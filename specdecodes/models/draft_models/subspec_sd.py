import torch
import nvtx

from specdecodes.models.utils.wandb_logger import wandb_logger

from ..utils.cpu_tree import Tree
from .classic_sd import ClassicSDDraftModel, TreeData, TreeMaskCache
from copy import deepcopy


def share_param_deepcopy(model):
    # Build the memo dictionary from the model's parameters (and optionally buffers)
    model_memo = {}
    for _, param in model.named_parameters():
        model_memo[id(param)] = param
    for _, buf in model.named_buffers():
        model_memo[id(buf)] = buf

    # Clone the model using the memo dictionary.
    share_model = deepcopy(model, memo=model_memo)
    return share_model

class SubSpecSDDraftModel(ClassicSDDraftModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.had_first_speculate = False
        self.postspec_count = 0
    
    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path=None,
        *model_args,
        target_model = None,
        torch_dtype=torch.float32,
        **model_kwargs
    ):
        # Remove the following arguments from model_kwargs, cause AutoModelForCausalLM does not accept them
        eos_token_id = model_kwargs.pop("eos_token_id", None)
        
        base_model = share_param_deepcopy(target_model)
        model = cls(base_model=base_model, eos_token_id=eos_token_id, *model_args, **model_kwargs)
        
        # Convert the model to the desired dtype and return
        model.to(dtype=torch_dtype)
        return model
    
    @torch.no_grad()
    def speculate(self, input_ids, **kwargs):
        self.had_first_speculate = True
        
        # 1) Obtain necessary parameters
        device = input_ids.device
        dtype = self.model.lm_head.weight.dtype
        batch_size, input_len = input_ids.shape
        max_cache_len = getattr(self.past_key_values.cache, "max_cache_len", None)
        assert batch_size == 1, "Only support batch_size=1 for now."
        assert input_len == 1, "Value of input_len should be 1, as this is the root node of the tree."

        # 2) Initialize kv_len & cache_position
        with nvtx.annotate("kv_init"):
            kv_len = self._get_kv_len_int()

        # 3) First forward pass
        with nvtx.annotate("draft_prefill", color="red"):
            cache_position = torch.arange(kv_len, kv_len + input_len, dtype=torch.long, device=device)
            sampled_probs = self(
                input_ids,
                with_softmax=True,
                past_key_values=self.past_key_values.cache,
                position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position,
                logits_to_keep=1,
            )
            kv_len += input_len

        with nvtx.annotate("draft_sample", color="green"):
            self.parent_probs = torch.ones((1, 1), device=device, dtype=dtype)
            token_ids, child_probs, parent_indices = self.topk_sampling(
                sampled_probs,
                self.parent_probs,
                self.draft_params.topk_len
            )
            self.parent_probs = child_probs
                                
        # 4) Initialize TreeData & TreeMaskCache to manage tree structure and intermediate data.
        root_id = input_ids[0, -1]
        self.tree = Tree(root_id, dtype)
        self.tree_data = TreeData()
        self.tree_mask_cache = TreeMaskCache(
            prefix_len=kv_len,
            sample_len=self.draft_params.topk_len,
            max_cache_len=max_cache_len,
            dtype=dtype,
            device=device,
        )
        
        if wandb_logger.get_flag("detailed_analysis", False):
            self.draft_prob = [torch.max(sampled_probs[:, -1:]).cpu().item()]

        # 5) First update of tree_data and tree_mask_cache
        with nvtx.annotate("tree_update", color="green"):
            self.tree_data.update(token_ids, child_probs, parent_indices)
            self.tree_mask_cache.update_tree_mask(parent_indices)
        
        # Set initial state for the speculative tree
        self.token_ids = token_ids
        self.position_ids = torch.full((batch_size, self.draft_params.topk_len), kv_len, device=device, dtype=torch.long)
        self.cache_position = torch.arange(kv_len, kv_len+self.draft_params.topk_len, dtype=torch.long, device=device)
        
        # 6) Main loop
        for depth_i in range(self.draft_params.max_depth-1):
            self.speculate_once()

        # Update and obtain the final tree
        self.update_tree(self.tree_data)
        return self.tree
    
    def init_postspec(self):
        self.tree_data = TreeData()
        self.postspec_count = 0
        
    @torch.no_grad()
    def postspec(self):
        if not self.had_first_speculate:
            return
        if self.postspec_count > (self.draft_params.max_depth - 1):
            return
        with nvtx.annotate("postspec_step", color="blue"):
            self.speculate_once()
        self.postspec_count += 1
    
    def update_tree_after_post(self):
        """Return the finalized draft tree after post-speculation."""
        # Update the tree data and mask cache before returning
        self.update_tree(self.tree_data)
        return self.tree