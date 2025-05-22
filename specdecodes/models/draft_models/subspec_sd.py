import torch
import nvtx

from ..utils.cpu_tree import Tree
from .base import DraftModelBase, TreeData, TreeMaskCache
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

class SubSpecSDDraftModel(DraftModelBase):
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
    
    def forward(self, input_ids, with_softmax=False, *model_args, **kwargs):
        logits = self.model(input_ids, *model_args, **kwargs).logits
        if with_softmax:
            logits = torch.softmax(logits/self.draft_params.temperature, dim=-1)
            
        return logits
    
    @torch.no_grad()
    def speculate(self, input_ids, past_key_values, **kwargs):
        # 1) Obtain necessary parameters
        device = input_ids.device
        dtype = self.model.lm_head.weight.dtype
        batch_size, input_len = input_ids.shape
        max_cache_len = getattr(past_key_values, "max_cache_len", None)
        assert batch_size == 1, "Only support batch_size=1 for now."

        # 2) Initialize kv_len & cache_position
        with nvtx.annotate("Initialize kv_len & cache_position"):
            kv_len = past_key_values.get_seq_length()
            # convert kv_len to int if it is a tensor
            if isinstance(kv_len, torch.Tensor):
                kv_len = kv_len.item()
            org_kv_len = kv_len
            cache_position = torch.arange(kv_len, kv_len+input_len, dtype=torch.long, device=device)
        
        # 3) First forward pass
        position_ids = torch.full((batch_size, input_len), kv_len, device=device, dtype=torch.long)
        with nvtx.annotate("ssm first forward", color="red"):
            sampled_probs = self(
                input_ids,
                with_softmax=True,
                past_key_values=past_key_values,
                position_ids=position_ids,
                cache_position=cache_position,
                logits_to_keep=1,
            )
            kv_len += input_len

        with nvtx.annotate("update parent_probs & position_ids & cache_position"):
            parent_probs = torch.ones((1, 1), device=device, dtype=dtype)
            position_ids = torch.full((batch_size, self.draft_params.topk_len), kv_len, device=device, dtype=torch.long)
            cache_position = torch.arange(kv_len, kv_len+self.draft_params.topk_len, dtype=torch.long, device=device)
        
        # 4) Create TreeData & TreeMaskCache to manage tree structure and intermediate data.
        root_id = input_ids[0, -1]
        tree_data = TreeData(
            root_id,
            sample_len=self.draft_params.topk_len,
            max_sample_depth=self.draft_params.max_depth,
            dtype=dtype,
            device=device,
        )
        tree_mask_cache = TreeMaskCache(
            prefix_len=kv_len,
            sample_len=self.draft_params.topk_len,
            max_cache_len=max_cache_len,
            dtype=dtype,
            device=device,
        )

        # 5) Main loop
        for depth_i in range(self.draft_params.max_depth):
            # --------------------------------------
            # A. Compute token distribution & Sample
            # --------------------------------------
            with nvtx.annotate("sample nodes", color="green"):
                token_ids, child_probs, parent_indices = self.topk_sampling(
                    sampled_probs,
                    parent_probs,
                    self.draft_params.topk_len
                )
                parent_probs = child_probs
            
            # --------------------------------------
            # B. Add new nodes to the CPU tree
            # --------------------------------------
            with nvtx.annotate("add nodes", color="green"):
                tree_data.update(token_ids, child_probs, parent_indices)
                
            with nvtx.annotate("tree mask"):
                tree_attention_mask = tree_mask_cache.update_tree_mask(parent_indices)
            
            with nvtx.annotate("ssm forward", color="red"):
                sampled_probs = self(
                    token_ids,
                    with_softmax=True,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    attention_mask=tree_attention_mask,
                    cache_position=cache_position,
                )
                kv_len += self.draft_params.topk_len
                
            with nvtx.annotate("update position_ids & cache"):
                position_ids += 1
                cache_position += self.draft_params.topk_len
        
        # Discard new calcs in KV cache after original input length
        with nvtx.annotate("crop kv"):
            past_key_values.crop(org_kv_len, kv_len, dim=2)

        # Obtain the final tree
        with nvtx.annotate("tree related"):
            with nvtx.annotate("get tree data"):
                data = tree_data.get_data()
            with nvtx.annotate("build tree"):
                tree = Tree(root_id, dtype)
                tree.add_nodes(*data)
            with nvtx.annotate("prune tree"):
                tree.prune_to_top_n(self.draft_params.max_verify_tokens)
        
        return tree