import torch
import nvtx
import os

from ..utils.cpu_tree import Tree
from .classic_seq_sd import ClassicSDDraftModel
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
    def speculate(self, input_ids, depth=None, **kwargs):
        self.had_first_speculate = True
        
        # 1) Obtain necessary parameters
        device = input_ids.device
        batch_size, input_len = input_ids.shape
        assert batch_size == 1, "Only support batch_size=1 for now."
        assert input_len == 1, "Value of input_len should be 1, as this is the root node of the tree."
        depth = depth if depth is not None else self.draft_params.max_depth

        # 2) Initialize kv_len & cache_position
        with nvtx.annotate("Initialize kv_len & cache_position"):
            kv_len = self.past_key_values.get_seq_length()
            # convert kv_len to int if it is a tensor
            if isinstance(kv_len, torch.Tensor):
                kv_len = kv_len.item()

        # 3) First forward pass
        cache_position = torch.arange(kv_len, kv_len+input_len, dtype=torch.long, device=device)
        with nvtx.annotate("ssm first forward", color="red"):
            sampled_probs = self(
                input_ids,
                with_softmax=True,
                past_key_values=self.past_key_values.cache,
                position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position,
                logits_to_keep=1,
            )
            kv_len += input_len

        with nvtx.annotate("sample nodes", color="green"):
            sampled_token = torch.argmax(sampled_probs[:, -1:], dim=-1)
                                
        # 4) Initialize TreeData & TreeMaskCache to manage tree structure and intermediate data.
        self.token_ids = []
        self.token_ids.append(input_ids[:, -1:])
        self.token_ids.append(sampled_token)
        self.cache_position = torch.arange(kv_len, kv_len+self.draft_params.topk_len, dtype=torch.long, device=device)

        if os.environ.get("DETAILED_ANALYSIS", "False") == "True":
            self.draft_prob = [torch.max(sampled_probs[:, -1:]).cpu().item()]

        # 6) Main loop
        for depth_i in range(depth-1):
            self.speculate_once()

        return torch.cat(self.token_ids, dim=-1)
    
    def init_postspec(self):
        self.postspec_count = 0
        
    @torch.no_grad()
    def postspec(self):
        if not self.had_first_speculate:
                #print("Post speculate before first speculate, skip.")
                pass
        elif self.postspec_count > (self.draft_params.max_depth - 1):
                #print("Post speculate reached max depth, skip.")
                pass
        else:
            with nvtx.annotate("post_speculate_once", color="blue"):
                self.speculate_once()
            self.postspec_count += 1
    
    def update_tree_after_post(self):
        """
        Get the tree structure 
        """
        # Update the tree data and mask cache before returning
        token_ids = torch.cat(self.token_ids, dim=-1)
        new_token_ids = token_ids[:, -self.postspec_count:]
        return new_token_ids