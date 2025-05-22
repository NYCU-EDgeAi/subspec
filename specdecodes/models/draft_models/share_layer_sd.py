import torch

from .classic_sd import ClassicSDDraftModel
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
    
class ShareLayerSDDraftModel(ClassicSDDraftModel):
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