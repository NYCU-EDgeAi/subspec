from typing import Callable
from .flashinfer import FusedRMSNorm


def _bind_method_to_module(module, method_name: str, new_method: Callable):
    # Binds a new method to a module instance so that self is passed as the first argument
    module.__dict__[method_name] = new_method.__get__(module, module.__class__)

def _patch_rms_norm_module(module, eps=1e-6):
    module.variance_epsilon = getattr(module, "variance_epsilon", None) or getattr(module, "eps", None) or eps
    _bind_method_to_module(module, "forward", FusedRMSNorm.forward)
    _bind_method_to_module(module, "extra_repr", FusedRMSNorm.extra_repr)
            
def apply_monkey_patch(
    model,
    rms_norm: bool = True,
) -> None:
    # get the base model from the model instance
    base_model = getattr(model, model.base_model_prefix, model)

    if rms_norm:
        _patch_rms_norm_module(base_model.norm)

    for decoder_layer in base_model.layers:
        if rms_norm:
            _patch_rms_norm_module(decoder_layer.input_layernorm)
            _patch_rms_norm_module(decoder_layer.post_attention_layernorm)