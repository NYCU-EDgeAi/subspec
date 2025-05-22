from typing import Callable
from .rms_norm import FiLlamaRMSNorm
from .attention import FiLlamaAttention
from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaAttention


def set_module_name(model, name, value):
    if "." in name:
        parent_name = name.rsplit(".", 1)[0]
        child_name = name[len(parent_name) + 1 :]
        parent = model.get_submodule(parent_name)
    else:
        parent_name = ""
        parent = model
        child_name = name

    setattr(parent, child_name, value)
    
def _bind_method_to_module(module, method_name: str, new_method: Callable):
    # Binds a new method to a module instance so that self is passed as the first argument
    module.__dict__[method_name] = new_method.__get__(module, module.__class__)

def _patch_rms_norm_module(module, eps=1e-6):
   
    module.variance_epsilon = getattr(module, "variance_epsilon", None) or getattr(module, "eps", None) or eps
   
    _bind_method_to_module(module, "forward", FiLlamaRMSNorm.forward)
    _bind_method_to_module(module, "extra_repr", FiLlamaRMSNorm.extra_repr)

def _patch_attention_module(module):
    # If you only want to override the forward method (keeping the rest), do:
    _bind_method_to_module(module, "forward", FiLlamaAttention.forward)

    # Alternatively, if you want to replace the class entirely, you can do:
    # module.__class__ = LigerAttention

def replace_llama_qkv_with_fused(model):
    for name, module in model.named_modules():
        if isinstance(module, LlamaAttention):
            qkv = FiLlamaAttention(
                module.config,
                module.q_proj,
                module.k_proj,
                module.v_proj,
                module.layer_idx,
                # module.o_proj,
            )
            set_module_name(model, name, qkv)
            
def apply_flashinfer_kernel_to_llama(
    attention: bool = True,
    # cross_entropy: bool = False,
    # fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply kernels to replace original implementation in HuggingFace Llama models (2 and 3)

    Args:
        attention (bool): Whether to apply Flashinfer's rotary position embedding and attention forward. Default is True.
        rms_norm (bool): Whether to apply Flashinfer's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Flashinfer's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    from transformers.models.llama import modeling_llama
    from transformers.models.llama.modeling_llama import LlamaModel

    # if attention:
        # modeling_llama.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_llama.LlamaRMSNorm = FiLlamaRMSNorm
    # if swiglu:
    #     modeling_llama.LlamaMLP = LigerSwiGLUMLP
    if attention:
        modeling_llama.LlamaAttention = FiLlamaAttention
        
    # replace_llama_qkv_with_fused(model)

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules (e.g. LlamaRMSNorm or LlamaMLP)

        # get the base model from the model instance
        base_model: LlamaModel = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)

        for decoder_layer in base_model.layers:
            # if swiglu:
            #     _bind_method_to_module(decoder_layer.mlp, "forward", LigerSwiGLUMLP.forward)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)
            if attention:
                _patch_attention_module(decoder_layer.self_attn)
                # decoder_layer.self_attn.fuse_qkv()