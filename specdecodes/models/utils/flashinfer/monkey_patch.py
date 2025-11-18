from typing import Callable
from .rms_norm import FiLlamaRMSNorm
from .attention import FiLlamaAttention, FiQwen3Attention
# from .ragged_attention import LlamaAttention as RaggedLlamaAttention
from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
    
def _bind_method_to_module(module, method_name: str, new_method: Callable):
    # Binds a new method to a module instance so that self is passed as the first argument
    module.__dict__[method_name] = new_method.__get__(module, module.__class__)

def _patch_rms_norm_module(module, eps=1e-6):
   
    module.variance_epsilon = getattr(module, "variance_epsilon", None) or getattr(module, "eps", None) or eps
   
    _bind_method_to_module(module, "forward", FiLlamaRMSNorm.forward)
    _bind_method_to_module(module, "extra_repr", FiLlamaRMSNorm.extra_repr)

def _patch_attention_module(module, use_ragged=False):
    # If you only want to override the forward method (keeping the rest), do:
    if use_ragged:
        pass
        # _bind_method_to_module(module, "forward", RaggedLlamaAttention.forward)
    else:
        if isinstance(module, LlamaAttention):
            _bind_method_to_module(module, "forward", FiLlamaAttention.forward)

        elif isinstance(module, Qwen3Attention):
            _bind_method_to_module(module, "forward", FiQwen3Attention.forward)
        else:
            raise ValueError(f"Unsupported attention module type, only suppoort llama and qwen for now: {type(module)}")
        
            
def apply_flashinfer_kernel_to_llama(
    attention: bool = True,
    # cross_entropy: bool = False,
    # fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
    use_ragged: bool = False,
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
    from transformers.models.qwen3 import modeling_qwen3

    if rms_norm:
        modeling_llama.LlamaRMSNorm = FiLlamaRMSNorm
        modeling_qwen3.Qwen3RMSNorm = FiLlamaRMSNorm
    if attention:
        if use_ragged:
            # modeling_llama.LlamaAttention = RaggedLlamaAttention
            pass
        else:
            modeling_llama.LlamaAttention = FiLlamaAttention
            modeling_qwen3.Qwen3Attention = FiQwen3Attention
    # replace_llama_qkv_with_fused(model)

    if model is not None:

        # for target model
        if hasattr(model, "base_model_prefix"):
            base_model = getattr(model, model.base_model_prefix, model)
        else:
            # fallback to your underlying base model, for draft models
            base_model = getattr(model, "model", model).model

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)

        for decoder_layer in base_model.layers:
            # if swiglu:
            #     _bind_method_to_module(decoder_layer.mlp, "forward", LigerSwiGLUMLP.forward)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)
            if attention:
                _patch_attention_module(decoder_layer.self_attn,use_ragged)
                # decoder_layer.self_attn.fuse_qkv()