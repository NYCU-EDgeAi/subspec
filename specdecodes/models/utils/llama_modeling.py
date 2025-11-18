import transformers
import logging
import nvtx

from typing import Callable, List, Optional, Tuple, Union
import torch
from torch import nn
from transformers import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, LlamaAttention, eager_attention_forward
from transformers.processing_utils import Unpack
from transformers.activations import ACT2FN

from transformers.models.llama import modeling_llama  # module we will patch

def _bind_method_to_module(module, method_name: str, new_method: Callable):
    # Binds a new method to a module instance so that self is passed as the first argument
    module.__dict__[method_name] = new_method.__get__(module, module.__class__)

def _get_decoder_layers(model: PreTrainedModel):
    """
    Returns (base_model, layers) where layers is an iterable of decoder layers.
    This handles common HF structures like model.model.layers, model.layers, etc.
    """
    # Try base_model_prefix (e.g., "model" for LlamaForCausalLM)
    base_model = None
    if hasattr(model, "base_model_prefix"):
        base_model = getattr(model, model.base_model_prefix, None)
    # Common fallback for LlamaForCausalLM -> .model
    if base_model is None:
        base_model = getattr(model, "model", None)
    # Some wrappers nest again (rare), normalize
    if hasattr(base_model, "model"):
        base_model = base_model.model
    if base_model is None:
        raise AttributeError("Could not locate base model. Tried base_model_prefix and `.model`.")
    if not hasattr(base_model, "layers"):
        raise AttributeError("Base model does not expose `.layers` – unexpected LLaMA structure.")
    return base_model, base_model.layers

def apply_nvtx_to_llama(
    model: Optional[PreTrainedModel] = None,
    *,
    patch_modeling_module: bool = True,
    verbose: bool = True,
):
    if patch_modeling_module:
        modeling_llama.LlamaAttention = nvtx_LlamaAttention
        modeling_llama.LlamaMLP = nvtx_LlamaMLP

    if model is None:
        return  # Only global monkey patch requested

    base_model, layers = _get_decoder_layers(model)
    config: LlamaConfig = getattr(model, "config", getattr(base_model, "config", None))
    if config is None:
        raise AttributeError("Could not find LLaMA config on model.")

    # Replace layer modules one by one
    for idx, decoder_layer in enumerate(layers):
        # self-attention
        if hasattr(decoder_layer, "self_attn") and not isinstance(decoder_layer.self_attn, nvtx_LlamaAttention):
           _bind_method_to_module(decoder_layer.self_attn, "forward", nvtx_LlamaAttention.forward)

        # mlp
        if hasattr(decoder_layer, "mlp") and not isinstance(decoder_layer.mlp, nvtx_LlamaMLP):
            _bind_method_to_module(decoder_layer.mlp, "forward", nvtx_LlamaMLP.forward)

class nvtx_LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        with nvtx.annotate("qkv forward", color="cyan"):
            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward

        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                print(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        with nvtx.annotate("attention forward", color="red"):
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        with nvtx.annotate("o forward", color="orange"):
            attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    
class nvtx_LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        with nvtx.annotate("mlp forward", color="green"):
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj