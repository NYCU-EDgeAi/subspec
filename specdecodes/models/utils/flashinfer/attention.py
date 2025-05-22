import transformers
import logging

from typing import Callable, List, Optional, Tuple, Union
import torch
from torch import nn
from transformers import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, LlamaAttention
from transformers.processing_utils import Unpack

from .attention_wrapper import (
    POS_ENCODING_MODE,
    AttentionRotaryParams,
)
# class FiLlamaAttention(LlamaAttention):
    
#     def __init__(
#         self,
#         config,
#         q: torch.nn.Linear,  # pylint: disable=invalid-name
#         k: torch.nn.Linear,  # pylint: disable=invalid-name
#         v: torch.nn.Linear,  # pylint: disable=invalid-name
#         layer_idx: int,
#     ):
#         super().__init__(config, layer_idx)
#         self.config = config
#         self.init_device = next(iter(q.state_dict().values())).device

        
#         self.q_proj = q
#         self.k_proj = k
#         self.v_proj = v
        
#         self.init_device = next(iter(q.state_dict().values())).device
#         # print(self.init_device)
        
#         # define equivalent fused qkv projection
#         self.out_features: List[int] = [q.out_features, k.out_features, v.out_features]
#         self.qkv_proj = torch.nn.Linear(
#             q.in_features, sum(self.out_features), bias=config.attention_bias
#         ).to(self.init_device)
#         # print("qkv_proj" , self.qkv_proj)
#         # overwrite initialized weights with pretrained weights
#         self.qkv_proj.weight.data = torch.cat(
#             (q.weight.data, k.weight.data, v.weight.data), dim=0
#         )
        
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         position_embeddings: Tuple[torch.Tensor, torch.Tensor],
#         attention_mask: Optional[torch.Tensor],
#         past_key_value: Optional[Cache] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#         **kwargs: Unpack[FlashAttentionKwargs],
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
#         flashinferWrapper = kwargs["flashinferWrapper"]
#         kvCachePool       = kwargs.get("kvCachePool", None)
#         mode              = kwargs.get("mode", "prefill")
#         batch_position    = kwargs.get("batch_position", None)
#         position_ids      = kwargs.get("position_ids", None)
        
#         rotaryParams = AttentionRotaryParams(pos_encoding_mode=POS_ENCODING_MODE.NONE)
    
#         input_shape = hidden_states.shape[:-1]
#         hidden_shape = (*input_shape, -1, self.head_dim)

#         query_states, key_states, value_states = self.qkv_proj(hidden_states).split(
#                 self.out_features, dim=-1
#             ) 
#         query_states = query_states.view(hidden_shape).transpose(1, 2)
#         key_states = key_states.view(hidden_shape).transpose(1, 2)
#         value_states = value_states.view(hidden_shape).transpose(1, 2)
        
#         # query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
#         # key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
#         # value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
#         if self.layer_idx == 0:
#             print(query_states.shape)
#             print(key_states.shape)
#         cos, sin = position_embeddings
#         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

#         query = query_states.transpose(1, 2).contiguous()
#         key = key_states.transpose(1, 2).contiguous()
#         value = value_states.transpose(1, 2).contiguous()

#         q, k, v = flashinferWrapper.reshape_qkv_for_attention(
#             query, key, value, batch_position
#         )
        
#         attn_output = flashinferWrapper.computeAttention(
#             q,
#             k,
#             v,
#             kvCachePool.cache_data[self.layer_idx] ,
#             mode,
#             batch_position,
#             rotaryParams,
#             self.layer_idx
#         )

#         attn_output = attn_output.reshape(*input_shape, -1).contiguous()
#         attn_output = self.o_proj(attn_output)
#         # The second return is `attn_weights`, which for flashinfer we typically skip/None
#         # print("q")
#         return attn_output, None

class FiLlamaAttention(nn.Module):
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
        
        flashinferWrapper = kwargs["flashinferWrapper"]
        kvCachePool       = kwargs.get("kvCachePool", None)
        mode              = kwargs.get("mode", "prefill")
        batch_position    = kwargs.get("batch_position", None)
        position_ids      = kwargs.get("position_ids", None)
        
        rotaryParams = AttentionRotaryParams(pos_encoding_mode=POS_ENCODING_MODE.NONE)
    
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        query = query_states.transpose(1, 2).contiguous()
        key = key_states.transpose(1, 2).contiguous()
        value = value_states.transpose(1, 2).contiguous()

        q, k, v = flashinferWrapper.reshape_qkv_for_attention(
            query, key, value, batch_position
        )
        
        attn_output = flashinferWrapper.computeAttention(
            q,
            k,
            v,
            kvCachePool.cache_data[self.layer_idx] ,
            mode,
            batch_position,
            rotaryParams,
            self.layer_idx
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        # The second return is `attn_weights`, which for flashinfer we typically skip/None
        # print("q")
        return attn_output, None