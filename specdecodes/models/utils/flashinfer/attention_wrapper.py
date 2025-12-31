from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union
import math
import torch
import flashinfer
from .cache_manager import KvCacheBatchPosition

FLASH_INFER_SUPPORTED_DIMS = [64, 128, 256]


class POS_ENCODING_MODE(Enum):
    ROPE_LLAMA = "ROPE_LLAMA"
    ALIBI = "ALIBI"
    NONE = "NONE"


@dataclass(frozen=True)
class AttentionRotaryParams:
    causal: bool = True
    pos_encoding_mode: POS_ENCODING_MODE = POS_ENCODING_MODE.ROPE_LLAMA
    rope_scale: float = 1.0
    rope_theta: float = 1.0e4


def find_padded_head_dim(head_dim):
    for dim in FLASH_INFER_SUPPORTED_DIMS:
        if head_dim <= dim:
            return dim
    raise ValueError("The head dimension is too large for FlashInfer")


class FlashinferAttentionWrapper:
    def __init__(
        self, num_attention_heads: int, num_key_value_heads: int, hidden_size: int, page_len:int,
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self._head_padded_dim = find_padded_head_dim(self.head_dim)
        self.page_len = page_len

        self.group_size = self.num_attention_heads // self.num_key_value_heads
        _workspace_buffer = torch.empty(
            256 * 1024 * 1024, dtype=torch.int8, device=torch.cuda.current_device()
        )
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            float_workspace_buffer=_workspace_buffer, kv_layout="NHD",
        )
        _use_tensor_cores = self.group_size in [7, 16]
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            float_workspace_buffer=_workspace_buffer,
            kv_layout="NHD",
            use_tensor_cores=_use_tensor_cores,
            
        )

        self.tree_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            float_workspace_buffer=_workspace_buffer, kv_layout="NHD"
        )
        
        # for cuda grapth
        batch_size = 1
        
        self.qo_indptr_buf = torch.zeros(
            (batch_size + 1,),  # Typically 2 for batch=1
            dtype=torch.int32,
            device=torch.cuda.current_device()
        )

        self.paged_kv_indptr_buf = torch.zeros(
            (batch_size + 1,),  # Also 2
            dtype=torch.int32,
            device=torch.cuda.current_device()
        )

        self.paged_kv_indices_buf = torch.zeros(
            1024,              # Example size; tune to your max seq length / heads
            dtype=torch.int32,
            device=torch.cuda.current_device()
        )

        self.paged_kv_last_page_len_buf = torch.zeros(
            (batch_size,),     # 1
            dtype=torch.int32,
            device=torch.cuda.current_device()
        )

        # If you do not use custom masks, you can omit these;
        # otherwise, set them as large as the max needed for your custom mask.
        self.custom_mask_buf = torch.zeros(
            100000*1000,  # Must be >= packed_mask.numel() to avoid OOB.
            dtype=torch.uint8,
            device=torch.cuda.current_device()
        )
        
        self.mask_indptr_buf = torch.zeros(
            (batch_size + 1,),  # Also 2
            dtype=torch.int32,
            device=torch.cuda.current_device()
        )
        
        self.tree_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            float_workspace_buffer=_workspace_buffer,
            kv_layout="NHD",
            use_cuda_graph=True,  # <-- The critical toggle
            qo_indptr_buf=self.qo_indptr_buf,
            paged_kv_indptr_buf=self.paged_kv_indptr_buf,
            paged_kv_indices_buf=self.paged_kv_indices_buf,
            paged_kv_last_page_len_buf=self.paged_kv_last_page_len_buf,
            custom_mask_buf=self.custom_mask_buf,
            mask_indptr_buf=self.mask_indptr_buf,
        )

    def prepareAttention(
        self,
        mode: str,
        batch_position: KvCacheBatchPosition,
        page_len: int,
        pos_encoding_mode: POS_ENCODING_MODE,
        dtype: torch.dtype,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if mode == "tree" and attention_mask is not None:
            self.tree_wrapper.begin_forward(
                    batch_position.seq_indptr,
                    batch_position.kv_page_indptr,
                    batch_position.kv_page_indices,
                    batch_position.kv_last_page_len,
                    self.num_attention_heads,
                    self.num_key_value_heads,
                    self._head_padded_dim,
                    page_len,
                    custom_mask = attention_mask,
                    causal = False,
                    non_blocking = True
            )
        elif mode == "tree" and attention_mask is None:
            self.tree_wrapper.begin_forward(
                    batch_position.seq_indptr,
                    batch_position.kv_page_indptr,
                    batch_position.kv_page_indices,
                    batch_position.kv_last_page_len,
                    self.num_attention_heads,
                    self.num_key_value_heads,
                    self._head_padded_dim,
                    page_len,
                    non_blocking=True,
                    causal = True,
            )
        elif mode == "prefill":
            self.prefill_wrapper.begin_forward(
                batch_position.seq_indptr,
                batch_position.kv_page_indptr,
                batch_position.kv_page_indices,
                batch_position.kv_last_page_len,
                self.num_attention_heads,
                self.num_key_value_heads,
                self._head_padded_dim,
                page_len,
                causal=True,
            )
        elif mode == "decode":
            self.decode_wrapper.begin_forward(
                batch_position.kv_page_indptr,
                batch_position.kv_page_indices,
                batch_position.kv_last_page_len,
                self.num_attention_heads,
                self.num_key_value_heads,
                self._head_padded_dim,
                page_len,
                # pos_encoding_mode="ROPE_LLAMA",
                data_type=dtype,
            )
        else :
            raise ValueError("the mode for attention must be prefill, decode or tree")

    def reshape_qkv_for_attention(self, q, k, v, batchPosition: KvCacheBatchPosition):
        return (
            q.view(
                -1,
                self.num_attention_heads,
                self.head_dim,
            ),
            k.view(
                -1,
                self.num_key_value_heads,
                self.head_dim,
            ),
            v.view(
                -1,
                self.num_key_value_heads,
                self.head_dim,
            ),
        )

    def computeAttention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cacheData: torch.Tensor,
        mode: str,
        batchPosition: KvCacheBatchPosition,
        rotaryParams: AttentionRotaryParams,
        layer_idx : int,
    ):

        q, k, v = self._pad_qkv(q, k, v)
        if mode == 'prefill':
            attn_output =  self._batchPrefill(q, k, v, cacheData, batchPosition, rotaryParams)
        elif mode == 'decode':
            attn_output = self._batchDecode(q, k, v, cacheData, batchPosition, rotaryParams)
        elif mode == 'tree':
            attn_output = self._treeDecode(q, k, v, cacheData, batchPosition, rotaryParams)

        return self._unpad_attention(attn_output)

    def _unpad_attention(self, attn_output):
        if self._head_padded_dim > self.head_dim:
            return attn_output[:, :, : self.head_dim].reshape(-1, self.hidden_size)
        else:
            return attn_output.view(-1, self.hidden_size)

    def _pad_qkv(self, q, k, v):
        if self._head_padded_dim > self.head_dim:
            q = torch.nn.functional.pad(q, (0, self._head_padded_dim - self.head_dim))
            k = torch.nn.functional.pad(k, (0, self._head_padded_dim - self.head_dim))
            v = torch.nn.functional.pad(v, (0, self._head_padded_dim - self.head_dim))
        return q, k, v

    # @torch._dynamo.disable
    def append_kv_cache(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_position,
        paged_kv_cache: torch.Tensor,
        page_len: int
    ):
        
        flashinfer.append_paged_kv_cache(
            append_key=k,
            append_value=v,
            batch_indices=batch_position.batch_indices,
            paged_kv_cache=paged_kv_cache,
            kv_indices=batch_position.kv_page_indices,
            positions=batch_position.positions,
            kv_indptr=batch_position.kv_page_indptr,
            kv_last_page_len=batch_position.kv_last_page_len,
        )

    def _batchPrefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cacheData: torch.Tensor,
        prefillBatchPosition: KvCacheBatchPosition,
        rotaryParams: AttentionRotaryParams,
    ):

        self.append_kv_cache (
            q,
            k,
            v,
            prefillBatchPosition,
            cacheData,
            self.page_len
        )

        attn_output_prefill = self.prefill_wrapper.forward(
            q,
            cacheData,
            # causal=rotaryParams.causal,
            causal=True,
            pos_encoding_mode=rotaryParams.pos_encoding_mode.value,
            sm_scale=1.0 / math.sqrt(self.head_dim),
            rope_scale=rotaryParams.rope_scale,
            rope_theta=rotaryParams.rope_theta,
        )


        return attn_output_prefill
    
    def _treeDecode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cacheData: torch.Tensor,
        treeBatchPosition: KvCacheBatchPosition,
        rotaryParams: AttentionRotaryParams,
    ):
        self.append_kv_cache (
            q,
            k,
            v,
            treeBatchPosition,
            cacheData,
            self.page_len
        )
        
        #  prefill
        attn_output = self.tree_wrapper.forward(
            q,
            cacheData,
            pos_encoding_mode=rotaryParams.pos_encoding_mode.value,
            sm_scale=1.0 / math.sqrt(self.head_dim),
            rope_scale=rotaryParams.rope_scale,
            rope_theta=rotaryParams.rope_theta,
        )

        return attn_output
    
    

    def _batchDecode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cacheData: torch.Tensor,
        decodeBatchPosition: KvCacheBatchPosition,
        rotaryParams: AttentionRotaryParams,
    ):
        self.append_kv_cache (
            q,
            k,
            v,
            decodeBatchPosition,
            cacheData,
            self.page_len
        )

        attn_output_decode = self.decode_wrapper.forward(
            q,
            cacheData,
            pos_encoding_mode=rotaryParams.pos_encoding_mode.value,
            sm_scale=1.0 / math.sqrt(self.head_dim),
            rope_scale=rotaryParams.rope_scale,
            rope_theta=rotaryParams.rope_theta,
        )

        return attn_output_decode