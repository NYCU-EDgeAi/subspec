import torch
from typing import Optional

from .cache_manager import RequestKvCache, getKvCacheBatchPosition


def flashinfer_chunked_prefill(
    *,
    target_model,
    flashinfer_wrapper,
    input_ids: torch.LongTensor,
    kv_cache_pool,
    request_kv_cache: RequestKvCache,
    prefill_chunk_size: Optional[int],
):
    """Chunked FlashInfer prefill.

    Infrastructure helper (not generator-specific):
    - increments request KV cache per chunk
    - prepares FlashInfer attention per chunk
    - only keeps logits on the final chunk

    Returns the last forward outputs.
    """

    prefill_length = input_ids.size(1)
    chunk_size = (
        prefill_length
        if prefill_chunk_size is None
        else min(prefill_length, int(prefill_chunk_size))
    )

    outputs = None
    for start in range(0, prefill_length, chunk_size):
        chunk = input_ids[:, start : start + chunk_size]
        num_new_tokens = chunk.size(1)

        request_kv_cache.increment(num_new_tokens)
        batch_position = getKvCacheBatchPosition(
            request_kv_caches=[request_kv_cache],
            mode="tree",
            device=input_ids.device,
            treeTokens=num_new_tokens,
        )
        flashinfer_wrapper.prepareAttention(
            "prefill",
            batch_position,
            kv_cache_pool.page_len,
            "NONE",  # POS_ENCODING_MODE.NONE,
            kv_cache_pool.cache_data[0].dtype,
        )

        outputs = target_model.prefill_forward(
            input_ids=chunk,
            past_key_values=None,
            use_cache=False,
            logits_to_keep=1 if (start + chunk_size) >= prefill_length else None,
            kvCachePool=kv_cache_pool,
            batch_position=batch_position,
            mode="prefill",
            flashinferWrapper=flashinfer_wrapper,
        )

    return outputs
