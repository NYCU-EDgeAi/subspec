from __future__ import annotations

import logging
from typing import Any


def reset_kv(past_key_values: Any, draft_past_key_values: Any) -> None:
    if past_key_values is not None:
        past_key_values.reset()
    if draft_past_key_values is not None:
        draft_past_key_values.reset()


def maybe_init_cuda_graph_runner(
    generator: Any,
    past_key_values: Any,
    draft_past_key_values: Any,
    device: str,
    warmup_iter: int,
) -> None:
    """Initialize CUDA Graph runner (capture/setup) if supported, then reset KV caches.

    This is only used for methods that run FlashInfer kernels (i.e., generators that
    expose `init_cuda_graph_runner`). Intended to be called AFTER warmup.
    """
    logger = logging.getLogger(__name__)
    init_fn = getattr(generator, "init_cuda_graph_runner", None)
    if not callable(init_fn):
        return

    if warmup_iter <= 0:
        logger.info(
            "warmup_iter is set to 0; CUDA-graph capture for FlashInfer kernels is not activated. "
            "Set warmup_iter > 0 to enable it and get the speedup benefit."
        )
        return

    logger.info("Initializing CUDA Graph runner (FlashInfer).")
    init_fn(device)
    reset_kv(past_key_values, draft_past_key_values)
