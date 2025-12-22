import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria
import logging
import nvtx

from .base import GeneratorBase
from ..utils.mixin import ProfilingMixin


class NaiveGeneratorBase(GeneratorBase):
    def __init__(self, generator_kwargs, *model_args, **kwargs):
        super().__init__(*model_args, **kwargs)
        self.prefill_chunk_size = generator_kwargs.get("prefill_chunk_size", None)
        print(f"prefill_chunk_size: {self.prefill_chunk_size}")

    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
        **model_kwargs,
    ):
        assert self.target_model is not None, "target_model must be provided"

        # Clone input_ids
        input_ids = input_ids.clone()
        batch_size, input_len = input_ids.shape
        assert batch_size == 1, "Only support batch_size=1 for now."

        # Prepare kv-cache and cache position
        if stopping_criteria.max_length is None:
            if self.cache_implementation == "static":
                raise ValueError(
                    "max_length is not set. Only 'dynamic' kv-cache is supported when max_length is unspecified."
                )

        if model_kwargs.get("past_key_values") is not None:
            past_key_values = model_kwargs["past_key_values"]
            max_cache_len = getattr(past_key_values.cache, "max_cache_len", None)
        else:
            raise ValueError("past_key_values should be provided")

        stream_callback = model_kwargs.get("stream_callback", None)

        kv_len = past_key_values.get_seq_length()
        cache_position = torch.arange(kv_len, input_len, dtype=torch.long, device=input_ids.device)

        # Prefill stage
        with nvtx.annotate("prefill", color="orange"):
            current_kv_len = past_key_values.get_seq_length()
            prefill_tokens = input_ids[:, current_kv_len:]
            prefill_length = prefill_tokens.size(1)
            chunk_size = prefill_length if self.prefill_chunk_size is None else min(prefill_length, self.prefill_chunk_size)
            next_token_logits = None
            for start in range(0, prefill_length, chunk_size):
                chunk = prefill_tokens[:, start:start + chunk_size]
                current_kv_len = past_key_values.get_seq_length()
                cache_position = torch.arange(
                    current_kv_len, current_kv_len + chunk.size(1),
                    dtype=torch.long, device=input_ids.device
                )
                # last iteration
                if start + chunk_size < prefill_length:
                    # does not need output logits, just update kv-cache
                    self.target_model.model(
                        chunk,
                        past_key_values=past_key_values.cache,
                        position_ids=cache_position.unsqueeze(0),
                        cache_position=cache_position,
                    )
                else:
                    outputs = self.target_model.prefill_forward(
                        chunk,
                        past_key_values=past_key_values.cache,
                        position_ids=cache_position.unsqueeze(0),
                        cache_position=cache_position,
                        logits_to_keep=1,
                    )
                    next_token_logits = outputs.logits
                    del outputs
                past_key_values.seq_len += chunk.size(1)

        with nvtx.annotate("sample tokens"):
            next_tokens = self._sample_token(next_token_logits, logits_processor, do_sample)

        with nvtx.annotate("update data"):
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            cache_position = cache_position[-1:] + 1
            self._maybe_stream(stream_callback, next_tokens)

        # Decoding loop
        with nvtx.annotate("decoding"):
            finished = False
            while not finished:
                with nvtx.annotate("llm forward", color="orange"):
                    outputs = self.target_model(
                        next_tokens,
                        past_key_values=past_key_values.cache,
                        position_ids=cache_position.unsqueeze(0),
                        cache_position=cache_position,
                    )
                    next_token_logits = outputs.logits

                with nvtx.annotate("sample tokens"):
                    next_tokens = self._sample_token(next_token_logits, logits_processor, do_sample)

                with nvtx.annotate("update data"):
                    input_ids = torch.cat([input_ids, next_tokens], dim=-1)
                    cache_position += 1
                    past_key_values.seq_len += 1
                    self._maybe_stream(stream_callback, next_tokens)

                with nvtx.annotate("stopping criteria"):
                    finished = stopping_criteria(input_ids, None)

        return input_ids

class NaiveGenerator(ProfilingMixin, NaiveGeneratorBase):
    pass