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
        logging.debug("prefill_chunk_size: %s", self.prefill_chunk_size)

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
        with nvtx.annotate("prefill_chunked", color="orange"):
            outputs = self._chunked_prefill_forward(
                input_ids,
                past_key_values,
                prefill_chunk_size=self.prefill_chunk_size,
                use_position_ids=True,
            )
            next_token_logits = outputs.logits
            del outputs

        with nvtx.annotate("sample"):
            next_tokens = self._sample_token(next_token_logits, logits_processor, do_sample)

        with nvtx.annotate("state_update"):
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            cache_position = cache_position[-1:] + 1
            self._maybe_stream(stream_callback, next_tokens)

        # Decoding loop
        with nvtx.annotate("decode_loop"):
            finished = False
            while not finished:
                with nvtx.annotate("target_forward", color="red"):
                    outputs = self.target_model(
                        next_tokens,
                        past_key_values=past_key_values.cache,
                        position_ids=cache_position.unsqueeze(0),
                        cache_position=cache_position,
                    )
                    next_token_logits = outputs.logits

                with nvtx.annotate("sample"):
                    next_tokens = self._sample_token(next_token_logits, logits_processor, do_sample)

                with nvtx.annotate("state_update"):
                    input_ids = torch.cat([input_ids, next_tokens], dim=-1)
                    cache_position += 1
                    past_key_values.seq_len += 1
                    self._maybe_stream(stream_callback, next_tokens)

                with nvtx.annotate("stop_check"):
                    finished = stopping_criteria(input_ids, None)

        return input_ids

class NaiveGenerator(ProfilingMixin, NaiveGeneratorBase):
    pass