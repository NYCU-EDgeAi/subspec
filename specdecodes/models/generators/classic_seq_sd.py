import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria
import nvtx

from .base import GeneratorBase
from ..utils.mixin import SDProfilingMixin

class ClassicSDGeneratorBase(GeneratorBase):
    def __init__(self, generator_kwargs, *model_args, **kwargs):
        super().__init__(*model_args, **kwargs)
        self.prefill_chunk_size = generator_kwargs.get("prefill_chunk_size", None)
        
    def _speculate(self, input_ids, *model_args, **kwargs):
        return self.draft_model.speculate(input_ids, *model_args, **kwargs)

    def _tree_decoding(self, draft_ids, cache_position, past_key_values):
        # Target model forward
        with nvtx.annotate("target_forward", color="red"):
            outputs = self.target_model(
                draft_ids,
                past_key_values=past_key_values.cache,
                position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position,
            )
        return outputs
    
    def _verify(self, draft_ids, root_ind, logits, logits_processor, do_sample, skip_nodes: int = 0):
        global_ids = self._sample_token(logits, logits_processor, do_sample, return_probs=False)  # [1, T]
        g0 = global_ids[0] # [T]
        d = draft_ids[0][root_ind:root_ind + g0.size(0)] # [T]

        valid = (d[1:] == g0[:-1]) & (g0[:-1] != self.draft_model.eos_token_id)
        accept_len = int(torch.cumprod(valid.to(torch.int64), dim=0).sum().item())
        cmp_len = g0.size(0) - 1
        total_len = cmp_len if accept_len == cmp_len else accept_len + 1

        sampled_tokens = g0[:accept_len + 1]

        return sampled_tokens.unsqueeze(0), None, (total_len, accept_len)
    
    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
        **model_kwargs,
    ):
        """
        Generate a token sequence with sequential speculative decoding.

        Stages:
        - Prefill: run the target model once on the prompt, then sample the next token.
        - Decode loop:
            1) Draft proposes a block of candidate tokens.
            2) Target scores the candidate block in one forward.
            3) Verify and accept a prefix, then update KV/state.

        Args:
            input_ids (torch.LongTensor): The input token IDs. 
            stopping_criteria (StoppingCriteria): The criteria to stop the generation.
            logits_processor (LogitsProcessor): The processor to modify the logits.
            do_sample (bool): Whether to sample tokens during generation. If False, the generation will be deterministic.

        Returns:
            input_ids (torch.LongTensor): The generated token IDs.
        """
        assert self.target_model is not None, "target_model must be provided"
        assert self.draft_model is not None, "draft_model must be provided"
        assert self.tokenizer is not None, "tokenizer must be provided"

        input_ids = input_ids.clone()
        batch_size, org_input_len = input_ids.shape
        assert batch_size == 1, "Only support batch_size=1 for now."

        # Raise error if max_length not set while using static cache
        if stopping_criteria.max_length is None:
            if self.cache_implementation == "static":
                raise ValueError(
                    "max_length is not set. Only 'dynamic' kv-cache is supported when max_length is unspecified."
                )
            
        if model_kwargs.get("past_key_values") is not None:
            past_key_values = model_kwargs["past_key_values"]
            max_cache_len = getattr(past_key_values.cache, "max_cache_len", None)

            if model_kwargs.get("draft_past_key_values") is not None:
                draft_past_key_values = model_kwargs["draft_past_key_values"]
                self.draft_model.set_past_key_values(draft_past_key_values)
        else:
            raise ValueError("past_key_values and draft_past_key_values should both be provided")

        stream_callback = model_kwargs.get("stream_callback", None)
        
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
            sampled_tokens = self._sample_token(next_token_logits, logits_processor, do_sample)

        with nvtx.annotate("state_update"):
            input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)
            cache_position = torch.arange(org_input_len, org_input_len+self.draft_params.max_verify_tokens, dtype=torch.long, device=input_ids.device)
            self._maybe_stream(stream_callback, sampled_tokens)

        with nvtx.annotate("decode_loop"):
            finished = False
            while not finished:
                with nvtx.annotate("speculate", color="cyan"):
                    input_ids = input_ids.clone(memory_format=torch.contiguous_format)
                    draft_ids = self._speculate(input_ids)
                    if self.cache_implementation == 'dynamic':
                        _, input_len = input_ids.shape
                        draft_past_key_values.crop(input_len)

                with nvtx.annotate("target_decode", color="orange"):
                    prev_kv_len = past_key_values.get_seq_length()
                    outputs = self._tree_decoding(draft_ids, cache_position, past_key_values)
                    next_token_logits = outputs.logits
                    del outputs

                with nvtx.annotate("verify"):
                    root_ind = 0
                    sampled_tokens, hidden_indices, (total_len, accept_len) = self._verify(
                                                        draft_ids, root_ind, next_token_logits, 
                                                        logits_processor,
                                                        do_sample
                                                    )
                    del next_token_logits
                    
                with nvtx.annotate("state_update"):
                    input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)
                    cache_position += sampled_tokens.shape[1]
                
                with nvtx.annotate("stop_check"):
                    finished, input_ids, kept, prune_tokens = self._apply_tokenwise_stopping_criteria(
                        input_ids=input_ids,
                        sampled_tokens=sampled_tokens,
                        stopping_criteria=stopping_criteria,
                    )
                if kept.numel() > 0:
                    self._maybe_stream(stream_callback, kept)
                                
                with nvtx.annotate("kv_update"):
                    if self.cache_implementation == 'dynamic':
                        past_key_values.crop(prev_kv_len + sampled_tokens.shape[1])
                    past_key_values.seq_len += sampled_tokens.shape[1]
                    
        return input_ids
    
class ClassicSDGenerator(SDProfilingMixin, ClassicSDGeneratorBase):
    pass