import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria
import logging
import nvtx
import torchaudio

from .base import GeneratorBase
from ..utils.mixin import SDProfilingMixin

class ClassicSDGeneratorBase(GeneratorBase):
    def __init__(self, generator_kwargs, *model_args, **kwargs):
        super().__init__(*model_args, **kwargs)
        self.prefill_chunk_size = generator_kwargs.get("prefill_chunk_size", None)
        
    def _speculate(self, input_ids, *model_args, **kwargs):
        return self.draft_model.speculate(input_ids, *model_args, **kwargs)

    def _tree_decoding(self, draft_ids, cache_position, past_key_values):
        # llm forward
        #TODO: Remove unnecessary squeeze(0) and unsqueeze(0) operations
        with nvtx.annotate("llm forward", color="red"):
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

        # print("draft_ids:", self.tokenizer.batch_decode(draft_ids.squeeze(0), skip_special_tokens=False))
        # print("global_ids:", self.tokenizer.batch_decode(global_ids.squeeze(0), skip_special_tokens=False))
        # print("sampled:", self.tokenizer.batch_decode(sampled_tokens, skip_special_tokens=False))

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
        Generate sequence of tokens with speculative decoding.

        This method consists of two main stages: prefill and decode.

        Prefill Stage:
        - Perform the model's initial forward pass.
        - Sample a token and append it to the input_ids.

        Decode Stage (with speculative decoding):
        - Iterate through the following steps:
            1. Perform SSM speculative sampling, returns sampled tokens in tree form.
            2. Decode the sampled tokens in parallel with the language model (LLM), generating probabilities for each token.
            3. Verify the sampled tokens by accepting or rejecting them, corresponding to the probabilities.
            4. Update the key-value cache and input_ids accordingly.

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

        # * clone input_ids 
        input_ids = input_ids.clone()
        batch_size, org_input_len = input_ids.shape
        assert batch_size == 1, "Only support batch_size=1 for now."

        # * prepare kv-cache
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
        
        # * prefill stage
        with nvtx.annotate("chunked prefill", color="orange"):
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
            sampled_tokens = self._sample_token(next_token_logits, logits_processor, do_sample)

        with nvtx.annotate("update data"):
            input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)
            cache_position = torch.arange(org_input_len, org_input_len+self.draft_params.max_verify_tokens, dtype=torch.long, device=input_ids.device)

        with nvtx.annotate("decoding"):
            finished = False
            while not finished:
                # * speculate
                with nvtx.annotate("speculate", color="cyan"):
                    input_ids = input_ids.clone(memory_format=torch.contiguous_format)
                    draft_ids = self._speculate(input_ids)
                    if self.cache_implementation == 'dynamic':
                        _, input_len = input_ids.shape
                        draft_past_key_values.crop(input_len)

                # * tree decoding
                with nvtx.annotate("tree_decoding", color="orange"):
                    prev_kv_len = past_key_values.get_seq_length()
                    outputs = self._tree_decoding(draft_ids, cache_position, past_key_values)
                    next_token_logits = outputs.logits
                    del outputs

                # * verify
                with nvtx.annotate("verify"):
                    root_ind = 0
                    sampled_tokens, hidden_indices, (total_len, accept_len) = self._verify(
                                                        draft_ids, root_ind, next_token_logits, 
                                                        logits_processor,
                                                        do_sample
                                                    )
                    del next_token_logits
                    # print(f"total_len: {total_len}, accept_len: {accept_len}, sampled_tokens.shape: {sampled_tokens.shape}")
                    
                # * update input_ids and cache_position
                with nvtx.annotate("update data"):
                    input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)
                    cache_position += sampled_tokens.shape[1]
                
                # * check stopping criteria
                with nvtx.annotate("stopping criteria"):
                    prune_tokens = 0
                    for k in range(sampled_tokens.shape[1]):    
                        finished = stopping_criteria(sampled_tokens[:, k:k+1], None).item()
                        if finished:
                            prune_tokens = sampled_tokens.shape[1]-k-1
                            input_ids = input_ids[:, :-prune_tokens] if prune_tokens > 0 else input_ids
                            break
                                
                with nvtx.annotate("update kv-cache"):
                    if self.cache_implementation == 'dynamic':
                        past_key_values.crop(prev_kv_len + sampled_tokens.shape[1])
                    past_key_values.seq_len += sampled_tokens.shape[1]
                    
        return input_ids
    
class ClassicSDGenerator(SDProfilingMixin, ClassicSDGeneratorBase):
    pass