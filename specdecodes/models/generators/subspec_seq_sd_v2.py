import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria
import logging
import nvtx

from .classic_seq_sd import ClassicSDGeneratorBase
from ..utils.mixin import SDProfilingMixin

class SubSpecSDGeneratorBase(ClassicSDGeneratorBase):
    def __init__(self, generator_kwargs, *model_args, **kwargs):
        super().__init__(generator_kwargs, *model_args, **kwargs)
        post_draft_params = kwargs.get("post_draft_params", None)
        if post_draft_params is None:
           raise ValueError("post_draft_params must be provided in subspec_sd_v2 generator.")
        self.post_draft_params = post_draft_params
        self.draft_model.post_draft_params = post_draft_params
        
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
            
            self.draft_model.set_past_key_values(past_key_values)
        else:
            raise ValueError("past_key_values is not provided")

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
                    with nvtx.annotate("chunked", color="blue"):
                        # does not need output logits, just update kv-cache
                        self.target_model.model(
                            chunk,
                            past_key_values=past_key_values.cache,
                            position_ids=cache_position.unsqueeze(0),
                            cache_position=cache_position,
                        )
                else:
                    with nvtx.annotate("last", color="purple"):
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
            # print(f"After prefill, sampled token: {self.tokenizer.decode(sampled_tokens[0])}")
            # return "end"

        with nvtx.annotate("update data"):
            input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)
            cache_position = torch.arange(org_input_len, org_input_len+self.draft_params.max_verify_tokens, dtype=torch.long, device=input_ids.device)
        
        with nvtx.annotate("speculate", color="cyan"):
            last_token_id = sampled_tokens[:, -1:].clone(memory_format=torch.contiguous_format)
            draft_ids = self._speculate(last_token_id)

        with nvtx.annotate("decoding"):
            finished = False
            last_tree_size = 0
            accept_post_tree = False

            post_count = 0
            regular_count = 0
            count = 0
            accept_post_tree = False

            while not finished:
                # * tree decoding
                
                self.draft_model.init_postspec()
                with nvtx.annotate("tree_decoding", color="orange"):
                    prev_kv_len = past_key_values.get_seq_length()
                    if self.cache_implementation == 'dynamic':
                        past_key_values.crop(prev_kv_len)
                    outputs = self._tree_decoding(draft_ids, cache_position, past_key_values)
                    next_token_logits = outputs.logits
                    del outputs

                with nvtx.annotate("update_post_tree", color="cyan"):
                    new_draft_ids = self.draft_model.update_tree_after_post()
                # * verify
                with nvtx.annotate("verify"):
                    # skip_nodes = last_tree_size if accept_post_tree else 0
                    root_ind = 0
                    sampled_tokens, hidden_indices, (total_len, accept_len) = self._verify(
                                                            draft_ids, root_ind, next_token_logits, 
                                                            logits_processor,
                                                            do_sample
                                                        )
                    
                    del next_token_logits
                    sampled_tokens = sampled_tokens.to(input_ids.device)
                   
                    if new_draft_ids[0, 0].item() == sampled_tokens[:, -1].item(): 
                        accept_post_tree = True
                    else:
                        accept_post_tree = False
                        
                
                with nvtx.annotate("update kv-cache"):
                    input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)
                    past_key_values.seq_len += sampled_tokens.shape[1]
                    count +=1
                    
                    if not accept_post_tree:
                        regular_count +=1
                        with nvtx.annotate("speculate", color="cyan"):
                            last_token_id = sampled_tokens[:, -1:].clone(memory_format=torch.contiguous_format)
                            draft_ids = self._speculate(last_token_id)
                    else:
                        post_count += 1
                        draft_ids = new_draft_ids
                    
                    cache_position = torch.arange(input_ids.shape[1]-1, input_ids.shape[1]+draft_ids.shape[1]-1, dtype=torch.long, device=input_ids.device)
                                                 
                # * check stopping criteria
                with nvtx.annotate("stopping criteria"):
                    # finished = stopping_criteria(input_ids, None).item()
                    for k in range(sampled_tokens.shape[1]):    
                        finished = stopping_criteria(sampled_tokens[:, k:k+1], None).item()
                        if finished:
                            input_ids = input_ids[:, :-(sampled_tokens.shape[1]-k-1)] if (sampled_tokens.shape[1]-k-1)>0 else input_ids
                            break
        print("count:", count, "post_count:", post_count, "regular_count:", regular_count) 
        return input_ids
    
class SubSpecSDGenerator(SDProfilingMixin, SubSpecSDGeneratorBase):
    pass