import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria
import logging
import nvtx

from .classic_sd import ClassicSDGeneratorBase
from ..utils.mixin import SDProfilingMixin
from ..utils.utils import DraftParams, invert_mask
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_EXCEPTION


class SubSpecSDGeneratorBase(ClassicSDGeneratorBase):
    def __init__(self, generator_kwargs, *model_args, **kwargs):
        super().__init__(generator_kwargs, *model_args, **kwargs)
        post_draft_params = kwargs.get("post_draft_params", None)
        if post_draft_params is None:
           raise ValueError("post_draft_params must be provided in subspec_sd_v2 generator.")
        self.post_draft_params = post_draft_params
        self.draft_model.post_draft_params = post_draft_params
        
    def _tree_decoding(self, tree, tree_mask, past_key_values, position_offset, cache_position, skip_nodes, device):
        # Preparing target_model's tree decoding data, also updates each node's index (node.ind).
        with nvtx.annotate("create attn mask"):
            node_data = tree.get_tree_data(skip_nodes)
            tree_input_ids = node_data['token_ids']
            tree_position_ids = node_data['depths'] + position_offset
            tree_mask_partial = tree.create_attention_mask(position_offset, skip_nodes)
          
        # Move to device
        with nvtx.annotate("mask to GPU"):
            tree_input_ids = tree_input_ids.to(device)
            tree_position_ids = tree_position_ids.to(device)
            tree_mask_partial = tree_mask_partial.to(device)
        
        # Assing to tree mask
        with nvtx.annotate("update mask"):
            tree_mask = self._update_tree_mask(tree_mask, tree_mask_partial)
            tree_mask = invert_mask(tree_mask, dtype=self.target_model.model.dtype)
        
        # llm forward
        #TODO: Remove unnecessary squeeze(0) and unsqueeze(0) operations
        with nvtx.annotate("llm forward", color="red"):
            # print("cache_position:", cache_position)
            outputs = self.target_model(
                tree_input_ids.unsqueeze(0),
                past_key_values=past_key_values.cache,
                attention_mask=tree_mask,
                position_ids=tree_position_ids.unsqueeze(0),
                cache_position=cache_position
            )
        return outputs
    
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
            tree_mask = self._init_tree_mask(
                self.draft_params.max_verify_tokens+self.post_draft_params.max_verify_tokens, max_cache_len, device=input_ids.device
            )
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
            position_offset = input_ids.shape[1]-1
            cache_position = torch.arange(org_input_len, org_input_len+self.draft_params.max_verify_tokens, dtype=torch.long, device=input_ids.device)

        with nvtx.annotate("speculate", color="cyan"):
            last_token_id = sampled_tokens[:, -1:].clone(memory_format=torch.contiguous_format)
            tree = self._speculate(last_token_id)

        with nvtx.annotate("decoding"):
            finished = False
            count = 0
            post_count = 0
            regular_count = 0

            accept_post_tree = False
            sampled_tokens_cache = None
            hidden_indices_cache = None
            last_tree_size = 0
            while not finished:
                # * tree decoding
                with nvtx.annotate("tree_decoding", color="orange"):
                    
                    skip_nodes = last_tree_size if accept_post_tree else 0
                    self.draft_model.init_postspec()
                    
                    outputs = self._tree_decoding(tree, tree_mask, past_key_values, position_offset=position_offset, cache_position=cache_position, skip_nodes=skip_nodes, device=input_ids.device)
                    next_token_logits = outputs.logits
                    tem_tree_size = tree.size()
                
                with nvtx.annotate("update_post_tree", color="cyan"):
                    tree = self.draft_model.update_tree_after_post()
                
                # * verify
                with nvtx.annotate("verify"): 
                    skip_nodes = last_tree_size if accept_post_tree else 0
                    root_ind = root_ind if accept_post_tree else 0
                    
                    sampled_tokens, hidden_indices, (total_len, sampled_len) = self._verify(
                                                            tree, root_ind ,next_token_logits, 
                                                            logits_processor,
                                                            do_sample, 
                                                            skip_nodes=skip_nodes,
                                                        )
                    
                    last_accepted_ind = hidden_indices[-1]
                    bonus_token = sampled_tokens[:, -1].item()
                    sampled_tokens = sampled_tokens.to(input_ids.device)
                    hidden_indices = hidden_indices.to(input_ids.device)

                    if accept_post_tree:
                        sampled_tokens_cache = torch.cat([sampled_tokens_cache, sampled_tokens], dim=-1)
                        hidden_indices_cache = torch.cat([hidden_indices_cache, hidden_indices], dim=-1)
                    else:
                        sampled_tokens_cache = sampled_tokens
                        hidden_indices_cache = hidden_indices

                # print(f"Current sampled ({sampled_tokens_cache.shape[1]}):", self.tokenizer.batch_decode(sampled_tokens_cache.squeeze(0), skip_special_tokens=False))
                
                last_tree_size = tem_tree_size                
                root_ind = tree.find_child_index(last_accepted_ind, bonus_token)
                
                if root_ind is not None:
                    accept_post_tree = True 
                else:
                    accept_post_tree = False 

                if (sampled_tokens_cache.shape[1]>100):
                    accept_post_tree = False 
                
                # * check stopping criteria
                with nvtx.annotate("stopping criteria"):
                    finished = stopping_criteria(sampled_tokens_cache, None).item()

                with nvtx.annotate("reorder kv"):
                    count += 1
                    if finished:
                        input_ids = torch.cat([input_ids, sampled_tokens_cache], dim=-1)
                    elif not accept_post_tree:
                        # print("reject post tree,re speculate")
                        regular_count += 1
                        past_key_values.reorder_cache_with_offset(hidden_indices_cache, offset=past_key_values.get_seq_length(), new_chunk_len=last_tree_size, dim=2)
                        past_key_values.seq_len += hidden_indices_cache.shape[0]
                        input_ids = torch.cat([input_ids, sampled_tokens_cache], dim=-1)
                        
                        with nvtx.annotate("speculate", color="cyan"):
                            last_token_id = sampled_tokens[:, -1:].clone(memory_format=torch.contiguous_format)
                            tree = self._speculate(last_token_id)
                            last_tree_size = tree.size()

                        position_offset = input_ids.shape[1] - 1
                        cache_position = torch.arange(input_ids.shape[1]-1, input_ids.shape[1]-1+tree.size(), dtype=torch.long, device=input_ids.device)
                    else:  
                        # print("accept post tree")
                        post_count += 1
                        position_offset = input_ids.shape[1] - 1 
                        cache_position = torch.arange(input_ids.shape[1]+last_tree_size -1, input_ids.shape[1]+tree.size()-1, dtype=torch.long, device=input_ids.device)                     
                        
        print("count:", count, "post_count:", post_count, "regular_count:", regular_count)        
        return input_ids
    
class SubSpecSDGenerator(SDProfilingMixin, SubSpecSDGeneratorBase):
    pass