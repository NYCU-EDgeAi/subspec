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
    def _draft_tree_decoding(self, tree, past_key_values, position_offset, cache_position, skip_nodes, device):
        # Preparing draft_model's tree decoding data, also updates each node's index (node.ind).
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
        with nvtx.annotate("get mask"):
            tree_mask = self._get_tree_mask(tree_mask_partial)
            tree_mask = invert_mask(tree_mask, dtype=self.draft_model.model.dtype)
        
        # draft_model llm forward
        with nvtx.annotate("draft llm forward", color="red"):
            # print("cache_position:", cache_position)
            next_token_logits = self.draft_model(
                tree_input_ids.unsqueeze(0),
                past_key_values=past_key_values.cache,
                attention_mask=tree_mask,
                position_ids=tree_position_ids.unsqueeze(0),
                cache_position=cache_position
            )
        return next_token_logits
    
    def _post_verify(self, tree, root_ind, past_key_values, position_offset, cache_position, last_tree_depth, skip_nodes, logits_processor, device):
        next_token_logits = self._draft_tree_decoding(tree, past_key_values, position_offset=position_offset, cache_position=cache_position, skip_nodes=skip_nodes, device=device)
        sampled_tokens, _, (total_len, sampled_len) = self._verify(
                                                tree, root_ind, next_token_logits, 
                                                logits_processor,
                                                False, 
                                                skip_nodes=skip_nodes,
                                            )
        
        # print("Speculative tree before prune:")
        # tree.print(tokenizer=self.tokenizer)
        # print(f"pv-sampled_tokens ({sampled_tokens.shape}):", self.tokenizer.batch_decode(sampled_tokens.squeeze(0)))
        # print("tree depth before prune:", tree.get_depth())
        
        # print("----- Prune tree -----")
        keep_tokens = sampled_tokens.size(1)
        tree.prune_to_depth(last_tree_depth+keep_tokens)
        # print("pruned to depth:", last_tree_depth+keep_tokens)
        # print("tree depth after prune:", tree.get_depth())
        
        # print("Speculative tree after prune:")
        # tree.print(tokenizer=self.tokenizer)
        
        # # speculate to refill the tree
        refill_steps = self.draft_params.max_depth - keep_tokens
        # print(f"refill_steps: {refill_steps}")
        if refill_steps > 0:
            # print("----- Refill tree -----")
            with nvtx.annotate("post speculate", color="cyan"):
                self.draft_model.init_postspec()
                for _ in range(refill_steps):
                    self.draft_model.postspec()
            tree = self.draft_model.update_tree_after_post()
            
        return tree

    def _tree_decoding(self, tree, past_key_values, position_offset, cache_position, skip_nodes, device):
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
        with nvtx.annotate("get mask"):
            tree_mask = self._get_tree_mask(tree_mask_partial)
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

        stream_callback = model_kwargs.get("stream_callback", None)

        # * prefill stage
        with nvtx.annotate("chunked prefill", color="orange"):
            self._init_tree_mask(
                self.draft_params.max_verify_tokens*2, max_cache_len, device=input_ids.device
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
            position_offset = input_ids.shape[1] - 1
            self._maybe_stream(stream_callback, sampled_tokens)

        with nvtx.annotate("decoding"):
            # Better naming:
            # - `post_verify_count`: previous speculative tree was fully accepted, so we run `post_verify` instead of re-speculating.
            # - `speculate_count`: we had to run a fresh speculation step.
            self.post_verify_count = 0
            self.speculate_count = 0

            finished = False
            is_prev_accepted = False
            hidden_indices_cache = None
            last_tree_size = 0
            last_tree_depth = 0

            while not finished:
                # * speculate only if not previous accepted
                if is_prev_accepted:
                    self.post_verify_count += 1
                    # print("----- Post-speculation -----")
                    skip_nodes = last_tree_size
                    cache_position = torch.arange(position_offset+last_tree_size, position_offset+tree.size(), dtype=torch.long, device=input_ids.device)
                    with nvtx.annotate("post_verify", color="cyan"):
                        tree = self._post_verify(tree, root_ind, past_key_values, position_offset, cache_position, last_tree_depth, skip_nodes, logits_processor, input_ids.device)

                    # NOTE: `_post_verify` can prune/refill the tree (post-spec), changing `tree.size()`.
                    # `cache_position` must be recomputed to match the *updated* tree slice we will decode.
                    cache_position = torch.arange(
                        position_offset + skip_nodes,
                        position_offset + tree.size(),
                        dtype=torch.long,
                        device=input_ids.device,
                    )
                    last_tree_size = tree.size()
                    last_tree_depth = tree.get_depth()

                else:
                    self.speculate_count += 1
                    # print("----- Regular speculation -----")
                    last_token_id = sampled_tokens[:, -1:].clone(memory_format=torch.contiguous_format)
                    with nvtx.annotate("speculate", color="cyan"):
                        tree = self._speculate(last_token_id)
                    last_tree_size = tree.size()
                    last_tree_depth = tree.get_depth()
                    
                    skip_nodes = 0
                    position_offset = input_ids.shape[1] - 1
                    cache_position = torch.arange(position_offset, position_offset+tree.size(), dtype=torch.long, device=input_ids.device)
                        
                # * tree decoding
                # print("----- Verify -----")
                with nvtx.annotate("tree_decoding", color="orange"):
                    self.draft_model.init_postspec()
                    outputs = self._tree_decoding(tree, past_key_values, position_offset=position_offset, cache_position=cache_position, skip_nodes=skip_nodes, device=input_ids.device)
                    next_token_logits = outputs.logits
                
                with nvtx.annotate("update_post_tree", color="cyan"):
                    tree = self.draft_model.update_tree_after_post()
                
                # * verify
                with nvtx.annotate("verify"):
                    root_ind = root_ind if is_prev_accepted else 0
                    sampled_tokens, hidden_indices, (total_len, sampled_len) = self._verify(
                                                            tree, root_ind, next_token_logits, 
                                                            logits_processor,
                                                            do_sample, 
                                                            skip_nodes=skip_nodes,
                                                        )
                    
                    last_accepted_ind = hidden_indices[-1]
                    bonus_token = sampled_tokens[:, -1].item()
                    sampled_tokens = sampled_tokens.to(input_ids.device)
                    hidden_indices = hidden_indices.to(input_ids.device)

                    if is_prev_accepted:
                        hidden_indices_cache = torch.cat([hidden_indices_cache, hidden_indices], dim=-1)
                    else:
                        hidden_indices_cache = hidden_indices
               
                root_ind = tree.find_child_index(last_accepted_ind, bonus_token)
                if root_ind >= 0:
                    is_prev_accepted = True
                else:
                    is_prev_accepted = False

                # print("sampled_tokens:", self.tokenizer.batch_decode(sampled_tokens.squeeze(0)))
                input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)
                
                # * check stopping criteria
                with nvtx.annotate("stopping criteria"):
                    prune_tokens = 0
                    for k in range(sampled_tokens.shape[1]):    
                        finished = stopping_criteria(sampled_tokens[:, k:k+1], None).item()
                        if finished:
                            prune_tokens = sampled_tokens.shape[1]-k-1
                            input_ids = input_ids[:, :-prune_tokens] if prune_tokens > 0 else input_ids
                            break

                kept = sampled_tokens if prune_tokens == 0 else sampled_tokens[:, : sampled_tokens.shape[1] - prune_tokens]
                if kept.numel() > 0:
                    self._maybe_stream(stream_callback, kept)
                    
                with nvtx.annotate("reorder kv"):
                    if not is_prev_accepted or finished:
                        past_key_values.reorder_cache_with_offset(hidden_indices_cache, offset=past_key_values.get_seq_length(), new_chunk_len=last_tree_size, dim=2)
                        past_key_values.seq_len += hidden_indices_cache.shape[0]
                        if finished:
                            past_key_values.seq_len -= prune_tokens


            # Normalize to plain ints for logging/consumers.
            self.post_verify_count = int(self.post_verify_count)
            self.speculate_count = int(self.speculate_count)

        return input_ids
    
class SubSpecSDGenerator(SDProfilingMixin, SubSpecSDGeneratorBase):
    pass