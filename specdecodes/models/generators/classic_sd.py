import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria
import logging
import nvtx

from .base import GeneratorBase
from ..utils.mixin import SDProfilingMixin
from ..utils.utils import DraftParams, invert_mask
from ..utils.tree_verify import verify_tree

class ClassicSDGeneratorBase(GeneratorBase):
    def __init__(self, generator_kwargs, *model_args, **kwargs):
        super().__init__(*model_args, **kwargs)
        self.generator_kwargs = generator_kwargs or {}
        self.prefill_chunk_size = self.generator_kwargs.get("prefill_chunk_size", None)
        
    def _speculate(self, input_ids):
        return self.draft_model.speculate(input_ids)
        
    def _init_tree_mask(self, max_verify_tokens, max_cache_len=None, device='cpu'):
        if not hasattr(self, 'tree_mask_update_method'):
            self.tree_mask_update_method = 'static' if max_cache_len is not None else 'dynamic'
            logging.debug(f"'max_cache_len' is {'set, uses static' if max_cache_len else 'not set, uses dynamic'} tree_mask.")

        tree_mask = (
            torch.zeros((1, 1, max_verify_tokens, max_cache_len), device=device, dtype=torch.bool)
            if max_cache_len is not None else None
        )
        self.base_tree_mask = tree_mask
        return tree_mask

    def _get_tree_mask(self, tree_mask_partial):
        if self.tree_mask_update_method == 'static':
            # Avoid prints in hot path; use logging if needed.
            _, _, K, D = tree_mask_partial.shape

            # If the preallocated buffer is missing or too small, fall back to the
            # partial mask (dynamic behavior). This prevents intermittent shape
            # mismatch errors when the cache reports an unexpected small length.
            if (
                self.base_tree_mask is None
                or self.base_tree_mask.shape[2] < K
                or self.base_tree_mask.shape[3] < D
            ):
                return tree_mask_partial

            # Slice to the same shape as the partial input
            tree_mask_view = self.base_tree_mask[:, :, :K, :].clone()
            tree_mask_view[:, :, :K, :D] = tree_mask_partial

            # Return view with the correct shape
            return tree_mask_view
        else:
            return tree_mask_partial

    def _tree_decoding(self, tree, past_key_values, position_offset, cache_position, device):
        # Preparing target_model's tree decoding data, also updates each node's index (node.ind).
        with nvtx.annotate("attn_mask/build"):
            node_data = tree.get_tree_data()
            tree_input_ids = node_data['token_ids']
            tree_position_ids = node_data['depths'] + position_offset
            tree_mask_partial = tree.create_attention_mask(position_offset)
        
        # Move to device
        with nvtx.annotate("attn_mask/to_device"):
            tree_input_ids = tree_input_ids.to(device)
            tree_position_ids = tree_position_ids.to(device)
            tree_mask_partial = tree_mask_partial.to(device)
        
        # Assign to tree mask
        with nvtx.annotate("attn_mask/prepare"):
            tree_mask = self._get_tree_mask(tree_mask_partial)
            tree_mask = invert_mask(tree_mask, dtype=self.target_model.model.dtype)
        
        # Target model forward
        # NOTE: Some squeeze/unsqueeze is legacy for shape alignment.
        with nvtx.annotate("target_forward", color="red"):
            outputs = self.target_model(
                tree_input_ids.unsqueeze(0),
                past_key_values=past_key_values.cache,
                attention_mask=tree_mask,
                position_ids=tree_position_ids.unsqueeze(0),
                cache_position=cache_position
            )
        return outputs
 
    def _verify_step(self, p, token_ids, logits_processor, do_sample):
        sampled_token_id = p.argmax() if not do_sample else p.multinomial(1).squeeze(-1)
        if torch.any(sampled_token_id == token_ids):
            return sampled_token_id, None
        else:
            return None, sampled_token_id
        
    def _verify(self, tree, root_ind ,logits, logits_processor, do_sample,skip_nodes=0):
        verify_method = str(self.generator_kwargs.get("verify_method", "exact") or "exact").strip().lower()
        verify_kwargs = dict(self.generator_kwargs.get("verify_kwargs") or {})

        return verify_tree(
            tree=tree,
            root_ind=int(root_ind),
            logits=logits,
            sample_token_fn=self._sample_token,
            verify_step_fn=self._verify_step,
            eos_token_id=getattr(self.draft_model, "eos_token_id", None),
            logits_processor=logits_processor,
            do_sample=do_sample,
            skip_nodes=int(skip_nodes),
            verify_method=verify_method,
            verify_kwargs=verify_kwargs,
        )


    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
        **model_kwargs,
    ):
        """Generate a token sequence with speculative decoding.

        Stages:
        - Prefill: run the target model on the prompt, then sample the next token.
        - Decode loop:
            1) Draft proposes candidate tokens (tree form).
            2) Target scores candidates in one forward.
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

        stream_callback = model_kwargs.get("stream_callback", None)
        
        # * prefill stage
        with nvtx.annotate("prefill_chunked", color="orange"):
            self._init_tree_mask(
                self.draft_params.max_verify_tokens, max_cache_len, device=input_ids.device
            )
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
                # * speculate
                with nvtx.annotate("speculate", color="cyan"):
                    input_ids = input_ids.clone(memory_format=torch.contiguous_format)
                    tree = self._speculate(input_ids)
                    if self.cache_implementation == 'dynamic':
                        _, input_len = input_ids.shape
                        draft_past_key_values.crop(input_len)

                # * tree decoding
                with nvtx.annotate("target_decode", color="orange"):
                    prev_kv_len = past_key_values.get_seq_length()
                    outputs = self._tree_decoding(tree, past_key_values, position_offset=input_ids.shape[1]-1, cache_position=cache_position, device=input_ids.device)
                    next_token_logits = outputs.logits
                    del outputs

                # * verify
                with nvtx.annotate("verify"):
                    root_ind = 0
                    sampled_tokens, hidden_indices, (total_len, accept_len) = self._verify(
                                                        tree, root_ind, next_token_logits, 
                                                        logits_processor,
                                                        do_sample
                                                    )
                    
                    sampled_tokens = sampled_tokens.to(input_ids.device)
                    del next_token_logits
                    
                # * update input_ids and cache_position
                with nvtx.annotate("state_update"):
                    input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)
                    cache_position += sampled_tokens.shape[1]
                
                # * check stopping criteria
                with nvtx.annotate("stop_check"):
                    finished, input_ids, kept, prune_tokens = self._apply_tokenwise_stopping_criteria(
                        input_ids=input_ids,
                        sampled_tokens=sampled_tokens,
                        stopping_criteria=stopping_criteria,
                    )
                if kept.numel() > 0:
                    self._maybe_stream(stream_callback, kept)
                                
                with nvtx.annotate("kv_reorder"):
                    past_key_values.reorder_cache_with_offset(hidden_indices, offset=prev_kv_len, new_chunk_len=self.draft_params.max_verify_tokens, dim=2)
                    past_key_values.seq_len += hidden_indices.shape[0]
                    if finished:
                        past_key_values.seq_len -= prune_tokens
                    
        return input_ids
    
class ClassicSDGenerator(SDProfilingMixin, ClassicSDGeneratorBase):
    pass