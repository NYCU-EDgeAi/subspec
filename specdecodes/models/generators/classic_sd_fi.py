import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria
import logging
import nvtx

from .base import GeneratorBase
from ..utils.mixin import SDProfilingMixin
from ..utils.utils import DraftParams, invert_mask
from ..utils.tree_verify import verify_tree
from ..utils.flashinfer.cache_manager import (
    KvCachePool,
    KvCacheBatchPosition,
    RequestKvCache,
    getKvCacheBatchPosition,
    FlashInferCache
)
from ..utils.flashinfer.attention_wrapper import FlashinferAttentionWrapper
from ..utils.flashinfer.prefill import flashinfer_chunked_prefill

class ClassicSDGeneratorBase(GeneratorBase):
    def __init__(self, generator_kwargs, *model_args, **kwargs):
        super().__init__(*model_args, **kwargs)

        # Then handle generator-specific kwargs here
        self.generator_kwargs = generator_kwargs or {}
        self.prefill_chunk_size = self.generator_kwargs.get("prefill_chunk_size", None)

    def init_cuda_graph_runner(self,device,kvCachePool=None):
        """
        Initialize the draft model CUDA-graph runner (FlashInfer path only).
        """
        if hasattr(self.draft_model, 'init_cuda_graph_runner') and callable(self.draft_model.init_cuda_graph_runner):
            self.draft_model.init_cuda_graph_runner(device=device)

    def _tree_decoding(self, tree, request_kv_cache, position_offset, cache_position, device):
        # Preparing target_model's tree decoding data, also updates each node's index (node.ind).
        with nvtx.annotate("attn_mask/build"):
            node_data = tree.get_tree_data()
            tree_input_ids = node_data['token_ids']
            tree_position_ids = node_data['depths'] + position_offset
            tree_mask_partial = tree.create_attention_mask(position_offset)
        
        # Move to device
        with nvtx.annotate("attn_mask/to_device"):
            tree_input_ids = tree_input_ids.to(device, non_blocking=True)
            tree_position_ids = tree_position_ids.to(device, non_blocking=True)
            tree_mask_partial = tree_mask_partial.to(device)
        
        # Assign to tree mask
        with nvtx.annotate("attn_mask/prepare"):
            tree_mask = self._get_tree_mask(tree_mask_partial)
               
        # Target model forward
        # NOTE: Some squeeze/unsqueeze is legacy for shape alignment.
        with nvtx.annotate("target_forward", color="red"):
            num_tokens = self.draft_params.max_verify_tokens
            kvCachePool = request_kv_cache.kvCachePool
            
            request_kv_cache.increment(num_tokens)

            batch_position = getKvCacheBatchPosition(
                request_kv_caches=[request_kv_cache],
                mode='tree',  # Set to False if you're doing incremental decoding
                device=device,
                treeTokens=num_tokens,
            )
            # batch_position.print_info() 
            self.flashinferWrapper.prepareAttention(
                'tree',
                batch_position,
                kvCachePool.page_len,
                "NONE", # POS_ENCODING_MODE.NONE,
                kvCachePool.cache_data[0].dtype,
                attention_mask=tree_mask,
            )
            # Check if the current instance has the attribute 'graph'
            if hasattr(self, 'graph'):
                outputs = self.tree_decoding_step(
                    input_ids=tree_input_ids.unsqueeze(0),
                    position_ids=tree_position_ids.unsqueeze(0),
                    batch_position=batch_position,
                )
            else:
                outputs = self.target_model(
                    input_ids=tree_input_ids.unsqueeze(0),
                    past_key_values=None,
                    position_ids=tree_position_ids.unsqueeze(0),
                    output_hidden_states=True,
                    use_cache=False,
                    kvCachePool=kvCachePool,
                    batch_position=batch_position,
                    mode='tree', 
                    flashinferWrapper = self.flashinferWrapper,
                )
        return outputs
    def _speculate(self, input_ids, request_kv_cache):
        return self.draft_model.speculate(
            input_ids,
            request_kv_cache=request_kv_cache,
            flashinferWrapper=self.flashinferWrapper,
        )
    
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

            # Slice to the same shape as the partial input
            tree_mask_view = self.base_tree_mask[:, :, :K, :].clone()
            tree_mask_view[:, :, :K, :D] = tree_mask_partial

            # Return view with the correct shape
            return tree_mask_view
        else:
            return tree_mask_partial

    def _verify_step(self, p, token_ids, logits_processor, do_sample):
        sampled_token_id = p.argmax() if not do_sample else p.multinomial(1).squeeze(-1)
        if torch.any(sampled_token_id == token_ids):
            return sampled_token_id, None
        else:
            return None, sampled_token_id
        
    def _verify(self, tree, root_ind ,logits, logits_processor, do_sample,skip_nodes=0):
        lossy_enabled = bool(getattr(self, "generator_kwargs", {}).get("lossy_verify", False))
        lossy_threshold = float(getattr(self.draft_params, "lossy_threshold", 0.0))
        lossy_window_size = int(getattr(self.draft_params, "lossy_window_size", 1))

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
            lossy=lossy_enabled,
            lossy_threshold=lossy_threshold,
            lossy_window_size=lossy_window_size,
        )


    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
        **model_kwargs,
    ):
        """Generate a token sequence with speculative decoding (FlashInfer-backed).

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
            max_cache_len = getattr(past_key_values, "max_cache_len", None)

            if model_kwargs.get("draft_past_key_values") is not None:
                draft_past_key_values = model_kwargs["draft_past_key_values"]
                # self.draft_model.set_past_key_values(draft_past_key_values)
        else:
            raise ValueError("past_key_values and draft_past_key_values should both be provided")

        stream_callback = model_kwargs.get("stream_callback", None)
        
        # * prefill stage
        with nvtx.annotate("prefill_chunked", color="orange"):
            self._init_tree_mask(
                self.draft_params.max_verify_tokens, max_cache_len, device=input_ids.device
            )
            if not hasattr(self, 'flashinferWrapper'):
                self.flashinferWrapper = FlashinferAttentionWrapper(
                    self.target_model.config.num_attention_heads, self.target_model.config.num_key_value_heads, self.target_model.config.hidden_size,past_key_values.page_len
                )

            self.kvCachePool = past_key_values
            request_kv_cache = RequestKvCache(
                kvCachePool=self.kvCachePool,
                page_len=self.kvCachePool.page_len,
                seq_init_len=0
            )
            draft_request_kv_cache = RequestKvCache(
                kvCachePool=draft_past_key_values,
                page_len=draft_past_key_values.page_len,
                seq_init_len=0
            )
            outputs = flashinfer_chunked_prefill(
                target_model=self.target_model,
                flashinfer_wrapper=self.flashinferWrapper,
                input_ids=input_ids,
                kv_cache_pool=self.kvCachePool,
                request_kv_cache=request_kv_cache,
                prefill_chunk_size=self.prefill_chunk_size,
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
                    last_token_ids = input_ids[:, draft_request_kv_cache.get_seq_length():].clone(memory_format=torch.contiguous_format)
                    tree = self._speculate(last_token_ids, draft_request_kv_cache)

                # * tree decoding
                with nvtx.annotate("target_decode", color="orange"):
                    prev_kv_len = request_kv_cache.get_seq_length() + 1
                    outputs = self._tree_decoding(tree, request_kv_cache, position_offset=input_ids.shape[1]-1, cache_position=cache_position, device=input_ids.device)
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
                    
                with nvtx.annotate("kv_reorder"):
                    num_new_tokens = self.draft_params.max_verify_tokens
                    request_kv_cache.reorder_cache_with_offset(hidden_indices, offset=prev_kv_len, num_new_tokens=num_new_tokens)
                    draft_request_kv_cache.reorder_cache_with_offset(hidden_indices, offset=draft_request_kv_cache.get_seq_length(), num_new_tokens=num_new_tokens)

                # * update input_ids and cache_position
                with nvtx.annotate("state_update"):
                    input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)
                    cache_position += sampled_tokens.shape[1]
                    self._maybe_stream(stream_callback, sampled_tokens)
                
                # * check stopping criteria
                with nvtx.annotate("stop_check"):
                    finished = stopping_criteria(input_ids, None).item()
        request_kv_cache.release()   
        draft_request_kv_cache.release()  
        return input_ids
    
class ClassicSDGenerator(SDProfilingMixin, ClassicSDGeneratorBase):
    pass