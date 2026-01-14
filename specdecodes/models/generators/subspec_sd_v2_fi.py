import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria
import nvtx

from .classic_sd import ClassicSDGeneratorBase
from ..utils.mixin import SDProfilingMixin
from ..utils.flashinfer.cache_manager import (
    RequestKvCache,
    getKvCacheBatchPosition,
)
from ..utils.flashinfer.attention_wrapper import FlashinferAttentionWrapper
from ..utils.flashinfer.prefill import flashinfer_chunked_prefill


class SubSpecSDGeneratorBase(ClassicSDGeneratorBase):
    def init_cuda_graph_runner(self, device, kvCachePool=None):
        """Initialize the draft-model CUDA graph runner (FlashInfer path)."""
        if hasattr(self.draft_model, "init_cuda_graph_runner") and callable(
            self.draft_model.init_cuda_graph_runner
        ):
            self.draft_model.init_cuda_graph_runner(device=device)

    def _draft_tree_decoding(self, tree, request_kv_cache, position_offset, skip_nodes, device):
        kv_cache_pool = request_kv_cache.kvCachePool
        tree_input_ids, tree_position_ids, tree_mask = self._prepare_tree_inputs_and_mask(
            tree,
            position_offset=position_offset,
            device=device,
            model_dtype=kv_cache_pool.cache_data[0].dtype,
            non_blocking=True,
            skip_nodes=skip_nodes,
            invert=False,
        )

        num_tokens = int(tree_input_ids.shape[0])
        if num_tokens == 0:
            return torch.empty(
                (1, 0, self.draft_model.model.config.vocab_size),
                device=device,
                dtype=self.draft_model.model.lm_head.weight.dtype,
            )

        with nvtx.annotate("draft_forward", color="red"):
            request_kv_cache.increment(num_tokens)
            batch_position = getKvCacheBatchPosition(
                request_kv_caches=[request_kv_cache],
                mode="tree",
                device=device,
                treeTokens=num_tokens,
            )
            self.flashinferWrapper.prepareAttention(
                "tree",
                batch_position,
                kv_cache_pool.page_len,
                "NONE",
                kv_cache_pool.cache_data[0].dtype,
                attention_mask=tree_mask,
            )
            logits = self.draft_model(
                tree_input_ids.unsqueeze(0),
                with_softmax=False,
                past_key_values=None,
                position_ids=tree_position_ids.unsqueeze(0),
                use_cache=False,
                kvCachePool=kv_cache_pool,
                batch_position=batch_position,
                mode="tree",
                flashinferWrapper=self.flashinferWrapper,
            )
        return logits

    def _post_verify(
        self,
        tree,
        root_ind,
        draft_request_kv_cache,
        position_offset,
        last_tree_depth,
        skip_nodes,
        logits_processor,
        device,
    ):
        next_token_logits = self._draft_tree_decoding(
            tree,
            draft_request_kv_cache,
            position_offset=position_offset,
            skip_nodes=skip_nodes,
            device=device,
        )
        sampled_tokens, _, _ = self._verify(
            tree,
            root_ind,
            next_token_logits,
            logits_processor,
            False,
            skip_nodes=skip_nodes,
        )

        accepted_len = sampled_tokens.size(1)
        tree.prune_to_depth(int(last_tree_depth) + int(accepted_len))

        refill_steps = int(self.draft_params.max_depth) - int(accepted_len)
        if refill_steps > 0:
            with nvtx.annotate("postspec_refill", color="cyan"):
                self.draft_model.init_postspec()
                for _ in range(refill_steps):
                    self.draft_model.postspec()
            tree = self.draft_model.update_tree_after_post()

        return tree

    def _tree_decoding(self, tree, request_kv_cache, position_offset, skip_nodes, device, batch_position=None):
        kv_cache_pool = request_kv_cache.kvCachePool

        tree_input_ids, tree_position_ids, tree_mask = self._prepare_tree_inputs_and_mask(
            tree,
            position_offset=position_offset,
            device=device,
            model_dtype=kv_cache_pool.cache_data[0].dtype,
            non_blocking=True,
            skip_nodes=skip_nodes,
            invert=False,
        )

        num_tokens = int(tree_input_ids.shape[0])
        if num_tokens == 0:
            return None

        with nvtx.annotate("target_forward", color="red"):
            self.flashinferWrapper.prepareAttention(
                "tree",
                batch_position,
                kv_cache_pool.page_len,
                "NONE",
                kv_cache_pool.cache_data[0].dtype,
                attention_mask=tree_mask,
            )

            outputs = self.target_model(
                input_ids=tree_input_ids.unsqueeze(0),
                past_key_values=None,
                position_ids=tree_position_ids.unsqueeze(0),
                output_hidden_states=True,
                use_cache=False,
                kvCachePool=kv_cache_pool,
                batch_position=batch_position,
                mode="tree",
                flashinferWrapper=self.flashinferWrapper,
            )
        return outputs

    def _speculate(self, input_ids, request_kv_cache):
        return self.draft_model.speculate(
            input_ids,
            request_kv_cache=request_kv_cache,
            flashinferWrapper=self.flashinferWrapper,
        )

    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
        **model_kwargs,
    ):
        """FlashInfer-backed SubSpec SD v2.

        This mirrors `subspec_sd_v2` control flow (post_verify + skip_nodes + delayed reorder),
        but uses paged KV cache via FlashInfer (`RequestKvCache`).
        """

        assert self.target_model is not None, "target_model must be provided"
        assert self.draft_model is not None, "draft_model must be provided"
        assert self.tokenizer is not None, "tokenizer must be provided"

        input_ids = input_ids.clone()
        batch_size, _ = input_ids.shape
        assert batch_size == 1, "Only support batch_size=1 for now."

        if stopping_criteria.max_length is None and self.cache_implementation == "static":
            raise ValueError(
                "max_length is not set. Only 'dynamic' kv-cache is supported when max_length is unspecified."
            )

        if model_kwargs.get("past_key_values") is None:
            raise ValueError("past_key_values should be provided")

        kv_cache_pool = model_kwargs["past_key_values"]
        max_cache_len = getattr(kv_cache_pool, "max_cache_len", None)

        stream_callback = model_kwargs.get("stream_callback", None)

        if not hasattr(self, "flashinferWrapper"):
            self.flashinferWrapper = FlashinferAttentionWrapper(
                self.target_model.config.num_attention_heads,
                self.target_model.config.num_key_value_heads,
                self.target_model.config.hidden_size,
                kv_cache_pool.page_len,
            )

        request_kv_cache = RequestKvCache(
            kvCachePool=kv_cache_pool,
            page_len=kv_cache_pool.page_len,
            seq_init_len=0,
        )
        with nvtx.annotate("prefill_chunked", color="orange"):
            # v2 can grow trees across iterations; allocate a bit more mask headroom.
            self._init_tree_mask(
                int(self.draft_params.max_verify_tokens) * 2,
                max_cache_len,
                device=input_ids.device,
            )
            outputs = flashinfer_chunked_prefill(
                target_model=self.target_model,
                flashinfer_wrapper=self.flashinferWrapper,
                input_ids=input_ids,
                kv_cache_pool=kv_cache_pool,
                request_kv_cache=request_kv_cache,
                prefill_chunk_size=self.prefill_chunk_size,
            )
            next_token_logits = outputs.logits
            del outputs

        with nvtx.annotate("sample"):
            sampled_tokens = self._sample_token(next_token_logits, logits_processor, do_sample)

        with nvtx.annotate("state_update"):
            input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)
            position_offset = input_ids.shape[1] - 1
            self._maybe_stream(stream_callback, sampled_tokens)

        with nvtx.annotate("decode_loop"):
            self.post_verify_count = 0
            self.speculate_count = 0

            finished = False
            is_prev_accepted = False
            hidden_indices_cache = None
            last_tree_size = 0
            last_tree_depth = 0
            root_ind = 0

            while not finished:
                if is_prev_accepted:
                    skip_nodes = last_tree_size
                    # with nvtx.annotate("post_verify", color="cyan"):
                    #     tree = self._post_verify(
                    #         tree,
                    #         root_ind,
                    #         request_kv_cache,
                    #         position_offset,
                    #         last_tree_depth,
                    #         skip_nodes,
                    #         logits_processor,
                    #         input_ids.device,
                    #     )
                    last_tree_size = tree.size()
                    last_tree_depth = tree.get_depth()

                else:
                    self.speculate_count += 1
                    with nvtx.annotate("speculate", color="cyan"):
                        last_token_id = sampled_tokens[:, -1:].clone(memory_format=torch.contiguous_format)
                        tree = self._speculate(last_token_id, request_kv_cache)

                    last_tree_size = tree.size()
                    last_tree_depth = tree.get_depth()
                    skip_nodes = 0
                    position_offset = input_ids.shape[1] - 1

                with nvtx.annotate("target_decode", color="orange"):
                    # Allow draft model to accumulate async post-spec data if it does so.
                    self.draft_model.init_postspec()
                    num_tokens = tree.size() - skip_nodes

                    batch_position = getKvCacheBatchPosition(
                        request_kv_caches=[request_kv_cache],
                        mode="tree",
                        device=input_ids.device,
                        treeTokens=num_tokens,
                    )
                    outputs = self._tree_decoding(
                        tree,
                        request_kv_cache,
                        position_offset=position_offset,
                        skip_nodes=skip_nodes,
                        device=input_ids.device,
                        batch_position=batch_position,
                    )
                    next_token_logits = outputs.logits if outputs is not None else None

                with nvtx.annotate("postspec_update", color="cyan"):
                    tree = self.draft_model.update_tree_after_post()

                with nvtx.annotate("verify"):
                    root_ind = root_ind if is_prev_accepted else 0
                    sampled_tokens, hidden_indices, _ = self._verify(
                        tree,
                        root_ind,
                        next_token_logits,
                        logits_processor,
                        do_sample,
                        skip_nodes=skip_nodes,
                    )

                    last_accepted_ind = hidden_indices[-1]
                    bonus_token = sampled_tokens[:, -1].item()
                    sampled_tokens = sampled_tokens.to(input_ids.device)
                    
                    hidden_indices = hidden_indices.to(input_ids.device)

                    if is_prev_accepted:
                        hidden_indices_cache = torch.cat(
                            [hidden_indices_cache, hidden_indices], dim=-1
                        )
                    else:
                        hidden_indices_cache = hidden_indices

                root_ind = tree.find_child_index(last_accepted_ind, bonus_token)
                is_prev_accepted = root_ind >= 0

                input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)

                with nvtx.annotate("stop_check"):
                    finished, input_ids, kept, prune_tokens = (
                        self._apply_tokenwise_stopping_criteria(
                            input_ids=input_ids,
                            sampled_tokens=sampled_tokens,
                            stopping_criteria=stopping_criteria,
                        )
                    )
                if kept.numel() > 0:
                    self._maybe_stream(stream_callback, kept)
                
                with nvtx.annotate("kv_reorder"):
                    if (not is_prev_accepted) or finished:
                        
                        num_new_tokens = tree.size()
                        base_target_offset = request_kv_cache.get_seq_length() - num_new_tokens + 1

                        request_kv_cache.reorder_cache_with_offset(
                            hidden_indices_cache,
                            offset=base_target_offset,
                            num_new_tokens=num_new_tokens,
                        )
                        
                        if finished:
                            request_kv_cache.decrement(int(prune_tokens))
                        
            self.post_verify_count = int(self.post_verify_count)
            self.speculate_count = int(self.speculate_count)

        request_kv_cache.release()
        return input_ids


class SubSpecSDGenerator(SDProfilingMixin, SubSpecSDGeneratorBase):
    pass
