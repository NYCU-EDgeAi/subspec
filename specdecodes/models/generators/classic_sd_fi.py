import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria
import logging
import nvtx

from .base import GeneratorBase
from ..utils.mixin import SDProfilingMixin
from ..utils.utils import DraftParams, invert_mask
from ..utils.flashinfer.cache_manager import (
    KvCachePool,
    KvCacheBatchPosition,
    RequestKvCache,
    getKvCacheBatchPosition,
    FlashInferCache
)
from ..utils.flashinfer.attention_wrapper import FlashinferAttentionWrapper

class ClassicSDGeneratorBase(GeneratorBase):
    def __init__(self, generator_kwargs, *model_args, **kwargs):
        super().__init__(*model_args, **kwargs)

        # Then handle generator-specific kwargs here
        self.generator_kwargs = generator_kwargs or {}
        self.prefill_chunk_size = self.generator_kwargs.get("prefill_chunk_size", None)

    def init_cuda_graph_runner(self,device,kvCachePool=None):
        """
        Example method to allocate a maximum-size buffer for kv_page_indices 
        and capture the forward pass using padding.
        """
        if hasattr(self.draft_model, 'init_cuda_graph_runner') and callable(self.draft_model.init_cuda_graph_runner):
            pass
            self.draft_model.init_cuda_graph_runner(device=device)
    
    def _tree_decoding(self, tree, request_kv_cache, position_offset, cache_position, device):
        # Preparing target_model's tree decoding data, also updates each node's index (node.ind).
        with nvtx.annotate("create attn mask"):
            node_data = tree.get_tree_data()
            tree_input_ids = node_data['token_ids']
            tree_position_ids = node_data['depths'] + position_offset
            tree_mask_partial = tree.create_attention_mask(position_offset)
        
        # Move to device
        with nvtx.annotate("mask to GPU"):
            tree_input_ids = tree_input_ids.to(device, non_blocking=True)
            tree_position_ids = tree_position_ids.to(device, non_blocking=True)
            tree_mask_partial = tree_mask_partial.to(device)
        
        # Assing to tree mask
        with nvtx.annotate("update mask"):
            tree_mask = self._get_tree_mask(tree_mask_partial)
               
        # llm forward
        #TODO: Remove unnecessary squeeze(0) and unsqueeze(0) operations
        with nvtx.annotate("llm forward", color="red"):
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
        # Obtain LLM sample logits
        # global_p = sample_token_method(logits, return_probs=True).squeeze(0).to(device='cpu') # remove batch dim
        global_p = self._sample_token(logits, logits_processor, do_sample, return_probs=True)
        global_p = global_p.squeeze(0).cpu()
        
        # Initialize variables
        sampled_tokens = torch.empty(0, dtype=torch.long, device='cpu')
        hidden_indices = torch.empty(0, dtype=torch.long, device='cpu')
        total_len = accept_len = 0
        # bonus_token = None
        
        # Iterate through draft tree, verify each node
        node_data = tree.get_tree_data(skip_nodes=skip_nodes)
        token_ids = node_data['token_ids']  # (num_nodes,)
        cur_ind = torch.tensor([root_ind], dtype=torch.long, device='cpu') # start at root
        children_inds = tree.get_children_indices(cur_ind)
        children_token_ids = token_ids[children_inds-skip_nodes]
        
        if(children_inds.numel() == 0):
            bonus_token = None
            
        torch.cuda.synchronize() # synchronize before starting the loop
        while children_inds.numel() > 0:
            total_len += 1
            dist = global_p[cur_ind-skip_nodes].squeeze(0)  # (vocab_size,)
            accept_token_id, bonus_token = self._verify_step(dist, children_token_ids, logits_processor, do_sample)
            
            # if accept_token_id is not None, it means the token is in the children
            # -> accept token, update cur_ind and children_inds
            # if accept_token_id is None, it means the token is not in the children
            # -> reject token, update global_p and break
            if accept_token_id is not None:
                accept_len += 1
                sampled_tokens = torch.cat([sampled_tokens, accept_token_id[None]])
                hidden_indices = torch.cat([hidden_indices, cur_ind])
                
                # Stop on EOS
                if accept_token_id == self.draft_model.eos_token_id:
                    break

                # Update
                cur_ind = children_inds[children_token_ids == accept_token_id]
                children_inds = tree.get_children_indices(cur_ind)
                children_token_ids = token_ids[children_inds-skip_nodes]
                # Stop if current index exceeds global_p shape
                if (children_inds-skip_nodes).numel() == 0 :
                    # logging.warning("No more children nodes to verify.")
                    break
                if (torch.min(children_inds-skip_nodes)).item() >= global_p.shape[0]:
                    # logging.warning(f"Current index {cur_ind-skip_nodes} exceeds global_p shape {global_p.shape[0]}.")
                    break
            
            # Reject token, break
            else:
                break
        
        # Generate bonus token, don't generate if eos token is the last token
        if sampled_tokens.numel() == 0 or sampled_tokens[-1] != self.draft_model.eos_token_id:
            if bonus_token is None:
                # print(f"Bonus token is None")
                dist = global_p[cur_ind-skip_nodes].squeeze(0)
                if do_sample:
                    bonus_token = dist.multinomial(num_samples=1).squeeze(-1)
                else:
                    bonus_token = dist.argmax()
            
            if bonus_token is not None:
                # print(f"Bonus token is not None")
                sampled_tokens = torch.cat([sampled_tokens, bonus_token.unsqueeze(0)])
                hidden_indices = torch.cat([hidden_indices, cur_ind])
        
        return sampled_tokens.unsqueeze(0), hidden_indices, (total_len, accept_len)


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
            max_cache_len = getattr(past_key_values, "max_cache_len", None)

            if model_kwargs.get("draft_past_key_values") is not None:
                draft_past_key_values = model_kwargs["draft_past_key_values"]
                # self.draft_model.set_past_key_values(draft_past_key_values)
        else:
            raise ValueError("past_key_values and draft_past_key_values should both be provided")
        
        # * prefill stage
        with nvtx.annotate("chunked prefill", color="orange"):
            self._init_tree_mask(
                self.draft_params.max_verify_tokens, max_cache_len, device=input_ids.device
            )
            # current_kv_len = past_key_values.get_seq_length()
            current_kv_len = 0
            prefill_tokens = input_ids[:, current_kv_len:]
            prefill_length = prefill_tokens.size(1)
            chunk_size = prefill_length if self.prefill_chunk_size is None else min(prefill_length, self.prefill_chunk_size)
            next_token_logits = None

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
            for start in range(0, prefill_length, chunk_size):
                chunk = prefill_tokens[:, start:start + chunk_size]
                num_new_tokens = chunk.size(1)
                
                request_kv_cache.increment(num_new_tokens)

                batch_position = getKvCacheBatchPosition(
                    request_kv_caches=[request_kv_cache],
                    mode='tree', 
                    device=input_ids.device,
                    treeTokens=num_new_tokens,
                )
                self.flashinferWrapper.prepareAttention(
                    'prefill',
                    batch_position,
                    self.kvCachePool.page_len,
                    "NONE", # POS_ENCODING_MODE.NONE,
                    self.kvCachePool.cache_data[0].dtype,
                )
                # current_kv_len = past_key_values.get_seq_length()
                # cache_position = torch.arange(
                #     current_kv_len, current_kv_len + chunk.size(1),
                #     dtype=torch.long, device=input_ids.device
                # )
                # last iteration
                if start + chunk_size < prefill_length:
                    # does not need output logits, just update kv-cache
                    outputs = self.target_model.prefill_forward(
                        input_ids=chunk,
                        past_key_values=None,
                        use_cache=False,
                        
                        kvCachePool=self.kvCachePool,
                        batch_position=batch_position,
                        mode='prefill', 
                        flashinferWrapper = self.flashinferWrapper,
                    )
                else:
                    outputs = self.target_model.prefill_forward(
                        input_ids=chunk,
                        past_key_values=None,
                        use_cache=False,
                        logits_to_keep=1,
                        
                        kvCachePool=self.kvCachePool,
                        batch_position=batch_position,
                        mode='prefill', 
                        flashinferWrapper = self.flashinferWrapper,
                    )
            next_token_logits = outputs.logits
            del outputs
                
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
                    last_token_ids = input_ids[:, draft_request_kv_cache.get_seq_length():].clone(memory_format=torch.contiguous_format)
                    tree = self._speculate(last_token_ids, draft_request_kv_cache)

                # * tree decoding
                with nvtx.annotate("tree_decoding", color="orange"):
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
                    # print(f"Current sampled ({sampled_tokens.shape[1]}):", self.tokenizer.batch_decode(sampled_tokens.squeeze(0), skip_special_tokens=False))
                    
                with nvtx.annotate("reorder kv"):
                    num_new_tokens = self.draft_params.max_verify_tokens
                    request_kv_cache.reorder_cache_with_offset(hidden_indices, offset=prev_kv_len, num_new_tokens=num_new_tokens)
                    print(f"draft_request_kv_cache seq_len before reorder: {draft_request_kv_cache.get_seq_length()}")
                    draft_request_kv_cache.reorder_cache_with_offset(hidden_indices, offset=draft_request_kv_cache.get_seq_length(), num_new_tokens=num_new_tokens)
                    print(f"draft_request_kv_cache seq_len after reorder: {draft_request_kv_cache.get_seq_length()}")
                    input("Press Enter to continue...")

                # * update input_ids and cache_position
                with nvtx.annotate("update data"):
                    input_ids = torch.cat([input_ids, sampled_tokens], dim=-1)
                    cache_position += sampled_tokens.shape[1]
                
                # * check stopping criteria
                with nvtx.annotate("stopping criteria"):
                    finished = stopping_criteria(input_ids, None).item()
        request_kv_cache.release()   
        draft_request_kv_cache.release()  
        return input_ids
    
class ClassicSDGenerator(SDProfilingMixin, ClassicSDGeneratorBase):
    pass