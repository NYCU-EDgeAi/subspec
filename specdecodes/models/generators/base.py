import logging
import torch
import torch.nn as nn
from typing import Any
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, LogitNormalization
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, MaxLengthCriteria, MaxTimeCriteria, EosTokenCriteria, StopStringCriteria
from specdecodes.models.utils.cache_utils import TreeDynamicCache, TreeStaticCache


# https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py
# Several functions are simplified from GenerationMixin class.
class GeneratorBase(nn.Module):
    def __init__(self, target_model, tokenizer, draft_model=None, draft_params=None, cache_implementation="dynamic", **generator_kwargs):
        super().__init__()
        self.target_model = target_model
        self.tokenizer = tokenizer

        if draft_model is not None:
            self.draft_model = draft_model
            self.draft_params = draft_params
            self.draft_model.draft_params = draft_params
        else:
            self.draft_model = None

        self.cache_implementation = cache_implementation
        
        # Set prefill function same as forward so torch.compile() forward will not execute on prefill phase)
        self.target_model.prefill_forward = self.target_model.forward

    @property
    def config(self):
        return self.target_model.config
    
    @property
    def dtype(self):
        return self.target_model.dtype
    
    @property
    def device(self):
        return self.target_model.device
        
    def _get_logits_processor(
        self,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
    ):
        """
        Simplified HuggingFace's `LogitsProcessorList` for multinomial sampling.
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsProcessor`] instances
        used for multinomial sampling.
        Visit https://github.com/huggingface/transformers/pull/5420/files for more details.
        """
        # Instantiate warpers list
        warpers = LogitsProcessorList()
        
        if temperature is not None and temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(temperature))
        if top_k is not None and top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=top_k))
        if top_p is not None and top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=top_p))
        
        return warpers
    
    def _get_stopping_criteria(
        self,
        input_ids_length: torch.LongTensor = None,
        max_new_tokens: int = None,
        max_length: int = None,
        max_time: float = None,
        eos_token_tensor: torch.LongTensor = None,
        stop_strings: list[str] = None,
    ):
        criteria = StoppingCriteriaList()
        if max_new_tokens is not None:
            if max_length is not None:
                logging.warning(
                    f"Both `max_new_tokens` (={max_new_tokens}) and `max_length`(="
                    f"{max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                )
            max_length = input_ids_length + max_new_tokens
            
        if max_length is not None:
            max_position_embeddings = getattr(self.target_model.config, "max_position_embeddings", None)
            criteria.append(
                MaxLengthCriteria(
                    max_length=max_length,
                    max_position_embeddings=max_position_embeddings,
                )
            )
        if max_time is not None:
            criteria.append(MaxTimeCriteria(max_time=max_time))
        if stop_strings is not None:
            if self.tokenizer is None:
                raise ValueError(
                    "There are one or more stop strings, either in the arguments to `generate` or in the "
                    "model's generation config, but we could not locate a tokenizer. When generating with "
                    "stop strings, you must pass the model's tokenizer to the `tokenizer` argument of `generate`."
                )
            criteria.append(StopStringCriteria(stop_strings=stop_strings, tokenizer=self.tokenizer))
        if eos_token_tensor is not None:
            # EosTokenCriteria only checks last input token,
            # make sure not token is appended after eos_token_tensor during generation
            criteria.append(EosTokenCriteria(eos_token_id=eos_token_tensor))
        
        return criteria
    
    def _sample_token(
        self,
        logits: torch.FloatTensor,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
        return_probs: bool = False,
    ):
        if do_sample:
            batch, seq_len, vocab_size = logits.shape
            
            # Flatten logits for sampling
            logits = logits.view(-1, vocab_size)
            
            # Apply logits warper
            next_token_scores = logits_processor(None, logits)
            
            # Apply softmax to get probabilities
            probs = torch.softmax(next_token_scores, dim=-1)
            
            if return_probs: # return sample prob
                return probs.view(batch, seq_len, vocab_size) # preserve shape
            else: # return sampled token
                token = torch.multinomial(probs, num_samples=1)
                return token.view(batch, seq_len) # preserve shape

        else:
            
            if return_probs: # return sample prob
                return torch.softmax(logits, dim=-1)
            else: # return sampled token
                return torch.argmax(logits, dim=-1)

    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_processor: LogitsProcessor,
        do_sample: bool,
        *args,
        **kwargs,
    ):
        r"""
        This method is expected to be implemented by subclasses.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        temperature=None,
        top_p=None,
        top_k=None,
        max_new_tokens=None,
        max_length=None,
        do_sample=True,
        stop_strings=None,
        stream_callback=None,
        **model_kwargs,
    ):        
        # 1. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            input_ids_length=input_ids.shape[1],
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            eos_token_tensor=self.tokenizer.eos_token_id,
            stop_strings=stop_strings
        )
        
        # 2. prepare logits processor (if `do_sample` is `True`)
        logits_processor = (
            self._get_logits_processor(
                temperature=temperature, 
                top_p=top_p, 
                top_k=top_k,
            ) if do_sample else None
        )
        
        # 3. generate
        if stream_callback is not None:
            model_kwargs["stream_callback"] = stream_callback
        results = self._generate(
            input_ids=input_ids,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            do_sample=do_sample,
            **model_kwargs,
        )
        return results

    def _maybe_stream(self, stream_callback, token_ids: torch.LongTensor):
        if stream_callback is None:
            return
        stream_callback(token_ids)

    def _chunked_prefill_forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Any,
        *,
        prefill_chunk_size: int | None,
        use_position_ids: bool = True,
        model_forward_kwargs: dict[str, Any] | None = None,
        prefill_forward_kwargs: dict[str, Any] | None = None,
    ):
        """Run prefill in chunks to reduce peak memory, returning the last forward outputs.

        This helper only performs forward passes and updates `past_key_values.seq_len`.
        Callers typically read `outputs.logits` from the returned object.
        """
        model_forward_kwargs = model_forward_kwargs or {}
        prefill_forward_kwargs = prefill_forward_kwargs or {}

        current_kv_len = past_key_values.get_seq_length()
        prefill_tokens = input_ids[:, current_kv_len:]
        prefill_length = prefill_tokens.size(1)

        chunk_size = (
            prefill_length
            if prefill_chunk_size is None
            else min(prefill_length, int(prefill_chunk_size))
        )

        outputs = None
        for start in range(0, prefill_length, chunk_size):
            chunk = prefill_tokens[:, start : start + chunk_size]
            current_kv_len = past_key_values.get_seq_length()
            cache_position = torch.arange(
                current_kv_len,
                current_kv_len + chunk.size(1),
                dtype=torch.long,
                device=input_ids.device,
            )

            forward_common_kwargs: dict[str, Any] = {
                "past_key_values": past_key_values.cache,
                "cache_position": cache_position,
            }
            if use_position_ids:
                forward_common_kwargs["position_ids"] = cache_position.unsqueeze(0)

            # Last chunk returns logits (and optionally other outputs); earlier chunks only update KV.
            if start + chunk_size < prefill_length:
                self.target_model.model(chunk, **forward_common_kwargs, **model_forward_kwargs)
            else:
                outputs = self.target_model.prefill_forward(
                    chunk,
                    **forward_common_kwargs,
                    logits_to_keep=1,
                    **prefill_forward_kwargs,
                )

            past_key_values.seq_len += chunk.size(1)

        return outputs

    def _apply_tokenwise_stopping_criteria(
        self,
        input_ids: torch.LongTensor,
        sampled_tokens: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
    ):
        """Apply stopping criteria token-by-token over a generated token block.

        Returns: (finished, updated_input_ids, kept_sampled_tokens, prune_tokens)
        where `prune_tokens` counts tokens removed from the tail after stop.
        """
        finished = False
        prune_tokens = 0

        for k in range(sampled_tokens.shape[1]):
            res = stopping_criteria(sampled_tokens[:, k : k + 1], None)
            finished = bool(res.item()) if hasattr(res, "item") else bool(res)
            if finished:
                prune_tokens = sampled_tokens.shape[1] - k - 1
                if prune_tokens > 0:
                    input_ids = input_ids[:, :-prune_tokens]
                break

        kept = (
            sampled_tokens
            if prune_tokens == 0
            else sampled_tokens[:, : sampled_tokens.shape[1] - prune_tokens]
        )
        return finished, input_ids, kept, prune_tokens
    
    def create_kv_cache(
        self,
        cache_implementation,
        max_cache_len=None,
        max_batch_size=None,
        config=None,
        device=None,
        dtype=None,
    ):
        if cache_implementation == "dynamic":
            return TreeDynamicCache()
        
        elif cache_implementation == "static":
            return TreeStaticCache(
                max_cache_len=max_cache_len,
                max_batch_size=max_batch_size,
                config=config,
                device=device,
                dtype=dtype,
            )