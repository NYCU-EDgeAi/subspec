import logging
import torch
import torch.nn as nn
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper, LogitNormalization
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, MaxLengthCriteria, MaxTimeCriteria, EosTokenCriteria
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
        **model_kwargs,
    ):        
        # 1. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            input_ids_length=input_ids.shape[1],
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            eos_token_tensor=self.tokenizer.eos_token_id
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
        results = self._generate(
            input_ids=input_ids,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            do_sample=do_sample,
            **model_kwargs,
        )
        return results
    
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