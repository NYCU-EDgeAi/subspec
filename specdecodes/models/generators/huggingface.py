import torch
import logging

from .base import GeneratorBase
from ..utils.mixin import ProfilingMixin

class HuggingFaceGeneratorBase(GeneratorBase):
    def __init__(self, *model_args, **kwargs):
        super().__init__(*model_args, **kwargs)
    
    def generate(
        self, 
        input_ids: torch.LongTensor, 
        temperature=None, top_p=None, top_k=None, 
        max_length=2048, do_sample=True, 
        *args,
        **kwargs
    ):
        assert self.target_model is not None, "target_model must be provided"
        
        if self.cache_implementation == "dynamic":
            self.target_model.generation_config.cache_implementation = None
        else:
            self.target_model.generation_config.cache_implementation = self.cache_implementation
        # Let HF manage its own cache.
        if 'past_key_values' in kwargs:
            kwargs.pop('past_key_values')
        # Not supported via this wrapper (handled at API layer if needed)
        if 'stream_callback' in kwargs:
            kwargs.pop('stream_callback')

        return self.target_model.generate(
            input_ids=input_ids,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_length=max_length,
            do_sample=do_sample,
            *args,
            **kwargs,
        )
        
class HuggingFaceGenerator(ProfilingMixin, HuggingFaceGeneratorBase):
    pass