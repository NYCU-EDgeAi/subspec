import logging
from .app_router import run_app
from .base_builder import GeneratorPipelineBuilder

import torch
from specdecodes.models.utils.utils import DraftParams
from specdecodes.models.utils.cache_utils import create_kv_cache
from specdecodes.models.draft_models.subspec_seq_sd import SubSpecSDDraftModel
from specdecodes.models.generators.subspec_seq_sd_v2 import SubSpecSDGenerator

from specdecodes.helpers.recipes.subspec.hqq_4bit_attn_4bit_mlp_postspec import Recipe

class SubSpecSDBuilder(GeneratorPipelineBuilder):
    def __init__(self):
        super().__init__()
        # Base configurations.
        self.vram_limit_gb = None
        self.seed = 0
        self.device = "cuda:0"
        self.dtype = torch.float16
        self.max_length = 10 * 1024
        
        # Model paths.
        self.llm_path = "Qwen/Qwen3-8B"
        # self.llm_path = "meta-llama/Llama-3.1-8B-Instruct"
        
        # Generation parameters.
        self.do_sample = False
        self.temperature = 0
        
        # Generator-specific configurations.
        self.generator_kwargs = {
            "prefill_chunk_size": 4096,
        }
        self.draft_params = DraftParams(
            temperature=1,
            max_depth=16,
            topk_len=1,
        )

        self.post_draft_params = DraftParams(
            temperature=1,
            max_depth=16,
            topk_len=1,
        )
        
        # Recipe for quantization and offloading.
        self.recipe = Recipe()
        
        # Additional configurations.
        self.cache_implementation = "static"
        self.warmup_iter = 1
        self.compile_mode = "max-autotune"
        
        # Profiling.
        self.generator_profiling = True
    
    def load_draft_model(self, target_model, tokenizer, draft_model_path):
        draft_model = SubSpecSDDraftModel.from_pretrained(
            draft_model_path,
            target_model=target_model,
            torch_dtype=self.dtype,
            eos_token_id=tokenizer.eos_token_id
        )
        return draft_model
    
    def load_kv_cache(self, target_model, draft_model):            
        if self.cache_implementation == "static":
            if self.max_length is not None:
                # Additional sample tokens may cause KV-Cache tp exceed max_length, share with draft model.
                max_cache_len = self.max_length + self.draft_params.max_sample_tokens
            else:
                raise ValueError("max_length should be set for static cache.")
            
            # Create static kv-cache
            past_key_values = create_kv_cache(
                "static",
                max_cache_len=max_cache_len,
                max_batch_size=1,
                config=target_model.model.config,
                device=self.device,
                dtype=target_model.model.dtype,
            )
        else:
            # Create dynamic kv-cache
            past_key_values = create_kv_cache("dynamic")
            
        # Target model shares cache with draft model.
        draft_past_key_values = None
        
        return past_key_values, draft_past_key_values
    
    def load_generator(self, target_model, tokenizer, draft_model=None):
        generator = SubSpecSDGenerator(
            target_model=target_model,
            tokenizer=tokenizer,
            draft_model=draft_model,
            draft_params=self.draft_params,
            post_draft_params=self.post_draft_params,
            cache_implementation=self.cache_implementation,
            profiling=self.generator_profiling,
            profiling_verbose=self.profiling_verbose,
            generator_kwargs=self.generator_kwargs,
        )
        return generator
    
    def compile_generator(self, generator):
        logging.info(f"Compiling generator with mode: {self.compile_mode}")
        # generator.target_model.forward = torch.compile(generator.target_model.forward, mode=self.compile_mode, dynamic=False, fullgraph=True)
        if getattr(generator, 'draft_model', None) is not None:
            generator.draft_model.forward = torch.compile(generator.draft_model.forward, mode=self.compile_mode, dynamic=False, fullgraph=True)
    
if __name__ == "__main__":
    run_app(SubSpecSDBuilder())