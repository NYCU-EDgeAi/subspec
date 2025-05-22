from .app_router import run_app
from .base_builder import GeneratorPipelineBuilder

import torch
from specdecodes.models.generators.naive import NaiveGenerator

class NaiveBuilder(GeneratorPipelineBuilder):
    def __init__(self):
        super().__init__()
        # Base configurations.
        self.vram_limit_gb = None
        self.device = "cuda:0"
        self.dtype = torch.float16
        
        # Model paths.
        self.llm_path = "meta-llama/Llama-3.1-8B-Instruct"
        
        # Generation parameters.
        self.do_sample = False
        self.temperature = 0
        
        # Generator-specific configurations.
        self.generator_kwargs = {
            "prefill_chunk_size": 256,
        }
        
        # Recipe for quantization and offloading.
        self.recipe = None
        
        # Additional configurations.
        self.cache_implementation = "static"
        self.warmup_iter = 3
        self.compile_mode = "max-autotune"
        
        # Profiling
        self.generator_profiling = True
        
    def load_generator(self, target_model, tokenizer, draft_model=None):
        generator = NaiveGenerator(
            target_model=target_model,
            tokenizer=tokenizer,
            draft_model=draft_model,
            device=self.device,
            dtype=self.dtype,
            do_sample=self.do_sample,
            temperature=self.temperature,
            profiling_verbose=self.profiling_verbose,
            generator_kwargs=self.generator_kwargs,
        )
        return generator
        
if __name__ == "__main__":
    run_app(NaiveBuilder())