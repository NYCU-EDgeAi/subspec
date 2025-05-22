import torch
from run.app_router import run_app
from run.vanilla import NaiveBuilder

import torch
from specdecodes.helpers.recipes.quant.hqq_4bit import Recipe

class NaiveBuilder(NaiveBuilder):
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
        self.recipe = Recipe()
        
        # Additional configurations.
        self.cache_implementation = "static"
        self.warmup_iter = 3
        self.compile_mode = "max-autotune"
        
        # Profiling
        self.generator_profiling = True
        
if __name__ == "__main__":
    run_app(NaiveBuilder())