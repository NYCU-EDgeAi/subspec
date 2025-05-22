import torch
from run.app_router import run_app
from run.vanilla import NaiveBuilder

from ..recipes.recipe_vanilla_qwen_32b import Recipe

class NaiveBuilder(NaiveBuilder):
    def __init__(self):
        super().__init__()
        # Base configurations.
        self.vram_limit_gb = 24
        self.device = "cuda:0"
        self.dtype = torch.float16
        self.max_length = 2048
        
        # Model paths.
        self.llm_path = "Qwen/QwQ-32B"
        
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
        self.warmup_iter = 1
        self.compile_mode = None # torch.compile does not benefit under offload settings!
        
        # Profiling
        self.generator_profiling = True

if __name__ == "__main__":
    run_app(NaiveBuilder())