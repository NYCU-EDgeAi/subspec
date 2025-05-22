import torch
import logging

from run.app_router import run_app
from run.classic_sd import ClassicSDBuilder
from specdecodes.models.utils.utils import DraftParams

from .recipes.recipe_classic_sd_1p5b_qwen_32b import Recipe

class ExpClassicSDBuilder(ClassicSDBuilder):
    def __init__(self):
        super().__init__()
        # Base configurations.
        self.vram_limit_gb = 24
        self.seed = 0
        self.device = "cuda:0"
        self.dtype = torch.float16
        self.max_length = 2048
        
        # Model paths.
        self.llm_path = "Qwen/Qwen2.5-32B-Instruct"
        self.draft_model_path = "Qwen/Qwen2.5-1.5B-Instruct"
        
        # Generation parameters.
        self.do_sample = False
        self.temperature = 0
        
        # Generator-specific configurations.
        self.generator_kwargs = {
            "prefill_chunk_size": 256,
        }
        self.draft_params = DraftParams(
            temperature=1,
            max_depth=32,
            topk_len=6,
        )
        
        # Recipe for quantization and offloading.
        self.recipe = Recipe()
        
        # Additional configurations.
        self.cache_implementation = "static"
        self.warmup_iter = 3
        self.compile_mode = "max-autotune"
        
    def compile_generator(self, generator):
        logging.info(f"Compiling generator with mode: {self.compile_mode}")
        # Cannot compile target model under offload settings!
        # generator.target_model.forward = torch.compile(generator.target_model.forward, mode=self.compile_mode, dynamic=False, fullgraph=True)
        if getattr(generator, 'draft_model', None) is not None:
            generator.draft_model.forward = torch.compile(generator.draft_model.forward, mode=self.compile_mode, dynamic=False, fullgraph=True)
        
if __name__ == "__main__":
    run_app(ExpClassicSDBuilder())