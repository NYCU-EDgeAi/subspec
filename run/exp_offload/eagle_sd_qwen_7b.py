from run.app_router import run_app
from run.eagle_sd import EagleSDBuilder

import torch
import logging
from specdecodes.models.utils.utils import DraftParams

from .recipes.recipe_eagle_sd_qwen_7b import Recipe
from specdecodes.models.draft_models.eagle_sd import EagleSDDraftModel
from specdecodes.models.generators.eagle_sd import EagleSDGenerator

from transformers import AutoTokenizer, AutoModelForCausalLM

class ExpEagleSDBuilder(EagleSDBuilder):
    def __init__(self):
        super().__init__()
        # Base configurations.
        self.vram_limit_gb = 8
        self.seed = 0
        self.device = "cuda:0"
        self.dtype = torch.float16
        self.max_length = 2048
        
        # Model paths.
        self.llm_path = "Qwen/Qwen2.5-7B-Instruct"
        self.draft_model_path = "~/checkpoints/eagle/official/EAGLE-Qwen2.5-7B-Instruct"
        
        # Generation parameters.
        self.do_sample = False
        self.temperature = 0
        
        # Generator-specific configurations.
        self.generator_kwargs = {} # chunk preffiling breaks eagle_sd, hence not used.
        self.draft_params = DraftParams(
            temperature=1,
            max_depth=6,
            topk_len=10,
        )
        
        # Recipe for quantization and offloading.
        self.recipe = Recipe()
        
        # Additional configurations.
        self.cache_implementation = "static"
        self.warmup_iter = 3
        self.compile_mode = "max-autotune"
        
        # Profiling.
        self.generator_profiling = True
    
    def compile_generator(self, generator):
        """
        Compile the generator's forward methods.
        """
        logging.info(f"Compiling generator with mode: {self.compile_mode}")
        # Cannot compile target model under offload settings!
        # generator.target_model.forward = torch.compile(generator.target_model.forward, mode=self.compile_mode, dynamic=False, fullgraph=True)
        if getattr(generator, 'draft_model', None) is not None:
            generator.draft_model.forward = torch.compile(generator.draft_model.forward, mode=self.compile_mode, dynamic=False, fullgraph=True)


if __name__ == "__main__":
    run_app(ExpEagleSDBuilder())