import torch
from run.app_router import run_app
from run.offload.subspec_sd import SubSpecSDBuilder
from specdecodes.models.utils.utils import DraftParams

from ..recipes.recipe_subspec_sd import Recipe

class ExpSubSpecSDBuilder(SubSpecSDBuilder):
    def __init__(self):
        super().__init__()
        # Base configurations.
        self.vram_limit_gb = 8
        self.seed = 0
        self.device = "cuda:0"
        self.dtype = torch.float16
        self.max_length = 2048
        
        # Model paths.
        self.llm_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        
        # Generation parameters.
        self.do_sample = False
        self.temperature = 0
        
        # Generator-specific configurations.
        self.generator_kwargs = {
            "prefill_chunk_size": 256,
        }
        self.draft_params = DraftParams(
            temperature=0.2,
            max_depth=48,
            topk_len=6,
        )
        
        # Recipe for quantization and offloading.
        self.recipe = Recipe()
        
        # Additional configurations.
        self.cache_implementation = "static"
        self.warmup_iter = 3
        self.compile_mode = "max-autotune"
        
if __name__ == "__main__":
    run_app(ExpSubSpecSDBuilder())