from .app_router import run_app
from .base_builder import GeneratorPipelineBuilder

import torch
from specdecodes.models.utils.utils import DraftParams
from specdecodes.models.draft_models.classic_sd import ClassicSDDraftModel
from specdecodes.models.generators.classic_sd import ClassicSDGenerator

class ClassicSDBuilder(GeneratorPipelineBuilder):
    def __init__(self):
        super().__init__()
        # Base configurations.
        self.vram_limit_gb = None
        self.seed = 0
        self.device = "cuda:0"
        self.dtype = torch.float16
        self.max_length = 10 * 1024
        
        # Model paths.
        self.llm_path = "Qwen/Qwen3-8B"#"meta-llama/Llama-3.1-8B-Instruct"
        self.draft_model_path = "Qwen/Qwen3-1.7B"#"meta-llama/Llama-3.2-1B-Instruct"
        
        # Generation parameters.
        self.do_sample = False
        self.temperature = 0
        
        # Generator-specific configurations.
        self.generator_kwargs = {
            "prefill_chunk_size": 4096,
        }
        self.draft_params = DraftParams(
            temperature=1,
            max_depth=8,
            topk_len=16,
        )
        
        # Recipe for quantization and offloading.
        self.recipe = None
        
        # Additional configurations.
        self.cache_implementation = "static"
        self.warmup_iter = 1
        self.compile_mode = "max-autotune"
        
        # Profiling.
        self.generator_profiling = True
    
    def load_draft_model(self, target_model, tokenizer, draft_model_path):
        draft_model = ClassicSDDraftModel.from_pretrained(
            draft_model_path,
            target_model=target_model,
            torch_dtype=self.dtype,
            device_map=self.device,
            eos_token_id=tokenizer.eos_token_id
        )
        return draft_model
    
    def load_generator(self, target_model, tokenizer, draft_model=None):
        generator = ClassicSDGenerator(
            target_model=target_model,
            tokenizer=tokenizer,
            draft_model=draft_model,
            draft_params=self.draft_params,
            cache_implementation=self.cache_implementation,
            profiling=self.generator_profiling,
            profiling_verbose=self.profiling_verbose,
            generator_kwargs=self.generator_kwargs,
        )
        return generator

if __name__ == "__main__":
    run_app(ClassicSDBuilder())