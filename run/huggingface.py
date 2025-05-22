from .app_router import run_app
from .base_builder import GeneratorPipelineBuilder

import torch
from specdecodes.models.generators.huggingface import HuggingFaceGenerator

class HuggingFaceBuilder(GeneratorPipelineBuilder):
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
        
        # Additional configurations.
        self.cache_implementation = "dynamic"
        self.warmup_iter = 0
        self.compile_mode = None
        
        # Profiling
        self.generator_profiling = True
        
    def load_generator(self, target_model, tokenizer, draft_model=None):
        generator = HuggingFaceGenerator(
            target_model=target_model,
            tokenizer=tokenizer,
            draft_model=draft_model,
            device=self.device,
            dtype=self.dtype,
            do_sample=self.do_sample,
            temperature=self.temperature,
            profiling_verbose=self.profiling_verbose
        )
        return generator
        
if __name__ == "__main__":
    run_app(HuggingFaceBuilder())