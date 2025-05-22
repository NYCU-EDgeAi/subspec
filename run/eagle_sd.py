from .app_router import run_app
from .base_builder import GeneratorPipelineBuilder

import torch
from specdecodes.models.utils.utils import DraftParams
from specdecodes.models.draft_models.eagle_sd import EagleSDDraftModel
from specdecodes.models.generators.eagle_sd import EagleSDGenerator

class EagleSDBuilder(GeneratorPipelineBuilder):
    def __init__(self):
        super().__init__()
        # Base configurations.
        self.vram_limit_gb = None
        self.seed = 0
        self.device = "cuda:0"
        self.dtype = torch.float16
        
        # Model paths.
        self.llm_path = "meta-llama/Llama-3.1-8B-Instruct"
        self.draft_model_path = "~/checkpoints/eagle/official/EAGLE-Llama-3.1-8B-Instruct"
        
        # Generation parameters.
        self.do_sample = False
        self.temperature = 0
        
        # Generator-specific configurations.
        self.generator_kwargs = {}
        self.draft_params = DraftParams(
            temperature=1,
            max_depth=6,
            topk_len=10,
        )
        
        # Recipe for quantization and offloading.
        self.recipe = None
        
        # Additional configurations.
        self.cache_implementation = "static"
        self.warmup_iter = 3
        self.compile_mode = "max-autotune"
        
        # Profiling.
        self.generator_profiling = True
    
    def load_draft_model(self, target_model, tokenizer, draft_model_path):
        draft_model = EagleSDDraftModel.from_pretrained(
            draft_model_path,
            target_model=target_model,
            torch_dtype=self.dtype,
            eos_token_id=tokenizer.eos_token_id
        ).to(self.device)
        draft_model.update_modules(embed_tokens=target_model.get_input_embeddings(), lm_head=target_model.lm_head)
        return draft_model
    
    def load_generator(self, target_model, tokenizer, draft_model=None):
        generator = EagleSDGenerator(
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
    run_app(EagleSDBuilder())