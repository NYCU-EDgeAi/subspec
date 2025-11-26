import logging
from .app_router import run_app
from .base_builder import GeneratorPipelineBuilder

import torch
from specdecodes.models.utils.utils import DraftParams
from specdecodes.models.utils.cache_utils import create_kv_cache
from specdecodes.models.draft_models.classic_sd_fi import ClassicSDDraftModel
from specdecodes.models.generators.classic_sd_fi import ClassicSDGenerator
from specdecodes.models.utils.flashinfer.monkey_patch import apply_flashinfer_kernel_to_llama
from specdecodes.models.utils.flashinfer.cache_manager import FlashInferCache

def set_torch_compile_config():
    import torch._dynamo.config
    import torch._inductor.config

    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future

    # FIXME: tmp workaround
    torch._dynamo.config.accumulated_cache_size_limit = 1024
    if hasattr(torch._dynamo.config, "cache_size_limit"):
        torch._dynamo.config.cache_size_limit = 1024

class ClassicSDBuilder(GeneratorPipelineBuilder):
    def __init__(self):
        super().__init__()
        # Base configurations.
        self.vram_limit_gb = None
        self.seed = 0
        self.device = "cuda:0"
        self.dtype = torch.float16
        self.max_length = 16 * 1024
        
        # Model paths.
        self.llm_path = "meta-llama/Llama-3.1-8B-Instruct"
        # self.llm_path = "Qwen/Qwen3-8B"
        self.draft_model_path = "meta-llama/Llama-3.2-1B-Instruct"
        # self.draft_model_path = "Qwen/Qwen3-1.7B"
        
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
        self.warmup_iter = 0
        self.compile_mode = None
        
        # Profiling.
        self.generator_profiling = True

    def load_kv_cache(self, target_model, draft_model):
        if self.cache_implementation == "static":
            if self.max_length is not None:
                # Additional sample tokens may cause KV-Cache tp exceed max_length, share with draft model.
                max_cache_len = self.max_length + self.draft_params.max_sample_tokens
            else:
                raise ValueError("max_length should be set for static cache.")
            
            past_key_values = FlashInferCache(target_model.config, max_tokens=max_cache_len, PAGE_LEN=max_cache_len).kvCachePool
        else:
            # Create dynamic kv-cache
            past_key_values = create_kv_cache("dynamic")
            
        draft_past_key_values = FlashInferCache(draft_model.config, max_tokens=max_cache_len, PAGE_LEN=max_cache_len).kvCachePool
        
        return past_key_values, draft_past_key_values

    def load_draft_model(self, target_model, tokenizer, draft_model_path):
        draft_model = ClassicSDDraftModel.from_pretrained(
            draft_model_path,
            target_model=target_model,
            torch_dtype=self.dtype,
            device_map=self.device,
            eos_token_id=tokenizer.eos_token_id
        )
        apply_flashinfer_kernel_to_llama(attention=True, rms_norm=True, swiglu=False, model=target_model)
        apply_flashinfer_kernel_to_llama(attention=True, rms_norm=True, swiglu=False, model=draft_model)
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
    
    def compile_generator(self, generator):
        logging.info(f"Compiling generator with mode: {self.compile_mode}")
        set_torch_compile_config()
        # generator.target_model.forward = torch.compile(generator.target_model.forward, mode=self.compile_mode, dynamic=False, fullgraph=True)
        if getattr(generator, 'draft_model', None) is not None:
            generator.draft_model.forward = torch.compile(generator.draft_model.forward, mode=self.compile_mode, dynamic=False, fullgraph=False)

if __name__ == "__main__":
    run_app(ClassicSDBuilder())