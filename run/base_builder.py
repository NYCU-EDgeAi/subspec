import logging
import os
import random
from typing import Any, Dict, Tuple, Optional
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from specdecodes.models.utils.cache_utils import create_kv_cache
from specdecodes.models.generators.naive import NaiveGenerator
from .app_router import run_app


LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=LOGLEVEL)

class GeneratorPipelineBuilder:
    """
    Builder class to construct the generation pipeline.
    
    This class handles:
      - Torch configuration (precision, seeding)
      - Loading the model and tokenizer
      - Generating configuration dictionaries via the recipe
      - Applying quantization and offloading through the recipe (if applicable)
      - Building and optionally compiling the generator pipeline
    """
    def __init__(self):
        # Base configurations.
        self.vram_limit_gb: Optional[int] = None
        self.seed = 0
        self.device = "cuda:0"
        self.dtype = torch.float16

        # Model paths.
        self.llm_path = "meta-llama/Llama-3.1-8B-Instruct"
        self.draft_model_path = None

        # Generation parameters.
        self.max_length = 2048
        self.do_sample = False
        self.temperature = 0.0

        # Generator-specific configurations.
        self.generator_kwargs = {}
        self.draft_params = None

        # Additional configurations.
        self.cache_implementation = "dynamic"
        self.warmup_iter = 0
        self.compile_mode = None

        # Recipe for quantization and offloading.
        # It can optionally perform quantization/offloading.
        self.recipe = None
        self.cpu_offload_gb: Optional[int] = None # This variable is not used in the current implementation.

        # Profiling and printing settings.
        self.generator_profiling: bool = True
        self.profiling_verbose: bool = True
        self.print_time: bool = True
        self.print_message: bool = True

        # Benchmarking/logging directories.
        self.out_dir: Optional[str] = None
        self.log_dir: str = "experiments"
        
    @property
    def args(self) -> Dict[str, Any]:
        """
        Return all attributes of the class as a dictionary.
        """
        my_dict = {k: v for k, v in self.__dict__.items() if not k.startswith("_") and not callable(v)}
        return SimpleNamespace(**my_dict)
        
    
    def configure_torch(self):
        """
        Set up torch configurations for reproducibility and performance.
        """
        torch.set_float32_matmul_precision('high')
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        
        # Set memory limit.
        total_memory = torch.cuda.get_device_properties(self.device).total_memory
        if self.vram_limit_gb is not None:
            memory_fraction = min(1.0, float(self.vram_limit_gb * (1024**3))/total_memory)
            torch.cuda.set_per_process_memory_fraction(memory_fraction, self.device)

    def load_model_and_tokenizer(self, model_path: str):
        """
        Load a model and tokenizer from the specified model path.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Use CPU if an offloader is provided via recipe; otherwise use the desired device.
        device_map = 'cpu' if (self.recipe and self.recipe.offloader) else self.device
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            device_map=device_map,
            _attn_implementation="sdpa",
        )
        return model, tokenizer

    def load_draft_model(self, target_model=None, tokenizer=None, draft_model_path=None):
        """
        Load a draft model if a draft model path is provided.
        Returns None if no draft model is needed.
        """
        if draft_model_path:
            # Implement draft model loading logic if needed.
            return None
        return None
    
    def load_kv_cache(self, target_model, draft_model):            
        if self.cache_implementation == "static":
            if self.max_length is not None:
                if draft_model is not None:
                    # Additional sample tokens may cause KV-Cache tp exceed max_length
                    max_cache_len = self.max_length + self.draft_params.max_verify_tokens
                else:
                    max_cache_len = self.max_length
            else:
                raise ValueError("max_length should be set for static cache.")
            
            # Create static kv-cache
            past_key_values = create_kv_cache(
                "static",
                max_cache_len=max_cache_len,
                max_batch_size=1,
                config=target_model.model.config,
                device=self.device,
                dtype=target_model.model.dtype,
            )
            # if generator.draft_model is not None:
            if draft_model is not None:
                draft_past_key_values = create_kv_cache(
                    "static",
                    max_cache_len=max_cache_len,
                    max_batch_size=1,
                    config=draft_model.model.config,
                    device=self.device,
                    dtype=draft_model.model.dtype,
                )
            else:
                draft_past_key_values = None
        else:
            # Create dynamic kv-cache
            past_key_values = create_kv_cache("dynamic")
            if draft_model is not None:
                draft_past_key_values = create_kv_cache("dynamic")
            else:
                draft_past_key_values = None
        
        return past_key_values, draft_past_key_values
    
    def load_generator(self, target_model, tokenizer, draft_model=None):
        """
        Initialize the generator with the target model, tokenizer, and draft model.
        """
        generator = NaiveGenerator(
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
        """
        Compile the generator's forward methods.
        """
        logging.info(f"Compiling generator with mode: {self.compile_mode}")
        generator.target_model.forward = torch.compile(generator.target_model.forward, mode=self.compile_mode, dynamic=False, fullgraph=True)
        if getattr(generator, 'draft_model', None) is not None:
            generator.draft_model.forward = torch.compile(generator.draft_model.forward, mode=self.compile_mode, dynamic=False, fullgraph=True)
    
    def post_process(self, generator, tokenizer, past_kv, draft_past_kv):
        pass
    
    def build_models_and_tokenizer(self):
        """
        Build and return the main model, draft model, and tokenizer.
        """
        self.configure_torch()
        model, tokenizer = self.load_model_and_tokenizer(self.llm_path)
        draft_model = self.load_draft_model(model, tokenizer, self.draft_model_path)

        if self.recipe:
            target_config, draft_config = self.recipe.generate_configurations(
                target_model=model,
                draft_model=draft_model,
                max_length=self.max_length,
                cpu_offload_gb=self.cpu_offload_gb,
                dtype=self.dtype,
                device=self.device,
            )
            
            if draft_model and draft_config and draft_config.get("quant_config"):
                self.recipe.apply_quantization(draft_model.model, draft_config["quant_config"], self.dtype, self.device)
            if target_config and target_config.get("quant_config"):
                self.recipe.apply_quantization(model, target_config["quant_config"], self.dtype, self.device)

            if draft_model and draft_config and draft_config.get("device_map"):
                self.recipe.apply_offloading(draft_model.model, draft_config["device_map"])
            if target_config and target_config.get("device_map"):
                self.recipe.apply_offloading(model, target_config["device_map"], draft_model=draft_model)

        return model, draft_model, tokenizer
    
    def build_generator_pipeline(self, model, draft_model, tokenizer):
        """
        Build the generator pipeline using pre-built model, draft_model, and tokenizer.
        """
        past_kv, draft_past_kv = self.load_kv_cache(model, draft_model)

        generator = self.load_generator(model, tokenizer, draft_model)
        generator.eval()

        if self.compile_mode is not None:
            self.compile_generator(generator)

        self.post_process(generator, tokenizer, past_kv, draft_past_kv)

        return generator, tokenizer, past_kv, draft_past_kv

    def build(self):
        """
        Build the full generation pipeline from scratch.
        """
        model, draft_model, tokenizer = self.build_models_and_tokenizer()
        return self.build_generator_pipeline(model, draft_model, tokenizer)


if __name__ == "__main__":
    run_app(GeneratorPipelineBuilder())