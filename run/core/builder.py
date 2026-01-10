import logging
import os
import random
from typing import Any, Optional, TYPE_CHECKING
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from specdecodes.models.utils.cache_utils import create_kv_cache
from specdecodes.models.generators.naive import NaiveGenerator
from .router import run_app
from .registry import ModelRegistry
from .config_utils import instantiate_recipe
# Type hint only, import inside init to avoid circular dependency from .registry import ModelRegistry
if TYPE_CHECKING:
    from .configuration import AppConfig


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
    def __init__(self, config: Optional["AppConfig"] = None):
        if config is None:
            from .configuration import AppConfig
            config = AppConfig()
        
        self.config = config

        # Expose config attributes directly for legacy call sites.
        self.__dict__.update(config.__dict__)

        # Normalize recipe from YAML/preset into an actual recipe object.
        # (Some entrypoints may construct AppConfig directly without going through run/main.py.)
        self.recipe = instantiate_recipe(getattr(self, "recipe", None))
        self.config.recipe = self.recipe
        
    @property
    def args(self) -> SimpleNamespace:
        my_dict = {k: v for k, v in self.__dict__.items() if not k.startswith("_") and not callable(v)}
        return SimpleNamespace(**my_dict)
        
    
    def configure_torch(self):
        """
        Set up torch configurations for reproducibility and performance.
        """
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        # Configure CUDA only when a CUDA device is requested and available.
        cuda_device: Optional[torch.device] = None
        if torch.cuda.is_available():
            if isinstance(self.device, torch.device):
                cuda_device = self.device if self.device.type == "cuda" else None
            elif isinstance(self.device, int):
                cuda_device = torch.device(f"cuda:{self.device}")
            elif isinstance(self.device, str) and self.device.startswith("cuda"):
                cuda_device = torch.device(self.device)

        if cuda_device is not None:
            torch.cuda.set_device(cuda_device)
            torch.cuda.manual_seed_all(self.seed)

            torch.set_float32_matmul_precision("high")
            torch._dynamo.config.allow_unspec_int_on_nn_module = True

            # Set memory limit.
            total_memory = torch.cuda.get_device_properties(cuda_device).total_memory
            if self.vram_limit_gb is not None:
                memory_fraction = min(1.0, float(self.vram_limit_gb * (1024**3)) / total_memory)
                torch.cuda.set_per_process_memory_fraction(memory_fraction, cuda_device)

    def load_model_and_tokenizer(self, model_path: str):
        """
        Load a model and tokenizer from the specified model path.
        """
        entry = ModelRegistry.get(self.config.method)
        if entry and entry.load_model_fn:
            return entry.load_model_fn(self, model_path)

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
        entry = ModelRegistry.get(self.config.method)
        if entry and entry.load_draft_model_fn:
            return entry.load_draft_model_fn(self, target_model, tokenizer, draft_model_path)
            
        draft_model_cls = entry.get_draft_model_cls() if entry else None
        if draft_model_cls:
            # Assuming standard from_pretrained pattern
            draft_model = draft_model_cls.from_pretrained(
                draft_model_path,
                target_model=target_model,
                torch_dtype=self.dtype,
                eos_token_id=tokenizer.eos_token_id,
                device_map=self.device
            )
            return draft_model
        return None
    
    def load_kv_cache(self, target_model, draft_model):    
        entry = ModelRegistry.get(self.config.method)
        # If there is no draft model, we never allocate a draft KV cache.
        if draft_model is None:
            if self.cache_implementation == "static":
                if self.max_length is None:
                    raise ValueError("max_length should be set for static cache.")
                past_key_values = create_kv_cache(
                    "static",
                    max_cache_len=self.max_length,
                    max_batch_size=1,
                    config=target_model.model.config,
                    device=self.device,
                    dtype=target_model.model.dtype,
                )
            else:
                past_key_values = create_kv_cache("dynamic")
            return past_key_values, None

        needs_draft_kv_cache = bool(getattr(entry, "needs_draft_kv_cache", True)) if entry else True

        if entry and entry.load_kv_cache_fn:
            past_key_values, draft_past_key_values = entry.load_kv_cache_fn(self, target_model, draft_model)
            return past_key_values, draft_past_key_values if needs_draft_kv_cache else None
                    
        if self.cache_implementation == "static":
            if self.max_length is not None:
                if draft_model is not None:
                    # Additional speculative tokens may cause KV-cache to exceed `max_length`.
                    # We allocate extra headroom based on draft params.
                    def _infer_max_verify_tokens(draft_params: Any) -> int:
                        if not draft_params:
                            return 0

                        # Support DraftParams dataclass, SimpleNamespace, or raw dict.
                        if isinstance(draft_params, dict):
                            if "max_verify_tokens" in draft_params and draft_params["max_verify_tokens"] is not None:
                                return int(draft_params["max_verify_tokens"])
                            if "max_sample_tokens" in draft_params and draft_params["max_sample_tokens"] is not None:
                                return int(draft_params["max_sample_tokens"])
                            if "num_nodes" in draft_params and draft_params["num_nodes"] is not None:
                                return int(draft_params["num_nodes"]) + 1
                            if "max_depth" in draft_params and "topk_len" in draft_params:
                                try:
                                    return int(draft_params["max_depth"]) * int(draft_params["topk_len"]) + 1
                                except Exception:
                                    return 0
                            return 0

                        if hasattr(draft_params, "max_verify_tokens") and getattr(draft_params, "max_verify_tokens") is not None:
                            return int(getattr(draft_params, "max_verify_tokens"))
                        if hasattr(draft_params, "max_sample_tokens") and getattr(draft_params, "max_sample_tokens") is not None:
                            return int(getattr(draft_params, "max_sample_tokens"))
                        if hasattr(draft_params, "num_nodes") and getattr(draft_params, "num_nodes") is not None:
                            return int(getattr(draft_params, "num_nodes")) + 1
                        if hasattr(draft_params, "max_depth") and hasattr(draft_params, "topk_len"):
                            try:
                                return int(getattr(draft_params, "max_depth")) * int(getattr(draft_params, "topk_len")) + 1
                            except Exception:
                                return 0
                        return 0

                    max_verify_tokens = _infer_max_verify_tokens(getattr(self, "draft_params", None))
                    max_cache_len = int(self.max_length) + int(max_verify_tokens)
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
            if needs_draft_kv_cache:
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
            if needs_draft_kv_cache:
                draft_past_key_values = create_kv_cache("dynamic")
            else:
                draft_past_key_values = None
        
        return past_key_values, draft_past_key_values
    
    def load_generator(self, target_model, tokenizer, draft_model=None):
        """
        Initialize the generator with the target model, tokenizer, and draft model.
        """
        entry = ModelRegistry.get(self.config.method)
        generator_cls = entry.get_generator_cls() if entry else None
        if generator_cls:
            generator = generator_cls(
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
            
        # Fallback to NaiveGenerator if not in registry (or default behavior)
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
        if not hasattr(torch, "compile"):
            raise RuntimeError(
                "compile_mode is set but torch.compile is unavailable. "
                "Please use PyTorch 2.x or disable compile_mode."
            )

        logging.info(f"Compiling generator with mode: {self.compile_mode}")
        
        # FlashInfer methods generally require fullgraph=False to work.
        method_name = str(getattr(self.config, "method", ""))
        fullgraph = not method_name.endswith("_fi")
        
        # If the target model uses offloading, torch.compile() (especially fullgraph/cudagraph-related paths)
        # is typically incompatible or provides little benefit. Skip compiling target_model in that case.
        has_offloader = bool(getattr(self.recipe, "offloader", None))
        if has_offloader:
            logging.info("Skipping torch.compile() for target_model because recipe.offloader is set.")
        else:
            generator.target_model.forward = torch.compile(generator.target_model.forward, mode=self.compile_mode, dynamic=False, fullgraph=fullgraph)

        # Compile draft model if it exists
        if getattr(generator, 'draft_model', None) is not None:
            generator.draft_model.forward = torch.compile(
                generator.draft_model.forward,
                mode=self.compile_mode,
                dynamic=False,
                fullgraph=fullgraph,
            )
    
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
            
            # Apply quantization first
            if draft_model and draft_config and draft_config.get("quant_config"):
                self.recipe.apply_quantization(draft_model.model, draft_config["quant_config"], self.dtype, self.device)
            if target_config and target_config.get("quant_config"):
                self.recipe.apply_quantization(model, target_config["quant_config"], self.dtype, self.device)

            # Then apply offloading
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