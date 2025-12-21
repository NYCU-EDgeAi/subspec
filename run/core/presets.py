import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from specdecodes.models.utils.cache_utils import create_kv_cache
from specdecodes.models.utils.utils import DraftParams
from .registry import ModelRegistry

# --- Custom Loader Hooks ---

def flashinfer_load_kv_cache(builder, target_model, draft_model):
    from specdecodes.models.utils.flashinfer.cache_manager import FlashInferCache
    
    if builder.cache_implementation == "static":
        if builder.max_length is not None:
            # Shared logic for max_cache_len calculation
            max_verify_tokens = 0
            if builder.draft_params:
                if hasattr(builder.draft_params, 'max_verify_tokens'):
                    max_verify_tokens = builder.draft_params.max_verify_tokens
                elif hasattr(builder.draft_params, 'max_sample_tokens'):
                    max_verify_tokens = builder.draft_params.max_sample_tokens
                elif hasattr(builder.draft_params, 'num_nodes'):
                        max_verify_tokens = builder.draft_params.num_nodes + 1
            
            max_cache_len = builder.max_length + max_verify_tokens
        else:
            raise ValueError("max_length should be set for static cache.")
        
        past_key_values = FlashInferCache(target_model.config, max_tokens=max_cache_len, PAGE_LEN=max_cache_len).kvCachePool
    else:
        past_key_values = create_kv_cache("dynamic")
        
    draft_past_key_values = FlashInferCache(draft_model.config, max_tokens=max_cache_len, PAGE_LEN=max_cache_len).kvCachePool
    return past_key_values, draft_past_key_values

def flashinfer_load_draft_model(builder, target_model, tokenizer, draft_model_path):
    from specdecodes.models.utils.flashinfer.monkey_patch import apply_flashinfer_kernel_to_llama
    
    # We need to get the class from the registry entry that is currently being used/loaded.
    # However, builder.config.method gives us the method name.
    entry = ModelRegistry.get(builder.config.method)
    draft_model_cls = entry.draft_model_cls
    
    draft_model = draft_model_cls.from_pretrained(
        draft_model_path,
        target_model=target_model,
        torch_dtype=builder.dtype,
        device_map=builder.device,
        eos_token_id=tokenizer.eos_token_id
    )
    apply_flashinfer_kernel_to_llama(attention=True, rms_norm=True, swiglu=False, model=target_model)
    apply_flashinfer_kernel_to_llama(attention=True, rms_norm=True, swiglu=False, model=draft_model)
    return draft_model

def eagle_load_draft_model(builder, target_model, tokenizer, draft_model_path):
    entry = ModelRegistry.get(builder.config.method)
    draft_model_cls = entry.draft_model_cls
    
    # Eagle usually needs .to(device) explicitly if device_map is not passed or if it behaves differently
    draft_model = draft_model_cls.from_pretrained(
        draft_model_path,
        target_model=target_model,
        torch_dtype=builder.dtype,
        eos_token_id=tokenizer.eos_token_id
    ).to(builder.device)
    
    draft_model.update_modules(embed_tokens=target_model.get_input_embeddings(), lm_head=target_model.lm_head)
    return draft_model

def quant_load_model(builder, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Use CPU if an offloader is provided via recipe; otherwise use the desired device.
    # Defaulting to behavior extracted from original other_quant.py
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        device_map="auto",
        quantization_config=GPTQConfig(bits=4, backend="triton")
    )
    return model, tokenizer

# --- Registry ---

def register_presets():
    
    # SubSpec SD (Original)
    try:
        from specdecodes.models.generators.subspec_sd import SubSpecSDGenerator
        from specdecodes.models.draft_models.subspec_sd import SubSpecSDDraftModel
        from specdecodes.helpers.recipes.subspec.hqq_4bit_attn_4bit_mlp import Recipe as SubSpecRecipe

        ModelRegistry.register(
            name="subspec_sd",
            generator_cls=SubSpecSDGenerator,
            draft_model_cls=SubSpecSDDraftModel,
            default_config={
                "llm_path": "meta-llama/Llama-3.2-1B-Instruct",
                "max_length": 2 * 1024,
                "generator_kwargs": {"prefill_chunk_size": 4096},
                "draft_params": DraftParams(temperature=0.2, max_depth=32, topk_len=6),
                "recipe": SubSpecRecipe(),
                "cache_implementation": "static",
                "warmup_iter": 1,
                "compile_mode": None,
                "generator_profiling": True,
            }
        )
    except ImportError:
        pass

    # Classic SD
    from specdecodes.models.generators.classic_sd import ClassicSDGenerator
    from specdecodes.models.draft_models.classic_sd import ClassicSDDraftModel
    
    ModelRegistry.register(
        name="classic_sd",
        generator_cls=ClassicSDGenerator,
        draft_model_cls=ClassicSDDraftModel,
        default_config={
            "llm_path": "meta-llama/Llama-3.1-8B-Instruct",
            "draft_model_path": "meta-llama/Llama-3.2-1B-Instruct",
            "max_length": 2 * 1024,
            "generator_kwargs": {"prefill_chunk_size": 4096},
            "draft_params": DraftParams(temperature=1, max_depth=8, topk_len=16),
            "recipe": None,
            "cache_implementation": "static",
            "warmup_iter": 1,
            "compile_mode": None,
            "generator_profiling": True,
        }
    )
    
    # Vanilla (Naive)
    from specdecodes.models.generators.naive import NaiveGenerator
    
    ModelRegistry.register(
        name="vanilla",
        generator_cls=NaiveGenerator,
        draft_model_cls=None,
        default_config={
            "llm_path": "meta-llama/Llama-3.1-8B-Instruct",
            "max_length": 16 * 1024,
            "generator_kwargs": {"prefill_chunk_size": 4096},
            "recipe": None,
            "cache_implementation": "static",
            "warmup_iter": 1,
            "compile_mode": None,
            "generator_profiling": True,
        }
    )

    # SubSpec SD V2
    try:
        from specdecodes.models.generators.subspec_sd_v2 import SubSpecSDGenerator as SubSpecSDGeneratorV2
        from specdecodes.helpers.recipes.subspec.hqq_4bit_attn_4bit_mlp_postspec import Recipe as SubSpecRecipeV2
        
        ModelRegistry.register(
            name="subspec_sd_v2",
            generator_cls=SubSpecSDGeneratorV2,
            draft_model_cls=SubSpecSDDraftModel,
            default_config={
                "llm_path": "meta-llama/Llama-3.1-8B-Instruct",
                "max_length": 10 * 1024,
                "generator_kwargs": {"prefill_chunk_size": 4096},
                "draft_params": DraftParams(temperature=0.2, max_depth=16, topk_len=6),
                "recipe": SubSpecRecipeV2(),
                "cache_implementation": "static",
                "warmup_iter": 3,
                "generator_profiling": True,
            }
        )
    except ImportError:
        pass

    # Classic SD FlashInfer
    try:
        from specdecodes.models.generators.classic_sd_fi import ClassicSDGenerator as ClassicSDGeneratorFI
        from specdecodes.models.draft_models.classic_sd_fi import ClassicSDDraftModel as ClassicSDDraftModelFI
        
        ModelRegistry.register(
            name="classic_sd_fi",
            generator_cls=ClassicSDGeneratorFI,
            draft_model_cls=ClassicSDDraftModelFI,
            default_config={
                "llm_path": "meta-llama/Llama-3.1-8B-Instruct",
                "draft_model_path": "meta-llama/Llama-3.2-1B-Instruct",
                "max_length": 2 * 1024,
                "generator_kwargs": {"prefill_chunk_size": 4096},
                "draft_params": DraftParams(temperature=1, max_depth=8, topk_len=16),
                "recipe": None,
                "cache_implementation": "static",
                "warmup_iter": 0,
                "compile_mode": None,
                "generator_profiling": True,
            },
            load_draft_model_fn=flashinfer_load_draft_model,
            load_kv_cache_fn=flashinfer_load_kv_cache
        )
    except ImportError:
        pass 

    # SubSpec SD FlashInfer
    try:
        from specdecodes.models.generators.subspec_sd_fi import SubSpecSDGenerator as SubSpecSDGeneratorFI
        from specdecodes.models.draft_models.subspec_sd_fi import SubSpecSDDraftModel as SubSpecSDDraftModelFI

        ModelRegistry.register(
            name="subspec_sd_fi",
            generator_cls=SubSpecSDGeneratorFI,
            draft_model_cls=SubSpecSDDraftModelFI,
            default_config={
                "llm_path": "meta-llama/Llama-3.1-8B-Instruct",
                "max_length": 2 * 1024,
                "generator_kwargs": {"prefill_chunk_size": 4096},
                "draft_params": DraftParams(temperature=0.2, max_depth=32, topk_len=6),
                "recipe": SubSpecRecipe(),
                "cache_implementation": "static",
                "warmup_iter": 3,
                "compile_mode": "max-autotune-no-cudagraphs",
                "generator_profiling": True,
            },
            load_draft_model_fn=flashinfer_load_draft_model,
            load_kv_cache_fn=flashinfer_load_kv_cache
        )
    except ImportError:
        pass

    # Classic SD Seq
    try:
        from specdecodes.models.generators.classic_seq_sd import ClassicSDGenerator as ClassicSDGeneratorSeq
        from specdecodes.models.draft_models.classic_seq_sd import ClassicSDDraftModel as ClassicSDDraftModelSeq

        ModelRegistry.register(
            name="classic_seq_sd",
            generator_cls=ClassicSDGeneratorSeq,
            draft_model_cls=ClassicSDDraftModelSeq,
            default_config={
                "llm_path": "meta-llama/Llama-3.1-8B-Instruct",
                "draft_model_path": "meta-llama/Llama-3.2-1B-Instruct",
                "max_length": 2 * 1024,
                "generator_kwargs": {"prefill_chunk_size": 4096},
                "draft_params": DraftParams(temperature=1, max_depth=8, topk_len=1),
                "recipe": None,
                "cache_implementation": "static",
                "warmup_iter": 1,
                "compile_mode": None,
                "generator_profiling": True,
            }
        )
    except ImportError:
        pass

    # SubSpec SD Seq
    try:
        from specdecodes.models.generators.subspec_seq_sd import SubSpecSDGenerator as SubSpecSDGeneratorSeq
        from specdecodes.models.draft_models.subspec_seq_sd import SubSpecSDDraftModel as SubSpecSDDraftModelSeq

        ModelRegistry.register(
            name="subspec_seq_sd",
            generator_cls=SubSpecSDGeneratorSeq,
            draft_model_cls=SubSpecSDDraftModelSeq,
            default_config={
                "llm_path": "meta-llama/Llama-3.1-8B-Instruct",
                "max_length": 2 * 1024,
                "generator_kwargs": {"prefill_chunk_size": 4096},
                "draft_params": DraftParams(temperature=1, max_depth=32, topk_len=1),
                "recipe": SubSpecRecipe(),
                "cache_implementation": "static",
                "warmup_iter": 1,
                "compile_mode": None,
                "generator_profiling": True,
            }
        )
    except ImportError:
        pass

    # SubSpec SD Seq V2
    try:
        from specdecodes.models.generators.subspec_seq_sd_v2 import SubSpecSDGenerator as SubSpecSDGeneratorSeqV2
        
        ModelRegistry.register(
            name="subspec_seq_sd_v2",
            generator_cls=SubSpecSDGeneratorSeqV2,
            draft_model_cls=SubSpecSDDraftModelSeq,
            default_config={
                "llm_path": "meta-llama/Llama-3.1-8B-Instruct",
                "max_length": 2 * 1024,
                "generator_kwargs": {"prefill_chunk_size": 4096},
                "draft_params": DraftParams(temperature=1, max_depth=16, topk_len=1),
                "recipe": SubSpecRecipeV2(),
                "cache_implementation": "static",
                "warmup_iter": 1,
                "compile_mode": None,
                "generator_profiling": True,
            }
        )
    except ImportError:
        pass

    # SubSpec SD No Offload
    try:
        from specdecodes.helpers.recipes.subspec.hqq_4bit_attn_4bit_mlp_no_offload import Recipe as SubSpecRecipeNoOffload
        
        ModelRegistry.register(
            name="subspec_sd_no_offload",
            generator_cls=SubSpecSDGenerator,
            draft_model_cls=SubSpecSDDraftModel,
            default_config={
                "llm_path": "meta-llama/Llama-3.1-8B-Instruct",
                "max_length": 2 * 1024,
                "generator_kwargs": {"prefill_chunk_size": 256},
                "draft_params": DraftParams(temperature=0.2, max_depth=48, topk_len=6),
                "recipe": SubSpecRecipeNoOffload(),
                "cache_implementation": "static",
                "warmup_iter": 3,
                "compile_mode": None,
                "generator_profiling": True,
            }
        )
    except ImportError:
        pass

    # Eagle SD
    try:
        from specdecodes.models.generators.eagle_sd import EagleSDGenerator
        from specdecodes.models.draft_models.eagle_sd import EagleSDDraftModel

        ModelRegistry.register(
            name="eagle_sd",
            generator_cls=EagleSDGenerator,
            draft_model_cls=EagleSDDraftModel,
            default_config={
                "llm_path": "meta-llama/Llama-3.1-8B-Instruct",
                "draft_model_path": "~/checkpoints/eagle/official/EAGLE-Llama-3.1-8B-Instruct",
                "max_length": 2 * 1024,
                "generator_kwargs": {},
                "draft_params": DraftParams(temperature=1, max_depth=6, topk_len=10),
                "recipe": None,
                "cache_implementation": "static",
                "warmup_iter": 3,
                "compile_mode": None,
                "generator_profiling": True,
            },
            load_draft_model_fn=eagle_load_draft_model
        )
    except ImportError:
        pass

    # HuggingFace
    try:
        from specdecodes.models.generators.huggingface import HuggingFaceGenerator

        ModelRegistry.register(
            name="huggingface",
            generator_cls=HuggingFaceGenerator,
            draft_model_cls=None,
            default_config={
                "llm_path": "meta-llama/Llama-3.1-8B-Instruct",
                "max_length": 2 * 1024, 
                "generator_kwargs": {},
                "recipe": None,
                "cache_implementation": "dynamic",
                "warmup_iter": 0,
                "compile_mode": None,
                "generator_profiling": True,
            }
        )
    except ImportError:
        pass
    
    # Share Layer SD
    try:
        from specdecodes.models.draft_models.share_layer_sd import ShareLayerSDDraftModel
        
        ModelRegistry.register(
            name="share_layer_sd",
            generator_cls=ClassicSDGenerator,
            draft_model_cls=ShareLayerSDDraftModel,
            default_config={
                "llm_path": "Qwen/Qwen2.5-7B-Instruct",
                "max_length": 2 * 1024,
                "generator_kwargs": {"prefill_chunk_size": 256},
                "draft_params": DraftParams(temperature=0.2, max_depth=48, topk_len=6),
                "recipe": SubSpecRecipe(),
                "cache_implementation": "static",
                "warmup_iter": 3,
                "compile_mode": None,
                "generator_profiling": True,
            }
        )
    except ImportError:
        pass
    
    # Vanilla Quant
    try:
        from specdecodes.helpers.recipes.quant.hqq_4bit import Recipe as HQQRecipe
        
        ModelRegistry.register(
            name="vanilla_quant",
            generator_cls=NaiveGenerator,
            draft_model_cls=None,
            default_config={
                "llm_path": "meta-llama/Llama-3.1-8B-Instruct",
                "max_length": 2 * 1024,
                "generator_kwargs": {"prefill_chunk_size": 256},
                "recipe": HQQRecipe(),
                "cache_implementation": "static",
                "warmup_iter": 3,
                "compile_mode": None,
                "generator_profiling": True,
            }
        )

        # Other Quant (GPTQ)
        ModelRegistry.register(
            name="other_quant",
            generator_cls=NaiveGenerator,
            draft_model_cls=None,
            default_config={
                "llm_path": "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
                "max_length": 2 * 1024, 
                "generator_kwargs": {"prefill_chunk_size": 256},
                "recipe": HQQRecipe(),
                "cache_implementation": "static",
                "warmup_iter": 3,
                "compile_mode": None,
                "generator_profiling": True,
            },
            load_model_fn=quant_load_model
        )
    except ImportError:
        pass
