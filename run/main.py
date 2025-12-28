import sys
import argparse
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

# Monkey patch for auto_gptq compatibility with optimum
try:
    import auto_gptq
    if not hasattr(auto_gptq, "QuantizeConfig") and hasattr(auto_gptq, "BaseQuantizeConfig"):
        auto_gptq.QuantizeConfig = auto_gptq.BaseQuantizeConfig
except ImportError:
    pass

from .core.configuration import AppConfig
from .core.registry import ModelRegistry
from .core.presets import register_presets
from .core.builder import GeneratorPipelineBuilder
from .core.router import run_app
from .core.config_utils import instantiate_recipe


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    if not override:
        return dict(base)
    out: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def _draft_params_to_dict(dp) -> Dict[str, Any]:
    if dp is None:
        return {}
    if is_dataclass(dp):
        return dict(asdict(dp))
    if hasattr(dp, "__dict__"):
        return dict(dp.__dict__)
    return {}


def _load_yaml_config(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as e:
        raise RuntimeError(
            "PyYAML is required for --config. Install it with `pip install pyyaml`."
        ) from e

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping/object at top-level, got {type(data).__name__}")
    return dict(data)




def _argv_has_flag(flag: str) -> bool:
    # Handles both: --flag value  and  --flag=value
    for a in sys.argv[1:]:
        if a == flag or a.startswith(flag + "="):
            return True
    return False


def _normalize_compile_mode(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
        return None
    return value


def _apply_yaml_overrides(default_config: Dict[str, Any], yaml_config: Dict[str, Any]) -> Dict[str, Any]:
    if not yaml_config:
        return dict(default_config)

    cfg = dict(yaml_config)
    cfg.pop("method", None)

    if "compile_mode" in cfg:
        cfg["compile_mode"] = _normalize_compile_mode(cfg.get("compile_mode"))

    # DraftParams can be specified as a dict in YAML.
    if isinstance(cfg.get("draft_params"), dict):
        from specdecodes.models.utils.utils import DraftParams

        base_dp = _draft_params_to_dict(default_config.get("draft_params"))
        merged_dp = _deep_merge_dict(base_dp, cfg["draft_params"])
        cfg["draft_params"] = DraftParams(**merged_dp)

    # generator_kwargs deep-merge.
    if isinstance(cfg.get("generator_kwargs"), dict):
        base_gk = default_config.get("generator_kwargs") or {}
        cfg["generator_kwargs"] = _deep_merge_dict(base_gk, cfg["generator_kwargs"])

    return _deep_merge_dict(default_config, cfg)

def main():
    # Reduce run-to-run drift from cuBLAS matmul reductions.
    # Important: set before the first CUDA context initialization.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    # 1. Register presets
    register_presets()

    # 2. Parse method + optional YAML config path first to load defaults
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--method", type=str, default="subspec_sd", help="Decoding method to use")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file. Values override method defaults; CLI args override YAML.",
    )
    
    # We use parse_known_args to get the method/config, then we can look up defaults
    args, _ = parser.parse_known_args()

    # Optional YAML config can supply defaults (and optionally method).
    yaml_config: Dict[str, Any] = {}
    if args.config:
        config_path = os.path.abspath(os.path.expanduser(args.config))
        if not os.path.exists(config_path):
            print(f"Config file not found: {config_path}")
            sys.exit(1)
        yaml_config = _load_yaml_config(config_path)

        # If user did NOT explicitly provide --method, allow YAML to select method.
        if not _argv_has_flag("--method") and isinstance(yaml_config.get("method"), str):
            args.method = yaml_config["method"]
    
    # 3. Get default config for the method
    method_entry = ModelRegistry.get(args.method)
    if method_entry is None:
        print(f"Unknown method: {args.method}. Available methods: {ModelRegistry.list_methods()}")
        sys.exit(1)
        
    default_config = method_entry.default_config.copy()

    # Merge YAML into default_config (defaults <- yaml).
    default_config = _apply_yaml_overrides(default_config, yaml_config)
    
    # 4. Create full parser for AppConfig
    # We populate arguments based on AppConfig fields, but respecting the method's defaults
    full_parser = argparse.ArgumentParser(parents=[parser], add_help=False)
    
    # Helper to add arguments from config dataclass
    # For simplicity, we manually add common ones or general ones. 
    # A robust solution would inspect the dataclass.
    # Here we just add the key ones we expect to override.
    
    full_parser.add_argument("--llm-path", type=str, default=default_config.get("llm_path", "meta-llama/Llama-3.1-8B-Instruct"))
    full_parser.add_argument("--draft-model-path", type=str, default=default_config.get("draft_model_path", None))
    full_parser.add_argument("--max-length", type=int, default=default_config.get("max_length", 2048))
    full_parser.add_argument("--seed", type=int, default=default_config.get("seed", 0))
    full_parser.add_argument("--device", type=str, default="cuda:0")
    full_parser.add_argument("--compile-mode", type=str, default=default_config.get("compile_mode", None))
    full_parser.add_argument("--temperature", type=float, default=default_config.get("temperature", 0.0))
    full_parser.add_argument("--do-sample", action="store_true", default=default_config.get("do_sample", False))
    full_parser.add_argument("--warmup-iter", type=int, default=default_config.get("warmup_iter", 0))

    full_parser.add_argument(
        "--cache-implementation",
        type=str,
        choices=["dynamic", "static"],
        default=default_config.get("cache_implementation", "dynamic"),
        help="KV-cache mode: dynamic or static",
    )

    # Common generator_kwargs override
    default_prefill = (default_config.get("generator_kwargs") or {}).get("prefill_chunk_size", None)
    full_parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=default_prefill,
        help="Generator prefill chunk size (sets generator_kwargs.prefill_chunk_size)",
    )

    # Allow turning profiling on/off explicitly.
    full_parser.add_argument(
        "--generator-profiling",
        action=argparse.BooleanOptionalAction,
        default=default_config.get("generator_profiling", True),
        help="Enable/disable generator profiling",
    )

    default_lossy_verify = bool(default_config.get("generator_kwargs", {}).get("lossy_verify", False))
    full_parser.add_argument(
        "--lossy-verify",
        action="store_true",
        default=default_lossy_verify,
        help="Use lossy verification (threshold + window) instead of exact verification for tree-based SD methods",
    )

    # DraftParams overrides (only applied when explicitly provided)
    full_parser.add_argument(
        "--lossy-threshold",
        type=float,
        default=None,
        help="Lossy SD: accept non-matching draft token if target prob >= this threshold",
    )
    full_parser.add_argument(
        "--lossy-window-size",
        type=int,
        default=None,
        help="Lossy SD: require this many future locally-correct draft nodes (lookahead)",
    )
    
    # Parse again with known args to override defaults
    # We still use parse_known_args because run_app (Typer) needs the rest
    config_args, typer_argv = full_parser.parse_known_args()
    
    # 5. Build AppConfig
    config = AppConfig()
    config.method = args.method
    
    # Update config from defaults
    config.update(default_config)
    
    # Update config from CLI args
    config.llm_path = config_args.llm_path
    config.draft_model_path = config_args.draft_model_path
    config.max_length = int(config_args.max_length)
    config.seed = int(config_args.seed)
    config.device = config_args.device
    config.compile_mode = _normalize_compile_mode(config_args.compile_mode)
    config.temperature = float(config_args.temperature)
    config.do_sample = bool(config_args.do_sample)
    config.warmup_iter = int(config_args.warmup_iter)
    config.cache_implementation = config_args.cache_implementation
    config.generator_profiling = bool(config_args.generator_profiling)

    # Apply generator_kwargs override(s)
    if config.generator_kwargs is None:
        config.generator_kwargs = {}
    if config_args.prefill_chunk_size is not None:
        config.generator_kwargs["prefill_chunk_size"] = int(config_args.prefill_chunk_size)

    # Verifier mode switch (applies to tree-based SD generators that route through verify_tree).
    config.generator_kwargs["lossy_verify"] = bool(config_args.lossy_verify)

    # Apply DraftParams overrides.
    # Note: these are verifier-specific and only affect methods that actually use draft_params.
    if config_args.lossy_threshold is not None or config_args.lossy_window_size is not None:
        from specdecodes.models.utils.utils import DraftParams

        if config.draft_params is None:
            config.draft_params = DraftParams()

        if config_args.lossy_threshold is not None:
            config.draft_params.lossy_threshold = float(config_args.lossy_threshold)
        if config_args.lossy_window_size is not None:
            config.draft_params.lossy_window_size = int(config_args.lossy_window_size)

    # Allow YAML to specify recipes via import path + kwargs.
    config.recipe = instantiate_recipe(getattr(config, "recipe", None))
    
    # 6. Build pipeline
    # We must patch sys.argv for Typer to work correctly on the subcommands
    # Typer expects [script, subcommand, options...]
    # We removed the config options, so we pass the rest.
    sys.argv = [sys.argv[0]] + typer_argv
    
    builder = GeneratorPipelineBuilder(config)
    run_app(builder)

if __name__ == "__main__":
    main()
