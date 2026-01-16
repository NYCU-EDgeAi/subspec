from __future__ import annotations

import sys
import argparse
import os
import shutil
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .core.configuration import AppConfig


BENCHMARK_COMMANDS = {"run-benchmark", "run-benchmark-acc", "run-benchmark-agent"}


def _configure_allocator_env(default: str = "expandable_segments:True") -> None:
    """Configure PyTorch allocator env vars.

    Some PyTorch builds still apply CUDA allocator settings more reliably via
    PYTORCH_CUDA_ALLOC_CONF, while newer versions encourage PYTORCH_ALLOC_CONF.
    We support both by mirroring values and providing stable defaults.
    """

    if "PYTORCH_ALLOC_CONF" in os.environ and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ["PYTORCH_ALLOC_CONF"]
        return

    if "PYTORCH_CUDA_ALLOC_CONF" in os.environ and "PYTORCH_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_ALLOC_CONF"] = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
        return

    os.environ.setdefault("PYTORCH_ALLOC_CONF", default)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", default)


def _maybe_patch_auto_gptq() -> None:
    """Monkey patch for auto_gptq compatibility with optimum (best-effort)."""

    try:
        import auto_gptq  # type: ignore[import-not-found]

        if not hasattr(auto_gptq, "QuantizeConfig") and hasattr(auto_gptq, "BaseQuantizeConfig"):
            auto_gptq.QuantizeConfig = auto_gptq.BaseQuantizeConfig
    except ImportError:
        pass


def _configure_runtime_environment() -> None:
    # Reduce run-to-run drift from cuBLAS matmul reductions.
    # Important: set before the first CUDA context initialization.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    # Keep allocator behavior stable by default (can be overridden via env).
    _configure_allocator_env(default="expandable_segments:True")


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


def _resolve_existing_path(path: str) -> str:
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(resolved):
        raise FileNotFoundError(resolved)
    return resolved


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
        merged_dp = _deep_merge_dict(base_dp, dict(cfg["draft_params"]))
        cfg["draft_params"] = DraftParams(**merged_dp)

    # generator_kwargs deep-merge.
    if isinstance(cfg.get("generator_kwargs"), dict):
        base_gk = default_config.get("generator_kwargs") or {}
        cfg["generator_kwargs"] = _deep_merge_dict(base_gk, cfg["generator_kwargs"])

    return _deep_merge_dict(default_config, cfg)


def _build_base_parser() -> argparse.ArgumentParser:
    # Important: disable allow_abbrev so Typer subcommand flags like --d/--k
    # don't get parsed as abbreviations for top-level options (e.g., --device).
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Decoding method to use (overrides YAML `method`)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML config file. Required. Values override method defaults; CLI args override YAML.",
    )

    # Research toggles (parsed early so we can optionally re-exec under Nsight Systems).
    parser.add_argument(
        "--nvtx-profiling",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable NVTX profiling via nsys re-exec (overrides YAML)",
    )
    parser.add_argument(
        "--nsys-output",
        type=str,
        default=None,
        help="Nsight Systems output base name (overrides YAML)",
    )
    parser.add_argument(
        "--detailed-analysis",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable detailed analysis logging (overrides YAML)",
    )
    return parser


def _maybe_reexec_with_nsys(enabled: bool, output: str) -> None:
    if not enabled:
        return

    # Avoid infinite recursion when we re-exec under nsys.
    if os.environ.get("SUBSPEC_NSYS_ACTIVE", "0") == "1":
        return

    if shutil.which("nsys") is None:
        print("Error: NVTX profiling requested but `nsys` was not found in PATH.")
        sys.exit(1)

    os.environ["SUBSPEC_NSYS_ACTIVE"] = "1"

    # Mirrors the previous wrapper-script settings, but lives in Python so it's config-driven.
    cmd = [
        "nsys",
        "profile",
        "-w",
        "true",
        "-t",
        "cuda,nvtx,osrt,cudnn,cublas",
        "-s",
        "cpu",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop-shutdown",
        "--cudabacktrace=all",
        "--force-overwrite=true",
        "--python-sampling-frequency=1000",
        "--python-sampling=true",
        "--cuda-memory-usage=true",
        "--gpuctxsw=true",
        "--python-backtrace",
        "-x",
        "true",
        "-o",
        output,
        sys.executable,
        "-m",
        "run.main",
        *sys.argv[1:],
    ]
    os.execvp(cmd[0], cmd)


def _build_full_parser(base_parser: argparse.ArgumentParser, default_config: Dict[str, Any]) -> argparse.ArgumentParser:
    full_parser = argparse.ArgumentParser(parents=[base_parser], add_help=False, allow_abbrev=False)

    full_parser.add_argument(
        "--llm-path",
        type=str,
        default=default_config.get("llm_path", "meta-llama/Llama-3.1-8B-Instruct"),
    )
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

    # generator_kwargs overrides
    default_prefill = (default_config.get("generator_kwargs") or {}).get("prefill_chunk_size", None)
    full_parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=default_prefill,
        help="Generator prefill chunk size (sets generator_kwargs.prefill_chunk_size)",
    )

    full_parser.add_argument(
        "--generator-profiling",
        action=argparse.BooleanOptionalAction,
        default=default_config.get("generator_profiling", True),
        help="Enable/disable generator profiling",
    )

    default_verify_method = str((default_config.get("generator_kwargs") or {}).get("verify_method", "exact") or "exact").strip().lower()
    default_threshold_method = str(
        ((default_config.get("generator_kwargs") or {}).get("verify_kwargs") or {}).get(
            "threshold_method", "entropy"
        )
        or "entropy"
    ).strip().lower()
    full_parser.add_argument(
        "--verify-method",
        type=str,
        choices=["exact", "lossy"],
        default=default_verify_method,
        help="Verification method for tree-based SD: exact or lossy",
    )

    full_parser.add_argument(
        "--threshold",
        "-e",
        type=float,
        default=None,
        help=(
            "Lossy verify threshold: entropy gate (h_j < threshold) when threshold_method=entropy, "
            "or target prob >= threshold when threshold_method=prob"
        ),
    )
    full_parser.add_argument(
        "--threshold-method",
        type=str,
        choices=["entropy", "prob"],
        default=default_threshold_method,
        help="Lossy verify threshold method: entropy (paper) or prob (probability)",
    )
    full_parser.add_argument(
        "--window-size",
        "-w",
        type=int,
        default=None,
        help="Lossy verify: require this many future locally-correct nodes (lookahead)",
    )

    return full_parser


def _enforce_benchmark_requires_config(typer_argv: list[str], config_path: str | None) -> None:
    if typer_argv and typer_argv[0] in BENCHMARK_COMMANDS and config_path is None:
        print(
            "Error: benchmark commands require a YAML config via --config.\n"
            "Example: python -m run.main --config configs/methods/subspec_sd.yaml run-benchmark --benchmarks mt-bench --max-samples 20"
        )
        sys.exit(2)


def _resolve_method(cli_method: str | None, yaml_config: Dict[str, Any]) -> str:
    if isinstance(cli_method, str) and cli_method.strip():
        return cli_method
    if isinstance(yaml_config.get("method"), str) and yaml_config["method"].strip():
        return yaml_config["method"]
    raise ValueError("Missing `method`: specify --method or set `method:` in the YAML config.")


def _apply_cli_overrides(config: AppConfig, config_args: argparse.Namespace) -> None:
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

    # Optional research toggles: only override when explicitly provided.
    if getattr(config_args, "detailed_analysis", None) is not None:
        config.detailed_analysis = bool(config_args.detailed_analysis)
    if getattr(config_args, "nvtx_profiling", None) is not None:
        config.nvtx_profiling = bool(config_args.nvtx_profiling)
    if getattr(config_args, "nsys_output", None) is not None:
        config.nsys_output = str(config_args.nsys_output)


def _apply_generator_kwargs_overrides(config: AppConfig, config_args: argparse.Namespace) -> None:
    if config.generator_kwargs is None:
        config.generator_kwargs = {}

    if config_args.prefill_chunk_size is not None:
        config.generator_kwargs["prefill_chunk_size"] = int(config_args.prefill_chunk_size)

    # Verifier selection + method kwargs.
    config.generator_kwargs["verify_method"] = str(getattr(config_args, "verify_method", "exact") or "exact").strip().lower()
    config.generator_kwargs.setdefault("verify_kwargs", {})
    if getattr(config_args, "threshold", None) is not None:
        config.generator_kwargs["verify_kwargs"]["threshold"] = float(config_args.threshold)
    if getattr(config_args, "threshold_method", None) is not None:
        config.generator_kwargs["verify_kwargs"]["threshold_method"] = str(
            config_args.threshold_method
        ).strip().lower()
    if getattr(config_args, "window_size", None) is not None:
        config.generator_kwargs["verify_kwargs"]["window_size"] = int(config_args.window_size)


def _apply_draft_params_overrides(config: AppConfig, config_args: argparse.Namespace) -> None:
    return


def _load_yaml_and_method(args: argparse.Namespace) -> tuple[str, Dict[str, Any], str]:
    try:
        config_path = _resolve_existing_path(args.config)
    except FileNotFoundError:
        print(f"Config file not found: {os.path.abspath(os.path.expanduser(args.config))}")
        sys.exit(1)

    yaml_config = _load_yaml_config(config_path)

    try:
        method = _resolve_method(args.method, yaml_config)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(2)

    return config_path, yaml_config, method


def _effective_nsys_settings(args: argparse.Namespace, yaml_config: Dict[str, Any]) -> tuple[bool, str]:
    enabled = (
        bool(args.nvtx_profiling)
        if args.nvtx_profiling is not None
        else bool(yaml_config.get("nvtx_profiling", False))
    )
    output = (
        str(args.nsys_output)
        if args.nsys_output is not None
        else str(yaml_config.get("nsys_output", "nsight_report"))
    )
    return enabled, output


def _configure_wandb_flags(config: "AppConfig") -> None:
    # Propagate global research flags via wandb_logger (avoids env var plumbing).
    try:
        from specdecodes.models.utils.wandb_logger import wandb_logger

        wandb_logger.set_flags(
            detailed_analysis=bool(getattr(config, "detailed_analysis", False)),
            nvtx_profiling=bool(getattr(config, "nvtx_profiling", False)),
        )
    except Exception:
        # Keep main robust even if wandb_logger isn't importable in some minimal setups.
        pass


def _build_app_config(
    *,
    AppConfig: type["AppConfig"],
    method: str,
    default_config: Dict[str, Any],
    config_args: argparse.Namespace,
) -> "AppConfig":
    config = AppConfig()
    config.method = method
    config.update(default_config)

    _apply_cli_overrides(config, config_args)
    _apply_generator_kwargs_overrides(config, config_args)
    return config


def main():
    # Configure env + compatibility patches before importing heavy GPU code.
    _configure_runtime_environment()
    _maybe_patch_auto_gptq()

    # 1) Parse method + YAML config path first to load defaults
    base_parser = _build_base_parser()
    args, _ = base_parser.parse_known_args()
    _, yaml_config, method = _load_yaml_and_method(args)

    # If enabled via YAML/CLI, re-exec under Nsight Systems *before* importing heavy GPU code.
    nsys_enabled, nsys_output = _effective_nsys_settings(args, yaml_config)
    _maybe_reexec_with_nsys(nsys_enabled, nsys_output)

    # Import project modules lazily so env defaults above apply before any torch/CUDA init.
    from .core.configuration import AppConfig
    from .core.registry import ModelRegistry
    from .core.presets import register_presets
    from .core.builder import GeneratorPipelineBuilder
    from .core.router import run_app
    from .core.config_utils import instantiate_recipe

    # 2) Register presets (after optional nsys re-exec)
    register_presets()
    
    # 3) Get default config for the method
    method_entry = ModelRegistry.get(method)
    if method_entry is None:
        print(f"Unknown method: {method}. Available methods: {ModelRegistry.list_methods()}")
        sys.exit(1)
        
    default_config = method_entry.default_config.copy()

    # Merge YAML into default_config (defaults <- yaml).
    default_config = _apply_yaml_overrides(default_config, yaml_config)
    
    # 4) Build full parser for AppConfig (method defaults <- YAML; CLI overrides both)
    full_parser = _build_full_parser(base_parser, default_config)
    
    # Parse again with known args to override defaults
    # We still use parse_known_args because run_app (Typer) needs the rest
    config_args, typer_argv = full_parser.parse_known_args()

    # (Kept for backward compatibility + explicit error messaging if this file is reused elsewhere.)
    _enforce_benchmark_requires_config(typer_argv, args.config)
    
    # 5) Build AppConfig
    config = _build_app_config(
        AppConfig=AppConfig,
        method=method,
        default_config=default_config,
        config_args=config_args,
    )
    _configure_wandb_flags(config)

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
