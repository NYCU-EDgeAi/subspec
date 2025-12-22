import sys
import argparse

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

def main():
    # 1. Register presets
    register_presets()

    # 2. Parse method argument first to load defaults
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--method", type=str, default="subspec_sd", help="Decoding method to use")
    
    # We use parse_known_args to get the method, then we can look up defaults
    args, remaining_argv = parser.parse_known_args()
    
    # 3. Get default config for the method
    method_entry = ModelRegistry.get(args.method)
    if method_entry is None:
        print(f"Unknown method: {args.method}. Available methods: {ModelRegistry.list_methods()}")
        sys.exit(1)
        
    default_config = method_entry.default_config.copy()
    
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
    full_parser.add_argument("--device", type=str, default="cuda:0")
    full_parser.add_argument("--compile-mode", type=str, default=default_config.get("compile_mode", None))
    full_parser.add_argument("--temperature", type=float, default=default_config.get("temperature", 0.0))
    full_parser.add_argument("--do-sample", action="store_true", default=default_config.get("do_sample", False))
    full_parser.add_argument("--warmup-iter", type=int, default=default_config.get("warmup_iter", 0))
    
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
    if config_args.draft_model_path:
        config.draft_model_path = config_args.draft_model_path
    config.max_length = config_args.max_length
    config.device = config_args.device
    config.compile_mode = config_args.compile_mode
    if config.compile_mode and config.compile_mode.lower() == "none":
        config.compile_mode = None
    config.temperature = config_args.temperature
    config.do_sample = config_args.do_sample
    config.warmup_iter = config_args.warmup_iter
    
    # 6. Build pipeline
    # We must patch sys.argv for Typer to work correctly on the subcommands
    # Typer expects [script, subcommand, options...]
    # We removed the config options, so we pass the rest.
    sys.argv = [sys.argv[0]] + typer_argv
    
    builder = GeneratorPipelineBuilder(config)
    run_app(builder)

if __name__ == "__main__":
    main()
