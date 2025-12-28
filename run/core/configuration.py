from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
import torch
from specdecodes.models.utils.utils import DraftParams

@dataclass
class AppConfig:
    # Base configurations
    method: str = "classic_sd"
    vram_limit_gb: Optional[int] = None
    seed: int = 0
    device: str = "cuda:0"
    dtype: torch.dtype = torch.float16
    
    # Model paths
    llm_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    draft_model_path: Optional[str] = None
    
    # Generation parameters
    max_length: int = 2048
    do_sample: bool = False
    temperature: float = 0.0
    
    # Generator-specific configurations
    generator_kwargs: Dict[str, Any] = field(default_factory=dict)
    draft_params: Optional[DraftParams] = None
    
    # Recipe
    recipe: Any = None
    cpu_offload_gb: Optional[int] = None
    
    # Additional configurations
    cache_implementation: str = "dynamic"
    warmup_iter: int = 0
    compile_mode: Optional[str] = None
    
    # Profiling
    generator_profiling: bool = True
    profiling_verbose: bool = True
    print_time: bool = True
    print_message: bool = True
    
    # Benchmarking/logging
    out_dir: Optional[str] = None
    log_dir: str = "experiments"

    # Research toggles (set via YAML/CLI)
    detailed_analysis: bool = False
    nvtx_profiling: bool = False
    nsys_output: str = "nsight_report"

    def update(self, new_config: Dict[str, Any]):
        for key, value in new_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
