import torch
from dataclasses import dataclass

def invert_mask(mask, dtype): 
    # Inversion using bitwise NOT and multiplication
    return (~mask).to(dtype) * torch.finfo(dtype).min

@dataclass
class DraftParams:
    temperature: float = 1
    max_depth: int = 6
    topk_len: int = 10
    max_verify_tokens: int = None
    # Lossy tree-based verification parameters
    lossy_threshold: float = 0.0
    lossy_window_size: int = 1
    
    def __post_init__(self):
        self.max_sample_tokens = self.max_depth * self.topk_len + 1
        self.max_verify_tokens = min(self.max_sample_tokens, self.max_verify_tokens) if self.max_verify_tokens is not None else self.max_sample_tokens