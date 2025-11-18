import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple, Union
from transformers.activations import ACT2FN

from flashinfer.activation import silu_and_mul
from flashinfer.norm import (
    fused_add_rmsnorm,
    rmsnorm,
)
        
class FiLlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        bsz, seq_len, hidden_size = hidden_states.size()
        if residual is not None:
            fused_add_rmsnorm(hidden_states, residual, self.weight.data, self.variance_epsilon)
            return hidden_states, residual
        
        hidden_states = rmsnorm(
            hidden_states.view(bsz * seq_len, hidden_size),
            self.weight,
            eps=self.variance_epsilon,
        )
        return hidden_states.view(bsz, seq_len, hidden_size)
    
class LigerSwiGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")
        
        self.act_fn = ACT2FN[config.hidden_act]
        # self.fi_act_fn = SiluAndMul()

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
            return down_proj
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            return down_proj
        
            # gate_output = self.gate_proj(x)
            # up_output = self.up_proj(x)
            intermediate_output = self.fi_act_fn(torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1))
            down_proj = self.down_proj(intermediate_output)
            
            # down_proj1 = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            
            # result = torch.allclose(down_proj, down_proj1, rtol=1e-03, atol=1e-05)

            # if result is False:
            #     raise ValueError("differ")

        return down_proj