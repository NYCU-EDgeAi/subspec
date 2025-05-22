import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from flashinfer.norm import (
    fused_add_rmsnorm,
    rmsnorm,
)

class FusedRMSNorm(nn.Module):
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