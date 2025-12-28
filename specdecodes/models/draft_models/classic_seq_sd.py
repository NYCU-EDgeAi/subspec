import torch
import torch.nn as nn
import nvtx

from specdecodes.models.utils.wandb_logger import wandb_logger

from .base import DraftModelBase

    
class ClassicSDDraftModel(DraftModelBase):
    def forward(self, input_ids, with_softmax=False, *model_args, **kwargs):
        logits = self.model(input_ids, *model_args, **kwargs).logits
        if with_softmax:
            logits = torch.softmax(logits/self.draft_params.temperature, dim=-1)
            
        return logits
    
    @torch.no_grad()
    def speculate(self, input_ids, **kwargs):
        # 1) Obtain necessary parameters
        device = input_ids.device
        batch_size, input_len = input_ids.shape
        assert batch_size == 1, "Only support batch_size=1 for now."
        
        # 2) Initialize kv_len & cache_position
        with nvtx.annotate("Initialize kv_len & cache_position"):
            kv_len = self.past_key_values.get_seq_length()
            # convert kv_len to int if it is a tensor
            if isinstance(kv_len, torch.Tensor):
                kv_len = kv_len.item()
            
        # 3) First forward pass
        cache_position = torch.arange(kv_len, input_len, dtype=torch.long, device=device)
        with nvtx.annotate("ssm first forward", color="red"):
            sampled_probs = self.prefill_forward(
                input_ids[:, kv_len:],
                with_softmax=True,
                past_key_values=self.past_key_values.cache,
                position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position,
                logits_to_keep=1,
            )
            kv_len = input_len
            self.past_key_values.seq_len = input_len
            
        with nvtx.annotate("sample nodes", color="green"): #torch.argmax(logits, dim=-1)
            sampled_token = torch.argmax(sampled_probs[:, -1:], dim=-1)
        
        # 4) Initialize TreeData & TreeMaskCache to manage tree structure and intermediate data. 
        self.token_ids = []
        self.token_ids.append(input_ids[:, -1:])
        self.token_ids.append(sampled_token)
        self.cache_position = torch.arange(kv_len, kv_len+self.draft_params.topk_len, dtype=torch.long, device=device)
        
        # 6) Main loop
        for depth_i in range(self.draft_params.max_depth-1):
            self.speculate_once()
        
        return torch.cat(self.token_ids, dim=-1)
    
    @torch.no_grad()
    def speculate_once(self, **kwargs):
        token_ids = self.token_ids
        cache_position = self.cache_position

        with nvtx.annotate("ssm forward", color="red"):
            sampled_probs = self(
                token_ids[-1], 
                with_softmax=True,
                position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position,
                past_key_values=self.past_key_values.cache,
            )
        with nvtx.annotate("sample nodes", color="green"):
            sampled_token = torch.argmax(sampled_probs[:, -1, :], dim=-1, keepdim=True)
            token_ids.append(sampled_token)
            
        if wandb_logger.get_flag("detailed_analysis", False):
            self.draft_prob.append(torch.max(sampled_probs[:, -1, :]).cpu().item())
            
        # Update internal state
        self.token_ids = token_ids
        self.cache_position += 1