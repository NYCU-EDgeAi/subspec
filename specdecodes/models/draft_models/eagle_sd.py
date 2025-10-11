import torch
import torch.nn as nn
from transformers import AutoModel
import nvtx

from copy import deepcopy
from ..utils.cpu_tree import Tree
from .classic_sd import ClassicSDDraftModel, TreeData, TreeMaskCache


class MergeLinear(nn.Module):
    def __init__(self, in_shape, out_shape, bias=True):
        super().__init__()
        self.fc = nn.Linear(in_shape, out_shape, bias=bias)

    def forward(self, x, emb):
        # swapped (x, emb) to (emb, x) to match official implementation of Eagle
        return self.fc(torch.cat((emb, x), dim=-1))

class EagleSDDraftModel(ClassicSDDraftModel):
    def init_base_model(self, target_model):
        draft_config = deepcopy(target_model.config)
        draft_config.num_hidden_layers = 1
        
        # No bias for Llama2
        if draft_config._name_or_path.startswith("meta-llama/Llama-2"):
            self.bias = False
        else:
            self.bias = True
        
        # No GQA for draft model on Qwen
        if draft_config._name_or_path.startswith("Qwen/Qwen2.5"):
            draft_config.num_key_value_heads=draft_config.num_attention_heads
        
        model = AutoModel.from_config(draft_config)
        
        # replace model.norm and first input_layernorm with nn.Identity
        model.norm = nn.Identity()
        model.layers[0].input_layernorm = nn.Identity()
        
        # remove embed_tokens
        if hasattr(model, "embed_tokens"): 
            del model.embed_tokens

        # set _init_weights to empty function
        model._init_weights = lambda x: None

        return model

    def init_additional_modules(self):
        self.fusion = MergeLinear(self.config.hidden_size*2, self.config.hidden_size, bias=self.bias)
        
    def update_modules(self, embed_tokens=None, lm_head=None, **kwargs):
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        if lm_head is not None:
            self.lm_head = lm_head
    
    def forward(self, input_ids, hidden_states, logits_to_keep=0, with_softmax=False, *model_args, **kwargs):
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = self.fusion(hidden_states, inputs_embeds)
        hidden_states = self.model(inputs_embeds=hidden_states, *model_args, **kwargs)[0][:, -logits_to_keep:]
        logits = self.lm_head(hidden_states)
        
        if with_softmax:
            logits = torch.softmax(logits/self.draft_params.temperature, dim=-1)
            
        return logits, hidden_states
    
    @torch.no_grad()
    def speculate(self, input_ids, hidden_states, **kwargs):
        # 1-1) Remove the first token from input_ids (shift by 1)
        input_ids = input_ids[:, 1:]
        
        # 1-2) Obtain necessary parameters
        device = input_ids.device
        if hasattr(self.model, "lm_head"):
            dtype = self.model.lm_head.weight.dtype
        else:
            dtype = self.lm_head.weight.dtype
        batch_size, input_len = input_ids.shape
        max_cache_len = getattr(self.past_key_values.cache, "max_cache_len", None)
        assert batch_size == 1, "Only support batch_size=1 for now."
        
        # 2) Initialize kv_len & cache_position
        with nvtx.annotate("Initialize kv_len & cache_position"):
            kv_len = self.past_key_values.get_seq_length()
            # convert kv_len to int if it is a tensor
            if isinstance(kv_len, torch.Tensor):
                kv_len = kv_len.item()
            
        # 5) First forward pass
        cache_position = torch.arange(kv_len, input_len, dtype=torch.long, device=device)
        with nvtx.annotate("ssm first forward", color="red"):
            sampled_probs, hidden_states = self.prefill_forward(
                input_ids[:, kv_len:],
                hidden_states=hidden_states,
                with_softmax=True,
                past_key_values=self.past_key_values.cache,
                position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position,
                logits_to_keep=1,
            )
            kv_len = input_len
            self.past_key_values.seq_len = input_len
            
        with nvtx.annotate("sample nodes", color="green"):
            self.parent_probs = torch.ones((1, 1), device=device, dtype=dtype)
            token_ids, child_probs, parent_indices = self.topk_sampling(
                sampled_probs,
                self.parent_probs,
                self.draft_params.topk_len
            )
            self.parent_probs = child_probs
            
        with nvtx.annotate("filter"):
            # Expand parent_indices to match hidden_states along the last dimension
            parent_indices_expanded = parent_indices.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
            hidden_states = torch.gather(hidden_states, dim=1, index=parent_indices_expanded)
        
        # 4) Initialize TreeData & TreeMaskCache to manage tree structure and intermediate data.
        root_id = input_ids[0, -1]
        self.tree = Tree(root_id, dtype)
        self.tree_data = TreeData()
        self.tree_mask_cache = TreeMaskCache(
            prefix_len=kv_len,
            sample_len=self.draft_params.topk_len,
            max_cache_len=max_cache_len,
            dtype=dtype,
            device=device,
        )
        
        # 5) First update of tree_data and tree_mask_cache
        with nvtx.annotate("update tree_data & tree_mask", color="green"):
            self.tree_data.update(token_ids, child_probs, parent_indices)
            self.tree_mask_cache.update_tree_mask(parent_indices)
        
        # Set initial state for the speculative tree
        self.token_ids = token_ids
        self.hidden_states = hidden_states
        self.position_ids = torch.full((batch_size, self.draft_params.topk_len), kv_len, device=device, dtype=torch.long)
        self.cache_position = torch.arange(kv_len, kv_len+self.draft_params.topk_len, dtype=torch.long, device=device)

        # 6) Main loop
        for depth_i in range(self.draft_params.max_depth-1):
            self.speculate_once()
             
        # Update and obtain the final tree
        self.update_tree(self.tree_data)
        return self.tree
    
    @torch.no_grad()
    def speculate_once(self, **kwargs):
        tree_attention_mask = self.tree_mask_cache.get_tree_mask()
        token_ids = self.token_ids
        hidden_states = self.hidden_states
        parent_probs = self.parent_probs
        position_ids = self.position_ids
        cache_position = self.cache_position
        
        with nvtx.annotate("ssm forward", color="red"):
            sampled_probs, hidden_states = self(
                token_ids,
                hidden_states=hidden_states,
                with_softmax=True,
                past_key_values=self.past_key_values.cache,
                position_ids=position_ids,
                attention_mask=tree_attention_mask,
                cache_position=cache_position,
            )
        
        with nvtx.annotate("sample nodes", color="green"):
            token_ids, child_probs, parent_indices = self.topk_sampling(
                sampled_probs,
                parent_probs,
                self.draft_params.topk_len
            )
            parent_probs = child_probs
            
        with nvtx.annotate("filter"):
            # Expand parent_indices to match hidden_states along the last dimension
            parent_indices_expanded = parent_indices.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
            hidden_states = torch.gather(hidden_states, dim=1, index=parent_indices_expanded)
            
        with nvtx.annotate("update tree_data & tree_mask", color="green"):
            self.tree_data.update(token_ids, child_probs, parent_indices)
            self.tree_mask_cache.update_tree_mask(parent_indices)
            
        # Update internal state
        self.token_ids = token_ids
        self.parent_probs = parent_probs
        self.position_ids += 1
        self.cache_position += self.draft_params.topk_len
    
    def final_update(self, input_ids, hidden_states, **kwargs):
        input_ids = input_ids[:, 1:]
        
        device = input_ids.device
        batch_size, input_len = input_ids.shape
        kv_len = self.past_key_values.get_seq_length()
        assert batch_size == 1, "Only support batch_size=1 for now."
        
        cache_position = torch.arange(kv_len, input_len, dtype=torch.long, device=device)
        self.prefill_forward(
            input_ids[:, kv_len:],
            with_softmax=True,
            hidden_states=hidden_states,
            past_key_values=self.past_key_values.cache,
            cache_position=cache_position,
            logits_to_keep=1,
        )
        self.past_key_values.seq_len = input_len