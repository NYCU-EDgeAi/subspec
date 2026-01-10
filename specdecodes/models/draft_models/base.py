import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.utils import is_torchdynamo_compiling
from safetensors.torch import load_model
from typing import List, Tuple, Optional, Dict
import logging
import os
import nvtx

from ..utils.utils import invert_mask


def load_custom_model(model, model_path, remove_embeddings=False):
    # Load the model
    missing_keys, unexpected_keys = load_model(model, model_path, strict=False)
    
    # Remove embed_tokens if not found (for custom models that uses LLM's embed_tokens)
    for key in missing_keys:
        if 'embed_tokens' in key:
            logging.info("embed_tokens not found. Use LLM's embed_tokens instead.")
            if remove_embeddings:
                del model.model.embed_tokens
    missing_keys = [key for key in missing_keys if 'embed_tokens' not in key]
    
    # error handling
    assert len(missing_keys) == 0 and len(unexpected_keys) == 0, f"Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}"
    
    return model

class TreeData(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_ids_data = []
        self.child_probs_data = []
        self.parent_indices_data = []
        
    def update(self, token_ids: torch.Tensor, child_probs: torch.Tensor, parent_indices: torch.Tensor) -> torch.Tensor:
        self.token_ids_data.append(token_ids)
        self.child_probs_data.append(child_probs)
        self.parent_indices_data.append(parent_indices)
    
    def get_data(self):
        token_ids_data = torch.cat(self.token_ids_data, dim=0).unsqueeze(0)
        child_probs_data = torch.cat(self.child_probs_data, dim=0).unsqueeze(0)
        parent_indices_data = torch.cat(self.parent_indices_data, dim=0).unsqueeze(0)
        return (token_ids_data, child_probs_data, parent_indices_data)
    
class TreeMaskCache:
    def __init__(self, prefix_len: int, sample_len: int, max_cache_len: int, dtype: str, device: str):
        self.prefix_len = prefix_len
        self.sample_len = sample_len
        self.max_cache_len = max_cache_len
        self.dtype = dtype
        self.device = device

        # Build static tree_mask only when the cache length is known and large enough.
        # Some cache implementations expose a small/non-max `max_cache_len` (e.g.,
        # current seq len), which would make the static mask too small and cause
        # shape mismatch errors during updates.
        use_static = (
            self.max_cache_len is not None
            and int(self.max_cache_len) >= int(self.prefix_len) + int(self.sample_len)
        )

        # build static tree_mask
        if use_static:
            self.tree_mask_update_method = 'static'
            self.tree_mask_cache = torch.zeros(
                (1, 1, self.sample_len, self.max_cache_len),
                device=self.device,
                dtype=torch.bool
            )
            if not is_torchdynamo_compiling():
                # Mark the buffer's address as static for optimization purposes
                torch._dynamo.mark_static_address(self.tree_mask_cache)
            
            # Initialize the first `prefix_len` elements to True
            self.tree_mask_cache[:, :, 0, :self.prefix_len] = True
            self.current_len = self.prefix_len
            
        # build dynamic tree_mask instead
        else:
            self.tree_mask_update_method = 'dynamic'
            self.tree_mask_cache = torch.ones(
                (1, 1, 1, self.prefix_len),
                device=self.device,
                dtype=torch.bool
            )
        # Create an identity block for later use
        self.eye_block = torch.eye(self.sample_len, device=self.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)

    def update_tree_mask(self, parent_indices: torch.Tensor,return_invert:bool=True) -> torch.Tensor:
        if self.tree_mask_update_method == 'static': # static tree mask update
            # Update existing mask based on parent indices
            self.tree_mask_cache[..., :self.current_len] = self.tree_mask_cache[..., parent_indices[0], :self.current_len]
            # Append the eye_block to the mask
            self.tree_mask_cache[..., self.current_len:self.current_len + self.sample_len] = self.eye_block
            # Update the current length
            self.current_len += self.sample_len
        else: 
            # Dynamically expand the mask by concatenating the eye_block
            tree_mask = self.tree_mask_cache[:, :, parent_indices[0]]
            self.tree_mask_cache = torch.concat((tree_mask, self.eye_block), dim=3)
        
        # Invert the mask and return
        if return_invert:
            return invert_mask(self.tree_mask_cache, dtype=self.dtype)
        else:
            return self.tree_mask_cache
    
    # return Inverted tree mask (same as update_tree_mask output)
    def get_tree_mask(self, return_invert:bool=True):
        if return_invert:
            return invert_mask(self.tree_mask_cache, dtype=self.dtype)
        else:
            return self.tree_mask_cache

class DraftModelBase(nn.Module):
    def __init__(self, base_model=None, target_model=None, eos_token_id=None, *model_args, **model_kwargs):
        super().__init__()
        self.eos_token_id = eos_token_id
        
        # Set model and config
        if base_model is not None and target_model is not None:
            raise ValueError("Only one of model or config must be provided.")   
        elif base_model is not None:
            self.model = base_model
        elif target_model is not None:
            self.model = self.init_base_model(target_model)
        else:
            raise ValueError("Either model or config must be provided.")

        # Initialize additional modules if needed
        self.init_additional_modules()
        
        # Set prefill function same as forward. 
        # prefill_forward() is used for prefill phase that cannot torch.compile()
        self.prefill_forward = self.forward
        
    @property
    def dtype(self):
        return self.model.dtype
    
    @property
    def device(self):
        return self.model.device
    
    @property
    def config(self):
        return self.model.config
    
        
    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path,
        *model_args,
        target_model = None,
        torch_dtype=torch.float32,
        remove_embeddings = False, #! Deprecated
        **model_kwargs
    ):
        # Remove the following arguments from model_kwargs, cause AutoModelForCausalLM does not accept them
        eos_token_id = model_kwargs.pop("eos_token_id", None)
        
        # Load HuggingFace model if config is not provided
        # if target_model is not None: 
        #if pretrained_model_name_or_path path exists on the local disk, load the model from the path
        if os.path.exists(pretrained_model_name_or_path):
            logging.info(f"Loading model from {pretrained_model_name_or_path}")
            draft_model_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
            model = cls(target_model=target_model, eos_token_id=eos_token_id, *model_args, **model_kwargs)
            load_custom_model(model, draft_model_path, remove_embeddings=remove_embeddings)

        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                *model_args, 
                **model_kwargs
            )
            model = cls(base_model, eos_token_id=eos_token_id, *model_args, **model_kwargs).to(dtype=torch_dtype)
        
        # Convert the model to the desired dtype and return
        model.to(dtype=torch_dtype)
        return model
        
    def init_base_model(self, target_model):
        raise NotImplementedError
    
    def init_additional_modules(self):
        pass
    
    def update_modules(self, **kwargs):
        pass
        
    def get_input_embeddings(self):
        # If the model has input embeddings, return it. Otherwise, return None
        if hasattr(self.model, "embed_tokens"):
            return self.model.embed_tokens
        else:
            return None
        
    @torch.no_grad()
    def forward(self, input_ids, *model_args, **kwargs):
        raise NotImplementedError
    
    @torch.no_grad()
    def speculate(self, input_ids, past_key_values, **kwargs):
        raise NotImplementedError
        
    # Currently not used. This may be used to match LLM's sampling behavior.
    @torch.no_grad()
    def _sample_probs(
        self,
        logits: torch.FloatTensor,
        logits_warper,
        do_sample: bool,
    ):
        if do_sample:
            batch, seq_len, vocab_size = logits.shape
            
            logits = logits.view(-1, vocab_size)
            next_token_scores = logits_warper(None, logits)
            probs = torch.softmax(next_token_scores, dim=-1)
            return probs.view(batch, seq_len, vocab_size) # preserve shape
        
        else:
            return torch.softmax(logits, dim=-1)
        
    @torch.no_grad()
    def topk_sampling(
        self,
        sampled_probs: torch.Tensor, 
        parent_probs: torch.Tensor, 
        sample_k: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        with nvtx.annotate("topk_sampling/contiguous"):
            sampled_probs = sampled_probs.contiguous()
            parent_probs = parent_probs.contiguous()

        return self._topk_flatten(sampled_probs, parent_probs, sample_k)

    def _topk_flatten(
        self,
        sampled_probs: torch.Tensor,
        parent_probs: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_leaves, vocab_size = sampled_probs.shape
        k = min(int(k), int(vocab_size))

        buf = getattr(self, "_topk_global_scores_buf", None)
        if (
            buf is None
            or buf.shape != sampled_probs.shape
            or buf.dtype != sampled_probs.dtype
            or buf.device != sampled_probs.device
        ):
            buf = torch.empty_like(sampled_probs)
            self._topk_global_scores_buf = buf

        torch.mul(sampled_probs, parent_probs.unsqueeze(-1), out=buf)
        flattened_probs = buf.view(batch_size, -1)
        topk_probs, topk_indices = torch.topk(flattened_probs, k, dim=1, sorted=True)
        parent_indices = (topk_indices // vocab_size).long()
        token_ids = (topk_indices % vocab_size).long()
        return token_ids, topk_probs, parent_indices

    def _topk_flatten_for_graph(
        self,
        sampled_probs_ref: torch.Tensor,
        parent_probs_buf: torch.Tensor,
        outbuf: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact flatten+topk path used by the CUDA-graph sampling runner."""
        batch_size, _, vocab_size = sampled_probs_ref.shape
        k = min(int(k), int(vocab_size))

        torch.mul(sampled_probs_ref, parent_probs_buf.unsqueeze(-1), out=outbuf)
        flattened_probs = outbuf.view(batch_size, -1)
        topk_probs, topk_indices = torch.topk(flattened_probs, k, dim=1, sorted=True)
        parent_indices = (topk_indices // vocab_size).long()
        token_ids = (topk_indices % vocab_size).long()
        return token_ids, topk_probs, parent_indices
    
    def set_past_key_values(self, past_key_values):
        self.past_key_values = past_key_values
        
    def get_tree(self):
        return self.tree

    def _get_kv_len_int(self) -> int:
        kv_len = self.past_key_values.get_seq_length()
        if isinstance(kv_len, torch.Tensor):
            kv_len = kv_len.item()
        return int(kv_len)