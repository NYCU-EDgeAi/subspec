from typing import Any, List, Optional, Dict, Tuple, Union
import torch
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.configuration_utils import PretrainedConfig

def create_kv_cache(
    cache_implementation = "dynamic",
    max_cache_len=None,
    max_batch_size=None,
    config=None,
    device='cpu',
    dtype='float16',
):
    if cache_implementation == "dynamic":
        return TreeDynamicCache()
    
    elif cache_implementation == "static":
        return TreeStaticCache(
            max_cache_len=max_cache_len,
            max_batch_size=max_batch_size,
            config=config,
            device=device,
            dtype=dtype,
        )

class TreeDynamicCache(DynamicCache):
    def __init__(self) -> None:
        super().__init__()
        
    def crop(self, start: int, end = None, dim=0):
        """Crop the past key/values up to a new `max_length` (negative removes from the end)."""
        if end is None:
            end = self.get_seq_length()
            
        if start < 0:
            start = end - abs(start)
        if end <= start:
            return

        self._seen_tokens = start
        for i in range(len(self.key_cache)):
            if self.key_cache[i] != []:
                self.key_cache[i] = self.key_cache[i][..., :start, :]
                self.value_cache[i] = self.value_cache[i][..., :start, :]
                
    def reorder_cache(self, beam_idx: torch.LongTensor, dim=0):
        """Reorder cache for beam search (classic approach)."""
        for i in range(len(self.key_cache)):
            dev = self.key_cache[i].device
            self.key_cache[i] = self.key_cache[i].index_select(dim, beam_idx.to(dev))
            self.value_cache[i] = self.value_cache[i].index_select(dim, beam_idx.to(dev))
            
    def reorder_cache_with_offset(self, beam_idx: torch.LongTensor, new_chunk_len=1, offset=0, dim=0):
        """
        Reorder the cache for beam search with an offset. 
        [:offset] remain unchanged; [offset:] is reordered.
        """
        # Build the full reorder indices
        full_beam_idx = torch.cat(
            [torch.arange(offset, device=beam_idx.device), beam_idx + offset], dim=0
        )
        beam_idx_device_cache = {}

        for i in range(len(self.key_cache)):
            dev = self.key_cache[i].device
            if dev not in beam_idx_device_cache:
                beam_idx_device_cache[dev] = full_beam_idx.to(dev)
            r_idx = beam_idx_device_cache[dev]
            
            self.key_cache[i] = self.key_cache[i].index_select(dim, r_idx)
            self.value_cache[i] = self.value_cache[i].index_select(dim, r_idx)
            
    def reset(self):
        """Resets the cache."""
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []


class TreeStaticCache(StaticCache):
    def __init__(
        self,
        config: PretrainedConfig,
        max_cache_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        max_batch_size: Optional[int] = None,
    ) -> None:
        super().__init__(
            config=config,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
            max_batch_size=max_batch_size,
        )

    def crop(self, start: int, end: Optional[int] = None, dim: int = 2) -> None:
        """
        Crop past key/values in [start : end] (along dimension 2).
        Negative start is counted from the end.
        Zero out tokens in the specified range.
        """
        if end is None:
            end = self.get_seq_length()
        if start < 0:
            start = end + start
        if end <= start:
            return

        # Group (k, v) pairs by device.
        device_groups = {}
        for k, v in zip(self.key_cache, self.value_cache):
            device_groups.setdefault(k.device, []).append((k, v))
        for dev, kv_list in device_groups.items():
            # For non‑MPS devices, use index_fill_ along dim 2.
            if dev.type != 'mps':
                idx = torch.arange(start, end, device=dev)
                for k, v in kv_list:
                    k.index_fill_(dim, idx, 0)
                    v.index_fill_(dim, idx, 0)
            else:
                # For MPS, use slicing.
                for k, v in kv_list:
                    k[:, :, start:end] = 0
                    v[:, :, start:end] = 0

    def reorder_cache_with_offset(
        self,
        beam_idx: torch.LongTensor,
        new_chunk_len: int = 1,
        offset: int = 0,
        dim: int = 0,
    ) -> None:
        """
        Reorder the slice [offset : offset + new_chunk_len] of each key/value cache
        according to the order specified by beam_idx, then zero out any leftover positions.
        The update is performed in batch for all layers on a device so that the underlying
        tensor objects (their memory pointers) remain unchanged—a requirement for CUDA graphs.
        
        Parameters:
          beam_idx (LongTensor): 1D tensor of indices indicating the new ordering.
          new_chunk_len (int): The new length of the updated slice.
          offset (int): The starting offset along dimension `dim` to update.
          dim (int): The dimension along which the update occurs.
        """
        slice_len = beam_idx.size(0)
        # Group cache indices by device.
        dev_groups = {}
        for i, (k, _) in enumerate(zip(self.key_cache, self.value_cache)):
            dev_groups.setdefault(k.device, []).append(i)
        
        # Process each device group.
        for dev, indices in dev_groups.items():
            # Ensure beam_idx is on the correct device.
            b_idx = beam_idx.to(dev)
            reorder_src = offset + b_idx
            reorder_dest = offset + torch.arange(slice_len, device=dev)
            leftover_range = (offset + torch.arange(slice_len, new_chunk_len, device=dev)
                              if new_chunk_len > slice_len else None)
            # Stack the caches for this device.
            k_cat = torch.stack([self.key_cache[i] for i in indices], dim=0)
            v_cat = torch.stack([self.value_cache[i] for i in indices], dim=0)
            # Batched update along dimension `dim`
            k_cat.index_copy_(dim+1, reorder_dest, k_cat.index_select(dim+1, reorder_src))
            v_cat.index_copy_(dim+1, reorder_dest, v_cat.index_select(dim+1, reorder_src))
            if leftover_range is not None:
                k_cat.index_fill_(dim+1, leftover_range, 0)
                v_cat.index_fill_(dim+1, leftover_range, 0)
            # Scatter the updated results back.
            for j, i in enumerate(indices):
                self.key_cache[i].copy_(k_cat[j], non_blocking=True)
                self.value_cache[i].copy_(v_cat[j], non_blocking=True)