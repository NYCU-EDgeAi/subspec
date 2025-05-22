from typing import Set, List
import math
import torch
import nvtx, os


# PAGE_LEN: int = 64

@torch.compile(mode="reduce-overhead")  # PyTorch 2.x: capture and optimize this function
def reorder_and_copy_back(
    cache_data: torch.Tensor,
    beam_idx: torch.Tensor,
    offset: int,
    num_new_tokens: int,
    page_len: int,
):
    """
    This function:
      1) Stacks all layers from `cache_data` into a single Tensor.
      2) Computes old/new flat indices based on `beam_idx + offset`.
      3) Reorders (K,V) in-place using `index_select` + `index_copy_`.
      4) Un-stacks and copies back to the original Tensors in `cache_data`.
         (Thus preserving their underlying storage addresses.)

    cache_data: list of Tensors each of shape (max_pages, 2, page_len, num_heads, head_dim)
    beam_idx: indices specifying the reordering.
    offset, num_new_tokens: standard offset logic.
    page_len: token capacity per page.
    """

    device = beam_idx.device
    beam_idx = beam_idx.to(device)
    beam_size = beam_idx.shape[0]

    # -- Flatten offset-based indices --
    old_indices = beam_idx + offset
    new_indices = torch.arange(offset, offset + beam_size, device=device)

    def to_flat_idx(idx: torch.Tensor):
        page_idx = idx // page_len
        token_idx = idx % page_len
        return page_idx * page_len + token_idx

    old_flat = to_flat_idx(old_indices)
    new_flat = to_flat_idx(new_indices)

    
    cache_stacked = cache_data
    L, max_pages, _, pl, num_heads, head_dim = cache_stacked.shape
    
    total_tokens = offset + num_new_tokens
    total_pages = (total_tokens + page_len - 1) // page_len
    if total_pages > max_pages:
        raise ValueError(
            f"Cache has only {max_pages} pages but needs {total_pages} for {total_tokens} tokens."
        )

    # -- Separate K / V and flatten along "page_len" dimension --
    # k_cat, v_cat => (L, max_pages*page_len, num_heads, head_dim)
    k_cat = cache_stacked[:, :, 0, :, :, :].reshape(L, max_pages * page_len, num_heads, head_dim)
    v_cat = cache_stacked[:, :, 1, :, :, :].reshape(L, max_pages * page_len, num_heads, head_dim)

    # -- Gather-then-scatter reorder --
    k_gathered = k_cat.index_select(1, old_flat)
    v_gathered = v_cat.index_select(1, old_flat)
    k_cat.index_copy_(1, new_flat, k_gathered)
    v_cat.index_copy_(1, new_flat, v_gathered)

    # -- Un-flatten => (L, max_pages, page_len, num_heads, head_dim) --
    k_cat = k_cat.view(L, max_pages, page_len, num_heads, head_dim)
    v_cat = v_cat.view(L, max_pages, page_len, num_heads, head_dim)

    # -- Copy back to preserve memory addresses --
    # We'll do a simple Python loop. As of PyTorch 2.x, this can still be captured
    # if L is consistent across calls. The loop is unrolled/flattened in the graph.
    for i in range(L):
        cache_data[i][:, 0, :, :, :].copy_(k_cat[i], non_blocking=True)
        cache_data[i][:, 1, :, :, :].copy_(v_cat[i], non_blocking=True)

class KvCacheBatchPosition:
    def __init__(
        self,
        seq_indptr: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        seq_lens: torch.Tensor,
        total_seq_len: int,
    ):
        self.total_seq_len = total_seq_len
        self.seq_indptr = seq_indptr
        self.kv_page_indptr = kv_page_indptr
        self.kv_page_indices = kv_page_indices
        self.seq_lens = seq_lens
        self.kv_last_page_len = kv_last_page_len
        
    def print_info(self):
        print(f"  q_indptr:       {self.seq_indptr}")
        print(f"  kv_page_indptr:   {self.kv_page_indptr}")
        print(f"  kv_page_indices:  {self.kv_page_indices}")
        print(f"  kv_last_page_len: {self.kv_last_page_len}")
        print(f"  seq_lens:         {self.seq_lens}")

class KvCachePool:
    def __init__(
        self,
        max_pages: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        # max_pages = 100
        
        self.cache_data =  torch.zeros(
                num_layers, max_pages, 2, page_len, num_heads, head_dim, dtype=dtype, device=device
            )
            # for _ in range(num_layers)
        self.num_layers = num_layers
        self.device = device
        self.max_pages = max_pages
        self.page_len = page_len
        self.free_page_mask = torch.ones(max_pages, dtype=torch.bool, device="cpu")
        self.num_heads = num_heads
        self.head_dims = head_dim
        self.dtype = dtype
        
    def reset(self):
        self.cache_data =  torch.zeros(
                self.num_layers, self.max_pages, 2, self.page_len, self.num_heads, self.head_dims, dtype=self.dtype, device=self.device
            )

    def count(self):
        total_nonzero = sum(tensor.count_nonzero().item() for tensor in self.cache_data)
        print(f"Total number of changed values: {total_nonzero/2/self.num_heads/self.head_dims/32}")

    def print_cache_sums(self):
        key_cache_sum = sum([torch.sum(layer[:, 0, :, :, :]) for layer in self.cache_data])
        value_cache_sum = sum([torch.sum(layer[:, 1, :, :, :]) for layer in self.cache_data])
        print(f"Sum of key_cache: {key_cache_sum}, Sum of value_cache: {value_cache_sum}")

    def num_free_pages(self):
        return self.free_page_mask.sum()

    def allocate(self, num_pages: int):
        free_page_indices = self.free_page_mask.nonzero()
        assert (
            len(free_page_indices) >= num_pages
        ), f"Out of available cache pages: asked {num_pages}, only {len(free_page_indices)} free pages"

        allocated_indices = free_page_indices[:num_pages]
        self.free_page_mask[allocated_indices] = False
        return allocated_indices.squeeze(1).tolist()

    def deallocate(self, kv_page_indices: List[int]):
        self.free_page_mask[kv_page_indices] = True
    
    
    def reorder_cache_with_offset(self, beam_idx: torch.LongTensor, offset=0, num_new_tokens=0):
        """
        Reorders the cache for speculative decoding, given the selected beam indices,
        while [:offset] remain unchanged. After reordering, sets the rest of the new tokens to zero.
        The cache layout is assumed to be (max_pages, 2, page_len, num_heads, head_dim)
        for each layer, and we want to preserve the same memory footprint.
        """
        with nvtx.annotate("to device", color="green"):
            device = beam_idx.device
            beam_idx = beam_idx.to(device)
            beam_size = beam_idx.size(0)

        # Convert old positions (beam_idx) to new positions:
        old_indices = beam_idx + offset  # [beam_size]
        new_indices = torch.arange(offset, offset + beam_size, device=device, dtype=torch.long)  # [beam_size]

        # Flatten the "page + token" dimension into a single index
        page_len = self.page_len
        
        def to_flat_idx(idx: torch.Tensor):
            """
            Given a tensor of positions in [0, total_tokens),
            map them to (page_idx, token_idx), then flatten as: page_idx * page_len + token_idx.
            """
            page_indices = idx // page_len
            token_indices = idx % page_len
            return page_indices, token_indices

        with nvtx.annotate("compute idx", color="blue"):
            old_page_indices, old_token_indices = to_flat_idx(old_indices)
            new_page_indices, new_token_indices = to_flat_idx(new_indices)

            old_flat = old_page_indices * page_len + old_token_indices  # [beam_size]
            new_flat = new_page_indices * page_len + new_token_indices  # [beam_size]

            total_tokens = offset + num_new_tokens
            total_pages = (total_tokens + page_len - 1) // page_len  # ceiling division
            max_flat_len = total_pages * page_len

        with nvtx.annotate("stack cache", color="red"):
            # Stack all layers into one big tensor:
            #   self.cache_data is a list of Tensors of shape (max_pages, 2, page_len, num_heads, head_dim)
            #   After stacking on dim=0 => shape (L, max_pages, 2, page_len, num_heads, head_dim)
            # cache_stacked = torch.stack(self.cache_data, dim=0)
            cache_stacked = self.cache_data
            L, max_pages, _, page_len_, num_heads, head_dim = cache_stacked.shape
            if page_len_ != page_len:
                raise ValueError(
                    f"Expected page_len={page_len}, found {page_len_} in cached data."
                )
            if total_pages > max_pages:
                raise ValueError(
                    f"Cache does not have enough pages ({max_pages}) for total tokens ({total_tokens})."
                )

        with nvtx.annotate("split k/v", color="green"):
            # Separate keys and values: (L, max_pages, page_len, num_heads, head_dim)
            k_cat = cache_stacked[:, :, 0, :, :, :].clone()
            v_cat = cache_stacked[:, :, 1, :, :, :].clone()

        with nvtx.annotate("flatten", color="blue"):
            # Flatten the (max_pages, page_len) => single "tokens" dimension for simpler index_copy
            k_cat = k_cat.view(L, max_pages * page_len, num_heads, head_dim)
            v_cat = v_cat.view(L, max_pages * page_len, num_heads, head_dim)

        with nvtx.annotate("reorder", color="yellow"):
            # Reorder keys (K)
            k_cat.index_copy_(
                1,                 # dimension = 1
                new_flat,          # where to copy
                k_cat.index_select(1, old_flat)  # what to copy
            )
            # Reorder values (V)
            v_cat.index_copy_(
                1,
                new_flat,
                v_cat.index_select(1, old_flat)
            )

        with nvtx.annotate("unflatten", color="green"):
        # (L, max_pages * page_len, num_heads, head_dim) => (L, max_pages, page_len, num_heads, head_dim)
            k_cat = k_cat.view(L, max_pages, page_len, num_heads, head_dim)
            v_cat = v_cat.view(L, max_pages, page_len, num_heads, head_dim)

        with nvtx.annotate("assign", color="purple"):
            # Place each layer's K and V back into original shape:
            # (max_pages, 2, page_len, num_heads, head_dim)
            # for layer_idx in range(L):
            #     self.cache_data[layer_idx][:, 0, :, :, :].copy_(k_cat[layer_idx], non_blocking=True)
            #     self.cache_data[layer_idx][:, 1, :, :, :].copy_(v_cat[layer_idx], non_blocking=True)
            self.cache_data[:, :, 0, :, :, :].copy_(k_cat,non_blocking=True)
            self.cache_data[:, :, 1, :, :, :].copy_(v_cat,non_blocking=True)
            
    # def reorder_cache_with_offset(self, beam_idx: torch.LongTensor, offset=0, num_new_tokens=0):
    #     """
    #     Public method that just delegates to the compiled reorder function.
    #     """
    #     with nvtx.annotate("Reorder + copy back (compiled)", color="green"):
    #         reorder_and_copy_back(
    #             cache_data=self.cache_data,
    #             beam_idx=beam_idx,
    #             offset=offset,
    #             num_new_tokens=num_new_tokens,
    #             page_len=self.page_len,
    #         )

class RequestKvCache:
    def __init__(self, kvCachePool: KvCachePool, page_len: int, seq_init_len: int):
        self.kvCachePool = kvCachePool
        self.page_len = page_len
        init_num_pages = math.ceil(seq_init_len / self.page_len)
        self.kv_last_page_len = seq_init_len - (init_num_pages - 1) * self.page_len
        self.kv_page_indices = kvCachePool.allocate(init_num_pages)
        self.kv_len = seq_init_len
        self.is_released = False

    def increment(self):
        self.kv_len += 1
        self.kv_last_page_len += 1
        if self.kv_last_page_len > self.page_len:
            self.kv_last_page_len -= self.page_len
            new_indices = self.kvCachePool.allocate(1)
            self.kv_page_indices.extend(new_indices)
            


    def release(self):
        self.kvCachePool.deallocate(self.kv_page_indices)
        self.is_released = True
    
    def reorder_cache_with_offset(self, beam_idx: torch.LongTensor, offset=0, num_new_tokens=0):
        """
        Reorders the cache for beam search, given the selected beam indices, while [:offset] remain unchanged.
        beam_idx: LongTensor of shape (batch_size * num_beams,)
        """
        if offset != 0:
            offset -=1
            
        self.kvCachePool.reorder_cache_with_offset(beam_idx,offset,num_new_tokens)
       
       
        # update  self.kv_last_page_len self.kv_page_indices self.kv_len
        self.kv_len = offset + beam_idx.size(0) 

        if self.kv_len == 0:
            self.kv_last_page_len = 0
        else:
            self.kv_last_page_len = (self.kv_len - 1) % self.page_len + 1

        num_pages_needed = (self.kv_len + self.page_len - 1) // self.page_len  # Ceiling division
        # Deallocate any extra pages that are no longer needed
        current_num_pages = len(self.kv_page_indices)
        if current_num_pages > num_pages_needed:
            # Identify extra pages to deallocate
            extra_pages = self.kv_page_indices[num_pages_needed:]
            # Deallocate the extra pages
            self.kvCachePool.deallocate(extra_pages)
            # Update kv_page_indices to keep only the needed pages
            self.kv_page_indices = self.kv_page_indices[:num_pages_needed]
            
        elif current_num_pages < num_pages_needed:
            # Should not happen in speculative decoding, but handle just in case
            # Allocate additional pages
            additional_pages_needed = num_pages_needed - current_num_pages
            new_indices = self.kvCachePool.allocate(additional_pages_needed)
            self.kv_page_indices.extend(new_indices)
            raise ValueError("need to allocate new pages in reorder cache, should not happen")
   
def getKvCacheBatchPosition(
    request_kv_caches: List[RequestKvCache], mode: str, device: torch.device, treeTokens :int = 0,
) -> KvCacheBatchPosition:
    kv_page_indices_list = []
    kv_page_indptr_list = []
    seq_indptr_list = []
    kv_last_page_len_list = []
    seq_lens_list = []
    cum_pages = 0
    cum_seq_len = 0
    for request_kv_cache in request_kv_caches:
        kv_page_indices_list.extend(request_kv_cache.kv_page_indices)
        kv_page_indptr_list.append(cum_pages)
        seq_indptr_list.append(cum_seq_len)
        kv_last_page_len_list.append(request_kv_cache.kv_last_page_len)
        seq_lens_list.append(request_kv_cache.kv_len)
        cum_pages += len(request_kv_cache.kv_page_indices)

        if mode == 'prefill':
            cum_seq_len += request_kv_cache.kv_len
        elif mode == 'decode' :
            cum_seq_len += 1
        elif mode == 'tree':
            cum_seq_len += treeTokens
        else :
            raise ValueError('invalid mode')
        
    kv_page_indptr_list.append(cum_pages)
    seq_indptr_list.append(cum_seq_len)
    kv_page_indices = torch.tensor(
        kv_page_indices_list, dtype=torch.int32, device=device
    )
    kv_page_indptr = torch.tensor(kv_page_indptr_list, dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor(
        kv_last_page_len_list, dtype=torch.int32, device=device
    )
    seq_indptr = torch.tensor(seq_indptr_list, dtype=torch.int32, device=device)
    seq_lens = torch.tensor(
        seq_lens_list,
        dtype=torch.int32,
        device=device,
    )
    return KvCacheBatchPosition(
        seq_indptr=seq_indptr,
        kv_page_indptr=kv_page_indptr,
        kv_page_indices=kv_page_indices,
        kv_last_page_len=kv_last_page_len,
        seq_lens=seq_lens,
        total_seq_len=cum_seq_len,
    )

# total_seq_len : numbers of canditate tokens 
# seq_lens : kv lens

# KvCacheBatchPosition(
#             seq_indptr=seq_indptr,
#             kv_page_indptr=kv_page_indptr,
#             kv_page_indices=kv_page_indices,
#             kv_last_page_len=kv_last_page_len,
#             seq_lens=seq_lens,
#             total_seq_len=cum_seq_len,
#         )

class FlashInferCache():
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self,config ,max_tokens:int = None,PAGE_LEN = 16) -> None:
        
        currentDevice = torch.device(f'cuda:{torch.cuda.current_device()}')
        # PAGE_LEN: int = 64
        dtype_size = torch.tensor([], dtype=torch.float16).element_size()
        self.config = config
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        MEMORY_FRACTION = float(os.getenv("CUDA_MEMORY_FRACTION", "1.0"))
        
        cache_page_size = (
                    2   * PAGE_LEN
                        * config.num_hidden_layers
                        * config.num_attention_heads
                        * head_dim
                        * dtype_size
        )

        total_free_memory, _ = torch.cuda.mem_get_info(currentDevice)
        total_gpu_memory = torch.cuda.get_device_properties(currentDevice).total_memory
        free_memory = max(0, total_free_memory - (1 - MEMORY_FRACTION) * total_gpu_memory)    
        num_pages_to_allocate = int(free_memory * 0.50 / cache_page_size)
        
        if max_tokens is not None and num_pages_to_allocate * PAGE_LEN > max_tokens:
            num_pages_to_allocate = max_tokens // PAGE_LEN + 1
            print(f"Reducing cache size to {num_pages_to_allocate * PAGE_LEN} tokens")
        
        self.kvCachePool = KvCachePool(
                max_pages = num_pages_to_allocate,
                num_layers = config.num_hidden_layers,
                num_heads = config.num_attention_heads,
                head_dim = head_dim,
                page_len=PAGE_LEN,
                dtype=torch.float16,
                device=currentDevice,
        )
    def reset(self):
        self.kvCachePool.reset()