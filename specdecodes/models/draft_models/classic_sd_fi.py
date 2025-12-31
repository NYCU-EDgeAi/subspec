import torch
import nvtx

from ..utils.cpu_tree import Tree
from .base import DraftModelBase, TreeData, TreeMaskCache
from copy import deepcopy

from ..utils.flashinfer.cache_manager import (
    KvCachePool,
    RequestKvCache,
    KvCacheBatchPosition,
    getKvCacheBatchPosition,
    FlashInferCache
)
from ..utils.flashinfer.attention_wrapper import FlashinferAttentionWrapper
import numpy as np
import pathlib 



class ClassicSDDraftModel(DraftModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.had_first_speculate = False
        self.postspec_count = 0

    def forward(self, input_ids, with_softmax=False, *model_args, **kwargs):
        logits = self.model(input_ids, *model_args, **kwargs).logits
        if with_softmax:
            logits = torch.softmax(logits/self.draft_params.temperature, dim=-1)
            
        return logits
    
    def init_cuda_graph_runner(
        self,
        device: torch.device,
    ):
        """
        Allocate *fixed‑size* staging buffers for a single‑batch, `tree`
        forward of length `topk_len` and capture it inside a CUDA Graph.

        Call once (e.g. the very first time `speculate` is invoked).
        """
        print("draft model has init_cuda_graph_runner. Initializing CUDA Graph runner...")
        if hasattr(self, "graph"):          # already captured
            return

        self.decode_chunk_size = self.draft_params.topk_len   # <- used by `tree_step`
        self.graph              = None
        self.output_buffer      = None
        self.model.eval()   
        kvCachePool = self.kvCachePool

        # ── staging buffers ───────────────────────────────────────
        B = 1
        L = self.decode_chunk_size
        PAGE_LEN = kvCachePool.page_len

        self.input_ids_buf    = torch.zeros((B, L),  dtype=torch.long,  device=device)
        self.position_ids_buf = torch.zeros((B, L),  dtype=torch.long,  device=device)

        self.seq_indptr_buf        = torch.zeros((B + 1,),          dtype=torch.int32, device=device)
        self.kv_page_indptr_buf    = torch.zeros((B + 1,),          dtype=torch.int32, device=device)
        self.kv_page_indices_buf   = torch.zeros((kvCachePool.max_pages,), dtype=torch.int32, device=device)
        self.kv_last_page_len_buf  = torch.zeros((B,),              dtype=torch.int32, device=device)
        self.batch_indices_buf     = torch.zeros((L,),              dtype=torch.int32, device=device)
        self.positions_buf         = torch.zeros((L,),              dtype=torch.int32, device=device)

        self.batch_position = KvCacheBatchPosition(
            seq_indptr       = self.seq_indptr_buf,
            kv_page_indptr   = self.kv_page_indptr_buf,
            kv_page_indices  = self.kv_page_indices_buf,
            kv_last_page_len = self.kv_last_page_len_buf,
            batch_indices    = self.batch_indices_buf,
            positions        = self.positions_buf,
        )

        # ── Flash‑Infer wrapper (shared weights, no extra allocation) ──
        if not hasattr(self, "flashinferWrapper"):
            raise ValueError("flashinferWrapper not found in draft model.")
        
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(stream):
            dummy_tok = torch.zeros((B, L), dtype=torch.long, device=device)
            dummy_pos = torch.zeros_like(dummy_tok)

            # two warm‑ups (outside the graph)
            for _ in range(2):
                _ = self(
                    dummy_tok,
                    with_softmax=True,
                    position_ids=dummy_pos,
                    kvCachePool=kvCachePool,
                    batch_position=self.batch_position,
                    mode="tree",
                    flashinferWrapper=self.flashinferWrapper,
                )

            torch.cuda.current_stream().wait_stream(stream)
            cg = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cg, stream=stream):
                self.output_buffer = self(
                    self.input_ids_buf,
                    with_softmax=True,
                    position_ids=self.position_ids_buf,
                    kvCachePool=kvCachePool,
                    batch_position=self.batch_position,
                    mode="tree",
                    flashinferWrapper=self.flashinferWrapper,
                )

        self.graph = cg
        print("finished capturing draft model CUDA graph")


    def tree_step(
        self,
        token_ids: torch.Tensor,           # [1, L]  – same L as topk_len
        position_ids: torch.Tensor,        # [1, L]
        batch_position: KvCacheBatchPosition,
    ):
        """
        Copy fresh data into the staging buffers and *replay* the graph.
        Returns the model outputs captured in `self.output_buffer`.
        """
        # basic shape checks
        B, L = token_ids.shape
        if B != 1 or L > self.decode_chunk_size:
            raise ValueError("tree_step expects shape [1, L<=topk_len]")

        # ---- buffer updates ------------------------------------------------
        self.input_ids_buf[:, :L].copy_(token_ids)
        self.position_ids_buf[:, :L].copy_(position_ids)

        self.seq_indptr_buf.copy_(batch_position.seq_indptr)
        self.kv_page_indptr_buf.copy_(batch_position.kv_page_indptr)
        self.kv_last_page_len_buf.copy_(batch_position.kv_last_page_len)
        self.batch_indices_buf.copy_(batch_position.batch_indices)
        self.positions_buf.copy_(batch_position.positions)

        n_pages = batch_position.kv_page_indptr[1].item()
        self.kv_page_indices_buf[:n_pages].copy_(batch_position.kv_page_indices[:n_pages])

        # ---- replay --------------------------------------------------------
        self.graph.replay()
        return self.output_buffer
    
    @torch.no_grad()
    def update_tree(self, tree_data):
        with nvtx.annotate("tree_finalize"):
            with nvtx.annotate("tree_data/get"):
                data = tree_data.get_data()
            with nvtx.annotate("tree/apply"):
                self.tree.add_nodes(*data)
        return self.tree
    
    @torch.no_grad()
    def speculate(self, input_ids, request_kv_cache, **kwargs):

        self.had_first_speculate = True

        # 1) Obtain necessary parameters
        device = input_ids.device
        dtype = self.model.lm_head.weight.dtype
        batch_size, input_len = input_ids.shape
        self.request_kv_cache = request_kv_cache
        
        max_cache_len = None
        # self.flashinferWrapper = kwargs["flashinferWrapper"]
        if not hasattr(self, 'flashinferWrapper'):
            self.flashinferWrapper = FlashinferAttentionWrapper(
                self.model.config.num_attention_heads, self.model.config.num_key_value_heads, self.model.config.hidden_size,request_kv_cache.kvCachePool.page_len
            )
        self.kvCachePool = request_kv_cache.kvCachePool

        assert (self.flashinferWrapper is not None)
        assert batch_size == 1, "Only support batch_size=1 for now."

        
        # 2) Initialize kv_len & cache_position
        with nvtx.annotate("kv_init"):
            kv_len = request_kv_cache.get_seq_length()
            # convert kv_len to int if it is a tensor
            if isinstance(kv_len, torch.Tensor):
                kv_len = kv_len.item()
            org_kv_len = kv_len

        # 3) First forward pass
        with nvtx.annotate("draft_prefill", color="red"):
            
            mode = "decode"
            # request_kv_cache.increment() 
            request_kv_cache.increment(input_len)

            # batch_position = getKvCacheBatchPosition(
            #     [request_kv_cache],
            #     mode=mode,
            #     device=device,
            #     treeTokens=None if mode != "tree" else input_len,
            # )
            batch_position = getKvCacheBatchPosition(
                request_kv_caches=[request_kv_cache],
                mode='tree', 
                device=input_ids.device,
                treeTokens=input_len,
            )
            # self.flashinferWrapper.prepareAttention(
            #     mode,
            #     batch_position,
            #     request_kv_cache.kvCachePool.page_len,
            #     "NONE", #POS_ENCODING_MODE.NONE
            #     request_kv_cache.kvCachePool.cache_data[0].dtype,
            # )  
            self.flashinferWrapper.prepareAttention(
                'prefill',
                batch_position,
                self.kvCachePool.page_len,
                "NONE", # POS_ENCODING_MODE.NONE,
                self.kvCachePool.cache_data[0].dtype,
            )
            # position_ids = torch.full((batch_size, input_len), kv_len, device=device, dtype=torch.long)
            position_ids = torch.arange(kv_len, kv_len + input_len, dtype=torch.long, device=device).unsqueeze(0)
            sampled_probs = self(
                input_ids,
                with_softmax=True,
                # use_cache=False,
                logits_to_keep=1,
                position_ids = position_ids,
                kvCachePool=request_kv_cache.kvCachePool,
                batch_position=batch_position,
                mode='prefill',
                flashinferWrapper=self.flashinferWrapper,
            )
           
            kv_len += input_len
            
        org_kv_len = kv_len
        with nvtx.annotate("draft_init_state"):
            parent_probs = torch.ones((1, 1), device=device, dtype=dtype)
            position_ids = torch.full((batch_size, self.draft_params.topk_len), kv_len, device=device, dtype=torch.long)
        
        # 4) Create TreeData & TreeMaskCache to manage tree structure and intermediate data.
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

        # 5) Main loop
        for depth_i in range(self.draft_params.max_depth):
            with nvtx.annotate("draft_sample", color="green"):
                token_ids, child_probs, parent_indices = self.topk_sampling(
                    sampled_probs,
                    parent_probs,
                    self.draft_params.topk_len
                )
                
                parent_probs = child_probs
                
            with nvtx.annotate("tree_data/update", color="green"):
                self.tree_data.update(token_ids, child_probs, parent_indices)
                
            with nvtx.annotate("tree_mask/update"):
                tree_attention_mask = self.tree_mask_cache.update_tree_mask(parent_indices,return_invert=False)
                
            with nvtx.annotate("draft_forward", color="red"):
                num_tokens = self.draft_params.topk_len
                request_kv_cache.increment(num_tokens)

                batch_position = getKvCacheBatchPosition(
                    request_kv_caches=[request_kv_cache],
                    mode='tree',
                    device=input_ids.device,
                    treeTokens=num_tokens,
                )
                
                self.flashinferWrapper.prepareAttention(
                    'tree',
                    batch_position,
                    request_kv_cache.kvCachePool.page_len,
                    "NONE", #POS_ENCODING_MODE.NONE
                    request_kv_cache.kvCachePool.cache_data[0].dtype,
                    attention_mask=tree_attention_mask,
                )
                if hasattr(self, "graph"):
                    # use CUDA graph
                    sampled_probs = self.tree_step(
                        token_ids,
                        position_ids,
                        batch_position=batch_position,
                    )
                else:
                    sampled_probs = self(
                        token_ids,
                        with_softmax=True,
                        past_key_values=None,
                        position_ids=position_ids,
                        kvCachePool=request_kv_cache.kvCachePool,
                        batch_position=batch_position,
                        mode='tree',
                        flashinferWrapper=self.flashinferWrapper,
                    )
                kv_len += self.draft_params.topk_len
                
            with nvtx.annotate("state_update"):
                position_ids += 1

        request_kv_cache.decrement(kv_len - org_kv_len)
        self.update_tree(self.tree_data)
        self.token_ids = token_ids
        self.position_ids = position_ids
        self.parent_probs = parent_probs

        return self.tree
    
    def init_postspec(self):
        self.tree_data = TreeData()
        self.postspec_count = 0
        
    @torch.no_grad()
    def postspec(self):
        if not self.had_first_speculate:
            return
        if self.postspec_count > (self.post_draft_params.max_depth - 1):
            return
        with nvtx.annotate("postspec_step", color="blue"):
            self.speculate_once()
        self.postspec_count += 1

    @torch.no_grad()
    def speculate_once(self, **kwargs):
        tree_attention_mask = self.tree_mask_cache.get_tree_mask()
        token_ids = self.token_ids
        parent_probs = self.parent_probs
        position_ids = self.position_ids

        request_kv_cache = self.request_kv_cache
        
        with nvtx.annotate("draft_forward", color="red"):
            num_tokens = self.draft_params.topk_len
            
            request_kv_cache.increment(num_tokens)

            batch_position = getKvCacheBatchPosition(
                request_kv_caches=[request_kv_cache],
                mode='tree',
                device=token_ids.device,                    
                treeTokens=num_tokens,
            )
            self.flashinferWrapper.prepareAttention(
                'tree',
                batch_position,
                request_kv_cache.kvCachePool.page_len,
                "NONE", #POS_ENCODING_MODE.NONE
                request_kv_cache.kvCachePool.cache_data[0].dtype,
                attention_mask=tree_attention_mask,
            )

            if hasattr(self, "graph"):
                # use CUDA graph
                sampled_probs = self.tree_step(
                    token_ids,
                    position_ids,
                    batch_position=batch_position,  
                )
            else:
                sampled_probs = self(
                    token_ids,
                    with_softmax=True,
                    past_key_values=None,
                    position_ids=position_ids,
                    kvCachePool=request_kv_cache.kvCachePool,
                    batch_position=batch_position,
                    mode='tree',
                    flashinferWrapper=self.flashinferWrapper,
                )

        with nvtx.annotate("draft_sample", color="green"):
            token_ids, child_probs, parent_indices = self.topk_sampling(
                sampled_probs,
                parent_probs,
                self.draft_params.topk_len
            )
            parent_probs = child_probs
            
        with nvtx.annotate("tree_update", color="green"):
            self.tree_data.update(token_ids, child_probs, parent_indices)
            self.tree_mask_cache.update_tree_mask(parent_indices)
            
        # Update internal state
        self.token_ids = token_ids
        self.parent_probs = parent_probs
        self.position_ids += 1


    def update_tree_after_post(self):
        """Return the finalized draft tree after post-speculation."""
        # Update the tree data and mask cache before returning
        self.update_tree(self.tree_data)
        return self.tree