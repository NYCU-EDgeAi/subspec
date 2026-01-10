import torch
import torch.nn as nn
import nvtx

from ..utils.cpu_tree import Tree
from .base import DraftModelBase, TreeData, TreeMaskCache

    
class ClassicSDDraftModel(DraftModelBase):
    def forward(self, input_ids, with_softmax=False, *model_args, **kwargs):
        input_ids, kwargs = self._align_forward_inputs_to_model_device(input_ids, kwargs)
        logits = self.model(input_ids, *model_args, **kwargs).logits
        if with_softmax:
            logits = torch.softmax(logits/self.draft_params.temperature, dim=-1)
            
        return logits
    
    @torch.no_grad()
    def speculate(self, input_ids, **kwargs):
        # 1) Obtain necessary parameters
        device = input_ids.device
        dtype = self.model.lm_head.weight.dtype
        batch_size, input_len = input_ids.shape
        max_cache_len = getattr(self.past_key_values.cache, "max_cache_len", None)
        assert batch_size == 1, "Only support batch_size=1 for now."
        
        # 2) Initialize kv_len & cache_position
        with nvtx.annotate("kv_init"):
            kv_len = self._get_kv_len_int()
            
        # 3) First forward pass
        with nvtx.annotate("draft_prefill", color="red"):
            cache_position = torch.arange(kv_len, input_len, dtype=torch.long, device=device)
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
            
        with nvtx.annotate("draft_sample", color="green"):
            self.parent_probs = torch.ones((1, 1), device=device, dtype=dtype)
            token_ids, child_probs, parent_indices = self.topk_sampling(
                sampled_probs,
                self.parent_probs,
                self.draft_params.topk_len
            )
            self.parent_probs = child_probs
        
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
        with nvtx.annotate("tree_update", color="green"):
            self.tree_data.update(token_ids, child_probs, parent_indices)
            self.tree_mask_cache.update_tree_mask(parent_indices)
        
        # Set initial state for the speculative tree
        self.token_ids = token_ids
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
        parent_probs = self.parent_probs
        position_ids = self.position_ids
        cache_position = self.cache_position
        
        with nvtx.annotate("draft_forward", color="red"):
            sampled_probs = self(
                token_ids,
                with_softmax=True,
                past_key_values=self.past_key_values.cache,
                position_ids=position_ids,
                attention_mask=tree_attention_mask,
                cache_position=cache_position,
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
        self.cache_position += self.draft_params.topk_len
        
    @torch.no_grad()
    def update_tree(self, tree_data):
        with nvtx.annotate("tree_finalize"):
            with nvtx.annotate("tree_data/get"):
                data = tree_data.get_data()
            with nvtx.annotate("tree/apply"):
                self.tree.add_nodes(*data)
        return self.tree