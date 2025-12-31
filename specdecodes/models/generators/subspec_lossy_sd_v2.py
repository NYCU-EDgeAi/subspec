import torch
from transformers.generation.logits_process import LogitsProcessorList

from .subspec_sd_v2 import SubSpecSDGeneratorBase as _SubSpecSDGeneratorBase
from ..utils.mixin import SDProfilingMixin
from ..utils.lossy_tree_verify import lossy_bottom_up_verify


class SubSpecLossySDGeneratorBase(_SubSpecSDGeneratorBase):
    """SubSpec SD v2 generator with lossy bottom-up verification.

    Notes:
    - Drafting is unchanged.
    - Verification uses lossy bottom-up verification and accounts for postspec extending the tree.
    """
    def _verify(self, tree, root_ind, logits, logits_processor, do_sample, skip_nodes: int = 0):
        # Build target probabilities (post-warp if do_sample=True).
        if do_sample and logits_processor is None:
            logits_processor = LogitsProcessorList()

        probs = self._sample_token(logits, logits_processor, do_sample, return_probs=True)
        probs = probs.squeeze(0).detach().cpu()  # (num_nodes_from_logits, vocab)

        # IMPORTANT: In v2, the draft model can extend the tree via postspec *after*
        # the target forward produced `logits`. So `tree.get_tree_data(skip_nodes=0)`
        # may now include *more* nodes than `logits`/`probs`.
        # We must slice tree metadata down to the exact number of rows in `probs`.
        num_nodes_from_logits = int(probs.shape[0])
        node_data = tree.get_tree_data(skip_nodes=skip_nodes)
        token_ids = node_data["token_ids"].cpu()[:num_nodes_from_logits]
        parent_indices = node_data["parent_indices"].cpu()[:num_nodes_from_logits]

        # Map original tree indices -> local indices inside this chunk.
        root_local = int(root_ind - skip_nodes)
        if root_local < 0 or root_local >= token_ids.numel():
            # Fallback to local root 0 if mapping is invalid.
            root_local = 0

        # Build children lists in local indexing.
        num_nodes_local = token_ids.numel()
        children_lists: list[list[int]] = [[] for _ in range(num_nodes_local)]

        # Children relationships are stored on the full tree in original indexing.
        # We rebuild the adjacency restricted to [skip_nodes, ...] range.
        for v_local in range(num_nodes_local):
            v_orig = v_local + skip_nodes
            for c_orig in tree.nodes[v_orig].children:
                c_local = c_orig - skip_nodes
                if 0 <= c_local < num_nodes_local:
                    children_lists[v_local].append(int(c_local))

        sampled_1d, hidden_1d, accept_len = lossy_bottom_up_verify(
            probs=probs,
            token_ids=token_ids,
            parent_indices=parent_indices - int(skip_nodes),
            children_lists=children_lists,
            root_index=root_local,
            eos_token_id=getattr(self.draft_model, "eos_token_id", None),
            do_sample=do_sample,
            threshold=float(getattr(self.draft_params, "lossy_threshold", 0.0)),
            window_size=int(getattr(self.draft_params, "lossy_window_size", 1)),
        )

        # Convert local hidden indices back to original indexing.
        hidden_indices = hidden_1d + int(skip_nodes)

        sampled_tokens = sampled_1d.unsqueeze(0)  # (1, L)
        total_len = int(sampled_1d.numel())
        return sampled_tokens, hidden_indices, (total_len, int(accept_len))


class SubSpecLossySDGenerator(SDProfilingMixin, SubSpecLossySDGeneratorBase):
    pass
