import torch
from transformers.generation.logits_process import LogitsProcessorList

from .subspec_sd import SubSpecSDGeneratorBase as _SubSpecSDGeneratorBase
from ..utils.mixin import SDProfilingMixin
from ..utils.lossy_tree_verify import lossy_bottom_up_verify


class SubSpecLossySDGeneratorBase(_SubSpecSDGeneratorBase):
    """SubSpec SD (v1) + lossy bottom-up verification."""

    def _verify(self, tree, root_ind, logits, logits_processor, do_sample, skip_nodes: int = 0):
        # v1 pipeline does not use skip_nodes, but keep signature for compatibility.
        if do_sample and logits_processor is None:
            logits_processor = LogitsProcessorList()

        probs = self._sample_token(logits, logits_processor, do_sample, return_probs=True)
        probs = probs.squeeze(0).detach().cpu()  # (num_nodes_from_logits, vocab)

        # Defensive alignment: ensure tree metadata matches the number of decoded logits.
        # This mirrors the v2 lossy generator fix.
        num_nodes_from_logits = int(probs.shape[0])
        node_data = tree.get_tree_data(skip_nodes=0)
        token_ids = node_data["token_ids"].cpu()[:num_nodes_from_logits]
        parent_indices = node_data["parent_indices"].cpu()[:num_nodes_from_logits]

        num_nodes = int(token_ids.numel())
        children_lists: list[list[int]] = [[] for _ in range(num_nodes)]
        for v in range(num_nodes):
            for c in tree.nodes[v].children:
                if 0 <= int(c) < num_nodes:
                    children_lists[v].append(int(c))

        sampled_1d, hidden_1d, accept_len = lossy_bottom_up_verify(
            probs=probs,
            token_ids=token_ids,
            parent_indices=parent_indices,
            children_lists=children_lists,
            root_index=int(root_ind),
            eos_token_id=getattr(self.draft_model, "eos_token_id", None),
            do_sample=do_sample,
            threshold=float(getattr(self.draft_params, "lossy_threshold", 0.0)),
            window_size=int(getattr(self.draft_params, "lossy_window_size", 1)),
        )

        sampled_tokens = sampled_1d.unsqueeze(0)  # (1, L)
        total_len = int(sampled_1d.numel())
        return sampled_tokens, hidden_1d, (total_len, int(accept_len))


class SubSpecLossySDGenerator(SDProfilingMixin, SubSpecLossySDGeneratorBase):
    pass
