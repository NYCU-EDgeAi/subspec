import torch
from typing import Optional, Tuple

from .lossy_tree_verify import lossy_bottom_up_verify


@torch.no_grad()
def verify_tree(
    *,
    tree,
    root_ind: int,
    logits: torch.Tensor,
    sample_token_fn,
    verify_step_fn,
    eos_token_id: Optional[int],
    logits_processor,
    do_sample: bool,
    skip_nodes: int = 0,
    lossy: bool = False,
    lossy_threshold: float = 0.0,
    lossy_window_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """Verify a speculative draft tree against target logits.

    This is a shared abstraction used by tree-based SD methods (classic/subspec/eagle/etc).

    Args:
      tree: Draft tree object (CPU tree in this repo).
      root_ind: Root node index in the *original tree indexing*.
      logits: Target model logits for the decoded tree slice.
      sample_token_fn: Callable like GeneratorBase._sample_token.
      verify_step_fn: Callable like ClassicSDGeneratorBase._verify_step.
      eos_token_id: EOS token id.
      logits_processor: HF LogitsProcessorList (or None if do_sample=False).
      do_sample: Whether to sample target token.
      skip_nodes: Number of leading nodes skipped for this verify call (SubSpec v2 post-spec).
      lossy: If True, use lossy bottom-up verification.
      lossy_threshold/window_size: Parameters for lossy verification.

    Returns:
      sampled_tokens: (1, L)
      hidden_indices: (L,) (indices into the original tree indexing)
      (total_len, accept_len): metrics (accept_len excludes bonus token)
    """

    if not lossy:
        # ---- Exact verifier (existing behavior) ----
        global_p = sample_token_fn(logits, logits_processor, do_sample, return_probs=True)
        global_p = global_p.squeeze(0).cpu()

        sampled_tokens = torch.empty(0, dtype=torch.long, device="cpu")
        hidden_indices = torch.empty(0, dtype=torch.long, device="cpu")
        total_len = 0
        accept_len = 0

        node_data = tree.get_tree_data(skip_nodes=skip_nodes)
        token_ids = node_data["token_ids"]

        cur_ind = torch.tensor([int(root_ind)], dtype=torch.long, device="cpu")
        children_inds = tree.get_children_indices(cur_ind)
        children_token_ids = token_ids[children_inds - int(skip_nodes)]

        bonus_token = None

        # Preserve existing synchronization behavior.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        while children_inds.numel() > 0:
            total_len += 1
            dist = global_p[cur_ind - int(skip_nodes)].squeeze(0)
            accept_token_id, bonus_token = verify_step_fn(dist, children_token_ids, logits_processor, do_sample)

            if accept_token_id is not None:
                accept_len += 1
                sampled_tokens = torch.cat([sampled_tokens, accept_token_id[None]])
                hidden_indices = torch.cat([hidden_indices, cur_ind])

                if eos_token_id is not None and int(accept_token_id) == int(eos_token_id):
                    break

                cur_ind = children_inds[children_token_ids == accept_token_id]
                children_inds = tree.get_children_indices(cur_ind)
                children_token_ids = token_ids[children_inds - int(skip_nodes)]

                if (children_inds - int(skip_nodes)).numel() == 0:
                    break
                if int(torch.min(children_inds - int(skip_nodes)).item()) >= int(global_p.shape[0]):
                    break
            else:
                break

        # Bonus token, unless EOS already emitted.
        if sampled_tokens.numel() == 0 or (eos_token_id is None) or (int(sampled_tokens[-1].item()) != int(eos_token_id)):
            if bonus_token is None:
                dist = global_p[cur_ind - int(skip_nodes)].squeeze(0)
                bonus_token = dist.multinomial(num_samples=1).squeeze(-1) if do_sample else dist.argmax()

            if bonus_token is not None:
                sampled_tokens = torch.cat([sampled_tokens, bonus_token.unsqueeze(0)])
                hidden_indices = torch.cat([hidden_indices, cur_ind])

        return sampled_tokens.unsqueeze(0), hidden_indices, (int(total_len), int(accept_len))

    # ---- Lossy bottom-up verifier ----
    if do_sample and logits_processor is None:
        from transformers.generation.logits_process import LogitsProcessorList

        logits_processor = LogitsProcessorList()

    probs = sample_token_fn(logits, logits_processor, do_sample, return_probs=True)
    probs = probs.squeeze(0).detach().cpu()  # (num_nodes_from_logits, vocab)

    num_nodes_from_logits = int(probs.shape[0])
    node_data = tree.get_tree_data(skip_nodes=skip_nodes)
    token_ids = node_data["token_ids"].cpu()[:num_nodes_from_logits]
    parent_indices = node_data["parent_indices"].cpu()[:num_nodes_from_logits]

    num_nodes_local = int(token_ids.numel())
    if num_nodes_local == 0:
        # Degenerate case: no nodes decoded; emit a single bonus token from root context if possible.
        # This should be rare; keep behavior conservative.
        global_p = probs
        bonus = global_p[0].multinomial(1).squeeze(-1) if do_sample else global_p[0].argmax()
        return bonus.view(1, 1), torch.tensor([int(root_ind)], dtype=torch.long), (1, 0)

    root_local = int(root_ind) - int(skip_nodes)
    if root_local < 0 or root_local >= num_nodes_local:
        root_local = 0

    children_lists: list[list[int]] = [[] for _ in range(num_nodes_local)]
    base = int(skip_nodes)
    end = base + num_nodes_local
    for v_orig in range(base, end):
        v_local = v_orig - base
        for c_orig in tree.nodes[v_orig].children:
            c_local = int(c_orig) - base
            if 0 <= c_local < num_nodes_local:
                children_lists[v_local].append(c_local)

    sampled_1d, hidden_1d_local, accept_len = lossy_bottom_up_verify(
        probs=probs,
        token_ids=token_ids,
        parent_indices=parent_indices - int(skip_nodes),
        children_lists=children_lists,
        root_index=root_local,
        eos_token_id=eos_token_id,
        do_sample=do_sample,
        threshold=float(lossy_threshold),
        window_size=int(lossy_window_size),
    )

    hidden_indices = hidden_1d_local + int(skip_nodes)
    sampled_tokens = sampled_1d.unsqueeze(0)
    total_len = int(sampled_1d.numel())
    return sampled_tokens, hidden_indices, (total_len, int(accept_len))
