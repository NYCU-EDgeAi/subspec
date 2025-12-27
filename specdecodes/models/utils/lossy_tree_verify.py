import torch
from typing import Tuple, Optional


@torch.no_grad()
def lossy_bottom_up_verify(
    *,
    probs: torch.Tensor,
    token_ids: torch.Tensor,
    parent_indices: torch.Tensor,
    children_lists: list[list[int]],
    root_index: int,
    eos_token_id: Optional[int],
    do_sample: bool,
    threshold: float,
    window_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Bottom-up lossy verification over a speculative *tree chunk*.

    Inputs are in *local chunk indexing* (0..num_nodes-1). `probs[i]` is the
    target distribution produced *at node i* (i.e., predicting the next token
    after token_ids[i]).

        Verification rule (as clarified):
            - If target's generated token matches a draft child token, accept it (same
                behavior as the classic verifier).
            - Otherwise, we may accept a *non-matching* draft child token c only if:
                    1) probs[parent, token_ids[c]] >= threshold, and
                    2) if we accept c and continue verifying, we can accept at least
                         `window_size` additional draft tokens afterward.

        This is implemented via a bottom-up DP `best_len[u]` which counts the maximum
        number of draft tokens we can accept starting from context node u.

    Returns:
      sampled_tokens: 1D (accept_len + 1,) tensor (accepted draft tokens + bonus)
      hidden_indices: 1D indices aligned with sampled_tokens semantics used in
        this repo (indices of context nodes whose logits were used to emit each
        sampled token).
      accept_len: number of accepted draft tokens (excluding bonus).
    """
    if probs.dim() != 2:
        raise ValueError(f"probs must be 2D (num_nodes, vocab), got {tuple(probs.shape)}")
    if token_ids.dim() != 1:
        raise ValueError(f"token_ids must be 1D (num_nodes,), got {tuple(token_ids.shape)}")
    if parent_indices.dim() != 1:
        raise ValueError(f"parent_indices must be 1D (num_nodes,), got {tuple(parent_indices.shape)}")

    num_nodes = int(probs.shape[0])
    if token_ids.size(0) != num_nodes or parent_indices.size(0) != num_nodes:
        raise ValueError("token_ids/parent_indices must match probs first dimension")
    if not (0 <= root_index < num_nodes):
        raise ValueError(f"root_index out of range: {root_index} (num_nodes={num_nodes})")

    # If probs is on CUDA, repeated scalar .item() calls will cause many synchronizations.
    # Move to CPU once for verification; this matches the existing verifier pattern in this repo.
    if probs.device.type != "cpu":
        probs = probs.cpu()

    threshold_f = float(threshold)
    required_lookahead = max(0, int(window_size))

    # Target token for each node distribution (one per node/context).
    if do_sample:
        target_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    else:
        target_tokens = probs.argmax(dim=-1)

    # Cache ids as Python ints to avoid repeated tensor -> Python conversions in inner loops.
    token_ids_i = token_ids.tolist()
    target_tokens_i = target_tokens.tolist()

    # Bottom-up DP over accepted length.
    best_len: list[int] = [0] * num_nodes
    best_next: list[int] = [-1] * num_nodes

    for u in range(num_nodes - 1, -1, -1):
        children = children_lists[u]
        if not children:
            continue

        tgt = target_tokens_i[u]

        # 1) Exact-match behaves as usual: if matching child exists, take it.
        match_child = next((c for c in children if token_ids_i[c] == tgt), None)
        if match_child is not None:
            best_next[u] = int(match_child)
            best_len[u] = 1 + best_len[int(match_child)]
            continue

        # 2) Otherwise, consider lossy acceptance (threshold + lookahead).
        best_c = -1
        best_child_len = -1
        best_p = -1.0

        for c in children:
            c = int(c)
            if best_len[c] < required_lookahead:
                continue

            tok = token_ids_i[c]
            p = float(probs[u, tok].item())
            if p < threshold_f:
                continue

            child_len = best_len[c]
            if child_len > best_child_len or (child_len == best_child_len and p > best_p):
                best_c = c
                best_child_len = child_len
                best_p = p

        if best_c >= 0:
            best_next[u] = best_c
            best_len[u] = 1 + best_len[best_c]

    # Top-down extraction for the chosen policy.
    sampled_tokens: list[int] = []
    hidden_indices: list[int] = []
    context = root_index
    accept_len = 0

    while True:
        nxt = best_next[context]
        if nxt < 0:
            break

        tok = token_ids_i[nxt]
        sampled_tokens.append(tok)
        hidden_indices.append(context)
        accept_len += 1

        if eos_token_id is not None and tok == int(eos_token_id):
            break

        context = nxt

    # Bonus token from target at the final context (or root if none accepted).
    if not sampled_tokens or (eos_token_id is None) or (sampled_tokens[-1] != int(eos_token_id)):
        sampled_tokens.append(target_tokens_i[context])
        hidden_indices.append(context)

    return (
        torch.tensor(sampled_tokens, dtype=torch.long),
        torch.tensor(hidden_indices, dtype=torch.long),
        accept_len,
    )
