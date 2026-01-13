import torch
from typing import Tuple, Optional

from .wandb_logger import wandb_logger


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

        # Consider both exact-match and lossy-acceptable children and choose the one
        # that yields the longest accept sequence.
        best_c = -1
        best_total_len = 0
        best_is_exact = False
        best_p = -1.0

        for c in children:
            c = int(c)
            tok = token_ids_i[c]

            is_exact = (tok == tgt)
            if is_exact:
                is_acceptable = True
                p = -1.0  # not used for exact-match unless tie-breaking falls through
            else:
                # Lossy acceptability: window constraint + probability threshold.
                if best_len[c] < required_lookahead:
                    continue
                p = float(probs[u, tok].item())
                if p < threshold_f:
                    continue
                is_acceptable = True

            if not is_acceptable:
                continue

            total_len = 1 + best_len[c]
            if total_len > best_total_len:
                best_c = c
                best_total_len = total_len
                best_is_exact = is_exact
                best_p = p
                continue

            if total_len == best_total_len:
                # Prefer exact-match when lengths are equal; otherwise prefer higher probability.
                if is_exact and not best_is_exact:
                    best_c = c
                    best_is_exact = True
                    best_p = p
                elif (not is_exact) and (not best_is_exact) and p > best_p:
                    best_c = c
                    best_p = p

        if best_c >= 0 and best_total_len > 0:
            best_next[u] = best_c
            best_len[u] = best_total_len

    # Top-down extraction for the chosen policy.
    sampled_tokens: list[int] = []
    hidden_indices: list[int] = []
    context = root_index
    accept_len = 0

    # Path-only diagnostics:
    # - mismatch: contexts we actually visited where target token is not among draft children.
    # - threshold-drop mismatch: mismatch contexts where window constraint is satisfiable, but
    #   we did not accept due to threshold.
    path_mismatch_steps = 0
    path_mismatch_eligible_steps = 0
    path_mismatch_threshold_drop_steps = 0

    mismatch_accepted_steps = 0

    mismatch_accepted_no_match_steps = 0

    def _welford_update(stat_key: str, x: float) -> None:
        """Update a running mean/std in wandb_logger.internal_data.

        We keep internal accumulators out of JSONL by not storing them in
        wandb_logger.log_data.
        """
        state = wandb_logger.internal_data.get(stat_key)
        if state is None:
            state = {"n": 0, "mean": 0.0, "M2": 0.0}

        n0 = int(state["n"])
        mean0 = float(state["mean"])
        M2_0 = float(state["M2"])

        n1 = n0 + 1
        delta = x - mean0
        mean1 = mean0 + delta / n1
        delta2 = x - mean1
        M2_1 = M2_0 + delta * delta2

        wandb_logger.internal_data[stat_key] = {"n": n1, "mean": mean1, "M2": M2_1}

    def _welford_mean_std(stat_key: str) -> tuple[float, float]:
        state = wandb_logger.internal_data.get(stat_key)
        if not state:
            return 0.0, 0.0
        n = int(state.get("n", 0))
        mean = float(state.get("mean", 0.0))
        M2 = float(state.get("M2", 0.0))
        if n <= 0:
            return 0.0, 0.0
        var = M2 / n
        return mean, float(max(var, 0.0) ** 0.5)

    def _is_mismatch_context(u: int) -> bool:
        children = children_lists[u]
        if not children:
            return False
        tgt = target_tokens_i[u]
        return all(token_ids_i[int(c)] != tgt for c in children)

    def _max_window_ok_child_prob(u: int) -> Optional[float]:
        """Return max target prob among children that satisfy the lookahead window.

        "window_ok" means the child has enough downstream acceptability such that
        best_len[child] >= window_size (required_lookahead).
        """
        max_p: Optional[float] = None
        for c in children_lists[u]:
            c = int(c)
            if best_len[c] < required_lookahead:
                continue
            tok = token_ids_i[c]
            p = float(probs[u, tok].item())
            if max_p is None or p > max_p:
                max_p = p
        return max_p

    while True:
        # Path-only mismatch statistics are computed for each visited context, including the
        # final context where we stop accepting and emit the bonus token.
        if _is_mismatch_context(context):
            path_mismatch_steps += 1

            best_window_ok_p = _max_window_ok_child_prob(context)
            if best_window_ok_p is not None:
                path_mismatch_eligible_steps += 1
                _welford_update("lossy/window_ok_best_prob", float(best_window_ok_p))

                # If we have an eligible (window-satisfying) child but its prob is below threshold,
                # then lowering the threshold to <= best_eligible_p would allow a lossy accept here.
                if best_window_ok_p < threshold_f:
                    path_mismatch_threshold_drop_steps += 1
                    _welford_update("lossy/window_ok_drop_best_prob", float(best_window_ok_p))

        nxt = best_next[context]
        if nxt < 0:
            break

        tok = token_ids_i[nxt]
        sampled_tokens.append(tok)
        hidden_indices.append(context)
        accept_len += 1

        # If target token doesn't match the accepted draft token, this is a lossy-accepted mismatch.
        if tok != target_tokens_i[context]:
            mismatch_accepted_steps += 1
            p = float(probs[context, tok].item())
            _welford_update("verify/mm_acc_p", float(p))

            # Count lossy accepts that occurred specifically in contexts where the
            # target token was not among any draft children.
            if _is_mismatch_context(context):
                mismatch_accepted_no_match_steps += 1

        if eos_token_id is not None and tok == int(eos_token_id):
            break

        context = nxt

    # Bonus token from target at the final context (or root if none accepted).
    if not sampled_tokens or (eos_token_id is None) or (sampled_tokens[-1] != int(eos_token_id)):
        sampled_tokens.append(target_tokens_i[context])
        hidden_indices.append(context)

    # Persist diagnostics into the per-generation log (accumulate across verify calls).
    # These are intended to be lightweight and easy to interpret.
    def _int_acc(key: str, value: float) -> None:
        wandb_logger.internal_data[key] = float(wandb_logger.internal_data.get(key, 0.0)) + float(value)

    # Keep raw counts internal (not written to JSONL) to avoid confusing logs.
    _int_acc("lossy/accept_tokens", float(accept_len))
    _int_acc("lossy/mm_ctx", float(path_mismatch_steps))
    _int_acc("lossy/mm_elig_ctx", float(path_mismatch_eligible_steps))
    _int_acc("lossy/mm_drop_ctx", float(path_mismatch_threshold_drop_steps))
    _int_acc("lossy/mm_acc_steps", float(mismatch_accepted_steps))
    _int_acc("lossy/mm_acc_nomatch_steps", float(mismatch_accepted_no_match_steps))

    # Export only actionable, easy-to-interpret metrics.
    accept_tokens = float(wandb_logger.internal_data.get("lossy/accept_tokens", 0.0))
    mm_ctx = float(wandb_logger.internal_data.get("lossy/mm_ctx", 0.0))
    mm_elig_ctx = float(wandb_logger.internal_data.get("lossy/mm_elig_ctx", 0.0))
    mm_drop_ctx = float(wandb_logger.internal_data.get("lossy/mm_drop_ctx", 0.0))
    mm_acc_steps = float(wandb_logger.internal_data.get("lossy/mm_acc_steps", 0.0))
    mm_acc_nomatch_steps = float(wandb_logger.internal_data.get("lossy/mm_acc_nomatch_steps", 0.0))

    wandb_logger.log_data["verify_accept_tokens"] = accept_tokens
    wandb_logger.log_data["verify_lossy_accept_tokens"] = mm_acc_steps
    wandb_logger.log_data["verify_lossy_accept_rate"] = (mm_acc_steps / accept_tokens) if accept_tokens > 0 else 0.0

    # "If I lower threshold, what would I unlock?" (only when window constraint is satisfiable)
    # window_ok := some child satisfies the window_size lookahead constraint.
    wandb_logger.log_data["lossy_window_ok_drop_rate"] = (mm_drop_ctx / mm_elig_ctx) if mm_elig_ctx > 0 else 0.0
    wandb_logger.log_data["lossy_accept_rate_when_no_match"] = (mm_acc_nomatch_steps / mm_ctx) if mm_ctx > 0 else 0.0

    window_ok_mean, window_ok_std = _welford_mean_std("lossy/window_ok_best_prob")
    wandb_logger.log_data["lossy_window_ok_best_prob_mean"] = window_ok_mean
    wandb_logger.log_data["lossy_window_ok_best_prob_std"] = window_ok_std

    drop_mean, drop_std = _welford_mean_std("lossy/window_ok_drop_best_prob")
    wandb_logger.log_data["lossy_window_ok_drop_best_prob_mean"] = drop_mean
    wandb_logger.log_data["lossy_window_ok_drop_best_prob_std"] = drop_std

    acc_p_mean, acc_p_std = _welford_mean_std("verify/mm_acc_p")
    wandb_logger.log_data["lossy_accepted_prob_mean"] = acc_p_mean
    wandb_logger.log_data["lossy_accepted_prob_std"] = acc_p_std

    return (
        torch.tensor(sampled_tokens, dtype=torch.long),
        torch.tensor(hidden_indices, dtype=torch.long),
        accept_len,
    )
