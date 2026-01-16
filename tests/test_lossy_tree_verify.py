import torch

from specdecodes.models.utils.lossy_tree_verify import lossy_bottom_up_verify
from specdecodes.models.utils.wandb_logger import wandb_logger


def _make_probs(rows):
    probs = torch.tensor(rows, dtype=torch.float32)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


def test_lossy_verify_exact_match_window0():
    # Tree:
    # 0(root)
    #  ├─ 1(tok=5)
    #  └─ 2(tok=7)
    token_ids = torch.tensor([0, 5, 7], dtype=torch.long)
    parent = torch.tensor([-1, 0, 0], dtype=torch.long)
    children = [[1, 2], [], []]

    # root predicts 5, child1 predicts 1 (bonus)
    probs = _make_probs(
        [
            [0.01, 0.01, 0.01, 0.01, 0.01, 0.90, 0.01, 0.03, 0.01, 0.01],
            [0.80, 0.10, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.00, 0.00],
            [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.00],
        ]
    )

    sampled, hidden, accept_len = lossy_bottom_up_verify(
        probs=probs,
        token_ids=token_ids,
        parent_indices=parent,
        children_lists=children,
        root_index=0,
        eos_token_id=None,
        do_sample=False,
        threshold=0.99,
        window_size=0,
    )

    assert accept_len == 1
    assert sampled.tolist() == [5, 0]  # bonus from node 1 argmax -> 0
    assert hidden.tolist() == [0, 1]


def test_lossy_verify_prob_gate_accepts_when_above_threshold_window0():
    # Tree: root -> {child(tok=5), child(tok=7)}
    token_ids = torch.tensor([0, 5, 7], dtype=torch.long)
    parent = torch.tensor([-1, 0, 0], dtype=torch.long)
    children = [[1, 2], [], []]

    # Root argmax is 9 (no matching child). We allow a lossy accept only when the
    # target assigns probability >= threshold to some draft child token.
    probs = _make_probs(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.30, 0.0, 0.20, 0.0, 0.50],
            [0.05, 0.90, 0.01, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.01],
            [0.05, 0.90, 0.01, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.01],
        ]
    )

    sampled, hidden, accept_len = lossy_bottom_up_verify(
        probs=probs,
        token_ids=token_ids,
        parent_indices=parent,
        children_lists=children,
        root_index=0,
        eos_token_id=None,
        do_sample=False,
        threshold=0.25,
        window_size=0,
    )

    assert accept_len == 1
    assert sampled[0].item() == 5
    assert hidden[0].item() == 0


def test_lossy_verify_prob_gate_rejects_when_below_threshold_window0():
    # Same tree, but now the target distribution does not match any child token and
    # assigns too little probability to draft child tokens. Lossy accept should be rejected.
    token_ids = torch.tensor([0, 5, 7], dtype=torch.long)
    parent = torch.tensor([-1, 0, 0], dtype=torch.long)
    children = [[1, 2], [], []]

    # Argmax is 9 (no matching child). Child tokens have low probability.
    probs = _make_probs(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.01, 0.0, 0.98],
            [0.05, 0.90, 0.01, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.01],
            [0.05, 0.90, 0.01, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.01],
        ]
    )

    sampled, hidden, accept_len = lossy_bottom_up_verify(
        probs=probs,
        token_ids=token_ids,
        parent_indices=parent,
        children_lists=children,
        root_index=0,
        eos_token_id=None,
        do_sample=False,
        threshold=0.4,
        window_size=0,
    )

    # No draft token accepted; we only emit the bonus token (argmax=9).
    assert accept_len == 0
    assert sampled.tolist() == [9]
    assert hidden.tolist() == [0]


def test_lossy_verify_window_lookahead():
    # Chain: root(0) -> a(5) -> b(9)
    # With window_size=1, token a can be accepted only if it has 1 future locally-correct node (b).
    # Note: the lossy verifier is non-regressing vs baseline exact-match.
    # Even if window gating would reject b (no future lookahead), we fall back to the
    # baseline exact-match chain when it yields a longer acceptance.
    token_ids = torch.tensor([0, 5, 9], dtype=torch.long)
    parent = torch.tensor([-1, 0, 1], dtype=torch.long)
    children = [[1], [2], []]

    probs = _make_probs(
        [
            # root predicts 5
            [0.01, 0.01, 0.01, 0.01, 0.01, 0.90, 0.01, 0.02, 0.01, 0.01],
            # a predicts 9
            [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01, 0.02, 0.90],
            # b predicts 1 (bonus)
            [0.01, 0.90, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        ]
    )

    sampled, hidden, accept_len = lossy_bottom_up_verify(
        probs=probs,
        token_ids=token_ids,
        parent_indices=parent,
        children_lists=children,
        root_index=0,
        eos_token_id=None,
        do_sample=False,
        threshold=0.99,
        window_size=1,
    )

    assert accept_len == 2
    assert sampled.tolist() == [5, 9, 1]  # bonus from node 2 argmax -> 1
    assert hidden.tolist() == [0, 1, 2]


def test_lossy_verify_logs_mismatch_prob_means_and_rates():
    # Tree: root -> {child(tok=5), child(tok=7)}
    # Target argmax is 9, so this is a mismatch context.
    # We accept tok=5 lossy because probs[root, 5] >= threshold.
    wandb_logger.clear_log_data()

    token_ids = torch.tensor([0, 5, 7], dtype=torch.long)
    parent = torch.tensor([-1, 0, 0], dtype=torch.long)
    children = [[1, 2], [], []]

    probs = _make_probs(
        [
            # root: argmax 9, but child(5)=0.4 is above threshold.
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.399, 0.0, 0.20, 0.0, 0.401],
            [0.05, 0.90, 0.01, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.01],
            [0.05, 0.90, 0.01, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.01],
        ]
    )

    sampled, hidden, accept_len = lossy_bottom_up_verify(
        probs=probs,
        token_ids=token_ids,
        parent_indices=parent,
        children_lists=children,
        root_index=0,
        eos_token_id=None,
        do_sample=False,
        threshold=0.30,
        window_size=0,
    )

    assert accept_len == 1
    assert sampled[0].item() == 5
    assert hidden[0].item() == 0

    # Token-level acceptance.
    assert wandb_logger.log_data["verify_accept_tokens"] == 1.0
    assert wandb_logger.log_data["verify_lossy_accept_tokens"] == 1.0
    assert abs(wandb_logger.log_data["verify_lossy_accept_rate"] - 1.0) < 1e-9

    # In this toy case, the mismatch is window-eligible and not threshold-blocked.
    assert abs(wandb_logger.log_data["lossy_window_ok_gate_mean"] - 0.399) < 1e-6
    assert abs(wandb_logger.log_data["lossy_window_ok_gate_std"] - 0.0) < 1e-9
    assert abs(wandb_logger.log_data["lossy_window_ok_drop_rate"] - 0.0) < 1e-9

    # Probability assigned by target to the accepted lossy token.
    assert abs(wandb_logger.log_data["lossy_accepted_prob_mean"] - 0.399) < 1e-6
    assert abs(wandb_logger.log_data["lossy_accepted_prob_std"] - 0.0) < 1e-9

    # Rate of lossy acceptance when the target token is missing among children.
    assert abs(wandb_logger.log_data["lossy_accept_rate_when_no_match"] - 1.0) < 1e-9


def test_lossy_verify_prefers_longer_lossy_over_short_exact_match():
    # Tree:
    # 0(root)
    #  ├─ 1(tok=5)   [exact match but dead end]
    #  └─ 2(tok=7)   [lossy-acceptable, leads to 3]
    #       └─ 3(tok=8) [exact match from node 2]
    token_ids = torch.tensor([0, 5, 7, 8], dtype=torch.long)
    parent = torch.tensor([-1, 0, 0, 2], dtype=torch.long)
    children = [[1, 2], [], [3], []]

    # At root, target argmax is 5, but tok=7 has prob >= threshold so lossy is allowed.
    # At node 2, target argmax is 8 (matching child 3).
    probs = _make_probs(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.60, 0.0, 0.35, 0.0, 0.05],
            [0.90, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.05, 0.80, 0.10],
            [0.90, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    sampled, hidden, accept_len = lossy_bottom_up_verify(
        probs=probs,
        token_ids=token_ids,
        parent_indices=parent,
        children_lists=children,
        root_index=0,
        eos_token_id=None,
        do_sample=False,
        threshold=0.30,
        window_size=0,
    )

    # Longest acceptable path should be lossy at root (7) then exact at node 2 (8).
    assert accept_len == 2
    assert sampled.tolist()[:2] == [7, 8]
    assert hidden.tolist()[:2] == [0, 2]


def test_lossy_verify_tie_break_prefers_exact_match():
    # Tree:
    # 0(root)
    #  ├─ 1(tok=5)   [exact match]
    #  └─ 2(tok=7)   [lossy-acceptable]
    # Both children are leaves, so both yield equal accept length (=1).
    # The verifier should prefer the exact-match child.
    wandb_logger.clear_log_data()

    token_ids = torch.tensor([0, 5, 7], dtype=torch.long)
    parent = torch.tensor([-1, 0, 0], dtype=torch.long)
    children = [[1, 2], [], []]

    probs = _make_probs(
        [
            # root target argmax is 5; tok=7 is above threshold but should lose the tie.
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.60, 0.0, 0.35, 0.0, 0.05],
            [0.90, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.90, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    sampled, hidden, accept_len = lossy_bottom_up_verify(
        probs=probs,
        token_ids=token_ids,
        parent_indices=parent,
        children_lists=children,
        root_index=0,
        eos_token_id=None,
        do_sample=False,
        threshold=0.30,
        window_size=0,
    )

    assert accept_len == 1
    assert sampled.tolist()[0] == 5
    assert hidden.tolist()[0] == 0


def test_lossy_verify_long_tree_window4_prefers_long_lookahead_branch():
    # Build a longer, branching tree where a higher-probability branch exists,
    # but it should be rejected by the window lookahead gating.
    #
    # Window semantics in lossy verifier:
    # window_size=4 means the mismatch child is eligible only if there are
    # >= 4 exact-match tokens after that child.
    #
    # Structure (indices -> token):
    # 0(root=0)
    # ├─ 1(tok=5)  [good long chain]
    # │   ├─ 3(tok=9)   [argmax distractor, leaf]
    # │   └─ 4(tok=6)
    # │       ├─ 6(tok=11) [argmax distractor, leaf]
    # │       └─ 7(tok=7)
    # │           ├─ 9(tok=13)  [argmax distractor, leaf]
    # │           └─ 10(tok=14)
    # │               ├─ 11(tok=15) [argmax distractor, leaf]
    # │               └─ 12(tok=16)
    # │                   ├─ 13(tok=17) [argmax distractor, leaf]
    # │                   └─ 14(tok=18)
    # │                       └─ 15(tok=19)
    # │                           └─ 16(tok=4)
    # └─ 2(tok=8)  [short chain: high prob but fails window gating]
    #     └─ 5(tok=10)
    #         └─ 8(tok=12)

    token_ids = torch.tensor(
        [0, 5, 8, 9, 6, 10, 11, 7, 12, 13, 14, 15, 16, 17, 18, 19, 4],
        dtype=torch.long,
    )
    parent = torch.tensor(
        [-1, 0, 0, 1, 1, 2, 4, 4, 5, 7, 7, 10, 10, 12, 12, 14, 15],
        dtype=torch.long,
    )
    children = [
        [1, 2],  # 0
        [3, 4],  # 1
        [5],
        [],
        [6, 7],
        [8],
        [],
        [9, 10],
        [],
        [],
        [11, 12],
        [],
        [13, 14],
        [],
        [15],
        [16],
        [],
    ]

    vocab = 20

    def row(main_tok: int, main_p: float, alt_tok: int, alt_p: float):
        r = [0.0] * vocab
        r[main_tok] = main_p
        r[alt_tok] = alt_p
        # tiny remainder to keep other tokens non-negative after normalization
        rem = max(0.0, 1.0 - main_p - alt_p)
        if rem > 0:
            r[0] += rem
        return r

    # We set do_sample=False and threshold=0.35.
    # Clarified window semantics:
    # - If target token matches a child, accept it (no window gating).
    # - If target token does NOT match any child, we may accept a non-matching child
    #   only if prob>=threshold AND we can accept `window_size` exact tokens afterward.
    probs_rows = [
        # 0: target token=3 (no matching child); lossy may choose 5 if it has long lookahead.
        # Give tok=5 enough probability to clear the threshold.
        [0.0] * vocab,
        row(6, 0.80, 9, 0.10),   # 1: exact match to tok=6 (node 4)
        row(1, 0.90, 10, 0.05),  # 2: no matching child and low prob for child tok=10 -> short branch dies
        row(0, 0.90, 1, 0.05),
        row(7, 0.80, 11, 0.10),  # 4: exact match to tok=7 (node 7)
        row(1, 0.90, 2, 0.05),
        row(0, 0.90, 1, 0.05),
        row(14, 0.80, 13, 0.10), # 7: exact match to tok=14 (node 10)
        row(0, 0.90, 1, 0.05),
        row(0, 0.90, 1, 0.05),
        row(16, 0.80, 15, 0.10), # 10: exact match to tok=16 (node 12)
        row(0, 0.90, 1, 0.05),
        row(18, 0.80, 17, 0.10), # 12: exact match to tok=18 (node 14)
        row(0, 0.90, 1, 0.05),
        row(19, 0.80, 1, 0.10),  # 14: exact match to tok=19 (node 15)
        row(4, 0.80, 2, 0.10),   # 15: exact match to tok=4 (node 16)
        row(15, 0.80, 0, 0.10),  # 16: leaf, bonus would be 15
    ]
    probs_rows[0][3] = 0.40
    probs_rows[0][5] = 0.36
    probs_rows[0][8] = 0.24
    probs = _make_probs(probs_rows)

    sampled, hidden, accept_len = lossy_bottom_up_verify(
        probs=probs,
        token_ids=token_ids,
        parent_indices=parent,
        children_lists=children,
        root_index=0,
        eos_token_id=None,
        do_sample=False,
        threshold=0.35,
        window_size=4,
    )

    # Root has no matching child, so lossy should pick tok=5 (node 1) only if there is
    # >= window_size exact-match tokens available after it.
    assert accept_len == 8
    assert sampled.tolist()[:8] == [5, 6, 7, 14, 16, 18, 19, 4]
    assert sampled.tolist()[-1] == 15  # bonus from node 16 argmax
    assert hidden.tolist() == [0, 1, 4, 7, 10, 12, 14, 15, 16]

if __name__ == "__main__":
    print("Running lossy tree verify tests...")
    test_lossy_verify_exact_match_window0()
    test_lossy_verify_prob_gate_accepts_when_above_threshold_window0()
    test_lossy_verify_prob_gate_rejects_when_below_threshold_window0()
    test_lossy_verify_window_lookahead()
    test_lossy_verify_long_tree_window4_prefers_long_lookahead_branch()
    print("All tests passed.")