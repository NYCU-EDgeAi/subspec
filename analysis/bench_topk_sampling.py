import argparse
import time
from typing import Optional, Tuple

import torch


def old_flatten_topk(
    sampled_probs: torch.Tensor, parent_probs: torch.Tensor, sample_k: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference implementation (old behavior): flatten [leaves*vocab] then topk."""
    batch_size, num_leaves, vocab_size = sampled_probs.shape
    global_probs = sampled_probs * parent_probs.unsqueeze(-1)
    flattened_probs = global_probs.view(batch_size, -1)
    topk_probs, topk_indices = torch.topk(flattened_probs, sample_k, dim=1, sorted=True)
    parent_indices = (topk_indices // vocab_size).long()
    token_ids = (topk_indices % vocab_size).long()
    return token_ids, topk_probs, parent_indices


def flatten_topk_outbuf(
    sampled_probs: torch.Tensor,
    parent_probs: torch.Tensor,
    sample_k: int,
    outbuf: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact: flatten [leaves*vocab] then topk, but avoid per-iter allocation via out=."""
    batch_size, num_leaves, vocab_size = sampled_probs.shape
    k = min(int(sample_k), int(vocab_size))

    if outbuf is None or outbuf.shape != sampled_probs.shape or outbuf.dtype != sampled_probs.dtype:
        outbuf = torch.empty_like(sampled_probs)

    torch.mul(sampled_probs, parent_probs.unsqueeze(-1), out=outbuf)
    flattened_probs = outbuf.view(batch_size, -1)
    topk_probs, topk_indices = torch.topk(flattened_probs, k, dim=1, sorted=True)
    parent_indices = (topk_indices // vocab_size).long()
    token_ids = (topk_indices % vocab_size).long()
    return token_ids, topk_probs, parent_indices


def two_stage_topk(
    sampled_probs: torch.Tensor, parent_probs: torch.Tensor, sample_k: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact optimized implementation: per-leaf topk then global topk over candidates."""
    batch_size, num_leaves, vocab_size = sampled_probs.shape
    k = min(int(sample_k), int(vocab_size))

    leaf_topk_probs, leaf_topk_token_ids = torch.topk(sampled_probs, k, dim=-1, sorted=True)
    global_probs = leaf_topk_probs * parent_probs.unsqueeze(-1)

    flattened_probs = global_probs.reshape(batch_size, -1)
    flattened_token_ids = leaf_topk_token_ids.reshape(batch_size, -1)

    leaf_indices = (
        torch.arange(num_leaves, device=sampled_probs.device)
        .view(1, num_leaves, 1)
        .expand(batch_size, num_leaves, k)
        .reshape(batch_size, -1)
    )

    topk_probs, topk_indices = torch.topk(flattened_probs, k, dim=1, sorted=True)
    parent_indices = leaf_indices.gather(1, topk_indices).long()
    token_ids = flattened_token_ids.gather(1, topk_indices).long()
    return token_ids, topk_probs, parent_indices


def _time_cuda(fn, iters: int) -> float:
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    starter.record()
    sink = None
    for _ in range(iters):
        out = fn()
        # Some kernels may run on auxiliary streams. Consume outputs on the
        # current stream to ensure the recorded CUDA events include the work.
        if isinstance(out, (tuple, list)):
            t = out[1]
        else:
            t = out
        s = t.sum()
        sink = s if sink is None else (sink + s)
    ender.record()
    torch.cuda.synchronize()
    # Prevent any chance of eliding the computation.
    if sink is not None:
        _ = float(sink.item())
    # ms -> s
    return float(starter.elapsed_time(ender)) / 1000.0


def _time_cpu(fn, iters: int) -> float:
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--vocab", type=int, default=32000)
    p.add_argument("--k", type=int, default=8)
    p.add_argument(
        "--sampled_dist",
        choices=["uniform", "softmax"],
        default="softmax",
        help="How to generate sampled_probs for benchmarking.",
    )
    # In the real decode, there are exactly k leaves each step.
    # This benchmark fixes num_leaves == k.
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)

    torch.manual_seed(0)

    leaves = int(args.k)

    if args.sampled_dist == "softmax":
        # More realistic: probabilities from a softmax over random logits.
        logits = torch.randn(args.batch, leaves, args.vocab, device=device, dtype=torch.float32)
        sampled_probs = torch.softmax(logits, dim=-1).to(dtype=dtype)
    else:
        sampled_probs = torch.rand(args.batch, leaves, args.vocab, device=device, dtype=dtype)

    parent_probs = torch.rand(args.batch, leaves, device=device, dtype=dtype)

    outbuf = torch.empty_like(sampled_probs)

    # warmup
    for _ in range(args.warmup):
        old_flatten_topk(sampled_probs, parent_probs, args.k)
        flatten_topk_outbuf(sampled_probs, parent_probs, args.k, outbuf=outbuf)
        two_stage_topk(sampled_probs, parent_probs, args.k)

    # correctness check (tolerate extremely rare tie edge cases)
    tok0, prob0, par0 = old_flatten_topk(sampled_probs, parent_probs, args.k)
    tokb, probb, parb = flatten_topk_outbuf(sampled_probs, parent_probs, args.k, outbuf=outbuf)
    tok1, prob1, par1 = two_stage_topk(sampled_probs, parent_probs, args.k)

    def _assert_matches(tok, prob, par):
        if torch.equal(tok0, tok) and torch.equal(par0, par) and torch.allclose(prob0, prob):
            return
        vocab = sampled_probs.shape[-1]
        scores0 = (sampled_probs * parent_probs.unsqueeze(-1)).view(args.batch, -1)
        ref = torch.topk(scores0, args.k, dim=1, sorted=True)
        opt_idx = (par * vocab + tok).long()
        opt_scores = scores0.gather(1, opt_idx)
        if not torch.allclose(opt_scores, ref.values):
            raise AssertionError("Optimized topk mismatch vs reference")

    _assert_matches(tok1, prob1, par1)
    _assert_matches(tokb, probb, parb)

    def run_old():
        return old_flatten_topk(sampled_probs, parent_probs, args.k)

    def run_outbuf():
        return flatten_topk_outbuf(sampled_probs, parent_probs, args.k, outbuf=outbuf)

    def run_two_stage():
        return two_stage_topk(sampled_probs, parent_probs, args.k)

    if device.type == "cuda":
        t_old = _time_cuda(run_old, args.iters)
        t_out = _time_cuda(run_outbuf, args.iters)
        t_two = _time_cuda(run_two_stage, args.iters)
    else:
        t_old = _time_cpu(run_old, args.iters)
        t_out = _time_cpu(run_outbuf, args.iters)
        t_two = _time_cpu(run_two_stage, args.iters)
    print(
        "shape=(B={b}, leaves={l}, vocab={v}), k={k}, dtype={dt}, device={dev}".format(
            b=args.batch, l=leaves, v=args.vocab, k=args.k, dt=args.dtype, dev=args.device
        )
    )
    print(f"sampled_dist  : {args.sampled_dist}")
    print(f"old_flatten_topk: {t_old:.6f}s total ({t_old/args.iters*1e3:.3f} ms/iter)")
    print(f"flatten_outbuf  : {t_out:.6f}s total ({t_out/args.iters*1e3:.3f} ms/iter)")
    print(f"two_stage_topk  : {t_two:.6f}s total ({t_two/args.iters*1e3:.3f} ms/iter)")
    print(f"two_stage speed : {t_old / max(t_two, 1e-12):.2f}x")
    print(f"outbuf speed    : {t_old / max(t_out, 1e-12):.2f}x")


if __name__ == "__main__":
    main()
