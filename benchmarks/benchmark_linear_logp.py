# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Benchmark NativeLinearLogpOp vs TritonLinearLogpOp.

The native op materializes the full [N, V] logits (a single F.linear +
log_softmax + gather); the Triton op streams the vocab through an online softmax
and never lands the logits, so its peak forward memory is independent of V. This
is the headline number: peak VRAM vs the materializing baseline, swept over
vocab size. Latency (forward and forward+backward) is reported too -- the win is
memory, not FLOPs (the backward recomputes the logit tiles).

Usage:
    python benchmarks/benchmark_linear_logp.py
    python benchmarks/benchmark_linear_logp.py --configs "4096,2048,32768;4096,2048,131072"
"""

import argparse

import torch
from tabulate import tabulate

from rl_engine.kernels.ops.pytorch.loss.linear_logp import NativeLinearLogpOp
from rl_engine.kernels.ops.triton.loss.linear_logp import TritonLinearLogpOp
from rl_engine.platforms.device import device_ctx
from rl_engine.utils.logger import logger

# (num_tokens, hidden_dim, vocab)
DEFAULT_CONFIGS = [
    (4096, 2048, 32768),
    (4096, 2048, 50257),
    (4096, 2048, 131072),
]


def _make_inputs(num_tokens, hidden_dim, vocab, device, dtype):
    hidden = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    weight = torch.randn(vocab, hidden_dim, device=device, dtype=dtype)
    target = torch.randint(0, vocab, (num_tokens,), device=device)
    return hidden, weight, target


def _time_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _peak_vram_gb(fn, warmup=3, iters=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - baseline) / (1024**3)


def run_benchmark(args):
    if device_ctx.device_type != "cuda":
        raise RuntimeError("linear_logp benchmark requires a CUDA device (Triton op is CUDA-only).")

    device = device_ctx.device
    dtype = torch.bfloat16
    native = NativeLinearLogpOp()
    triton_op = TritonLinearLogpOp()

    logger.info(f"linear_logp benchmark on {device} (dtype={dtype})")

    rows = []
    for num_tokens, hidden_dim, vocab in args.configs:
        hidden, weight, target = _make_inputs(num_tokens, hidden_dim, vocab, device, dtype)

        def native_fwd(h=hidden, w=weight):
            with torch.no_grad():
                native(h, w, target)

        def triton_fwd(h=hidden, w=weight):
            with torch.no_grad():
                triton_op(h, w, target)

        def native_fwd_bwd():
            h = hidden.clone().requires_grad_(True)
            w = weight.clone().requires_grad_(True)
            native(h, w, target).sum().backward()

        def triton_fwd_bwd():
            h = hidden.clone().requires_grad_(True)
            w = weight.clone().requires_grad_(True)
            triton_op(h, w, target).sum().backward()

        n_fwd = _time_ms(native_fwd, args.warmup, args.iters)
        t_fwd = _time_ms(triton_fwd, args.warmup, args.iters)
        n_fb = _time_ms(native_fwd_bwd, args.warmup, args.iters)
        t_fb = _time_ms(triton_fwd_bwd, args.warmup, args.iters)
        n_vram = _peak_vram_gb(native_fwd)
        t_vram = _peak_vram_gb(triton_fwd)

        rows.append(
            [
                f"{num_tokens}x{hidden_dim}x{vocab}",
                f"{n_fwd:.3f}",
                f"{t_fwd:.3f}",
                f"{n_fwd/t_fwd:.2f}x",
                f"{n_fb:.3f}",
                f"{t_fb:.3f}",
                f"{n_fb/t_fb:.2f}x",
                f"{n_vram*1024:.0f}",
                f"{t_vram*1024:.0f}",
            ]
        )

    headers = [
        "shape (N x H x V)",
        "native fwd ms",
        "triton fwd ms",
        "fwd speedup",
        "native f+b ms",
        "triton f+b ms",
        "f+b speedup",
        "native fwd MB",
        "triton fwd MB",
    ]
    print(tabulate(rows, headers=headers, tablefmt="github"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help="Semicolon-separated 'tokens,hidden,vocab' tuples, "
        "e.g. '4096,2048,32768;4096,2048,131072'.",
    )
    args = parser.parse_args()
    if args.configs:
        args.configs = [tuple(int(x) for x in tup.split(",")) for tup in args.configs.split(";")]
    else:
        args.configs = DEFAULT_CONFIGS
    return args


if __name__ == "__main__":
    run_benchmark(parse_args())
