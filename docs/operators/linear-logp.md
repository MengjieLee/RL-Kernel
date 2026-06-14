# Fused Linear LogP

Fused Linear LogP computes per-token selected log-probabilities directly from
**hidden states and the LM-head weight** — `log_softmax(hidden @ Wᵀ + b)[target]` —
**without ever materializing the `[N, V]` logits**. It targets large-vocabulary RL
post-training, where the `[B, S, V]` logits activation (and its gradient) dominate
memory. The forward streams the vocab in blocks through an online softmax; the
backward recomputes the logit tiles instead of storing them, trading compute for
memory. It is differentiable w.r.t. `hidden`, `lm_head_weight`, and `bias`.

This differs from [Fused LogP](fused-logp.md), which takes already-materialized
logits as input. Here the LM-head projection is fused into the reduction, so the
`[N, V]` tensor never lands in HBM.

## Entry Point

```python
from rl_engine.kernels.registry import kernel_registry

linear_logp = kernel_registry.get_op("linear_logp")

logp = linear_logp(
    hidden,         # [B, S, D] or [N, D]  (differentiable)
    lm_head_weight, # [V, D]               (differentiable)
    target_ids,     # [B, S] or [N]        int, in [0, V)
    bias=None,      # [V] optional         (differentiable)
)                   # -> [B, S] or [N], float32

logp.sum().backward()  # gradients flow into hidden, lm_head_weight, bias
```

## Backends

| Backend | Wrapper | Status |
| --- | --- | --- |
| CUDA SM90 (Hopper) | `FusedLinearLogpSM90Op` | TMA + WGMMA streaming forward, online softmax in smem; chunked backward. Compiles for `sm_90a`; **numerics pending validation on an SM90 GPU.** |
| CUDA / ROCm (Triton) | `TritonLinearLogpOp` | Triton online-softmax forward; Liger-style chunked backward (cuBLAS matmuls, deterministic). Phase 1. |
| PyTorch native | `NativeLinearLogpOp` | Naive `F.linear` + `log_softmax` + `gather` reference; CPU / Triton-less fallback. |

The SM90 backend (`csrc/cuda/fused_linear_logp_sm90.cu`) streams hidden/weight
tiles via TMA, computes each logit tile with WGMMA (`m64n64k16`), folds it into a
per-row online softmax in shared memory, and gathers the target logit — never
materializing `[N, V]`. It is **build-guarded**: only compiled when the extension
is built with `KERNEL_ALIGN_FORCE_SM90=1` on an SM90 device (WGMMA is Hopper-only,
`sm_90a`), and the registry only selects it when `cc_major == 9` and the symbol is
present. The forward kernel requires bf16 hidden/weight; the backward reuses the
deterministic chunked path. It assembles cleanly under CUDA 13.1 `ptxas`, but the
layout-sensitive pieces (WGMMA descriptors, accumulator→smem mapping, B-operand
transpose) require validation on Hopper hardware.

The backward chunks over the token dimension: for each chunk it materializes only
`[chunk, V]` logits, recomputes the softmax from scratch, and forms the three
gradients with cuBLAS matmuls (`grad_hidden = dz @ W`, `grad_weight += dzᵀ @ X`).
`grad_weight` is accumulated in sequential loop order, so it is **atomic-free and
bitwise-deterministic** while peak backward memory stays `chunk·V` instead of `N·V`.

The native op materializes the full `[N, V]` logits and is the correctness oracle;
the Triton op is the portable, fp32-accurate baseline that the future CUDA generic,
CUDA SM90 (TMA/WGMMA), and native ROCm backends are validated against. See the
[design doc](../design/fused-linear-logp.md) for the phased plan.

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `hidden` | `[N, D]` / `[B, S, D]` | bf16 / fp16 / fp32 | Differentiable; contiguous. |
| `lm_head_weight` | `[V, D]` | bf16 / fp16 / fp32 | Differentiable; contiguous (row-major over V). |
| `target_ids` | `[N]` / `[B, S]` | int | Token id per position, in `[0, V)`. |
| `bias` | `[V]` | float | Optional; differentiable. |
| Output | `[N]` / `[B, S]` | float32 | `z[target] − logsumexp(z)` per position. |

Gradients flow into `hidden`, `lm_head_weight`, and `bias`; `target_ids` is
integer and non-differentiable.

## Reference Semantics

```python
logits = torch.nn.functional.linear(hidden.float(), weight.float(), bias)  # [N, V]
logp = torch.log_softmax(logits, dim=-1)
out = logp.gather(-1, target_ids.long().unsqueeze(-1)).squeeze(-1)
```

The Triton kernel accumulates the matmul and softmax in float32, so it matches the
float32 reference to `atol≈1e-3`. For bf16/fp16 inputs it matches the **fp32-upcast**
reference (it is more accurate than a bf16 `F.linear`, which rounds the logits).

## Performance

```bash
python benchmarks/benchmark_linear_logp.py
python benchmarks/benchmark_linear_logp.py --configs "4096,2048,32768;4096,2048,131072"
```

Indicative results (RTX PRO 6000, SM120, bf16, N=4096, D=2048; native vs Triton):

| shape (N×H×V) | fwd | fwd+bwd | peak fwd VRAM (native → Triton) |
| --- | --- | --- | --- |
| 4096×2048×32768 | 0.53× | 0.43× | 1280 MB → ~0 MB |
| 4096×2048×50257 | 0.99× | 0.46× | 1965 MB → ~0 MB |
| 4096×2048×131072 | 0.64× | 0.23× | 5120 MB → ~0 MB |

The headline is memory: the native path allocates the `[N, V]` logits (forward peak
scales with `V`), while the fused op streams them online — its forward peak is
**independent of `V`**, and the chunked backward only ever holds `chunk·V`. The
forward matmul keeps operands in their native dtype, so bf16/fp16 inputs run on
tensor cores (fp32 accumulation) and the forward lands near cuBLAS parity; the fp32
path stays full-precision (`input_precision="ieee"`) for its role as the tolerance
reference. The backward runs at ~2–4× native: it recomputes the logits per chunk
(native keeps the `[N, V]` it already materialized), which is the compute-for-memory
trade. Closing that gap and the absolute forward latency are the job of the native
CUDA generic / SM90 (TMA/WGMMA) backends in later phases; Phase 1 delivers the
memory reduction and the correctness/tolerance baseline.

## Tests

```bash
python -m pytest tests/test_linear_logp.py -v
```

Covers the native reference vs the materialized definition, Triton forward (fp32 and
bf16) vs native, Triton backward vs native autograd (with and without bias), leading-
shape preservation, a large-vocab smoke test, and registry dispatch. Triton tests
skip without CUDA + Triton.

## Implementation Files

- `rl_engine/kernels/ops/triton/loss/linear_logp.py`
- `rl_engine/kernels/ops/pytorch/loss/linear_logp.py`
- `rl_engine/kernels/ops/cuda/loss/linear_logp.py` (SM90 wrapper + chunked backward)
- `csrc/cuda/fused_linear_logp_sm90.cu`, `csrc/ops.cpp`, `setup.py` (SM90 kernel + build)
- `rl_engine/kernels/registry.py`
- `tests/test_linear_logp.py`
- `benchmarks/benchmark_linear_logp.py`
- `docs/design/fused-linear-logp.md`
