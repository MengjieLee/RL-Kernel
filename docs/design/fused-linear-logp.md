# Design: Fused Linear LogProb (no logit materialization) + fused backward

## Motivation

Every current log-prob path — `logp`, `ratio_kl`, `grpo_loss` — takes the
**logits** `[B, S, V]` as input, so the `[B, S, V]` tensor is already resident in
HBM before any fusion happens. For large-vocabulary models that tensor dominates
memory: at `V≈150k`, `B·S≈8k` in bf16 it is ~2.4 GB for the forward activation
alone, plus the same again for its gradient.

This op fuses the **LM-head projection itself** into the log-prob reduction:

```
logits = hidden @ Wᵀ (+ b)      # [N, V] — NEVER materialized
logp[n] = logits[n, t[n]] − logsumexp(logits[n, :])
```

The forward streams the vocab in blocks (block matmul → online softmax), keeping
only per-row scalars. The backward recomputes the logit tiles from `hidden`/`W`
instead of storing them — trading a second matmul for the `[N, V]` storage. The
freed memory buys larger batches / longer CoT.

This is the "cut cross-entropy" / fused-linear-cross-entropy pattern, specialized
to **selected-token log-prob** (the RL training quantity) rather than mean CE.

## Scope

- **New op type** `linear_logp` — does **not** replace `logp` (still useful when
  logits already exist). It is the differentiable hidden→logp primitive.
- **In scope**: forward + backward w.r.t. `hidden` and `lm_head_weight` (and bias);
  Triton, CUDA generic, CUDA SM90 TMA, ROCm backends; native PyTorch reference;
  registry wiring; tests; benchmarks; docs.
- **Out of scope (follow-ups)**: rewiring `ratio_kl`/`grpo_loss` to consume hidden
  states directly; tensor/sequence-parallel vocab sharding; fp8 weights.

## Math

Per row `n`, target token `t = t[n]`, logits `z = Hₙ Wᵀ (+ b) ∈ ℝ^V`:

**Forward**
```
lse = logsumexp(z);   logp = z[t] − lse
```
Computed by streaming vocab blocks `Vblk`: each block does a `[Nblk, D]·[D, Vblk]`
matmul tile, folds the tile into an online-softmax state `(m, s)` (running max,
rescaled sum), and captures `z[t]` when `t` lands in the block. Saved for backward:
**only** `lse` (or `(m,s)`) and the gathered `z[t]` — per-row scalars, no `[N,V]`.

**Backward** — given `g = dL/dlogp ∈ ℝ^N`:
```
dL/dz[v]  = g · (1[v==t] − p[v]),   p = softmax(z)            # never stored; p recomputed
dL/dHₙ    = g · (W[t] − Σ_v p[v]·W[v])                         # = (dL/dz) @ W
dL/dW[v] += g · (1[v==t] − p[v]) · Hₙ      (reduce over n)
dL/db[v] += g · (1[v==t] − p[v])           (reduce over n)
```
The backward re-streams vocab blocks, recomputes `z_blk = Hₙ W_blkᵀ`, then
`p_blk = exp(z_blk − lse)`, and accumulates the three gradients. `dL/dW` and
`dL/db` are reductions **over the token dimension** → handled by atomic-add into
the `[V,D]`/`[V]` grad buffers (or a token-blocked two-pass to avoid atomics).

All reductions/accumulation in **fp32**; inputs may be bf16/fp16. The `c·(onehot −
softmax)` shape is identical to the existing `_ratio_kl_bwd_kernel` — the only new
piece is the surrounding matmul.

## Public API

```python
linear_logp = kernel_registry.get_op("linear_logp")

logp = linear_logp(
    hidden,        # [B, S, D] or [N, D]  (differentiable)
    lm_head_weight,# [V, D]               (differentiable)
    target_ids,    # [B, S] or [N]        int
    bias=None,     # [V] optional         (differentiable)
)                  # -> [B, S] or [N], differentiable w.r.t. hidden, weight, bias
```

Wrapped in a `torch.autograd.Function` so it is a drop-in differentiable node.
Mask handling follows the existing convention (caller zeroes masked positions, or
pass an ignore-index for `target_ids`).

## Tensor contract

| Arg | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| `hidden` | `[N, D]` | bf16/fp16/fp32 | contiguous; leading dims flattened |
| `lm_head_weight` | `[V, D]` | bf16/fp16/fp32 | contiguous (row-major over V) |
| `target_ids` | `[N]` | int32/int64 | in `[0, V)`; optional ignore-index |
| `bias` | `[V]` | float | optional |
| `logp` (out) | `[N]` | fp32 | `z[t] − logsumexp(z)` |

## Implementation phases

Ordering (per request): **Triton → CUDA generic → CUDA SM90 TMA → ROCm.** Each
phase is independently shippable and gated on matching the prior phase's reference
to tolerance.

### Phase 1 — Triton (portable baseline + tolerance target)

The semantic reference that CUDA/ROCm must match.

- `rl_engine/kernels/ops/pytorch/loss/linear_logp.py` — `NativeLinearLogpOp`:
  **naive** pure-PyTorch reference — a single `F.linear` (materializes the full
  `[N, V]` logits) → `log_softmax` → `gather`, differentiable via autograd. No
  chunking: this is the straightforward, obviously-correct oracle that the fused
  kernels are validated against (and the baseline the benchmark measures the VRAM
  win against). Also the CPU / Triton-less fallback.
- `rl_engine/kernels/ops/triton/loss/linear_logp.py`:
  - `_linear_logp_fwd_kernel`: program per token-block; loop vocab blocks, `tl.dot`
    for the `[Nblk,D]·[D,Vblk]` tile, online-softmax state in registers; store
    `logp`, `lse`, gathered `z[t]`.
  - `_linear_logp_bwd_kernel`: re-stream vocab, recompute `p_blk`, accumulate
    `dH` (per token-block, local) and `dW`/`db` (atomic-add into global buffers).
  - `_LinearLogpFunction(autograd.Function)` + `TritonLinearLogpOp`.
- Tests `tests/test_linear_logp.py`: native-vs-`F.linear→log_softmax→gather`;
  Triton fwd vs native; Triton bwd vs autograd (gradcheck-style on `hidden`,
  `weight`, `bias`) at `atol=1e-3` bf16 / tighter fp32; ignore-index; large `V`
  (50257, 131072) memory-flat check; registry dispatch.
- Benchmark `benchmarks/benchmark_linear_logp.py`: sweep `V`; report fwd / fwd+bwd
  latency and **peak VRAM vs the `F.linear`+`log_softmax` baseline** (the headline
  number — should be flat in `V`).
- Docs `docs/operators/linear-logp.md` (+ `.nav.yml`, `operators/README.md`).

**Acceptance**: Triton matches native within tolerance; benchmark shows peak VRAM
independent of `V` and a clear win vs the materializing baseline.

### Phase 2 — CUDA generic fallback (`fused_linear_logp_kernel.cu`)

Portable CUDA (no arch intrinsics) for all SM≥70.

- `csrc/fused_linear_logp_kernel.cu`: forward + backward. Block per token-tile;
  `cp.async` double-buffer `W` tiles into shared memory; FMA or `mma.sync` matmul;
  online-softmax state in registers (reuse `LogSumExpState` / `merge_logsumexp_state`
  and `blockReduce*` from `fused_logp_kernel.cu`); fp32 accumulate. Backward
  recomputes tiles; atomic-add into `dW`/`db`.
- Bind in `csrc/ops.cpp` under the existing `KERNEL_ALIGN_WITH_CUDA` guard:
  `fused_linear_logp_forward`, `fused_linear_logp_backward`.
- `setup.py`: add `csrc/fused_linear_logp_kernel.cu` to `cuda_sources`; env-var
  block-size toggles mirroring the `FUSED_LOGP_*` macros.
- `rl_engine/kernels/ops/cuda/loss/linear_logp.py` — `FusedLinearLogpGenericOp`
  wrapping fwd/bwd in an `autograd.Function`.

**Acceptance**: matches Triton/native to tolerance; ≥ parity with Triton latency
on the generic path; same flat-VRAM property.

### Phase 3 — CUDA SM90 TMA/WGMMA (`fused_linear_logp_sm90.cu`)

Advanced streaming for Hopper/Blackwell-class (`cc ∈ {9,10,12}`).

- `csrc/cuda/fused_linear_logp_sm90.cu`: TMA (`cuTensorMapEncodeTiled`) bulk loads
  of `H` and `W` tiles into shared memory; WGMMA (`wgmma.mma_async`) for the
  matmul; warp-specialized producer (TMA) / consumer (online softmax) with
  `mbarrier`; keep state in registers. Backward symmetric with tile recompute.
  **Heed the prior TMA lessons** (see issue #91 / the SM120 work): box inner dim
  ≤ 256 elems, CUDA 12.9+ for `cuda::maximum`, arch `compute_90a`/`120a`.
- Gate behind `KERNEL_ALIGN_FORCE_SM90` in `setup.py` (arch-specific gencode), bind
  under `KERNEL_ALIGN_WITH_SM90` in `ops.cpp`.
- `FusedLinearLogpSM90Op` in the CUDA wrapper; registry auto-prioritizes it in
  `_adjust_priority_for_hardware` when `hasattr(_C, "fused_linear_logp_sm90")` and
  `cc_major in (9,10,12)` (same pattern as `fused_logp_sm90`).

**Acceptance**: matches reference to tolerance on SM90+; measurable speedup over
the generic CUDA path; builds cleanly on SM120/CUDA 13 (the dev box) and is cleanly
disabled where unsupported.

### Phase 4 — ROCm / CDNA

- **Free coverage first**: the Phase-1 Triton kernel compiles on ROCm — register
  `TRITON_LINEAR_LOGP` in the `rocm` priority map so ROCm works from Phase 1.
- **Native HIP** (`csrc/rocm/...` or HIP-ified): wavefront=64-aware reductions and
  LDS layouts; replace TMA with **manual double-buffering** into LDS; CDNA MFMA for
  the matmul where available. Build via the ROCm/HIP path in `setup.py`.

**Acceptance**: ROCm Triton matches reference; native HIP ≥ Triton on CDNA.

## Registry wiring

Add to `OpBackend`: `TRITON_LINEAR_LOGP`, `PYTORCH_LINEAR_LOGP`,
`CUDA_FUSED_LINEAR_LOGP_GENERIC`, `CUDA_FUSED_LINEAR_LOGP_SM90`. Add `linear_logp`
to each platform map:

```
cuda:  [CUDA_FUSED_LINEAR_LOGP_GENERIC, TRITON_LINEAR_LOGP, PYTORCH_LINEAR_LOGP]
       (+ CUDA_FUSED_LINEAR_LOGP_SM90 inserted at front by _adjust_priority_for_hardware)
rocm:  [TRITON_LINEAR_LOGP, PYTORCH_LINEAR_LOGP]   (+ native HIP once landed)
cpu:   [PYTORCH_LINEAR_LOGP]
```

## Numerics & testing strategy

- The **Triton kernel is the tolerance target**; native PyTorch is the correctness
  oracle. Every native backend is gradchecked against `F.linear → log_softmax →
  gather` autograd.
- fp32 accumulation throughout; bf16/fp16 I/O. Tolerance `atol≈1e-3` (bf16),
  tighter for fp32.
- Edge cases: `V` not a multiple of the block; ignore-index targets; single-token
  rows; very large `V` (memory-flat assertion).

## Risks / open questions

- **`dW` atomics**: atomic-add into `[V,D]` may contend at small `V`. Fallback is a
  token-blocked two-pass (deterministic, atomic-free) — decide per-backend by
  benchmark. Determinism flag may be wanted for reproducible RL runs.
- **bf16 weight gradient precision**: accumulate `dW` in fp32, cast on store.
- **Backward recompute cost**: ~2× forward matmul FLOPs in backward; net win is the
  memory, not FLOPs — benchmark must report both so the tradeoff is explicit.
- **TMA portability**: the SM90 path repeats the constraints that bit us before
  (box dims, CUDA version, arch flags); keep it default-off and feature-detected.

## Relevant files (to add / touch)

- `rl_engine/kernels/ops/{pytorch,triton,cuda}/loss/linear_logp.py`
- `csrc/fused_linear_logp_kernel.cu`, `csrc/cuda/fused_linear_logp_sm90.cu`
- `csrc/ops.cpp`, `setup.py`, `rl_engine/kernels/registry.py`
- `tests/test_linear_logp.py`, `benchmarks/benchmark_linear_logp.py`
- `docs/operators/linear-logp.md`, `docs/.nav.yml`, `docs/operators/README.md`
