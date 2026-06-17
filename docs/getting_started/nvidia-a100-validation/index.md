# NVIDIA A100 Environment Validation

This page records one verified RL-Kernel onboarding run on an NVIDIA A100 cluster node.
It is a concrete reference for CUDA users, not a general hardware support matrix.

The accompanying notebook is available at [a100_benchmark_notes.ipynb](a100_benchmark_notes.ipynb).

## Verified Environment

| Item | Observed value |
| --- | --- |
| GPUs | 8 x NVIDIA A100 80GB PCIe |
| NVIDIA driver | 535.230.02 |
| `nvidia-smi` CUDA version | 12.2 |
| Python | 3.12.3 |
| PyTorch | 2.9.1+cu129 |
| `torch.cuda.is_available()` | `True` |
| PyTorch CUDA devices | 8 |
| Device capability | 8.0 |
| ROCm | unavailable: `rocminfo` not found, `torch.version.hip` is `None` |

## Environment Probe Commands

```bash
nvidia-smi
```

```bash
python - <<'PY'
import sys
print('python', sys.version)
try:
    import torch
    print('torch', torch.__version__)
    print('cuda_available', torch.cuda.is_available())
    print('cuda_version', torch.version.cuda)
    print('hip_version', getattr(torch.version, 'hip', None))
    if torch.cuda.is_available():
        print('device_count', torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print('device', i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
except Exception as exc:
    print('torch_error', repr(exc))
PY
```

## Commands Run

### Dispatch Smoke Tests

```bash
python -m pytest rl_engine/tests/test_dispatch.py -v
```

Result:

```text
3 passed in 9.33s
```

### GRPO Example Smoke Tests

```bash
python -m pytest tests/test_grpo_single_gpu_example.py -v
```

Result:

```text
2 passed in 9.85s
```

### CUDA GRPO Smoke Example

```bash
python examples/grpo_single_gpu.py \
  --device cuda \
  --steps 2 \
  --num-prompts 1 \
  --samples-per-prompt 2 \
  --prompt-len 2 \
  --completion-len 3 \
  --vocab-size 16 \
  --hidden-dim 8
```

Result excerpt:

```text
starting grpo_single_gpu device=cuda backend=FusedLogpGenericOp batch=2x3 active_tokens=5
step=0 loss=0.200001 policy_loss=0.200001 kl=0.000000 train_logp_source=autograd_reference kernel_max_abs_error=2.384186e-07
step=1 loss=0.179493 policy_loss=0.179481 kl=0.001267 train_logp_source=autograd_reference kernel_max_abs_error=2.384186e-07
completed grpo_single_gpu steps=2 device=cuda backend=FusedLogpGenericOp
```

### Documentation Build

```bash
mkdocs build --strict -f mkdocs.yaml
```

Result after adding this page to navigation:

```text
Documentation built in 2.25 seconds
```

The build emitted only plugin warnings that this new page has no git history yet, which
is expected before the page is committed.

## What This Validates

- PyTorch can see and use CUDA on this A100 node.
- RL-Kernel dispatch smoke tests pass in this environment.
- The GRPO example CPU smoke tests pass in this environment.
- The small CUDA GRPO example runs successfully with `FusedLogpGenericOp`.
- The documentation site builds in strict mode after adding this page.

## What This Does Not Claim

- This does not validate AMD ROCm; ROCm was unavailable on this node.
- This does not validate H100, SM90, or TMA-specific fused LogP behavior; A100 is SM80.
- This does not reproduce the full benchmark tables in the project README.
- This does not claim that every CUDA, driver, or PyTorch combination is supported.
- This does not validate strict fused mode with `--require-fused-logp`; run the strict command separately when that is the target.

## Next Checks

For stricter NVIDIA validation, build the extension and require fused dispatch explicitly:

```bash
MAX_JOBS=2 python setup.py build_ext --inplace
python examples/grpo_single_gpu.py \
  --device cuda \
  --require-fused-logp \
  --steps 2 \
  --num-prompts 1 \
  --samples-per-prompt 2 \
  --prompt-len 2 \
  --completion-len 3 \
  --vocab-size 16 \
  --hidden-dim 8
```

Only report that strict path as validated after running it on the target hardware.
