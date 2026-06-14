# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.kernels.ops.pytorch.loss.linear_logp import NativeLinearLogpOp

try:
    import triton  # noqa: F401

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

requires_triton_cuda = pytest.mark.skipif(
    not (_HAS_TRITON and torch.cuda.is_available()),
    reason="Triton linear log-prob requires a CUDA device and Triton.",
)

# Deliberately non-multiples of the kernel block sizes (32 / 64 / 64).
_N = 40
_D = 80
_V = 300


def _inputs(seed, *, device, dtype=torch.float32, bias=True, lead=None):
    gen = torch.Generator(device=device).manual_seed(seed)
    lead = lead or (_N,)
    hidden = torch.randn(*lead, _D, generator=gen, device=device, dtype=dtype)
    weight = torch.randn(_V, _D, generator=gen, device=device, dtype=dtype)
    bias_t = torch.randn(_V, generator=gen, device=device, dtype=dtype) if bias else None
    target = torch.randint(0, _V, lead, generator=gen, device=device)
    return hidden, weight, target, bias_t


def _manual_reference(hidden, weight, target, bias):
    """The semantic definition: materialize logits, log_softmax, gather."""
    logits = torch.nn.functional.linear(
        hidden.float(), weight.float(), None if bias is None else bias.float()
    )
    logp = torch.log_softmax(logits, dim=-1)
    idx = target.reshape(-1).long()
    sel = logp.reshape(-1, logp.size(-1)).gather(-1, idx.unsqueeze(1)).squeeze(1)
    return sel.reshape(target.shape)


def test_native_matches_manual_reference():
    native = NativeLinearLogpOp()
    hidden, weight, target, bias = _inputs(0, device="cpu")
    out = native(hidden, weight, target, bias)
    ref = _manual_reference(hidden, weight, target, bias)
    assert out.dtype == torch.float32
    assert torch.allclose(out, ref, atol=1e-5)


def test_native_rejects_shape_mismatch():
    native = NativeLinearLogpOp()
    hidden, weight, _, bias = _inputs(0, device="cpu")
    with pytest.raises(ValueError):
        native(hidden, weight, torch.zeros(_N + 1, dtype=torch.long), bias)


@requires_triton_cuda
def test_triton_forward_matches_native_fp32():
    from rl_engine.kernels.ops.triton.loss.linear_logp import TritonLinearLogpOp

    native, trit = NativeLinearLogpOp(), TritonLinearLogpOp()
    hidden, weight, target, bias = _inputs(1, device="cuda")
    ref = native(hidden, weight, target, bias)
    out = trit(hidden, weight, target, bias)
    assert torch.allclose(out, ref, atol=1e-3)


@requires_triton_cuda
def test_triton_forward_matches_native_bf16():
    from rl_engine.kernels.ops.triton.loss.linear_logp import TritonLinearLogpOp

    native, trit = NativeLinearLogpOp(), TritonLinearLogpOp()
    hidden, weight, target, bias = _inputs(2, device="cuda", dtype=torch.bfloat16)
    # The kernel accumulates in fp32, so the oracle uses the fp32-upcast inputs.
    ref = native(hidden.float(), weight.float(), target, bias.float())
    out = trit(hidden, weight, target, bias)
    assert torch.allclose(out, ref, atol=2e-2)


@requires_triton_cuda
@pytest.mark.parametrize("use_bias", [True, False])
def test_triton_backward_matches_native(use_bias):
    from rl_engine.kernels.ops.triton.loss.linear_logp import TritonLinearLogpOp

    native, trit = NativeLinearLogpOp(), TritonLinearLogpOp()
    hidden, weight, target, bias = _inputs(3, device="cuda", bias=use_bias)
    grad_out = torch.randn(_N, device="cuda")

    def run(op, h, w, b):
        h = h.detach().clone().requires_grad_(True)
        w = w.detach().clone().requires_grad_(True)
        b = b.detach().clone().requires_grad_(True) if b is not None else None
        op(h, w, target, b).backward(grad_out)
        return h.grad, w.grad, (b.grad if b is not None else None)

    th, tw, tb = run(trit, hidden, weight, bias)
    nh, nw, nb = run(native, hidden, weight, bias)
    assert torch.allclose(th, nh, atol=2e-3)
    assert torch.allclose(tw, nw, atol=2e-3)
    if use_bias:
        assert torch.allclose(tb, nb, atol=2e-3)


@requires_triton_cuda
def test_triton_gradients_flow_to_inputs_only():
    from rl_engine.kernels.ops.triton.loss.linear_logp import TritonLinearLogpOp

    trit = TritonLinearLogpOp()
    hidden, weight, target, bias = _inputs(4, device="cuda")
    hidden = hidden.requires_grad_(True)
    weight = weight.requires_grad_(True)
    bias = bias.requires_grad_(True)
    trit(hidden, weight, target, bias).sum().backward()
    assert hidden.grad is not None and weight.grad is not None and bias.grad is not None
    assert target.grad is None  # integer targets are non-differentiable


@requires_triton_cuda
def test_triton_preserves_leading_shape():
    from rl_engine.kernels.ops.triton.loss.linear_logp import TritonLinearLogpOp

    native, trit = NativeLinearLogpOp(), TritonLinearLogpOp()
    hidden, weight, target, bias = _inputs(5, device="cuda", lead=(4, 7))
    out = trit(hidden, weight, target, bias)
    assert out.shape == (4, 7)
    assert torch.allclose(out, native(hidden, weight, target, bias), atol=1e-3)


@requires_triton_cuda
def test_triton_large_vocab_smoke():
    from rl_engine.kernels.ops.triton.loss.linear_logp import TritonLinearLogpOp

    trit = TritonLinearLogpOp()
    hidden = torch.randn(8, 64, device="cuda")
    weight = torch.randn(50257, 64, device="cuda")
    target = torch.randint(0, 50257, (8,), device="cuda")
    out = trit(hidden, weight, target)
    assert out.shape == (8,) and torch.isfinite(out).all()


def test_registry_dispatch_matches_native():
    from rl_engine.kernels.registry import kernel_registry
    from rl_engine.platforms.device import device_ctx

    op = kernel_registry.get_op("linear_logp")
    device = device_ctx.device if device_ctx.device_type == "cuda" else "cpu"
    hidden, weight, target, bias = _inputs(6, device=device)
    out = op(hidden, weight, target, bias)
    ref = NativeLinearLogpOp()(hidden, weight, target, bias)
    assert torch.allclose(out.cpu(), ref.cpu(), atol=1e-3)
