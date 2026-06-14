# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import Optional

import torch


class NativeLinearLogpOp:
    """Naive PyTorch reference for fused linear log-prob.

    Materializes the full ``[N, V]`` logits with a single ``F.linear`` and runs
    ``log_softmax`` + ``gather``. This is the obviously-correct oracle the fused
    kernels are validated against (and the baseline the benchmark measures the
    VRAM win against); it is also the CPU / Triton-less fallback. Differentiable
    w.r.t. ``hidden``, ``lm_head_weight`` and ``bias`` through autograd.
    """

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.apply(hidden, lm_head_weight, target_ids, bias)

    def apply(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Selected-token log-prob ``z[t] - logsumexp(z)``, returned in float32."""
        if hidden.shape[:-1] != target_ids.shape:
            raise ValueError(
                f"hidden leading shape {tuple(hidden.shape[:-1])} must match "
                f"target_ids shape {tuple(target_ids.shape)}"
            )
        if lm_head_weight.size(-1) != hidden.size(-1):
            raise ValueError(
                f"hidden dim {hidden.size(-1)} must match lm_head_weight dim "
                f"{lm_head_weight.size(-1)}"
            )

        lead_shape = hidden.shape[:-1]
        hidden_2d = hidden.reshape(-1, hidden.size(-1))
        logits = torch.nn.functional.linear(hidden_2d, lm_head_weight, bias)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        target_1d = target_ids.reshape(-1).to(device=logits.device, dtype=torch.long)
        selected = torch.gather(log_probs, dim=-1, index=target_1d.unsqueeze(1)).squeeze(-1)
        return selected.reshape(lead_shape)
