"""Grad-tracking context selection for solver entry points.

Design:
    - Pure function (no state, no config).
    - Isolates the one branch ("does this call need autograd?") so
      ``IterativeSolverBase.solve`` stays a single ``with`` statement.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import torch


def inference_unless_differentiable(differentiable: bool) -> AbstractContextManager[None]:
    """Return the grad-tracking context a solve call should run under.

    Solves default to ``torch.inference_mode()``: most callers run a frozen,
    already-trained preconditioner and never backpropagate through the
    solve, so tracking autograd by default only builds and discards graphs
    nobody uses. Pass ``differentiable=True`` to keep normal autograd
    tracking instead — e.g. to unroll the iteration and train a
    preconditioner end-to-end through it.

    Args:
        differentiable: Whether the caller needs gradients through the solve.

    Returns:
        AbstractContextManager[None]: ``nullcontext()`` when
            ``differentiable`` is ``True``, otherwise ``torch.inference_mode()``.
    """
    return nullcontext() if differentiable else torch.inference_mode()
