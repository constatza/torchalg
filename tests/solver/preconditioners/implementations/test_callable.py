"""Tests for CallablePreconditioner wrapper.

Tests the adapter that wraps arbitrary callables as preconditioners.
"""

from __future__ import annotations

import torch

from torchalg.preconditioners.implementations import CallablePreconditioner


def test_callable_preconditioner_wraps_function(torch_dtype: torch.dtype) -> None:
    """Verify CallablePreconditioner wraps an arbitrary function."""

    def custom_precond(r: torch.Tensor) -> torch.Tensor:
        return r * 0.5  # Simple damping.

    precond = CallablePreconditioner(custom_precond)
    r = torch.tensor([2.0, 4.0, 6.0], dtype=torch_dtype)
    z = precond.apply(r)

    torch.testing.assert_close(z, torch.tensor([1.0, 2.0, 3.0], dtype=torch_dtype))


def test_callable_preconditioner_requires_flexible_cg(torch_dtype: torch.dtype) -> None:
    """A ``CallablePreconditioner`` is non-linear by construction - requires Flexible CG."""
    precond = CallablePreconditioner(lambda r: r)
    assert precond.requires_flexible_cg is True
