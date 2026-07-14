"""Tests for LinearOperatorPreconditioner wrapper.

Tests the adapter that wraps an arbitrary linear-operator callable
(``Callable[[torch.Tensor], torch.Tensor]``) as a preconditioner. Unlike the
reference (which wraps ``scipy.sparse.linalg.LinearOperator``), ``torchalg``
has no sparse dependency at all (see ``docs/plan.md``'s dense-only
directive) - a plain callable is the abstraction here.
"""

from __future__ import annotations

import pytest
import torch

from torchalg.preconditioners.implementations import LinearOperatorPreconditioner


def test_linear_operator_preconditioner_wraps_callable(torch_dtype: torch.dtype) -> None:
    """Verify LinearOperatorPreconditioner wraps an arbitrary matvec callable."""
    diag = torch.tensor([2.0, 4.0, 1.0], dtype=torch_dtype)

    precond = LinearOperatorPreconditioner(lambda r: r / diag)
    r = torch.tensor([2.0, 4.0, 1.0], dtype=torch_dtype)
    z = precond.apply(r)

    # z = r / diag = [2/2, 4/4, 1/1] = [1, 1, 1]
    torch.testing.assert_close(z, torch.ones_like(r))


def test_linear_operator_preconditioner_reshapes_mismatched_output(
    torch_dtype: torch.dtype,
) -> None:
    """A flattened/misshapen operator output is reshaped to match the residual."""
    precond = LinearOperatorPreconditioner(lambda r: r.reshape(-1, 1).flatten())
    r = torch.tensor([1.0, 2.0, 3.0], dtype=torch_dtype)

    z = precond.apply(r)

    assert z.shape == r.shape


def test_linear_operator_preconditioner_rejects_none_output(torch_dtype: torch.dtype) -> None:
    """A ``None``-returning operator raises ``ValueError``."""
    precond = LinearOperatorPreconditioner(lambda r: None)  # ty: ignore[invalid-argument-type]
    r = torch.tensor([1.0, 2.0, 3.0], dtype=torch_dtype)

    with pytest.raises(ValueError, match="returned None"):
        precond.apply(r)
