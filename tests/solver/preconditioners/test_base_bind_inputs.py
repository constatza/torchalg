"""Tests that Preconditioner base has no extra-input interface, and BindableInputs Protocol works."""

from __future__ import annotations

import torch

from torchalg.preconditioners.base import BindableInputs
from torchalg.preconditioners.implementations.jacobi import JacobiPreconditioner


def test_standard_preconditioner_is_not_bindable(jacobi: JacobiPreconditioner) -> None:
    """A plain ``LinearPreconditioner`` does not structurally satisfy ``BindableInputs``."""
    assert not isinstance(jacobi, BindableInputs)


def test_standard_preconditioner_has_no_extra_input_names(jacobi: JacobiPreconditioner) -> None:
    """A plain preconditioner has no ``extra_input_names`` attribute."""
    assert not hasattr(jacobi, "extra_input_names")


def test_standard_preconditioner_has_no_bind_inputs(jacobi: JacobiPreconditioner) -> None:
    """A plain preconditioner has no ``bind_inputs`` method."""
    assert not hasattr(jacobi, "bind_inputs")


def test_standard_preconditioner_apply_unchanged(
    jacobi: JacobiPreconditioner,
    small_diagonal_matrix_residual: torch.Tensor,
) -> None:
    """``apply`` still works normally for a non-bindable preconditioner."""
    result = jacobi.apply(small_diagonal_matrix_residual)
    torch.testing.assert_close(result, torch.ones_like(small_diagonal_matrix_residual))
