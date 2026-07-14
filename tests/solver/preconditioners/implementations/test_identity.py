"""Tests for Identity preconditioner.

Tests the noop Identity preconditioner to ensure:
- Returns a clone of the input (not the same tensor).
- Doesn't modify input values.
"""

from __future__ import annotations

import torch

from torchalg.preconditioners.implementations import Identity


def test_identity_preconditioner_is_noop(residual_vector: torch.Tensor) -> None:
    """Verify identity preconditioner returns a clone of the input.

    Theory:
        Identity preconditioner: M = I, so z = M^{-1}r = r. Should return an
        exact clone without modification.
    """
    precond = Identity()
    z = precond.apply(residual_vector)

    torch.testing.assert_close(z, residual_vector)
    assert z is not residual_vector


def test_identity_preconditioner_does_not_alias_input(residual_vector: torch.Tensor) -> None:
    """Verify mutating the output does not affect the original residual."""
    precond = Identity()
    original_first_entry = residual_vector[0].clone()

    z = precond.apply(residual_vector)
    z[0] = 999.0

    assert residual_vector[0] == original_first_entry
