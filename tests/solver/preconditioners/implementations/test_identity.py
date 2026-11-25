"""Tests for Identity preconditioner.

Tests the noop Identity preconditioner to ensure:
- Returns the input tensor itself (``z = r``, no copy).
- Doesn't modify input values.
"""

from __future__ import annotations

import torch

from torchalg.preconditioners.implementations import Identity


def test_identity_preconditioner_returns_the_input_unchanged(
    residual_vector: torch.Tensor,
) -> None:
    """Verify identity preconditioner returns the input tensor itself.

    Theory:
        Identity preconditioner: M = I, so z = M^{-1}r = r. No solver writes
        ``r`` or ``z`` in place, so a copy would only cost a memory pass.
    """
    expected = residual_vector.clone()

    z = Identity().apply(residual_vector)

    assert z is residual_vector
    assert torch.equal(z, expected)
