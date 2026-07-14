"""Tests for ``torchalg.strategies.orthogonalization``."""

from __future__ import annotations

import pytest
import torch

from torchalg.strategies.orthogonalization import (
    ModifiedGramSchmidt,
    PeriodicRestartOrthogonalization,
    TruncatedGramSchmidt,
    create_fcg_orthogonalization,
)


def test_truncated_gram_schmidt_removes_a_conjugate_component(
    orthogonalization_probe: torch.Tensor,
    previous_direction: torch.Tensor,
    previous_matrix_product: torch.Tensor,
) -> None:
    """Truncated Gram-Schmidt enforces d_i^T A d_j = 0 for the stored direction."""
    result, report = TruncatedGramSchmidt(window_size=2).orthogonalize(
        vector=orthogonalization_probe,
        d_vectors=(previous_direction,),
        q_vectors=(previous_matrix_product,),
    )

    assert torch.allclose(result, torch.tensor([0.0, 1.0], dtype=result.dtype))
    assert report.coefficients == (2.0,)
    assert report.breakdown is False


def test_modified_gram_schmidt_uses_all_history(
    orthogonalization_probe: torch.Tensor,
    previous_direction: torch.Tensor,
    previous_matrix_product: torch.Tensor,
) -> None:
    """Modified Gram-Schmidt advertises unlimited history and orthogonalizes."""
    strategy = ModifiedGramSchmidt()
    result, report = strategy.orthogonalize(
        vector=orthogonalization_probe,
        d_vectors=(previous_direction,),
        q_vectors=(previous_matrix_product,),
    )

    assert strategy.window_size is None
    assert torch.allclose(result, torch.tensor([0.0, 1.0], dtype=result.dtype))
    assert report.breakdown is False


def test_create_fcg_orthogonalization_handles_full_window_sentinel() -> None:
    """The factory maps m_max=-1 to an unlimited periodic-restart strategy."""
    strategy = create_fcg_orthogonalization(m_max=-1)
    assert isinstance(strategy, PeriodicRestartOrthogonalization)
    assert strategy.window_size is None


def test_create_fcg_orthogonalization_rejects_zero_window() -> None:
    """A zero orthogonalization window is invalid."""
    with pytest.raises(ValueError, match="m_max cannot be 0"):
        create_fcg_orthogonalization(m_max=0)
