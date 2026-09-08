"""Tests for ``torchalg.utils.spectral``.

Exercises ``condition_number`` against hand-verifiable eigenvalue oracles
(SPD matrices built via an orthogonal similarity transform of a known
diagonal, so the true condition number is exact regardless of the random
basis), including the ill-conditioned/near-singular-cluster regime that
naive power-iteration estimators are known to get wrong (see the module
docstring in ``torchalg.utils.spectral`` for the reference this replaces).
"""

from __future__ import annotations

import math

import pytest
import torch

from torchalg.utils.spectral import condition_number


class TestConditionNumber:
    """Tests for ``condition_number``."""

    def test_matches_known_condition_number(
        self,
        well_conditioned_spd_matrix: torch.Tensor,
    ) -> None:
        """A matrix with exact condition number 4.0 is recovered exactly."""
        assert condition_number(well_conditioned_spd_matrix) == pytest.approx(4.0, rel=1e-10)

    def test_identity_has_condition_number_one(self, torch_dtype: torch.dtype) -> None:
        """The identity matrix is perfectly conditioned."""
        identity = torch.eye(5, dtype=torch_dtype)
        assert condition_number(identity) == pytest.approx(1.0, rel=1e-10)

    def test_matches_true_condition_number_for_heterogeneous_stiffness(
        self,
        heterogeneous_stiffness_matrix: tuple[torch.Tensor, float],
    ) -> None:
        """Wide eigenvalue spread (stiff cube / soft inclusion) is estimated exactly.

        This is the regime where the folded-spectrum shifted-power-iteration
        estimator (the ``neuralls`` reference's ``compute_condition_numbers``)
        catastrophically fails: clustered near-zero eigenvalues destroy its
        convergence and its final ``shift - power_iterate(...)`` subtraction
        loses all precision to cancellation, occasionally emitting a negative
        "eigenvalue". The exact eigendecomposition used here has no such
        failure mode.
        """
        matrix, true_condition_number = heterogeneous_stiffness_matrix
        estimated = condition_number(matrix)
        assert estimated > 0.0
        assert estimated == pytest.approx(true_condition_number, rel=1e-8)

    def test_never_returns_negative_or_nonfinite(
        self,
        heterogeneous_stiffness_matrix: tuple[torch.Tensor, float],
    ) -> None:
        """The estimate is always a finite, positive number for an SPD input."""
        matrix, _ = heterogeneous_stiffness_matrix
        estimated = condition_number(matrix)
        assert math.isfinite(estimated)
        assert estimated > 0.0

    def test_singular_matrix_returns_infinity(self, singular_matrix: torch.Tensor) -> None:
        """A matrix with an exact zero eigenvalue has an infinite condition number."""
        assert condition_number(singular_matrix) == math.inf

    def test_raises_on_non_square_matrix(self, torch_dtype: torch.dtype) -> None:
        """A non-square matrix has no well-defined eigenvalue condition number."""
        with pytest.raises(ValueError, match="square"):
            condition_number(torch.ones((2, 3), dtype=torch_dtype))
