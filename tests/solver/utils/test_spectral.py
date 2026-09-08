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
from typing import TYPE_CHECKING

import pytest
import torch

from torchalg.utils.spectral import condition_number, preconditioned_condition_number

if TYPE_CHECKING:
    from collections.abc import Callable


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


class TestPreconditionedConditionNumber:
    """Tests for ``preconditioned_condition_number``.

    The preconditioned operator ``M^-1 A`` is generally non-symmetric even
    for SPD ``M``/``A``, so it cannot go through ``condition_number``
    directly (``eigvalsh`` would silently read only the lower triangle and
    give a wrong answer). These tests check the function against the
    literature definition it implements: ``kappa(M^-1 A) :=
    kappa(M^-1/2 A M^-1/2)`` (Saad, "Iterative Methods for Sparse Linear
    Systems", 2nd ed., Ch. 9) - independently computed here via the
    already-verified ``condition_number`` on the explicit congruence
    transform, using the fact that ``M`` is diagonal in these fixtures so
    ``M^-1/2`` is just an elementwise ``rsqrt``.
    """

    def test_identity_preconditioner_matches_plain_condition_number(
        self,
        heterogeneous_stiffness_matrix: tuple[torch.Tensor, float],
    ) -> None:
        """An identity preconditioner (M=I) reduces to the plain condition number."""
        matrix, true_condition_number = heterogeneous_stiffness_matrix
        estimated = preconditioned_condition_number(matrix, lambda v: v)
        assert estimated == pytest.approx(true_condition_number, rel=1e-6)

    def test_matches_symmetric_congruence_transform_for_jacobi(
        self,
        jacobi_preconditioned_spd_system: tuple[
            torch.Tensor, Callable[[torch.Tensor], torch.Tensor]
        ],
    ) -> None:
        """Matches kappa(D^-1/2 A D^-1/2) computed independently for a Jacobi preconditioner."""
        matrix, apply_jacobi = jacobi_preconditioned_spd_system
        diag_values = torch.diagonal(matrix).clone()
        # Only exact because this fixture's preconditioner is exactly
        # Jacobi (M = diag(A)); apply_jacobi itself is treated as an
        # opaque callable by preconditioned_condition_number.
        inv_sqrt_diag = diag_values.rsqrt()
        congruence_transformed = inv_sqrt_diag.unsqueeze(1) * matrix * inv_sqrt_diag.unsqueeze(0)
        expected = condition_number(congruence_transformed)

        estimated = preconditioned_condition_number(matrix, apply_jacobi)

        assert estimated == pytest.approx(expected, rel=1e-6)

    def test_never_returns_negative_or_nonfinite(
        self,
        heterogeneous_stiffness_matrix: tuple[torch.Tensor, float],
    ) -> None:
        """The estimate is always a finite, positive number for an SPD-preconditioned SPD input."""
        matrix, _ = heterogeneous_stiffness_matrix
        estimated = preconditioned_condition_number(matrix, lambda v: v)
        assert math.isfinite(estimated)
        assert estimated > 0.0
