"""Tests for the dense zero-fill Incomplete Cholesky (IC(0)) preconditioner.

Tests the dense masked IC(0) factorization preconditioner to ensure:
- Correctly factorizes SPD matrices (``L @ L.T ≈ A``).
- Preserves the original sparsity pattern (no fill-in beyond it).
- Raises ``ValueError`` at construction on breakdown (non-SPD or
  near-singular input hitting a non-positive pivot), instead of silently
  propagating ``nan``/``inf`` through ``apply()``.
"""

from __future__ import annotations

import pytest
import torch

from torchalg.preconditioners.implementations import IC0Preconditioner


def test_ic0_factorization_approximates_matrix(tridiagonal_spd_small_torch: torch.Tensor) -> None:
    """Verify IC(0) factorization produces L such that L @ L.T ≈ A.

    Theory:
        IC(0) computes incomplete Cholesky factorization: ``A ≈ L @ L.T``.
        For sparse SPD matrices, the approximation should be reasonable;
        for a tridiagonal matrix, IC(0) preserves the sparsity pattern
        exactly and reproduces ``A`` almost exactly (no fill-in is needed).
    """
    a = tridiagonal_spd_small_torch
    precond = IC0Preconditioner()
    precond.setup(a)
    factor = precond._operator

    reconstructed = factor @ factor.T
    torch.testing.assert_close(reconstructed, a, rtol=1e-2, atol=1e-2)


def test_ic0_preserves_sparsity_pattern(tridiagonal_spd_small_torch: torch.Tensor) -> None:
    """Verify IC(0) factor L has the same sparsity pattern as the lower triangle of A.

    Theory:
        Zero-level incomplete Cholesky maintains the sparsity pattern:
        ``L[i, j] == 0`` wherever ``A[i, j] == 0`` (for the lower
        triangle), so every non-zero stored value is pattern.
    """
    a = tridiagonal_spd_small_torch
    a_lower_mask = torch.tril(a).abs() >= 1e-14

    precond = IC0Preconditioner()
    precond.setup(a)
    factor = precond._operator

    assert torch.equal(factor != 0, a_lower_mask)
    assert torch.count_nonzero(factor) <= torch.count_nonzero(torch.tril(a))


def test_ic0_raises_on_non_spd_matrix(non_spd_matrix_2x2_torch: torch.Tensor) -> None:
    """Verify IC(0) raises ValueError setting up from a non-SPD matrix.

    Note:
        IC(0) is designed for symmetric positive definite matrices. This
        matrix's elimination hits a negative pivot (no real square root
        exists), which is a genuine breakdown - it must raise clearly
        instead of producing a factor full of ``nan``.
    """
    with pytest.raises(ValueError, match="breakdown"):
        IC0Preconditioner().setup(non_spd_matrix_2x2_torch)


def test_ic0_improves_convergence(
    tridiagonal_system_known_solution_torch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    integration_tolerances: tuple[float, float],
) -> None:
    """Verify IC(0) reduces CG iterations vs no preconditioning.

    Theory:
        IC(0) approximates ``A^{-1}`` better than identity for SPD
        matrices, so it should require fewer iterations:
        ``iterations_ic0 < iterations_identity``.
    """
    from torchalg import flexible_cg
    from torchalg.preconditioners.implementations import Identity

    a, b, _ = tridiagonal_system_known_solution_torch
    rtol, atol = integration_tolerances

    _, result_identity = flexible_cg(
        a, b, preconditioner=Identity(), rtol=rtol, atol=atol, maxiter=200
    )
    ic0_precond = IC0Preconditioner()
    ic0_precond.setup(a)
    _, result_ic0 = flexible_cg(a, b, preconditioner=ic0_precond, rtol=rtol, atol=atol, maxiter=200)

    assert result_identity.converged
    assert result_ic0.converged
    assert result_ic0.iterations < result_identity.iterations


def test_ic0_compares_favorably_with_jacobi(
    tridiagonal_system_known_solution_torch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    integration_tolerances: tuple[float, float],
) -> None:
    """Verify IC(0) performs comparably or better than Jacobi for SPD systems.

    Theory:
        IC(0) approximates full Cholesky factorization, which is optimal
        for SPD systems, so it should match or outperform Jacobi diagonal
        scaling: ``iterations_ic0 <= iterations_jacobi``.
    """
    from torchalg import flexible_cg
    from torchalg.preconditioners.implementations import JacobiPreconditioner

    a, b, _ = tridiagonal_system_known_solution_torch
    rtol, atol = integration_tolerances

    jacobi_precond = JacobiPreconditioner()
    jacobi_precond.setup(a)
    _, result_jacobi = flexible_cg(
        a, b, preconditioner=jacobi_precond, rtol=rtol, atol=atol, maxiter=200
    )
    ic0_precond = IC0Preconditioner()
    ic0_precond.setup(a)
    _, result_ic0 = flexible_cg(a, b, preconditioner=ic0_precond, rtol=rtol, atol=atol, maxiter=200)

    assert result_jacobi.converged
    assert result_ic0.converged
    assert result_ic0.iterations <= result_jacobi.iterations


class TestIC0Str:
    def test_str_is_the_class_summary(self) -> None:
        """`str()` reports the (parameterless) configuration."""
        assert str(IC0Preconditioner()) == "IC0()"
