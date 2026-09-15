"""Tests for the dense zero-fill Incomplete Cholesky (IC(0)) preconditioner.

Tests the dense masked IC(0) factorization preconditioner to ensure:
- Correctly factorizes SPD matrices (``L @ L.T ≈ A``).
- Preserves the original sparsity pattern (no fill-in beyond it).
- Handles the ``threshold`` drop-tolerance parameter.
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
    precond = IC0Preconditioner(a)
    factor = precond._operator

    reconstructed = factor @ factor.T
    torch.testing.assert_close(reconstructed, a, rtol=1e-2, atol=1e-2)


def test_ic0_preserves_sparsity_pattern(tridiagonal_spd_small_torch: torch.Tensor) -> None:
    """Verify IC(0) factor L has the same sparsity pattern as the lower triangle of A.

    Theory:
        Zero-level incomplete Cholesky maintains the sparsity pattern:
        ``L[i, j] == 0`` wherever ``A[i, j] == 0`` (for the lower
        triangle). With ``threshold=0``, this holds exactly. With the
        default threshold, ``L`` can be sparser (small entries dropped).
    """
    a = tridiagonal_spd_small_torch
    a_lower_mask = torch.tril(a).abs() >= 1e-14

    precond_strict = IC0Preconditioner(a, threshold=0.0)
    factor_strict = precond_strict._operator
    factor_strict_mask = factor_strict != 0

    assert torch.equal(factor_strict_mask, a_lower_mask)

    precond_default = IC0Preconditioner(a)
    factor_default = precond_default._operator
    nnz_factor_default = torch.count_nonzero(factor_default)
    nnz_a_lower = torch.count_nonzero(torch.tril(a))

    assert nnz_factor_default <= nnz_a_lower


def test_ic0_raises_on_non_spd_matrix(non_spd_matrix_2x2_torch: torch.Tensor) -> None:
    """Verify IC(0) raises ValueError constructing from a non-SPD matrix.

    Note:
        IC(0) is designed for symmetric positive definite matrices. This
        matrix's elimination hits a negative pivot (no real square root
        exists), which is a genuine breakdown - it must raise clearly
        instead of producing a factor full of ``nan``.
    """
    with pytest.raises(ValueError, match="breakdown"):
        IC0Preconditioner(non_spd_matrix_2x2_torch)


def test_ic0_raises_on_near_singular_matrix(near_singular_matrix_torch: torch.Tensor) -> None:
    """Verify IC(0) raises ValueError on a matrix whose pivot the threshold drops to zero.

    Note:
        The near-zero diagonal entry (``1e-15``) falls below the default
        drop tolerance and is excluded from the sparsity pattern entirely,
        leaving a zero pivot - a non-positive pivot is a breakdown by the
        same definition as a negative one (Saad, Sec. 10.3), so this must
        raise rather than silently produce a singular, unusable factor.
    """
    with pytest.raises(ValueError, match="breakdown"):
        IC0Preconditioner(near_singular_matrix_torch)


def test_ic0_threshold_drops_small_entries(
    tridiagonal_spd_small_perturbed_torch: torch.Tensor,
) -> None:
    """Verify the threshold parameter drops small entries.

    Theory:
        Threshold parameter controls sparsity: a smaller threshold means
        more fill-in survives (better approximation); a larger threshold
        drops more small entries (sparser, less accurate).
    """
    a = tridiagonal_spd_small_perturbed_torch

    ic0_strict = IC0Preconditioner(a, threshold=1e-16)
    nnz_strict = torch.count_nonzero(ic0_strict._operator)

    ic0_aggressive = IC0Preconditioner(a, threshold=1e-10)
    nnz_aggressive = torch.count_nonzero(ic0_aggressive._operator)

    assert nnz_aggressive <= nnz_strict


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
    _, result_ic0 = flexible_cg(
        a, b, preconditioner=IC0Preconditioner(a), rtol=rtol, atol=atol, maxiter=200
    )

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

    _, result_jacobi = flexible_cg(
        a, b, preconditioner=JacobiPreconditioner(a), rtol=rtol, atol=atol, maxiter=200
    )
    _, result_ic0 = flexible_cg(
        a, b, preconditioner=IC0Preconditioner(a), rtol=rtol, atol=atol, maxiter=200
    )

    assert result_jacobi.converged
    assert result_ic0.converged
    assert result_ic0.iterations <= result_jacobi.iterations
