"""Tests for the dense ILU(0) preconditioner.

Tests the dense masked incomplete LU factorization preconditioner to
ensure:
- Its combined ``L``/``U`` factorization approximately satisfies
  ``A ≈ LU``, checked against ``scipy.sparse.linalg.spilu`` as an
  independent test-only oracle (see ``docs/plan.md``'s dense-only
  directive: scipy stays a legitimate *test* dependency for exactly this
  kind of ground-truth comparison, even though ``src/`` is scipy-free).
- Accepts dense matrices and produces finite, correctly shaped output.
- Preserves the working dtype.
- On a diagonal matrix, exactly matches Jacobi (ILU(0) is exact for
  diagonal/tridiagonal matrices - no fill-in is ever needed).
"""

from __future__ import annotations

import torch

from torchalg.preconditioners.implementations import ILUPreconditioner


def test_ilu_improves_convergence(
    tridiagonal_system_known_solution_torch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    integration_tolerances: tuple[float, float],
) -> None:
    """Verify ILU(0) reduces CG iterations vs no preconditioning.

    Theory:
        ILU(0) approximates ``A^{-1}`` better than identity, so it should
        require no more iterations: ``iterations_ilu <= iterations_identity``.
    """
    from torchalg import flexible_cg
    from torchalg.preconditioners.implementations import Identity

    a, b, _ = tridiagonal_system_known_solution_torch
    rtol, atol = integration_tolerances

    _, result_identity = flexible_cg(
        a, b, preconditioner=Identity(), rtol=rtol, atol=atol, maxiter=200
    )
    _, result_ilu = flexible_cg(
        a, b, preconditioner=ILUPreconditioner(a), rtol=rtol, atol=atol, maxiter=200
    )

    assert result_identity.converged
    assert result_ilu.converged
    assert result_ilu.iterations <= result_identity.iterations


def test_ilu_factorization_matches_scipy_spilu_oracle(
    tridiagonal_spd_small_torch: torch.Tensor,
) -> None:
    """Verify the dense ILU(0) factorization matches scipy's ``spilu`` oracle.

    Theory:
        For a tridiagonal matrix, Gaussian elimination produces no fill-in
        under natural ordering - the (L, U) factors are exact regardless of
        "incompleteness", so both this dense masked ILU(0) and scipy's
        ``spilu`` should reduce ``M^{-1} @ A`` to (numerically) the
        identity, and closely agree with each other.
    """
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import spilu

    a = tridiagonal_spd_small_torch
    n = a.shape[0]
    a_csc = csc_matrix(a.numpy())
    scipy_ilu = spilu(a_csc)

    precond = ILUPreconditioner(a)
    identity = torch.eye(n, dtype=a.dtype)

    m_inv_a_torch = torch.stack([precond.apply(a @ identity[:, i]) for i in range(n)], dim=1)
    m_inv_a_scipy = torch.stack(
        [
            torch.from_numpy(scipy_ilu.solve((a @ identity[:, i]).numpy())).to(a.dtype)
            for i in range(n)
        ],
        dim=1,
    )

    # Loose threshold, matching the reference's expected ILU approximation
    # quality bound.
    ilu_approximation_threshold = 0.5
    assert torch.linalg.norm(m_inv_a_torch - identity, ord="fro") < ilu_approximation_threshold

    # Tighter, direct oracle comparison: for this tridiagonal fixture, dense
    # ILU(0) and scipy's spilu agree almost to machine precision (verified
    # empirically - both reduce to the exact LU factorization here).
    torch.testing.assert_close(m_inv_a_torch, m_inv_a_scipy, rtol=1e-8, atol=1e-8)


def test_ilu_on_diagonal_matrix_matches_jacobi(
    diagonal_system_known_solution_torch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Verify ILU(0) on a diagonal matrix gives the same result as Jacobi.

    Theory:
        For diagonal matrices, ILU factorization is exact: ``L = I``,
        ``U = D``. Applying ``(LU)^{-1}`` is therefore identical to Jacobi's
        ``D^{-1}``.
    """
    a, b, _ = diagonal_system_known_solution_torch
    precond = ILUPreconditioner(a)

    z = precond.apply(b)
    expected = b / torch.diagonal(a)

    torch.testing.assert_close(z, expected, rtol=1e-12, atol=1e-14)


def test_ilu_preconditioner_from_matrix(
    dense_spd_matrix: torch.Tensor,
    dense_spd_matrix_residual: torch.Tensor,
) -> None:
    """Verify ILU preconditioner computes a usable factorization from a matrix."""
    precond = ILUPreconditioner(dense_spd_matrix)
    z = precond.apply(dense_spd_matrix_residual)

    assert torch.isfinite(z).all()
    assert z.shape == dense_spd_matrix_residual.shape

    # ILU should approximate A^{-1}, so A @ z ≈ r.
    result = dense_spd_matrix @ z
    torch.testing.assert_close(result, dense_spd_matrix_residual, rtol=1e-2, atol=1e-2)


def test_ilu_preconditioner_with_tridiagonal(
    tridiagonal_spd_small_torch: torch.Tensor,
    rhs_ones_small_torch: torch.Tensor,
) -> None:
    """Verify ILU with a larger (10x10) tridiagonal matrix produces finite output."""
    precond = ILUPreconditioner(tridiagonal_spd_small_torch)
    z = precond.apply(rhs_ones_small_torch)

    assert torch.isfinite(z).all()
    assert z.shape == rhs_ones_small_torch.shape


def test_ilu_preconditioner_accepts_dense_matrix(dense_spd_matrix: torch.Tensor) -> None:
    """Test ILUPreconditioner with dense matrix input produces correctly shaped output."""
    precond = ILUPreconditioner(dense_spd_matrix)

    residual = torch.ones(5, dtype=dense_spd_matrix.dtype)
    result = precond.apply(residual)

    assert result.shape == (5,)
    assert result.dtype == torch.float64


def test_ilu_preserves_dtype(
    dense_spd_matrix: torch.Tensor,
    dense_spd_matrix_residual: torch.Tensor,
    torch_dtype: torch.dtype,
) -> None:
    """Test that ILUPreconditioner preserves the working dtype."""
    precond = ILUPreconditioner(dense_spd_matrix)
    result = precond.apply(dense_spd_matrix_residual)

    assert result.dtype == torch_dtype
