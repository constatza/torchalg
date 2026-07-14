"""Tests for the ICholesky preconditioner.

Tests the externally-supplied-factor Incomplete Cholesky preconditioner to
ensure:
- Acts as an exact inverse when given an exact Cholesky factor.
- Preserves shapes for both single-vector and batched residual input.
"""

from __future__ import annotations

import torch

from torchalg.preconditioners.implementations import ICholeskyPreconditioner


def test_icholesky_preconditioner_exact_inverse(
    random_spd_small_torch: torch.Tensor,
    cholesky_factor_random_spd_small_torch: torch.Tensor,
    rhs_random_small_torch: torch.Tensor,
) -> None:
    """Verify ICholeskyPreconditioner solves Mz=r exactly when given the exact L.

    Since the supplied factor is the true Cholesky factor of ``A``
    (``M = L @ L.T = A``), the preconditioner's ``apply`` is exactly
    ``A^{-1}``: ``A @ z`` should reconstruct ``r``.
    """
    precond = ICholeskyPreconditioner(cholesky_factor_random_spd_small_torch)

    z = precond.apply(rhs_random_small_torch)
    reconstructed_r = random_spd_small_torch @ z

    torch.testing.assert_close(reconstructed_r, rhs_random_small_torch, rtol=1e-10, atol=1e-10)


def test_icholesky_preconditioner_shapes(
    cholesky_factor_identity_5: torch.Tensor,
    ones_residual_5: torch.Tensor,
    ones_residual_5x2: torch.Tensor,
) -> None:
    """Verify shape handling for both single-vector and batched residual input."""
    precond = ICholeskyPreconditioner(cholesky_factor_identity_5)

    z = precond.apply(ones_residual_5)
    assert z.shape == ones_residual_5.shape

    z_batch = precond.apply(ones_residual_5x2)
    assert z_batch.shape == ones_residual_5x2.shape
    # Since the factorized matrix M = L @ L.T = I here, z_batch == r_batch.
    torch.testing.assert_close(z_batch, ones_residual_5x2)
