"""Tests for Jacobi preconditioner.

Tests diagonal scaling preconditioner to ensure:
- Correctly computes z_i = r_i / A_ii.
- Preserves signs correctly.
- Handles near-zero diagonal elements.
- Registers its precomputed inverse diagonal as an ``nn.Module`` buffer that
  actually moves under ``.to(dtype=...)``/``.to(device=...)`` - the whole
  point of choosing ``nn.Module`` for this class (see
  ``src/torchalg/preconditioners/implementations/jacobi.py``).
"""

from __future__ import annotations

import pytest
import torch

from torchalg.preconditioners.implementations import JacobiPreconditioner


def test_jacobi_preserves_signs(
    mixed_sign_diagonal_matrix: torch.Tensor,
    mixed_sign_residual: torch.Tensor,
) -> None:
    """Verify Jacobi preconditioner preserves sign of residual components.

    Theory:
        Jacobi: z_i = r_i / A_ii. For positive diagonal A_ii > 0,
        sign(z_i) = sign(r_i). For negative diagonal A_ii < 0, sign(z_i) =
        sign(r_i) (inverted twice). This test ensures Jacobi handles
        negative diagonal elements correctly.
    """
    precond = JacobiPreconditioner(mixed_sign_diagonal_matrix)
    z = precond.apply(mixed_sign_residual)

    expected = mixed_sign_residual / torch.diagonal(mixed_sign_diagonal_matrix)
    torch.testing.assert_close(z, expected, rtol=1e-14, atol=0.0)
    assert torch.equal(torch.sign(z), torch.sign(expected))


def test_jacobi_correctness_on_well_conditioned_diagonal(
    well_conditioned_matrix: torch.Tensor,
    residual_vector: torch.Tensor,
) -> None:
    """Verify Jacobi preconditioner computes z_i = r_i / A_ii correctly.

    Theory:
        Jacobi preconditioner: z = diag(A)^{-1} @ r. For a diagonal matrix,
        this simplifies to element-wise division.
    """
    precond = JacobiPreconditioner(well_conditioned_matrix)
    z = precond.apply(residual_vector)

    expected = residual_vector / torch.diagonal(well_conditioned_matrix)
    torch.testing.assert_close(z, expected, rtol=1e-14, atol=1e-14)


def test_jacobi_preconditioner_from_matrix(torch_dtype: torch.dtype) -> None:
    """Verify Jacobi preconditioner computes diagonal inverse from matrix."""
    matrix = torch.diag(torch.tensor([2.0, 4.0, 1.0], dtype=torch_dtype))
    precond = JacobiPreconditioner(matrix)

    r = torch.diagonal(matrix).clone()
    z = precond.apply(r)

    # z_i = r_i / A_ii = [2/2, 4/4, 1/1] = [1.0, 1.0, 1.0]
    torch.testing.assert_close(z, torch.ones_like(r))


def test_jacobi_preconditioner_with_tridiagonal(
    tridiagonal_spd_small_torch: torch.Tensor,
    rhs_ones_small_torch: torch.Tensor,
) -> None:
    """Verify Jacobi preconditioner from an actual SPD matrix."""
    precond = JacobiPreconditioner(tridiagonal_spd_small_torch)
    z = precond.apply(rhs_ones_small_torch)

    expected = rhs_ones_small_torch / torch.diagonal(tridiagonal_spd_small_torch)
    torch.testing.assert_close(z, expected)


def test_jacobi_handles_near_zero_diagonal(torch_dtype: torch.dtype) -> None:
    """Verify Jacobi preconditioner protects against near-zero diagonals."""
    matrix = torch.diag(torch.tensor([2.0, 1e-15, 1.0], dtype=torch_dtype))
    precond = JacobiPreconditioner(matrix)

    r = torch.tensor([2.0, 4.0, 1.0], dtype=torch_dtype)
    z = precond.apply(r)

    # Near-zero diagonal is treated as 1.0: [2/2, 4/1, 1/1] = [1.0, 4.0, 1.0]
    expected = torch.tensor([1.0, 4.0, 1.0], dtype=torch_dtype)
    torch.testing.assert_close(z, expected)


def test_jacobi_preserves_dtype(
    well_conditioned_matrix: torch.Tensor,
    residual_vector: torch.Tensor,
    torch_dtype: torch.dtype,
) -> None:
    """Verify JacobiPreconditioner preserves the working dtype."""
    precond = JacobiPreconditioner(well_conditioned_matrix)
    result = precond.apply(residual_vector)

    assert result.dtype == torch_dtype


class TestJacobiIsNnModule:
    """Regression tests proving ``JacobiPreconditioner``'s ``nn.Module`` buffer actually moves.

    ``JacobiPreconditioner`` is the first preconditioner to multiply-inherit
    ``nn.Module`` specifically so its precomputed ``inv_diag`` tensor can
    ride ``nn.Module``'s battle-tested ``.to(device/dtype)`` propagation
    (see ``docs/plan.md``'s ``nn.Module`` architecture decision). These
    tests exist to prove that payoff is real, not just that ``.apply()``
    gives the right answer.
    """

    def test_is_nn_module_instance(self, well_conditioned_matrix: torch.Tensor) -> None:
        """``JacobiPreconditioner`` is a genuine ``nn.Module``."""
        precond = JacobiPreconditioner(well_conditioned_matrix)
        assert isinstance(precond, torch.nn.Module)

    def test_inv_diag_is_registered_buffer(self, well_conditioned_matrix: torch.Tensor) -> None:
        """``inv_diag`` is registered via ``register_buffer``, not a plain attribute."""
        precond = JacobiPreconditioner(well_conditioned_matrix)
        buffer_names = dict(precond.named_buffers())
        assert "inv_diag" in buffer_names
        assert buffer_names["inv_diag"] is precond.inv_diag

    def test_to_dtype_moves_the_buffer(self, well_conditioned_matrix: torch.Tensor) -> None:
        """``.to(dtype=...)`` actually converts the registered buffer's dtype."""
        precond = JacobiPreconditioner(well_conditioned_matrix)
        assert precond.inv_diag.dtype == torch.float64

        precond = precond.to(dtype=torch.float32)

        assert precond.inv_diag.dtype == torch.float32
        r = torch.tensor([4.0, 3.0, 2.0, 2.0], dtype=torch.float32)
        z = precond.apply(r)
        torch.testing.assert_close(z, torch.ones_like(r), rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
    def test_to_device_moves_the_buffer(self, well_conditioned_matrix: torch.Tensor) -> None:
        """``.to(device=...)`` actually moves the registered buffer across devices."""
        precond = JacobiPreconditioner(well_conditioned_matrix)
        assert precond.inv_diag.device.type == "cpu"

        precond = precond.to(device="cuda")

        assert precond.inv_diag.device.type == "cuda"
        r = torch.tensor([4.0, 3.0, 2.0, 2.0], dtype=torch.float64, device="cuda")
        z = precond.apply(r)
        torch.testing.assert_close(z, torch.ones_like(r))


def test_jacobi_improves_convergence(
    tridiagonal_system_known_solution_torch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    integration_tolerances: tuple[float, float],
) -> None:
    """Verify Jacobi preconditioner reduces iterations vs identity.

    Theory:
        For diagonally dominant matrices, Jacobi preconditioning improves
        the condition number, leading to faster convergence:
        ``iterations_jacobi <= iterations_identity``.
    """
    from torchalg import flexible_cg
    from torchalg.preconditioners.implementations import Identity

    A, b, _ = tridiagonal_system_known_solution_torch
    rtol, atol = integration_tolerances

    _, result_identity = flexible_cg(
        A, b, preconditioner=Identity(), rtol=rtol, atol=atol, maxiter=200
    )
    _, result_jacobi = flexible_cg(
        A, b, preconditioner=JacobiPreconditioner(A), rtol=rtol, atol=atol, maxiter=200
    )

    assert result_identity.converged
    assert result_jacobi.converged
    assert result_jacobi.iterations <= result_identity.iterations
