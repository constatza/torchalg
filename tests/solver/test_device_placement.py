"""Tests for automatic device placement in ``IterativeSolverBase.solve()``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from torchalg.conjugate_gradient import PCGSolver
from torchalg.preconditioners.implementations.jacobi import JacobiPreconditioner
from torchalg.strategies.convergence import CombinedToleranceCriterion

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tests.conftest import ToTorch

# (rtol, atol) sized to each dtype's headroom - loose for float32, tight for float64.
_PRECISION_TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    torch.float32: (1e-4, 1e-6),
    torch.float64: (1e-10, 1e-14),
}


def test_solve_places_system_on_resolved_device(
    tridiagonal_system_known_solution_torch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """A small system with no GPU policy win is solved on the CPU.

    Runs unconditionally: a 4x4 dense system is below every GPU threshold, so
    this also holds on a machine with CUDA.
    """
    matrix, rhs, _ = tridiagonal_system_known_solution_torch
    solver = PCGSolver(convergence_criterion=CombinedToleranceCriterion(rtol=1e-10, atol=1e-14))

    solution, _ = solver.solve(matrix, rhs, maxiter=100)

    assert solution.device == torch.device("cpu")


def test_solve_moves_module_preconditioner_to_resolved_device(
    tridiagonal_system_known_solution_torch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """A ``nn.Module``-based preconditioner built on CPU ends up on the resolved device.

    Exercises the exact hazard the design guards against: without moving the
    preconditioner alongside the system, a CPU-built ``JacobiPreconditioner``
    would crash inside ``apply()`` the moment the residual moved to a
    different device.
    """
    matrix, rhs, _ = tridiagonal_system_known_solution_torch
    preconditioner = JacobiPreconditioner().setup(matrix)
    solver = PCGSolver(
        preconditioner=preconditioner,
        convergence_criterion=CombinedToleranceCriterion(rtol=1e-10, atol=1e-14),
    )

    solver.solve(matrix, rhs, maxiter=100)

    assert preconditioner.inv_diag.device == torch.device("cpu")


def test_place_on_device_failure_leaves_preconditioner_untouched(
    tridiagonal_system_known_solution_torch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed move of ``b`` must not have already mutated the preconditioner.

    Simulates the CUDA-OOM scenario: ``A`` is moved successfully but ``b``'s
    move fails. Per the ordering rationale in ``_place_on_device``, the
    preconditioner (moved last, in place) must be untouched so the caller
    can safely retry.
    """
    matrix, rhs, _ = tridiagonal_system_known_solution_torch
    preconditioner = JacobiPreconditioner().setup(matrix)
    solver = PCGSolver(preconditioner=preconditioner)
    original_inv_diag = preconditioner.inv_diag

    real_to = torch.Tensor.to

    def _failing_to(self: torch.Tensor, device: torch.device) -> torch.Tensor:
        if self is rhs:
            raise RuntimeError("simulated device-move failure")
        return real_to(self, device)

    monkeypatch.setattr(torch.Tensor, "to", _failing_to)

    with pytest.raises(RuntimeError, match="simulated device-move failure"):
        solver.solve(matrix, rhs, maxiter=100)

    assert preconditioner.inv_diag is original_inv_diag


def test_solve_leaves_matvec_callable_device_untouched(
    diagonal_system_known_solution_torch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """A matvec callable for ``A`` is never touched by device placement.

    Only a plain tensor ``A`` is moved; an opaque callable's captured
    tensors remain the caller's responsibility, matching torchalg's
    "device follows the caller" convention for anything it doesn't own.
    """
    matrix, rhs, x_exact = diagonal_system_known_solution_torch

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        assert vector.device == rhs.device
        return matrix @ vector

    solver = PCGSolver(convergence_criterion=CombinedToleranceCriterion(rtol=1e-10, atol=1e-14))
    solution, result = solver.solve(matvec, rhs, maxiter=100)

    assert result.converged is True
    assert torch.allclose(solution, x_exact.to(solution.device), rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_solve_preserves_input_dtype(
    tridiagonal_spd_small: NDArray,
    rhs_ones_small: NDArray,
    to_torch: ToTorch,
    dtype: torch.dtype,
) -> None:
    """Device placement in solve() never coerces dtype - float32 stays float32, float64 stays float64."""
    matrix = to_torch(tridiagonal_spd_small, dtype=dtype)
    rhs = to_torch(rhs_ones_small, dtype=dtype)
    rtol, atol = _PRECISION_TOLERANCES[dtype]
    solver = PCGSolver(convergence_criterion=CombinedToleranceCriterion(rtol=rtol, atol=atol))

    solution, _ = solver.solve(matrix, rhs, maxiter=200)

    assert solution.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_solve_preserves_preconditioner_buffer_dtype(
    tridiagonal_spd_small: NDArray,
    rhs_ones_small: NDArray,
    to_torch: ToTorch,
    dtype: torch.dtype,
) -> None:
    """Moving an nn.Module preconditioner during placement never coerces its buffer dtype."""
    matrix = to_torch(tridiagonal_spd_small, dtype=dtype)
    rhs = to_torch(rhs_ones_small, dtype=dtype)
    preconditioner = JacobiPreconditioner().setup(matrix)
    rtol, atol = _PRECISION_TOLERANCES[dtype]
    solver = PCGSolver(
        preconditioner=preconditioner,
        convergence_criterion=CombinedToleranceCriterion(rtol=rtol, atol=atol),
    )

    solver.solve(matrix, rhs, maxiter=200)

    assert preconditioner.inv_diag.dtype == dtype


@pytest.fixture
def cuda_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend no CUDA device is visible, so the cost policy alone would pick the CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def meta_system(
    tridiagonal_system_known_solution_torch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """The small system on the ``meta`` device, standing in for an accelerator.

    A copy out of ``meta`` raises, so any move to the CPU fails loudly.
    """
    matrix, rhs, _ = tridiagonal_system_known_solution_torch
    return matrix.to("meta"), rhs.to("meta")


@pytest.mark.usefixtures("cuda_absent")
def test_operands_already_on_an_accelerator_stay_there(
    meta_system: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """With no explicit device, a non-CPU system matrix keeps its device.

    The cost policy would choose the CPU for this small dense system; it decides
    only for CPU-resident input, so no per-solve device-to-host copy happens.
    """
    matrix, rhs = meta_system
    solver = PCGSolver()

    placed_matrix, placed_rhs, placed_guess = solver._place_on_device(
        matrix, rhs, torch.zeros_like(rhs)
    )

    assert isinstance(placed_matrix, torch.Tensor)
    assert placed_matrix.device == matrix.device
    assert placed_rhs.device == matrix.device
    assert placed_guess is not None
    assert placed_guess.device == matrix.device


@pytest.fixture
def csr_system(
    tridiagonal_system_known_solution_torch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """The small system with a CPU CSR matrix carrying PyTorch's default int64 indices."""
    matrix, rhs, _ = tridiagonal_system_known_solution_torch
    csr = matrix.to_sparse_csr()
    assert csr.crow_indices().dtype == csr.col_indices().dtype == torch.int64
    return csr, rhs


def test_csr_indices_are_narrowed_before_the_move(
    csr_system: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """A CPU int64 CSR matrix leaves placement with int32 indices (half the H2D index bytes)."""
    matrix, rhs = csr_system

    placed, _, _ = PCGSolver()._place_on_device(matrix, rhs, None)

    assert isinstance(placed, torch.Tensor)
    assert placed.crow_indices().dtype == placed.col_indices().dtype == torch.int32
    assert matrix.crow_indices().dtype == torch.int64


def test_csr_that_requires_grad_keeps_its_indices(
    csr_system: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """A tracked matrix is not rebuilt, so its autograd graph stays intact."""
    matrix, rhs = csr_system
    tracked = matrix.clone().requires_grad_(True)

    placed, _, _ = PCGSolver()._place_on_device(tracked, rhs, None)

    assert placed is tracked


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_solve_runs_on_cuda_when_available(
    tridiagonal_system_known_solution_torch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """End-to-end: CPU-constructed tensors actually execute and solve on CUDA."""
    matrix, rhs, _ = tridiagonal_system_known_solution_torch
    solver = PCGSolver(convergence_criterion=CombinedToleranceCriterion(rtol=1e-10, atol=1e-14))

    solution, result = solver.solve(matrix, rhs, maxiter=100)

    assert solution.device.type == "cuda"
    assert result.converged is True
