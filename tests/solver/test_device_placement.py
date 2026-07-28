"""Tests for automatic device placement in ``IterativeSolverBase.solve()``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from torchalg.conjugate_gradient import PCGSolver
from torchalg.preconditioners.implementations.jacobi import JacobiPreconditioner
from torchalg.strategies.convergence import CombinedToleranceCriterion
from torchalg.utils.device import resolve_device

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tests.conftest import ToTorch

# (rtol, atol) sized to each dtype's headroom - loose for float32, tight for float64.
_PRECISION_TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    torch.float32: (1e-4, 1e-6),
    torch.float64: (1e-10, 1e-14),
}


def test_resolve_device_returns_cpu_when_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """``resolve_device`` falls back to CPU when CUDA is not available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device() == torch.device("cpu")


def test_resolve_device_returns_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """``resolve_device`` picks CUDA when ``torch.cuda.is_available()`` is True."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device() == torch.device("cuda")


def test_solve_places_system_on_resolved_device(
    tridiagonal_system_known_solution_torch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """The solution returned by ``solve()`` lands on whatever device is resolved.

    Runs unconditionally (CPU-only CI still exercises the code path, since
    ``resolve_device()`` resolves to CPU there too) — regression safety for
    the existing CPU-only behavior.
    """
    matrix, rhs, _ = tridiagonal_system_known_solution_torch
    solver = PCGSolver(convergence_criterion=CombinedToleranceCriterion(rtol=1e-10, atol=1e-14))

    solution, _ = solver.solve(matrix, rhs, maxiter=100)

    assert solution.device == resolve_device()


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
    preconditioner = JacobiPreconditioner(matrix)
    solver = PCGSolver(
        preconditioner=preconditioner,
        convergence_criterion=CombinedToleranceCriterion(rtol=1e-10, atol=1e-14),
    )

    solver.solve(matrix, rhs, maxiter=100)

    assert preconditioner.inv_diag.device == resolve_device()


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
    preconditioner = JacobiPreconditioner(matrix)
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
        assert vector.device == resolve_device()
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
    preconditioner = JacobiPreconditioner(matrix)
    rtol, atol = _PRECISION_TOLERANCES[dtype]
    solver = PCGSolver(
        preconditioner=preconditioner,
        convergence_criterion=CombinedToleranceCriterion(rtol=rtol, atol=atol),
    )

    solver.solve(matrix, rhs, maxiter=200)

    assert preconditioner.inv_diag.dtype == dtype


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
