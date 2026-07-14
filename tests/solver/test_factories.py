"""Tests for ``torchalg.factories`` public solver entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from torchalg.factories import flexible_cg, pcg
from torchalg.monitoring import TraceMode
from torchalg.preconditioners.base import (
    BindableInputs,
    Preconditioner,
    PreconditionerContext,
)
from torchalg.preconditioners.implementations.jacobi import JacobiPreconditioner

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


class _RecordingBindablePreconditioner(Preconditioner, BindableInputs):
    """Preconditioner test double that records bound extra tensors and contexts."""

    def __init__(self) -> None:
        """Initialize empty call records."""
        self.bound_inputs: dict[str, torch.Tensor] = {}
        self.contexts: list[PreconditionerContext | None] = []

    @property
    def extra_input_names(self) -> tuple[str, ...]:
        """Declare one accepted extra input."""
        return ("matrix",)

    def bind_inputs(self, **inputs: torch.Tensor) -> None:
        """Record filtered extra inputs."""
        self.bound_inputs = dict(inputs)

    def apply(
        self,
        residual: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> torch.Tensor:
        """Record context and apply identity preconditioning."""
        self.contexts.append(context)
        return residual.clone()


def test_pcg_solves_small_spd_system(
    tridiagonal_spd_small: NDArray,
    rhs_ones_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
) -> None:
    """PCG converges to the dense torch solve on a small SPD system."""
    matrix = to_torch(tridiagonal_spd_small)
    rhs = to_torch(rhs_ones_small)

    solution, result = pcg(
        matrix,
        rhs,
        rtol=1e-12,
        atol=1e-14,
        maxiter=100,
    )

    expected = torch.linalg.solve(matrix, rhs)
    assert result.converged is True
    assert result.breakdown is False
    assert result.iterations <= matrix.shape[0]
    assert torch.allclose(solution, expected, rtol=1e-10, atol=1e-12)


def test_flexible_cg_solves_with_jacobi_preconditioner(
    diagonal_spd_small: NDArray,
    rhs_ones_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
) -> None:
    """FCG accepts a concrete preconditioner and converges on a diagonal system."""
    matrix = to_torch(diagonal_spd_small)
    rhs = to_torch(rhs_ones_small)
    preconditioner = JacobiPreconditioner(matrix)

    solution, result = flexible_cg(
        matrix,
        rhs,
        preconditioner=preconditioner,
        m_max=3,
        rtol=1e-12,
        atol=1e-14,
        maxiter=20,
    )

    expected = torch.linalg.solve(matrix, rhs)
    assert result.converged is True
    assert torch.allclose(solution, expected, rtol=1e-10, atol=1e-12)


def test_factory_full_trace_records_vectors(
    identity_matrix_small: NDArray,
    rhs_ones_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
) -> None:
    """FULL trace mode records residual and solution vectors on the result."""
    matrix = to_torch(identity_matrix_small)
    rhs = to_torch(rhs_ones_small)

    _, result = pcg(matrix, rhs, trace_mode=TraceMode.FULL, maxiter=5)

    assert result.residual_vectors is not None
    assert result.solution_vectors is not None
    assert result.residual_vectors.shape[1:] == rhs.shape
    assert result.solution_vectors.shape[1:] == rhs.shape


def test_factory_disabled_trace_omits_histories(
    identity_matrix_small: NDArray,
    rhs_ones_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
) -> None:
    """DISABLED trace mode avoids storing residual histories."""
    matrix = to_torch(identity_matrix_small)
    rhs = to_torch(rhs_ones_small)

    _, result = pcg(matrix, rhs, trace_mode="disabled", maxiter=5)

    assert result.residual_history_abs is None
    assert result.residual_history_rel is None
    assert result.residual_vectors is None
    assert result.solution_vectors is None


def test_factory_binds_extra_inputs_before_solve(
    tridiagonal_spd_small: NDArray,
    rhs_ones_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
) -> None:
    """Factories thread named extra tensors into bindable preconditioners."""
    matrix = to_torch(tridiagonal_spd_small)
    rhs = to_torch(rhs_ones_small)
    preconditioner = _RecordingBindablePreconditioner()

    _, result = flexible_cg(
        matrix,
        rhs,
        preconditioner=preconditioner,
        extra_inputs={
            "matrix": matrix,
            "ignored": rhs,
        },
        maxiter=20,
    )

    assert result.converged is True
    assert preconditioner.bound_inputs == {"matrix": matrix}
    assert preconditioner.contexts
    assert preconditioner.contexts[0] is not None
