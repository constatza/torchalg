"""Integration tests: FCG's orthogonalization breakdown reaches ``TerminationDiagnostics``.

``OrthogonalizationReport.breakdown`` (a near-zero/degenerate Gram-Schmidt
result) was previously computed by ``OrthogonalizationStrategy.orthogonalize``
and then silently discarded by ``strategies/direction.py`` - never surfaced
anywhere, despite ``TerminationDiagnostics.ortho_breakdown_at`` existing
specifically to carry it. These tests use a stub ``OrthogonalizationStrategy``
that unconditionally reports breakdown, wired directly into ``FCGSolver``
(bypassing the ``flexible_cg()`` factory, which has no way to inject a
custom orthogonalization strategy), to verify the wiring in isolation from
whether a real SPD system happens to trigger genuine numerical breakdown.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch

from torchalg.conjugate_gradient import FCGSolver
from torchalg.strategies.orthogonalization import (
    OrthogonalizationReport,
    OrthogonalizationStrategy,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class _AlwaysBreaksDown(OrthogonalizationStrategy):
    """Stub orthogonalization strategy that reports breakdown every call."""

    @property
    def window_size(self) -> int | None:
        """Unlimited history - irrelevant for this stub."""
        return None

    def orthogonalize(
        self,
        vector: torch.Tensor,
        d_vectors: Sequence[torch.Tensor],
        q_vectors: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, OrthogonalizationReport]:
        """Pass ``vector`` through unchanged, always flagged as breakdown."""
        return vector, OrthogonalizationReport(coefficients=(), breakdown=True)


class _NeverBreaksDown(OrthogonalizationStrategy):
    """Stub orthogonalization strategy that never reports breakdown."""

    @property
    def window_size(self) -> int | None:
        """Unlimited history - irrelevant for this stub."""
        return None

    def orthogonalize(
        self,
        vector: torch.Tensor,
        d_vectors: Sequence[torch.Tensor],
        q_vectors: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, OrthogonalizationReport]:
        """Pass ``vector`` through unchanged, never flagged as breakdown."""
        return vector, OrthogonalizationReport(coefficients=(), breakdown=False)


def test_ortho_breakdown_reaches_termination_diagnostics(
    tridiagonal_spd_small: NDArray,
    rhs_ones_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
) -> None:
    """A stub strategy that always breaks down sets ``ortho_breakdown_at`` to iteration 1."""
    matrix = to_torch(tridiagonal_spd_small)
    rhs = to_torch(rhs_ones_small)
    solver = FCGSolver(orthogonalization=_AlwaysBreaksDown())

    _, result = solver.solve(matrix, rhs, rtol=1e-10, atol=1e-14, maxiter=5)

    assert result.diagnostics.ortho_breakdown_at == 1


def test_ortho_breakdown_at_is_none_when_never_reported(
    tridiagonal_spd_small: NDArray,
    rhs_ones_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
) -> None:
    """A stub strategy that never breaks down leaves ``ortho_breakdown_at`` unset."""
    matrix = to_torch(tridiagonal_spd_small)
    rhs = to_torch(rhs_ones_small)
    solver = FCGSolver(orthogonalization=_NeverBreaksDown())

    _, result = solver.solve(matrix, rhs, rtol=1e-10, atol=1e-14, maxiter=5)

    assert result.diagnostics.ortho_breakdown_at is None


def test_ortho_breakdown_at_records_first_occurrence_only(
    tridiagonal_spd_small: NDArray,
    rhs_ones_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
) -> None:
    """``ortho_breakdown_at`` stays at the first iteration it occurred, not the last."""
    matrix = to_torch(tridiagonal_spd_small)
    rhs = to_torch(rhs_ones_small)
    solver = FCGSolver(orthogonalization=_AlwaysBreaksDown())

    _, result = solver.solve(matrix, rhs, rtol=1e-10, atol=1e-14, maxiter=5)

    # Breakdown is reported every iteration by the stub, but the diagnostic
    # records the *first* occurrence (iteration 1), not a running/last value.
    assert result.iterations > 1
    assert result.diagnostics.ortho_breakdown_at == 1


def test_ortho_breakdown_is_purely_diagnostic_not_terminal(
    tridiagonal_spd_small: NDArray,
    rhs_ones_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
) -> None:
    """Orthogonalization breakdown does not stop the solve or mark it broken.

    Per Notay (2000): truncated-orthogonalization quality loss is a
    convergence-rate signal, not a hard breakdown requiring termination
    (see ``models/state.py::CGState.ortho_breakdown_at``'s docstring).
    """
    matrix = to_torch(tridiagonal_spd_small)
    rhs = to_torch(rhs_ones_small)
    solver = FCGSolver(orthogonalization=_AlwaysBreaksDown())

    _, result = solver.solve(matrix, rhs, rtol=1e-10, atol=1e-14, maxiter=20)

    assert result.converged is True
    assert result.breakdown is False
    assert result.diagnostics.ortho_breakdown_at == 1
