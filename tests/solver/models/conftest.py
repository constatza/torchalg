"""Fixtures for ``torchalg.models`` tests.

Modular, composable fixtures per project convention - no inline test data.
Builds on the session-wide ``torch_dtype`` fixture from ``tests/conftest.py``.
"""

from __future__ import annotations

import pytest
import torch

from torchalg.models.diagnostics import TerminationDiagnostics
from torchalg.models.history import DirectionHistory, ResidualHistory
from torchalg.models.result import IterationContext, SolverResult
from torchalg.models.state import CGState, KrylovState, SolverState

STATE_VECTOR_SIZE = 5


# =============================================================================
# state.py fixtures
# =============================================================================


@pytest.fixture
def krylov_vectors(
    torch_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Five distinct working vectors (u, r, w, d, q) for Krylov state fixtures.

    Returns:
        tuple[torch.Tensor, ...]: ``(u, r, w, d, q)``, each shape ``(5,)``,
            each a distinct constant fill so a test can confirm state fields
            are not accidentally aliased or swapped.
    """
    n = STATE_VECTOR_SIZE
    u = torch.full((n,), 1.0, dtype=torch_dtype)
    r = torch.full((n,), 2.0, dtype=torch_dtype)
    w = torch.full((n,), 3.0, dtype=torch_dtype)
    d = torch.full((n,), 4.0, dtype=torch_dtype)
    q = torch.full((n,), 5.0, dtype=torch_dtype)
    return u, r, w, d, q


@pytest.fixture
def solver_state() -> SolverState:
    """A minimal valid base ``SolverState``.

    Returns:
        SolverState: State at iteration 0, not converged/broken/diverged.
    """
    return SolverState(
        iteration=0,
        converged=False,
        breakdown=False,
        divergence=False,
        residual_norm=1.0,
        rhs_norm=1.0,
    )


@pytest.fixture
def krylov_state(
    krylov_vectors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> KrylovState:
    """A minimal valid ``KrylovState`` built from ``krylov_vectors``.

    Returns:
        KrylovState: State at iteration 0 carrying the five working vectors.
    """
    u, r, w, d, q = krylov_vectors
    return KrylovState(
        iteration=0,
        converged=False,
        breakdown=False,
        divergence=False,
        residual_norm=1.0,
        rhs_norm=1.0,
        u=u,
        r=r,
        w=w,
        d=d,
        q=q,
    )


@pytest.fixture
def cg_state(
    krylov_vectors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> CGState:
    """A ``CGState`` built via ``CGState.create_initial``.

    Uses ``krylov_vectors`` for u/r/w/d/q and a distinct rhs magnitude
    (``10.0``) so the relative-residual-norm branch in ``create_initial`` is
    exercised meaningfully (rather than degenerating to the ``rhs_norm == 0``
    edge case).

    Returns:
        CGState: Freshly created state at iteration 0 with empty histories.
    """
    u, r, w, d, q = krylov_vectors
    return CGState.create_initial(
        u=u,
        r=r,
        w=w,
        d=d,
        q=q,
        residual_norm=float(torch.linalg.norm(r)),
        rhs_norm=10.0,
        max_history=3,
    )


# =============================================================================
# history.py fixtures
# =============================================================================


@pytest.fixture
def empty_direction_history() -> DirectionHistory:
    """An empty ``DirectionHistory`` with a window size of 2.

    Returns:
        DirectionHistory: Empty history, ``max_size=2`` (small enough to
            exercise truncation with only 3 ``add`` calls).
    """
    return DirectionHistory.empty(max_size=2)


@pytest.fixture
def empty_residual_history() -> ResidualHistory:
    """An empty ``ResidualHistory``.

    Returns:
        ResidualHistory: History with no recorded norms.
    """
    return ResidualHistory.empty()


@pytest.fixture
def direction_vector_sequence(
    torch_dtype: torch.dtype,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Three distinct ``(d, q)`` pairs for exercising ``DirectionHistory.add``.

    Returns:
        list[tuple[torch.Tensor, torch.Tensor]]: Three ``(d, q)`` pairs,
            each vector a distinct constant fill so truncation/ordering is
            unambiguous to assert on.
    """
    n = STATE_VECTOR_SIZE
    return [
        (
            torch.full((n,), float(i), dtype=torch_dtype),
            torch.full((n,), float(i + 100), dtype=torch_dtype),
        )
        for i in range(3)
    ]


# =============================================================================
# result.py fixtures
# =============================================================================


@pytest.fixture
def minimal_solver_result() -> SolverResult:
    """A ``SolverResult`` built with only the required fields.

    Returns:
        SolverResult: Converged result with every optional field left at
            its default (``diagnostics=TerminationDiagnostics()``, histories
            ``None``, etc.).
    """
    return SolverResult(
        converged=True,
        iterations=5,
        residual=1e-8,
        residual_abs=1e-8,
        rhs_norm=1.0,
        breakdown=False,
    )


@pytest.fixture
def populated_diagnostics() -> TerminationDiagnostics:
    """A ``TerminationDiagnostics`` with every field set to a non-default value.

    Returns:
        TerminationDiagnostics: Diagnostics describing a solve that
            converged at iteration 7 (no breakdown).
    """
    return TerminationDiagnostics(converged_at=7, breakdown_at=None, breakdown_reason=None)


@pytest.fixture
def iteration_context(
    torch_dtype: torch.dtype,
) -> IterationContext:
    """An ``IterationContext`` for a small 3-dimensional system.

    Returns:
        IterationContext: Context at iteration 2 with hand-distinguishable
            residual/solution/matrix/rhs tensors.
    """
    return IterationContext(
        iteration=2,
        residual=torch.tensor([1.0, 1.0, 1.0], dtype=torch_dtype),
        solution=torch.tensor([0.5, 0.5, 0.5], dtype=torch_dtype),
        matrix=torch.eye(3, dtype=torch_dtype),
        rhs=torch.tensor([1.0, 1.0, 1.0], dtype=torch_dtype),
    )
