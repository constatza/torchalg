"""Primitive: CG vector operations and iteration cost with the identity preconditioner.

Covers the stable dot, PCG and FCG iterations, and the FCG orthogonalization window at the
default ``m_max``.
"""

from __future__ import annotations

import pytest
import torch

from benchmarks.cases import MatrixCase
from benchmarks.conftest import Record
from benchmarks.harness import Level
from benchmarks.primitives._support import case_params
from torchalg import flexible_cg, pcg
from torchalg.models.history import DirectionHistory
from torchalg.strategies.orthogonalization import DEFAULT_M_MAX, create_fcg_orthogonalization
from torchalg.utils.numerics import stable_dot_product

PCG_ITERATIONS = [1, 20]
FCG_ITERATIONS = 25
"""More than ``DEFAULT_M_MAX`` so the orthogonalization window has filled."""
SOLVER_REPEATS = 3
NO_TOLERANCE = 0.0
"""Tolerance that never triggers, so exactly ``maxiter`` iterations run even on box3d."""


def test_dot_product(case: MatrixCase, rhs: torch.Tensor, record: Record) -> None:
    other = rhs.flip(0)
    torch.testing.assert_close(stable_dot_product(rhs, other), torch.dot(rhs, other))
    record(
        Level.PRIMITIVE,
        "cg.dot",
        case_params(case, dot="stable"),
        lambda: stable_dot_product(rhs, other),
        repeats=20,
        warmups=3,
    )
    record(
        Level.PRIMITIVE,
        "cg.dot",
        case_params(case, dot="torch"),
        lambda: torch.dot(rhs, other),
        repeats=20,
        warmups=3,
    )


@pytest.mark.parametrize("iterations", PCG_ITERATIONS)
@pytest.mark.parametrize("format_name", ["dense", "csr"])
def test_pcg_identity(
    iterations: int,
    format_name: str,
    case: MatrixCase,
    rhs: torch.Tensor,
    request: pytest.FixtureRequest,
    record: Record,
) -> None:
    matrix = request.getfixturevalue("dense_matrix" if format_name == "dense" else "csr_matrix_")
    _, result = pcg(matrix, rhs, maxiter=iterations, rtol=NO_TOLERANCE, atol=NO_TOLERANCE)
    assert result.iterations == iterations
    record(
        Level.PRIMITIVE,
        "cg.pcg_identity",
        case_params(case, format=format_name, maxiter=iterations),
        lambda: pcg(matrix, rhs, maxiter=iterations, rtol=NO_TOLERANCE, atol=NO_TOLERANCE),
        repeats=SOLVER_REPEATS,
    )


def test_fcg_identity(
    case: MatrixCase, csr_matrix_: torch.Tensor, rhs: torch.Tensor, record: Record
) -> None:
    _, result = flexible_cg(
        csr_matrix_,
        rhs,
        m_max=DEFAULT_M_MAX,
        maxiter=FCG_ITERATIONS,
        rtol=NO_TOLERANCE,
        atol=NO_TOLERANCE,
    )
    assert result.iterations == FCG_ITERATIONS
    record(
        Level.PRIMITIVE,
        "cg.fcg_identity",
        case_params(
            case,
            format="csr",
            m_max=DEFAULT_M_MAX,
            maxiter=FCG_ITERATIONS,
        ),
        lambda: flexible_cg(
            csr_matrix_,
            rhs,
            m_max=DEFAULT_M_MAX,
            maxiter=FCG_ITERATIONS,
            rtol=NO_TOLERANCE,
            atol=NO_TOLERANCE,
        ),
        repeats=SOLVER_REPEATS,
    )


def test_fcg_orthogonalization_cycle(
    case: MatrixCase,
    rhs: torch.Tensor,
    orthogonalization_history: tuple[list[torch.Tensor], list[torch.Tensor]],
    record: Record,
) -> None:
    """One full sawtooth period (window 1..m_max) of the periodic-restart strategy."""
    directions, images = orthogonalization_history
    history = DirectionHistory.empty(max_size=DEFAULT_M_MAX)
    for d, q in zip(directions, images, strict=True):
        history.append(d, q, stable_dot_product(d, q))
    window = history.window(like=rhs)

    def cycle() -> None:
        strategy = create_fcg_orthogonalization(m_max=DEFAULT_M_MAX)
        for _ in range(DEFAULT_M_MAX):
            strategy.orthogonalize(rhs, window)

    strategy = create_fcg_orthogonalization(m_max=DEFAULT_M_MAX)
    orthogonalized, _ = strategy.orthogonalize(rhs, window)
    assert orthogonalized.shape == rhs.shape
    record(
        Level.PRIMITIVE,
        "cg.fcg_orthogonalization",
        case_params(case, m_max=DEFAULT_M_MAX, calls=DEFAULT_M_MAX),
        cycle,
        repeats=SOLVER_REPEATS,
    )
