"""Fixtures for the algorithm benchmarks: one parametrized cell per (preconditioner, matrix).

The matrix cases are the default ladder plus the CSR-only large cases, which are marked
``slow`` (select ``-m "benchmark and not slow"`` to skip them).
"""

from __future__ import annotations

import pytest
import torch

from benchmarks.algorithms.registry import (
    REGISTRY,
    PreconditionerSpec,
    Problem,
    Tree,
    make_rhs_block,
    make_snapshots,
)
from benchmarks.cases import MatrixCase, default_cases, large_csr_cases

CASES = (
    *(pytest.param(case, id=case.id) for case in default_cases()),
    *(pytest.param(case, id=case.id, marks=pytest.mark.slow) for case in large_csr_cases()),
)


@pytest.fixture(params=CASES)
def case(request: pytest.FixtureRequest) -> MatrixCase:
    """Overrides the shared ladder with the large CSR-only cases added."""
    return request.param


@pytest.fixture(params=REGISTRY, ids=lambda spec: spec.id)
def spec(request: pytest.FixtureRequest) -> PreconditionerSpec:
    """One registry entry per test cell."""
    return request.param


@pytest.fixture
def problem(
    request: pytest.FixtureRequest,
    spec: PreconditionerSpec,
    case: MatrixCase,
    dtype: torch.dtype,
    device: torch.device,
) -> Problem:
    """The cell's system in its tree's format; skips cells over the spec's size cap.

    The cap check runs before any matrix is materialized so excluded dense cells cost nothing.
    """
    if case.n > spec.max_n:
        pytest.skip(f"{spec.id} capped at n<={spec.max_n} ({spec.cap_reason}); case n={case.n}")
    fixture_name = "dense_matrix" if spec.tree is Tree.DENSE else "csr_matrix_"
    return Problem(
        case=case,
        matrix=request.getfixturevalue(fixture_name),
        rhs=request.getfixturevalue("rhs"),
        snapshots=make_snapshots(case, dtype, device),
        rhs_block=make_rhs_block(case, dtype, device),
    )
