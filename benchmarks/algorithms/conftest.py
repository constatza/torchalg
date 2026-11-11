"""Fixtures for the algorithm benchmarks: one parametrized cell per (preconditioner, matrix)."""

from __future__ import annotations

import pytest
import torch

from benchmarks.algorithms.registry import (
    REGISTRY,
    PreconditionerSpec,
    Problem,
    Tree,
    make_snapshots,
)
from benchmarks.cases import MatrixCase


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
    )
