"""Primitive: coarsest-level direct solve, split into one-off preparation and per-cycle apply.

``coarse_solve.prepare`` is the factorization (LU or pseudo-inverse) paid once per
hierarchy; ``coarse_solve.apply`` is what every V-cycle application pays. The sparse-tree
factories additionally densify the CSR coarse matrix, once, inside ``prepare``.
"""

from __future__ import annotations

import pytest
import torch

from benchmarks.conftest import Record
from benchmarks.harness import Level, ParamValue
from benchmarks.primitives._support import CURRENT_VARIANT
from torchalg.multigrid.coarse_solve import (
    CoarseSolverFactory,
    lu_coarse_solver,
    pseudo_inverse_coarse_solver,
)
from torchalg.sparse.preconditioners.amg.coarse_solve import (
    dense_lu_coarse_solver,
    dense_pseudo_inverse_coarse_solver,
)

SOLVE_RTOL = 1e-8
"""The fixture is well conditioned (shifted Gram), so both solvers agree to near eps."""


def _params(matrix: torch.Tensor, solver: str, fmt: str) -> dict[str, ParamValue]:
    return {
        "n": matrix.shape[0],
        "solver": solver,
        "format": fmt,
        "variant": CURRENT_VARIANT,
    }


def _assert_solves(matrix: torch.Tensor, rhs: torch.Tensor, solution: torch.Tensor) -> None:
    dense = matrix.to_dense() if matrix.is_sparse_csr else matrix
    torch.testing.assert_close(dense @ solution, rhs, rtol=SOLVE_RTOL, atol=SOLVE_RTOL)


def _measure(
    factory: CoarseSolverFactory,
    matrix: torch.Tensor,
    shape_source: torch.Tensor,
    rhs: torch.Tensor,
    solver: str,
    fmt: str,
    record: Record,
) -> None:
    solve = factory(matrix)
    _assert_solves(matrix, rhs, solve(rhs))
    params = _params(shape_source, solver, fmt)
    record(Level.PRIMITIVE, "coarse_solve.prepare", params, lambda: factory(matrix))
    record(Level.PRIMITIVE, "coarse_solve.apply", params, lambda: solve(rhs))


@pytest.mark.parametrize(
    ("name", "factory"), [("solve", lu_coarse_solver), ("pinv", pseudo_inverse_coarse_solver)]
)
def test_dense_coarse_solve(
    name: str,
    factory: CoarseSolverFactory,
    coarse_matrix: torch.Tensor,
    coarse_rhs: torch.Tensor,
    record: Record,
) -> None:
    _measure(factory, coarse_matrix, coarse_matrix, coarse_rhs, name, "dense", record)


@pytest.mark.parametrize(
    ("name", "factory"),
    [("solve", dense_lu_coarse_solver), ("pinv", dense_pseudo_inverse_coarse_solver)],
)
def test_sparse_tree_coarse_solve(
    name: str,
    factory: CoarseSolverFactory,
    coarse_matrix: torch.Tensor,
    coarse_csr: torch.Tensor,
    coarse_rhs: torch.Tensor,
    record: Record,
) -> None:
    _measure(factory, coarse_csr, coarse_matrix, coarse_rhs, name, "csr", record)
