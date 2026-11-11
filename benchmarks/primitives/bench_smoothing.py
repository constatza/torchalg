"""Primitive: smoothing sweeps (weighted Jacobi, symmetric Gauss-Seidel) and triangular solves.

Each smoother is split into ``prepare`` (matrix-dependent one-off work: Jacobi's
Arnoldi damping estimate and inverse diagonal, Gauss-Seidel's triangular factors or level
schedules) and the per-sweep apply the cycles pay on every application.
"""

from __future__ import annotations

from typing import Literal

import pytest
import torch

from benchmarks.cases import MatrixCase
from benchmarks.conftest import Record
from benchmarks.harness import Level
from benchmarks.primitives._support import DENSE_N_CAP, case_params
from torchalg.preconditioners.implementations.amg.smoothers import (
    GaussSeidelSmoother as DenseGaussSeidelSmoother,
)
from torchalg.preconditioners.implementations.amg.smoothers import (
    JacobiSmoother as DenseJacobiSmoother,
)
from torchalg.sparse.kernels.triangular import PreparedTriangularSolve, level_schedule
from torchalg.sparse.preconditioners.amg._gauss_seidel import PreparedSymmetricGaussSeidel
from torchalg.sparse.preconditioners.amg.smoothers import JacobiSmoother as SparseJacobiSmoother

FIXED_OMEGA = 0.7
SWEEPS = 1
GAUSS_SEIDEL_REPEATS = 3
PREPARE_REPEATS = 3


def _residual_norm(matrix: torch.Tensor, rhs: torch.Tensor, x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(rhs - matrix @ x))


def _assert_reduces_residual(
    matrix: torch.Tensor, rhs: torch.Tensor, start: torch.Tensor, smoothed: torch.Tensor
) -> None:
    assert _residual_norm(matrix, rhs, smoothed) < _residual_norm(matrix, rhs, start)


# --- weighted Jacobi -------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["dense", "csr"])
@pytest.mark.parametrize("omega", ["auto", "fixed"])
def test_jacobi_prepare(
    fmt: str,
    omega: str,
    case: MatrixCase,
    capped_dense_matrix: torch.Tensor,
    csr_matrix_: torch.Tensor,
    record: Record,
) -> None:
    smoother_class = DenseJacobiSmoother if fmt == "dense" else SparseJacobiSmoother
    matrix = capped_dense_matrix if fmt == "dense" else csr_matrix_
    smoother = smoother_class(omega=FIXED_OMEGA if omega == "fixed" else None)
    record(
        Level.PRIMITIVE,
        "smoothing.jacobi.prepare",
        case_params(case, format=fmt, omega=omega),
        lambda: smoother.prepare(matrix),
        repeats=PREPARE_REPEATS,
        warmups=0,
    )


@pytest.mark.parametrize("fmt", ["dense", "csr"])
@pytest.mark.parametrize("omega", ["auto_warm", "fixed"])
@pytest.mark.parametrize("entry", ["general", "from_zero"])
def test_jacobi_apply(
    fmt: str,
    omega: str,
    entry: str,
    case: MatrixCase,
    capped_dense_matrix: torch.Tensor,
    csr_matrix_: torch.Tensor,
    rhs: torch.Tensor,
    zero_guess: torch.Tensor,
    record: Record,
) -> None:
    smoother_class = DenseJacobiSmoother if fmt == "dense" else SparseJacobiSmoother
    matrix = capped_dense_matrix if fmt == "dense" else csr_matrix_
    prepared = smoother_class(omega=FIXED_OMEGA if omega == "fixed" else None).prepare(matrix)
    smoothed = prepared.smooth(rhs, zero_guess, SWEEPS)
    _assert_reduces_residual(matrix, rhs, zero_guess, smoothed)
    torch.testing.assert_close(prepared.smooth_from_zero(rhs, SWEEPS), smoothed)
    if fmt == "csr" and omega == "auto_warm" and case.n <= DENSE_N_CAP:
        dense = DenseJacobiSmoother().prepare(csr_matrix_.to_dense())
        torch.testing.assert_close(smoothed, dense.smooth(rhs, zero_guess, SWEEPS))
    params = case_params(case, format=fmt, omega=omega)
    if entry == "from_zero":
        params = {**params, "entry": entry}
        run = lambda: prepared.smooth_from_zero(rhs, SWEEPS)
    else:
        run = lambda: prepared.smooth(rhs, zero_guess, SWEEPS)
    record(Level.PRIMITIVE, "smoothing.jacobi", params, run)


# --- symmetric Gauss-Seidel ------------------------------------------------------------


def test_dense_symmetric_gauss_seidel(
    case: MatrixCase,
    capped_dense_matrix: torch.Tensor,
    rhs: torch.Tensor,
    zero_guess: torch.Tensor,
    record: Record,
) -> None:
    smoother = DenseGaussSeidelSmoother()
    prepared = smoother.prepare(capped_dense_matrix)
    smoothed = prepared.smooth(rhs, zero_guess, SWEEPS)
    _assert_reduces_residual(capped_dense_matrix, rhs, zero_guess, smoothed)
    params = case_params(case, format="dense")
    record(
        Level.PRIMITIVE,
        "smoothing.symmetric_gauss_seidel.prepare",
        params,
        lambda: smoother.prepare(capped_dense_matrix),
        repeats=GAUSS_SEIDEL_REPEATS,
    )
    record(
        Level.PRIMITIVE,
        "smoothing.symmetric_gauss_seidel",
        params,
        lambda: prepared.smooth(rhs, zero_guess, SWEEPS),
        repeats=GAUSS_SEIDEL_REPEATS,
    )


def test_sparse_symmetric_gauss_seidel(
    case: MatrixCase,
    csr_matrix_: torch.Tensor,
    rhs: torch.Tensor,
    zero_guess: torch.Tensor,
    record: Record,
) -> None:
    prepared = PreparedSymmetricGaussSeidel.build(csr_matrix_)
    smoothed = prepared.smooth(rhs, zero_guess, SWEEPS)
    _assert_reduces_residual(csr_matrix_, rhs, zero_guess, smoothed)
    if case.n <= DENSE_N_CAP:
        dense = DenseGaussSeidelSmoother().prepare(csr_matrix_.to_dense())
        torch.testing.assert_close(smoothed, dense.smooth(rhs, zero_guess, SWEEPS))
    params = case_params(case, format="csr")
    record(
        Level.PRIMITIVE,
        "smoothing.symmetric_gauss_seidel.prepare",
        params,
        lambda: PreparedSymmetricGaussSeidel.build(csr_matrix_),
        repeats=GAUSS_SEIDEL_REPEATS,
    )
    record(
        Level.PRIMITIVE,
        "smoothing.symmetric_gauss_seidel",
        params,
        lambda: prepared.smooth(rhs, zero_guess, SWEEPS),
        repeats=GAUSS_SEIDEL_REPEATS,
    )


# --- sparse triangular solve (IC0/ILU/ICholesky shape) ---------------------------------


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_sparse_level_schedule(
    direction: Literal["forward", "backward"],
    case: MatrixCase,
    csr_matrix_: torch.Tensor,
    record: Record,
) -> None:
    schedule = level_schedule(csr_matrix_, direction=direction)
    assert int(schedule.level_sizes.sum()) == case.n
    record(
        Level.PRIMITIVE,
        "triangular.level_schedule",
        case_params(case, direction=direction, levels=len(schedule.level_sizes)),
        lambda: level_schedule(csr_matrix_, direction=direction),
        repeats=GAUSS_SEIDEL_REPEATS,
    )


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_sparse_triangular_solve(
    direction: Literal["forward", "backward"],
    case: MatrixCase,
    csr_matrix_: torch.Tensor,
    rhs: torch.Tensor,
    record: Record,
) -> None:
    schedule = level_schedule(csr_matrix_, direction=direction)
    prepared = PreparedTriangularSolve(csr_matrix_, direction, schedule=schedule)
    solution = prepared.solve(rhs)
    if case.n <= DENSE_N_CAP:
        dense = csr_matrix_.to_dense()
        triangle = torch.tril(dense) if direction == "forward" else torch.triu(dense)
        reference = torch.linalg.solve_triangular(
            triangle, rhs.unsqueeze(-1), upper=direction == "backward"
        ).squeeze(-1)
        torch.testing.assert_close(solution, reference)
    params = case_params(case, direction=direction, levels=len(schedule.level_sizes))
    record(
        Level.PRIMITIVE,
        "triangular.prepare",
        params,
        lambda: PreparedTriangularSolve(csr_matrix_, direction, schedule=schedule),
        repeats=GAUSS_SEIDEL_REPEATS,
    )
    record(
        Level.PRIMITIVE,
        "triangular.solve",
        params,
        lambda: prepared.solve(rhs),
        repeats=GAUSS_SEIDEL_REPEATS,
    )
