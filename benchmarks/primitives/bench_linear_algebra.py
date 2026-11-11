"""Primitive: Galerkin product ``PᵀAP`` (dense versus CSR) and CSR transpose."""

from __future__ import annotations

import torch

from benchmarks.cases import MatrixCase
from benchmarks.conftest import Record
from benchmarks.harness import Level
from benchmarks.primitives._support import DENSE_N_CAP, case_params
from torchalg.sparse.kernels.galerkin import form_sparse_sparse

GALERKIN_REPEATS = 3


def _dense_galerkin(prolongation: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return prolongation.T @ matrix @ prolongation


def test_dense_galerkin(
    case: MatrixCase,
    capped_dense_matrix: torch.Tensor,
    prolongation_dense: torch.Tensor,
    record: Record,
) -> None:
    record(
        Level.PRIMITIVE,
        "galerkin",
        case_params(case, format="dense"),
        lambda: _dense_galerkin(prolongation_dense, capped_dense_matrix),
        repeats=GALERKIN_REPEATS,
    )


def test_sparse_galerkin(
    case: MatrixCase,
    csr_matrix_: torch.Tensor,
    prolongation_csr: torch.Tensor,
    record: Record,
) -> None:
    coarse = form_sparse_sparse(prolongation_csr, csr_matrix_)
    if case.n <= DENSE_N_CAP:
        dense_coarse = _dense_galerkin(prolongation_csr.to_dense(), csr_matrix_.to_dense())
        torch.testing.assert_close(coarse.to_dense(), dense_coarse)
    record(
        Level.PRIMITIVE,
        "galerkin",
        case_params(case, format="csr"),
        lambda: form_sparse_sparse(prolongation_csr, csr_matrix_),
        repeats=GALERKIN_REPEATS,
    )


def test_csr_transpose(case: MatrixCase, prolongation_csr: torch.Tensor, record: Record) -> None:
    transposed = prolongation_csr.t().to_sparse_csr()
    torch.testing.assert_close(transposed.to_dense(), prolongation_csr.to_dense().T)
    record(
        Level.PRIMITIVE,
        "csr_transpose",
        case_params(case, route="t_to_csr"),
        lambda: prolongation_csr.t().to_sparse_csr(),
    )
    record(
        Level.PRIMITIVE,
        "csr_transpose",
        case_params(case, route="via_coo"),
        lambda: prolongation_csr.to_sparse_coo().t().to_sparse_csr(),
    )
