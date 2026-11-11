"""Primitive: ``A @ x`` for dense and CSR operands (the cost every cycle level pays)."""

from __future__ import annotations

import torch

from benchmarks.cases import MatrixCase
from benchmarks.harness import Level


def test_dense_matvec(case: MatrixCase, dense_matrix, rhs, record) -> None:
    measurement = record(
        Level.PRIMITIVE,
        "matvec",
        {"n": case.n, "nnz": case.nnz, "format": "dense", "matrix": case.id},
        lambda: dense_matrix @ rhs,
    )
    assert measurement.median_s > 0


def test_csr_matvec(case: MatrixCase, csr_matrix_, rhs, record) -> None:
    reference = case.dense(rhs.dtype, rhs.device) @ rhs
    torch.testing.assert_close(csr_matrix_ @ rhs, reference)
    record(
        Level.PRIMITIVE,
        "matvec",
        {"n": case.n, "nnz": case.nnz, "format": "csr", "matrix": case.id},
        lambda: csr_matrix_ @ rhs,
    )
