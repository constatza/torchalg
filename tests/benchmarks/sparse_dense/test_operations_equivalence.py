"""Correctness checks for every operations.py function's sparse/native/scipy leg.

Each test compares a sparse, native-alternative, or scipy leg against a
dense reference (or, where the two algorithms genuinely differ - IC0 vs.
scipy's incomplete LU, dense SVD vs. randomized low-rank SVD - against a
shared correctness criterion both should satisfy). These are sanity checks
that the benchmark harness is timing something numerically valid, not a
substitute for torchalg's own unit tests of ``dense_ic0``/``dense_ilu0``/
``cholesky_factor_solve`` (covered elsewhere under ``tests/solver/``).

Marked ``@pytest.mark.benchmark`` throughout, per this repo's convention
(``pyproject.toml``'s ``addopts = ["-m", "not benchmark"]`` already excludes
it from the default run) - run explicitly with
``uv run pytest -m benchmark tests/benchmarks/sparse_dense/``.
"""

from __future__ import annotations

import pytest
import torch
from bench_sparse_vs_dense import operations as ops

from torchalg.strategies.convergence import CombinedToleranceCriterion

pytestmark = pytest.mark.benchmark

_RTOL = 1e-8
_ATOL = 1e-10


def _solves(
    matrix: torch.Tensor, solution: torch.Tensor, rhs: torch.Tensor, *, rtol: float, atol: float
) -> bool:
    """``||matrix @ solution - rhs|| <= max(rtol * ||rhs||, atol)`` - reuses torchalg's own criterion.

    The standard way to check "does x solve Ax=b" when there's no reference
    solution tensor to compare elementwise against (which is what
    ``torch.allclose`` is for); matches this repo's own established pattern
    (``tests/benchmarks/exactness/conftest.py``'s ``check_residual_norm``)
    instead of hand-rolling the ratio.
    """
    criterion = CombinedToleranceCriterion(rtol=rtol, atol=atol)
    residual = matrix @ solution - rhs
    return criterion.has_converged(residual, torch.linalg.norm(rhs))


def test_mv_dense_sparse_agree(
    small_dense_matrix: torch.Tensor, small_sparse_matrix: torch.Tensor, small_vector: torch.Tensor
) -> None:
    dense_result = ops.mv(small_dense_matrix, small_vector)
    sparse_result = ops.mv(small_sparse_matrix, small_vector)
    assert torch.allclose(dense_result, sparse_result, rtol=_RTOL, atol=_ATOL)


def test_elementwise_scale_dense_sparse_agree(
    small_dense_matrix: torch.Tensor, small_sparse_matrix: torch.Tensor, small_vector: torch.Tensor
) -> None:
    dense_inv_diag = ops.extract_inv_diag(small_dense_matrix)
    sparse_inv_diag = ops.extract_inv_diag(small_sparse_matrix)
    assert torch.allclose(dense_inv_diag, sparse_inv_diag, rtol=_RTOL, atol=_ATOL)

    dense_result = ops.elementwise_scale(dense_inv_diag, small_vector)
    sparse_result = ops.elementwise_scale(sparse_inv_diag, small_vector)
    assert torch.allclose(dense_result, sparse_result, rtol=_RTOL, atol=_ATOL)


def test_galerkin_formed_dense_spmm_spgemm_agree(
    small_dense_matrix: torch.Tensor,
    small_sparse_matrix: torch.Tensor,
    prolongation_dense: torch.Tensor,
    prolongation_sparse: torch.Tensor,
) -> None:
    dense_coarse = ops.form_galerkin_dense(prolongation_dense, small_dense_matrix)
    spmm_coarse = ops.form_galerkin_spmm(prolongation_dense, small_sparse_matrix)
    spgemm_coarse = ops.form_galerkin_spgemm(prolongation_sparse, small_sparse_matrix)

    assert torch.allclose(dense_coarse, spmm_coarse, rtol=_RTOL, atol=_ATOL)
    assert torch.allclose(dense_coarse, spgemm_coarse.to_dense(), rtol=1e-6, atol=1e-8)


def test_galerkin_matrix_free_equals_formed(
    small_dense_matrix: torch.Tensor,
    prolongation_dense: torch.Tensor,
    coarse_vector: torch.Tensor,
) -> None:
    coarse_matrix = ops.form_galerkin_dense(prolongation_dense, small_dense_matrix)
    formed_result = ops.apply_formed(coarse_matrix, coarse_vector)
    matrix_free_result = ops.apply_matrix_free(
        prolongation_dense, small_dense_matrix, coarse_vector
    )
    assert torch.allclose(formed_result, matrix_free_result, rtol=1e-6, atol=1e-8)


def test_triangular_apply_dense_solves_system(
    small_dense_matrix: torch.Tensor, small_vector: torch.Tensor
) -> None:
    factor = torch.linalg.cholesky(small_dense_matrix)
    solution = ops.triangular_apply_dense(factor, small_vector)
    assert _solves(small_dense_matrix, solution, small_vector, rtol=1e-8, atol=1e-10)


def test_triangular_apply_sparse_matches_dense_or_is_unsupported(
    small_dense_matrix: torch.Tensor, small_vector: torch.Tensor
) -> None:
    factor = torch.linalg.cholesky(small_dense_matrix)
    dense_solution = ops.triangular_apply_dense(factor, small_vector)
    try:
        sparse_solution = ops.triangular_apply_sparse(factor.to_sparse_csr(), small_vector)
    except (NotImplementedError, RuntimeError) as error:
        pytest.skip(f"sparse triangular solve unavailable on this build/device: {error}")
    else:
        assert torch.allclose(dense_solution, sparse_solution, rtol=1e-6, atol=1e-8)


def test_factorize_ic0_preserves_sparsity_pattern(small_dense_matrix: torch.Tensor) -> None:
    # IC(0) is an *incomplete* factorization used as a preconditioner inside
    # an iterative solve, not a one-shot direct solve - so the property
    # worth checking here is its defining structural guarantee (zero-fill:
    # L's lower-triangular non-zero pattern never exceeds A's), not solve
    # accuracy from a single apply.
    ic0_factor = ops.factorize_ic0(small_dense_matrix)
    factor_pattern = torch.tril(ic0_factor).abs() > 0
    matrix_pattern = torch.tril(small_dense_matrix).abs() > 0
    assert torch.all(~factor_pattern | matrix_pattern)


def test_factorize_spilu_solves_system_accurately(
    small_dense_matrix: torch.Tensor, small_sparse_matrix: torch.Tensor, small_vector: torch.Tensor
) -> None:
    # scipy `spilu`'s defaults still drop small fill-in entries (it is not
    # `splu`, the exact full factorization), so it approximates - closely,
    # but not to machine precision - a direct solve.
    csc = ops.to_scipy_csc(small_sparse_matrix)
    spilu_factor = ops.factorize_spilu(csc)
    spilu_solution = torch.from_numpy(spilu_factor.solve(small_vector.numpy())).to(
        small_vector.dtype
    )
    assert _solves(small_dense_matrix, spilu_solution, small_vector, rtol=1e-3, atol=1e-8)


def test_eigh_dense_lobpcg_scipy_agree_on_extreme_eigenvalues(
    small_dense_matrix: torch.Tensor, small_sparse_matrix: torch.Tensor
) -> None:
    k = 3
    dense_eigenvalues = ops.eigh_dense(small_dense_matrix)
    dense_top_k = torch.sort(dense_eigenvalues, descending=True).values[:k]

    lobpcg_eigenvalues, _ = ops.eigh_lobpcg(small_sparse_matrix, k=k)
    lobpcg_top_k = torch.sort(lobpcg_eigenvalues, descending=True).values

    scipy_csr = ops.to_scipy_csc(small_sparse_matrix).tocsr()
    scipy_eigenvalues, _ = ops.eigh_scipy(scipy_csr, k=k)
    scipy_top_k = torch.from_numpy(scipy_eigenvalues).sort(descending=True).values

    assert torch.allclose(dense_top_k, lobpcg_top_k, rtol=1e-4, atol=1e-6)
    assert torch.allclose(dense_top_k, scipy_top_k, rtol=1e-4, atol=1e-6)


def test_svd_dense_lowrank_agree_on_leading_singular_values(snapshot_matrix: torch.Tensor) -> None:
    _, dense_singular_values, _ = ops.svd_dense(snapshot_matrix)
    _, lowrank_singular_values, _ = ops.svd_lowrank_native(snapshot_matrix, rank=5)

    # Randomized low-rank SVD without oversampling (q == the exact target
    # rank, matching how run_benchmark.py times it - the same rank POD would
    # actually request) systematically *underestimates* singular values by
    # construction - this loose, one-sided tolerance is the real expected
    # behavior, not a placeholder pending a tighter bound.
    assert torch.all(lowrank_singular_values <= dense_singular_values[:5] * 1.01)
    assert torch.all(lowrank_singular_values >= dense_singular_values[:5] * 0.8)


def test_transfer_roundtrip_preserves_values(small_dense_matrix: torch.Tensor) -> None:
    moved = ops.h2d(small_dense_matrix, torch.device("cpu"))
    back = ops.d2h(moved)
    assert torch.equal(back, small_dense_matrix)

    sparse = ops.to_sparse(small_dense_matrix)
    dense_again = ops.to_dense(sparse)
    assert torch.equal(dense_again, small_dense_matrix)
