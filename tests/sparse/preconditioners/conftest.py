"""Fixtures for sparse preconditioner tests: same-structure and changed-structure variants.

``poisson_2d_csr`` (9x9, 5-point stencil) is the base matrix; every variant keeps
the dtype and ``n`` so only the stored structure or non-zero mask differs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from torchalg.sparse.kernels.index_dtype import compact_csr_indices
from torchalg.sparse.preconditioners import _pattern_index
from torchalg.sparse.preconditioners._ic0_pattern import IC0Pattern, analyze_ic0
from torchalg.sparse.preconditioners._ilu0_pattern import ILU0Pattern, analyze_ilu0

if TYPE_CHECKING:
    from collections.abc import Callable

PERTURBATION = 0.1
"""Relative size of the seeded value perturbation (keeps diagonal dominance)."""

SCALE = 1.5
"""Uniform scale applied to every stored value of the same-structure variant."""


def _with_values(matrix: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return torch.sparse_csr_tensor(
        matrix.crow_indices(),
        matrix.col_indices(),
        values,
        size=matrix.shape,
        check_invariants=False,
    )


@pytest.fixture
def poisson_2d_rescaled_csr(poisson_2d_csr: torch.Tensor, test_seed: int) -> torch.Tensor:
    """Same stored structure and non-zero mask as ``poisson_2d_csr``, new values."""
    generator = torch.Generator().manual_seed(test_seed)
    noise = torch.rand(poisson_2d_csr._nnz(), generator=generator, dtype=poisson_2d_csr.dtype)
    values = poisson_2d_csr.values() * SCALE * (1.0 + PERTURBATION * noise)
    return _with_values(poisson_2d_csr, values)


@pytest.fixture
def poisson_2d_explicit_zero_csr(poisson_2d_csr: torch.Tensor) -> torch.Tensor:
    """Same ``crow``/``col`` as ``poisson_2d_csr`` with one lower entry stored as zero."""
    row = torch.repeat_interleave(
        torch.arange(poisson_2d_csr.shape[0]),
        poisson_2d_csr.crow_indices().diff(),
    )
    lower_offdiagonal = (poisson_2d_csr.col_indices() < row).nonzero()[0, 0]
    values = poisson_2d_csr.values().clone()
    values[lower_offdiagonal] = 0.0
    return _with_values(poisson_2d_csr, values)


@pytest.fixture
def poisson_2d_nan_entry_csr(poisson_2d_csr: torch.Tensor) -> torch.Tensor:
    """Same ``crow``/``col`` as ``poisson_2d_csr`` with one lower entry stored as NaN."""
    row = torch.repeat_interleave(
        torch.arange(poisson_2d_csr.shape[0]),
        poisson_2d_csr.crow_indices().diff(),
    )
    lower_offdiagonal = (poisson_2d_csr.col_indices() < row).nonzero()[0, 0]
    values = poisson_2d_csr.values().clone()
    values[lower_offdiagonal] = float("nan")
    return _with_values(poisson_2d_csr, values)


@pytest.fixture
def poisson_2d_extra_entry_csr(poisson_2d_dense: torch.Tensor) -> torch.Tensor:
    """``poisson_2d_dense`` plus one symmetric coupling: same ``n``, different ``crow``/``col``."""
    dense = poisson_2d_dense.clone()
    dense[0, -1] = dense[-1, 0] = -0.25
    return dense.to_sparse_csr()


@pytest.fixture
def indefinite_csr(torch_dtype: torch.dtype) -> torch.Tensor:
    """2x2 symmetric indefinite matrix: IC(0) breaks down at the second pivot."""
    return torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch_dtype).to_sparse_csr()


@pytest.fixture
def poisson_2d_int32_csr(poisson_2d_csr: torch.Tensor) -> torch.Tensor:
    """``poisson_2d_csr`` with int32 ``crow``/``col`` (the width apply-time operators use)."""
    return compact_csr_indices(poisson_2d_csr)


@pytest.fixture
def tridiagonal_csr(torch_dtype: torch.dtype) -> torch.Tensor:
    """6x6 SPD tridiagonal matrix: row 5 stores columns 4 and 5."""
    n = 6
    dense = 2.0 * torch.eye(n, dtype=torch_dtype)
    dense -= torch.diag(torch.ones(n - 1, dtype=torch_dtype), 1)
    dense -= torch.diag(torch.ones(n - 1, dtype=torch_dtype), -1)
    return dense.to_sparse_csr()


@pytest.fixture
def redirect_last_row_entry() -> Callable[[torch.Tensor], None]:
    """Edit in place the first stored column of the last row to column 3 (was ``n - 2``)."""

    def redirect(matrix: torch.Tensor) -> None:
        matrix.col_indices()[matrix.crow_indices()[-2]] = 3

    return redirect


@pytest.fixture
def single_entry_csr(torch_dtype: torch.dtype) -> torch.Tensor:
    """1x1 SPD matrix."""
    return torch.tensor([[4.0]], dtype=torch_dtype).to_sparse_csr()


@pytest.fixture
def diagonal_csr(torch_dtype: torch.dtype) -> torch.Tensor:
    """Diagonal-only SPD matrix: no off-diagonal entry, so every elimination level is empty."""
    return torch.diag(torch.tensor([2.0, 4.0, 8.0], dtype=torch_dtype)).to_sparse_csr()


@pytest.fixture
def infinite_corner_csr(torch_dtype: torch.dtype) -> torch.Tensor:
    """5x5 lower matrix with ``a00 = inf`` whose row 4 couples columns 1 and 2 but ``(2, 1)`` is absent.

    The absent ``(2, 1)`` coupling is an unfound gather position: it points at
    storage index 0, which holds the infinite ``a00``.
    """
    inf = float("inf")
    dense = torch.tensor(
        [
            [inf, 0.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 9.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 16.0, 0.0],
            [0.0, 2.0, 3.0, 0.0, 27.0],
        ],
        dtype=torch_dtype,
    )
    return dense.to_sparse_csr()


@pytest.fixture
def explicit_zero_diagonal_csr(torch_dtype: torch.dtype) -> torch.Tensor:
    """3x3 tridiagonal whose middle diagonal entry is stored explicitly as zero."""
    return torch.sparse_csr_tensor(
        torch.tensor([0, 2, 5, 7]),
        torch.tensor([0, 1, 0, 1, 2, 1, 2]),
        torch.tensor([4.0, -1.0, -1.0, 0.0, -1.0, -1.0, 4.0], dtype=torch_dtype),
        size=(3, 3),
        check_invariants=False,
    )


@pytest.fixture
def poisson_2d_rescaled_int32_csr(poisson_2d_rescaled_csr: torch.Tensor) -> torch.Tensor:
    """``poisson_2d_rescaled_csr`` with int32 ``crow``/``col``."""
    return compact_csr_indices(poisson_2d_rescaled_csr)


@pytest.fixture
def ramp_rhs(poisson_2d_csr: torch.Tensor) -> torch.Tensor:
    """Deterministic right-hand side ``1, 2, ..., n`` matching ``poisson_2d_csr``."""
    return torch.arange(1, poisson_2d_csr.shape[0] + 1, dtype=poisson_2d_csr.dtype)


@pytest.fixture
def narrow_pattern(poisson_2d_csr: torch.Tensor) -> IC0Pattern:
    """Pattern of ``poisson_2d_csr`` under the real int32 limit (indices fit, so int32)."""
    return analyze_ic0(poisson_2d_csr)


@pytest.fixture
def wide_pattern(poisson_2d_csr: torch.Tensor, monkeypatch: pytest.MonkeyPatch) -> IC0Pattern:
    """Pattern of ``poisson_2d_csr`` with the limit below its size, so indices stay int64."""
    monkeypatch.setattr(_pattern_index, "INT32_INDEX_LIMIT", 1)
    return analyze_ic0(poisson_2d_csr)


@pytest.fixture
def asymmetric_csr(torch_dtype: torch.dtype) -> torch.Tensor:
    """Tridiagonal matrix with only the upper off-diagonal perturbed (non-symmetric)."""
    n = 8
    dense = 2.0 * torch.eye(n, dtype=torch_dtype)
    dense -= torch.diag(torch.ones(n - 1, dtype=torch_dtype), 1)
    dense -= torch.diag(torch.ones(n - 1, dtype=torch_dtype), -1)
    dense += 0.3 * torch.diag(torch.ones(n - 1, dtype=torch_dtype), 1)
    return dense.to_sparse_csr()


@pytest.fixture
def asymmetric_rescaled_csr(asymmetric_csr: torch.Tensor, test_seed: int) -> torch.Tensor:
    """Same stored structure as ``asymmetric_csr``, new values."""
    generator = torch.Generator().manual_seed(test_seed)
    noise = torch.rand(asymmetric_csr._nnz(), generator=generator, dtype=asymmetric_csr.dtype)
    return _with_values(
        asymmetric_csr, asymmetric_csr.values() * SCALE * (1.0 + PERTURBATION * noise)
    )


@pytest.fixture
def narrow_ilu0_pattern(poisson_2d_csr: torch.Tensor) -> ILU0Pattern:
    """ILU(0) pattern of ``poisson_2d_csr`` under the real int32 limit (indices fit, so int32)."""
    return analyze_ilu0(poisson_2d_csr)


@pytest.fixture
def wide_ilu0_pattern(poisson_2d_csr: torch.Tensor, monkeypatch: pytest.MonkeyPatch) -> ILU0Pattern:
    """ILU(0) pattern of ``poisson_2d_csr`` with the limit below its size, so indices stay int64."""
    monkeypatch.setattr(_pattern_index, "INT32_INDEX_LIMIT", 1)
    return analyze_ilu0(poisson_2d_csr)
