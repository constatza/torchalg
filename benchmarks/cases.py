"""Seeded matrix cases for benchmarks.

Two families, both SPD and built without files so runs are reproducible anywhere:

- ``lap2d``: 5-point lattice Laplacian (≈5 nnz/row), the cheap scaling workhorse.
- ``box3d``: 27-point box stencil on a 3D grid (≈27 nnz/row, diagonally dominant), the
  closest synthetic stand-in for the paper's spectral-3d matrices (n = 2946, ≈37 nnz/row)
  whose heavier rows make dense-versus-sparse costs visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import torch
from scipy.sparse import csr_matrix, identity, kron

from benchmarks.sparse_dense.matrices import (
    lattice_laplacian,
    to_torch_dense,
    to_torch_sparse_csr,
)

BENCHMARK_SEED = 20240611
BOX_DIAGONAL_MARGIN = 0.5
"""Added to the 26 neighbour weights so the box stencil is strictly diagonally dominant."""


class MatrixFamily(StrEnum):
    """Synthetic matrix families available to benchmarks."""

    LAP2D = "lap2d"
    BOX3D = "box3d"


@dataclass(frozen=True, slots=True)
class MatrixCase:
    """A named benchmark matrix, convertible to dense or CSR on demand."""

    family: MatrixFamily
    side: int
    scipy_matrix: csr_matrix

    @property
    def n(self) -> int:
        return int(self.scipy_matrix.shape[0])

    @property
    def nnz(self) -> int:
        return int(self.scipy_matrix.nnz)

    @property
    def id(self) -> str:
        return f"{self.family}-n{self.n}"

    def dense(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return to_torch_dense(self.scipy_matrix, dtype).to(device)

    def csr(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return to_torch_sparse_csr(self.scipy_matrix, dtype).to(device)

    def rhs(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        generator = torch.Generator().manual_seed(BENCHMARK_SEED)
        return torch.randn(self.n, dtype=dtype, generator=generator).to(device)


def _box_stencil_3d(side: int) -> csr_matrix:
    # The 27-point box is (I + shift + shiftᵀ) Kronecker-ed over the three axes.
    line = csr_matrix(
        np.diag(np.ones(side)) + np.diag(np.ones(side - 1), 1) + np.diag(np.ones(side - 1), -1)
    )
    box = kron(kron(line, line), line, format="csr")
    box.setdiag(0)
    box.eliminate_zeros()
    degree = float(box.sum(axis=1).max())
    return (-box + identity(box.shape[0]) * (degree + BOX_DIAGONAL_MARGIN)).tocsr()


def lap2d_case(target_n: int) -> MatrixCase:
    side = max(2, round(target_n**0.5))
    return MatrixCase(MatrixFamily.LAP2D, side, lattice_laplacian(side, 2).tocsr())


def box3d_case(target_n: int) -> MatrixCase:
    side = max(3, round(target_n ** (1 / 3)))
    return MatrixCase(MatrixFamily.BOX3D, side, _box_stencil_3d(side))


PAPER_LIKE_N = 2946
"""Matrix size of the paper's spectral-3d comparison; ``box3d`` rounds to 14**3 = 2744."""

SIZE_LADDER = (484, 2744, 10000)
"""Default sweep: tiny (launch-bound), paper-like (n ≈ 3000), and large (bandwidth-bound)."""


def default_cases() -> tuple[MatrixCase, ...]:
    return (lap2d_case(SIZE_LADDER[0]), box3d_case(PAPER_LIKE_N), lap2d_case(SIZE_LADDER[2]))
