"""Choice between the level-scheduled CSR and the dense-backed Gauss-Seidel sweep.

The CSR sweep costs about one kernel launch per dependency level and is flat in
``n``; the dense sweep costs ``O(n^2)`` whatever the density. Measured on CPU
(6 threads, ``docs/plan.md`` Phase 4): dense is 20-50x faster below ``n = 300``,
the two tie near ``n = 2500-2750`` for both 5 and 23 nonzeros per row, and CSR wins
from about ``n = 3600``. Density therefore moves the crossover only through the
level count (99 levels at ``n = 2500`` for a 5-point stencil, 92 at ``n = 2744`` for a
27-point stencil), so the decision is by size outside a band and by an estimated
cost inside it, using the level count the CSR path needs anyway.

The constants are CPU estimates. On CUDA a dense ``trsm`` is faster and a launch
costs more, so the crossover there is higher and unmeasured.
"""

from __future__ import annotations

import torch

from torchalg.sparse.kernels.triangular import level_schedule

from ._dense_gauss_seidel import DenseBackedSymmetricGaussSeidel
from ._gauss_seidel import PreparedSymmetricGaussSeidel

type SymmetricGaussSeidel = PreparedSymmetricGaussSeidel | DenseBackedSymmetricGaussSeidel

DENSE_ALWAYS_MAX_N = 1500
"""At or below this size the dense sweep wins at every measured density."""

CSR_ALWAYS_MIN_N = 4000
"""At or above this size the CSR sweep wins at every measured density."""

DENSE_MS_PER_N_SQUARED = 8e-7
"""Measured cost of one dense symmetric sweep pair, in ms per ``n^2``."""

CSR_MS_PER_LEVEL = 0.06
"""Measured cost of one CSR symmetric sweep pair, in ms per dependency level."""


def _forward_level_count(matrix: torch.Tensor) -> int:
    return len(level_schedule(matrix, direction="forward").level_sizes)


def _prefers_dense(matrix: torch.Tensor) -> bool:
    n = matrix.shape[0]
    if n <= DENSE_ALWAYS_MAX_N:
        return True
    if n >= CSR_ALWAYS_MIN_N:
        return False
    return DENSE_MS_PER_N_SQUARED * n * n < CSR_MS_PER_LEVEL * _forward_level_count(matrix)


def prepare_gauss_seidel(matrix: torch.Tensor) -> SymmetricGaussSeidel:
    """Prepare symmetric Gauss-Seidel on ``matrix`` in the cheaper of the two formats.

    Args:
        matrix (torch.Tensor): Sparse CSR matrix ``A``, shape ``(n, n)``.

    Returns:
        SymmetricGaussSeidel: Same result either way; only the cost differs.
    """
    if _prefers_dense(matrix):
        return DenseBackedSymmetricGaussSeidel.build(matrix)
    return PreparedSymmetricGaussSeidel.build(matrix)
