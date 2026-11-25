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

Roles along the preparation chain: ``GaussSeidelSmoother.prepare`` and
``GaussSeidelMemo.prepare`` (``smoothers.py``) are the smoother protocol and
its per-matrix memo; ``prepare_gauss_seidel`` here only picks the format;
``prepare_csr_gauss_seidel`` / ``prepare_dense_gauss_seidel`` build the chosen
frozen ``Prepared*`` object; ``PreparedTriangularSolve`` is the kernel inside.
"""

from __future__ import annotations

import torch

from torchalg.sparse.kernels.triangular import level_schedule

from ._dense_gauss_seidel import PreparedDenseGaussSeidel, prepare_dense_gauss_seidel
from ._gauss_seidel import PreparedSymmetricGaussSeidel, prepare_csr_gauss_seidel

type PreparedGaussSeidel = PreparedSymmetricGaussSeidel | PreparedDenseGaussSeidel

DENSE_ALWAYS_MAX_N = 1500
"""At or below this size the dense sweep wins at every measured density."""

CSR_ALWAYS_MIN_N = 4000
"""At or above this size the CSR sweep wins at every measured density."""

DENSE_MS_PER_N_SQUARED = 8e-7
"""Measured cost of one dense symmetric sweep pair, in ms per ``n^2``."""

CSR_MS_PER_LEVEL = 0.06
"""Measured cost of one CSR symmetric sweep pair, in ms per dependency level."""


def prepare_gauss_seidel(matrix: torch.Tensor) -> PreparedGaussSeidel:
    """Prepare symmetric Gauss-Seidel on ``matrix`` in the cheaper of the two formats.

    Inside the size band the forward level schedule that estimates the CSR cost
    is handed to the CSR sweep, which needs the same schedule, so it is built once.

    Args:
        matrix (torch.Tensor): Sparse CSR matrix ``A``, shape ``(n, n)``.

    Returns:
        PreparedGaussSeidel: Same result either way; only the cost differs.
    """
    n = matrix.shape[0]
    if n <= DENSE_ALWAYS_MAX_N:
        return prepare_dense_gauss_seidel(matrix)
    if n >= CSR_ALWAYS_MIN_N:
        return prepare_csr_gauss_seidel(matrix)
    schedule = level_schedule(matrix, direction="forward")
    if DENSE_MS_PER_N_SQUARED * n * n < CSR_MS_PER_LEVEL * len(schedule.level_sizes):
        return prepare_dense_gauss_seidel(matrix)
    return prepare_csr_gauss_seidel(matrix, forward_schedule=schedule)
