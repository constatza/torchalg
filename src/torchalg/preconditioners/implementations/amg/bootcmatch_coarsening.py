"""BootCMatch coarsening strategy (one hierarchy level), dense sibling.

Implements ``CoarseningStrategy`` (``torchalg.multigrid.protocols``) using
D'Ambra, Filippone, Vassilevski's compatible-weighted-matching coarsening
(ACM TOMS 44(4), 2018, Algorithm 2): per level, compute BootCMatch edge
weights (Component A - ``_bootcmatch_weights.bootcmatch_edge_weights``),
run the parallel pointer-based matching (Component B -
``_bootcmatch_matching.bootcmatch_parallel_matching``), then build the
tentative prolongator from the matching (Component C -
``torchalg.utils.bootcmatch_aggregation.bootcmatch_aggregates_from_matching``,
shared with the sparse sibling since it is format-agnostic).

Same per-level shape as ``bootstrap.BAMGCoarsening``/
``coarsening.AggregationCoarsening`` elsewhere in this package: holds the
current smooth vector ``w`` keyed by matrix dimension (restricted
level-to-level exactly like those classes restrict their test vectors),
and ``build_transfer`` does the per-level work.

``P`` is materialized dense via a plain scatter-assignment (the same
pattern ``BAMGCoarsening._prolongation`` already uses for its own
coarse-identity-row scatter step) - this class never writes a new
materialization kernel. The Galerkin coarse operator reuses a plain
``P.T @ A @ P`` matmul chain, exactly as the dense ``BAMGCoarsening``
already does for its own Galerkin product.
"""

from __future__ import annotations

import torch

from torchalg.utils.bootcmatch_aggregation import bootcmatch_aggregates_from_matching
from torchalg.utils.dense_transfer import DenseTransferOperator

from ._bootcmatch_matching import bootcmatch_parallel_matching
from ._bootcmatch_weights import bootcmatch_edge_weights

__all__ = ["BootCMatchCoarsening"]


class BootCMatchCoarsening:
    """One coarse level of BootCMatch (TOMS Algorithm 2), dense.

    Stateful ``CoarseningStrategy``: holds the current smooth vector ``w``
    keyed by matrix dimension, restricted to the coarse grid at the end of
    every ``build_transfer`` call, the same dimension-keyed-dict pattern
    ``BAMGCoarsening`` uses for its test vectors.

    Args:
        w (torch.Tensor): Smooth vector for this level, shape ``(n,)``.
    """

    def __init__(self, w: torch.Tensor) -> None:
        """Store the finest-level smooth vector.

        Args:
            w (torch.Tensor): Smooth vector, shape ``(n,)``.
        """
        self._vectors: dict[int, torch.Tensor] = {w.shape[0]: w}
        self._last_prolongation: torch.Tensor | None = None

    def smooth_vector_for(self, matrix: torch.Tensor) -> torch.Tensor:
        """Currently stored smooth vector for ``matrix``'s dimension.

        Args:
            matrix (torch.Tensor): Level matrix, shape ``(n, n)``.

        Returns:
            torch.Tensor: Smooth vector, shape ``(n,)``.
        """
        return self._vectors[matrix.shape[0]]

    def set_smooth_vector(self, matrix: torch.Tensor, w: torch.Tensor) -> None:
        """Overwrite the stored smooth vector for ``matrix``'s dimension.

        Args:
            matrix (torch.Tensor): Level matrix, shape ``(n, n)``.
            w (torch.Tensor): New smooth vector, shape ``(n,)``.
        """
        self._vectors[matrix.shape[0]] = w

    @property
    def last_prolongation(self) -> torch.Tensor:
        """Dense prolongation tensor ``P`` from the most recent ``build_transfer`` call.

        Returns:
            torch.Tensor: Prolongation matrix, shape ``(n, n_c)``.

        Raises:
            RuntimeError: If ``build_transfer`` has not been called yet.
        """
        if self._last_prolongation is None:
            raise RuntimeError("build_transfer has not been called yet")
        return self._last_prolongation

    def build_transfer(self, A: torch.Tensor) -> tuple[torch.Tensor, DenseTransferOperator]:
        """Build one BootCMatch coarse level from dense ``A``.

        Cost:
            Dense ``(n, n)`` matrix, ``c`` the coarse dimension (about ``n / 2``,
            one column per matched pair or singleton), ``R`` the matching rounds
            (data dependent; the loop runs ``R`` rounded up to a multiple of 4),
            ``V = 8`` bytes per ``float64``, once per level. Edge weights (eq. 10,
            every ``(i, j)`` position, then masked to the stored edges):
            ``W`` about ``10n^2``, ``Q`` about ``20Vn^2``, about 20 kernels of
            constant depth. Matching: per round an ``n^2`` mask, ``where``, copy and
            row ``argmax`` (``n^2`` comparisons, ``Q`` about ``5Vn^2``, depth
            ``ceil(log2 n) + O(1)``, about 18 kernels), with one host read of the
            stop condition every fourth round. Aggregates: ``O(n)`` vectorized work
            and one host read of ``c``. Prolongator scatter into a dense ``n x c``
            tensor, Galerkin ``(P^T A) P`` and the coarse smooth vector ``P^T w``.
            In total ``W = 2cn^2 + 2c^2 n + 2nc + (10 + R)n^2 + O(n)``,
            ``Q`` about ``V(20 + 5R)n^2 + 2Vn c``,
            ``D = R(ceil(log2 n) + O(1)) + O(log n)``, ``L = 18R + 50`` approximately and
            ``S = ceil(R / 4) + 1`` (the stop checks and the coarse dimension).
            Roofline: the Galerkin GEMMs are compute-bound; the weight and matching
            passes are memory-bound ``n^2`` sweeps.
            Work-depth: every round is fully parallel over the ``n^2`` positions, so
            the depth is set by ``R``, not by ``n`` (unlike the host-sequential
            aggregation of ``AggregationCoarsening``).

        Args:
            A (torch.Tensor): Fine-grid matrix, shape ``(n, n)``.

        Returns:
            tuple[torch.Tensor, DenseTransferOperator]: ``(A_coarse, transfer)``.
        """
        w = self._vectors[A.shape[0]]
        weights, fine_only_mask = bootcmatch_edge_weights(w, A)
        match_of = bootcmatch_parallel_matching(weights, fine_only_mask)
        row, col, value, n_coarse = bootcmatch_aggregates_from_matching(w, match_of)

        n = A.shape[0]
        prolongation = torch.zeros(n, n_coarse, dtype=A.dtype, device=A.device)
        prolongation[row, col] = value.to(A.dtype)

        coarse_matrix = prolongation.T @ A @ prolongation
        self._last_prolongation = prolongation
        self._vectors[coarse_matrix.shape[0]] = prolongation.T @ w
        return coarse_matrix, DenseTransferOperator(prolongation)
