"""Bootstrap AMG (BAMG) coarsening strategy, sparse-CSR sibling.

**Recommended for dense use; prefer** ``BootCMatchPreconditioner``
(``torchalg.sparse.preconditioners.amg.bootcmatch``) **for sparse/large-scale
use.** This class's coarsening is driven by compatible-relaxation's greedy,
strict-priority-order selection (``_compatible_relaxation
._independent_set_of``) - each decision depends on every earlier one, so it
is a CPU-resident Python loop with no batched-tensor reformulation, by the
algorithm's own mathematical structure, not an implementation gap (see that
module's own docstring). It remains correct and is kept here - dense and
sparse stay full siblings, oracle-tested against each other, per this
project's own convention - but it does not scale the way BootCMatch's
round-based parallel matching does: BootCMatch is a different,
also-published Bootstrap AMG coarsening (compatible *weighted matching*
rather than compatible relaxation) whose coarse-point selection is a fixed
number of fully-batched rounds instead of one Python iteration per node,
and it is the preconditioner to reach for once a sparse system is large
enough that setup time matters (measured ~100x+ faster setup at N=15,876 in
this codebase's own benchmark). See ``bootcmatch.py``'s module docstring for
the paper references.

Sparse-CSR counterpart of
``preconditioners.implementations.amg.bootstrap.BAMGCoarsening`` (dense;
kept unmodified for comparison).
Same algorithm, separate implementation, tested against the dense result: the
sparse tree never imports the dense tree (enforced by ``tach.toml``).
Wires together this session's new sparse kernels
(``kernels.algebraic_distance.sparse_algebraic_distance``,
``kernels.strength.sparse_strength_graph``,
``kernels.depth_neighborhood.sparse_depth_neighborhood``,
``kernels.lsr_correction.sparse_lsr_correction``,
``kernels.prolongation.sparse_interpolation_prolongation``) with the
format-agnostic promoted leaves (``torchalg.utils.test_vector_weights``,
``torchalg.utils.ls_interpolation``) and this package's own
``_compatible_relaxation.compatible_relaxation_coarsening``, following
the [BAMG11]/[STATUS14] per-level setup block exactly as the dense
class does.

Every change from the dense class follows directly from storage format, not
a different algorithm:

- **Transpose-needing sparse-sparse/sparse-dense products** (``T_l = P_l^T
  P_l``, composite prolongation chaining ``P_0 P_1``, ``P^T @
  test_vectors``) route through COO for the transpose step, mirroring
  ``kernels.galerkin.form_sparse_sparse``'s own documented
  CSR-transpose-yields-CSC workaround.
- **``T_l`` is never formed.** The only use of ``T_l = C^T C`` is the
  weight numerator ``<T v, v> = ||C v||^2`` with dense ``test_vectors``, so
  the sparse composite interpolation ``C`` itself is passed down and
  multiplied against the test vectors (one sparse-dense product per level).
- **The LS-ring/coarse-neighbor candidate lookup** (dense:
  ``torch.nonzero(coarse_neighborhood, as_tuple=False).tolist()`` grouped by
  row) is replaced by directly reading ``sparse_depth_neighborhood``'s own
  ``crow_indices()``/``col_indices()`` per fine row - *simpler* than the
  dense version, not more complex, since CSR already hands back row-grouped
  neighbor lists for free (same precedent as SA-AMG's own sparse
  ``build_transfer``).
- **The empty-interpolatory-set fallback**'s strength lookup
  (``distance_cpu[i]``, a dense row) becomes one batched sparse lookup over every
  fallback row (``_strongest_candidates_sparse``, searchsorted over the stored
  row-major keys, 0.0 where absent - exactly matching what the dense row's
  zero-fill outside the neighborhood pattern already means).
- **``P`` is built directly as sparse CSR** via
  ``sparse_interpolation_prolongation`` from accumulated ``(row, col,
  value)`` triples (coarse-point identity rows plus each fine row's
  LS-fitted entries), never materialized dense at any point - each row
  contributes at most ``caliber`` nonzeros, provably sparse by construction,
  the same shape SA-AMG's own sparse ``P`` already has.

``select_interpolatory_set``/``ls_interpolation_row`` (from
``torchalg.utils.ls_interpolation``) and ``_strongest_candidate`` (identical
to the dense class's own static method, operating only on already-gathered
dense tensors) are otherwise unchanged from the dense algorithm - confirmed
format-agnostic, see those modules' own docstrings.

**Known numerical sensitivity, not a correctness bug: ``caliber >= k`` (the
test-vector count) breaks dense-vs-sparse bit-parity.** ``select_
interpolatory_set``'s greedy caliber-growth picks the candidate minimizing
the LS residual at each step ([AD11] Sec. 4.3). With ``k`` test vectors, the
LS fit becomes exact (residual pinned at machine-epsilon, not merely small)
once any ``k`` linearly-independent points are chosen - so every candidate
beyond the ``k``-th is "chosen" by comparing residuals that are already at
the floating-point noise floor (confirmed by direct measurement: residuals
differing by ~1e-30 between two candidates whose *true* residual is
identically zero). Dense ``@`` and sparse CSR matmul accumulate test-vector
weights/fit-vectors in a different summation order, so at exactly this
noise floor they can legitimately disagree on which candidate "wins" -
correct behavior, not a bug (the project's existing "float summation order
changes" precedent, see ``sparse_ic0``'s own docstring, for a quantity
whose *value* is tolerance-robust; this is the sharper case where a
*discrete choice* flips instead, which `assert_close`'s tolerance cannot
absorb the way it absorbs a perturbed value). Confirmed empirically: 8
random anisotropic-grid seeds with ``caliber=4 >= k=3`` test vectors all
diverged from the dense result (up to ~1.5 in the Galerkin-amplified
``A_coarse``, after a handful of per-row tie-flips); the identical 8 seeds
with ``caliber=3 < k=6`` all matched to ~1e-14. Callers who need
reproducible dense/sparse parity (e.g. this module's own tests) should keep
``caliber < k``; production use is unaffected either way, since any
candidate the greedy search picks at this noise floor is, by construction,
an equally valid (near-zero-residual) interpolation point.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from torchalg.multigrid import AMGPreconditioner
from torchalg.multigrid.bootstrap_setup import BootstrapAMGResult, BootstrapSetup
from torchalg.preconditioners.base import PreconditionerContext, PreconditionerNotReadyError
from torchalg.sparse.kernels.algebraic_distance import sparse_algebraic_distance
from torchalg.sparse.kernels.depth_neighborhood import sparse_depth_neighborhood
from torchalg.sparse.kernels.diagonal import sparse_diagonal
from torchalg.sparse.kernels.galerkin import form_sparse_sparse
from torchalg.sparse.kernels.lsr_correction import _NEAR_ZERO_DIAGONAL_TOL
from torchalg.sparse.kernels.prolongation import sparse_interpolation_prolongation
from torchalg.sparse.kernels.strength import sparse_strength_graph
from torchalg.sparse.preconditioners.amg.transfer import (
    SparseTransferOperator,
    TransferOperatorMemo,
)
from torchalg.utils.ls_interpolation import (
    batched_ls_interpolation_rows,
    batched_select_interpolatory_set,
    padded_candidate_sets,
)
from torchalg.utils.test_vector_weights import test_vector_weights
from torchalg.utils.weak_cache import weak_memo

from ._compatible_relaxation import compatible_relaxation_coarsening
from ._presets import PrebuiltCoarsening, gs_setup_cycle, prebuilt_cycle, seeded_draw
from .smoothers import GaussSeidelMemo, resolve_jacobi_default

if TYPE_CHECKING:
    from typing import Self

    from torchalg.multigrid.hierarchy import MultigridHierarchy
    from torchalg.multigrid.protocols import MultigridSmoother

_LSR_TARGET_FRACTION = 0.2
"""Matches the dense sibling's value - see its docstring."""

type NeighborhoodOf = Callable[[torch.Tensor, int], torch.Tensor]
"""``(matrix, depth) -> boolean CSR neighborhood``: ``sparse_depth_neighborhood`` or its memo."""


def _seeded_draw(seed: int) -> Callable[[int], torch.Tensor]:
    """Stateful uniform ``[0, 1)`` source seeded for reproducibility.

    Trivial duplicate of ``preconditioners.implementations.amg._presets
    .seeded_draw`` - same precedent as this package's own duplication of
    other small, entangled boilerplate (see ``_compatible_relaxation.py``'s
    module docstring) rather than a cross-tree import for one function this
    small.

    Args:
        seed (int): Generator seed.

    Returns:
        Callable[[int], torch.Tensor]: ``n -> float64`` tensor of length ``n``.
    """
    generator = torch.Generator().manual_seed(seed)
    return lambda n: torch.rand(n, generator=generator, dtype=torch.float64)


def _entry_rows(matrix: torch.Tensor) -> torch.Tensor:
    """Row index of every stored entry of a CSR matrix, without a host read.

    Args:
        matrix (torch.Tensor): Sparse CSR matrix.

    Returns:
        torch.Tensor: Long tensor, one row index per stored entry.
    """
    crow = matrix.crow_indices()
    return torch.repeat_interleave(
        torch.arange(matrix.shape[0], device=matrix.device),
        crow[1:] - crow[:-1],
        output_size=matrix.col_indices().numel(),
    )


def _symmetrize_sparse_bool(matrix: torch.Tensor) -> torch.Tensor:
    """Union-symmetrize a sparse CSR boolean adjacency (``M | M.T``), sparse CSR.

    Sparse CSR has no direct ``.t()`` (yields CSC); builds the transpose by
    swapping row/col indices directly instead, avoiding the CSR->CSC
    roundtrip entirely, then relies on ``coalesce()`` to merge any position
    stored in both directions into one entry before rebuilding as boolean.

    Args:
        matrix (torch.Tensor): Sparse CSR boolean matrix, shape ``(n, n)``.

    Returns:
        torch.Tensor: Sparse CSR boolean matrix, ``matrix | matrix.T``.
    """
    coo = matrix.to_sparse_coo().coalesce()
    row, col = coo.indices()
    sym_row = torch.cat([row, col])
    sym_col = torch.cat([col, row])
    sym_indices = torch.stack([sym_row, sym_col])
    deduped = torch.sparse_coo_tensor(
        sym_indices,
        torch.ones(sym_row.numel(), dtype=torch.float64, device=matrix.device),
        size=matrix.shape,
        check_invariants=False,
    ).coalesce()
    return torch.sparse_coo_tensor(
        deduped.indices(),
        torch.ones(deduped.indices().shape[1], dtype=torch.bool, device=matrix.device),
        size=matrix.shape,
        check_invariants=False,
    ).to_sparse_csr()


class BAMGCoarsening:
    """One coarse level of Bootstrap AMG, sparse-CSR sibling of the dense ``BAMGCoarsening``.

    Stateful ``CoarseningStrategy``: holds the current test vectors keyed by
    matrix dimension - see the dense sibling's docstring for the full
    per-level-block algorithm description (CR coarsening guided by the
    algebraic-distance strength graph, per-F-point LS/LSR interpolatory-set
    fitting, Galerkin coarse operator), unchanged here except for storage
    format.

    Args:
        test_vectors (torch.Tensor): Test vectors for this level, shape
            ``(n, k)``.
        relaxation (MultigridSmoother): Smoother CR binds to the level matrix
            once per coarsening (e.g. the sparse ``GaussSeidelSmoother()``).
        nu (int): CR sweeps per stage (Sec. 2.1 default ``5``).
        delta (float): CR stopping tolerance (Sec. 2.1 default ``0.7``).
        theta_ad (float): Algebraic-distance strength threshold (Sec. 2.2
            default ``0.5``).
        caliber (int): Maximum interpolatory-set size ``c``.
        gamma (float): Interpolatory-set caliber-growth penalization
            exponent (Sec. 3.3 default ``1.5``).
        use_lsr (bool): Apply the LSR residual correction (Sec. 3.2) before
            fitting; plain LS (Sec. 3.1) if ``False``.
        depth (int): Algebraic-distance search depth ``d``; the LS-ring
            candidate neighborhood uses ``d_LS = d + 2`` (eq. 4.5).
        draw (Callable[[int], torch.Tensor] | None): Uniform ``[0, 1)``
            random-vector source for CR; a seeded default if ``None``.
        seed (int): Seed of the default draw source.
    """

    def __init__(
        self,
        test_vectors: torch.Tensor,
        relaxation: MultigridSmoother,
        *,
        nu: int = 5,
        delta: float = 0.7,
        theta_ad: float = 0.5,
        caliber: int = 4,
        gamma: float = 1.5,
        use_lsr: bool = True,
        depth: int = 1,
        draw: Callable[[int], torch.Tensor] | None = None,
        seed: int = 0,
        transfer_operator: Callable[
            [torch.Tensor], SparseTransferOperator
        ] = SparseTransferOperator,
        neighborhood_of: NeighborhoodOf = sparse_depth_neighborhood,
    ) -> None:
        """Store hyperparameters and the finest-level test vectors.

        Args:
            test_vectors (torch.Tensor): Test vectors for this level.
            relaxation (MultigridSmoother): Smoother CR binds to the level matrix.
            nu (int): CR sweeps per stage.
            delta (float): CR stopping tolerance.
            theta_ad (float): Algebraic-distance strength threshold.
            caliber (int): Maximum interpolatory-set size.
            gamma (float): Caliber-growth penalization exponent.
            use_lsr (bool): Apply the LSR residual correction before fitting.
            depth (int): Algebraic-distance search depth.
            draw (Callable[[int], torch.Tensor] | None): Uniform ``[0, 1)``
                random-vector source for CR.
            seed (int): Seed of the default draw source.
            transfer_operator (Callable[[torch.Tensor], SparseTransferOperator]):
                Wraps each built prolongation. Pass the setup stage's
                ``TransferOperatorMemo`` so ``build_transfer`` and the
                hierarchy builder share one operator (one transpose) per
                prolongation instead of building a throwaway.
            neighborhood_of (NeighborhoodOf): Evaluates the LS-ring
                neighborhood of a level matrix. Pass a per-setup memo so the
                finest matrix's neighborhood, which depends on ``A`` alone, is
                built once across the initial and every improvement hierarchy.
        """
        self._neighborhood_of = neighborhood_of
        self._transfer_operator = transfer_operator
        self._relaxation = relaxation
        self._nu = nu
        self._delta = delta
        self._theta_ad = theta_ad
        self._caliber = caliber
        self._gamma = gamma
        self._use_lsr = use_lsr
        self._depth = depth
        self._draw = draw if draw is not None else _seeded_draw(seed)
        self._vectors: dict[int, torch.Tensor] = {test_vectors.shape[0]: test_vectors}
        self._last_prolongation: torch.Tensor | None = None
        self._composite: torch.Tensor | None = None

    def test_vectors_for(self, matrix: torch.Tensor) -> torch.Tensor:
        """Currently stored test vectors for ``matrix``'s dimension.

        Args:
            matrix (torch.Tensor): Level matrix, shape ``(n, n)``.

        Returns:
            torch.Tensor: Test vectors, shape ``(n, k)``.
        """
        return self._vectors[matrix.shape[0]]

    def set_test_vectors(self, matrix: torch.Tensor, vectors: torch.Tensor) -> None:
        """Overwrite the stored test vectors for ``matrix``'s dimension.

        Args:
            matrix (torch.Tensor): Level matrix, shape ``(n, n)``.
            vectors (torch.Tensor): New test vectors, shape ``(n, k)``.
        """
        self._vectors[matrix.shape[0]] = vectors

    @property
    def last_prolongation(self) -> torch.Tensor:
        """Sparse CSR prolongation tensor ``P`` from the most recent ``build_transfer`` call.

        Returns:
            torch.Tensor: Sparse CSR prolongation matrix, shape ``(n, n_c)``.

        Raises:
            RuntimeError: If ``build_transfer`` has not been called yet.
        """
        if self._last_prolongation is None:
            raise RuntimeError("build_transfer has not been called yet")
        return self._last_prolongation

    def build_transfer(self, A: torch.Tensor) -> tuple[torch.Tensor, SparseTransferOperator]:
        """Build one Bootstrap AMG coarse level from sparse CSR ``A``.

        Cost:
            Per level, ``z = nnz(A)``, ``k`` test vectors, ``s`` compatible-relaxation
            stages (data dependent), ``nu`` CR sweeps per stage, ``m`` fine rows,
            ``m_b`` rows at padded width ``c_b`` in each candidate-count chunk,
            ``caliber`` the interpolatory-set size, ``d`` the algebraic-distance depth. Stages:
            algebraic distance ``W = O(zk)`` plus its depth-``d`` neighborhood;
            compatible relaxation ``s (nu W_sweep + O(n))`` with ``W_sweep = 4z + 2n``
            (one Gauss-Seidel iteration, see ``compatible_relaxation_coarsening``) and
            ``D = s (nu D_sweep + n)``, the ``n`` being the host-sequential
            independent set, which dominates ``D`` of the whole step; the LSR
            correction ``W = 2zk``; the depth ``d + 2`` neighborhood
            (``sparse_depth_neighborhood``, memoized per level matrix and setup run,
            so the finest level pays it once, not once per hierarchy build);
            the interpolatory-set selection ``W = (8 caliber - 1) k sum_b m_b c_b`` over
            row-width chunks (each chunk padded to its own largest candidate count,
            not the global maximum) and the row fits ``O(m caliber^2 k)``
            (``batched_select_interpolatory_set``); the
            Galerkin product ``W = 2 sum nnz(P_k:) + 2 sum nnz((AP)_i:)``
            (``form_sparse_sparse``) and the restriction ``P^T V``, ``W = 2 nnz(P) k``.
            Totals: ``W = O(zk) + s (nu W_sweep + O(n)) + 2zk + W_sel + W_gal
            + 2 nnz(P) k``, ``D = s (nu D_sweep + n) + O(caliber log z)``,
            ``L = s (nu L_sweep + O(1)) + O(caliber + d)`` (the GS sweep has
            ``L_sweep = 6``; the selection, fits and products are a constant number of
            batched kernels per interpolatory-set step) and ``S = s + O(1)``: one
            convergence read per CR stage, and in the prolongation the ``nonzero``
            splits into C and F points, the candidate-block width, the trust test of
            each of the two row fits and one ``nonzero`` sizing the empty-selection
            fallback rows (whose strongest candidates are one batched ``searchsorted``
            lookup), never one read per fallback row. ``I`` is the neighborhood, the
            independent set and the candidate packing.
            Roofline: every stage has ``W/Q`` below 1 flop per byte (the index work
            has none), so the step is memory- and host-bound.
            Work-depth: the selection, fits and SpGEMMs are parallel (large ``W/D``);
            the CR coarsening is a host-sequential chain (``W/D`` about 1), so setup
            does not speed up with workers.

        Args:
            A (torch.Tensor): Sparse CSR fine-grid matrix, shape ``(n, n)``.

        Returns:
            tuple[torch.Tensor, SparseTransferOperator]: ``(A_coarse, transfer)``.
        """
        test_vectors = self._vectors[A.shape[0]]
        weights = test_vector_weights(test_vectors, A, composite=self._composite)
        distance = sparse_algebraic_distance(test_vectors, A, depth=self._depth, weights=weights)
        coarse_mask = self._coarse_mask(A, distance)
        prolongation = self._prolongation(A, test_vectors, coarse_mask, distance, weights)
        coarse_matrix = form_sparse_sparse(prolongation, A)
        self._last_prolongation = prolongation

        p_coo = prolongation.to_sparse_coo()
        restricted_vectors = torch.sparse.mm(p_coo.t(), test_vectors)
        self._vectors[coarse_matrix.shape[0]] = restricted_vectors

        if self._composite is None:
            self._composite = prolongation
        else:
            composed = torch.sparse.mm(self._composite.to_sparse_coo(), p_coo)
            self._composite = composed.to_sparse_csr()
        return coarse_matrix, self._transfer_operator(prolongation)

    def _coarse_mask(self, A: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
        """Compatible-relaxation coarse set, guided by the algebraic-distance graph.

        Args:
            A (torch.Tensor): Sparse CSR fine-grid matrix, shape ``(n, n)``.
            distance (torch.Tensor): Sparse CSR algebraic-distance matrix
                ``r``, shape ``(n, n)`` ([AD11] eq. 4.3).

        Returns:
            torch.Tensor: Boolean coarse mask, shape ``(n,)``.
        """
        all_fine = torch.ones(A.shape[0], dtype=torch.bool, device=A.device)
        directed = sparse_strength_graph(distance, all_fine, theta_ad=self._theta_ad)
        return compatible_relaxation_coarsening(
            A,
            self._relaxation,
            nu=self._nu,
            delta=self._delta,
            draw=self._draw,
            guidance_graph=_symmetrize_sparse_bool(directed),
        )

    def _fit_vectors(
        self,
        A: torch.Tensor,
        test_vectors: torch.Tensor,
        fine_points: torch.Tensor,
    ) -> torch.Tensor:
        """Test vectors the LS fit uses: LSR-corrected if enabled, else as-is.

        Args:
            A (torch.Tensor): Sparse CSR fine-grid matrix, shape ``(n, n)``.
            test_vectors (torch.Tensor): Test vectors, shape ``(n, k)``.
            fine_points (torch.Tensor): Long tensor of fine-point indices.

        Returns:
            torch.Tensor: Test vectors to fit, shape ``(n, k)``.
        """
        if not self._use_lsr or fine_points.numel() == 0:
            return test_vectors
        top_count = max(
            min(math.ceil(_LSR_TARGET_FRACTION * fine_points.numel()), fine_points.numel()), 1
        )
        residual = A @ test_vectors
        residual_magnitude = residual[fine_points].abs()
        top_local = torch.topk(residual_magnitude, top_count, dim=0).indices
        target_rows = fine_points[top_local]
        diagonal = sparse_diagonal(A)
        diagonal_at_targets = diagonal[target_rows]
        diagonal_safe = torch.where(
            diagonal_at_targets.abs() > _NEAR_ZERO_DIAGONAL_TOL,
            diagonal_at_targets,
            torch.ones_like(diagonal_at_targets),
        )
        residual_at_targets = torch.gather(residual, 0, target_rows)
        values_at_targets = (
            torch.gather(test_vectors, 0, target_rows) - residual_at_targets / diagonal_safe
        )
        corrected = test_vectors.clone()
        corrected.scatter_(0, target_rows, values_at_targets)
        return corrected

    @staticmethod
    def _strongest_candidate(candidates: torch.Tensor, strengths: torch.Tensor) -> torch.Tensor:
        """Fallback interpolatory set: the single algebraically closest candidate (dense strengths row).

        Args:
            candidates (torch.Tensor): Long tensor of candidate ``C``-indices.
            strengths (torch.Tensor): Algebraic distances ``r_ij`` from the
                target row ``i`` to every node, shape ``(n,)``.

        Returns:
            torch.Tensor: Long tensor of shape ``(1,)``.
        """
        return candidates[torch.argmax(strengths[candidates])].reshape(1)

    @staticmethod
    def _strongest_candidates_sparse(
        candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
        targets: torch.Tensor,
        distance: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback interpolatory point per row: its candidate with the largest stored ``r_ij``.

        Each ``(target, candidate)`` pair is looked up in ``distance``'s storage at once by
        its row-major key ``target * n + candidate``, which ascends over a CSR matrix with
        sorted columns; an unstored pair reads ``0.0``, the dense row's zero-fill. Equal
        strengths resolve to the lowest slot (``argmax`` returns the first maximum), so a
        row with no stored entry takes its first candidate.

        Args:
            candidates (torch.Tensor): Long tensor, shape ``(m, w)``, each row's candidates
                left-aligned.
            candidate_mask (torch.Tensor): Bool tensor, shape ``(m, w)``, a non-empty
                prefix per row.
            targets (torch.Tensor): Long tensor, shape ``(m,)``, each row's fine index.
            distance (torch.Tensor): Sparse CSR algebraic-distance matrix, shape ``(n, n)``.

        Returns:
            torch.Tensor: Long tensor, shape ``(m,)``, each row's chosen candidate.
        """
        n = distance.shape[1]
        stored_keys = _entry_rows(distance) * n + distance.col_indices()
        query = targets.unsqueeze(1) * n + candidates
        strengths = torch.zeros(candidates.shape, dtype=distance.dtype, device=distance.device)
        if stored_keys.numel() > 0:
            position = torch.searchsorted(stored_keys, query).clamp(max=stored_keys.numel() - 1)
            found = stored_keys[position] == query
            strengths = torch.where(found, distance.values()[position], strengths)
        strengths = torch.where(candidate_mask, strengths, float("-inf"))
        return candidates.gather(1, strengths.argmax(dim=1, keepdim=True)).squeeze(1)

    def _prolongation(
        self,
        A: torch.Tensor,
        test_vectors: torch.Tensor,
        coarse_mask: torch.Tensor,
        distance: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """LS/LSR-fitted sparse CSR prolongation ``P`` (identity on ``C``, fitted rows on ``F``).

        Cost:
            The depth ``d + 2`` neighborhood is one ``sparse_depth_neighborhood`` call
            per distinct level matrix per setup run (``neighborhood_of`` memo), not per
            hierarchy build, so the finest level's index-work ``I`` is paid once instead
            of once per bootstrap pass.

        Args:
            A (torch.Tensor): Sparse CSR fine-grid matrix, shape ``(n, n)``.
            test_vectors (torch.Tensor): Test vectors, shape ``(n, k)``.
            coarse_mask (torch.Tensor): Boolean coarse mask, shape ``(n,)``.
            distance (torch.Tensor): Sparse CSR algebraic-distance matrix
                ``r``, shape ``(n, n)``, used by the empty-interpolatory-set
                fallback.
            weights (torch.Tensor): Per-test-vector LS/LSR fit weights
                ``omega_kappa``, shape ``(k,)``, from ``test_vector_weights`` with the
                sparse CSR composite interpolation ``C`` (``T = C^T C``).

        Returns:
            torch.Tensor: Sparse CSR prolongation matrix, shape
            ``(n, n_c)``.
        """
        n = A.shape[0]
        device = A.device
        coarse_points = torch.nonzero(coarse_mask).flatten()
        fine_points = torch.nonzero(~coarse_mask).flatten()
        n_fine = fine_points.numel()
        n_coarse = coarse_points.numel()
        coarse_index = torch.full((n,), -1, dtype=torch.long, device=device)
        coarse_index[coarse_points] = torch.arange(n_coarse, device=device)

        fit_vectors = self._fit_vectors(A, test_vectors, fine_points)
        neighborhood = self._neighborhood_of(A, self._depth + 2)

        row_indices: list[torch.Tensor] = [coarse_points]
        col_indices: list[torch.Tensor] = [torch.arange(n_coarse, device=device)]
        values: list[torch.Tensor] = [torch.ones(n_coarse, dtype=A.dtype, device=device)]

        if n_fine > 0:
            fine_row, fine_col, fine_value = self._fine_interpolation_triples(
                neighborhood=neighborhood,
                coarse_mask=coarse_mask,
                coarse_points=coarse_points,
                coarse_index=coarse_index,
                fine_points=fine_points,
                fit_vectors=fit_vectors,
                weights=weights,
                distance=distance,
                n=n,
                device=device,
            )
            row_indices.append(fine_row)
            col_indices.append(fine_col)
            values.append(fine_value)

        row_tensor = torch.cat(row_indices)
        col_tensor = torch.cat(col_indices)
        value_tensor = torch.cat(values).to(A.dtype)
        return sparse_interpolation_prolongation(
            row_tensor, col_tensor, value_tensor, (n, n_coarse)
        )

    def _fine_interpolation_triples(
        self,
        *,
        neighborhood: torch.Tensor,
        coarse_mask: torch.Tensor,
        coarse_points: torch.Tensor,
        coarse_index: torch.Tensor,
        fine_points: torch.Tensor,
        fit_vectors: torch.Tensor,
        weights: torch.Tensor,
        distance: torch.Tensor,
        n: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized ``(row, col, value)`` triples for every fine row's LS-fitted entries.

        Replaces the old per-fine-row Python loop: one CSR-to-COO expansion
        of ``neighborhood`` to gather every fine row's coarse-neighbor
        candidates in a single pass, one padded-batch call each to
        ``batched_select_interpolatory_set``/``batched_ls_interpolation_rows``,
        and a vectorized mask-based scatter - see this module's caller
        docstring/the originating plan for the full derivation.

        Args:
            neighborhood (torch.Tensor): Sparse CSR LS-ring neighborhood,
                shape ``(n, n)``.
            coarse_mask (torch.Tensor): Boolean coarse mask, shape ``(n,)``.
            coarse_points (torch.Tensor): Long tensor of coarse indices.
            coarse_index (torch.Tensor): Long tensor, shape ``(n,)``, mapping
                a global coarse index to its column in ``P``.
            fine_points (torch.Tensor): Long tensor of fine indices.
            fit_vectors (torch.Tensor): Test vectors the LS fit uses, shape
                ``(n, k)``.
            weights (torch.Tensor): Per-test-vector weights, shape ``(k,)``.
            distance (torch.Tensor): Sparse CSR algebraic-distance matrix,
                shape ``(n, n)``, used only by the empty-interpolatory-set
                fallback.
            n (int): Fine-level dimension.
            device (torch.device): Common device of all of the above.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ``(row, col,
            value)`` triples, each a 1-D tensor, ready to concatenate with
            the coarse-identity-row triples and pass to
            ``sparse_interpolation_prolongation``.
        """
        n_fine = fine_points.numel()

        all_row = _entry_rows(neighborhood)
        all_col = neighborhood.col_indices()

        keep = (~coarse_mask)[all_row] & coarse_mask[all_col]
        cand_row = all_row[keep]
        cand_col = all_col[keep]

        fine_index = torch.full((n,), -1, dtype=torch.long, device=device)
        fine_index[fine_points] = torch.arange(n_fine, device=device)
        local_row = fine_index[cand_row]

        padded_candidates, candidate_mask = padded_candidate_sets(
            local_row, cand_col, n_fine, coarse_points
        )

        targets = fine_points
        chosen, chosen_mask = batched_select_interpolatory_set(
            padded_candidates,
            candidate_mask,
            fit_vectors,
            targets,
            weights,
            self._caliber,
            gamma=self._gamma,
        )
        rows = batched_ls_interpolation_rows(fit_vectors, targets, chosen, chosen_mask, weights)

        had_candidates = candidate_mask.any(dim=-1)
        empty_selection = (chosen_mask.sum(dim=-1) == 0) & had_candidates

        # Empty-interpolatory-set fallback (candidates existed but the greedy batched
        # selection accepted none): the strongest candidate of each such row, found for
        # all of them at once. `nonzero` is the one read; it sizes the fallback rows.
        affected = torch.nonzero(empty_selection).flatten()
        if affected.numel() > 0:
            chosen[affected, 0] = self._strongest_candidates_sparse(
                padded_candidates[affected], candidate_mask[affected], targets[affected], distance
            )
            chosen_mask[affected, 0] = True
            chosen_mask[affected, 1:] = False
            rows = batched_ls_interpolation_rows(fit_vectors, targets, chosen, chosen_mask, weights)

        flat_row = fine_points.unsqueeze(1).expand(-1, chosen.shape[1])[chosen_mask]
        flat_col = coarse_index[chosen[chosen_mask]]
        flat_value = rows[chosen_mask]
        return flat_row, flat_col, flat_value


class BootstrapAMGPreconditioner(AMGPreconditioner):
    """Bootstrap AMG preconditioner (BAMG) for sparse CSR systems, sparse-CSR sibling.

    **Recommended for dense use; prefer** ``BootCMatchPreconditioner``
    (``torchalg.sparse.preconditioners.amg.bootcmatch``) **for sparse/
    large-scale use** - see this module's own ``BAMGCoarsening`` docstring
    for why (compatible relaxation's coarse-point selection is sequential by
    the algorithm's own structure, not an implementation gap).

    Sparse-CSR counterpart of
    ``preconditioners.implementations.amg.bootstrap.BootstrapAMGPreconditioner``
    (dense; kept unmodified for comparison). Runs ``BootstrapSetup.run`` in
    ``setup()`` (the shared, format-agnostic outer loop promoted to
    ``torchalg.multigrid.bootstrap_setup`` this session - see that module's
    docstring), injecting this module's own ``BAMGCoarsening``,
    ``SparseTransferOperator``, the sparse Gauss-Seidel and this package's
    ``_presets.gs_setup_cycle`` in place of the dense equivalents. Every
    ``setup()`` builds a fresh ``BootstrapSetup`` around its own
    ``GaussSeidelMemo`` and ``TransferOperatorMemo``, so the level
    schedules of each level matrix and the transpose of each prolongation are
    built once per setup run, however many compatible-relaxation sweeps,
    test-vector relaxations and improvement cycles revisit them.
    Kept as a subclass (not converted to a factory function like
    ``vcycle_amg``/``wcycle_amg``): it overrides ``setup`` (running the
    setup algorithm, then ``_commit``) with
    genuinely different hierarchy-construction logic (the prebuilt-levels
    pattern), the same "earned inheritance" reasoning that keeps the dense
    class a subclass too (a preset subclasses only when it overrides behavior;
    pure constructor wiring stays a factory function).

    **MGE (``k_e``) is not exposed here** - multigrid-eigensolver enrichment
    has no sparse implementation yet (see
    ``torchalg.multigrid.bootstrap_setup``'s module docstring); this preset
    always runs with ``k_e=0``/``eigensolver=None``, matching the dense
    class's own historical (pre-MGE) default behavior exactly.

    Cost:
        Setup, once per matrix, ``N`` levels with ``n_l`` rows and ``z_l`` stored
        entries, ``H = n_bootstrap_cycles + 1`` hierarchy builds:
        ``BootstrapSetup.run``'s Cost with the sparse ``BAMGCoarsening.build_transfer``
        (``bt``) per level and build, plus ``n_bootstrap_cycles`` improvement passes;
        the discarded builds are the algorithm's design (the improvement passes refine
        test vectors, not the kept hierarchy). Each build also relaxes ``k_r`` test
        vectors by ``eta`` symmetric Gauss-Seidel iterations per level,
        ``eta (4 z_l + 2 n_l) k_r``, and each improvement pass runs ``eta``
        GS-smoothed V(1,1) cycles on the ``N - 1`` sub-hierarchies, ``k_r`` sum over
        sub-hierarchies of ``(9 z_l + 4 nnz(P_l) + O(n_l)) + 2c^2`` per cycle (``c``
        the coarsest order). Totals: ``W = H sum_l [W_bt,l + eta (4z_l + 2n_l) k_r]
        + (H - 1) eta W_improve``, ``D = H sum_l [D_bt,l + eta (lv_f,l + lv_b,l)]
        + (H - 1) eta D_improve`` with ``D_bt,l = O(n_l)`` per CR stage from the
        host-sequential independent set, ``L = H sum_l (L_bt,l + 6 eta) + (H - 1) eta
        L_improve`` and ``S = H sum_l S_bt,l + O(H N)``, dominated by the per-stage
        convergence reads (``O(H N s)``). Smoothers and transposes are prepared once
        per level matrix and shared across the run. The setup does not speed up with
        workers and is the largest term of the method's total cost.
        Apply, one V(``a``, ``b``) cycle with weighted-Jacobi smoothing, per
        non-coarsest level ``l``: ``2(a + b) z_l + 4 nnz(P_l) + 3(a + b) n_l`` flops
        and ``4(a + b) + 2`` launches, plus the coarsest cached-inverse product
        ``2c^2``: ``W = sum_l [2(a + b) z_l + 4 nnz(P_l) + 3(a + b) n_l] k + 2c^2 k``,
        ``L = sum_l [4(a + b) + 2] + 1``, ``S = 0``, and
        ``D = sum_l [(a + b)(ceil(log2 m_l) + 4) + 2 (ceil(log2 m_P) + 1)]
        + ceil(log2 c) + 1``, shallow because Jacobi and the transfers have no
        recurrence. The hierarchy is deeper than aggregation's (coarsening is
        slower), so the launch count, not the depth, dominates its apply.
        Roofline: apply ``W/Q`` is about 0.15 flop per byte (memory-bound).

    Args:
        k_r (int): Number of relaxation-derived test vectors.
        eta (int): Relaxation sweeps per test vector per level.
        n_bootstrap_cycles (int): Number of bootstrap-cycle passes.
        nu (int): CR sweeps per stage.
        delta (float): CR stopping tolerance.
        theta_ad (float): Algebraic-distance strength threshold.
        caliber (int): Maximum interpolatory-set size.
        gamma (float): Caliber-growth penalization exponent.
        use_lsr (bool): Apply the LSR residual correction before fitting.
        max_levels (int): Maximum number of levels.
        max_coarse (int): Stop coarsening at this many coarse nodes.
        seed (int): Seed of the random test-vector source.
        draw (Callable[[int], torch.Tensor] | None): Explicit uniform
            ``[0, 1)`` source overriding ``seed``.
        seed_vectors (torch.Tensor | None): Known near-null vectors, shape
            ``(n, k_seed)``, seeded alongside the ``k_r`` random draws;
            ``None`` for no seeding.
        test_vector_draw (Callable[[int], torch.Tensor] | None):
            Random-vector source for the initial ``k_r`` test vectors
            specifically; ``None`` reuses ``draw``/``seed`` for this too.
        smoother_omega (float | None): Damping for the default
            weighted-Jacobi solve smoother; ``None`` selects
            ``1 / rho(D^-1 A)`` per level.
        smoother (MultigridSmoother | None): Explicit solve-time smoother.
            ``None`` selects weighted Jacobi. When supplied,
            ``smoother_omega`` must remain ``None``.
        n_pre (int): Solve-time pre-smoothing sweeps.
        n_post (int): Solve-time post-smoothing sweeps.

    References:
        - Brandt, Brannick, Kahl, Livshits (2011), SIAM J. Sci. Comput. 33(2).
    """

    def __init__(
        self,
        k_r: int = 8,
        eta: int = 4,
        n_bootstrap_cycles: int = 2,
        nu: int = 5,
        delta: float = 0.7,
        theta_ad: float = 0.5,
        caliber: int = 4,
        gamma: float = 1.5,
        use_lsr: bool = True,
        max_levels: int = 10,
        max_coarse: int = 10,
        seed: int = 0,
        draw: Callable[[int], torch.Tensor] | None = None,
        seed_vectors: torch.Tensor | None = None,
        test_vector_draw: Callable[[int], torch.Tensor] | None = None,
        smoother_omega: float | None = None,
        smoother: MultigridSmoother | None = None,
        n_pre: int = 1,
        n_post: int = 1,
    ) -> None:
        """Store setup hyperparameters; ``setup()`` runs the sparse Bootstrap AMG algorithm."""

        def new_bootstrap_setup() -> BootstrapSetup:
            stage = GaussSeidelMemo()
            transfers = TransferOperatorMemo()
            neighborhoods: NeighborhoodOf = weak_memo(sparse_depth_neighborhood)
            return BootstrapSetup(
                coarsening_factory=lambda vectors, relaxation, draw: BAMGCoarsening(
                    vectors,
                    relaxation,
                    nu=nu,
                    delta=delta,
                    theta_ad=theta_ad,
                    caliber=caliber,
                    gamma=gamma,
                    use_lsr=use_lsr,
                    draw=draw,
                    transfer_operator=transfers,
                    neighborhood_of=neighborhoods,
                ),
                transfer_operator_factory=transfers,
                relaxation=stage,
                setup_cycle=gs_setup_cycle(stage),
                eta=eta,
                k_r=k_r,
                n_bootstrap_cycles=n_bootstrap_cycles,
                max_levels=max_levels,
                max_coarse=max_coarse,
            )

        self._new_bootstrap_setup = new_bootstrap_setup
        self._run_draw = draw if draw is not None else seeded_draw(seed)
        self._seed_vectors = seed_vectors
        self._test_vector_draw = test_vector_draw
        self._result: BootstrapAMGResult | None = None
        super().__init__(
            coarsening=PrebuiltCoarsening("bootstrap AMG"),
            cycle=prebuilt_cycle(
                resolve_jacobi_default(smoother, smoother_omega), n_pre=n_pre, n_post=n_post
            ),
            n_levels=2,
            linear=True,
        )

    def setup(
        self,
        matrix: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> Self:
        """Run the sparse Bootstrap AMG algorithm and build the resulting hierarchy.

        Args:
            matrix (torch.Tensor): SPD system matrix A (n x n), sparse CSR.
            context (PreconditionerContext | None): Ignored.

        Returns:
            Self: This preconditioner, now ready for ``apply()``.

        Raises:
            ValueError: If the setup produced a single level (nothing to
                coarsen).
        """
        result = self._new_bootstrap_setup().run(
            matrix,
            self._run_draw,
            seed_vectors=self._seed_vectors,
            test_vector_draw=self._test_vector_draw,
        )
        if len(result.matrices) < 2:
            raise ValueError("bootstrap AMG setup produced a single level; lower max_coarse")
        self._commit(self._hierarchy_of(result, matrix))
        self._result = result
        self._n_levels = len(result.matrices)
        return self

    @property
    def result(self) -> BootstrapAMGResult:
        """The realized Bootstrap AMG hierarchy.

        Returns:
            BootstrapAMGResult: Levels, prolongations, and relaxation-derived
                test vectors from the setup that last ran.
        """
        if self._result is None:
            raise PreconditionerNotReadyError(
                f"{type(self).__name__}.setup() must be called before the result is available."
            )
        return self._result

    def __str__(self) -> str:
        """Human-readable structural summary.

        Returns:
            str: e.g. ``"BAMG(n_levels=3, k_r=8, coarse_dim=5)"``.
        """
        if self._result is None:
            return "BAMG(not set up)"
        return (
            f"BAMG(n_levels={len(self._result.matrices)}, "
            f"k_r={self._result.candidates.shape[1]}, "
            f"coarse_dim={int(self._result.matrices[-1].shape[0])})"
        )

    @staticmethod
    def _hierarchy_of(result: BootstrapAMGResult, matrix: torch.Tensor) -> MultigridHierarchy:
        """Prebuilt levels of ``result`` moved to the device/dtype of ``matrix``.

        Args:
            result (BootstrapAMGResult): Realized setup hierarchy.
            matrix (torch.Tensor): System matrix being bound.

        Returns:
            MultigridHierarchy: The Bootstrap AMG hierarchy.
        """
        moved = BootstrapAMGResult(
            matrices=tuple(m.to(matrix) for m in result.matrices),
            prolongations=tuple(p.to(matrix) for p in result.prolongations),
            candidates=result.candidates,
            transfer_operator_factory=result.transfer_operator_factory,
        )
        return moved.hierarchy
