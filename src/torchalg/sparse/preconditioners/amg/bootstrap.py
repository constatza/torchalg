"""Bootstrap AMG (BAMG) coarsening strategy, sparse-CSR sibling.

Sparse-CSR counterpart of
``preconditioners.implementations.amg.bootstrap.BAMGCoarsening`` (dense;
kept unmodified for comparison - see ``docs/plan.md``'s "Correction: dense
and sparse must be separate implementations, not an internal branch").
Wires together this session's new sparse kernels
(``kernels.algebraic_distance.sparse_algebraic_distance``,
``kernels.strength.sparse_strength_graph``,
``kernels.depth_neighborhood.sparse_depth_neighborhood``,
``kernels.lsr_correction.sparse_lsr_correction``,
``kernels.prolongation.sparse_interpolation_prolongation``) with the
format-agnostic promoted leaves (``torchalg.utils.test_vector_weights``,
``torchalg.utils.ls_interpolation``) and this package's own
``_compatible_relaxation.compatible_relaxation_coarsening``, following
``docs/bootstrap-amg.md`` Sec. 5's per-level block exactly as the dense
class does.

Every change from the dense class follows directly from storage format, not
a different algorithm:

- **Transpose-needing sparse-sparse/sparse-dense products** (``T_l = P_l^T
  P_l``, composite prolongation chaining ``P_0 P_1``, ``P^T @
  test_vectors``) route through COO for the transpose step, mirroring
  ``kernels.galerkin.form_sparse_sparse``'s own documented
  CSR-transpose-yields-CSC workaround.
- **``T_l`` is materialized dense** at the one point it is actually read
  (``_current_T``): it only ever multiplies dense ``test_vectors``
  downstream either way, it is bounded by the *current* level's size (not
  growing across levels), and this matches the established coarsest-level-
  densification precedent (``kernels.coarse_solve.dense_coarse_solve``) -
  densifying one small per-level Gram operator, never ``A`` or ``P``
  themselves.
- **The LS-ring/coarse-neighbor candidate lookup** (dense:
  ``torch.nonzero(coarse_neighborhood, as_tuple=False).tolist()`` grouped by
  row) is replaced by directly reading ``sparse_depth_neighborhood``'s own
  ``crow_indices()``/``col_indices()`` per fine row - *simpler* than the
  dense version, not more complex, since CSR already hands back row-grouped
  neighbor lists for free (same precedent as SA-AMG's own sparse
  ``build_transfer``).
- **The empty-interpolatory-set fallback**'s strength lookup
  (``distance_cpu[i]``, a dense row) becomes a per-row sparse lookup
  (``_strongest_candidate_sparse``, searchsorted over that row's own stored
  ``(col, value)`` pairs, 0.0 where absent - exactly matching what the
  dense row's zero-fill outside the neighborhood pattern already means).
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
from torchalg.sparse.kernels.algebraic_distance import sparse_algebraic_distance
from torchalg.sparse.kernels.depth_neighborhood import sparse_depth_neighborhood
from torchalg.sparse.kernels.galerkin import form_sparse_sparse
from torchalg.sparse.kernels.lsr_correction import sparse_lsr_correction
from torchalg.sparse.kernels.prolongation import sparse_interpolation_prolongation
from torchalg.sparse.kernels.strength import sparse_strength_graph
from torchalg.sparse.preconditioners.amg.transfer import SparseTransferOperator
from torchalg.utils.ls_interpolation import ls_interpolation_row, select_interpolatory_set
from torchalg.utils.test_vector_weights import test_vector_weights

from ._compatible_relaxation import compatible_relaxation_coarsening
from ._presets import GS_SETUP_CYCLE, PrebuiltCoarsening, prebuilt_cycle, seeded_draw
from .smoothers import GaussSeidelSmoother, resolve_jacobi_default

if TYPE_CHECKING:
    from torchalg.multigrid.hierarchy import MultigridHierarchy
    from torchalg.multigrid.protocols import MultigridSmoother

_Relaxation = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor]
"""Shape of a ``SmootherBase.smooth``-style relaxation callable: ``(A, rhs, x, steps) -> x``."""

_LSR_TARGET_FRACTION = 0.2
"""Matches the dense sibling's value - see its docstring."""


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
        relaxation (_Relaxation): Relaxation callable used by CR,
            ``(A, rhs, x, steps) -> x`` (e.g. the sparse
            ``GaussSeidelSmoother().smooth``).
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
        relaxation: _Relaxation,
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
    ) -> None:
        """Store hyperparameters and the finest-level test vectors.

        Args:
            test_vectors (torch.Tensor): Test vectors for this level.
            relaxation (_Relaxation): Relaxation callable used by CR.
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
        """
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

    def _current_T(self) -> torch.Tensor | None:
        """Composite-interpolation Gram operator ``T_l = P_l^T P_l`` for the level about to be processed.

        Materialized dense - see the module docstring's rationale.

        Returns:
            torch.Tensor | None: Dense ``T_l``, shape ``(n, n)`` for the
            current level's dimension ``n``, or ``None`` for ``T = I``.
        """
        if self._composite is None:
            return None
        composite_coo = self._composite.to_sparse_coo()
        return torch.sparse.mm(composite_coo.t(), composite_coo).to_dense()

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

        Args:
            A (torch.Tensor): Sparse CSR fine-grid matrix, shape ``(n, n)``.

        Returns:
            tuple[torch.Tensor, SparseTransferOperator]: ``(A_coarse, transfer)``.
        """
        test_vectors = self._vectors[A.shape[0]]
        T = self._current_T()
        distance = sparse_algebraic_distance(test_vectors, A, depth=self._depth, T=T)
        coarse_mask = self._coarse_mask(A, distance)
        prolongation = self._prolongation(A, test_vectors, coarse_mask, distance, T=T)
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
        return coarse_matrix, SparseTransferOperator(prolongation)

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
        top_count = min(math.ceil(_LSR_TARGET_FRACTION * fine_points.numel()), fine_points.numel())
        residual_magnitude = (A @ test_vectors)[fine_points].abs()
        corrected = test_vectors.clone()
        for column in range(test_vectors.shape[1]):
            top_local = torch.topk(residual_magnitude[:, column], max(top_count, 1)).indices
            target_rows = fine_points[top_local]
            corrected_column = sparse_lsr_correction(
                test_vectors[:, column : column + 1], A, target_rows
            )
            corrected[:, column] = corrected_column[:, 0]
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
    def _strongest_candidate_sparse(
        candidates: torch.Tensor, row_cols: torch.Tensor, row_vals: torch.Tensor
    ) -> torch.Tensor:
        """Fallback interpolatory set, given one sparse row's own stored ``(col, value)`` pairs.

        Looks up each candidate's ``r_ij`` within this row's stored
        entries, 0.0 where absent - matching exactly what the dense row's
        zero-fill outside its neighborhood pattern already means.

        Args:
            candidates (torch.Tensor): Long tensor of candidate ``C``-indices.
            row_cols (torch.Tensor): This row's stored column indices
                (ascending, as CSR guarantees), shape ``(deg,)``.
            row_vals (torch.Tensor): This row's stored values, same shape as
                ``row_cols``.

        Returns:
            torch.Tensor: Long tensor of shape ``(1,)``.
        """
        if row_cols.numel() == 0:
            return candidates[0].reshape(1)
        positions = torch.searchsorted(row_cols, candidates)
        clamped = positions.clamp(max=row_cols.numel() - 1)
        found = (positions < row_cols.numel()) & (row_cols[clamped] == candidates)
        strengths = torch.where(found, row_vals[clamped], torch.zeros_like(row_vals[clamped]))
        return candidates[torch.argmax(strengths)].reshape(1)

    def _prolongation(
        self,
        A: torch.Tensor,
        test_vectors: torch.Tensor,
        coarse_mask: torch.Tensor,
        distance: torch.Tensor,
        T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """LS/LSR-fitted sparse CSR prolongation ``P`` (identity on ``C``, fitted rows on ``F``).

        Args:
            A (torch.Tensor): Sparse CSR fine-grid matrix, shape ``(n, n)``.
            test_vectors (torch.Tensor): Test vectors, shape ``(n, k)``.
            coarse_mask (torch.Tensor): Boolean coarse mask, shape ``(n,)``.
            distance (torch.Tensor): Sparse CSR algebraic-distance matrix
                ``r``, shape ``(n, n)``, used by the empty-interpolatory-set
                fallback.
            T (torch.Tensor | None): Dense composite-interpolation Gram
                operator for the LS/LSR fit weights; ``None`` for the
                ``T = I`` reduction.

        Returns:
            torch.Tensor: Sparse CSR prolongation matrix, shape
            ``(n, n_c)``.
        """
        n = A.shape[0]
        device = A.device
        coarse_points = torch.nonzero(coarse_mask).flatten()
        fine_points = torch.nonzero(~coarse_mask).flatten()
        n_coarse = coarse_points.numel()
        coarse_index = torch.full((n,), -1, dtype=torch.long, device=device)
        coarse_index[coarse_points] = torch.arange(n_coarse, device=device)

        weights = test_vector_weights(test_vectors, A, T=T)
        fit_vectors = self._fit_vectors(A, test_vectors, fine_points)
        neighborhood = sparse_depth_neighborhood(A, self._depth + 2)

        neighborhood_cpu = neighborhood.cpu()
        neighborhood_crow = neighborhood_cpu.crow_indices()
        neighborhood_col = neighborhood_cpu.col_indices()
        coarse_mask_cpu = coarse_mask.cpu()

        distance_cpu = distance.cpu()
        distance_crow = distance_cpu.crow_indices()
        distance_col = distance_cpu.col_indices()
        distance_values = distance_cpu.values()

        fit_vectors_cpu = fit_vectors.cpu()
        weights_cpu = weights.cpu()
        coarse_index_cpu = coarse_index.cpu()
        coarse_points_cpu = coarse_points.cpu()

        row_indices: list[int] = coarse_points_cpu.tolist()
        col_indices: list[int] = list(range(n_coarse))
        values: list[float] = [1.0] * n_coarse

        for i in fine_points.tolist():
            n_start, n_end = int(neighborhood_crow[i]), int(neighborhood_crow[i + 1])
            row_cols_all = neighborhood_col[n_start:n_end]
            candidate_columns = row_cols_all[coarse_mask_cpu[row_cols_all]]
            candidates = candidate_columns if candidate_columns.numel() > 0 else coarse_points_cpu

            interp_set = select_interpolatory_set(
                candidates, fit_vectors_cpu, A, i, weights_cpu, self._caliber, gamma=self._gamma
            )
            if interp_set.numel() == 0 and candidates.numel() > 0:
                d_start, d_end = int(distance_crow[i]), int(distance_crow[i + 1])
                interp_set = self._strongest_candidate_sparse(
                    candidates, distance_col[d_start:d_end], distance_values[d_start:d_end]
                )
            if interp_set.numel() == 0:
                continue
            row = ls_interpolation_row(fit_vectors_cpu, i, interp_set, weights_cpu)
            cols = coarse_index_cpu[interp_set]
            row_indices.extend([i] * interp_set.numel())
            col_indices.extend(cols.tolist())
            values.extend(row.tolist())

        row_tensor = torch.tensor(row_indices, dtype=torch.long, device=device)
        col_tensor = torch.tensor(col_indices, dtype=torch.long, device=device)
        value_tensor = torch.tensor(values, dtype=A.dtype, device=device)
        return sparse_interpolation_prolongation(
            row_tensor, col_tensor, value_tensor, (n, n_coarse)
        )


class BootstrapAMGPreconditioner(AMGPreconditioner):
    """Bootstrap AMG preconditioner (BAMG) for sparse CSR systems, sparse-CSR sibling.

    Sparse-CSR counterpart of
    ``preconditioners.implementations.amg.bootstrap.BootstrapAMGPreconditioner``
    (dense; kept unmodified for comparison). Runs ``BootstrapSetup.run`` at
    construction (the shared, format-agnostic outer loop promoted to
    ``torchalg.multigrid.bootstrap_setup`` this session - see that module's
    docstring), injecting this module's own ``BAMGCoarsening``,
    ``SparseTransferOperator``, the sparse ``GaussSeidelSmoother``, and this
    package's ``_presets.GS_SETUP_CYCLE`` in place of the dense equivalents.
    Kept as a subclass (not converted to a factory function like
    ``vcycle_amg``/``wcycle_amg``): it overrides ``_make_hierarchy`` with
    genuinely different hierarchy-construction logic (the prebuilt-levels
    pattern), the same "earned inheritance" reasoning that keeps the dense
    class a subclass too (see ``docs/plan.md``'s "Correction: validation
    placement, and inheritance vs. composition for presets").

    **MGE (``k_e``) is not exposed here** - multigrid-eigensolver enrichment
    has no sparse implementation yet (see
    ``torchalg.multigrid.bootstrap_setup``'s module docstring); this preset
    always runs with ``k_e=0``/``eigensolver=None``, matching the dense
    class's own historical (pre-MGE) default behavior exactly.

    Args:
        matrix (torch.Tensor): SPD system matrix A (n x n), sparse CSR.
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
        matrix: torch.Tensor,
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
        """Run the sparse Bootstrap AMG setup and wrap the resulting hierarchy.

        Raises:
            ValueError: If the setup produced a single level (nothing to
                coarsen).
        """
        setup = BootstrapSetup(
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
            ),
            transfer_operator_factory=SparseTransferOperator,
            relaxation=GaussSeidelSmoother().smooth,
            setup_cycle=GS_SETUP_CYCLE,
            eta=eta,
            k_r=k_r,
            n_bootstrap_cycles=n_bootstrap_cycles,
            max_levels=max_levels,
            max_coarse=max_coarse,
        )
        result = setup.run(
            matrix,
            draw if draw is not None else seeded_draw(seed),
            seed_vectors=seed_vectors,
            test_vector_draw=test_vector_draw,
        )
        if len(result.matrices) < 2:
            raise ValueError("bootstrap AMG setup produced a single level; lower max_coarse")
        super().__init__(
            matrix=matrix,
            coarsening=PrebuiltCoarsening("bootstrap AMG"),
            cycle=prebuilt_cycle(
                resolve_jacobi_default(smoother, smoother_omega), n_pre=n_pre, n_post=n_post
            ),
            n_levels=len(result.matrices),
            linear=True,
        )
        self._result = result

    @property
    def result(self) -> BootstrapAMGResult:
        """The realized Bootstrap AMG hierarchy.

        Returns:
            BootstrapAMGResult: Levels, prolongations, and relaxation-derived
                test vectors from the setup that ran at construction.
        """
        return self._result

    def __str__(self) -> str:
        """Human-readable structural summary.

        Returns:
            str: e.g. ``"BAMG(n_levels=3, k_r=8, coarse_dim=5)"``.
        """
        return (
            f"BAMG(n_levels={len(self._result.matrices)}, "
            f"k_r={self._result.candidates.shape[1]}, "
            f"coarse_dim={int(self._result.matrices[-1].shape[0])})"
        )

    def _make_hierarchy(self) -> MultigridHierarchy:
        """Prebuilt levels moved to the current device/dtype of the system matrix.

        Returns:
            MultigridHierarchy: The Bootstrap AMG hierarchy.
        """
        moved = BootstrapAMGResult(
            matrices=tuple(m.to(self._matrix) for m in self._result.matrices),
            prolongations=tuple(p.to(self._matrix) for p in self._result.prolongations),
            candidates=self._result.candidates,
            transfer_operator_factory=self._result.transfer_operator_factory,
        )
        return moved.hierarchy
