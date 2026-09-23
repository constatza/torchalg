"""Bootstrap AMG (BAMG): coarsening strategy, bootstrap setup loop, preconditioner preset.

Wires together the pure kernels of ``_compatible_relaxation.py`` (Sec. 2.1),
``_algebraic_distance.py`` (Sec. 2.2) and ``_least_squares.py`` (Sec. 3) into
a working ``CoarseningStrategy``/preconditioner pair, following
``docs/bootstrap-amg.md`` Sec. 5's setup algorithm and the "prebuilt
hierarchy" preset pattern ``adaptive.py`` already established for alpha-SA.

``docs/bootstrap-amg.md`` Sec. 5 is the authoritative per-level/outer-loop
recipe; every section number cited below refers to it (or, for MGE, to
Sec. 4.2, which this module deliberately does not implement - see below).

References:
    - Brandt, A., Brannick, J., Kahl, K., & Livshits, I. (2011). Bootstrap
      AMG. SIAM J. Sci. Comput. 33(2), 612-632. Cited as [BAMG11].
    - Brandt, A., Brannick, J., Kahl, K., & Livshits, I. (2011). An algebraic
      distances measure of AMG strength of connection. arXiv:1106.5990.
      Cited as [AD11].

Scope decisions (v1, matching ``docs/amg-integration-architecture.md`` Sec.
3.2/Sec. 7):

- **No multigrid eigensolver (MGE, Sec. 4.2)**. Bootstrap cycles here are
  the "improve test vectors by running the current AMG cycle on ``A x = 0``"
  step alone ([BAMG11] Sec. 3, Sec. 5's "for l=0..L-1: relax; ... repeat:
  improve V ... recompute C_l, P_l"), with no eigenvector enrichment. The
  paper's own numbers show plain ``V^mu``-setup is already competitive
  without MGE (Sec. 4.1's "V^2-setup with MGE ~= W-setup without it" note).
- **Guidance-graph simplification.** ``compatible_relaxation_coarsening``'s
  ``guidance_graph`` (the Gap-1 fix below) is, by Task 1's own design, a
  single fixed argument for the *whole* CR outer loop - never recomputed as
  the coarse set ``C`` grows, exactly like the plain-matrix-graph default it
  replaces. This module follows the same shape: the algebraic-distance
  strength graph ``M_d`` (eq. 4.4) is built once, before any point is marked
  coarse (every node is an "F" candidate for eq. 4.4's own ``i, j in F``
  filter at that point), and held fixed through CR's loop. This does not
  admit any edge the true, iteration-refreshed ``M_d`` would have excluded:
  ``_independent_set_of``'s own ``eligible``/``candidates`` bookkeeping
  already restricts every lookup to genuinely still-eligible (not-yet-coarse)
  nodes regardless of what the guidance graph itself contains.
- **Finest-level-only test-vector improvement.** [BAMG11] Sec. 5's outer
  loop improves the test vectors on *every* level; ``BootstrapSetup.run``
  improves only the **finest** level's vectors per bootstrap cycle (running
  the current hierarchy's cycle on ``A x = 0``) and then re-derives every
  coarse level's vectors from scratch inside ``_build_levels``, by
  restriction plus ``eta`` relaxation sweeps, rather than improving them in
  place. Defensible because the whole hierarchy is rebuilt from the improved
  finest-level vectors anyway, so the coarse vectors already reflect the
  improvement; disclosed here because it is a real simplification of Sec. 5.
- **LSR practical schedule ([BAMG11] Sec. 4).** The paper's default schedule
  corrects only the single largest-weight test vector at the 20% of F-points
  with the largest residual; ``lsr_correction`` (Task 3) corrects every test
  vector at whichever rows it is given (matching the paper's own alternate,
  "LSR applied to every test vector" mode - Sec. 3.2 of
  ``docs/bootstrap-amg.md``). This module uses that mode: every test vector
  is corrected at the 20% of F-points with the largest combined residual
  magnitude.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ._algebraic_distance import (
    algebraic_distance,
    depth_neighborhood,
    strength_graph,
    test_vector_weights,
)
from ._compatible_relaxation import compatible_relaxation_coarsening
from ._least_squares import ls_interpolation_row, lsr_correction, select_interpolatory_set
from ._presets import GS_SETUP_CYCLE, PrebuiltCoarsening, prebuilt_cycle, seeded_draw
from .amg import AMGPreconditioner
from .hierarchy import MultigridHierarchy, MultigridLevel
from .smoothers import GaussSeidelSmoother, resolve_jacobi_default
from .transfer import DenseTransferOperator

if TYPE_CHECKING:
    from .protocols import MultigridSmoother

_Relaxation = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor]
"""Shape of a ``SmootherBase.smooth``-style relaxation callable: ``(A, rhs, x, steps) -> x``."""

_LSR_TARGET_FRACTION = 0.2
"""Fraction of F-points corrected by LSR - [BAMG11] Sec. 4's practical schedule."""


def _random_test_vectors(
    matrix: torch.Tensor, k: int, draw: Callable[[int], torch.Tensor]
) -> torch.Tensor:
    """``k`` random test vectors on ``matrix``'s grid ([BAMG11] Sec. 4.1: TVs start random).

    Args:
        matrix (torch.Tensor): Level matrix, shape ``(n, n)``.
        k (int): Number of test vectors.
        draw (Callable[[int], torch.Tensor]): Uniform ``[0, 1)`` random-vector
            source, ``n -> tensor of length n``.

    Returns:
        torch.Tensor: Test vectors, shape ``(n, k)``.
    """
    columns = [draw(matrix.shape[0]).to(dtype=matrix.dtype, device=matrix.device) for _ in range(k)]
    return torch.stack(columns, dim=1)


def _relax_columns(
    matrix: torch.Tensor, vectors: torch.Tensor, relaxation: _Relaxation, sweeps: int
) -> torch.Tensor:
    """Relax every column of ``vectors`` on ``matrix @ x = 0`` ([BAMG11] eq. 3.1).

    Args:
        matrix (torch.Tensor): Level matrix, shape ``(n, n)``.
        vectors (torch.Tensor): Test vectors, shape ``(n, k)``.
        relaxation (_Relaxation): Relaxation callable, ``(A, rhs, x, steps)
            -> x``.
        sweeps (int): Number of sweeps ``eta``.

    Returns:
        torch.Tensor: Relaxed test vectors, shape ``(n, k)``.
    """
    zero_rhs = torch.zeros(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    columns = [relaxation(matrix, zero_rhs, vectors[:, k], sweeps) for k in range(vectors.shape[1])]
    return torch.stack(columns, dim=1)


def _improve_test_vectors(
    matrix: torch.Tensor,
    vectors: torch.Tensor,
    hierarchy: MultigridHierarchy,
    iterations: int,
) -> torch.Tensor:
    """Improve ``vectors`` by running the current AMG cycle on ``matrix @ x = 0`` ([BAMG11] Sec. 3).

    Uses linearity of the cycle for a zero right-hand side: one cycle from
    ``x`` is ``x - cycle.apply(hierarchy, matrix @ x)`` (``adaptive.py``'s
    ``_solve`` uses the same identity).

    Args:
        matrix (torch.Tensor): Finest-level matrix, shape ``(n, n)``.
        vectors (torch.Tensor): Test vectors to improve, shape ``(n, k)``.
        hierarchy (MultigridHierarchy): Current (partial) hierarchy.
        iterations (int): Number of cycles per test vector.

    Returns:
        torch.Tensor: Improved test vectors, shape ``(n, k)``.
    """
    columns = []
    for k in range(vectors.shape[1]):
        x = vectors[:, k]
        for _ in range(iterations):
            x = x - GS_SETUP_CYCLE.apply(hierarchy, matrix @ x)
        columns.append(x)
    return torch.stack(columns, dim=1)


def _hierarchy_of(
    levels: list[torch.Tensor] | tuple[torch.Tensor, ...],
    prolongations: list[torch.Tensor] | tuple[torch.Tensor, ...],
) -> MultigridHierarchy:
    """Engine hierarchy over plain level-matrix/prolongation lists.

    Args:
        levels (list[torch.Tensor] | tuple[torch.Tensor, ...]): Level
            matrices, finest first.
        prolongations (list[torch.Tensor] | tuple[torch.Tensor, ...]):
            Prolongations, one per level except the coarsest.

    Returns:
        MultigridHierarchy: Hierarchy sharing the given tensors.
    """
    last = len(levels) - 1
    return MultigridHierarchy(
        tuple(
            MultigridLevel(
                matrix=matrix,
                transfer=None if index == last else DenseTransferOperator(prolongations[index]),
            )
            for index, matrix in enumerate(levels)
        )
    )


@dataclass(frozen=True)
class BootstrapAMGResult:
    """Result of the Bootstrap AMG setup.

    Mirrors ``adaptive.py``'s ``AdaptiveSAResult`` shape (same field names
    where the concept matches), so ``BootstrapAMGPreconditioner._make_hierarchy``
    can be nearly identical code to ``AdaptiveSAPreconditioner._make_hierarchy``.

    Attributes:
        matrices (tuple[torch.Tensor, ...]): Level matrices ``A_l``, finest
            first.
        prolongations (tuple[torch.Tensor, ...]): ``P_l`` mapping level
            ``l+1`` to level ``l`` (``R = P^T``).
        candidates (torch.Tensor): Finest-level test vectors after the final
            bootstrap cycle, shape ``(n, k_r)``.
    """

    matrices: tuple[torch.Tensor, ...]
    prolongations: tuple[torch.Tensor, ...]
    candidates: torch.Tensor

    @property
    def hierarchy(self) -> MultigridHierarchy:
        """The multigrid hierarchy for the engine's cycles.

        Returns:
            MultigridHierarchy: Levels with dense transfer operators.
        """
        return _hierarchy_of(self.matrices, self.prolongations)


class BAMGCoarsening:
    """One coarse level of Bootstrap AMG (``docs/bootstrap-amg.md`` Sec. 5's per-level block).

    Stateful ``CoarseningStrategy``: holds the current test vectors keyed by
    matrix dimension (coarsening strictly shrinks ``n``, so the dimension
    identifies the level - ``docs/amg-integration-architecture.md`` Sec.
    3.2's dimension-keyed-dict pattern). ``BootstrapSetup`` relaxes/improves
    the stored vectors between levels and bootstrap cycles via
    ``set_test_vectors``; ``build_transfer`` reads them, coarsens, and
    stores the restricted vectors under the coarse dimension for the next
    level.

    One ``build_transfer(A)`` call implements CR-coarsening (Sec. 2.1,
    guided by the algebraic-distance strength graph of Sec. 2.2 - see the
    module docstring's guidance-graph note), per-F-point interpolatory-set
    selection (Sec. 3.3), LS/LSR fitting (Sec. 3.1-3.2) and the Galerkin
    coarse operator - everything in Sec. 5's per-level block except the
    leading ``eta``-sweep relaxation, which is ``BootstrapSetup``'s
    responsibility (the same relaxed vectors also seed the CR loop itself).

    **Robustness addition, not part of the literal paper algorithm:** if
    ``select_interpolatory_set`` returns an empty ``C_i`` for an F-point,
    that row falls back to the single strongest candidate by algebraic
    distance (``argmax`` of ``r_ij`` over the candidate set, [AD11] eq.
    4.3) instead of being left all-zero. An all-zero row of ``P`` makes
    that F-point permanently invisible to the coarse grid - it receives no
    coarse-grid correction at all - which silently degrades the cycle far
    more than a merely suboptimal caliber-one row would. [AD11] Sec. 4.3's
    greedy rule has no such case because its penalization test is stated on
    a near-order-one relative residual; this is the defensive floor for the
    degenerate inputs that can still reach it. The one case that keeps an
    all-zero row is an empty coarse set (``C = {}``, a legitimate CR
    outcome - see ``BootstrapSetup._build_levels``), where there is simply
    nothing to interpolate from; that pass is discarded by the caller.

    Args:
        test_vectors (torch.Tensor): Test vectors for this level, shape
            ``(n, k)``.
        relaxation (_Relaxation): Relaxation callable used by CR,
            ``(A, rhs, x, steps) -> x``.
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
        self._draw = draw if draw is not None else seeded_draw(seed)
        self._vectors: dict[int, torch.Tensor] = {test_vectors.shape[0]: test_vectors}
        self._last_prolongation: torch.Tensor | None = None

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
        """Raw prolongation tensor ``P`` from the most recent ``build_transfer`` call.

        Returns:
            torch.Tensor: Prolongation matrix, shape ``(n, n_c)``.

        Raises:
            RuntimeError: If ``build_transfer`` has not been called yet.
        """
        if self._last_prolongation is None:
            raise RuntimeError("build_transfer has not been called yet")
        return self._last_prolongation

    def build_transfer(self, A: torch.Tensor) -> tuple[torch.Tensor, DenseTransferOperator]:
        """Build one Bootstrap AMG coarse level from ``A``.

        Args:
            A (torch.Tensor): Fine-grid matrix, shape ``(n, n)``.

        Returns:
            tuple[torch.Tensor, DenseTransferOperator]: ``(A_coarse, transfer)``.
        """
        test_vectors = self._vectors[A.shape[0]]
        distance = algebraic_distance(test_vectors, A, depth=self._depth)
        coarse_mask = self._coarse_mask(A, distance)
        prolongation = self._prolongation(A, test_vectors, coarse_mask, distance)
        coarse_matrix = prolongation.T @ A @ prolongation
        self._last_prolongation = prolongation
        self._vectors[coarse_matrix.shape[0]] = prolongation.T @ test_vectors
        return coarse_matrix, DenseTransferOperator(prolongation)

    def _coarse_mask(self, A: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
        """Compatible-relaxation coarse set, guided by the algebraic-distance graph.

        ``strength_graph`` is directional by design (``r_ij != r_ji``, see
        ``_algebraic_distance.py``'s module docstring), while
        ``_independent_set_of`` consults only row ``i`` of whatever graph it
        is handed - so a one-directional edge ``j -> i`` would let both ``i``
        and ``j`` be marked coarse in the same stage, which is not an
        independent set with respect to the strength relation. The graph is
        therefore symmetrized *here*, at the call site that owns the choice,
        by union (``M | M^T``: an edge exists if either direction is strong)
        rather than intersection: the union is the conservative option, since
        it blocks strictly more simultaneous coarse picks and so can only
        make the independent set safer, whereas the intersection would keep
        exactly the one-directional edges that cause the problem.

        Args:
            A (torch.Tensor): Fine-grid matrix, shape ``(n, n)``.
            distance (torch.Tensor): Algebraic-distance matrix ``r``, shape
                ``(n, n)`` ([AD11] eq. 4.3).

        Returns:
            torch.Tensor: Boolean coarse mask, shape ``(n,)``.
        """
        all_fine = torch.ones(A.shape[0], dtype=torch.bool, device=A.device)
        directed = strength_graph(distance, all_fine, theta_ad=self._theta_ad)
        return compatible_relaxation_coarsening(
            A,
            self._relaxation,
            nu=self._nu,
            delta=self._delta,
            draw=self._draw,
            guidance_graph=directed | directed.T,
        )

    def _fit_vectors(
        self, A: torch.Tensor, test_vectors: torch.Tensor, fine_points: torch.Tensor
    ) -> torch.Tensor:
        """Test vectors the LS fit uses: LSR-corrected if enabled, else as-is.

        Args:
            A (torch.Tensor): Fine-grid matrix, shape ``(n, n)``.
            test_vectors (torch.Tensor): Test vectors, shape ``(n, k)``.
            fine_points (torch.Tensor): Long tensor of fine-point indices.

        Returns:
            torch.Tensor: Test vectors to fit, shape ``(n, k)``.
        """
        if not self._use_lsr or fine_points.numel() == 0:
            return test_vectors
        residual_magnitude = (A @ test_vectors)[fine_points].abs().sum(dim=1)
        top_count = min(math.ceil(_LSR_TARGET_FRACTION * fine_points.numel()), fine_points.numel())
        top_local = torch.topk(residual_magnitude, max(top_count, 1)).indices
        target_rows = fine_points[top_local]
        return lsr_correction(test_vectors, A, target_rows)

    @staticmethod
    def _strongest_candidate(candidates: torch.Tensor, strengths: torch.Tensor) -> torch.Tensor:
        """Fallback interpolatory set: the single algebraically closest candidate.

        Args:
            candidates (torch.Tensor): Long tensor of candidate ``C``-indices.
            strengths (torch.Tensor): Algebraic distances ``r_ij`` from the
                target row ``i`` to every node, shape ``(n,)``; larger is
                stronger ([AD11] eq. 4.3).

        Returns:
            torch.Tensor: Long tensor of shape ``(1,)``.
        """
        return candidates[torch.argmax(strengths[candidates])].reshape(1)

    def _prolongation(
        self,
        A: torch.Tensor,
        test_vectors: torch.Tensor,
        coarse_mask: torch.Tensor,
        distance: torch.Tensor,
    ) -> torch.Tensor:
        """LS/LSR-fitted prolongation matrix ``P`` (identity on ``C``, fitted rows on ``F``).

        Args:
            A (torch.Tensor): Fine-grid matrix, shape ``(n, n)``.
            test_vectors (torch.Tensor): Test vectors, shape ``(n, k)``.
            coarse_mask (torch.Tensor): Boolean coarse mask, shape ``(n,)``.
            distance (torch.Tensor): Algebraic-distance matrix ``r``, shape
                ``(n, n)``, used by the empty-interpolatory-set fallback.

        Returns:
            torch.Tensor: Prolongation matrix, shape ``(n, n_c)``.
        """
        n = A.shape[0]
        device = A.device
        coarse_points = torch.nonzero(coarse_mask).flatten()
        fine_points = torch.nonzero(~coarse_mask).flatten()
        n_coarse = coarse_points.numel()
        coarse_index = torch.full((n,), -1, dtype=torch.long, device=device)
        coarse_index[coarse_points] = torch.arange(n_coarse, device=device)

        weights = test_vector_weights(test_vectors, A)
        fit_vectors = self._fit_vectors(A, test_vectors, fine_points)
        neighborhood = depth_neighborhood(A, self._depth + 2)

        # One nonzero() + tolist() for every fine row's coarse-neighbor set
        # up front, instead of one `torch.nonzero(...)` (a device sync) per
        # fine point inside the loop below - same grouping the aggregation
        # fix in `_aggregation.py::standard_aggregation` uses, for the same
        # reason: nonzero() already returns (row, col) pairs in row-major
        # order, so grouping them reproduces each row's original candidate
        # order with one sync total instead of one per row.
        coarse_neighborhood = neighborhood[fine_points] & coarse_mask
        candidate_lists: list[list[int]] = [[] for _ in range(fine_points.numel())]
        for local_row, col in torch.nonzero(coarse_neighborhood, as_tuple=False).tolist():
            candidate_lists[local_row].append(col)

        # `select_interpolatory_set`/`ls_interpolation_row` run once per
        # fine point (thousands, for a real mesh), each a greedy loop with
        # data-dependent stopping that does several `.item()`-synced tiny
        # linear solves. That's thousands of host<->device round trips for
        # genuinely tiny (caliber-sized) work if left on CUDA - each row's
        # stopping point depends on the previous row's chosen set only
        # through shared `fit_vectors`/`weights`, not through any cross-row
        # state, so unlike `fit_candidates`' aggregate loop (batched in
        # `_tentative.py`, all aggregates fit in lockstep) this loop's
        # per-row variable trip count doesn't reduce to a fixed small
        # number of batched steps without padding every row to the
        # worst-case caliber-growth trajectory - not worth the complexity
        # here. Running it CPU-resident instead removes the sync cost
        # entirely (CPU `.item()` needs no device round trip), paying one
        # one-time transfer of the row-loop's working tensors up front. `A`
        # itself is NOT among those transfers: `select_interpolatory_set`
        # accepts a `matrix` argument only for call-signature parity with
        # every other per-row AMG kernel and never reads it (see its
        # docstring), so shipping the whole dense (n, n) `A` to host on
        # every call here would be a pure-waste PCIe transfer, dwarfing
        # everything this loop actually saves. Pass `A` through unchanged.
        fit_vectors_cpu = fit_vectors.cpu()
        weights_cpu = weights.cpu()
        distance_cpu = distance.cpu()
        coarse_index_cpu = coarse_index.cpu()
        coarse_points_cpu = coarse_points.cpu()

        prolongation = torch.zeros(n, n_coarse, dtype=A.dtype)
        prolongation[coarse_points_cpu, torch.arange(n_coarse)] = 1.0
        for local_row, i in enumerate(fine_points.tolist()):
            candidate_columns = candidate_lists[local_row]
            candidates = (
                torch.tensor(candidate_columns, dtype=torch.long)
                if candidate_columns
                else coarse_points_cpu
            )
            interp_set = select_interpolatory_set(
                candidates, fit_vectors_cpu, A, i, weights_cpu, self._caliber, gamma=self._gamma
            )
            if interp_set.numel() == 0 and candidates.numel() > 0:
                interp_set = self._strongest_candidate(candidates, distance_cpu[i])
            if interp_set.numel() == 0:
                continue
            row = ls_interpolation_row(fit_vectors_cpu, i, interp_set, weights_cpu)
            prolongation[i, coarse_index_cpu[interp_set]] = row
        return prolongation.to(device)


@dataclass(frozen=True)
class BootstrapSetup:
    """Bootstrap AMG setup parameters and the outer setup loop.

    Implements ``docs/bootstrap-amg.md`` Sec. 4.1/Sec. 5: builds one
    hierarchy from ``k_r`` random test vectors relaxed ``eta`` sweeps per
    level, coarsening while ``len(levels) < max_levels and
    levels[-1].shape[0] > max_coarse`` (the same stopping-condition shape as
    ``adaptive.py``'s ``_smoothed_aggregation_levels``), then for
    ``n_bootstrap_cycles`` cycles replaces plain relaxation with a full
    ``VCycle.apply`` on ``A x = 0`` to improve the **finest** level's test
    vectors and rebuilds the whole hierarchy from them. Each coarse level's
    vectors are re-derived from scratch during that rebuild (restriction by
    ``P^T`` plus ``eta`` relaxation sweeps in ``_build_levels``), not
    improved in place - see the module docstring's scope note.

    Attributes:
        nu (int): CR sweeps per stage.
        delta (float): CR stopping tolerance.
        theta_ad (float): Algebraic-distance strength threshold.
        caliber (int): Maximum interpolatory-set size.
        gamma (float): Caliber-growth penalization exponent.
        eta (int): Relaxation sweeps per test vector per level.
        k_r (int): Number of relaxation-derived test vectors.
        use_lsr (bool): Apply the LSR residual correction before fitting.
        n_bootstrap_cycles (int): Number of bootstrap-cycle passes.
        max_levels (int): Maximum number of levels.
        max_coarse (int): Stop coarsening at this many coarse nodes.
    """

    nu: int = 5
    delta: float = 0.7
    theta_ad: float = 0.5
    caliber: int = 4
    gamma: float = 1.5
    eta: int = 4
    k_r: int = 8
    use_lsr: bool = True
    n_bootstrap_cycles: int = 2
    max_levels: int = 10
    max_coarse: int = 10

    def _coarsening_from(
        self,
        vectors: torch.Tensor,
        relaxation: _Relaxation,
        draw: Callable[[int], torch.Tensor],
    ) -> BAMGCoarsening:
        """Build a fresh ``BAMGCoarsening`` seeded with ``vectors``.

        Args:
            vectors (torch.Tensor): Finest-level test vectors.
            relaxation (_Relaxation): Relaxation callable used by CR.
            draw (Callable[[int], torch.Tensor]): Uniform ``[0, 1)``
                random-vector source for CR.

        Returns:
            BAMGCoarsening: Fresh coarsening strategy.
        """
        return BAMGCoarsening(
            vectors,
            relaxation,
            nu=self.nu,
            delta=self.delta,
            theta_ad=self.theta_ad,
            caliber=self.caliber,
            gamma=self.gamma,
            use_lsr=self.use_lsr,
            draw=draw,
        )

    def _build_levels(
        self, matrix: torch.Tensor, coarsening: BAMGCoarsening, relaxation: _Relaxation
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Coarsen ``matrix`` down to ``max_levels``/``max_coarse``.

        Compatible relaxation can legitimately converge with ``C = empty
        set`` on an already-coarsened level (``rho_f <= delta`` from the
        start, [AD11] Algorithm 1's own valid stopping case) - producing a
        zero-column prolongation and a zero-size coarse matrix that would
        contribute exactly zero coarse-grid correction. A coarsening pass is
        discarded, and the loop stops with ``current`` standing as the
        final, coarsest level, whenever it is degenerate in either
        direction: ``coarse_matrix.shape[0] == 0`` (the empty-``C`` case
        above) or ``coarse_matrix.shape[0] >= current.shape[0]`` (no real
        coarsening happened at all). This is a stopping-condition fix, not a
        change to ``compatible_relaxation_coarsening``'s own semantics.

        Args:
            matrix (torch.Tensor): Finest-level matrix.
            coarsening (BAMGCoarsening): Coarsening strategy, already seeded
                with the finest-level test vectors.
            relaxation (_Relaxation): Relaxation callable applied to the test
                vectors before each level is coarsened.

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: Level matrices
            (finest first) and prolongations.
        """
        levels = [matrix]
        prolongations: list[torch.Tensor] = []
        while len(levels) < self.max_levels and levels[-1].shape[0] > self.max_coarse:
            current = levels[-1]
            relaxed = _relax_columns(
                current, coarsening.test_vectors_for(current), relaxation, self.eta
            )
            coarsening.set_test_vectors(current, relaxed)
            coarse_matrix, _ = coarsening.build_transfer(current)
            if coarse_matrix.shape[0] == 0 or coarse_matrix.shape[0] >= current.shape[0]:
                break
            levels.append(coarse_matrix)
            prolongations.append(coarsening.last_prolongation)
        return levels, prolongations

    def run(self, matrix: torch.Tensor, draw: Callable[[int], torch.Tensor]) -> BootstrapAMGResult:
        """Run the Bootstrap AMG setup algorithm.

        Args:
            matrix (torch.Tensor): SPD system matrix, shape ``(n, n)``.
            draw (Callable[[int], torch.Tensor]): Uniform ``[0, 1)``
                random-vector source.

        Returns:
            BootstrapAMGResult: Level matrices, prolongations and the final
            finest-level test vectors.
        """
        relaxation = GaussSeidelSmoother().smooth
        vectors = _random_test_vectors(matrix, self.k_r, draw)
        coarsening = self._coarsening_from(vectors, relaxation, draw)
        levels, prolongations = self._build_levels(matrix, coarsening, relaxation)
        vectors = coarsening.test_vectors_for(matrix)

        for _ in range(self.n_bootstrap_cycles):
            if len(levels) < 2:
                break
            hierarchy = _hierarchy_of(levels, prolongations)
            vectors = _improve_test_vectors(matrix, vectors, hierarchy, self.eta)
            coarsening = self._coarsening_from(vectors, relaxation, draw)
            levels, prolongations = self._build_levels(matrix, coarsening, relaxation)
            vectors = coarsening.test_vectors_for(matrix)

        return BootstrapAMGResult(
            matrices=tuple(levels), prolongations=tuple(prolongations), candidates=vectors
        )


class BootstrapAMGPreconditioner(AMGPreconditioner):
    """Bootstrap AMG preconditioner (BAMG), ``docs/bootstrap-amg.md``.

    Runs ``BootstrapSetup.run`` at construction, then applies one
    V(1,1)-cycle with weighted-Jacobi smoothing by default and a
    pseudo-inverse coarse solve - a fixed symmetric linear operator, so plain
    PCG is valid (same reasoning as ``AdaptiveSAPreconditioner``). Setup cost
    is several multiples of a classical-AMG setup (Sec. 8), so it pays off
    when one matrix is reused across many solves. Compatible-relaxation
    coarsening keeps symmetric Gauss-Seidel unconditionally: CR's relaxation
    *is* the coarsening criterion ([BAMG11] Sec. 2.1's F-relaxation
    convergence test), not a smoothing style, so swapping it would change
    which nodes are marked coarse rather than merely change speed;
    ``smoother`` affects only the solve-time cycle, exactly like
    ``AdaptiveSAPreconditioner``.

    Args:
        matrix (torch.Tensor): SPD system matrix A (n x n).
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
        smoother_omega (float | None): Damping for the default
            weighted-Jacobi solve smoother; ``None`` selects
            ``1 / rho(D^-1 A)`` per level.
        smoother (MultigridSmoother | None): Explicit solve-time smoother.
            ``None`` selects weighted Jacobi. When supplied,
            ``smoother_omega`` must remain ``None``.

    Note (DIP):
        Preset/factory leaf class, same pattern as ``AdaptiveSAPreconditioner``.

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
        smoother_omega: float | None = None,
        smoother: MultigridSmoother | None = None,
    ) -> None:
        """Run the Bootstrap AMG setup and wrap the resulting hierarchy.

        Args:
            matrix (torch.Tensor): SPD system matrix A (n x n).
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
            smoother_omega (float | None): Damping for the default
                weighted-Jacobi solve smoother; ``None`` selects the
                per-level spectral rule.
            smoother (MultigridSmoother | None): Explicit solve-time
                smoother, or ``None`` for weighted Jacobi.

        Raises:
            ValueError: If the setup produced a single level (nothing to
                coarsen).
        """
        setup = BootstrapSetup(
            nu=nu,
            delta=delta,
            theta_ad=theta_ad,
            caliber=caliber,
            gamma=gamma,
            eta=eta,
            k_r=k_r,
            use_lsr=use_lsr,
            n_bootstrap_cycles=n_bootstrap_cycles,
            max_levels=max_levels,
            max_coarse=max_coarse,
        )
        result = setup.run(matrix, draw if draw is not None else seeded_draw(seed))
        if len(result.matrices) < 2:
            raise ValueError("bootstrap AMG setup produced a single level; lower max_coarse")
        super().__init__(
            matrix=matrix,
            coarsening=PrebuiltCoarsening("bootstrap AMG"),
            cycle=prebuilt_cycle(resolve_jacobi_default(smoother, smoother_omega)),
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

        Overrides ``AMGPreconditioner.__str__`` — see
        ``AdaptiveSAPreconditioner.__str__``'s docstring for why (same
        prebuilt-coarsening-placeholder reasoning applies here).

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
        )
        return moved.hierarchy
