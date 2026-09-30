"""Bootstrap AMG (BAMG): coarsening strategy, bootstrap setup loop, preconditioner preset.

Wires together the pure kernels of ``_compatible_relaxation.py`` (Sec. 2.1),
``_algebraic_distance.py`` (Sec. 2.2) and ``_least_squares.py`` (Sec. 3) into
a working ``CoarseningStrategy``/preconditioner pair, following
``docs/bootstrap-amg.md`` Sec. 5's setup algorithm and the "prebuilt
hierarchy" preset pattern ``adaptive.py`` already established for alpha-SA.

``docs/bootstrap-amg.md`` Sec. 5 is the authoritative per-level/outer-loop
recipe; every section number cited below refers to it (or, for MGE, to
Sec. 4.2, implemented in ``_mge.py`` and opt-in via ``k_e`` - see below).

References:
    - Brandt, A., Brannick, J., Kahl, K., & Livshits, I. (2011). Bootstrap
      AMG. SIAM J. Sci. Comput. 33(2), 612-632. Cited as [BAMG11].
    - Brandt, A., Brannick, J., Kahl, K., & Livshits, I. (2011). An algebraic
      distances measure of AMG strength of connection. arXiv:1106.5990.
      Cited as [AD11].
    - Brandt, A., Brannick, J., Kahl, K., & Livshits, I. (2015). Bootstrap
      algebraic multigrid: status report, open problems, and outlook. Numer.
      Math. Theor. Meth. Appl. 8(1). arXiv:1406.1819. Cited as [STATUS14].

Scope decisions (v1, matching ``docs/amg-integration-architecture.md`` Sec.
3.2/Sec. 7). Each is tagged against what [BAMG11]/[STATUS14] themselves say:
**mandatory** means the paper describes no sanctioned variant without it, so
skipping it is a correctness gap, not a style choice; **optional** means the
paper itself presents the omitted piece as an add-on enhancement over an
already-valid baseline, so the right fix is to expose it as a configurable
knob, not to silently hardcode one setting.

- **Multigrid eigensolver (MGE, [BAMG11] Sec. 4.2 / [STATUS14] Sec. 4) -
  RESOLVED: implemented in ``_mge.py``, opt-in via ``k_e``.** Bootstrap
  cycles improve test vectors by running the current AMG cycle on ``A x =
  0`` alone by default ([BAMG11] Sec. 3, Sec. 5's "for l=0..L-1: relax; ...
  repeat: improve V ... recompute C_l, P_l"); [STATUS14] frames MGE as an
  enhancement layered on this already-functional relaxation-only baseline
  (its Tables 1-3 report working two-grid solvers before MGE is even
  introduced; MGE then "consistently improves the performance... when
  compared to" those results) - not as required infrastructure, so ``k_e=0``
  (the default) keeps the historical MGE-free behavior exactly.
  ``k_e > 0`` runs ``_mge.multigrid_eigensolver`` each bootstrap cycle:
  solve the coarsest level's small generalized eigenproblem directly, then
  interpolate and relax the eigenvectors up through every finer level
  (``_mge.py``'s own module docstring has the index-convention translation
  from the papers' numbering). The resulting eigenvector columns are
  appended to the relaxation-improved test vectors at every level
  (``k = k_r + k_e`` total for the LS/LSR fit, per [BAMG11] Table 4.4-4.5).
  ``test_vector_weights``'s ``T = I`` reduction (used in both
  ``_algebraic_distance.algebraic_distance`` and this module's
  ``_prolongation``) is only paper-valid "on the finest level, or before
  any MGE enrichment" - **partially resolved, TODO(bamg-fidelity,
  needs-check) remains.** ``BAMGCoarsening`` now tracks the composite
  prolongation ``P_l = P_0 P_1 ... P_{l-1}`` incrementally as
  ``build_transfer`` runs level by level (``_current_T``, ``None`` -
  meaning ``T = I`` - only at the true finest level of each bootstrap
  cycle's coarsening pass), and threads the resulting ``T_l = P_l^H P_l``
  into both call sites via an optional ``T`` parameter (default ``None``,
  preserving every existing ``k_e=0`` caller's exact behavior). ``T_l =
  P_l^H P_l`` and the composite-prolongation chaining formula are confirmed
  directly against [STATUS14] Sec. 4 (arXiv:1406.1819) - that part is
  solid. **What is NOT confirmed:** that ``omega_kappa`` (the LS/LSR fit
  weight this ``T`` now feeds) is actually supposed to use this same
  ``T_l``. That specific link rests only on ``docs/bootstrap-amg.md``'s own
  transcription, attributed to [BAMG11] Sec. 2/eq. 4.1 - a paper that could
  not be fetched from any mirror tried this session. A direct [STATUS14]
  check of ``omega_kappa`` itself found it described only as ``||v||_A^2 =
  <Av,v>``, with no ``T`` mentioned and an explicit statement that "there is
  no statement linking this T operator to the omega_kappa weights" in that
  paper - not a contradiction (STATUS14's own tables are MGE-free, so it may
  simply not need the fuller formula), but not confirmation either, and
  ``docs/bootstrap-amg.md`` has already been shown wrong once this session
  on an unverifiable [BAMG11]-attributed claim (the LSR single-vector
  schedule). TODO: get [BAMG11] itself (or other independent confirmation)
  before treating this wiring as more than plausible; until then it is
  live but unverified for the weight-formula application (MGE's own use of
  ``T_l`` for the Rayleigh quotient, in ``_mge.py``, is unaffected and
  independently confirmed).
  **Known limitation, not a deviation from [BAMG11]/[STATUS14]:** at small
  problem sizes with ``k_e`` comparable to ``k_r`` (e.g. ``k_r=k_e=8`` on
  ``N=31`` 1D Poisson), the enriched test-vector set can occasionally push
  compatible-relaxation coarsening (Sec. 2.1) to a degenerate single-level
  outcome (measured: 2 of 8 setup seeds at ``N=31``; seed-sensitive, not
  deterministic - CR's own coarsening ratio is inherently less predictable
  than the papers' geometric mesh-doubling coarsening, a pre-existing
  fragility this module's ``_build_levels`` already handles for the
  non-MGE case, not a new failure mode MGE introduces). Callers hitting
  ``BootstrapAMGPreconditioner``'s "produced a single level" ``ValueError``
  with ``k_e > 0`` should lower ``k_e`` relative to ``k_r`` or ``max_coarse``
  relative to ``matrix``'s size.
- **Guidance-graph simplification (CONFIRMED CORRECT against [AD11]
  Algorithm 1).** ``compatible_relaxation_coarsening``'s ``guidance_graph``
  (the Gap-1 fix below) is, by Task 1's own design, a single fixed argument
  for the *whole* CR outer loop - never recomputed as the coarse set ``C``
  grows, exactly like the plain-matrix-graph default it replaces. This
  module follows the same shape: the algebraic-distance strength graph
  ``M_d`` (eq. 4.4) is built once, before any point is marked coarse (every
  node is an "F" candidate for eq. 4.4's own ``i, j in F`` filter at that
  point), and held fixed through CR's loop. This does not admit any edge the
  true, iteration-refreshed ``M_d`` would have excluded:
  ``_independent_set_of``'s own ``eligible``/``candidates`` bookkeeping
  already restricts every lookup to genuinely still-eligible (not-yet-coarse)
  nodes regardless of what the guidance graph itself contains. Checked
  directly against [AD11] (arXiv:1106.5990) Algorithm 1: it computes its
  algebraic-distance matrix once, at initialization, and never recomputes it
  as the independent-set loop proceeds - this module already matches that.
- **Every-level test-vector improvement (RESOLVED: no longer finest-only).**
  [BAMG11] Sec. 5's outer loop, and [STATUS14]'s own description of the base
  bootstrap loop (independent of MGE), improve the test vectors on *every*
  level; neither paper offers a finest-only variant as a sanctioned
  alternative, so this was a mandatory fix, not an optional knob.
  ``BootstrapSetup._improve_levels`` now runs the current partial
  hierarchy's cycle rooted at *each* level ``l`` (sub-hierarchy from ``l``
  down to the coarsest) on that level's own ``A_l x_l = 0`` - the same
  linearity identity ``_improve_test_vectors`` already used for the finest
  level alone, generalized to every level that has a coarser level below it.
  ``_build_levels`` accepts the resulting per-level vectors as an optional
  override of its restriction-derived defaults (``level_vectors``), used
  only during bootstrap-refinement passes, not the initial hierarchy build.
- **LSR practical schedule - RESOLVED against [STATUS14] Sec. 3 directly
  (Table 1 caption, verified against arXiv:1406.1819, not merely a secondary
  transcription).** Quoted exactly: "the update ... used in the LSR
  formulation is applied only to 20% of the entries of the TVs for which the
  associated values of the residual r_i^(kappa) are largest in absolute
  value." Every test vector is corrected - the paper never restricts this
  to a single vector anywhere, a claim checked by an explicit text search of
  the full document - each at the 20% of F-points with the largest absolute
  value of **that same vector's own** residual ``r^(kappa) = A v^(kappa)``
  (eq. 7), not a residual combined across vectors
  (``BAMGCoarsening._fit_vectors``). The paper's own wording is ambiguous
  between a combined-across-vectors and a per-vector reading of "20% of the
  entries of the TVs"; per-vector is used here, since it is the only
  reading that does not implicitly assume every test vector's residual sits
  on a comparable scale to every other's - the same fairness problem this
  fix closes for a single vector's row selection would otherwise resurface
  one level up, across vectors.

  An earlier version of this module instead read [docs/bootstrap-amg.md]'s
  (itself claiming to transcribe [BAMG11] 2011, a PDF that could not be
  independently fetched or verified in this session - every mirror tried
  failed) description of a "single TV with the largest weight" schedule,
  and exposed it as an optional ``lsr_single_vector`` toggle alongside a
  combined-residual "every vector" default. Once [STATUS14] was read
  directly and found to contradict the single-vector claim, that toggle was
  removed rather than kept as an unverifiable option: the per-vector,
  every-vector schedule above is now the only behavior, with a regression
  test (``test_bamg_coarsening_fit_vectors_corrects_every_vector_at_its_own_top_residual_points``)
  constructed so a combined-residual selection and a per-vector one
  disagree and are distinguishable.
- **Near-null-space seed vectors (RESOLVED: exposed as ``seed_vectors``).**
  [STATUS14] Table 3 presents seeding a known near-null vector (e.g. the
  constant vector ``1``) as a "when known" enhancement over an
  already-valid random-only baseline, not a requirement. ``BootstrapSetup.run``
  and ``BootstrapAMGPreconditioner.__init__`` accept ``seed_vectors: torch.Tensor
  | None`` (default ``None``, preserving the historical random-only
  behavior), concatenated alongside the ``k_r`` random draws by
  ``_seeded_test_vectors`` - deliberately additive rather than replacing,
  unlike ``adaptive.py``'s ``initial_candidates`` (which replaces the
  initial random start): that asymmetry follows the two papers' own
  conventions, not an inconsistency between the two presets.
- **Test-vector distribution vs. CR's internal probe distribution - FIXED
  (real bug, not just a fidelity gap): exposed as ``test_vector_draw``.**
  [STATUS14] Sec. 3's Table 1 specifies the initial ``k_r`` test vectors as
  "generated randomly with a normal distribution ... N(0,1)"; this module's
  single injectable ``draw`` source was, before this fix, reused both for
  that and for compatible relaxation's own internal convergence-rate probe
  (``cr_rate`` in ``_compatible_relaxation.py``, documented to expect a
  uniform ``[0, 1)`` start - its finite, ``nu``-sweep power-iteration-style
  estimate is sensitive to how strongly the start vector is already biased
  toward the near-constant/smooth mode, which a uniform ``[0,1)`` draw is
  and a mean-zero N(0,1) draw is not). Reusing one ``draw`` for both meant
  switching the test-vector distribution to match [STATUS14] silently broke
  CR: reproduced directly, not merely inferred - a normal-distributed
  ``draw`` collapsed compatible-relaxation coarsening to a single level on
  every matrix tried, including a plain 256-node isotropic 2D Poisson
  fixture, while the identical setup with a uniform ``draw`` (or with CR's
  own probe kept uniform while only the test vectors go normal) coarsened
  normally. ``BootstrapSetup.run``/``BootstrapAMGPreconditioner.__init__``
  now accept ``test_vector_draw: Callable[[int], torch.Tensor] | None``
  (default ``None``, reusing ``draw`` - the historical, still-supported
  behavior for callers who never need the two distributions to differ),
  used only for the initial ``k_r`` test vectors; ``draw`` keeps seeding
  CR's probe regardless. See ``test_bootstrap.py``'s
  ``test_bootstrap_setup_run_normal_test_vectors_do_not_break_cr_coarsening``
  for the regression case, and
  ``tests/benchmarks/preconditioners/test_bootstrap_amg.py``'s
  ``test_bootstrap_amg_matches_status14_table1_exact_setup`` for the fix
  applied to reproduce Table 1's exact setup end-to-end.
- **Solve-time cycle sweep counts (RESOLVED: exposed as ``n_pre``/``n_post``).**
  [STATUS14] treats smoothing-sweep counts as experiment-specific, not
  mandated (its Table 1 sweeps eta = 2, 4, 6, 8; V(2,2) is "results from
  [7]," a reported configuration, not a required setting).
  ``BootstrapAMGPreconditioner.__init__`` accepts ``n_pre``/``n_post``
  (default ``1, 1``, preserving the historical hardcoded V(1,1)), threaded
  through ``_presets.prebuilt_cycle`` to the underlying ``VCycle``, which
  already supported both.
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
from ._mge import multigrid_eigensolver
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
"""Fraction of F-points corrected by LSR - [STATUS14] Table 1's caption gives
this practical 20%/largest-residual schedule; [STATUS14] Sec. 3's own text
attributes the same result to "[7]" ([BAMG11]) but the exact section within
[BAMG11] (previously cited here as "Sec. 4") was never independently
confirmed, since [BAMG11] could not be fetched this session."""


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


def _seeded_test_vectors(
    matrix: torch.Tensor,
    k_r: int,
    draw: Callable[[int], torch.Tensor],
    seed_vectors: torch.Tensor | None,
) -> torch.Tensor:
    """``k_r`` random test vectors, plus known near-null vectors if given ([STATUS14] Table 3).

    [STATUS14] Table 3 seeds a known near-null vector (e.g. the constant
    vector ``1``) *alongside* the random draws, not instead of them -
    deliberately different from ``adaptive.py``'s ``initial_candidates``,
    which replaces the initial random start: that asymmetry follows the two
    papers' own conventions, not an inconsistency between the two presets.

    Args:
        matrix (torch.Tensor): Level matrix, shape ``(n, n)``.
        k_r (int): Number of random test vectors to draw.
        draw (Callable[[int], torch.Tensor]): Uniform ``[0, 1)`` random-vector
            source, ``n -> tensor of length n``.
        seed_vectors (torch.Tensor | None): Known near-null vectors, shape
            ``(n, k_seed)``; ``None`` for no seeding (the historical
            behavior).

    Returns:
        torch.Tensor: Test vectors, shape ``(n, k_r + k_seed)`` (or
        ``(n, k_r)`` when ``seed_vectors`` is ``None``).
    """
    random_vectors = _random_test_vectors(matrix, k_r, draw)
    if seed_vectors is None:
        return random_vectors
    return torch.cat(
        [random_vectors, seed_vectors.to(dtype=matrix.dtype, device=matrix.device)], dim=1
    )


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
        self._composite: torch.Tensor | None = None
        """Composite prolongation ``P_l = P_0 P_1 ... P_{l-1}`` from the
        *current* level ``l`` up to the finest level, built incrementally as
        ``build_transfer`` runs level by level; ``None`` at the finest level
        (no coarsening has happened yet within this pass, so ``T_0 = I``
        trivially - see ``_current_T``). A fresh ``BAMGCoarsening`` per
        bootstrap cycle (``BootstrapSetup._coarsening_from``) means this
        always starts ``None`` at the true finest level, exactly matching
        [BAMG11]/[STATUS14]'s own ``T_0 = I``."""

    def _current_T(self) -> torch.Tensor | None:
        """Composite-interpolation Gram operator ``T_l = P_l^H P_l`` for the level about to be processed.

        [BAMG11] Sec. 4.2 / ``docs/bootstrap-amg.md`` line 234: ``T`` only
        differs from the identity once at least one coarser level has been
        built within this pass, i.e. never at the finest level. Returning
        ``None`` (rather than materializing an explicit identity matrix)
        lets ``test_vector_weights``/``algebraic_distance`` take their
        cheaper ``T = I`` path directly.

        Returns:
            torch.Tensor | None: ``T_l``, shape ``(n, n)`` for the current
            level's dimension ``n``, or ``None`` for ``T = I``.
        """
        return self._composite.T @ self._composite if self._composite is not None else None

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
        T = self._current_T()
        distance = algebraic_distance(test_vectors, A, depth=self._depth, T=T)
        coarse_mask = self._coarse_mask(A, distance)
        prolongation = self._prolongation(A, test_vectors, coarse_mask, distance, T=T)
        coarse_matrix = prolongation.T @ A @ prolongation
        self._last_prolongation = prolongation
        self._vectors[coarse_matrix.shape[0]] = prolongation.T @ test_vectors
        self._composite = (
            prolongation if self._composite is None else self._composite @ prolongation
        )
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
        # `distance` (and therefore `directed`) is computed once per level
        # from the vectors at level entry and never refreshed as C grows
        # within this call - confirmed to match [AD11] Algorithm 1, which
        # does the same (see the module docstring's "Guidance-graph
        # simplification" entry).
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
        self,
        A: torch.Tensor,
        test_vectors: torch.Tensor,
        fine_points: torch.Tensor,
    ) -> torch.Tensor:
        """Test vectors the LS fit uses: LSR-corrected if enabled, else as-is.

        [STATUS14] Sec. 3 (Table 1 caption, verified directly against
        arXiv:1406.1819): "the update ... used in the LSR formulation is
        applied only to 20% of the entries of the TVs for which the
        associated values of the residual ``r_i^(kappa)`` are largest in
        absolute value." Every test vector is corrected - the paper never
        restricts this to a single vector anywhere - each at the 20% of
        F-points with the *largest absolute value of that same vector's own*
        residual ``r^(kappa) = A v^(kappa)`` (eq. 7); not a residual combined
        across vectors, which would make the selection for one vector
        depend on how large another, unrelated vector's residual happens to
        be at the same points - the paper's own text is ambiguous between a
        combined-across-vectors and a per-vector reading, so this reads it
        as per-vector, since that is the only interpretation that does not
        implicitly assume every test vector's residual sits on a comparable
        scale to every other's.

        Args:
            A (torch.Tensor): Fine-grid matrix, shape ``(n, n)``.
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
            corrected_column = lsr_correction(test_vectors[:, column : column + 1], A, target_rows)
            corrected[:, column] = corrected_column[:, 0]
        return corrected

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
        T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """LS/LSR-fitted prolongation matrix ``P`` (identity on ``C``, fitted rows on ``F``).

        Args:
            A (torch.Tensor): Fine-grid matrix, shape ``(n, n)``.
            test_vectors (torch.Tensor): Test vectors, shape ``(n, k)``.
            coarse_mask (torch.Tensor): Boolean coarse mask, shape ``(n,)``.
            distance (torch.Tensor): Algebraic-distance matrix ``r``, shape
                ``(n, n)``, used by the empty-interpolatory-set fallback.
            T (torch.Tensor | None): Composite-interpolation Gram operator
                for the LS/LSR fit weights ([BAMG11] eq. 4.1); ``None`` for
                the ``T = I`` reduction (see ``_current_T``).

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

        weights = test_vector_weights(test_vectors, A, T=T)
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
        k_e (int): Number of multigrid-eigensolver (MGE) eigenvector
            approximations per bootstrap cycle ([BAMG11] Algorithm 1,
            ``_mge.py``); ``0`` disables MGE (the historical behavior -
            [STATUS14] frames MGE as an optional enhancement over an
            already-valid relaxation-only baseline, not a requirement).
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
    k_e: int = 0
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

    def _improve_levels(
        self,
        levels: list[torch.Tensor],
        prolongations: list[torch.Tensor],
        level_vectors: dict[int, torch.Tensor],
    ) -> dict[int, torch.Tensor]:
        """Improve every level's relaxation-derived test vectors via its own sub-hierarchy cycle.

        [BAMG11] Sec. 5's outer loop, and [STATUS14]'s base bootstrap loop
        (independent of MGE), improve the test vectors on *every* level each
        bootstrap cycle - "the current AMG cycle replaces plain relaxation"
        for the ``for l=0,...,L-1: relax on A_l x_l=0`` step ([BAMG11] eq.
        3.1). This applies that replacement at every level ``l`` that has a
        coarser level below it (``0..len(levels)-2``), rooting the cycle at
        ``levels[l]`` with the sub-hierarchy from ``l`` down to the
        coarsest - the same linearity identity ``_improve_test_vectors``
        already uses, just no longer hardcoded to level ``0``. The coarsest
        level is excluded: it has no coarser level to correct with, so a
        "cycle" rooted there is just a direct solve, not an improvement step.

        Takes ``level_vectors`` directly (the relaxation-derived family
        ``V^r`` alone) rather than reading a ``BAMGCoarsening``'s stored
        state, because that state may also carry MGE's ``V^e`` columns
        appended for the *previous* cycle's LS/LSR fit ([BAMG11] Algorithm
        1) - ``V^r`` must keep exactly ``run``'s original column count across
        every cycle, or MGE columns would compound cycle over cycle instead
        of being freshly recomputed each time (``multigrid_eigensolver``
        never "improves" a stored eigenvector across cycles; it re-solves
        the coarsest eigenproblem from scratch every call).

        Args:
            levels (list[torch.Tensor]): Level matrices, finest first.
            prolongations (list[torch.Tensor]): Prolongations, one per level
                except the coarsest.
            level_vectors (dict[int, torch.Tensor]): Each level's current
                ``V^r`` test vectors, keyed by matrix dimension (every level
                except the coarsest).

        Levels whose dimension has no entry in ``level_vectors`` are skipped
        (not included in the returned dict) rather than raising: coarsening
        decisions can shift between cycles (they are guided by the test
        vectors, which change), so an intermediate level from the previous
        cycle's hierarchy is not guaranteed to reappear at the same
        dimension. The finest level's dimension never changes, so it is
        always present.

        Returns:
            dict[int, torch.Tensor]: Improved ``V^r`` test vectors keyed by
            each improved level's matrix dimension (every level except the
            coarsest, and skipping any level whose dimension is not found).
        """
        return {
            levels[index].shape[0]: _improve_test_vectors(
                levels[index],
                level_vectors[levels[index].shape[0]],
                _hierarchy_of(levels[index:], prolongations[index:]),
                self.eta,
            )
            for index in range(len(levels) - 1)
            if levels[index].shape[0] in level_vectors
        }

    def _merge_level_vectors(
        self, relaxed: dict[int, torch.Tensor], enriched: dict[int, torch.Tensor]
    ) -> dict[int, torch.Tensor]:
        """Concatenate MGE eigenvector columns onto the relaxation-improved vectors, per level.

        [BAMG11] Algorithm 1: "MGE-enriched TVs are appended to the
        relaxation-derived ones" (``k = k_r + k_e`` total for the LS/LSR
        fit) - ``relaxed`` and ``enriched`` share the same keys
        (``BootstrapSetup._improve_levels`` and ``multigrid_eigensolver``
        both cover every level except the coarsest, for the same reason).

        Args:
            relaxed (dict[int, torch.Tensor]): Per-level vectors from
                ``_improve_levels``, keyed by matrix dimension.
            enriched (dict[int, torch.Tensor]): Per-level MGE eigenvector
                approximations from ``multigrid_eigensolver``, same keys.

        Returns:
            dict[int, torch.Tensor]: Concatenated per-level vectors, same
            keys as both inputs.
        """
        return {
            dimension: torch.cat([vectors, enriched[dimension]], dim=1)
            for dimension, vectors in relaxed.items()
        }

    @staticmethod
    def _v_r_after_build(
        coarsening: BAMGCoarsening, levels: list[torch.Tensor], k_r_total: int
    ) -> dict[int, torch.Tensor]:
        """The ``V^r`` (relaxation-derived) columns of every level's post-``_build_levels`` vectors.

        ``_build_levels`` applies its own leading ``eta``-sweep relaxation to
        whatever is stored per level (Sec. 5's per-level block), independent
        of any MGE columns a cycle may have appended for the LS/LSR fit
        (``_merge_level_vectors``). Concatenation order is fixed (``V^r``
        first, then ``V^e``) and relaxation is column-independent
        (``_relax_columns`` relaxes one column at a time), so slicing to the
        first ``k_r_total`` columns after the build recovers exactly ``V^r``
        with that leading relaxation applied - the correct starting point
        for the *next* cycle's ``_improve_levels`` call, matching what a
        single shared, mutated ``coarsening`` object would have given before
        MGE made "the vectors currently stored per level" ambiguous between
        ``V^r`` alone and ``V^r`` concatenated with a transient ``V^e``.

        Args:
            coarsening (BAMGCoarsening): Coarsening strategy just returned
                from a ``_build_levels`` call.
            levels (list[torch.Tensor]): That same call's returned levels.
            k_r_total (int): ``V^r``'s fixed column count (``k_r`` plus any
                ``seed_vectors``), constant for the whole ``run`` call.

        Returns:
            dict[int, torch.Tensor]: ``V^r`` columns per level, keyed by
            matrix dimension (every level except the coarsest).
        """
        return {
            level.shape[0]: coarsening.test_vectors_for(level)[:, :k_r_total]
            for level in levels[:-1]
        }

    def _build_levels(
        self,
        matrix: torch.Tensor,
        coarsening: BAMGCoarsening,
        relaxation: _Relaxation,
        level_vectors: dict[int, torch.Tensor] | None = None,
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
            level_vectors (dict[int, torch.Tensor] | None): Per-level
                test vectors from ``_improve_levels``, keyed by matrix
                dimension, overriding ``coarsening``'s restriction-derived
                starting vectors for that level before its leading
                ``eta``-sweep relaxation; ``None`` on the very first
                hierarchy build, which has no prior levels to improve.

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: Level matrices
            (finest first) and prolongations.
        """
        levels = [matrix]
        prolongations: list[torch.Tensor] = []
        while len(levels) < self.max_levels and levels[-1].shape[0] > self.max_coarse:
            current = levels[-1]
            if level_vectors is not None and current.shape[0] in level_vectors:
                coarsening.set_test_vectors(current, level_vectors[current.shape[0]])
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

    def run(
        self,
        matrix: torch.Tensor,
        draw: Callable[[int], torch.Tensor],
        seed_vectors: torch.Tensor | None = None,
        test_vector_draw: Callable[[int], torch.Tensor] | None = None,
    ) -> BootstrapAMGResult:
        """Run the Bootstrap AMG setup algorithm.

        Args:
            matrix (torch.Tensor): SPD system matrix, shape ``(n, n)``.
            draw (Callable[[int], torch.Tensor]): Uniform ``[0, 1)``
                random-vector source for compatible relaxation's own
                internal convergence-rate probe (``cr_rate``); CR's finite,
                ``nu``-sweep power-iteration-style estimate is documented to
                expect this specific distribution (a uniform start is
                biased toward the near-constant/smooth mode CR's rate
                measures alignment with) - passing a signed distribution
                here (e.g. N(0,1)) can make CR's stopping criterion misfire
                and collapse coarsening to a single level, independent of
                what distribution the test vectors themselves come from.
            seed_vectors (torch.Tensor | None): Known near-null vectors,
                shape ``(n, k_seed)``, seeded alongside the ``k_r`` random
                draws ([STATUS14] Table 3); ``None`` for no seeding.
            test_vector_draw (Callable[[int], torch.Tensor] | None):
                Random-vector source for the initial ``k_r`` test vectors
                specifically ([STATUS14] Sec. 3's Table 1 uses N(0,1) here,
                a different distribution than CR needs); ``None`` reuses
                ``draw`` for this too, the historical behavior.

        Returns:
            BootstrapAMGResult: Level matrices, prolongations and the final
            finest-level test vectors.
        """
        relaxation = GaussSeidelSmoother().smooth
        vectors = _seeded_test_vectors(
            matrix,
            self.k_r,
            test_vector_draw if test_vector_draw is not None else draw,
            seed_vectors,
        )
        k_r_total = vectors.shape[1]
        coarsening = self._coarsening_from(vectors, relaxation, draw)
        levels, prolongations = self._build_levels(matrix, coarsening, relaxation)
        vectors = coarsening.test_vectors_for(matrix)
        # V^r only (no MGE columns yet) - the persistent state `_improve_levels`
        # improves cycle over cycle; kept separate from whatever combined
        # (V^r + V^e) vectors a given cycle feeds to `_build_levels` for
        # fitting, so MGE columns never compound across cycles (see
        # `_improve_levels`'s docstring).
        relaxed_level_vectors = self._v_r_after_build(coarsening, levels, k_r_total)

        for _ in range(self.n_bootstrap_cycles):
            if len(levels) < 2:
                break
            relaxed_level_vectors = self._improve_levels(
                levels, prolongations, relaxed_level_vectors
            )
            level_vectors = relaxed_level_vectors
            if self.k_e > 0:
                enriched = multigrid_eigensolver(
                    levels, prolongations, self.k_e, relaxation, self.eta
                )
                level_vectors = self._merge_level_vectors(relaxed_level_vectors, enriched)
            vectors = level_vectors[matrix.shape[0]]
            coarsening = self._coarsening_from(vectors, relaxation, draw)
            levels, prolongations = self._build_levels(
                matrix, coarsening, relaxation, level_vectors=level_vectors
            )
            vectors = coarsening.test_vectors_for(matrix)
            relaxed_level_vectors = self._v_r_after_build(coarsening, levels, k_r_total)

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
        k_e (int): Number of MGE eigenvector approximations per bootstrap
            cycle ([BAMG11] Algorithm 1); ``0`` disables MGE (the
            historical behavior - [STATUS14] frames MGE as an optional
            enhancement over an already-valid baseline, not a requirement).
        max_levels (int): Maximum number of levels.
        max_coarse (int): Stop coarsening at this many coarse nodes.
        seed (int): Seed of the random test-vector source.
        draw (Callable[[int], torch.Tensor] | None): Explicit uniform
            ``[0, 1)`` source overriding ``seed``.
        seed_vectors (torch.Tensor | None): Known near-null vectors, shape
            ``(n, k_seed)``, seeded alongside the ``k_r`` random draws
            ([STATUS14] Table 3); ``None`` for no seeding.
        smoother_omega (float | None): Damping for the default
            weighted-Jacobi solve smoother; ``None`` selects
            ``1 / rho(D^-1 A)`` per level.
        smoother (MultigridSmoother | None): Explicit solve-time smoother.
            ``None`` selects weighted Jacobi. When supplied,
            ``smoother_omega`` must remain ``None``.
        n_pre (int): Solve-time pre-smoothing sweeps.
        n_post (int): Solve-time post-smoothing sweeps.

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
        k_e: int = 0,
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
            k_e (int): Number of MGE eigenvector approximations per
                bootstrap cycle; ``0`` disables MGE.
            max_levels (int): Maximum number of levels.
            max_coarse (int): Stop coarsening at this many coarse nodes.
            seed (int): Seed of the random test-vector source.
            draw (Callable[[int], torch.Tensor] | None): Explicit uniform
                ``[0, 1)`` source overriding ``seed``.
            seed_vectors (torch.Tensor | None): Known near-null vectors,
                seeded alongside the ``k_r`` random draws; ``None`` for no
                seeding.
            test_vector_draw (Callable[[int], torch.Tensor] | None):
                Random-vector source for the initial ``k_r`` test vectors
                specifically, distinct from ``draw``/``seed`` (which also
                seed compatible relaxation's own internal probe - see
                ``BootstrapSetup.run``'s docstring for why the two must not
                be conflated); ``None`` reuses ``draw``/``seed`` for this
                too, the historical behavior.
            smoother_omega (float | None): Damping for the default
                weighted-Jacobi solve smoother; ``None`` selects the
                per-level spectral rule.
            smoother (MultigridSmoother | None): Explicit solve-time
                smoother, or ``None`` for weighted Jacobi.
            n_pre (int): Solve-time pre-smoothing sweeps.
            n_post (int): Solve-time post-smoothing sweeps.

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
            k_e=k_e,
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
