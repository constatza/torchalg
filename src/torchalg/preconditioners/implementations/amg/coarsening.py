"""Coarsening strategies for AMG hierarchy construction.

Ported from ``neuralls.domain.solver.preconditioners.implementations.amg.coarsening``
(see ``docs/plan.md``'s dense-only directive). ``SparseAggregationCoarsening``
is renamed to ``AggregationCoarsening`` and now builds dense ``torch.Tensor``
P/R operators instead of ``scipy.sparse`` matrices - the five-step SA-AMG
algorithm itself (Vanek, Mandel & Brezina 1996) is unchanged; only the
storage of the intermediate strength/aggregation/prolongation arrays moves
from CSR to dense. The step-by-step kernels live in ``_aggregation.py``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from ._aggregation import (
    piecewise_constant_prolongation,
    smoothed_prolongation,
    standard_aggregation,
    strength_of_connection,
)
from ._jacobi_omega import PROLONGATION_NOMINAL, jacobi_omega
from ._theta_search import adaptive_theta_scan
from .transfer import DenseTransferOperator, NeuralTransferOperator

if TYPE_CHECKING:
    import torch

    from ...ports import ExtraInputPredictorPort


@lru_cache(maxsize=1024)
def _cached_aggregation_build_transfer(
    A: torch.Tensor, theta: float, omega: float | None
) -> tuple[torch.Tensor, DenseTransferOperator]:
    """Build (and cache) one `AggregationCoarsening` candidate for `(A, theta, omega)`.

    `A` hashes and compares by object identity (the default for
    `torch.Tensor`), so this only dedupes calls sharing the exact matrix
    object already resident from one search - never a false hit across two
    distinct, coincidentally-equal matrices. Backs
    `TargetDimensionCoarsening`'s `cache_candidates=True` option.
    """
    return AggregationCoarsening(theta=theta, omega=omega).build_transfer(A)


class AggregationCoarsening:
    """Smoothed aggregation AMG coarsening using dense tensors (SA-AMG).

    Implements the five-step coarsening algorithm from Vanek, Mandel & Brezina
    (1996):

    1. Strength-of-connection: mark strong off-diagonal entries via
       ``|a_ij| / sqrt(|a_ii| * |a_jj|) >= theta`` (Vanek, Mandel & Brezina
       1996's Cauchy-Schwarz strength measure, not Stuben's classical/RS
       measure - see ``strength_of_connection``'s docstring).
    2. Aggregation: partition nodes into non-overlapping aggregates using
       the strong-connection graph via ``standard_aggregation``'s
       field-standard three-pass algorithm (PyAMG's
       ``amg_core::standard_aggregation`` - see that function's docstring
       for the full three-pass description).
    3. Tentative prolongation P0: piecewise-constant indicator matrix
       (aggregate membership).
    4. Smoothed prolongation: ``P = (I - omega D^{-1} A) P0`` with
       ``omega = (4/3) / rho(D^{-1}A)`` by default (Vanek et al. 1996,
       Eq. 3.2; PyAMG's rule), ~= 2/3 for isotropic SPD.
    5. Galerkin coarse matrix: ``A_coarse = P.T @ A @ P``.

    Args:
        theta (float): Strength-of-connection threshold theta in (0, 1).
            Default 0.25 is a literature/practice-standard default (e.g.
            hypre BoomerAMG's ``strong_threshold``, PyAMG's
            ``smoothed_aggregation_solver``), not from Stuben (2001).
        omega (float | None): Jacobi damping for the prolongation smoother.
            ``None`` (default) is ``(4/3) / rho(D^{-1}A)``, estimated once per
            level matrix and shared with the relaxation (see
            ``_jacobi_omega``); a float fixes it.

    References:
        - Vanek, P., Mandel, J., & Brezina, M. (1996). Algebraic multigrid by
          smoothed aggregation for second and fourth order elliptic problems.
          Computing, 56(3), 179-196.
        - Xu, J., & Zikatanov, L. (2017). Algebraic multigrid methods. Acta
          Numerica, 26, 591-721 (arXiv:1611.01917).
    """

    def __init__(self, theta: float = 0.25, omega: float | None = None) -> None:
        """Store the strength-of-connection threshold and smoothing damping factor.

        Args:
            theta (float): Strength-of-connection threshold theta in (0, 1).
            omega (float | None): Jacobi damping, or ``None`` for ``(4/3) / rho``.
        """
        self._theta = theta
        self._omega = omega

    def build_transfer(self, A: torch.Tensor) -> tuple[torch.Tensor, DenseTransferOperator]:
        """Build one coarse level from fine-grid matrix A.

        Args:
            A (torch.Tensor): Fine-grid matrix ``A``, shape ``(n, n)``. Named
                to match the ``CoarseningStrategy`` protocol's parameter
                name exactly (structural typing checks parameter names for
                positional-or-keyword parameters).

        Returns:
            tuple[torch.Tensor, DenseTransferOperator]: ``(A_coarse,
                transfer)`` where ``A_coarse`` is a dense ``(n_c, n_c)``
                tensor.
        """
        strength = strength_of_connection(A, self._theta)
        aggregate = standard_aggregation(strength)
        tentative = piecewise_constant_prolongation(aggregate, dtype=A.dtype)
        omega = jacobi_omega(A, PROLONGATION_NOMINAL, self._omega)
        prolongation = smoothed_prolongation(A, tentative, omega)

        coarse_matrix = prolongation.T @ A @ prolongation
        return coarse_matrix, DenseTransferOperator(prolongation)


class TargetDimensionCoarsening:
    """Reaches a target coarse dimension by searching `AggregationCoarsening`'s theta.

    `theta` is a strength-of-connection threshold — the coarse dimension it
    produces is emergent, not chosen, and empirically a *step function* of
    `theta` rather than a smooth or even monotonic one (a small change in
    `theta` can jump the realized dimension by a large amount, or not move
    it at all). This wraps `AggregationCoarsening` from the outside — it
    only calls its existing public ``build_transfer`` and never modifies it
    — to give aggregation coarsening the same "set the coarse dimension
    directly" ergonomics a fixed-rank strategy like `PODCoarseningStrategy`
    already has.

    **Why exhaustive grid search, not bisection or a black-box optimizer**:
    re-verified directly against `standard_aggregation` (the current
    field-standard three-pass algorithm, not the single-pass
    `greedy_aggregation` this class originally wrapped) that realized
    dimension is genuinely, not just theoretically, non-monotonic in
    `theta` on realistic heterogeneous stiffness matrices — a 5x5 2D
    5-point-stencil grid (n=25) with a central disk-shaped "soft
    inclusion" patch (radius 2, 13 of 25 nodes, stencil weights scaled
    1000x softer — the "soft sphere in a stiffer medium" case this class
    exists to serve) jumps from a realized dimension of 6 at theta=0.01 to
    8 at theta=0.02, and a 20x20 analog with the same soft-inclusion
    construction (49-node patch, both 10x and 1000x softness) shows further
    non-monotone jumps deeper into its theta range (e.g. 70 -> 74
    aggregates from theta=0.08 to 0.09). See
    `tests/.../conftest.py::soft_inclusion_2d_stiffness` and
    `TestTargetDimensionCoarsening.test_realized_dimension_is_not_monotonic_in_theta`
    for the exact fixture and reproduction. The mechanism: `standard_aggregation`'s
    Pass 1 is a left-to-right, order-dependent seeding process (which node
    becomes unblocked first, and therefore seeds, depends on the exact
    strength graph at each `theta`), so a stricter `theta` can change which
    node claims a given neighbor and shift the aggregate count either way,
    even though the underlying strength-of-connection graph only ever
    shrinks as `theta` grows. On the flat, homogeneous `poisson_1d` fixture
    this never surfaces (see
    `tests/.../test_amg.py::TestTargetDimensionCoarsening`'s two-plateau
    docstring) — uniform stencils don't create the local competition
    between seeds that produces it — which is exactly why that fixture
    alone was insufficient to settle the question; heterogeneous stiffness
    is required to trigger it. Bisection assumes monotonicity and can
    converge to the wrong plateau given a matrix like the one above. A
    sampler built for expensive, high-dimensional, exploitably-smooth
    objectives (Optuna, Bayesian optimization) earns nothing here either:
    `theta` is a single bounded scalar, each trial is one cheap dimension
    probe, and there is no smooth trend to model. `_search` uses
    `adaptive_theta_scan` (`_theta_search.py`): a `step`-spaced *coarse*
    pass (log-spaced, not linear — see that module's docstring) followed
    by bisection of every disagreeing interval down past `step`. A plain
    fixed `step` grid, linear or log, is not enough on its own: on a
    heterogeneous 25x25 soft-inclusion grid (see `.claude/plan.md`'s
    investigation notes), the true `theta -> dim` staircase has plateaus
    only ~0.003-0.006 wide sitting between much wider ones, so a `step`
    anywhere near a typical default (0.01-0.1) can straddle and skip them
    outright — silently handing back the same, coarser-than-intended
    dimension for two genuinely different `target_coarse_dim` requests.
    `adaptive_theta_scan` closes that: any plateau boundary the coarse
    pass detects (i.e. two adjacent coarse samples disagree) gets bisected
    down near machine precision, not just to `step`, so a plateau
    narrower than `step` but sandwiched between two disagreeing coarse
    points is still found. The one case no finite-budget grid can catch
    is a plateau narrower than the coarse spacing whose neighbors happen
    to agree on *both* sides.

    The search itself is isolated in `_search` precisely so it can be
    swapped later (e.g. for bisection) without touching `build_transfer`
    or any caller, should a future coarsening strategy make the
    `theta -> realized dimension` relationship monotonic.

    Args:
        target_coarse_dim (int): Desired realized coarse dimension.
        theta_min (float): Lower bound of the `theta` search grid.
        theta_max (float): Upper bound of the `theta` search grid.
        step (float): Grid spacing; smaller values give finer resolution at
            proportionally higher cost (one `build_transfer` call per step).
        omega (float): Prolongation Jacobi-smoothing damping, forwarded to
            each candidate `AggregationCoarsening`.
        cache_candidates (bool): If ``True``, the winning candidate's full
            `AggregationCoarsening.build_transfer` is memoized by
            `(A, theta, omega)` (object identity on `A`) via a shared
            `functools.lru_cache`, so a sibling instance searching the same
            matrix and landing on the same `theta` reuses the build instead
            of repeating it. Off by default - opt in only when multiple
            instances against the same matrix are expected (e.g. several
            `target_coarse_dim` values in one comparison sweep); the cache
            keeps matrix tensors alive up to `maxsize` entries, an
            unnecessary memory trade-off for a single one-shot instance.
    """

    def __init__(
        self,
        target_coarse_dim: int,
        *,
        theta_min: float,
        theta_max: float,
        step: float,
        omega: float | None = None,
        cache_candidates: bool = False,
    ) -> None:
        """Store the target dimension and search parameters; unbuilt until `build_transfer`.

        Args:
            target_coarse_dim (int): Desired realized coarse dimension.
            theta_min (float): Lower bound of the `theta` search grid.
            theta_max (float): Upper bound of the `theta` search grid.
            step (float): Grid spacing for the exhaustive search.
            omega (float | None): Prolongation Jacobi-smoothing damping, or
                ``None`` for the spectral rule ``(4/3) / rho(D^-1 A)``.
            cache_candidates (bool): Share the winning candidate's full
                build across instances against the same matrix - see the
                class docstring.
        """
        self._target_coarse_dim = target_coarse_dim
        self._theta_min = theta_min
        self._theta_max = theta_max
        self._step = step
        self._omega = omega
        self._cache_candidates = cache_candidates
        self._theta: float | None = None
        self._realized_coarse_dim: int | None = None

    def build_transfer(self, A: torch.Tensor) -> tuple[torch.Tensor, DenseTransferOperator]:
        """Search `theta`, then build the coarse level at the closest-matching value.

        The winning `theta` and its realized coarse dimension are cached as
        ``self._theta``/``self._realized_coarse_dim`` afterward (mirroring
        `AggregationCoarsening`'s own private-attribute convention), so
        callers that already have this object can read the result directly
        instead of triggering a second full search.

        Args:
            A (torch.Tensor): Fine-grid matrix ``A``, shape ``(n, n)``. Named
                to match the ``CoarseningStrategy`` protocol's parameter
                name exactly (structural typing checks parameter names for
                positional-or-keyword parameters).

        Returns:
            tuple[torch.Tensor, DenseTransferOperator]: ``(A_coarse,
                transfer)`` from the `AggregationCoarsening` candidate whose
                realized coarse dimension is closest to
                ``target_coarse_dim``.
        """
        theta, a_coarse, transfer = self._search(A)
        self._theta = theta
        self._realized_coarse_dim = int(a_coarse.shape[0])
        return a_coarse, transfer

    def _search(self, A: torch.Tensor) -> tuple[float, torch.Tensor, DenseTransferOperator]:
        """Adaptively scan the `theta` grid, keeping the closest match to `target_coarse_dim`.

        The sole extension point for swapping the search algorithm (e.g.
        for bisection, if a future coarsening strategy makes
        ``theta -> realized dimension`` monotonic) — everything else on
        this class (`build_transfer`'s caching, the constructor) is
        independent of how the search itself is performed.

        Sampling uses a cheap dimension-only probe (strength + aggregation,
        skipping prolongation smoothing and the Galerkin product) for every
        candidate `theta`; the full `AggregationCoarsening.build_transfer`
        is called exactly once, at the winning `theta`.

        Args:
            A (torch.Tensor): Fine-grid matrix, shape ``(n, n)``.

        Returns:
            tuple[float, torch.Tensor, DenseTransferOperator]: ``(theta,
                A_coarse, transfer)`` for the candidate whose realized
                coarse dimension is closest to ``target_coarse_dim``.
        """

        def realized_dimension(theta: float) -> int:
            aggregate = standard_aggregation(strength_of_connection(A, theta))
            return int(aggregate.max().item()) + 1 if aggregate.numel() else 0

        samples = adaptive_theta_scan(
            self._theta_min, self._theta_max, self._step, realized_dimension
        )
        theta = min(samples, key=lambda sample: abs(sample[1] - self._target_coarse_dim))[0]
        if self._cache_candidates:
            a_coarse, transfer = _cached_aggregation_build_transfer(A, theta, self._omega)
        else:
            a_coarse, transfer = AggregationCoarsening(
                theta=theta, omega=self._omega
            ).build_transfer(A)
        return theta, a_coarse, transfer


class NeuralCoarseningStrategy:
    """Coarsening with neural prolongation/restriction operators.

    Implements both ``CoarseningStrategy`` and ``BindableInputs`` so that
    static domain data (positions, theta) can be bound before the hierarchy
    is built. Injected predictors are already loaded; this class does not
    load models (that is the factory's job).

    Args:
        prolongator (ExtraInputPredictorPort): Neural predictor for
            coarse->fine mapping.
        restrictor (ExtraInputPredictorPort | None): Neural predictor for
            fine->coarse mapping, or ``None`` to use a placeholder
            (``build_transfer`` will raise).

    Note:
        ``build_transfer`` raises ``NotImplementedError`` until the neural
        operators are fully tested against a specific checkpoint format.
        The class structure is stable; only the forward pass needs wiring.
    """

    def __init__(
        self,
        prolongator: ExtraInputPredictorPort,
        restrictor: ExtraInputPredictorPort | None,
    ) -> None:
        """Store the injected neural predictors (already loaded).

        Args:
            prolongator (ExtraInputPredictorPort): Neural predictor for
                coarse->fine mapping.
            restrictor (ExtraInputPredictorPort | None): Neural predictor
                for fine->coarse mapping, or ``None`` to use a placeholder.
        """
        self._prolongator = prolongator
        self._restrictor = restrictor
        self._bound: dict[str, torch.Tensor] = {}

    @property
    def extra_input_names(self) -> tuple[str, ...]:
        """Union of required inputs from both predictors, deduplicated.

        Returns:
            tuple[str, ...]: Deduplicated names of extra tensors required by
                the prolongator and (if bound) the restrictor.
        """
        names: list[str] = list(self._prolongator.required_inputs)
        if self._restrictor is not None:
            names.extend(self._restrictor.required_inputs)
        return tuple(dict.fromkeys(names))

    def bind_inputs(self, **inputs: torch.Tensor) -> None:
        """Pre-bind domain data (positions, theta, ...) before ``build_transfer``.

        Args:
            **inputs (torch.Tensor): Named tensors; only those in
                ``extra_input_names`` are kept.
        """
        self._bound = {k: v for k, v in inputs.items() if k in self.extra_input_names}

    def build_transfer(self, A: torch.Tensor) -> tuple[torch.Tensor, NeuralTransferOperator]:
        """Build neural transfer operator for this coarse level.

        Args:
            A (torch.Tensor): Fine-grid matrix ``A``, shape ``(n, n)``. Named
                to match the ``CoarseningStrategy`` protocol's parameter
                name exactly (structural typing checks parameter names for
                positional-or-keyword parameters).

        Returns:
            tuple[torch.Tensor, NeuralTransferOperator]: ``(A_coarse,
                transfer)`` - A_coarse uses Galerkin projection; transfer
                applies the neural P/R operators.

        Raises:
            NotImplementedError: Until the neural forward pass is wired to
                a specific checkpoint format and the coarse matrix computation
                is verified.
        """
        raise NotImplementedError(
            "NeuralCoarseningStrategy.build_transfer is not yet implemented. "
            "Wire the neural prolongator/restrictor to produce P, then compute "
            "A_coarse = P.T @ A @ P and return NeuralTransferOperator."
        )
