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

from typing import TYPE_CHECKING

from ._aggregation import (
    greedy_aggregation,
    piecewise_constant_prolongation,
    smoothed_prolongation,
    strength_of_connection,
)
from .transfer import DenseTransferOperator, NeuralTransferOperator

if TYPE_CHECKING:
    import torch

    from ...ports import ExtraInputPredictorPort


class AggregationCoarsening:
    """Smoothed aggregation AMG coarsening using dense tensors (SA-AMG).

    Implements the five-step coarsening algorithm from Vanek, Mandel & Brezina
    (1996):

    1. Strength-of-connection: mark strong off-diagonal entries via
       ``|a_ij| / sqrt(|a_ii| * |a_jj|) >= theta`` (Stuben 2001, Section 2.1).
    2. Greedy aggregation: partition nodes into non-overlapping aggregates
       using the strong-connection graph.
    3. Tentative prolongation P0: piecewise-constant indicator matrix
       (aggregate membership).
    4. Smoothed prolongation: ``P = (I - omega D^{-1} A) P0``, omega ~= 2/3
       for isotropic SPD (Vanek et al. 1996, Eq. 3.2).
    5. Galerkin coarse matrix: ``A_coarse = P.T @ A @ P``.

    Args:
        theta (float): Strength-of-connection threshold theta in (0, 1).
            Default 0.25 follows Stuben (2001) Section 2.1.
        omega (float): Jacobi-smoothing damping omega in (0, 1) for the
            prolongation smoother. Default 0.67 ~= 2/3 assumes
            ``rho(D^{-1}A) ~= 2`` (isotropic SPD).

    References:
        - Vanek, P., Mandel, J., & Brezina, M. (1996). Algebraic multigrid by
          smoothed aggregation for second and fourth order elliptic problems.
          Computing, 56(3), 179-196.
        - Stuben, K. (2001). A review of algebraic multigrid.
          J. Comput. Appl. Math., 128(1-2), 281-309.
    """

    def __init__(self, theta: float = 0.25, omega: float = 0.67) -> None:
        """Store the strength-of-connection threshold and smoothing damping factor.

        Args:
            theta (float): Strength-of-connection threshold theta in (0, 1).
            omega (float): Jacobi-smoothing damping omega in (0, 1).
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
        aggregate = greedy_aggregation(strength)
        tentative = piecewise_constant_prolongation(aggregate, dtype=A.dtype)
        prolongation = smoothed_prolongation(A, tentative, self._omega)

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
    verified directly against real stiffness matrices that
    `AggregationCoarsening`'s realized dimension is not monotonic in
    `theta` — greedy aggregation is a first-come-first-served, order-
    dependent heuristic (standard SA-AMG, not a bug: see Vanek, Mandel &
    Brezina 1996), so a stricter `theta` can still occasionally *reduce*
    the aggregate count by changing which node claims which neighbor
    first, even though the underlying strength-of-connection graph only
    ever shrinks as `theta` grows. Bisection assumes monotonicity and can
    converge to the wrong plateau as a result. A sampler built for
    expensive, high-dimensional, exploitably-smooth objectives (Optuna,
    Bayesian optimization) earns nothing here either: `theta` is a single
    bounded scalar, each trial is one cheap `build_transfer` call, and
    there is no smooth trend to model. A plain `step`-spaced exhaustive
    scan is simpler and strictly more reliable: correct up to `step`
    resolution, guaranteed, with no risk of settling on the wrong plateau.

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
    """

    def __init__(
        self,
        target_coarse_dim: int,
        *,
        theta_min: float,
        theta_max: float,
        step: float,
        omega: float,
    ) -> None:
        """Store the target dimension and search parameters; unbuilt until `build_transfer`.

        Args:
            target_coarse_dim (int): Desired realized coarse dimension.
            theta_min (float): Lower bound of the `theta` search grid.
            theta_max (float): Upper bound of the `theta` search grid.
            step (float): Grid spacing for the exhaustive search.
            omega (float): Prolongation Jacobi-smoothing damping.
        """
        self._target_coarse_dim = target_coarse_dim
        self._theta_min = theta_min
        self._theta_max = theta_max
        self._step = step
        self._omega = omega
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
        """Exhaustively scan the `theta` grid, keeping the closest match to `target_coarse_dim`.

        The sole extension point for swapping the search algorithm (e.g.
        for bisection, if a future coarsening strategy makes
        ``theta -> realized dimension`` monotonic) — everything else on
        this class (`build_transfer`'s caching, the constructor) is
        independent of how the search itself is performed.

        Args:
            A (torch.Tensor): Fine-grid matrix, shape ``(n, n)``.

        Returns:
            tuple[float, torch.Tensor, DenseTransferOperator]: ``(theta,
                A_coarse, transfer)`` for the candidate whose realized
                coarse dimension is closest to ``target_coarse_dim``.
        """
        n_steps = round((self._theta_max - self._theta_min) / self._step) + 1
        candidates = [self._theta_min + i * self._step for i in range(n_steps)]
        results = [
            (theta, *AggregationCoarsening(theta=theta, omega=self._omega).build_transfer(A))
            for theta in candidates
        ]
        return min(results, key=lambda result: abs(result[1].shape[0] - self._target_coarse_dim))


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
