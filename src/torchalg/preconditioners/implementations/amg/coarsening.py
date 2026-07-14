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
