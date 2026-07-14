"""Transfer operators for multigrid prolongation and restriction.

Ported from ``neuralls.domain.solver.preconditioners.implementations.amg.transfer``
(see ``docs/plan.md``'s dense-only directive). ``SparseTransferOperator``
(scipy-sparse-backed in the reference) is renamed to ``DenseTransferOperator``
and now wraps a dense ``torch.Tensor`` prolongation matrix ``P`` instead of a
``scipy.sparse`` matrix - the algorithm (Galerkin restriction ``R = P.T``) is
unchanged, only the storage. The name also anticipates Stage 6 (POD), whose
reference README already calls its own dense-P/R wrapper
``DenseTransferOperator``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from ...ports import ExtraInputPredictorPort


class DenseTransferOperator:
    """Transfer operator backed by a dense prolongation matrix P.

    Prolongation: ``P @ coarse`` (inject coarse correction to fine grid).
    Restriction: ``P.T @ fine`` (Galerkin restriction, R = P^T).

    Args:
        P (torch.Tensor): Prolongation matrix (n_fine x n_coarse), dense.
    """

    def __init__(self, P: torch.Tensor) -> None:
        """Store the prolongation matrix and its transpose (Galerkin restriction).

        Args:
            P (torch.Tensor): Prolongation matrix (n_fine x n_coarse), dense.
        """
        self._P = P
        self._R = P.T

    def prolongate(self, coarse: torch.Tensor) -> torch.Tensor:
        """Interpolate coarse-grid vector to fine grid.

        Args:
            coarse (torch.Tensor): Coarse-grid vector of length n_coarse.

        Returns:
            torch.Tensor: Fine-grid vector of length n_fine.
        """
        return self._P @ coarse

    def restrict(self, fine: torch.Tensor) -> torch.Tensor:
        """Restrict fine-grid vector to coarse grid (R = P^T).

        Args:
            fine (torch.Tensor): Fine-grid vector of length n_fine.

        Returns:
            torch.Tensor: Coarse-grid vector of length n_coarse.
        """
        return self._R @ fine


class NeuralTransferOperator:
    """Transfer operator implemented by two neural predictors.

    Prolongation and restriction are computed by separate neural networks.
    Extra inputs (positions, stiffness matrix) are pre-bound at construction
    time by ``NeuralCoarseningStrategy``; nothing outside that class sets them.

    Args:
        prolongator (ExtraInputPredictorPort): Neural predictor for
            coarse->fine mapping.
        restrictor (ExtraInputPredictorPort): Neural predictor for
            fine->coarse mapping.
        **bound_inputs (torch.Tensor): Pre-bound extra tensors forwarded on
            every apply call.
    """

    def __init__(
        self,
        prolongator: ExtraInputPredictorPort,
        restrictor: ExtraInputPredictorPort,
        **bound_inputs: torch.Tensor,
    ) -> None:
        """Store the neural predictors and their pre-bound extra inputs.

        Args:
            prolongator (ExtraInputPredictorPort): Neural predictor for
                coarse->fine mapping.
            restrictor (ExtraInputPredictorPort): Neural predictor for
                fine->coarse mapping.
            **bound_inputs (torch.Tensor): Pre-bound extra tensors forwarded
                on every apply call.
        """
        self._prolongator = prolongator
        self._restrictor = restrictor
        self._bound = bound_inputs

    def prolongate(self, coarse: torch.Tensor) -> torch.Tensor:
        """Apply neural prolongator to coarse-grid vector.

        Args:
            coarse (torch.Tensor): Coarse-grid vector.

        Returns:
            torch.Tensor: Fine-grid vector.
        """
        return self._prolongator.apply(coarse, **self._bound)

    def restrict(self, fine: torch.Tensor) -> torch.Tensor:
        """Apply neural restrictor to fine-grid vector.

        Args:
            fine (torch.Tensor): Fine-grid vector.

        Returns:
            torch.Tensor: Coarse-grid vector.
        """
        return self._restrictor.apply(fine, **self._bound)
