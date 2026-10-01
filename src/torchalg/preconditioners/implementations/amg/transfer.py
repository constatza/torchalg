"""Transfer operators for multigrid prolongation and restriction.

Ported from ``neuralls.domain.solver.preconditioners.implementations.amg.transfer``
(see ``docs/plan.md``'s dense-only directive). ``SparseTransferOperator``
(scipy-sparse-backed in the reference) was renamed to ``DenseTransferOperator``
and now wraps a dense ``torch.Tensor`` prolongation matrix ``P`` instead of a
``scipy.sparse`` matrix - the algorithm (Galerkin restriction ``R = P.T``) is
unchanged, only the storage. ``DenseTransferOperator`` itself has since been
promoted to ``torchalg.utils.dense_transfer`` (it was already fully
format-agnostic - see that module's docstring) and is re-exported here so
every existing ``from ...amg.transfer import DenseTransferOperator`` call
site keeps working unchanged. The name also anticipated Stage 6 (POD), whose
reference README already calls its own dense-P/R wrapper
``DenseTransferOperator`` - and indeed the sparse POD sibling
(``torchalg.sparse.preconditioners.pod.coarsening``) reuses the very same
promoted class, since a POD basis is always dense.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchalg.utils.dense_transfer import DenseTransferOperator

if TYPE_CHECKING:
    import torch

    from ...ports import ExtraInputPredictorPort

__all__ = ["DenseTransferOperator", "NeuralTransferOperator"]


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
