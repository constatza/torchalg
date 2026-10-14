"""Neural network preconditioner."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..base import NonLinearPreconditioner, PreconditionerContext, StaticSetupPreconditioner
from ..ports import ExtraInputPredictorPort

if TYPE_CHECKING:
    from typing import Self

    import torch

logger = logging.getLogger(__name__)


class NeuralPreconditioner(NonLinearPreconditioner, StaticSetupPreconditioner):
    """Neural network preconditioner built on a caller-supplied predictor.

    Applies the predictor to residuals in ``apply()``. Extra named inputs
    (matrix, coordinates, parameters) declared in ``extra_input_names`` are
    pre-bound via ``bind_inputs()`` before the CG loop and forwarded to the
    predictor on each ``apply()`` call.

    The predictor is injected, already loaded: this class knows nothing about
    checkpoints, files or frameworks, so how a model is loaded, where it lives
    and when it is released are the caller's concern. It follows that the
    caller *owns* the predictor and releases it (``predictor.cleanup()`` or its
    context manager); this class never does, which is also what makes one
    loaded predictor safe to share between several preconditioners.

    This class owns no persistent tensor state of its own - it delegates
    inference to the injected predictor and only caches the currently-bound
    extra-input tensors (replaced wholesale on every ``bind_inputs()`` call,
    not accumulated). It therefore stays a plain class, matching
    ``ScheduledPreconditioner``/``CallablePreconditioner`` (wrapper/delegator
    preconditioners that hold references to other objects, not their own
    tensor buffers, do not need ``nn.Module`` - see ``docs/plan.md``'s
    ``nn.Module`` architecture decision).

    Args:
        predictor (ExtraInputPredictorPort): Loaded predictor. Not released
            by this class.
        extra_input_names (tuple[str, ...]): Names of extra inputs the model
            expects beyond the residual (e.g. ``("matrix",)``). Matched
            against ``bind_inputs()`` keys. When empty, the predictor's own
            ``required_inputs`` apply.

    Example:
        >>> with adapter.create_predictor(Path("model.ckpt")) as predictor:
        ...     precond = NeuralPreconditioner(predictor, extra_input_names=("matrix",))
        ...     precond.bind_inputs(matrix=A)  # done once before CG loop
        ...     z = precond.apply(residual)  # called each CG iteration
    """

    def __init__(
        self,
        predictor: ExtraInputPredictorPort,
        extra_input_names: tuple[str, ...] = (),
    ) -> None:
        """Store the predictor; nothing is left to load.

        Args:
            predictor (ExtraInputPredictorPort): Loaded predictor, owned and
                released by the caller.
            extra_input_names (tuple[str, ...]): Names of extra inputs the
                model expects beyond the residual (e.g. ``("matrix",)``).
                Matched against ``bind_inputs()`` keys.
        """
        self._predictor = predictor
        self._extra_input_names: tuple[str, ...] = extra_input_names
        self._extra_inputs: dict[str, torch.Tensor] = {}
        self._mark_ready()

    def setup(self, context: PreconditionerContext | None = None) -> Self:
        """No-op: the predictor is already usable after construction.

        Args:
            context (PreconditionerContext | None): Ignored.

        Returns:
            Self: This preconditioner, already ready.
        """
        self._mark_ready()
        return self

    @property
    def extra_input_names(self) -> tuple[str, ...]:
        """Names of extra inputs this preconditioner expects beyond the residual.

        When the constructor argument is empty, falls back to the
        predictor's own declaration so callers do not need to duplicate
        what the model config states.

        Returns:
            tuple[str, ...]: Declared extra input names.
        """
        if self._extra_input_names:
            return self._extra_input_names
        return self._predictor.required_inputs

    def bind_inputs(self, **inputs: torch.Tensor) -> None:
        """Store extra named inputs for forwarding on each ``apply()`` call.

        Only keys declared in ``extra_input_names`` are retained; others are
        ignored with a warning so a misconfigured ``extra_input_names`` is
        detectable.

        Args:
            **inputs (torch.Tensor): Named tensors matching
                ``extra_input_names`` entries.
        """
        declared = self.extra_input_names
        unexpected = set(inputs) - set(declared)
        if unexpected:
            logger.warning(
                "bind_inputs: keys %s not in extra_input_names %s - ignored",
                unexpected,
                declared,
            )
        self._extra_inputs = {k: v for k, v in inputs.items() if k in declared}

    def apply(
        self,
        residual: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> torch.Tensor:
        """Apply neural network to residual.

        Args:
            residual (torch.Tensor): Current residual vector r_k.
            context (PreconditionerContext | None): Ignored (neural
                preconditioner doesn't use context).

        Returns:
            torch.Tensor: Preconditioned residual ``z_k = network(r_k, **extra_inputs)``.
        """
        return self._predictor.apply(residual, **self._extra_inputs)
