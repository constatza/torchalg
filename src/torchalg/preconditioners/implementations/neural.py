"""Neural network preconditioner."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..base import NonLinearPreconditioner, PreconditionerContext
from ..ports import ExtraInputPredictorPort, PredictorAdapter

if TYPE_CHECKING:
    from pathlib import Path

    import torch

logger = logging.getLogger(__name__)


class NeuralPreconditioner(NonLinearPreconditioner):
    """Neural network preconditioner with automatic cleanup.

    Loads a trained model from checkpoint and applies it to residuals.
    Extra named inputs (matrix, coordinates, parameters) declared in
    ``extra_input_names`` are pre-bound via ``bind_inputs()`` before the CG
    loop and forwarded to the predictor on each ``apply()`` call.

    GPU resources are automatically freed when the preconditioner is garbage
    collected.

    This class owns no persistent tensor state of its own - it delegates
    inference to the injected ``adapter``/predictor and only caches the
    currently-bound extra-input tensors (replaced wholesale on every
    ``bind_inputs()`` call, not accumulated). It therefore stays a plain
    class, matching ``ScheduledPreconditioner``/``CallablePreconditioner``
    (wrapper/delegator preconditioners that hold references to other
    objects, not their own tensor buffers, do not need ``nn.Module`` - see
    ``docs/plan.md``'s ``nn.Module`` architecture decision). The one real
    ``nn.Module`` in this picture is the caller-supplied predictor model
    living behind ``adapter``/``PredictorPort``, which this class touches
    only through that plain ABC boundary, never subclassing or wrapping it.

    Args:
        checkpoint_path (Path): Path to trained model checkpoint.
        config_path (Path | None): Optional model configuration.
        data_config_path (Path | None): Optional data configuration.
        adapter (PredictorAdapter | None): Predictor adapter supplied by the
            caller. Must not be ``None`` at runtime.
        extra_input_names (tuple[str, ...]): Names of extra inputs the model
            expects beyond the residual (e.g. ``("matrix",)``). Matched
            against ``bind_inputs()`` keys.

    Example:
        >>> precond = NeuralPreconditioner(
        ...     Path("model.ckpt"), adapter=adapter, extra_input_names=("matrix",)
        ... )
        >>> precond.bind_inputs(matrix=A)  # done once before CG loop
        >>> z = precond.apply(residual)  # called each CG iteration
    """

    def __init__(
        self,
        checkpoint_path: Path,
        config_path: Path | None = None,
        data_config_path: Path | None = None,
        adapter: PredictorAdapter | None = None,
        extra_input_names: tuple[str, ...] = (),
    ) -> None:
        """Initialize neural preconditioner from checkpoint.

        Args:
            checkpoint_path (Path): Path to trained model checkpoint.
            config_path (Path | None): Optional model configuration.
            data_config_path (Path | None): Optional data configuration.
            adapter (PredictorAdapter | None): Predictor adapter supplied by
                the caller. Must not be ``None`` at runtime.
            extra_input_names (tuple[str, ...]): Names of extra inputs the
                model expects beyond the residual (e.g. ``("matrix",)``).
                Matched against ``bind_inputs()`` keys.

        Raises:
            ValueError: If ``adapter`` is ``None``.
        """
        if adapter is None:
            raise ValueError("NeuralPreconditioner requires an explicit predictor adapter.")

        self._extra_input_names: tuple[str, ...] = extra_input_names
        self._extra_inputs: dict[str, torch.Tensor] = {}

        # Load predictor (GPU model); adapter contract guarantees ExtraInputPredictorPort.
        self._predictor: ExtraInputPredictorPort = adapter.create_predictor(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            data_config_path=data_config_path,
        )

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

    def cleanup(self) -> None:
        """Free GPU resources manually.

        Called automatically by ``__del__``, but can be called explicitly
        if you want to free resources before garbage collection.
        """
        # Check if _predictor exists (might not if __init__ failed).
        if hasattr(self, "_predictor") and hasattr(self._predictor, "cleanup"):
            self._predictor.cleanup()

    def __del__(self) -> None:
        """Automatic cleanup on garbage collection.

        Ensures GPU resources are freed when the preconditioner is no
        longer referenced.
        """
        self.cleanup()
