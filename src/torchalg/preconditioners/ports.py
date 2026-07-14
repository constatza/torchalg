"""Predictor ports - framework-agnostic abstractions for neural preconditioners.

Ported from ``neuralls.domain.solver.preconditioners.ports`` (see
``docs/plan.md``'s Stage 7 entry), following the Ports & Adapters (Hexagonal
Architecture) pattern. Unlike the reference, this module is natively
``torch.Tensor``-typed rather than numpy: the reference's ``NDArray`` in/out
signature exists partly to shield the solver from the *actual* PyTorch model
living behind ``neuralls.platform.dlkit.predictor_adapter``, but ``torchalg``
has no numpy boundary to hide behind in the first place - it is torch-native
throughout. Framework-agnosticism here means "no dependency on a specific
predictor-loading framework (DLKit, TorchScript, ...)", not "no torch".

Design Principles:
    - Framework-agnostic w.r.t. model-loading frameworks: no DLKit or other
      ML-platform imports, but ``torch.Tensor`` in/out (torchalg is
      torch-native, not framework-agnostic w.r.t. the tensor library itself).
    - Lifecycle management: context-manager support for resource cleanup.
    - SOLID: ABC enforces contracts at runtime (unlike Protocol).
    - OCP: base port is minimal; subtype ``ExtraInputPredictorPort`` extends it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class PredictorPort(ABC):
    """Abstract port for neural predictors with lifecycle management.

    This port defines the minimal interface for neural network-based
    preconditioners without coupling to a specific predictor-loading
    framework. Adapters implement this port for different frameworks.

    Implementors that do not need named extra inputs beyond the residual
    should inherit from this class directly. For models that require
    additional named tensors (e.g. stiffness matrix, coordinates), use
    :class:`ExtraInputPredictorPort` instead.

    Lifecycle:
        1. Create predictor (loads model).
        2. Apply predictions (multiple calls).
        3. Cleanup (free GPU memory).

    Example:
        >>> with adapter.create_predictor(checkpoint) as predictor:
        ...     result = predictor.apply(residual)
        ...     # Automatic cleanup on exit
    """

    @abstractmethod
    def apply(self, residual: torch.Tensor, **extra_inputs: torch.Tensor) -> torch.Tensor:
        """Apply neural network to residual vector.

        Args:
            residual (torch.Tensor): Residual vector, any dtype/device.
            **extra_inputs (torch.Tensor): Optional named tensors forwarded
                to the model (e.g. ``matrix=A``). Implementations that do
                not need extra inputs should accept and ignore them.

        Returns:
            torch.Tensor: Predicted correction.

        Raises:
            RuntimeError: If predictor not loaded or GPU OOM.
            ValueError: If residual shape incompatible.
        """
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Release resources (GPU memory, model weights). Idempotent."""
        ...

    def __enter__(self) -> PredictorPort:
        """Context manager entry.

        Returns:
            PredictorPort: This predictor instance.
        """
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Context manager exit - automatic cleanup.

        Args:
            exc_type (object): Exception type, if the ``with`` block raised.
            exc_val (object): Exception value, if the ``with`` block raised.
            exc_tb (object): Exception traceback, if the ``with`` block raised.
        """
        self.cleanup()


class ExtraInputPredictorPort(PredictorPort):
    """Port for predictors that actively use named extra inputs.

    Inherit from this (instead of ``PredictorPort`` directly) when the model
    needs more than the residual - e.g. stiffness matrix, node coordinates,
    PDE parameters.

    Implementors must declare ``required_inputs``, which names the extra
    tensors the model expects. This eliminates the need to re-declare input
    names anywhere else: the model config is the single source of truth for
    which arrays a neural predictor needs (see
    ``NeuralPreconditioner.extra_input_names`` for how a caller-supplied
    override falls back to this property).

    Example:
        >>> class MyPredictor(ExtraInputPredictorPort):
        ...     @property
        ...     def required_inputs(self) -> tuple[str, ...]:
        ...         return ("matrix", "positions")
        ...
        ...     def apply(
        ...         self, residual: torch.Tensor, **extra_inputs: torch.Tensor
        ...     ) -> torch.Tensor:
        ...         return self._model(residual, extra_inputs["matrix"])
        ...
        ...     def cleanup(self) -> None: ...
    """

    @property
    @abstractmethod
    def required_inputs(self) -> tuple[str, ...]:
        """Names of extra tensors this model expects beyond the residual.

        Derived from the model config; eliminates the need to re-declare
        this anywhere else - the model config is the single source of truth.

        Returns:
            tuple[str, ...]: Names expected in ``**extra_inputs`` on ``apply()``.
        """
        ...


class PredictorAdapter(ABC):
    """Abstract adapter for creating predictors from checkpoints.

    Adapters implement framework-specific loading and provide
    framework-agnostic predictors through the ``PredictorPort`` interface.

    Design:
        - Adapter pattern: converts framework-specific loading APIs to the
          port interface.
        - Factory method: ``create_predictor()`` returns an
          ``ExtraInputPredictorPort``.
        - Dependency inversion: callers depend on the port, not the adapter.
    """

    @abstractmethod
    def create_predictor(
        self,
        checkpoint_path: Path,
        config_path: Path | None = None,
        data_config_path: Path | None = None,
    ) -> ExtraInputPredictorPort:
        """Create predictor from checkpoint.

        Args:
            checkpoint_path (Path): Path to model checkpoint.
            config_path (Path | None): Optional model config (unused by
                some adapters).
            data_config_path (Path | None): Optional data config (unused by
                some adapters).

        Returns:
            ExtraInputPredictorPort: Predictor instance (use as a context
                manager for cleanup).

        Raises:
            FileNotFoundError: If checkpoint doesn't exist.
            RuntimeError: If model loading fails.
            ImportError: If the required framework is not installed.

        Example:
            >>> adapter = SomeFrameworkAdapter()
            >>> with adapter.create_predictor(Path("model.ckpt")) as pred:
            ...     result = pred.apply(residual)
        """
        ...
