"""Scheduled preconditioner for iteration-based switching."""

from __future__ import annotations

import torch

from ..base import BindableInputs, NonLinearPreconditioner, Preconditioner, PreconditionerContext


class ScheduledPreconditioner(NonLinearPreconditioner, BindableInputs):
    """Preconditioner that switches between primary and fallback by iteration.

    Uses the fallback before ``start_iter``, the primary while the schedule
    is active, then the fallback again after ``limit_iters`` if a limit is
    configured. Useful for expensive preconditioners that should be
    delayed, bounded, or both.

    Composes two other ``Preconditioner`` instances via constructor
    injection rather than holding its own tensor buffers, so it stays a
    plain class - no ``nn.Module`` needed here, even though the wrapped
    ``primary``/``fallback`` preconditioners may themselves be ``nn.Module``
    subclasses (e.g. ``JacobiPreconditioner``).

    Args:
        primary (Preconditioner): Main preconditioner to apply.
        fallback (Preconditioner | None): Fallback preconditioner (default:
            ``Identity``).
        limit_iters (int | None): Iterations to apply primary after
            ``start_iter`` (``None`` = unlimited).
        start_iter (int): Iteration at which primary becomes active.

    Example:
        >>> # Use Jacobi from iterations 5 through 14, then Identity
        >>> from .identity import Identity
        >>> from .jacobi import JacobiPreconditioner
        >>> scheduled = ScheduledPreconditioner(
        ...     primary=JacobiPreconditioner(matrix),
        ...     fallback=Identity(),
        ...     limit_iters=10,
        ...     start_iter=5,
        ... )
    """

    def __init__(
        self,
        primary: Preconditioner,
        fallback: Preconditioner | None = None,
        limit_iters: int | None = None,
        start_iter: int = 0,
    ) -> None:
        """Initialize scheduled preconditioner.

        Args:
            primary (Preconditioner): Main preconditioner to apply.
            fallback (Preconditioner | None): Fallback preconditioner
                (default: ``Identity``).
            limit_iters (int | None): Iterations to apply primary after
                ``start_iter``.
            start_iter (int): Iteration at which primary becomes active.

        Raises:
            ValueError: If ``start_iter`` or ``limit_iters`` is negative.
        """
        if start_iter < 0:
            raise ValueError("start_iter must be non-negative")
        if limit_iters is not None and limit_iters < 0:
            raise ValueError("limit_iters must be non-negative")
        self._primary = primary
        self._fallback = fallback
        self._limit_iters = limit_iters
        self._start_iter = start_iter

    @property
    def primary(self) -> Preconditioner:
        """The main preconditioner applied while the schedule is active.

        Returns:
            Preconditioner: The value passed at construction.
        """
        return self._primary

    @property
    def extra_input_names(self) -> tuple[str, ...]:
        """Aggregate extra input names from primary (and fallback if bindable).

        Returns:
            tuple[str, ...]: Union of ``extra_input_names`` from primary and
                fallback, de-duplicated.
        """
        names: list[str] = []
        if isinstance(self._primary, BindableInputs):
            names.extend(self._primary.extra_input_names)
        if isinstance(self._fallback, BindableInputs):
            names.extend(self._fallback.extra_input_names)
        return tuple(dict.fromkeys(names))

    def bind_inputs(self, **inputs: torch.Tensor) -> None:
        """Propagate extra inputs to primary and fallback preconditioners.

        Args:
            **inputs (torch.Tensor): Named tensors to forward.
        """
        if isinstance(self._primary, BindableInputs):
            self._primary.bind_inputs(**inputs)
        if isinstance(self._fallback, BindableInputs):
            self._fallback.bind_inputs(**inputs)

    def apply(
        self,
        residual: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> torch.Tensor:
        """Apply primary or fallback based on the schedule.

        Args:
            residual (torch.Tensor): Current residual vector r_k.
            context (PreconditionerContext | None): Iteration state
                (required for scheduling).

        Returns:
            torch.Tensor: Preconditioned residual z_k.

        Raises:
            ValueError: If ``context`` is ``None``.
        """
        if context is None:
            raise ValueError("ScheduledPreconditioner requires context for iteration tracking")

        if self._fallback is None:
            from .identity import Identity

            self._fallback = Identity()

        precond = self._primary if self._uses_primary(context.iteration) else self._fallback

        return precond.apply(residual, context)

    def _uses_primary(self, iteration: int) -> bool:
        """Return whether primary is active for the given zero-based iteration.

        Args:
            iteration (int): Zero-based solver iteration number.

        Returns:
            bool: ``True`` if the primary preconditioner is active at
                ``iteration``.
        """
        if iteration < self._start_iter:
            return False
        if self._limit_iters is None:
            return True
        return iteration < self._start_iter + self._limit_iters
