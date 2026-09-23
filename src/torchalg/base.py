"""Template-method base for torch-native iterative solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import cast

import torch

from torchalg.models.protocols import HasEnergyDecrement, HasVectors
from torchalg.models.result import SolverResult
from torchalg.models.state import SolverState
from torchalg.monitoring import IterationHistory, TraceMode
from torchalg.preconditioners.base import Preconditioner, PreconditionerContext
from torchalg.strategies.convergence import IConvergenceCriterion
from torchalg.strategies.norms import euclidean_norm
from torchalg.utils.device import resolve_device
from torchalg.utils.grad_mode import inference_unless_differentiable
from torchalg.utils.validation import validate_matrix, validate_rhs_vector

DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-14

type LinearSystemOperator = Callable[[torch.Tensor], torch.Tensor] | torch.Tensor


class IterativeSolverBase[S: SolverState](ABC):
    """Abstract base class for iterative solvers using template method."""

    def __init__(
        self,
        *,
        iteration_history: IterationHistory | None = None,
        trace_mode: TraceMode = TraceMode.MINIMAL,
    ) -> None:
        """Initialize common solver monitoring configuration."""
        self.iteration_history = iteration_history
        self.trace_mode = trace_mode

    def solve(
        self,
        A: LinearSystemOperator,
        b: torch.Tensor,
        x0: torch.Tensor | None = None,
        *,
        rtol: float | None = None,
        atol: float | None = None,
        maxiter: int | None = None,
        differentiable: bool = False,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, SolverResult]:
        """Solve ``A x = b`` using the concrete iterative method.

        ``A``/``b``/``x0`` (and any ``nn.Module``-based preconditioner) are
        moved to CUDA automatically when available (see
        :func:`torchalg.utils.device.resolve_device`); the returned solution
        and ``SolverResult`` stay on that same device rather than being
        forced back to CPU — callers needing NumPy interop already convert
        explicitly (``.detach().cpu().numpy()``), and forcing a transfer
        here would be a wasted round-trip for callers chaining further GPU
        work.

        Args:
            A: System matrix, or a callable computing ``A @ v``.
            b: Right-hand side, shape ``(n,)``.
            x0: Initial guess; defaults to the zero vector.
            rtol: Relative residual tolerance; defaults to
                :data:`DEFAULT_RTOL`.
            atol: Absolute residual tolerance; defaults to
                :data:`DEFAULT_ATOL`.
            maxiter: Maximum iterations; defaults to ``10 * n``.
            differentiable: When ``False`` (default), the whole solve runs
                under ``torch.inference_mode()`` — most callers use a
                frozen preconditioner and never need gradients through the
                solve, so tracking autograd by default only builds graphs
                nobody uses. Set ``True`` to keep normal autograd tracking,
                e.g. to backpropagate through an unrolled solve.
            device: Overrides automatic CUDA/CPU resolution when given. The
                convergence check pulls a Python float off the residual
                every iteration, which forces a blocking device sync — for
                many small, short solves (e.g. per-sample dataset
                generation) that sync cost dwarfs the matvec it guards, so
                auto-placing onto CUDA there is actively slower than CPU.
                Callers who know their system is small should pin
                ``torch.device("cpu")`` explicitly instead of relying on
                automatic resolution.
        """
        with inference_unless_differentiable(differentiable):
            self._validate_system(A, b, x0)
            A, b, x0 = self._place_on_device(A, b, x0, device=device)

            rtol_eff = DEFAULT_RTOL if rtol is None else rtol
            atol_eff = DEFAULT_ATOL if atol is None else atol
            maxiter_eff = 10 * b.numel() if maxiter is None else maxiter
            linear_op = self._prepare_operator(A)

            state = self._initialize_state(linear_op, b, x0, maxiter=maxiter_eff)
            self._log_state(state)

            while not self._check_stopping(state, rtol_eff, atol_eff, maxiter_eff):
                state = self._iterate_step(linear_op, state)
                self._log_state(state)

            if not isinstance(state, HasVectors):
                # Internal invariant check, not caller input validation - the loop
                # above must always leave state satisfying HasVectors; RuntimeError
                # is correct here, not TypeError (TRY004 doesn't distinguish the two).
                raise RuntimeError("Final solution not found in state")  # noqa: TRY004

            return state.u, self._build_result(state, rtol_eff, atol_eff, maxiter=maxiter_eff)

    def check_convergence(
        self,
        residual: torch.Tensor,
        rhs_norm: float,
        criterion: IConvergenceCriterion,
    ) -> bool:
        """Delegate convergence checking to the injected criterion."""
        return criterion.has_converged(residual, rhs_norm)

    def _check_convergence_with_criterion(
        self,
        state: S,
        criterion: IConvergenceCriterion,
    ) -> bool:
        """Check convergence for states that expose residual vectors.

        ``state.residual_norm`` is always the Euclidean norm of ``state.r``
        (``_iterate_step`` computes it that way unconditionally). When the
        criterion's own norm is that same default, its convergence check is
        answered directly from the already-known value instead of calling
        ``criterion.has_converged(state.r, ...)``, which would recompute
        ``||state.r||`` from the tensor — a second, redundant pass that on a
        CUDA residual is a second blocking device sync for a number this
        method already has. A criterion using any other norm (e.g. the
        A-norm) still takes the tensor path, since only the Euclidean case
        matches what ``state.residual_norm`` holds.
        """
        if not isinstance(state, HasVectors):
            return False
        # Convert tensor-valued norms to float if needed
        residual_norm = state.residual_norm
        if isinstance(residual_norm, torch.Tensor):
            residual_norm = float(residual_norm)
        rhs_norm = state.rhs_norm
        if isinstance(rhs_norm, torch.Tensor):
            rhs_norm = float(rhs_norm)
        if criterion.norm is euclidean_norm:
            return criterion.has_converged_from_norm(residual_norm, rhs_norm)
        return self.check_convergence(state.r, rhs_norm, criterion)

    def _validate_system(
        self,
        A: LinearSystemOperator,
        b: torch.Tensor,
        x0: torch.Tensor | None,
    ) -> None:
        """Validate system dimensions and finiteness."""
        if b.ndim != 1:
            raise ValueError(f"b must be 1D vector, got shape {tuple(b.shape)}")
        validate_rhs_vector(b)

        if isinstance(A, torch.Tensor):
            validate_matrix(A)
            if A.shape[0] != b.shape[0]:
                raise ValueError(f"A shape {tuple(A.shape)} incompatible with b {tuple(b.shape)}")

        if x0 is not None:
            validate_rhs_vector(x0)
            if x0.shape != b.shape:
                raise ValueError(f"x0 shape {tuple(x0.shape)} != b shape {tuple(b.shape)}")

    def _place_on_device(
        self,
        A: LinearSystemOperator,
        b: torch.Tensor,
        x0: torch.Tensor | None,
        *,
        device: torch.device | None = None,
    ) -> tuple[LinearSystemOperator, torch.Tensor, torch.Tensor | None]:
        """Move the system and preconditioner onto the given or autodetected device.

        Mirrors Lightning-style automatic GPU placement by default: callers
        who pass no ``device`` get CUDA whenever it's available. Passing an
        explicit ``device`` opts out of that resolution entirely instead of
        merely overriding its result, since a caller who already knows a
        small solve is faster kept on CPU should not pay for a CUDA check it
        doesn't need. ``A``/``b``/``x0`` are moved first since a failed
        ``.to()`` on a local variable leaves nothing mutated;
        ``self.preconditioner`` is moved last since ``nn.Module.to()``
        mutates its buffers in place and could otherwise be left half-moved
        by a failure moving the (usually larger) system tensors. A matvec
        callable for ``A`` is left untouched — its device is the caller's
        responsibility, since torchalg cannot reach inside an opaque
        closure.

        Args:
            A (LinearSystemOperator): System matrix or matvec callable.
            b (torch.Tensor): RHS vector.
            x0 (torch.Tensor | None): Optional initial guess.
            device (torch.device | None): Explicit target device; ``None``
                auto-resolves via :func:`torchalg.utils.device.resolve_device`.

        Returns:
            tuple[LinearSystemOperator, torch.Tensor, torch.Tensor | None]:
                ``A``, ``b``, ``x0`` moved onto the resolved device.
        """
        device = device if device is not None else resolve_device()
        if isinstance(A, torch.Tensor):
            A = A.to(device)
        b = b.to(device)
        if x0 is not None:
            x0 = x0.to(device)
        preconditioner = getattr(self, "preconditioner", None)
        if isinstance(preconditioner, torch.nn.Module):
            preconditioner.to(device)
        return A, b, x0

    def _prepare_operator(self, A: LinearSystemOperator) -> Callable[[torch.Tensor], torch.Tensor]:
        """Wrap dense tensors and callables behind a matvec callable."""
        if callable(A):
            return cast(Callable[[torch.Tensor], torch.Tensor], A)

        matrix = A

        def _matvec(vector: torch.Tensor) -> torch.Tensor:
            return matrix @ vector

        return _matvec

    @abstractmethod
    def _initialize_state(
        self,
        linear_op: Callable[[torch.Tensor], torch.Tensor],
        b: torch.Tensor,
        x0: torch.Tensor | None,
        maxiter: int | None = None,
    ) -> S:
        """Initialize solver state before the iteration loop."""

    @abstractmethod
    def _iterate_step(
        self,
        linear_op: Callable[[torch.Tensor], torch.Tensor],
        state: S,
    ) -> S:
        """Execute one solver iteration."""

    @abstractmethod
    def _check_stopping(
        self,
        state: S,
        rtol: float,
        atol: float,
        maxiter: int,
    ) -> bool:
        """Return whether the solver should stop."""

    @abstractmethod
    def _build_result(
        self,
        state: S,
        rtol: float,
        atol: float,
        maxiter: int | None = None,
    ) -> SolverResult:
        """Build the final solver result."""

    def _log_state(self, state: S) -> None:
        """Log continuous iteration telemetry if enabled."""
        if self.iteration_history is None:
            return

        residual = None
        solution = None
        direction = None
        needs_error_norm = self.iteration_history.x_exact is not None
        if isinstance(state, HasVectors) and (
            self.trace_mode == TraceMode.FULL or needs_error_norm
        ):
            residual = state.r
            solution = state.u
        if self.trace_mode == TraceMode.FULL and isinstance(state, HasVectors):
            direction = state.d

        energy_decrement = None
        if isinstance(state, HasEnergyDecrement):
            energy_decrement = state.energy_decrement

        # residual_norm/energy_decrement may be 0-d tensors here; passed through
        # unconverted since ScalarHistory.add() already casts to float internally
        # (monitoring/storage.py) — converting here too would be a second,
        # redundant device sync for the same value.
        self.iteration_history.log_iteration(
            residual_norm=state.residual_norm,
            residual=residual,
            solution=solution,
            direction=direction,
            energy_decrement=energy_decrement,
        )

    def _apply_preconditioner(
        self,
        preconditioner: Preconditioner,
        residual: torch.Tensor,
        state: SolverState,
    ) -> torch.Tensor:
        """Apply preconditioner with a uniform context object."""
        context = PreconditionerContext(
            iteration=state.iteration,
            residual_norm=state.residual_norm,
            rhs_norm=state.rhs_norm,
        )
        result = preconditioner.apply(residual, context)
        if result.shape != residual.shape:
            raise ValueError(
                f"Preconditioner output shape {tuple(result.shape)} != input "
                f"{tuple(residual.shape)}"
            )
        return result.to(dtype=residual.dtype, device=residual.device)

    def _extract_histories_from_iteration_history(
        self,
        rhs_norm: float,
    ) -> tuple[
        tuple[float, ...] | None,
        tuple[float, ...] | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        tuple[float, ...] | None,
        tuple[float, ...] | None,
    ]:
        """Extract result histories from the configured iteration history."""
        empty_result = (None, None, None, None, None, None, None)
        if self.iteration_history is None:
            return empty_result

        residual_history_abs = tuple(self.iteration_history.residual_norms.to_list())
        if not residual_history_abs:
            return empty_result

        if rhs_norm > 0:
            residual_history_rel = tuple(value / rhs_norm for value in residual_history_abs)
        else:
            residual_history_rel = residual_history_abs

        residual_vectors = None
        if self.iteration_history.residuals is not None:
            residual_vectors = self.iteration_history.residuals.to_tensor()

        solution_vectors = None
        if self.iteration_history.solutions is not None:
            solution_vectors = self.iteration_history.solutions.to_tensor()

        direction_vectors = None
        if self.iteration_history.directions is not None:
            direction_vectors = self.iteration_history.directions.to_tensor()

        error_history_a_norm = tuple(self.iteration_history.error_norms.to_list()) or None
        energy_decrements = tuple(self.iteration_history.energy_decrements.to_list()) or None

        return (
            residual_history_abs,
            residual_history_rel,
            residual_vectors,
            solution_vectors,
            direction_vectors,
            error_history_a_norm,
            energy_decrements,
        )
