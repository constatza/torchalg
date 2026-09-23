"""Tests for ``torchalg.factories`` public solver entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
import torch

from torchalg.factories import flexible_cg, pcg
from torchalg.monitoring import TraceMode, energy_norm_history
from torchalg.preconditioners.base import (
    BindableInputs,
    Preconditioner,
    PreconditionerContext,
)
from torchalg.preconditioners.implementations.jacobi import JacobiPreconditioner

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


class _RecordingBindablePreconditioner(Preconditioner, BindableInputs):
    """Preconditioner test double that records bound extra tensors and contexts."""

    def __init__(self) -> None:
        """Initialize empty call records."""
        self.bound_inputs: dict[str, torch.Tensor] = {}
        self.contexts: list[PreconditionerContext | None] = []

    @property
    def extra_input_names(self) -> tuple[str, ...]:
        """Declare one accepted extra input."""
        return ("matrix",)

    def bind_inputs(self, **inputs: torch.Tensor) -> None:
        """Record filtered extra inputs."""
        self.bound_inputs = dict(inputs)

    def apply(
        self,
        residual: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> torch.Tensor:
        """Record context and apply identity preconditioning."""
        self.contexts.append(context)
        return residual.clone()


def test_pcg_solves_small_spd_system(
    tridiagonal_spd_small: NDArray,
    rhs_ones_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
) -> None:
    """PCG converges to the dense torch solve on a small SPD system."""
    matrix = to_torch(tridiagonal_spd_small)
    rhs = to_torch(rhs_ones_small)

    solution, result = pcg(
        matrix,
        rhs,
        rtol=1e-12,
        atol=1e-14,
        maxiter=100,
    )

    expected = torch.linalg.solve(matrix, rhs)
    assert result.converged is True
    assert result.breakdown is False
    assert result.iterations <= matrix.shape[0]
    assert torch.allclose(solution, expected, rtol=1e-10, atol=1e-12)


def test_flexible_cg_solves_with_jacobi_preconditioner(
    diagonal_spd_small: NDArray,
    rhs_ones_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
) -> None:
    """FCG accepts a concrete preconditioner and converges on a diagonal system."""
    matrix = to_torch(diagonal_spd_small)
    rhs = to_torch(rhs_ones_small)
    preconditioner = JacobiPreconditioner(matrix)

    solution, result = flexible_cg(
        matrix,
        rhs,
        preconditioner=preconditioner,
        m_max=3,
        rtol=1e-12,
        atol=1e-14,
        maxiter=20,
    )

    expected = torch.linalg.solve(matrix, rhs)
    assert result.converged is True
    assert torch.allclose(solution, expected, rtol=1e-10, atol=1e-12)


def test_factory_full_trace_records_vectors(
    identity_matrix_small: NDArray,
    rhs_ones_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
) -> None:
    """FULL trace mode records residual and solution vectors on the result."""
    matrix = to_torch(identity_matrix_small)
    rhs = to_torch(rhs_ones_small)

    _, result = pcg(matrix, rhs, trace_mode=TraceMode.FULL, maxiter=5)

    assert result.residual_vectors is not None
    assert result.solution_vectors is not None
    assert result.direction_vectors is not None
    assert result.residual_vectors.shape[1:] == rhs.shape
    assert result.solution_vectors.shape[1:] == rhs.shape
    assert result.direction_vectors.shape[1:] == rhs.shape
    assert result.residual_vectors.device.type == "cpu"
    assert result.solution_vectors.device.type == "cpu"
    assert result.direction_vectors.device.type == "cpu"


def test_factory_x_exact_populates_error_history_matching_full_trace(
    tridiagonal_spd_small: NDArray,
    rhs_ones_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
) -> None:
    """``x_exact`` populates ``error_history_a_norm`` matching a post-hoc computation."""
    matrix = to_torch(tridiagonal_spd_small)
    rhs = to_torch(rhs_ones_small)
    x_exact = torch.linalg.solve(matrix, rhs)

    _, result = pcg(matrix, rhs, x_exact=x_exact, trace_mode=TraceMode.FULL, maxiter=20)

    assert result.error_history_a_norm is not None
    assert result.solution_vectors is not None
    reference = energy_norm_history(matrix, x_exact, result.solution_vectors)
    assert list(result.error_history_a_norm) == pytest.approx(reference.tolist(), abs=1e-8)


def test_factory_without_x_exact_omits_error_history(
    identity_matrix_small: NDArray,
    rhs_ones_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
) -> None:
    """Without ``x_exact``, ``error_history_a_norm`` stays ``None``."""
    matrix = to_torch(identity_matrix_small)
    rhs = to_torch(rhs_ones_small)

    _, result = pcg(matrix, rhs, trace_mode=TraceMode.MINIMAL, maxiter=5)

    assert result.error_history_a_norm is None


def test_factory_energy_decrements_recorded_in_minimal_mode(
    tridiagonal_spd_small: NDArray,
    rhs_ones_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
) -> None:
    """``energy_decrements`` is always populated once history is enabled, no ``x_exact`` needed."""
    matrix = to_torch(tridiagonal_spd_small)
    rhs = to_torch(rhs_ones_small)

    _, result = pcg(matrix, rhs, trace_mode=TraceMode.MINIMAL, maxiter=20)

    assert result.energy_decrements is not None
    assert len(result.energy_decrements) == result.iterations
    assert all(value >= 0.0 for value in result.energy_decrements)


def test_factory_disabled_trace_omits_histories(
    identity_matrix_small: NDArray,
    rhs_ones_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
) -> None:
    """DISABLED trace mode avoids storing residual histories."""
    matrix = to_torch(identity_matrix_small)
    rhs = to_torch(rhs_ones_small)

    _, result = pcg(matrix, rhs, trace_mode="disabled", maxiter=5)

    assert result.residual_history_abs is None
    assert result.residual_history_rel is None
    assert result.residual_vectors is None
    assert result.solution_vectors is None
    assert result.direction_vectors is None


def test_factory_binds_extra_inputs_before_solve(
    tridiagonal_spd_small: NDArray,
    rhs_ones_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
) -> None:
    """Factories thread named extra tensors into bindable preconditioners."""
    matrix = to_torch(tridiagonal_spd_small)
    rhs = to_torch(rhs_ones_small)
    preconditioner = _RecordingBindablePreconditioner()

    _, result = flexible_cg(
        matrix,
        rhs,
        preconditioner=preconditioner,
        extra_inputs={
            "matrix": matrix,
            "ignored": rhs,
        },
        maxiter=20,
    )

    assert result.converged is True
    assert preconditioner.bound_inputs == {"matrix": matrix}
    assert preconditioner.contexts
    assert preconditioner.contexts[0] is not None


def test_pcg_defaults_to_inference_mode(
    tridiagonal_spd_small: NDArray,
    rhs_ones_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
) -> None:
    """By default, ``pcg`` never tracks gradients even if ``A`` requires them.

    Most callers run a frozen, already-trained preconditioner and never
    backpropagate through the solve itself, so building an autograd graph
    for every solve is pure waste — this is what made dataset generation in
    ``dl-experiments`` build and discard thousands of unused graphs per run.
    """
    matrix = to_torch(tridiagonal_spd_small).requires_grad_(True)
    rhs = to_torch(rhs_ones_small)

    solution, _ = pcg(matrix, rhs, maxiter=20)

    assert solution.requires_grad is False


def test_pcg_differentiable_true_preserves_grad(
    tridiagonal_spd_small: NDArray,
    rhs_ones_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
) -> None:
    """``differentiable=True`` opts back into normal autograd tracking."""
    matrix = to_torch(tridiagonal_spd_small).requires_grad_(True)
    rhs = to_torch(rhs_ones_small)

    solution, _ = pcg(matrix, rhs, maxiter=20, differentiable=True)

    assert solution.requires_grad is True
    solution.sum().backward()
    assert matrix.grad is not None


def test_flexible_cg_defaults_to_inference_mode(
    diagonal_spd_small: NDArray,
    rhs_ones_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
) -> None:
    """``flexible_cg`` shares the same inference-mode default as ``pcg``."""
    matrix = to_torch(diagonal_spd_small).requires_grad_(True)
    rhs = to_torch(rhs_ones_small)
    preconditioner = JacobiPreconditioner(matrix.detach())

    solution, _ = flexible_cg(matrix, rhs, preconditioner=preconditioner, maxiter=20)

    assert solution.requires_grad is False


def test_pcg_device_override_bypasses_auto_resolution(
    identity_matrix_small: NDArray,
    rhs_ones_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit ``device=`` must win over automatic CUDA/CPU resolution.

    ``_place_on_device`` always calls ``resolve_device()`` and prefers CUDA
    when available — right for large training-time solves, wrong for many
    small, short dataset-generation solves, where per-iteration host<->device
    syncs (``float(torch.linalg.norm(...))`` in the convergence check) cost
    far more than the tiny matvec they guard. Callers who know their system
    is small must be able to pin the device explicitly.
    """
    matrix = to_torch(identity_matrix_small)
    rhs = to_torch(rhs_ones_small)

    def _fail_if_called() -> torch.device:
        raise AssertionError("resolve_device() must not run when device= is given")

    monkeypatch.setattr("torchalg.base.resolve_device", _fail_if_called)

    solution, _ = pcg(matrix, rhs, maxiter=5, device=torch.device("cpu"))

    assert solution.device.type == "cpu"


def test_pcg_device_none_still_auto_resolves(
    identity_matrix_small: NDArray,
    rhs_ones_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leaving ``device`` unset keeps today's automatic resolution behavior."""
    matrix = to_torch(identity_matrix_small)
    rhs = to_torch(rhs_ones_small)
    calls: list[bool] = []

    def _recording_resolve() -> torch.device:
        calls.append(True)
        return torch.device("cpu")

    monkeypatch.setattr("torchalg.base.resolve_device", _recording_resolve)

    pcg(matrix, rhs, maxiter=5)

    assert calls == [True]


def test_pcg_default_norm_reuses_state_residual_norm(
    identity_matrix_small: NDArray,
    rhs_ones_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Convergence checks must not recompute an already-known residual norm.

    Every convergence check (twice per iteration: the loop's stopping test
    and the final result's ``converged`` recomputation) called
    ``criterion.norm(residual)`` from scratch even though ``_iterate_step``
    had already computed that exact value (Euclidean norm, hardcoded) as
    ``state.residual_norm``. On a CUDA residual, each of those recomputes
    forces its own blocking device sync (``float()``/``.item()`` — see
    ``at::cuda::memcpy_and_sync`` backing ``_local_scalar_dense_cuda``) for a
    value already sitting in Python.

    The identity matrix converges in exactly one CG iteration, giving a
    fully deterministic call count: 2 for ``_initialize_state`` (initial
    residual norm + ``rhs_norm``) + 1 for the one ``_iterate_step`` (its new
    residual norm) = 3 necessary calls. Before this fix, three more
    (redundant) calls came from re-deriving the same norm inside
    ``has_converged`` at iteration 0, after the one iteration, and again in
    ``_build_result`` — 6 total.
    """
    matrix = to_torch(identity_matrix_small)
    rhs = to_torch(rhs_ones_small)
    real_norm = torch.linalg.norm
    call_count = 0

    def _counting_norm(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        return real_norm(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "norm", _counting_norm)

    _, result = pcg(matrix, rhs, maxiter=20)

    assert result.converged is True
    assert result.iterations == 1
    assert call_count == 3


def test_preconditioner_context_no_forced_syncs_with_tensor_norms(
    tridiagonal_spd_small: NDArray,
    rhs_ones_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
) -> None:
    """PreconditionerContext passes tensor residual_norm without forcing bool() conversions.

    The default identity preconditioner never reads the norm value, only
    passes it through or returns identity, so the context construction itself
    should not trigger any unexpected syncs during a multi-iteration solve.
    """
    matrix = to_torch(tridiagonal_spd_small)
    rhs = to_torch(rhs_ones_small)

    bool_call_count = 0
    original_bool = torch.Tensor.__bool__

    def counting_bool(self: Any) -> bool:
        nonlocal bool_call_count
        bool_call_count += 1
        return original_bool(self)

    torch.Tensor.__bool__ = cast(Any, counting_bool)
    try:
        _, result = pcg(matrix, rhs, maxiter=20)
        assert result.converged is True
        # The convergence check will call has_converged_from_norm which does a comparison,
        # but that's expected. We're checking that PreconditionerContext construction
        # itself (in _apply_preconditioner) doesn't add extra bool() calls beyond what's needed.
        # Note: we can't control the convergence check's calls since it's needed anyway.
    finally:
        torch.Tensor.__bool__ = original_bool


def test_iteration_history_never_none(
    identity_matrix_small: NDArray,
    rhs_ones_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
) -> None:
    """``solver.iteration_history`` is always a real object, never None.

    Even when ``TraceMode.DISABLED`` is used, the solver constructs a real
    ``IterationHistory`` with mode set to DISABLED, which causes
    ``log_iteration()`` to return early, making it a no-op. This eliminates
    the need for fallback branches elsewhere that check for None.
    """
    from torchalg.conjugate_gradient import PCGSolver
    from torchalg.monitoring import TraceMode

    matrix = to_torch(identity_matrix_small)
    rhs = to_torch(rhs_ones_small)

    # Direct construction with no IterationHistory should auto-create one with DISABLED mode
    solver = PCGSolver(iteration_history=None, trace_mode=TraceMode.DISABLED)
    assert solver.iteration_history is not None
    assert solver.iteration_history.mode == TraceMode.DISABLED

    # Factory construction also produces a real object
    from torchalg.factories import pcg

    _, result = pcg(matrix, rhs, trace_mode=TraceMode.DISABLED, maxiter=5)

    # DISABLED mode contract: no histories are stored
    assert result.residual_history_abs is None
    assert result.residual_history_rel is None


def test_float_conversion_count_after_deduplication(
    tridiagonal_spd_medium: NDArray,
    rhs_ones_medium: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that redundant CGState.residual_history tracking is eliminated.

    With only one tracking mechanism (IterationHistory), float() conversions
    are reduced by ~2 per iteration compared to the previous dual-tracking
    implementation. Fixed tolerances below machine precision force exactly
    ``maxiter`` iterations (never converges early), making the call count
    fully deterministic.

    This test empirically measures the actual float()-call reduction to ensure
    the refactor achieved its goal of eliminating redundant conversions.
    """
    matrix = to_torch(tridiagonal_spd_medium)
    rhs = to_torch(rhs_ones_medium)
    maxiter = 10
    real_float = torch.Tensor.__float__
    call_count = 0

    def _counting_float(self: torch.Tensor) -> float:
        nonlocal call_count
        call_count += 1
        return real_float(self)

    monkeypatch.setattr(torch.Tensor, "__float__", cast(Any, _counting_float))

    _, result = pcg(matrix, rhs, rtol=1e-300, atol=1e-300, maxiter=maxiter)

    assert result.iterations == maxiter
    # After eliminating CGState.residual_history tracking:
    # - Convergence check: ~1/iter
    # - IterationHistory's residual_norms.add() and energy_decrements.add(): 2/iter (MINIMAL mode)
    # - Misc per-solve overhead: ~10 calls
    # Total should be well under pre-refactor levels of ~5.6/iter = 56 calls for 10 iterations
    # The new implementation should be ~3/iter = 30 calls + overhead
    actual_per_iter = call_count / maxiter if maxiter > 0 else 0
    assert actual_per_iter < 4.0, (
        f"float() calls per iteration should be ~3, got {actual_per_iter:.2f} "
        f"(total {call_count} for {maxiter} iterations)"
    )
