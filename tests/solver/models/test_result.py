"""Tests for ``torchalg.models.result``."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from torchalg.models.diagnostics import TerminationDiagnostics
from torchalg.models.result import IterationContext, SolverResult


class TestSolverResult:
    """Tests for ``SolverResult``."""

    def test_required_fields_are_set(self, minimal_solver_result: SolverResult) -> None:
        """Constructing with only required fields preserves their values."""
        assert minimal_solver_result.converged is True
        assert minimal_solver_result.iterations == 5
        assert minimal_solver_result.breakdown is False

    def test_diagnostics_defaults_to_empty_instance(
        self,
        minimal_solver_result: SolverResult,
    ) -> None:
        """``diagnostics`` defaults to a fresh, all-``None`` ``TerminationDiagnostics``.

        This is the replacement for the reference's ``event_log: EventLog | None
        = None`` default - a non-optional record whose fields are optional,
        instead of an optional record.
        """
        assert minimal_solver_result.diagnostics == TerminationDiagnostics()

    def test_each_result_gets_its_own_diagnostics_instance(self) -> None:
        """``field(default_factory=...)`` prevents a shared mutable default.

        Not that ``TerminationDiagnostics`` is mutable (it is frozen), but this
        guards against a future accidental switch to ``field(default=...)``
        silently reintroducing a shared-default bug.
        """
        first = SolverResult(
            converged=False,
            iterations=0,
            residual=1.0,
            residual_abs=1.0,
            rhs_norm=1.0,
            breakdown=False,
        )
        second = SolverResult(
            converged=False,
            iterations=0,
            residual=1.0,
            residual_abs=1.0,
            rhs_norm=1.0,
            breakdown=False,
        )
        assert first.diagnostics is not second.diagnostics

    def test_diagnostics_can_be_supplied_explicitly(self) -> None:
        """A caller-supplied ``TerminationDiagnostics`` is stored as-is."""
        diagnostics = TerminationDiagnostics(converged_at=3)
        result = SolverResult(
            converged=True,
            iterations=3,
            residual=1e-9,
            residual_abs=1e-9,
            rhs_norm=1.0,
            breakdown=False,
            diagnostics=diagnostics,
        )
        assert result.diagnostics is diagnostics

    def test_result_is_frozen(self, minimal_solver_result: SolverResult) -> None:
        """Mutating a field after construction raises (reference's ``SolverResult`` was mutable)."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            minimal_solver_result.iterations = 99  # ty: ignore[invalid-assignment]

    def test_residual_history_accepts_tuple(self) -> None:
        """Residual histories are typed as immutable tuples, not lists."""
        result = SolverResult(
            converged=True,
            iterations=2,
            residual=1e-8,
            residual_abs=1e-8,
            rhs_norm=1.0,
            breakdown=False,
            residual_history_rel=(1.0, 0.1, 1e-8),
            residual_history_abs=(1.0, 0.1, 1e-8),
        )
        assert result.residual_history_rel == (1.0, 0.1, 1e-8)


class TestIterationContext:
    """Tests for ``IterationContext``."""

    def test_fields_are_preserved(self, iteration_context: IterationContext) -> None:
        """Constructed fields are accessible without transformation."""
        assert iteration_context.iteration == 2
        expected = torch.ones(3, dtype=iteration_context.residual.dtype)
        assert torch.equal(iteration_context.residual, expected)

    def test_is_frozen(self, iteration_context: IterationContext) -> None:
        """Mutating a field after construction raises."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            iteration_context.iteration = 3  # ty: ignore[invalid-assignment]
