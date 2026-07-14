"""Tests for ``torchalg.models.diagnostics``.

No reference equivalent: ``TerminationDiagnostics`` is new code replacing
the reference's ``EventLog``/``SolverEvent``/``EventType`` (see
``docs/plan.md``'s "Architectural corrections" section), so these tests are
written fresh rather than ported.
"""

from __future__ import annotations

import dataclasses

import pytest

from torchalg.models.diagnostics import TerminationDiagnostics


class TestTerminationDiagnostics:
    """Tests for ``TerminationDiagnostics``."""

    def test_default_construction_is_all_none(self) -> None:
        """The "nothing happened yet" state has every field ``None``."""
        diagnostics = TerminationDiagnostics()
        assert diagnostics.converged_at is None
        assert diagnostics.breakdown_at is None
        assert diagnostics.breakdown_reason is None
        assert diagnostics.ortho_breakdown_at is None

    def test_converged_only(self) -> None:
        """A converged, non-broken-down solve records only ``converged_at``."""
        diagnostics = TerminationDiagnostics(converged_at=12)
        assert diagnostics.converged_at == 12
        assert diagnostics.breakdown_at is None
        assert diagnostics.breakdown_reason is None

    def test_breakdown_only(self) -> None:
        """Fields set together are all preserved independently."""
        diagnostics = TerminationDiagnostics(breakdown_at=4, breakdown_reason="nan_in_solution")
        assert diagnostics.breakdown_at == 4
        assert diagnostics.breakdown_reason == "nan_in_solution"
        assert diagnostics.converged_at is None

    def test_converged_at_matches_populated_fixture(
        self,
        populated_diagnostics: TerminationDiagnostics,
    ) -> None:
        """The shared ``populated_diagnostics`` fixture describes a clean convergence."""
        assert populated_diagnostics.converged_at == 7
        assert populated_diagnostics.breakdown_at is None

    def test_ortho_breakdown_is_independent_of_breakdown(self) -> None:
        """Orthogonalization breakdown is a distinct fact from solver breakdown."""
        diagnostics = TerminationDiagnostics(ortho_breakdown_at=3)
        assert diagnostics.ortho_breakdown_at == 3
        assert diagnostics.breakdown_at is None

    def test_equality_is_value_based(self) -> None:
        """Two instances with identical field values compare equal (plain dataclass eq)."""
        assert TerminationDiagnostics(converged_at=5) == TerminationDiagnostics(converged_at=5)
        assert TerminationDiagnostics(converged_at=5) != TerminationDiagnostics(converged_at=6)

    def test_is_frozen(self) -> None:
        """Mutating a field after construction raises."""
        diagnostics = TerminationDiagnostics()
        with pytest.raises(dataclasses.FrozenInstanceError):
            diagnostics.converged_at = 1  # ty: ignore[invalid-assignment]
