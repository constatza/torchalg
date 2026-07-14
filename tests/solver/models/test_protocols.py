"""Tests for ``torchalg.models.protocols``.

``HasVectors``/``HasDirectionHistory`` are ported protocols (structural
checks translated from the reference's numpy-typed versions).
``SolverProtocol`` has no reference equivalent (see ``docs/plan.md``'s
"Architectural corrections" section), so its tests are written fresh,
checking that both the test-support scipy oracle and a future
``IterativeSolverBase``-shaped object satisfy it structurally, and that an
unrelated object does not.
"""

from __future__ import annotations

from torchalg.models.protocols import HasDirectionHistory, HasVectors, SolverProtocol
from torchalg.models.state import CGState, KrylovState, SolverState


class TestHasVectors:
    """Tests for the ``HasVectors`` structural protocol."""

    def test_krylov_state_satisfies_protocol(self, krylov_state: KrylovState) -> None:
        """A ``KrylovState`` structurally satisfies ``HasVectors`` (u, r, w, d, q)."""
        assert isinstance(krylov_state, HasVectors)

    def test_cg_state_satisfies_protocol(self, cg_state: CGState) -> None:
        """``CGState`` inherits the working vectors, so it satisfies ``HasVectors`` too."""
        assert isinstance(cg_state, HasVectors)

    def test_base_solver_state_does_not_satisfy_protocol(
        self,
        solver_state: SolverState,
    ) -> None:
        """A bare ``SolverState`` has no working vectors and fails the structural check."""
        assert not isinstance(solver_state, HasVectors)


class TestHasDirectionHistory:
    """Tests for the ``HasDirectionHistory`` structural protocol."""

    def test_cg_state_satisfies_protocol(self, cg_state: CGState) -> None:
        """``CGState`` carries both histories and satisfies the protocol."""
        assert isinstance(cg_state, HasDirectionHistory)

    def test_krylov_state_does_not_satisfy_protocol(self, krylov_state: KrylovState) -> None:
        """``KrylovState`` has no history fields and fails the structural check."""
        assert not isinstance(krylov_state, HasDirectionHistory)


class TestSolverProtocol:
    """Tests for the new ``SolverProtocol`` structural contract."""

    def test_scipy_cg_solver_satisfies_protocol(self) -> None:
        """The test-support scipy oracle structurally satisfies ``SolverProtocol``.

        This is the concrete case the protocol exists for: it lets
        ``SciPyCGSolver`` be typed against the same interface as the future
        production solvers without either one inheriting from the other
        (see docs/plan.md's "Architectural corrections" section).
        """
        from tests.support.scipy_reference.scipy_wrapper import SciPyCGSolver

        solver = SciPyCGSolver()
        assert isinstance(solver, SolverProtocol)

    def test_object_without_solve_does_not_satisfy_protocol(self) -> None:
        """An unrelated object with no ``solve`` method fails the structural check."""
        assert not isinstance(object(), SolverProtocol)

    def test_unrelated_class_with_matching_method_name_satisfies_protocol(self) -> None:
        """A plain class with its own ``solve`` method satisfies the protocol.

        No inheritance from ``SolverProtocol`` (or anything else) is
        required - this is the whole point of structural typing (PEP 544)
        over an ABC: ``IterativeSolverBase`` subclasses (Stage 9) and
        ``SciPyCGSolver`` share no common base class today, and still both
        satisfy ``SolverProtocol``.
        """

        class MinimalSolver:
            def solve(self, A: object, b: object, x0: object = None, **kwargs: object) -> object:
                return None

        assert isinstance(MinimalSolver(), SolverProtocol)
