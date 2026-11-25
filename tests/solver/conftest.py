"""Shared fixtures for solver tests.

Ported from the reference (``dl-experiments``'s ``tests/solver/conftest.py``)
taxonomy of composable, modular fixtures for testing CG-family solvers. All
test data is created via fixtures — never inline in test functions — per
project convention.

Base matrix/RHS/system fixtures stay plain numpy so they are bit-identical
inputs to both the scipy oracle (``tests/support/scipy_reference``) and the
torch-native solver code later stages will port. Fixtures with a ``_torch``
suffix are thin ``to_torch``-adapted twins for torch-side tests.

Deviations from the reference:
    - ``solver_plots_dir``/``save_convergence_plot`` are not ported: they
      depend on matplotlib, which is not a torchalg dependency, and are
      unrelated to correctness verification.
    - Deprecated/legacy tolerance aliases the reference kept only for its
      own internal migration (``default_tolerances``, ``tight_tolerances``,
      ``default_assert_rtol``) are not carried forward; the
      ``integration_*``/``convergence_*`` tier fixtures are the ones actually
      used going forward.
    - ``solver_factories["fcg"]``/``["pcg"]`` resolve ``torchalg``
      lazily, at call time, exactly like the reference resolves
      ``neuralls.domain.solver`` lazily inside the fixture body. Those
      solvers do not exist until Stage 9 of the migration; requesting this
      fixture is safe today, but calling the ``"fcg"``/``"pcg"`` entries
      before Stage 9 raises ``ImportError`` by design (this repo's "tests
      before src" rule means no Stage-0 test exercises them yet).
      ``solver_factories["scipy_cg"]`` is fully functional today: it wraps
      the ``tests/support/scipy_reference`` oracle with the same call
      signature the future ``torchalg`` factories will have.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch
from scipy.sparse import diags

from tests.support.scipy_reference.preconditioners import (
    CallablePreconditioner,
    Identity,
    Preconditioner,
)
from tests.support.scipy_reference.scipy_wrapper import SciPyCGSolver

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

# =============================================================================
# Constants for Test Configuration
# =============================================================================
SMALL_MATRIX_SIZE = 10
MEDIUM_MATRIX_SIZE = 50
TRIDIAGONAL_DIAG_VALUE = 4.0
TRIDIAGONAL_OFFDIAG_VALUE = -1.0

# Tolerance constants for two test levels
INTEGRATION_RTOL = 1e-6  # Integration test relative tolerance
INTEGRATION_ATOL = 1e-14  # Integration test absolute tolerance
CONVERGENCE_RTOL = 1e-12  # Convergence test relative tolerance
CONVERGENCE_ATOL = 1e-14  # Convergence test absolute tolerance

DEFAULT_MAX_ITER = 1000


# =============================================================================
# SPD Matrix Fixtures
# =============================================================================


@pytest.fixture
def identity_matrix_small() -> NDArray:
    """Small identity matrix (10x10) - trivial SPD system.

    Returns:
        10x10 identity matrix as dense array.

    Theory:
        Identity matrix has eigenvalues all equal to 1, condition number 1.
        CG converges in exactly 1 iteration for any preconditioner.
    """
    return np.eye(SMALL_MATRIX_SIZE, dtype=np.float64)


@pytest.fixture
def tridiagonal_spd_small() -> NDArray:
    """Small tridiagonal SPD matrix (10x10) - well-conditioned.

    Returns:
        10x10 tridiagonal matrix: diag=4, off-diag=-1.

    Theory:
        Tridiagonal matrix with diag=4, off-diag=-1 is SPD.
        Eigenvalues: lambda_k = 4 - 2*cos(k*pi/(n+1)) for k=1..n.
        Condition number kappa ~= 4n^2/pi^2 for large n.
        For n=10: kappa ~= 40, well-conditioned.
    """
    n = SMALL_MATRIX_SIZE
    sparse_mat = diags(
        [TRIDIAGONAL_OFFDIAG_VALUE, TRIDIAGONAL_DIAG_VALUE, TRIDIAGONAL_OFFDIAG_VALUE],
        [-1, 0, 1],
        shape=(n, n),
        format="csr",
        dtype=np.float64,
    )
    return sparse_mat.toarray()


@pytest.fixture
def diagonal_spd_small() -> NDArray:
    """Small diagonal SPD matrix (10x10) with varying diagonal entries.

    Returns:
        10x10 diagonal matrix with entries [1, 2, ..., 10].

    Theory:
        Diagonal matrix is trivially SPD if all diagonal entries > 0.
        Eigenvalues are the diagonal entries.
        Condition number kappa = 10/1 = 10 (well-conditioned).
    """
    n = SMALL_MATRIX_SIZE
    return np.diag(np.arange(1, n + 1, dtype=np.float64))


@pytest.fixture
def random_spd_small(test_seed: int) -> NDArray:
    """Small random SPD matrix (10x10) with fixed seed.

    Args:
        test_seed: Random seed for reproducibility.

    Returns:
        10x10 random SPD matrix.

    Theory:
        Construct SPD matrix as A = Q @ D @ Q.T where:
        - Q is random orthogonal matrix (from QR decomposition)
        - D is diagonal with positive eigenvalues
        This guarantees SPD with controlled condition number.
    """
    n = SMALL_MATRIX_SIZE
    rng = np.random.default_rng(test_seed)

    random_mat = rng.standard_normal((n, n))
    q, _ = np.linalg.qr(random_mat)

    eigenvalues = rng.uniform(1.0, 10.0, n)
    d = np.diag(eigenvalues)

    return q @ d @ q.T


@pytest.fixture
def ill_conditioned_spd_small(test_seed: int) -> NDArray:
    """Small ill-conditioned SPD matrix (10x10) - challenging system.

    Args:
        test_seed: Random seed for reproducibility.

    Returns:
        10x10 SPD matrix with condition number ~1000.

    Theory:
        Ill-conditioned systems have large condition number kappa = lambda_max
        / lambda_min. CG convergence rate depends on sqrt(kappa). For kappa =
        1000: expect ~20 iterations to reach rtol=1e-6.
    """
    n = SMALL_MATRIX_SIZE
    rng = np.random.default_rng(test_seed)

    random_mat = rng.standard_normal((n, n))
    q, _ = np.linalg.qr(random_mat)

    eigenvalues = np.concatenate([np.linspace(0.01, 0.09, n - 1), [10.0]])
    d = np.diag(eigenvalues)

    return q @ d @ q.T


# =============================================================================
# Medium Matrix Fixtures (for convergence tests)
# =============================================================================


@pytest.fixture
def tridiagonal_spd_medium() -> NDArray:
    """Medium tridiagonal SPD matrix (50x50) - for convergence tests.

    Returns:
        50x50 tridiagonal matrix: diag=4, off-diag=-1.

    Theory:
        Same structure as small version but larger size for more realistic
        convergence testing. Condition number kappa ~= 160.
    """
    n = MEDIUM_MATRIX_SIZE
    sparse_mat = diags(
        [TRIDIAGONAL_OFFDIAG_VALUE, TRIDIAGONAL_DIAG_VALUE, TRIDIAGONAL_OFFDIAG_VALUE],
        [-1, 0, 1],
        shape=(n, n),
        format="csr",
        dtype=np.float64,
    )
    return sparse_mat.toarray()


@pytest.fixture
def diagonal_spd_medium() -> NDArray:
    """Medium diagonal SPD matrix (50x50) - for convergence tests.

    Returns:
        50x50 diagonal matrix with entries [1, 2, ..., 50].

    Theory:
        Diagonal matrix with entries 1 to 50.
        Condition number kappa = 50/1 = 50 (well-conditioned).
    """
    n = MEDIUM_MATRIX_SIZE
    return np.diag(np.arange(1, n + 1, dtype=np.float64))


# =============================================================================
# RHS Vector Fixtures
# =============================================================================


@pytest.fixture
def rhs_ones_small() -> NDArray:
    """RHS vector of all ones (size 10).

    Returns:
        10D vector of ones.
    """
    return np.ones(SMALL_MATRIX_SIZE, dtype=np.float64)


@pytest.fixture
def rhs_random_small(test_seed: int) -> NDArray:
    """Random RHS vector (size 10) with fixed seed.

    Args:
        test_seed: Random seed for reproducibility.

    Returns:
        10D random vector from standard normal distribution.
    """
    rng = np.random.default_rng(test_seed)
    return rng.standard_normal(SMALL_MATRIX_SIZE)


@pytest.fixture
def rhs_zero_small() -> NDArray:
    """Zero RHS vector (size 10) - trivial solution x=0.

    Returns:
        10D zero vector.
    """
    return np.zeros(SMALL_MATRIX_SIZE, dtype=np.float64)


@pytest.fixture
def rhs_ones_medium() -> NDArray:
    """RHS vector of all ones (size 50) - for convergence tests.

    Returns:
        50D vector of ones.
    """
    return np.ones(MEDIUM_MATRIX_SIZE, dtype=np.float64)


# =============================================================================
# Test System Fixtures (Matrix + RHS with Known Solutions)
# =============================================================================


@pytest.fixture
def identity_system_known_solution(
    identity_matrix_small: NDArray,
    rhs_ones_small: NDArray,
) -> tuple[NDArray, NDArray, NDArray]:
    """Identity matrix system with known solution x = b.

    Args:
        identity_matrix_small: Identity matrix.
        rhs_ones_small: RHS vector.

    Returns:
        Tuple of (A, b, x_exact) where A @ x_exact = b.
    """
    a = identity_matrix_small
    b = rhs_ones_small
    x_exact = b.copy()
    return a, b, x_exact


@pytest.fixture
def tridiagonal_system_known_solution(
    tridiagonal_spd_small: NDArray,
    rhs_ones_small: NDArray,
) -> tuple[NDArray, NDArray, NDArray]:
    """Tridiagonal SPD system with known solution.

    Args:
        tridiagonal_spd_small: Tridiagonal matrix.
        rhs_ones_small: RHS vector.

    Returns:
        Tuple of (A, b, x_exact) where A @ x_exact = b.
    """
    a = tridiagonal_spd_small
    b = rhs_ones_small
    x_exact = np.linalg.solve(a, b)
    return a, b, x_exact


@pytest.fixture
def diagonal_system_known_solution(
    diagonal_spd_small: NDArray,
    rhs_ones_small: NDArray,
) -> tuple[NDArray, NDArray, NDArray]:
    """Diagonal SPD system with known solution.

    Args:
        diagonal_spd_small: Diagonal matrix.
        rhs_ones_small: RHS vector.

    Returns:
        Tuple of (A, b, x_exact) where A @ x_exact = b.
    """
    a = diagonal_spd_small
    b = rhs_ones_small
    x_exact = b / np.diag(a)
    return a, b, x_exact


@pytest.fixture
def tridiagonal_system_medium(
    tridiagonal_spd_medium: NDArray,
    rhs_ones_medium: NDArray,
) -> tuple[NDArray, NDArray, NDArray]:
    """Medium tridiagonal SPD system with known solution - for convergence tests.

    Args:
        tridiagonal_spd_medium: 50x50 tridiagonal matrix.
        rhs_ones_medium: 50D RHS vector.

    Returns:
        Tuple of (A, b, x_exact) where A @ x_exact = b.
    """
    a = tridiagonal_spd_medium
    b = rhs_ones_medium
    x_exact = np.linalg.solve(a, b)
    return a, b, x_exact


@pytest.fixture
def diagonal_system_medium(
    diagonal_spd_medium: NDArray,
    rhs_ones_medium: NDArray,
) -> tuple[NDArray, NDArray, NDArray]:
    """Medium diagonal SPD system with known solution - for convergence tests.

    Args:
        diagonal_spd_medium: 50x50 diagonal matrix.
        rhs_ones_medium: 50D RHS vector.

    Returns:
        Tuple of (A, b, x_exact) where A @ x_exact = b.
    """
    a = diagonal_spd_medium
    b = rhs_ones_medium
    x_exact = b / np.diag(a)
    return a, b, x_exact


# =============================================================================
# Torch-Tensor Twin Fixtures (thin adapter over the numpy systems above)
# =============================================================================


@pytest.fixture
def identity_system_known_solution_torch(
    identity_system_known_solution: tuple[NDArray, NDArray, NDArray],
    to_torch: Callable[[NDArray], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch-tensor twin of ``identity_system_known_solution``.

    Args:
        identity_system_known_solution: Numpy (A, b, x_exact) triple.
        to_torch: Numpy -> torch adapter fixture.

    Returns:
        Tuple of (A, b, x_exact) as cloned ``torch.float64`` tensors.
    """
    a, b, x_exact = identity_system_known_solution
    return to_torch(a), to_torch(b), to_torch(x_exact)


@pytest.fixture
def tridiagonal_system_known_solution_torch(
    tridiagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
    to_torch: Callable[[NDArray], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch-tensor twin of ``tridiagonal_system_known_solution``.

    Args:
        tridiagonal_system_known_solution: Numpy (A, b, x_exact) triple.
        to_torch: Numpy -> torch adapter fixture.

    Returns:
        Tuple of (A, b, x_exact) as cloned ``torch.float64`` tensors.
    """
    a, b, x_exact = tridiagonal_system_known_solution
    return to_torch(a), to_torch(b), to_torch(x_exact)


@pytest.fixture
def diagonal_system_known_solution_torch(
    diagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
    to_torch: Callable[[NDArray], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch-tensor twin of ``diagonal_system_known_solution``.

    Args:
        diagonal_system_known_solution: Numpy (A, b, x_exact) triple.
        to_torch: Numpy -> torch adapter fixture.

    Returns:
        Tuple of (A, b, x_exact) as cloned ``torch.float64`` tensors.
    """
    a, b, x_exact = diagonal_system_known_solution
    return to_torch(a), to_torch(b), to_torch(x_exact)


# =============================================================================
# Preconditioner Fixtures
# =============================================================================


@pytest.fixture
def identity_preconditioner() -> Callable[[NDArray, object], NDArray]:
    """Identity preconditioner M = I.

    Returns:
        Preconditioner function precond(r, ctx) -> z = r.
    """

    def precond(r: NDArray, context: object) -> NDArray:
        """Apply identity preconditioner.

        Args:
            r: Residual vector.
            context: Iteration context (unused).

        Returns:
            z = r (no preconditioning).
        """
        return r.copy()

    return precond


@pytest.fixture
def jacobi_preconditioner_factory() -> Callable[[NDArray], Callable[[NDArray], NDArray]]:
    """Factory for Jacobi preconditioner M = diag(A).

    Returns:
        Factory function that creates preconditioner from matrix A.

    Theory:
        Jacobi preconditioner: M = diag(A), so z_i = r_i / A_ii.
        Effective for diagonally dominant matrices, cheap to apply.
    """

    def factory(a: NDArray) -> Callable[[NDArray], NDArray]:
        """Create Jacobi preconditioner for matrix A.

        Args:
            a: System matrix.

        Returns:
            Preconditioner function.
        """
        diag_inv = 1.0 / np.diag(a)

        def precond(r: NDArray) -> NDArray:
            """Apply Jacobi preconditioner.

            Args:
                r: Residual vector.

            Returns:
                z = diag(A)^{-1} r.
            """
            return diag_inv * r

        return precond

    return factory


@pytest.fixture
def jacobi_preconditioner_tridiagonal(
    tridiagonal_spd_small: NDArray,
    jacobi_preconditioner_factory: Callable[[NDArray], Callable[[NDArray], NDArray]],
) -> Callable[[NDArray], NDArray]:
    """Jacobi preconditioner for tridiagonal test matrix.

    Args:
        tridiagonal_spd_small: Tridiagonal matrix.
        jacobi_preconditioner_factory: Factory for creating Jacobi preconditioner.

    Returns:
        Preconditioner function for tridiagonal matrix.
    """
    return jacobi_preconditioner_factory(tridiagonal_spd_small)


@pytest.fixture
def jacobi_preconditioner_medium(
    jacobi_preconditioner_factory: Callable[[NDArray], Callable[[NDArray], NDArray]],
    tridiagonal_spd_medium: NDArray,
) -> Callable[[NDArray], NDArray]:
    """Jacobi preconditioner for medium tridiagonal matrix - for convergence tests."""
    return jacobi_preconditioner_factory(tridiagonal_spd_medium)


@pytest.fixture
def ilu_preconditioner_factory() -> Callable[[NDArray], Callable[[NDArray], NDArray]]:
    """Factory for ILU preconditioner using scipy.sparse.linalg.spilu.

    Returns:
        Factory function that creates ILU preconditioner from matrix A.

    Theory:
        ILU (Incomplete LU) preconditioner: approximates A ~= LU with sparse
        factors. Uses scipy's spilu, which performs sparse ILU factorization
        with threshold dropping to control fill-in.
    """

    def factory(a: NDArray) -> Callable[[NDArray], NDArray]:
        """Create ILU preconditioner for matrix A.

        Args:
            a: System matrix.

        Returns:
            Preconditioner function.
        """
        from scipy.sparse import csc_matrix
        from scipy.sparse.linalg import spilu

        a_csc = csc_matrix(a)
        ilu = spilu(a_csc)

        def precond(r: NDArray) -> NDArray:
            """Apply ILU preconditioner.

            Args:
                r: Residual vector.

            Returns:
                z = (LU)^{-1} r via forward/backward substitution.
            """
            return ilu.solve(r)

        return precond

    return factory


@pytest.fixture
def ilu_preconditioner_tridiagonal(
    tridiagonal_spd_small: NDArray,
    ilu_preconditioner_factory: Callable[[NDArray], Callable[[NDArray], NDArray]],
) -> Callable[[NDArray], NDArray]:
    """ILU preconditioner for tridiagonal test matrix.

    Args:
        tridiagonal_spd_small: Tridiagonal matrix.
        ilu_preconditioner_factory: Factory for creating ILU preconditioner.

    Returns:
        Preconditioner function for tridiagonal matrix.
    """
    return ilu_preconditioner_factory(tridiagonal_spd_small)


@pytest.fixture
def ilu_preconditioner_diagonal(
    diagonal_spd_small: NDArray,
    ilu_preconditioner_factory: Callable[[NDArray], Callable[[NDArray], NDArray]],
) -> Callable[[NDArray], NDArray]:
    """ILU preconditioner for diagonal test matrix.

    Args:
        diagonal_spd_small: Diagonal matrix.
        ilu_preconditioner_factory: Factory for creating ILU preconditioner.

    Returns:
        Preconditioner function for diagonal matrix.

    Theory:
        For diagonal matrices, ILU factorization is exact: L=I, U=D. Should
        give identical results to Jacobi preconditioner.
    """
    return ilu_preconditioner_factory(diagonal_spd_small)


@pytest.fixture
def ilu_preconditioner_medium(
    ilu_preconditioner_factory: Callable[[NDArray], Callable[[NDArray], NDArray]],
    tridiagonal_spd_medium: NDArray,
) -> Callable[[NDArray], NDArray]:
    """ILU preconditioner for medium tridiagonal matrix - for convergence tests."""
    return ilu_preconditioner_factory(tridiagonal_spd_medium)


@pytest.fixture
def nan_preconditioner() -> Callable[[NDArray, object], NDArray]:
    """Broken preconditioner that returns NaN - for breakdown testing.

    Returns:
        Preconditioner function that returns NaN vector.

    Theory:
        Used to test breakdown detection. Should trigger breakdown flag
        immediately.
    """

    def precond(r: NDArray, context: object) -> NDArray:
        """Return NaN vector to trigger breakdown.

        Args:
            r: Residual vector.
            context: Iteration context (unused).

        Returns:
            Vector of NaNs.
        """
        return np.full_like(r, np.nan)

    return precond


# =============================================================================
# Tolerance Fixtures
# =============================================================================


@pytest.fixture
def integration_tolerances() -> tuple[float, float]:
    """Integration test tolerances (rtol=1e-6, atol=1e-14).

    Use for functionality tests, smoke tests, and general solver validation.

    Returns:
        Tuple of (rtol, atol) for integration testing.
    """
    return INTEGRATION_RTOL, INTEGRATION_ATOL


@pytest.fixture
def integration_rtol() -> float:
    """Integration test relative tolerance (1e-6).

    Returns:
        Integration test rtol value.
    """
    return INTEGRATION_RTOL


@pytest.fixture
def integration_atol() -> float:
    """Integration test absolute tolerance (1e-14).

    Returns:
        Integration test atol value.
    """
    return INTEGRATION_ATOL


@pytest.fixture
def convergence_tolerances() -> tuple[float, float]:
    """Convergence test tolerances (rtol=1e-12, atol=1e-14).

    Use for high-precision validation and comparison against direct solvers.

    Returns:
        Tuple of (rtol, atol) for convergence testing.
    """
    return CONVERGENCE_RTOL, CONVERGENCE_ATOL


@pytest.fixture
def convergence_rtol() -> float:
    """Convergence test relative tolerance (1e-12).

    Returns:
        Convergence test rtol value.
    """
    return CONVERGENCE_RTOL


@pytest.fixture
def convergence_atol() -> float:
    """Convergence test absolute tolerance (1e-14).

    Returns:
        Convergence test atol value.
    """
    return CONVERGENCE_ATOL


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def default_maxiter() -> int:
    """Default maximum iterations for solver.

    Returns:
        Maximum iteration count.
    """
    return DEFAULT_MAX_ITER


@pytest.fixture
def zero_initial_guess_small() -> NDArray:
    """Zero initial guess (size 10).

    Returns:
        10D zero vector for x0.
    """
    return np.zeros(SMALL_MATRIX_SIZE, dtype=np.float64)


@pytest.fixture
def diagonal_3x3_torch(torch_dtype: torch.dtype) -> torch.Tensor:
    """Small 3x3 diagonal SPD matrix for residual-history solver tests."""
    return torch.diag(torch.tensor([2.0, 3.0, 4.0], dtype=torch_dtype))


@pytest.fixture
def rhs_ones_3_torch(torch_dtype: torch.dtype) -> torch.Tensor:
    """Three-element all-ones RHS vector for residual-history solver tests."""
    return torch.ones(3, dtype=torch_dtype)


@pytest.fixture
def diagonal_4x4_torch(torch_dtype: torch.dtype) -> torch.Tensor:
    """Small 4x4 diagonal SPD matrix for preconditioned solver tests."""
    return torch.diag(torch.tensor([2.0, 4.0, 6.0, 8.0], dtype=torch_dtype))


@pytest.fixture
def rhs_progression_4_torch(torch_dtype: torch.dtype) -> torch.Tensor:
    """Four-element RHS vector ``[1, 2, 3, 4]`` for solver tests."""
    return torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch_dtype)


@pytest.fixture
def random_rhs_10_torch(torch_dtype: torch.dtype) -> torch.Tensor:
    """Deterministic random RHS vector of length 10 for identity-system tests."""
    generator = torch.Generator().manual_seed(42)
    return torch.randn(10, generator=generator, dtype=torch_dtype)


def _scipy_cg(
    A: NDArray,
    b: NDArray,
    x0: NDArray | None = None,
    *,
    preconditioner: Callable[[NDArray], NDArray] | Preconditioner | None = None,
    rtol: float = INTEGRATION_RTOL,
    atol: float = INTEGRATION_ATOL,
    maxiter: int | None = None,
    trace_mode: str = "minimal",
) -> tuple[NDArray, object]:
    """Solve ``A x = b`` via the scipy CG oracle with a factory-style call signature.

    Mirrors the call shape the future ``torchalg`` factories
    (``flexible_cg``, ``pcg``) will have, so ``solver_factories`` can offer
    all three under one uniform interface. Thin glue over
    ``tests.support.scipy_reference.scipy_wrapper.SciPyCGSolver`` (the frozen
    oracle) — not itself part of the frozen snapshot.

    Args:
        A: System matrix, shape (n, n). Must be square and SPD.
        b: Right-hand side vector, shape (n,).
        x0: Initial guess. If None, uses zero vector.
        preconditioner: Callable implementing M^{-1}, a ``Preconditioner``
            instance, or None (identity).
        rtol: Relative tolerance for convergence.
        atol: Absolute tolerance for convergence.
        maxiter: Maximum number of iterations.
        trace_mode: Tracing granularity ("minimal" or "full").

    Returns:
        Tuple of (x, result): solution vector and ``SolverResult``.
    """
    if preconditioner is None:
        precond_strategy: Preconditioner = Identity()
    elif isinstance(preconditioner, Preconditioner):
        precond_strategy = preconditioner
    else:
        precond_strategy = CallablePreconditioner(preconditioner)

    solver = SciPyCGSolver(preconditioner=precond_strategy)
    return solver.solve(A, b, x0, rtol=rtol, atol=atol, maxiter=maxiter, trace_mode=trace_mode)


@pytest.fixture
def solver_factories() -> dict[str, Callable]:
    """Map of solver names to factory functions.

    Returns:
        Dictionary mapping solver names to factory functions:
        {"fcg": flexible_cg, "pcg": pcg, "scipy_cg": scipy_cg}

    Use this fixture for parametrized tests that need to test all solvers::

        @pytest.mark.parametrize("solver_name", ["fcg", "pcg", "scipy_cg"])
        def test_something(solver_name, solver_factories):
            solver = solver_factories[solver_name]
            x, result = solver(A, b, ...)

    Note:
        ``"fcg"``/``"pcg"`` resolve ``torchalg`` lazily, at call time
        (they do not exist until Stage 9 of the migration); requesting this
        fixture is safe today, but calling those two entries before Stage 9
        raises ``ImportError`` by design. ``"scipy_cg"`` is fully functional
        today.
    """

    def _fcg(*args: Any, **kwargs: Any) -> tuple[Any, object]:
        """Call the torch-native FCG factory through the shared fixture map."""
        from torchalg import flexible_cg

        return flexible_cg(*args, **kwargs)

    def _pcg(*args: Any, **kwargs: Any) -> tuple[Any, object]:
        """Call the torch-native PCG factory through the shared fixture map."""
        from torchalg import pcg

        return pcg(*args, **kwargs)

    return {
        "fcg": _fcg,
        "pcg": _pcg,
        "scipy_cg": _scipy_cg,
    }
