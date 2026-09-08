"""Fixtures for ``torchalg.utils`` (numerics/validation) tests.

Modular, composable fixtures per project convention — no inline test data.
Builds on the session-wide ``torch_dtype``/``to_torch`` fixtures from
``tests/conftest.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

# =============================================================================
# stable_dot_product fixtures
# =============================================================================


@pytest.fixture
def orthogonal_dot_vectors(torch_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Two small vectors with a hand-computable dot product (11.0).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(a, b)`` with ``a . b == 11.0``.
    """
    a = torch.tensor([1.0, 2.0], dtype=torch_dtype)
    b = torch.tensor([3.0, 4.0], dtype=torch_dtype)
    return a, b


@pytest.fixture
def overflow_cancelling_vectors() -> tuple[torch.Tensor, torch.Tensor]:
    """Vectors whose naive per-element products overflow but whose true dot is 0.

    ``a = [1e200, 1e200]``, ``b = [1e200, -1e200]``: each product
    (``1e200 * 1e200 = 1e400``) exceeds ``float64``'s representable range
    (~1.8e308) and would overflow to ``+-inf`` under a naive
    ``torch.dot``, then to ``nan`` on summation (``inf + -inf``). The
    balanced-scaling algorithm keeps every intermediate finite, so the exact
    mathematical answer (``0.0``, via perfect cancellation) is recovered.
    Deliberately ``float64`` regardless of the session default: the scenario
    is only meaningful relative to ``float64``'s own overflow threshold.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(a, b)`` with true dot product
            ``0.0``, but naive (unscaled) evaluation would overflow to
            ``nan``.
    """
    a = torch.tensor([1e200, 1e200], dtype=torch.float64)
    b = torch.tensor([1e200, -1e200], dtype=torch.float64)
    return a, b


# =============================================================================
# validate_matrix / validate_rhs_vector / validate_ax_equals_b fixtures
# =============================================================================


@pytest.fixture
def valid_spd_system(
    tridiagonal_system_known_solution: tuple[NDArray, NDArray, NDArray],
    to_torch: Callable[[NDArray], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch-tensor (A, b, x_exact) triple satisfying ``A @ x_exact == b``.

    Reuses ``tridiagonal_system_known_solution`` from
    ``tests/solver/conftest.py`` so the validation tests exercise a system
    consistent with the rest of the suite rather than one-off inline data.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ``(A, b, x_exact)``.
    """
    a, b, x_exact = tridiagonal_system_known_solution
    return to_torch(a), to_torch(b), to_torch(x_exact)


@pytest.fixture
def non_square_matrix(torch_dtype: torch.dtype) -> torch.Tensor:
    """A 2x3 matrix - invalid input for ``validate_matrix``.

    Returns:
        torch.Tensor: Non-square matrix, shape ``(2, 3)``.
    """
    return torch.ones((2, 3), dtype=torch_dtype)


@pytest.fixture
def non_finite_matrix(torch_dtype: torch.dtype) -> torch.Tensor:
    """A 2x2 matrix containing a ``nan`` entry.

    Returns:
        torch.Tensor: Square matrix with one non-finite entry.
    """
    return torch.tensor([[1.0, float("nan")], [0.0, 1.0]], dtype=torch_dtype)


@pytest.fixture
def non_finite_vector(torch_dtype: torch.dtype) -> torch.Tensor:
    """A length-3 vector containing an ``inf`` entry.

    Returns:
        torch.Tensor: Vector with one non-finite entry.
    """
    return torch.tensor([1.0, float("inf"), 3.0], dtype=torch_dtype)


@pytest.fixture
def finite_solution_vector(torch_dtype: torch.dtype) -> torch.Tensor:
    """A length-3 fully-finite vector.

    Returns:
        torch.Tensor: Finite solution vector.
    """
    return torch.tensor([1.0, 2.0, 3.0], dtype=torch_dtype)


@pytest.fixture
def nan_solution_vector(torch_dtype: torch.dtype) -> torch.Tensor:
    """A length-3 vector containing only a ``nan`` (no ``inf``).

    Returns:
        torch.Tensor: Solution vector with a ``nan`` entry.
    """
    return torch.tensor([1.0, float("nan"), 3.0], dtype=torch_dtype)


@pytest.fixture
def inf_solution_vector(torch_dtype: torch.dtype) -> torch.Tensor:
    """A length-3 vector containing only an ``inf`` (no ``nan``).

    Returns:
        torch.Tensor: Solution vector with an ``inf`` entry.
    """
    return torch.tensor([1.0, float("inf"), 3.0], dtype=torch_dtype)


@pytest.fixture
def nan_and_inf_solution_vector(torch_dtype: torch.dtype) -> torch.Tensor:
    """A length-4 vector containing both a ``nan`` and an ``inf``.

    Returns:
        torch.Tensor: Solution vector with both a ``nan`` and an ``inf``
            entry.
    """
    return torch.tensor([1.0, float("nan"), float("inf"), 3.0], dtype=torch_dtype)


# =============================================================================
# condition_number fixtures
# =============================================================================


@pytest.fixture
def well_conditioned_spd_matrix(torch_dtype: torch.dtype, test_seed: int) -> torch.Tensor:
    """Dense SPD matrix with a modest, hand-verifiable condition number.

    Built as ``Q @ diag([1, 2, 3, 4]) @ Q.T`` for a random orthonormal
    ``Q``, so its exact 2-norm condition number is ``4 / 1 == 4.0``
    regardless of ``Q``, since an orthogonal similarity transform preserves
    eigenvalues.

    Returns:
        torch.Tensor: Symmetric ``(4, 4)`` matrix with condition number
            exactly ``4.0``.
    """
    generator = torch.Generator().manual_seed(test_seed)
    orthonormal_basis, _ = torch.linalg.qr(
        torch.randn(4, 4, generator=generator, dtype=torch_dtype)
    )
    eigenvalues = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch_dtype)
    matrix = orthonormal_basis @ torch.diag(eigenvalues) @ orthonormal_basis.T
    return 0.5 * (matrix + matrix.T)


@pytest.fixture
def heterogeneous_stiffness_matrix(
    torch_dtype: torch.dtype, test_seed: int
) -> tuple[torch.Tensor, float]:
    """Dense SPD matrix mimicking a stiff cube with a 1000x-softer inclusion.

    Half the eigenvalues sit at ``O(1)`` (the stiff bulk material) and half
    at ``O(1e-6)`` (a much softer embedded region, e.g. a soft sphere ~1e3x
    less stiff than the surrounding cube, further compounded by the usual
    mesh-size-driven spread of an FEM stiffness spectrum) — the exact regime
    where naive power-iteration-based condition-number estimators are known
    to fail (see ``torchalg.utils.spectral``). Built via an orthogonal
    similarity transform of a diagonal matrix, so the true condition number
    is known exactly: it is invariant under conjugation by an orthogonal
    matrix.

    Returns:
        tuple[torch.Tensor, float]: ``(matrix, true_condition_number)``.
    """
    n = 64
    generator = torch.Generator().manual_seed(test_seed)
    orthonormal_basis, _ = torch.linalg.qr(
        torch.randn(n, n, generator=generator, dtype=torch_dtype)
    )
    eigenvalues = torch.empty(n, dtype=torch_dtype)
    eigenvalues[: n // 2] = 1.0 + torch.rand(n // 2, generator=generator, dtype=torch_dtype)
    eigenvalues[n // 2 :] = 1e-6 * (
        1.0 + torch.rand(n - n // 2, generator=generator, dtype=torch_dtype)
    )
    matrix = orthonormal_basis @ torch.diag(eigenvalues) @ orthonormal_basis.T
    matrix = 0.5 * (matrix + matrix.T)
    true_condition_number = float(eigenvalues.max() / eigenvalues.min())
    return matrix, true_condition_number


@pytest.fixture
def singular_matrix(torch_dtype: torch.dtype) -> torch.Tensor:
    """A symmetric matrix with an exact zero eigenvalue (singular, not SPD).

    Returns:
        torch.Tensor: ``diag([0.0, 1.0, 2.0])`` - condition number is
            undefined (infinite).
    """
    return torch.diag(torch.tensor([0.0, 1.0, 2.0], dtype=torch_dtype))
