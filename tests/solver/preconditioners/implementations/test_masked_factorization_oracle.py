"""Oracle tests: the right-looking masked factorizations equal the original scalar loops.

``_old_dense_ilu0`` and ``_old_dense_ic0`` are verbatim copies of the
implementations before the right-looking rewrite (Python IKJ triple loop with a
host read per mask test; per-pivot ``pivot <= 0`` synchronization). The new
kernels must reproduce them on random matrices with sparsity masks.
"""

from __future__ import annotations

import pytest
import torch

from torchalg.preconditioners.implementations import ILUPreconditioner
from torchalg.preconditioners.implementations._masked_factorization import (
    dense_ic0,
    dense_ilu0,
    ic0_sparsity_mask,
)

FACTOR_RTOL = 1e-13
MATRIX_SIZES = [1, 2, 17, 40]
DENSITIES = [0.1, 0.4, 1.0]


def _old_dense_ilu0(matrix: torch.Tensor) -> torch.Tensor:
    n = matrix.shape[0]
    mask = matrix != 0
    factors = matrix.clone()
    for i in range(1, n):
        for k in range(i):
            if not mask[i, k]:
                continue
            factors[i, k] = factors[i, k] / factors[k, k]
            for j in range(k + 1, n):
                if mask[i, j]:
                    factors[i, j] = factors[i, j] - factors[i, k] * factors[k, j]
    return factors


def _old_dense_ic0(matrix: torch.Tensor, threshold: float) -> torch.Tensor:
    n = matrix.shape[0]
    sparsity_mask = ic0_sparsity_mask(matrix, threshold)
    not_sparsity_mask = ~sparsity_mask
    factor = torch.where(sparsity_mask, torch.tril(matrix), torch.zeros_like(matrix))
    for k in range(n):
        pivot = factor[k, k]
        if pivot <= 0:
            raise ValueError(
                f"IC(0) breakdown at pivot {k}: diagonal value {pivot.item()} is "
                "non-positive, so no real factor L exists for this matrix "
                "(Saad, Iterative Methods for Sparse Linear Systems, 2nd ed., "
                "Sec. 10.3) - not every SPD matrix admits an IC(0) factorization."
            )
        factor[k, k] = torch.sqrt(pivot)
        if k + 1 >= n:
            break
        column = factor[k + 1 :, k] / factor[k, k]
        factor[k + 1 :, k] = column
        update = torch.outer(column, column)
        update.masked_fill_(not_sparsity_mask[k + 1 :, k + 1 :], 0)
        factor[k + 1 :, k + 1 :] -= update
    return factor


def _random_pattern(n: int, density: float, generator: torch.Generator) -> torch.Tensor:
    return torch.rand(n, n, generator=generator, dtype=torch.float64) < density


@pytest.fixture(params=[(n, density) for n in MATRIX_SIZES for density in DENSITIES])
def nonsymmetric_dominant_matrix(request: pytest.FixtureRequest) -> torch.Tensor:
    """Seeded diagonally dominant matrix with a random nonsymmetric sparsity pattern."""
    n, density = request.param
    generator = torch.Generator().manual_seed(100 * n + int(10 * density))
    values = torch.randn(n, n, generator=generator, dtype=torch.float64)
    off_diagonal = torch.where(
        _random_pattern(n, density, generator), values, torch.zeros_like(values)
    )
    off_diagonal.fill_diagonal_(0.0)
    row_sums = off_diagonal.abs().sum(dim=1)
    return off_diagonal + torch.diag(row_sums + 1.0)


@pytest.fixture(params=[(n, density) for n in MATRIX_SIZES for density in DENSITIES])
def sparse_spd_matrix(request: pytest.FixtureRequest) -> torch.Tensor:
    """Seeded symmetric, strictly diagonally dominant (hence SPD) matrix with a random pattern."""
    n, density = request.param
    generator = torch.Generator().manual_seed(7 * n + int(10 * density))
    values = torch.randn(n, n, generator=generator, dtype=torch.float64)
    symmetric = torch.tril(values, -1) + torch.tril(values, -1).T
    pattern = _random_pattern(n, density, generator)
    symmetric = torch.where(pattern & pattern.T, symmetric, torch.zeros_like(symmetric))
    return symmetric + torch.diag(symmetric.abs().sum(dim=1) + 1.0)


@pytest.fixture
def indefinite_matrix() -> torch.Tensor:
    """Symmetric matrix whose third IC(0) pivot is negative (first two are fine)."""
    return torch.tensor([[4.0, 1.0, 0.0], [1.0, 4.0, 3.0], [0.0, 3.0, 1.0]], dtype=torch.float64)


@pytest.fixture
def residual_matrix(sparse_spd_matrix: torch.Tensor) -> torch.Tensor:
    """Seeded multi-column right-hand side matching ``sparse_spd_matrix``."""
    generator = torch.Generator().manual_seed(5)
    return torch.randn(sparse_spd_matrix.shape[0], 3, generator=generator, dtype=torch.float64)


def test_dense_ilu0_matches_scalar_loop(nonsymmetric_dominant_matrix: torch.Tensor) -> None:
    """Right-looking masked ILU(0) reproduces the IKJ triple loop."""
    new = dense_ilu0(nonsymmetric_dominant_matrix)
    old = _old_dense_ilu0(nonsymmetric_dominant_matrix)

    torch.testing.assert_close(new, old, rtol=FACTOR_RTOL, atol=FACTOR_RTOL)


def test_dense_ilu0_never_fills_outside_the_pattern(
    nonsymmetric_dominant_matrix: torch.Tensor,
) -> None:
    """Entries that were zero in ``A`` stay exactly zero."""
    factors = dense_ilu0(nonsymmetric_dominant_matrix)

    assert torch.count_nonzero(factors[nonsymmetric_dominant_matrix == 0]) == 0


def test_dense_ic0_matches_scalar_loop(sparse_spd_matrix: torch.Tensor) -> None:
    """Gathered right-looking IC(0) reproduces the trailing-block reference."""
    new = dense_ic0(sparse_spd_matrix, 0.0)
    old = _old_dense_ic0(sparse_spd_matrix, 0.0)

    torch.testing.assert_close(new, old, rtol=FACTOR_RTOL, atol=FACTOR_RTOL)


def test_dense_ic0_matches_scalar_loop_with_drop_threshold(sparse_spd_matrix: torch.Tensor) -> None:
    """A drop tolerance shrinks the pattern identically in both versions."""
    threshold = 0.3
    try:
        old = _old_dense_ic0(sparse_spd_matrix, threshold)
    except ValueError as expected:
        with pytest.raises(ValueError, match="breakdown") as raised:
            dense_ic0(sparse_spd_matrix, threshold)
        assert str(raised.value) == str(expected)
        return

    torch.testing.assert_close(
        dense_ic0(sparse_spd_matrix, threshold), old, rtol=FACTOR_RTOL, atol=FACTOR_RTOL
    )


def test_dense_ic0_breakdown_names_the_first_bad_pivot_like_the_old_check(
    indefinite_matrix: torch.Tensor,
) -> None:
    """The deferred check raises the same error, with the same pivot index and value."""
    with pytest.raises(ValueError, match="breakdown") as old_error:
        _old_dense_ic0(indefinite_matrix, 0.0)

    with pytest.raises(ValueError, match="breakdown") as new_error:
        dense_ic0(indefinite_matrix, 0.0)

    assert str(new_error.value) == str(old_error.value)
    assert "pivot 2" in str(new_error.value)


def test_ilu_apply_matches_explicit_triangular_factors(
    sparse_spd_matrix: torch.Tensor, residual_matrix: torch.Tensor
) -> None:
    """Applying the packed factor equals solving with separately built ``L`` and ``U``."""
    preconditioner = ILUPreconditioner().setup(sparse_spd_matrix)
    packed = preconditioner._operator
    n = packed.shape[0]
    lower = torch.tril(packed, -1) + torch.eye(n, dtype=packed.dtype)
    upper = torch.triu(packed)
    expected = torch.linalg.solve_triangular(
        upper,
        torch.linalg.solve_triangular(lower, residual_matrix, upper=False, unitriangular=True),
        upper=True,
    )

    torch.testing.assert_close(
        preconditioner.apply(residual_matrix), expected, rtol=FACTOR_RTOL, atol=FACTOR_RTOL
    )


def test_ilu_apply_builds_no_triangle_or_identity_matrices(
    sparse_spd_matrix: torch.Tensor,
    residual_matrix: torch.Tensor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``apply`` does no per-call ``eye``/``tril``/``triu`` construction."""
    preconditioner = ILUPreconditioner().setup(sparse_spd_matrix)

    def forbidden(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("apply() rebuilt a factor matrix")

    for name in ("eye", "tril", "triu"):
        monkeypatch.setattr(torch, name, forbidden)

    preconditioner.apply(residual_matrix)
