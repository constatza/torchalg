"""Shared fixtures for ``tests/utils/test_ls_interpolation.py``.

Kept minimal and local to this directory since the batched LS-interpolation
tests are the only consumers so far; promote to the root conftest if a
second consumer appears (single source of truth, no premature sharing).
"""

from __future__ import annotations

import pytest
import torch

from tests.utils.selection_problem import SelectionProblem


@pytest.fixture
def ls_test_vectors(test_seed: int) -> torch.Tensor:
    """Small random dense test-vector matrix ``V``, shape ``(12, 7)``.

    ``k=7`` is deliberately larger than the test suite's ``caliber=4``, so
    every trial interpolatory set stays strictly rank-deficient relative to
    ``k`` and its LS residual stays safely away from the ``~0`` floating-
    point noise floor a square/overdetermined (``|C_i| >= k``) exact fit
    would hit - at ``|C_i| == k`` the normal-equations residual for *every*
    remaining candidate collapses to numerical noise, making "the minimizing
    candidate" an unstable, batched-vs-looped-computation-order-dependent
    tie rather than a real comparison, which previously made this fixture
    (at ``k=3``) produce spurious batched/single-row mismatches unrelated to
    the functions under test.

    Args:
        test_seed: Fixed seed for reproducibility.

    Returns:
        Dense float64 tensor, shape ``(n=12, k=7)``.
    """
    generator = torch.Generator().manual_seed(test_seed)
    return torch.randn(12, 7, generator=generator, dtype=torch.float64)


@pytest.fixture
def ls_weights() -> torch.Tensor:
    """Per-test-vector weights ``omega``, shape ``(7,)``.

    Returns:
        Positive float64 weights.
    """
    return torch.tensor([1.0, 2.0, 0.5, 1.5, 0.8, 1.2, 0.6], dtype=torch.float64)


_SELECTION_ROWS = 50
_SELECTION_MAX_CANDIDATES = 10
_SELECTION_TEST_VECTORS = 8
_DUPLICATED_BLOCK = 10
"""Rows ``[10, 20)`` of the degenerate problem duplicate rows ``[0, 10)``; rows ``[20, 25)`` are zero."""


def _assemble_problem(
    vectors: torch.Tensor,
    candidate_pool: int,
    counts: torch.Tensor,
    generator: torch.Generator,
) -> SelectionProblem:
    """Draw per-row candidates from ``[0, candidate_pool)`` and pad like the real call site.

    Padded slots duplicate each row's candidate 0 and are masked ``False``,
    exactly the layout ``BAMGCoarsening`` produces.
    """
    n, k = vectors.shape
    drawn = torch.randint(
        0, candidate_pool, (_SELECTION_ROWS, _SELECTION_MAX_CANDIDATES), generator=generator
    )
    mask = torch.arange(_SELECTION_MAX_CANDIDATES) < counts.unsqueeze(1)
    padded = torch.where(mask, drawn, drawn[:, :1].expand_as(drawn))
    targets = torch.randint(candidate_pool, n, (_SELECTION_ROWS,), generator=generator)
    weights = 0.5 + torch.rand(k, generator=generator, dtype=torch.float64)
    return SelectionProblem(
        padded_candidates=padded,
        candidate_mask=mask,
        test_vectors=vectors,
        targets=targets,
        weights=weights.to(vectors.dtype),
    )


@pytest.fixture(params=[torch.float32, torch.float64], ids=["float32", "float64"])
def float_dtype(request: pytest.FixtureRequest) -> torch.dtype:
    """Floating dtype the selection-stage tests run in."""
    return request.param


@pytest.fixture
def random_selection_problem(test_seed: int, float_dtype: torch.dtype) -> SelectionProblem:
    """Well-conditioned random vectors, candidate counts in ``[1, max_cand]``."""
    generator = torch.Generator().manual_seed(test_seed)
    vectors = torch.randn(200, _SELECTION_TEST_VECTORS, generator=generator, dtype=torch.float64)
    counts = torch.randint(
        1, _SELECTION_MAX_CANDIDATES + 1, (_SELECTION_ROWS,), generator=generator
    )
    return _assemble_problem(vectors.to(float_dtype), 150, counts, generator)


@pytest.fixture
def short_candidate_selection_problem(test_seed: int, float_dtype: torch.dtype) -> SelectionProblem:
    """Rows with 0, 1, 2 and 3 valid candidates (fewer than the caliber) mixed with full rows."""
    generator = torch.Generator().manual_seed(test_seed + 1)
    vectors = torch.randn(200, _SELECTION_TEST_VECTORS, generator=generator, dtype=torch.float64)
    short_counts = torch.arange(_SELECTION_ROWS) % 4
    full = torch.full((_SELECTION_ROWS,), _SELECTION_MAX_CANDIDATES)
    counts = torch.where(torch.arange(_SELECTION_ROWS) % 5 == 0, full, short_counts)
    return _assemble_problem(vectors.to(float_dtype), 150, counts, generator)


@pytest.fixture
def degenerate_selection_problem(test_seed: int, float_dtype: torch.dtype) -> SelectionProblem:
    """Candidate pool holding exact duplicates and all-zero rows (singular Gram systems).

    Candidates are drawn from the first 30 vector rows, of which ``[10, 20)``
    duplicate ``[0, 10)`` and ``[20, 25)`` are zero; many trial Gram matrices
    are exactly rank-deficient, and chosen indices may repeat the same vector.
    """
    generator = torch.Generator().manual_seed(test_seed + 2)
    vectors = torch.randn(120, _SELECTION_TEST_VECTORS, generator=generator, dtype=torch.float64)
    vectors[_DUPLICATED_BLOCK : 2 * _DUPLICATED_BLOCK] = vectors[:_DUPLICATED_BLOCK]
    vectors[2 * _DUPLICATED_BLOCK : 2 * _DUPLICATED_BLOCK + 5] = 0.0
    counts = torch.randint(
        1, _SELECTION_MAX_CANDIDATES + 1, (_SELECTION_ROWS,), generator=generator
    )
    return _assemble_problem(vectors.to(float_dtype), 30, counts, generator)
