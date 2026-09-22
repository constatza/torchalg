"""LS/LSR interpolation and caliber-bounded set-selection tests (Bootstrap AMG, docs/bootstrap-amg.md Sec. 3).

Fixtures live in this module, matching ``test_algebraic_distance.py``'s and
``test_compatible_relaxation.py``'s precedent: each ``implementations/amg/``
test module keeps its fixtures local rather than pre-emptively factoring
them into a shared conftest before a second consumer exists.
"""

from __future__ import annotations

import pytest
import torch

from torchalg.preconditioners.implementations.amg._least_squares import (
    ls_interpolation_row,
    lsr_correction,
    select_interpolatory_set,
)


@pytest.fixture
def full_rank_test_vectors(torch_dtype: torch.dtype) -> torch.Tensor:
    """Test vectors, shape ``(3, 2)``, whose first two rows span R^2 exactly."""
    return torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]], dtype=torch_dtype)


@pytest.fixture
def full_rank_interp_set() -> torch.Tensor:
    """Interpolatory set ``{0, 1}`` - matches ``full_rank_test_vectors``'s full-rank rows."""
    return torch.tensor([0, 1], dtype=torch.long)


@pytest.fixture
def uniform_weights_2(torch_dtype: torch.dtype) -> torch.Tensor:
    """Unit weights, shape ``(2,)``."""
    return torch.ones(2, dtype=torch_dtype)


def test_ls_interpolation_row_exact_when_full_rank(
    full_rank_test_vectors: torch.Tensor,
    full_rank_interp_set: torch.Tensor,
    uniform_weights_2: torch.Tensor,
) -> None:
    """``|C_i| == k`` and ``V_{C_i}`` full rank: ``p_i`` must reproduce ``v_i`` exactly (Remark 3.1)."""
    p = ls_interpolation_row(
        full_rank_test_vectors, target=2, interp_set=full_rank_interp_set, weights=uniform_weights_2
    )
    reconstructed = p @ full_rank_test_vectors[full_rank_interp_set]
    assert torch.allclose(reconstructed, full_rank_test_vectors[2], atol=1e-8)


@pytest.fixture
def rank_deficient_test_vectors(torch_dtype: torch.dtype) -> torch.Tensor:
    """Test vectors, shape ``(3, 2)``, whose rows 0 and 1 are identical (rank-1 ``V_{C_i}``)."""
    return torch.tensor([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]], dtype=torch_dtype)


@pytest.fixture
def rank_deficient_interp_set() -> torch.Tensor:
    """Interpolatory set ``{0, 1}`` - the duplicate rows of ``rank_deficient_test_vectors``."""
    return torch.tensor([0, 1], dtype=torch.long)


def test_ls_interpolation_row_falls_back_to_lstsq_when_rank_deficient(
    rank_deficient_test_vectors: torch.Tensor,
    rank_deficient_interp_set: torch.Tensor,
    uniform_weights_2: torch.Tensor,
) -> None:
    """Duplicate rows in ``V_{C_i}`` make the Gram matrix singular; ``lstsq`` must still return a finite row."""
    p = ls_interpolation_row(
        rank_deficient_test_vectors,
        target=2,
        interp_set=rank_deficient_interp_set,
        weights=uniform_weights_2,
    )
    assert p.shape == (2,)
    assert torch.isfinite(p).all()


@pytest.fixture
def lsr_no_increase_test_vectors(torch_dtype: torch.dtype) -> torch.Tensor:
    """Seeded random test vectors, shape ``(16, 3)``, for the LSR non-increase check."""
    torch.manual_seed(2)
    return torch.randn(16, 3, dtype=torch_dtype)


@pytest.fixture
def lsr_no_increase_interp_set() -> torch.Tensor:
    """Interpolatory set ``{0, 1, 2}`` used by the LSR non-increase check."""
    return torch.tensor([0, 1, 2], dtype=torch.long)


@pytest.fixture
def lsr_no_increase_weights(torch_dtype: torch.dtype) -> torch.Tensor:
    """Unit weights, shape ``(3,)``, used by the LSR non-increase check."""
    return torch.ones(3, dtype=torch_dtype)


@pytest.fixture
def lsr_no_increase_target_rows() -> torch.Tensor:
    """Single target row (index 5) used by the LSR non-increase check."""
    return torch.tensor([5], dtype=torch.long)


def test_lsr_correction_does_not_increase_fit_error(
    anisotropic_2d_factory,
    lsr_no_increase_test_vectors: torch.Tensor,
    lsr_no_increase_interp_set: torch.Tensor,
    lsr_no_increase_weights: torch.Tensor,
    lsr_no_increase_target_rows: torch.Tensor,
) -> None:
    """LSR's residual correction is one step of the local LS minimization: it cannot worsen the fit."""
    matrix = anisotropic_2d_factory(4, 0.1)
    target = 5
    before = ls_interpolation_row(
        lsr_no_increase_test_vectors, target, lsr_no_increase_interp_set, lsr_no_increase_weights
    )
    corrected = lsr_correction(lsr_no_increase_test_vectors, matrix, lsr_no_increase_target_rows)
    after = ls_interpolation_row(
        corrected, target, lsr_no_increase_interp_set, lsr_no_increase_weights
    )
    residual_before = (
        (
            lsr_no_increase_test_vectors[target]
            - before @ lsr_no_increase_test_vectors[lsr_no_increase_interp_set]
        )
        .pow(2)
        .sum()
    )
    residual_after = (
        (corrected[target] - after @ corrected[lsr_no_increase_interp_set]).pow(2).sum()
    )
    assert residual_after <= residual_before + 1e-8


@pytest.fixture
def poisson_lsr_test_vectors(torch_dtype: torch.dtype) -> torch.Tensor:
    """Non-random test vectors, shape ``(4, 1)``, for the exact eq. 2.3 arithmetic check."""
    return torch.tensor([[1.0], [2.0], [4.0], [8.0]], dtype=torch_dtype)


@pytest.fixture
def lsr_target_rows() -> torch.Tensor:
    """Single target row (index 1) for the exact eq. 2.3 arithmetic check."""
    return torch.tensor([1], dtype=torch.long)


@pytest.fixture
def lsr_expected_corrected(torch_dtype: torch.dtype) -> torch.Tensor:
    """Expected result of eq. 2.3 applied to ``poisson_lsr_test_vectors`` at row 1.

    Row 1 of the tridiagonal ``poisson_1d_factory(4)`` matrix is
    ``[-1, 2, -1, 0]``, so ``(A v)_1 = -1*1 + 2*2 - 1*4 = -1``; with
    ``a_11 = 2``, ``v_1 <- 2 - (-1)/2 = 2.5``. Every other row is untouched.
    """
    return torch.tensor([[1.0], [2.5], [4.0], [8.0]], dtype=torch_dtype)


def test_lsr_correction_matches_eq_2_3_and_does_not_mutate_input(
    poisson_1d_factory,
    poisson_lsr_test_vectors: torch.Tensor,
    lsr_target_rows: torch.Tensor,
    lsr_expected_corrected: torch.Tensor,
) -> None:
    """``lsr_correction`` reproduces eq. 2.3's exact arithmetic and returns a new tensor."""
    matrix = poisson_1d_factory(4)
    original = poisson_lsr_test_vectors.clone()

    corrected = lsr_correction(poisson_lsr_test_vectors, matrix, lsr_target_rows)

    assert torch.allclose(corrected, lsr_expected_corrected, atol=1e-12)
    assert torch.equal(poisson_lsr_test_vectors, original)


@pytest.fixture
def caliber_bound_test_vectors(torch_dtype: torch.dtype) -> torch.Tensor:
    """Seeded random test vectors, shape ``(16, 4)``, for the caliber-bound check."""
    torch.manual_seed(3)
    return torch.randn(16, 4, dtype=torch_dtype)


@pytest.fixture
def caliber_bound_candidates() -> torch.Tensor:
    """Candidate index set for the caliber-bound check."""
    return torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)


@pytest.fixture
def caliber_bound_weights(torch_dtype: torch.dtype) -> torch.Tensor:
    """Unit weights, shape ``(4,)``, for the caliber-bound check."""
    return torch.ones(4, dtype=torch_dtype)


def test_select_interpolatory_set_respects_caliber(
    anisotropic_2d_factory,
    caliber_bound_test_vectors: torch.Tensor,
    caliber_bound_candidates: torch.Tensor,
    caliber_bound_weights: torch.Tensor,
) -> None:
    """The returned set never exceeds ``caliber`` and only draws from ``candidates``."""
    matrix = anisotropic_2d_factory(4, 0.1)
    interp_set = select_interpolatory_set(
        caliber_bound_candidates,
        caliber_bound_test_vectors,
        matrix,
        target=0,
        weights=caliber_bound_weights,
        caliber=3,
    )
    assert len(interp_set) <= 3
    assert set(interp_set.tolist()) <= set(caliber_bound_candidates.tolist())


@pytest.fixture
def scale_factors() -> tuple[float, ...]:
    """Test-vector magnitudes spanning the range bootstrap cycles actually produce.

    ``BootstrapSetup.run`` relaxes/cycles on ``A x = 0`` (exact solution
    ``0``), driving the test vectors' magnitude down by many orders of
    magnitude - measured around ``1e-08`` at N=31 after the default two
    bootstrap cycles. The selection rule must behave identically at every
    one of these scales.
    """
    return (1.0, 1e-4, 1e-8, 1e-12)


def test_select_interpolatory_set_is_non_empty_in_the_non_degenerate_case(
    anisotropic_2d_factory,
    caliber_bound_test_vectors: torch.Tensor,
    caliber_bound_candidates: torch.Tensor,
    caliber_bound_weights: torch.Tensor,
    scale_factors: tuple[float, ...],
) -> None:
    """Reasonable test vectors + reasonable caliber + available candidates must give a non-empty ``C_i``.

    A genuine lower bound (the upper bound is
    ``test_select_interpolatory_set_respects_caliber``): an empty set means
    an all-zero prolongation row, i.e. an F-point invisible to the coarse
    grid. Asserted at every scale in ``scale_factors`` because [AD11] Sec.
    4.3's penalization rule is only meaningful on a *relative* residual -
    an implementation comparing raw functional values collapses to the
    empty set as soon as the test vectors are small.
    """
    matrix = anisotropic_2d_factory(4, 0.1)
    for scale in scale_factors:
        interp_set = select_interpolatory_set(
            caliber_bound_candidates,
            caliber_bound_test_vectors * scale,
            matrix,
            target=0,
            weights=caliber_bound_weights,
            caliber=3,
        )
        assert interp_set.numel() > 0, f"empty interpolatory set at test-vector scale {scale}"


def test_select_interpolatory_set_is_scale_invariant(
    anisotropic_2d_factory,
    caliber_bound_test_vectors: torch.Tensor,
    caliber_bound_candidates: torch.Tensor,
    caliber_bound_weights: torch.Tensor,
    scale_factors: tuple[float, ...],
) -> None:
    """Scaling every test vector by a constant must not change the chosen set.

    ``LS_W`` scales as ``||V||^2``, so a raw-value penalization comparison
    is scale-dependent; the normalized comparison this module implements is
    not. Regression guard for the bug where bootstrap-shrunk test vectors
    made growth stall at the empty set.
    """
    matrix = anisotropic_2d_factory(4, 0.1)
    reference = select_interpolatory_set(
        caliber_bound_candidates,
        caliber_bound_test_vectors,
        matrix,
        target=0,
        weights=caliber_bound_weights,
        caliber=3,
    )
    for scale in scale_factors:
        scaled = select_interpolatory_set(
            caliber_bound_candidates,
            caliber_bound_test_vectors * scale,
            matrix,
            target=0,
            weights=caliber_bound_weights,
            caliber=3,
        )
        assert scaled.tolist() == reference.tolist(), f"set changed at scale {scale}"


@pytest.fixture
def single_tv_test_vectors(torch_dtype: torch.dtype) -> torch.Tensor:
    """Single-test-vector data, shape ``(3, 1)``: candidate 0 alone perfectly fits the target row."""
    return torch.tensor([[2.0], [1.0], [4.0]], dtype=torch_dtype)


@pytest.fixture
def single_tv_candidates() -> torch.Tensor:
    """Candidate index set ``{0, 1}`` for the penalization-stop check."""
    return torch.tensor([0, 1], dtype=torch.long)


@pytest.fixture
def single_tv_weights(torch_dtype: torch.dtype) -> torch.Tensor:
    """Unit weight, shape ``(1,)``, for the penalization-stop check."""
    return torch.ones(1, dtype=torch_dtype)


def test_select_interpolatory_set_stops_growth_when_penalization_fails(
    poisson_1d_factory,
    single_tv_test_vectors: torch.Tensor,
    single_tv_candidates: torch.Tensor,
    single_tv_weights: torch.Tensor,
) -> None:
    """Candidate 0 (value 2.0) exactly fits the target (4.0 = 2 * 2.0), driving ``LS_{W'}`` to 0.

    Once ``LS_{W'} == 0``, the penalization threshold ``(LS_{W'})^gamma`` is
    also 0, and no further addition's ``LS_{W''}`` (a nonnegative sum of
    squares) can be strictly less than 0 - so growth must stop at caliber 1
    even though ``caliber=2`` would allow a second entry.
    """
    matrix = poisson_1d_factory(3)
    interp_set = select_interpolatory_set(
        single_tv_candidates,
        single_tv_test_vectors,
        matrix,
        target=2,
        weights=single_tv_weights,
        caliber=2,
    )
    assert interp_set.tolist() == [0]
