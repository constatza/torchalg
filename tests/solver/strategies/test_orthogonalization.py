"""Tests for ``torchalg.strategies.orthogonalization``."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest
import torch

from torchalg.models.history import DirectionWindow
from torchalg.strategies.orthogonalization import (
    ModifiedGramSchmidt,
    PeriodicRestartOrthogonalization,
    TruncatedGramSchmidt,
    create_fcg_orthogonalization,
)


def test_truncated_gram_schmidt_removes_a_conjugate_component(
    orthogonalization_probe: torch.Tensor,
    previous_direction: torch.Tensor,
    previous_matrix_product: torch.Tensor,
    make_window: Callable[[Sequence[torch.Tensor], Sequence[torch.Tensor]], DirectionWindow],
) -> None:
    """Truncated Gram-Schmidt enforces d_i^T A d_j = 0 for the stored direction."""
    result, report = TruncatedGramSchmidt(window_size=2).orthogonalize(
        vector=orthogonalization_probe,
        window=make_window((previous_direction,), (previous_matrix_product,)),
    )

    assert torch.allclose(result, torch.tensor([0.0, 1.0], dtype=result.dtype))
    assert len(report.coefficients) == 1
    assert report.coefficients[0].item() == 2.0
    assert bool(report.breakdown) is False


def test_modified_gram_schmidt_uses_all_history(
    orthogonalization_probe: torch.Tensor,
    previous_direction: torch.Tensor,
    previous_matrix_product: torch.Tensor,
    make_window: Callable[[Sequence[torch.Tensor], Sequence[torch.Tensor]], DirectionWindow],
) -> None:
    """Modified Gram-Schmidt advertises unlimited history and orthogonalizes."""
    strategy = ModifiedGramSchmidt()
    result, report = strategy.orthogonalize(
        vector=orthogonalization_probe,
        window=make_window((previous_direction,), (previous_matrix_product,)),
    )

    assert strategy.window_size is None
    assert torch.allclose(result, torch.tensor([0.0, 1.0], dtype=result.dtype))
    assert bool(report.breakdown) is False


def test_create_fcg_orthogonalization_handles_full_window_sentinel() -> None:
    """The factory maps m_max=-1 to an unlimited periodic-restart strategy."""
    strategy = create_fcg_orthogonalization(m_max=-1)
    assert isinstance(strategy, PeriodicRestartOrthogonalization)
    assert strategy.window_size is None


def test_create_fcg_orthogonalization_rejects_zero_window() -> None:
    """A zero orthogonalization window is invalid."""
    with pytest.raises(ValueError, match="m_max cannot be 0"):
        create_fcg_orthogonalization(m_max=0)


def test_periodic_restart_window_follows_notay_sawtooth(
    periodic_restart_history: DirectionWindow,
    torch_dtype: torch.dtype,
) -> None:
    """FCG(m) window sizes cycle ``1, 2, ..., m_max, 1, 2, ...`` (period ``m_max``).

    This is Notay (2000)'s periodic-restart truncation as documented by
    PETSc's ``KSPFCG``/``KSPPIPEFCG``/``KSPPIPEGCR`` ``notay`` truncation
    type (``src/ksp/ksp/impls/fcg/fcg.c``: ``mi = ((i - 1) % mmax) + 1``,
    i.e. a clean sawtooth of period ``mmax``) — see the module docstring
    for ``torchalg.strategies.orthogonalization``. History is held fixed at
    5 stored directions throughout (>= ``m_max``) so only the cycling
    formula is under test, not the separate ``min(m_i, n_history)`` ramp-up
    while history is still being filled.
    """
    strategy = PeriodicRestartOrthogonalization(m_max=3.0)
    probe = torch.ones(5, dtype=torch_dtype)

    window_sizes = [
        len(strategy.orthogonalize(probe, periodic_restart_history)[1].coefficients)
        for _ in range(8)
    ]

    assert window_sizes == [1, 2, 3, 1, 2, 3, 1, 2]


@pytest.mark.parametrize(
    "strategy",
    [
        PeriodicRestartOrthogonalization(m_max=3.0),
        TruncatedGramSchmidt(window_size=3),
        ModifiedGramSchmidt(),
    ],
)
def test_empty_history_breakdown_matches_vector_device(
    strategy: PeriodicRestartOrthogonalization | TruncatedGramSchmidt | ModifiedGramSchmidt,
    torch_dtype: torch.dtype,
) -> None:
    """The empty-history early return must not hardcode a CPU breakdown flag.

    ``OrthogonalizationReport.breakdown``'s dataclass default is a bare
    ``torch.tensor(False)`` (always CPU, since no device was specified) —
    the empty-history branch in each strategy must construct its own
    device-matched flag explicitly instead of relying on that default,
    since the solver folds every iteration's flag into one device-side record
    (see ``conjugate_gradient.py::_record_first_breakdown``); a stray CPU
    flag mixed into a GPU solve would fail with a device-mismatch
    error that only surfaces on GPU, never on CPU-only test runs — which is
    exactly why this test uses the ``"meta"`` device rather than comparing
    against a real CPU vector: on CPU alone, a hardcoded
    ``torch.tensor(False)`` and a correctly device-matched one are both
    ``"cpu"`` and indistinguishable, so the bug this guards against would
    slip past a same-device comparison in a GPU-less environment.
    """
    vector = torch.ones(5, dtype=torch_dtype, device="meta")

    _, report = strategy.orthogonalize(vector, DirectionWindow.empty(vector))

    assert report.breakdown.device == vector.device


@pytest.fixture
def conjugate_window_inputs(
    torch_dtype: torch.dtype,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """A seeded vector plus six random ``(d_j, q_j = A d_j)`` pairs for an SPD ``A``."""
    generator = torch.Generator().manual_seed(3)
    n = 40
    factor = torch.randn(n, n, generator=generator, dtype=torch_dtype)
    matrix = factor @ factor.T / n + torch.eye(n, dtype=torch_dtype)
    directions = [torch.randn(n, generator=generator, dtype=torch_dtype) for _ in range(6)]
    products = [matrix @ d for d in directions]
    vector = torch.randn(n, generator=generator, dtype=torch_dtype)
    return vector, directions, products


def _per_pair_classical(
    vector: torch.Tensor, directions: list[torch.Tensor], products: list[torch.Tensor]
) -> torch.Tensor:
    """Previous implementation: sequential subtraction with per-pair stable dot products."""
    from torchalg.utils.numerics import stable_dot_product

    result = vector.clone()
    for d_j, q_j in zip(directions, products, strict=True):
        coeff = stable_dot_product(vector, q_j) / stable_dot_product(d_j, q_j)
        result = result - coeff * d_j
    return result


@pytest.mark.parametrize("window_size", [1, 3, 6])
def test_classical_orthogonalization_matches_per_pair_loop(
    conjugate_window_inputs: tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]],
    make_window: Callable[[Sequence[torch.Tensor], Sequence[torch.Tensor]], DirectionWindow],
    window_size: int,
) -> None:
    """One matrix-vector product per side equals the sequential sum to rounding."""
    vector, directions, products = conjugate_window_inputs
    strategy = TruncatedGramSchmidt(window_size=window_size)

    result, _ = strategy.orthogonalize(vector, make_window(directions, products))

    expected = _per_pair_classical(vector, directions[-window_size:], products[-window_size:])
    torch.testing.assert_close(result, expected, rtol=1e-13, atol=1e-13)


def test_overflowing_projections_fall_back_to_scaled_dot_products(
    make_window: Callable[[Sequence[torch.Tensor], Sequence[torch.Tensor]], DirectionWindow],
) -> None:
    """Projections past the float range are recomputed by the rescaling dot product."""
    vector = torch.tensor([1e200, 1e200, 0.0], dtype=torch.float64)
    d = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    q = torch.tensor([1e200, -1e200, 0.0], dtype=torch.float64)
    window = make_window((d,), (q,))
    assert not torch.isfinite(window.q @ vector).all()

    _, report = TruncatedGramSchmidt(window_size=1).orthogonalize(vector, window)

    assert torch.isfinite(report.coefficients).all()


def test_modified_gram_schmidt_is_sequential_and_uses_stored_curvature(
    conjugate_window_inputs: tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]],
    make_window: Callable[[Sequence[torch.Tensor], Sequence[torch.Tensor]], DirectionWindow],
) -> None:
    """MGS still takes each numerator from the already-updated vector."""
    from torchalg.utils.numerics import stable_dot_product

    vector, directions, products = conjugate_window_inputs
    expected = vector.clone()
    for d_j, q_j in zip(directions, products, strict=True):
        expected = expected - (stable_dot_product(expected, q_j) / torch.dot(d_j, q_j)) * d_j

    result, _ = ModifiedGramSchmidt().orthogonalize(vector, make_window(directions, products))

    torch.testing.assert_close(result, expected, rtol=1e-14, atol=1e-14)
