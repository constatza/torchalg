"""Tests for ``torchalg.strategies.orthogonalization``."""

from __future__ import annotations

import pytest
import torch

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
) -> None:
    """Truncated Gram-Schmidt enforces d_i^T A d_j = 0 for the stored direction."""
    result, report = TruncatedGramSchmidt(window_size=2).orthogonalize(
        vector=orthogonalization_probe,
        d_vectors=(previous_direction,),
        q_vectors=(previous_matrix_product,),
    )

    assert torch.allclose(result, torch.tensor([0.0, 1.0], dtype=result.dtype))
    assert len(report.coefficients) == 1
    assert report.coefficients[0].item() == 2.0
    assert bool(report.breakdown) is False


def test_modified_gram_schmidt_uses_all_history(
    orthogonalization_probe: torch.Tensor,
    previous_direction: torch.Tensor,
    previous_matrix_product: torch.Tensor,
) -> None:
    """Modified Gram-Schmidt advertises unlimited history and orthogonalizes."""
    strategy = ModifiedGramSchmidt()
    result, report = strategy.orthogonalize(
        vector=orthogonalization_probe,
        d_vectors=(previous_direction,),
        q_vectors=(previous_matrix_product,),
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
    periodic_restart_history: tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]],
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
    d_vectors, q_vectors = periodic_restart_history
    strategy = PeriodicRestartOrthogonalization(m_max=3.0)
    probe = torch.ones(5, dtype=torch_dtype)

    window_sizes = [
        len(strategy.orthogonalize(probe, d_vectors, q_vectors)[1].coefficients) for _ in range(8)
    ]

    # Coefficients are now stored as 0-d tensors, but still indexable by length
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
    since ``CGState.breakdown_history`` later stacks every iteration's flag
    together (see ``conjugate_gradient.py::_build_result``); a stray CPU
    entry mixed into a GPU solve's stack would fail with a device-mismatch
    error that only surfaces on GPU, never on CPU-only test runs — which is
    exactly why this test uses the ``"meta"`` device rather than comparing
    against a real CPU vector: on CPU alone, a hardcoded
    ``torch.tensor(False)`` and a correctly device-matched one are both
    ``"cpu"`` and indistinguishable, so the bug this guards against would
    slip past a same-device comparison in a GPU-less environment.
    """
    vector = torch.ones(5, dtype=torch_dtype, device="meta")

    _, report = strategy.orthogonalize(vector, d_vectors=(), q_vectors=())

    assert report.breakdown.device == vector.device
