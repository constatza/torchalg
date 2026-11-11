"""Stage-scoped reuse inside the dense Bootstrap AMG setup."""

from __future__ import annotations

import pytest
import torch

from torchalg.preconditioners.implementations.amg import (
    _algebraic_distance,
    _graph,
    bootstrap,
)
from torchalg.preconditioners.implementations.amg.bootstrap import BootstrapAMGPreconditioner


@pytest.fixture
def neighborhood_requests(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """``(id(matrix), depth)`` of every ``depth_neighborhood`` evaluation, in call order."""
    requests: list[tuple[int, int]] = []
    original = _graph.depth_neighborhood

    def recording(matrix: torch.Tensor, depth: int) -> torch.Tensor:
        requests.append((id(matrix), depth))
        return original(matrix, depth)

    for module in (_graph, _algebraic_distance, bootstrap):
        monkeypatch.setattr(module, "depth_neighborhood", recording)
    return requests


def test_setup_evaluates_each_neighborhood_once(
    poisson_1d_factory,
    neighborhood_requests: list[tuple[int, int]],
) -> None:
    """The finest matrix is revisited by every bootstrap pass but its neighborhoods never change."""
    matrix = poisson_1d_factory(64)

    BootstrapAMGPreconditioner(
        k_r=6, eta=2, n_bootstrap_cycles=2, nu=3, caliber=3, max_levels=3, max_coarse=4, seed=5
    ).setup(matrix)

    assert neighborhood_requests
    assert len(set(neighborhood_requests)) == len(neighborhood_requests)
