"""Host reads of the tentative-prolongator fit (``torchalg.utils.tentative``).

Every adaptive-SA candidate build fits each level, so a host read inside the fit is paid
per level and per candidate. The fit sizes its outputs from three integers of the
aggregate (largest id, largest aggregate, isolated-node count) read together once; nothing
else in the dense fit forces a host synchronization on an accelerator (the CSR build adds
only its data-dependent filter of exact zeros, a boolean mask no dispatch counter sees).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch
from pyamg.aggregation.tentative import fit_candidates as pyamg_fit_candidates

from tests.support.dispatch_counter import DispatchCounter
from tests.support.pyamg_reference import aggregation_operator
from torchalg.utils.tentative import fit_candidates, fit_candidates_csr

if TYPE_CHECKING:
    from collections.abc import Callable

ATOL = 1e-11
DOFS_PER_NODE = 2
CANDIDATES = 3
SYNC_FORCING_OPS = frozenset(
    {
        "aten._local_scalar_dense.default",
        "aten.item.default",
        "aten.is_nonzero.default",
        "aten.nonzero.default",
        "aten.bincount.default",
    }
)
"""Dispatched ops that read a device value on the host or size their output from one."""


@pytest.fixture
def irregular_aggregate() -> torch.Tensor:
    """Interleaved aggregates of sizes 3, 1 and 4 plus two isolated nodes."""
    return torch.tensor([2, 0, -1, 2, 0, 1, 2, -1, 0, 2], dtype=torch.long)


@pytest.fixture
def isolated_aggregate() -> torch.Tensor:
    """Every node isolated."""
    return torch.full((4,), -1, dtype=torch.long)


@pytest.fixture
def block_candidates(torch_dtype: torch.dtype) -> Callable[[torch.Tensor], torch.Tensor]:
    """Seeded ``(DOFS_PER_NODE * n_nodes, CANDIDATES)`` candidates for an aggregate."""

    def build(aggregate: torch.Tensor) -> torch.Tensor:
        generator = torch.Generator().manual_seed(31)
        rows = aggregate.shape[0] * DOFS_PER_NODE
        return torch.randn(rows, CANDIDATES, generator=generator, dtype=torch_dtype)

    return build


@pytest.fixture
def tolist_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Counts ``Tensor.tolist`` calls (a host read that dispatches no ATen op)."""
    calls: list[int] = []
    original = torch.Tensor.tolist

    def counting(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        calls.append(self.numel())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "tolist", counting)
    return calls


def test_fit_matches_pyamg_on_irregular_aggregates(
    irregular_aggregate: torch.Tensor, block_candidates: Callable[[torch.Tensor], torch.Tensor]
) -> None:
    vectors = block_candidates(irregular_aggregate)
    q_ref, r_ref = pyamg_fit_candidates(aggregation_operator(irregular_aggregate), vectors.numpy())

    q, r = fit_candidates(irregular_aggregate, vectors)

    np.testing.assert_allclose(q.numpy(), q_ref.toarray(), atol=ATOL)
    np.testing.assert_allclose(r.numpy(), r_ref, atol=ATOL)


@pytest.mark.parametrize("fit", [fit_candidates, fit_candidates_csr], ids=["dense", "csr"])
def test_fit_reads_the_host_once(
    fit: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    irregular_aggregate: torch.Tensor,
    block_candidates: Callable[[torch.Tensor], torch.Tensor],
    tolist_calls: list[int],
) -> None:
    vectors = block_candidates(irregular_aggregate)

    with DispatchCounter() as counter:
        fit(irregular_aggregate, vectors)

    assert {name: counter.ops[name] for name in SYNC_FORCING_OPS if counter.ops[name]} == {}
    assert len(tolist_calls) == 1


def test_all_isolated_nodes_give_an_empty_fit(
    isolated_aggregate: torch.Tensor, block_candidates: Callable[[torch.Tensor], torch.Tensor]
) -> None:
    vectors = block_candidates(isolated_aggregate)

    q, r = fit_candidates(isolated_aggregate, vectors)

    assert q.shape == (vectors.shape[0], CANDIDATES)
    assert torch.count_nonzero(q) == 0
    assert torch.equal(r, torch.zeros(CANDIDATES, CANDIDATES, dtype=vectors.dtype))
