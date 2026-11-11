"""Primitive: AMG setup kernels, dense versus sparse where both exist.

Covers neighborhoods, algebraic distance, strength, aggregation, LS interpolation and
BootCMatch matching.

Dense variants materialize ``n × n`` temporaries and are skipped above ``DENSE_N_CAP``.
"""

from __future__ import annotations

import pytest
import torch

from benchmarks.cases import MatrixCase
from benchmarks.conftest import Record
from benchmarks.harness import Level
from benchmarks.primitives._support import (
    CURRENT_VARIANT,
    DENSE_N_CAP,
    STRENGTH_THETA,
    case_params,
)
from benchmarks.primitives.conftest import (
    LS_CALIBER,
    LS_MAX_CANDIDATES,
    LS_ROWS,
    LsProblem,
    LsSelection,
)
from torchalg.preconditioners.implementations.amg._aggregation import (
    standard_aggregation,
    strength_of_connection,
)
from torchalg.preconditioners.implementations.amg._algebraic_distance import (
    algebraic_distance,
    strength_graph,
)
from torchalg.preconditioners.implementations.amg._graph import depth_neighborhood
from torchalg.sparse.kernels.algebraic_distance import sparse_algebraic_distance
from torchalg.sparse.kernels.bootcmatch_matching import bootcmatch_parallel_matching
from torchalg.sparse.kernels.bootcmatch_weights import bootcmatch_edge_weights
from torchalg.sparse.kernels.depth_neighborhood import sparse_depth_neighborhood
from torchalg.sparse.kernels.strength import (
    sparse_strength_graph,
    sparse_strength_of_connection,
)
from torchalg.sparse.preconditioners.amg.aggregation import sparse_standard_aggregation
from torchalg.utils.ls_interpolation import (
    batched_ls_interpolation_rows,
    batched_select_interpolatory_set,
)

DEPTHS = [1, 3]
KERNEL_REPEATS = 3
LS_REPEATS = 3
FIT_TOLERANCE = 1e-9
"""Slack for the "LS fit never worse than the empty set" check in float64."""


def _assert_same_pattern(sparse: torch.Tensor, dense: torch.Tensor) -> None:
    assert torch.equal(sparse.to_dense().to(dense.dtype), dense)


# --- depth neighborhood ----------------------------------------------------------------


@pytest.mark.parametrize("depth", DEPTHS)
def test_dense_depth_neighborhood(
    depth: int, case: MatrixCase, capped_dense_matrix: torch.Tensor, record: Record
) -> None:
    record(
        Level.PRIMITIVE,
        "amg_setup.depth_neighborhood",
        case_params(case, format="dense", depth=depth),
        lambda: depth_neighborhood(capped_dense_matrix, depth),
        repeats=KERNEL_REPEATS,
    )


@pytest.mark.parametrize("depth", DEPTHS)
def test_sparse_depth_neighborhood(
    depth: int, case: MatrixCase, csr_matrix_: torch.Tensor, record: Record
) -> None:
    neighborhood = sparse_depth_neighborhood(csr_matrix_, depth)
    if case.n <= DENSE_N_CAP:
        reference = depth_neighborhood(csr_matrix_.to_dense(), depth)
        assert torch.equal(neighborhood.to_dense() != 0, reference)
    record(
        Level.PRIMITIVE,
        "amg_setup.depth_neighborhood",
        case_params(case, format="csr", depth=depth),
        lambda: sparse_depth_neighborhood(csr_matrix_, depth),
        repeats=KERNEL_REPEATS,
    )


# --- algebraic distance and strength graph ---------------------------------------------


def test_dense_algebraic_distance(
    case: MatrixCase,
    capped_dense_matrix: torch.Tensor,
    test_vectors: torch.Tensor,
    record: Record,
) -> None:
    record(
        Level.PRIMITIVE,
        "amg_setup.algebraic_distance",
        case_params(case, format="dense", depth=1),
        lambda: algebraic_distance(test_vectors, capped_dense_matrix, 1),
        repeats=KERNEL_REPEATS,
    )


def test_sparse_algebraic_distance(
    case: MatrixCase, csr_matrix_: torch.Tensor, test_vectors: torch.Tensor, record: Record
) -> None:
    distance = sparse_algebraic_distance(test_vectors, csr_matrix_, 1)
    if case.n <= DENSE_N_CAP:
        reference = algebraic_distance(test_vectors, csr_matrix_.to_dense(), 1)
        torch.testing.assert_close(distance.to_dense(), reference)
    record(
        Level.PRIMITIVE,
        "amg_setup.algebraic_distance",
        case_params(case, format="csr", depth=1),
        lambda: sparse_algebraic_distance(test_vectors, csr_matrix_, 1),
        repeats=KERNEL_REPEATS,
    )


def test_dense_strength_graph(
    case: MatrixCase, distance_dense: torch.Tensor, fine_mask: torch.Tensor, record: Record
) -> None:
    record(
        Level.PRIMITIVE,
        "amg_setup.strength_graph",
        case_params(case, format="dense"),
        lambda: strength_graph(distance_dense, fine_mask),
    )


def test_sparse_strength_graph(
    case: MatrixCase, distance_csr: torch.Tensor, fine_mask: torch.Tensor, record: Record
) -> None:
    graph = sparse_strength_graph(distance_csr, fine_mask)
    if case.n <= DENSE_N_CAP:
        reference = strength_graph(distance_csr.to_dense(), fine_mask)
        _assert_same_pattern(graph, reference)
    record(
        Level.PRIMITIVE,
        "amg_setup.strength_graph",
        case_params(case, format="csr"),
        lambda: sparse_strength_graph(distance_csr, fine_mask),
    )


# --- strength of connection and standard aggregation -----------------------------------


def test_dense_strength_of_connection(
    case: MatrixCase, capped_dense_matrix: torch.Tensor, record: Record
) -> None:
    record(
        Level.PRIMITIVE,
        "amg_setup.strength_of_connection",
        case_params(case, format="dense"),
        lambda: strength_of_connection(capped_dense_matrix, STRENGTH_THETA),
    )


def test_sparse_strength_of_connection(
    case: MatrixCase, csr_matrix_: torch.Tensor, record: Record
) -> None:
    strength = sparse_strength_of_connection(csr_matrix_, STRENGTH_THETA)
    if case.n <= DENSE_N_CAP:
        reference = strength_of_connection(csr_matrix_.to_dense(), STRENGTH_THETA)
        _assert_same_pattern(strength, reference)
    record(
        Level.PRIMITIVE,
        "amg_setup.strength_of_connection",
        case_params(case, format="csr"),
        lambda: sparse_strength_of_connection(csr_matrix_, STRENGTH_THETA),
    )


def test_dense_standard_aggregation(
    case: MatrixCase, strength_dense: torch.Tensor, record: Record
) -> None:
    aggregates = standard_aggregation(strength_dense)
    assert int(aggregates.max()) + 1 < case.n
    record(
        Level.PRIMITIVE,
        "amg_setup.standard_aggregation",
        case_params(case, format="dense"),
        lambda: standard_aggregation(strength_dense),
        repeats=KERNEL_REPEATS,
    )


def test_sparse_standard_aggregation(
    case: MatrixCase, strength_csr: torch.Tensor, record: Record
) -> None:
    aggregates = sparse_standard_aggregation(strength_csr)
    if case.n <= DENSE_N_CAP:
        reference = standard_aggregation(strength_csr.to_dense().to(torch.bool))
        assert torch.equal(aggregates, reference)
    record(
        Level.PRIMITIVE,
        "amg_setup.standard_aggregation",
        case_params(case, format="csr"),
        lambda: sparse_standard_aggregation(strength_csr),
        repeats=KERNEL_REPEATS,
    )


# --- LS interpolation (bootstrap AMG selection and fit) --------------------------------


def _ls_params(**extra: str | float) -> dict[str, str | int | float]:
    return {
        "m": LS_ROWS,
        "max_candidates": LS_MAX_CANDIDATES,
        "caliber": LS_CALIBER,
        "variant": CURRENT_VARIANT,
        **extra,
    }


def test_ls_select_interpolatory_set(ls_problem: LsProblem, record: Record) -> None:
    args = (
        ls_problem.padded_candidates,
        ls_problem.candidate_mask,
        ls_problem.test_vectors,
        ls_problem.targets,
        ls_problem.weights,
        LS_CALIBER,
    )
    chosen, chosen_mask = batched_select_interpolatory_set(*args)
    assert chosen_mask.any()
    # Every accepted slot is drawn from that row's own eligible candidates.
    picked = chosen.unsqueeze(-1) == ls_problem.padded_candidates.unsqueeze(1)
    eligible = (picked & ls_problem.candidate_mask.unsqueeze(1)).any(dim=-1)
    assert bool(eligible[chosen_mask].all())
    record(
        Level.PRIMITIVE,
        "amg_setup.ls_select",
        _ls_params(),
        lambda: batched_select_interpolatory_set(*args),
        repeats=LS_REPEATS,
    )


def test_ls_interpolation_rows(
    ls_problem: LsProblem, ls_selection: LsSelection, record: Record
) -> None:
    args = (
        ls_problem.test_vectors,
        ls_problem.targets,
        ls_selection.chosen,
        ls_selection.chosen_mask,
        ls_problem.weights,
    )
    rows = batched_ls_interpolation_rows(*args)
    assert torch.equal(
        rows[~ls_selection.chosen_mask], torch.zeros_like(rows[~ls_selection.chosen_mask])
    )
    target = ls_problem.test_vectors[ls_problem.targets]
    fitted = torch.einsum("mc,mck->mk", rows, ls_problem.test_vectors[ls_selection.chosen])
    empty_energy = (ls_problem.weights * target**2).sum(dim=-1)
    fit_energy = (ls_problem.weights * (target - fitted) ** 2).sum(dim=-1)
    assert bool((fit_energy <= empty_energy + FIT_TOLERANCE).all())
    record(
        Level.PRIMITIVE,
        "amg_setup.ls_fit",
        _ls_params(),
        lambda: batched_ls_interpolation_rows(*args),
        repeats=LS_REPEATS,
    )


# --- BootCMatch weights and matching ---------------------------------------------------


def test_bootcmatch_edge_weights(
    case: MatrixCase, csr_matrix_: torch.Tensor, smooth_vector: torch.Tensor, record: Record
) -> None:
    weights, fine_only = bootcmatch_edge_weights(smooth_vector, csr_matrix_)
    assert weights.shape == csr_matrix_.shape
    assert fine_only.shape == (case.n,)
    record(
        Level.PRIMITIVE,
        "amg_setup.bootcmatch_weights",
        case_params(case, format="csr"),
        lambda: bootcmatch_edge_weights(smooth_vector, csr_matrix_),
    )


def test_bootcmatch_matching(
    case: MatrixCase, csr_matrix_: torch.Tensor, smooth_vector: torch.Tensor, record: Record
) -> None:
    weights, fine_only = bootcmatch_edge_weights(smooth_vector, csr_matrix_)
    match_of = bootcmatch_parallel_matching(weights, fine_only)
    matched = match_of >= 0
    assert bool(matched.any())
    assert torch.equal(
        match_of[match_of[matched]], torch.arange(case.n, device=match_of.device)[matched]
    )
    record(
        Level.PRIMITIVE,
        "amg_setup.bootcmatch_matching",
        case_params(case, format="csr"),
        lambda: bootcmatch_parallel_matching(weights, fine_only),
        repeats=KERNEL_REPEATS,
    )
