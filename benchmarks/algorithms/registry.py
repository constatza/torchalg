"""Registry of the preconditioners the algorithm benchmarks measure.

Each :class:`PreconditionerSpec` is one (method, tree) cell family: how to build a fresh,
not-yet-set-up object, which operand its ``setup()`` takes, the largest system it is run
on, and the PCG iteration cap its end-to-end correctness check allows. Timed ``setup``
means ``spec.fresh(problem).setup(operand)``: every cost a caller pays before the first
``apply()``, including constructor-time work such as ``vcycle_amg`` wiring or the POD
basis fit.

Size caps keep the whole suite within minutes on a CPU: dense IC(0)/ILU are masked
column sweeps and dense adaptive/bootstrap setups cost tens of seconds near n = 3000,
so they skip the n = 10000 case. Sparse variants run every case.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from math import ceil, sqrt

import torch
from scipy.sparse.linalg import splu

from benchmarks.cases import BENCHMARK_SEED, MatrixCase
from torchalg.multigrid import AMGPreconditioner, VCycle
from torchalg.preconditioners.base import MatrixSetupPreconditioner
from torchalg.preconditioners.implementations import (
    AdaptiveSAPreconditioner,
    BootstrapAMGPreconditioner,
    IC0Preconditioner,
    ICholeskyPreconditioner,
    ILUPreconditioner,
    JacobiPreconditioner,
    pod2g_preconditioner,
)
from torchalg.preconditioners.implementations._masked_factorization import dense_ic0
from torchalg.preconditioners.implementations.amg import vcycle_amg
from torchalg.preconditioners.implementations.amg.bootcmatch import BootCMatchPreconditioner
from torchalg.sparse.preconditioners.amg.adaptive import (
    AdaptiveSAPreconditioner as SparseAdaptiveSAPreconditioner,
)
from torchalg.sparse.preconditioners.amg.bootcmatch import (
    BootCMatchPreconditioner as SparseBootCMatchPreconditioner,
)
from torchalg.sparse.preconditioners.amg.bootstrap import (
    BootstrapAMGPreconditioner as SparseBootstrapAMGPreconditioner,
)
from torchalg.sparse.preconditioners.amg.coarse_solve import dense_lu_coarse_solver
from torchalg.sparse.preconditioners.amg.smoothers import resolve_jacobi_default
from torchalg.sparse.preconditioners.amg.variants import vcycle_amg as sparse_vcycle_amg
from torchalg.sparse.preconditioners.ic0 import IC0Preconditioner as SparseIC0Preconditioner
from torchalg.sparse.preconditioners.ic0 import sparse_ic0
from torchalg.sparse.preconditioners.icholesky import (
    ICholeskyPreconditioner as SparseICholeskyPreconditioner,
)
from torchalg.sparse.preconditioners.ilu import ILUPreconditioner as SparseILUPreconditioner
from torchalg.sparse.preconditioners.jacobi import (
    JacobiPreconditioner as SparseJacobiPreconditioner,
)
from torchalg.sparse.preconditioners.pod.coarsening import (
    PODCoarseningStrategy as SparsePODCoarseningStrategy,
)

ALGORITHM_SEED = 0
"""Seed handed to every randomized preconditioner so hierarchies are reproducible."""

AMG_LEVELS = 3
"""Aggregation-AMG depth, as in ``benchmarks/sparse_dense/scaling.py``."""

AMG_THETA = 0.02
"""Strength threshold of the aggregation AMG. The ``box3d`` stencil couples each neighbour at
1/26.5 of the diagonal, so the default 0.25 marks every connection weak and leaves nothing to
aggregate (see the report on the empty-aggregate crash)."""

BOOTCMATCH_MAX_LEVELS = 5
"""BootCMatch depth cap, as in ``benchmarks/sparse_dense/scaling.py``."""

BOOTCMATCH_CAP = 400
"""BootCMatch needs >80 PCG iterations on the n=10000 Laplacian (5 levels leave a large
coarsest grid), far more than the other AMG variants."""

POD_RANK = 16
POD_SNAPSHOTS = 32
"""POD-2G fit data: ``POD_SNAPSHOTS`` solves of ``A x = b`` for seeded random ``b``,
truncated to ``POD_RANK`` modes (two grids, so a 16-dimensional coarse space)."""

SOLVE_RTOL = 1e-8
"""PCG tolerance of the ``solve`` phase."""

MAX_TRUE_RELATIVE_RESIDUAL = 1e-6
"""Bound on ``||b - A x|| / ||b||`` recomputed from the returned solution."""

NO_SIZE_CAP = 10**9


class Tree(StrEnum):
    """Which implementation tree a spec belongs to; doubles as the ``format`` parameter."""

    DENSE = "dense"
    CSR = "csr"


@dataclass(frozen=True, slots=True)
class Problem:
    """One benchmark system in the format its tree expects, plus POD fit data."""

    case: MatrixCase
    matrix: torch.Tensor
    rhs: torch.Tensor
    snapshots: torch.Tensor


def make_snapshots(case: MatrixCase, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Solve ``A x_i = b_i`` for seeded random ``b_i``; rows are the POD snapshot ensemble."""
    generator = torch.Generator().manual_seed(BENCHMARK_SEED + 1)
    right_hand_sides = torch.randn(POD_SNAPSHOTS, case.n, dtype=torch.float64, generator=generator)
    solutions = splu(case.scipy_matrix.tocsc()).solve(right_hand_sides.numpy().T).T
    return torch.from_numpy(solutions.copy()).to(dtype=dtype, device=device)


@dataclass(frozen=True, slots=True)
class PreconditionerSpec:
    """How to build, size-limit and correctness-bound one preconditioner in one tree.

    ``fresh`` returns a constructed, not-yet-set-up object; ``operand`` produces the tensor
    its ``setup()`` takes (the system matrix, except ICholesky which takes a factor and is
    built once outside the timed region); ``max_iterations`` is the generous PCG cap the
    solve check asserts against.
    """

    name: str
    tree: Tree
    fresh: Callable[[Problem], MatrixSetupPreconditioner]
    max_iterations: Callable[[int], int]
    max_n: int = NO_SIZE_CAP
    cap_reason: str = ""
    operand: Callable[[Problem], torch.Tensor] = lambda problem: problem.matrix
    setup_repeats: int = 3
    setup_warmups: int = 1

    @property
    def id(self) -> str:
        """Test id: method and tree."""
        return f"{self.name}-{self.tree}"


def _sqrt_cap(factor: float) -> Callable[[int], int]:
    return lambda n: ceil(factor * sqrt(n))


def _fixed_cap(iterations: int) -> Callable[[int], int]:
    return lambda _n: iterations


def _pod_amg(problem: Problem) -> AMGPreconditioner:
    coarsening = SparsePODCoarseningStrategy(rank=POD_RANK)
    coarsening.fit(problem.snapshots)
    return AMGPreconditioner(
        matrix=problem.matrix,
        coarsening=coarsening,
        cycle=VCycle(
            resolve_jacobi_default(None, None),
            n_pre=2,
            n_post=2,
            coarse_solver_factory=dense_lu_coarse_solver,
        ),
        n_levels=2,
        linear=True,
    )


DENSE_FACTOR_CAP = 2744
DENSE_FACTOR_REASON = "dense masked factorization is O(n^3) work; capped below n=10000"
DENSE_AMG_CAP = 2744
DENSE_AMG_REASON = (
    "dense Galerkin/transfer products are O(n^2)+ memory and time; capped below n=10000"
)
DENSE_BOOTSTRAP_CAP = 2744
DENSE_BOOTSTRAP_REASON = "dense bootstrap/adaptive setup takes tens of seconds at n~3000"

# Iteration caps are generous multiples of the counts first observed; a method needing
# more than its cap has regressed, not merely slowed.
REGISTRY: tuple[PreconditionerSpec, ...] = (
    PreconditionerSpec(
        "jacobi",
        Tree.DENSE,
        lambda _p: JacobiPreconditioner(),
        _sqrt_cap(12),
        setup_repeats=5,
    ),
    PreconditionerSpec(
        "jacobi",
        Tree.CSR,
        lambda _p: SparseJacobiPreconditioner(),
        _sqrt_cap(12),
        setup_repeats=5,
    ),
    PreconditionerSpec(
        "ic0",
        Tree.DENSE,
        lambda _p: IC0Preconditioner(),
        _sqrt_cap(6),
        max_n=DENSE_FACTOR_CAP,
        cap_reason=DENSE_FACTOR_REASON,
        setup_repeats=1,
        setup_warmups=0,
    ),
    PreconditionerSpec(
        "ic0",
        Tree.CSR,
        lambda _p: SparseIC0Preconditioner(),
        _sqrt_cap(6),
    ),
    PreconditionerSpec(
        "icholesky",
        Tree.DENSE,
        lambda _p: ICholeskyPreconditioner(),
        _sqrt_cap(6),
        operand=lambda p: dense_ic0(p.matrix, 0.0),
        max_n=DENSE_FACTOR_CAP,
        cap_reason=DENSE_FACTOR_REASON,
        setup_repeats=5,
    ),
    PreconditionerSpec(
        "icholesky",
        Tree.CSR,
        lambda _p: SparseICholeskyPreconditioner(),
        _sqrt_cap(6),
        operand=lambda p: sparse_ic0(p.matrix),
        setup_repeats=5,
    ),
    PreconditionerSpec(
        "ilu",
        Tree.DENSE,
        lambda _p: ILUPreconditioner(),
        _sqrt_cap(6),
        max_n=DENSE_FACTOR_CAP,
        cap_reason=DENSE_FACTOR_REASON,
        setup_repeats=1,
        setup_warmups=0,
    ),
    PreconditionerSpec(
        "ilu",
        Tree.CSR,
        lambda _p: SparseILUPreconditioner(),
        _sqrt_cap(6),
    ),
    PreconditionerSpec(
        "amg",
        Tree.DENSE,
        lambda p: vcycle_amg(p.matrix, n_levels=AMG_LEVELS, theta=AMG_THETA),
        _fixed_cap(60),
        max_n=DENSE_AMG_CAP,
        cap_reason=DENSE_AMG_REASON,
    ),
    PreconditionerSpec(
        "amg",
        Tree.CSR,
        lambda p: sparse_vcycle_amg(p.matrix, n_levels=AMG_LEVELS, theta=AMG_THETA),
        _fixed_cap(60),
    ),
    PreconditionerSpec(
        "adaptive_sa",
        Tree.DENSE,
        lambda _p: AdaptiveSAPreconditioner(seed=ALGORITHM_SEED),
        _fixed_cap(80),
        max_n=DENSE_BOOTSTRAP_CAP,
        cap_reason=DENSE_BOOTSTRAP_REASON,
        setup_repeats=1,
        setup_warmups=0,
    ),
    PreconditionerSpec(
        "adaptive_sa",
        Tree.CSR,
        lambda _p: SparseAdaptiveSAPreconditioner(seed=ALGORITHM_SEED),
        _fixed_cap(80),
        setup_repeats=1,
        setup_warmups=0,
    ),
    PreconditionerSpec(
        "bootstrap",
        Tree.DENSE,
        lambda _p: BootstrapAMGPreconditioner(seed=ALGORITHM_SEED),
        _fixed_cap(150),
        max_n=DENSE_BOOTSTRAP_CAP,
        cap_reason=DENSE_BOOTSTRAP_REASON,
        setup_repeats=1,
        setup_warmups=0,
    ),
    PreconditionerSpec(
        "bootstrap",
        Tree.CSR,
        lambda _p: SparseBootstrapAMGPreconditioner(seed=ALGORITHM_SEED),
        _fixed_cap(150),
        setup_repeats=1,
        setup_warmups=0,
    ),
    PreconditionerSpec(
        "bootcmatch",
        Tree.DENSE,
        lambda _p: BootCMatchPreconditioner(seed=ALGORITHM_SEED, max_levels=BOOTCMATCH_MAX_LEVELS),
        _fixed_cap(BOOTCMATCH_CAP),
        max_n=DENSE_AMG_CAP,
        cap_reason=DENSE_AMG_REASON,
        setup_repeats=2,
        setup_warmups=0,
    ),
    PreconditionerSpec(
        "bootcmatch",
        Tree.CSR,
        lambda _p: SparseBootCMatchPreconditioner(
            seed=ALGORITHM_SEED, max_levels=BOOTCMATCH_MAX_LEVELS
        ),
        _fixed_cap(BOOTCMATCH_CAP),
        setup_repeats=2,
        setup_warmups=0,
    ),
    PreconditionerSpec(
        "pod2g",
        Tree.DENSE,
        lambda p: pod2g_preconditioner(p.matrix, p.snapshots, rank=POD_RANK),
        _sqrt_cap(12),
        max_n=DENSE_AMG_CAP,
        cap_reason=DENSE_AMG_REASON,
    ),
    PreconditionerSpec(
        "pod2g",
        Tree.CSR,
        _pod_amg,
        _sqrt_cap(12),
    ),
)
