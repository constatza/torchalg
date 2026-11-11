"""Time to solution per preconditioner: setup, per-solve cost, and break-even against Jacobi.

Run: ``uv run python -m benchmarks.throughput.time_to_solution`` (CPU, CSR tree).
For each case and method: setup seconds, one PCG solve to ``RTOL`` (median), iteration
count, and the number of right-hand sides ``k*`` after which setup plus ``k`` solves beats
``k`` Jacobi-PCG solves: ``k* = setup / (t_jacobi - t_method)`` (``inf`` if the method's
solve is not faster).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import torch

from benchmarks.algorithms.registry import AMG_LEVELS, POD_RANK, make_snapshots
from benchmarks.cases import MatrixCase, box3d_case, lap2d_case
from benchmarks.harness import time_callable
from torchalg import pcg
from torchalg.multigrid import AMGPreconditioner, VCycle
from torchalg.preconditioners.base import MatrixSetupPreconditioner
from torchalg.sparse.preconditioners.amg.adaptive import AdaptiveSAPreconditioner
from torchalg.sparse.preconditioners.amg.bootstrap import BootstrapAMGPreconditioner
from torchalg.sparse.preconditioners.amg.coarse_solve import dense_lu_coarse_solver
from torchalg.sparse.preconditioners.amg.smoothers import resolve_jacobi_default
from torchalg.sparse.preconditioners.amg.variants import vcycle_amg
from torchalg.sparse.preconditioners.ic0 import IC0Preconditioner
from torchalg.sparse.preconditioners.ilu import ILUPreconditioner
from torchalg.sparse.preconditioners.jacobi import JacobiPreconditioner
from torchalg.sparse.preconditioners.pod.coarsening import PODCoarseningStrategy

CASES = (lap2d_case(10000), box3d_case(27000), lap2d_case(90000))
RTOL = 1e-8
MAX_ITERATIONS = 5000
REPEATS = 3
CPU = torch.device("cpu")
AMG_THETA = 0.02
SKIP_BOOTSTRAP_ABOVE_N = 30000


def _pod(matrix: torch.Tensor, case: MatrixCase) -> MatrixSetupPreconditioner:
    coarsening = PODCoarseningStrategy(rank=POD_RANK)
    coarsening.fit(make_snapshots(case, torch.float64, CPU))
    return AMGPreconditioner(
        matrix=matrix,
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


def _methods(
    matrix: torch.Tensor, case: MatrixCase
) -> dict[str, Callable[[], MatrixSetupPreconditioner]]:
    methods: dict[str, Callable[[], MatrixSetupPreconditioner]] = {
        "jacobi": JacobiPreconditioner,
        "ic0": IC0Preconditioner,
        "ilu": ILUPreconditioner,
        "amg": lambda: vcycle_amg(matrix, n_levels=AMG_LEVELS, theta=AMG_THETA),
        "adaptive_sa": AdaptiveSAPreconditioner,
        "pod2g": lambda: _pod(matrix, case),
    }
    if case.n <= SKIP_BOOTSTRAP_ABOVE_N:
        methods["bootstrap"] = BootstrapAMGPreconditioner
    return methods


def _run_case(case: MatrixCase) -> None:
    matrix = case.csr(torch.float64, CPU)
    rhs = case.rhs(torch.float64, CPU)
    rows: list[tuple[str, float, float, int]] = []
    for name, build in _methods(matrix, case).items():
        start = time.perf_counter()
        preconditioner = build().setup(matrix)
        setup_s = time.perf_counter() - start
        _, result = pcg(
            matrix,
            rhs,
            preconditioner=preconditioner,
            rtol=RTOL,
            maxiter=MAX_ITERATIONS,
            device=CPU,
        )
        solve_s = time_callable(
            lambda p=preconditioner: pcg(
                matrix, rhs, preconditioner=p, rtol=RTOL, maxiter=MAX_ITERATIONS, device=CPU
            ),
            device=CPU,
            repeats=REPEATS,
            warmups=0,
        )[0]
        rows.append((name, setup_s, solve_s, int(result.iterations)))
    jacobi_solve = rows[0][2]
    for name, setup_s, solve_s, iterations in rows:
        gain = jacobi_solve - solve_s
        k_star = setup_s / gain if gain > 0 else math.inf
        print(
            f"{case.id:14} {name:12} {setup_s:8.3f} {solve_s * 1e3:9.1f} {iterations:6d} "
            f"{k_star:7.1f}"
        )


def main() -> None:
    print(f"{'case':14} {'method':12} {'setup s':>8} {'solve ms':>9} {'iters':>6} {'k*':>7}")
    for case in CASES:
        _run_case(case)


if __name__ == "__main__":
    main()
