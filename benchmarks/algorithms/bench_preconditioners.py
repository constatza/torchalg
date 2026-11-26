"""Algorithm level: every preconditioner timed as separate setup, apply and solve phases.

Per (preconditioner, matrix) cell, recorded as ``<name>.<phase>``:

- ``setup``: ``spec.fresh(problem).setup(operand)`` on a fresh object each repeat.
- ``first_apply``: the first ``apply()`` after ``setup()`` on a fresh object each repeat;
  its gap to ``apply`` exposes lazy one-off work (spectral estimates, factorizations,
  schedules) that ``setup()`` should have done.
- ``apply``: steady-state ``apply()`` after warm-up.
- ``solve``: ``pcg`` to ``SOLVE_RTOL`` with the set-up preconditioner. This is also the
  end-to-end correctness check: it must converge within the spec's iteration cap and the
  recomputed relative residual must be small. Time is never asserted.
- ``solve_k``: one fresh ``setup()`` followed by ``pcg`` on each of ``SOLVE_K`` distinct
  right-hand sides, the amortized cost of a matrix shared by ``k`` solves (``k`` is in the
  params). Every solve must converge within the iteration cap.
"""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import pytest
import torch

from benchmarks.algorithms.registry import (
    MAX_TRUE_RELATIVE_RESIDUAL,
    SOLVE_K,
    SOLVE_RTOL,
    PreconditionerSpec,
    Problem,
)
from benchmarks.harness import Level, Params
from torchalg import pcg
from torchalg.models import SolverResult
from torchalg.preconditioners.base import MatrixSetupPreconditioner

APPLY_REPEATS = 10
APPLY_WARMUPS = 2
SOLVE_REPEATS = 3
SLOW_SOLVE_S = 5.0
"""A first solve slower than this is recorded as a single run (it is already the warm-up)."""


def _relative_residual(problem: Problem, solution: torch.Tensor) -> float:
    norm = torch.linalg.vector_norm
    return float(norm(problem.rhs - problem.matrix @ solution) / norm(problem.rhs))


def _cuda_peak(device: torch.device) -> dict[str, int]:
    return {"cuda_peak_bytes": torch.cuda.max_memory_allocated(device)}


@pytest.mark.benchmark
def test_preconditioner(
    spec: PreconditionerSpec, problem: Problem, device, record, result_sink
) -> None:
    """Time the four phases of one cell, then assert the end-to-end solve is correct."""
    params: Params = {
        "n": problem.case.n,
        "nnz": problem.case.nnz,
        "format": str(spec.tree),
        "matrix": problem.case.id,
    }
    operand = spec.operand(problem)
    built: list[MatrixSetupPreconditioner] = []

    def setup() -> None:
        preconditioner = spec.fresh(problem)
        preconditioner.setup(operand)
        built.append(preconditioner)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    record(
        Level.ALGORITHM,
        f"{spec.name}.setup",
        params,
        setup,
        repeats=spec.setup_repeats,
        warmups=spec.setup_warmups,
    )
    if device.type == "cuda":
        # Peak is only known after the timed runs, so patch it onto the stored measurement.
        sink = result_sink.measurements
        sink[-1] = replace(sink[-1], extra=_cuda_peak(device))

    # Every object built above is untouched, so each repeat's first apply is a true first use.
    fresh_objects = list(built)
    used: list[MatrixSetupPreconditioner] = []

    def first_apply() -> None:
        preconditioner = fresh_objects.pop()
        preconditioner.apply(problem.rhs)
        used.append(preconditioner)

    record(
        Level.ALGORITHM,
        f"{spec.name}.first_apply",
        params,
        first_apply,
        repeats=len(fresh_objects),
        warmups=0,
    )
    prepared = used[0]

    record(
        Level.ALGORITHM,
        f"{spec.name}.apply",
        params,
        lambda: prepared.apply(problem.rhs),
        repeats=APPLY_REPEATS,
        warmups=APPLY_WARMUPS,
    )

    iteration_cap = spec.max_iterations(problem.case.n)

    def solve() -> tuple[torch.Tensor, SolverResult]:
        return pcg(
            problem.matrix,
            problem.rhs,
            preconditioner=prepared,
            rtol=SOLVE_RTOL,
            maxiter=iteration_cap,
        )

    start = perf_counter()
    solution, result = solve()
    first_solve_s = perf_counter() - start
    repeats, warmups = (1, 0) if first_solve_s > SLOW_SOLVE_S else (SOLVE_REPEATS, 0)
    true_residual = _relative_residual(problem, solution)
    record(
        Level.ALGORITHM,
        f"{spec.name}.solve",
        params,
        solve,
        repeats=repeats,
        warmups=warmups,
        extra={
            "iterations": result.iterations,
            "residual_true": true_residual,
            "residual_reported": result.residual,
            "iteration_cap": iteration_cap,
            "converged": int(result.converged),
        },
    )

    def solve_k() -> list[SolverResult]:
        preconditioner = spec.fresh(problem)
        preconditioner.setup(operand)
        return [
            pcg(
                problem.matrix,
                b,
                preconditioner=preconditioner,
                rtol=SOLVE_RTOL,
                maxiter=iteration_cap,
            )[1]
            for b in problem.rhs_block
        ]

    start = perf_counter()
    k_results = solve_k()
    first_solve_k_s = perf_counter() - start
    # The setup repeat count already reflects what the cell's setup cost affords.
    repeats = 1 if first_solve_k_s > SLOW_SOLVE_S else spec.setup_repeats
    record(
        Level.ALGORITHM,
        f"{spec.name}.solve_k",
        {**params, "k": SOLVE_K},
        solve_k,
        repeats=repeats,
        warmups=0,
        extra={"max_iterations": max(r.iterations for r in k_results)},
    )

    assert result.converged, f"{spec.id} did not converge within {iteration_cap} iterations"
    assert result.iterations <= iteration_cap
    assert true_residual <= MAX_TRUE_RELATIVE_RESIDUAL
    assert all(r.converged for r in k_results), f"{spec.id} solve_k exceeded {iteration_cap}"
