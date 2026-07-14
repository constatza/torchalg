"""Benchmark: VCycleAMG vs WCycleAMG convergence on larger Poisson problems.

Ported from the reference (``dl-experiments``'s
``tests/benchmarks/preconditioners/test_amg_variants.py``), adapted from
numpy to torch tensors and using the shared ``poisson_1d_factory`` fixture
(``tests/conftest.py``) instead of a locally duplicated ``_poisson_1d``
helper (the reference duplicates that helper between this file and
``tests/solver/preconditioners/implementations/test_amg.py``; this repo's
DRY convention consolidates it into one fixture instead - see
``tests/conftest.py::poisson_1d_factory``'s docstring).

Both convergence tests below require ``flexible_cg`` (``torchalg``),
which landed in Stage 9.

Theoretical Basis (Briggs, Henson & McCormick 2000, Section 3.3):
    W-cycle applies gamma = 2 coarse-grid corrections per level vs.
    V-cycle's gamma = 1. For SPD problems, W-cycle iteration counts are
    <= V-cycle counts at higher cost per cycle. Both should converge in
    O(1) iterations independent of n.

Expected Outcome:
    - Both variants converge on all problem sizes.
    - WCycleAMG iterations <= VCycleAMG iterations for the same problem.
    - Iteration counts remain bounded as n grows (mesh-independent convergence).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from torchalg.preconditioners.implementations.amg import VCycleAMG, WCycleAMG

if TYPE_CHECKING:
    from collections.abc import Callable

# (n, n_levels) - larger n needs more levels to stay mesh-independent
TEST_CASES = [
    (50, 3),
    (100, 3),
    (200, 4),
    (500, 4),
]

AMG_CLASSES = [
    pytest.param(VCycleAMG, id="VCycleAMG"),
    pytest.param(WCycleAMG, id="WCycleAMG"),
]


def _is_slow(n: int) -> bool:
    return n >= 500


@pytest.mark.benchmark
@pytest.mark.parametrize("n,n_levels", TEST_CASES)
@pytest.mark.parametrize("amg_class", AMG_CLASSES)
def test_amg_variant_converges(
    n: int,
    n_levels: int,
    amg_class: type,
    request: pytest.FixtureRequest,
    poisson_1d_factory: Callable[[int], torch.Tensor],
) -> None:
    """Both AMG variants must converge on 1D Poisson of size n.

    Args:
        n: Problem size.
        n_levels: Hierarchy depth.
        amg_class: VCycleAMG or WCycleAMG.
        request: Pytest fixture request.
        poisson_1d_factory: Size-parametrized Poisson matrix factory.
    """
    from torchalg import flexible_cg

    if _is_slow(n):
        request.applymarker(pytest.mark.slow)

    a = poisson_1d_factory(n)
    b = torch.ones(n, dtype=a.dtype)
    precond = amg_class(a, n_levels=n_levels)

    x, info = flexible_cg(a, b, preconditioner=precond, rtol=1e-8, maxiter=200)

    assert info.converged, (
        f"{amg_class.__name__}(n={n}, levels={n_levels}) did not converge "
        f"in {info.iterations} iterations"
    )
    torch.testing.assert_close(a @ x, b, rtol=1e-6, atol=1e-6)


@pytest.mark.benchmark
@pytest.mark.parametrize("n,n_levels", TEST_CASES)
def test_wcycle_iters_not_worse_than_vcycle(
    n: int,
    n_levels: int,
    request: pytest.FixtureRequest,
    poisson_1d_factory: Callable[[int], torch.Tensor],
) -> None:
    """WCycleAMG must need <= iterations than VCycleAMG for every problem size.

    Args:
        n: Problem size.
        n_levels: Hierarchy depth.
        request: Pytest fixture request.
        poisson_1d_factory: Size-parametrized Poisson matrix factory.
    """
    from torchalg import flexible_cg

    if _is_slow(n):
        request.applymarker(pytest.mark.slow)

    a = poisson_1d_factory(n)
    b = torch.ones(n, dtype=a.dtype)

    _, info_v = flexible_cg(
        a, b, preconditioner=VCycleAMG(a, n_levels=n_levels), rtol=1e-8, maxiter=200
    )
    _, info_w = flexible_cg(
        a, b, preconditioner=WCycleAMG(a, n_levels=n_levels), rtol=1e-8, maxiter=200
    )

    assert info_v.converged and info_w.converged, "At least one variant did not converge"
    assert info_w.iterations <= info_v.iterations, (
        f"WCycle ({info_w.iterations}) used more iterations than VCycle ({info_v.iterations}) "
        f"for n={n}"
    )
