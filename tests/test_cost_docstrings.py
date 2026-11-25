"""Every symbol whose cost drives a placement or algorithm choice documents it.

The convention puts a ``Cost:`` section before ``Args:`` in the docstring, stating flops,
bytes moved, depth, kernel launches and host synchronisations as formulas; this test keeps
the documented symbols from
losing it.
"""

from __future__ import annotations

import importlib
from functools import reduce

import pytest

_AMG = "torchalg.sparse.preconditioners.amg"

COST_DOCUMENTED = [
    ("torchalg.conjugate_gradient", "PCGSolver"),
    ("torchalg.conjugate_gradient", "ConjugateGradientSolver._iterate_step"),
    ("torchalg.utils.numerics", "stable_dot_product"),
    ("torchalg.preconditioners.base", "Identity"),
    ("torchalg.sparse.preconditioners.jacobi", "JacobiPreconditioner"),
    (f"{_AMG}._gauss_seidel", "PreparedSymmetricGaussSeidel"),
    (f"{_AMG}._compatible_relaxation", "_independent_set_of"),
    (f"{_AMG}._compatible_relaxation", "compatible_relaxation_coarsening"),
    (f"{_AMG}.aggregation", "sparse_standard_aggregation"),
    (f"{_AMG}.coarsening", "AggregationCoarsening.build_transfer"),
    (f"{_AMG}.adaptive", "adaptive_sa_hierarchy"),
    (f"{_AMG}.adaptive", "AdaptiveSAPreconditioner"),
    (f"{_AMG}.bootstrap", "BAMGCoarsening.build_transfer"),
    (f"{_AMG}.bootstrap", "BootstrapAMGPreconditioner"),
    (f"{_AMG}.transfer", "SparseTransferOperator"),
    (f"{_AMG}._jacobi_omega", "jacobi_omega"),
    ("torchalg.sparse.kernels.triangular", "level_schedule"),
    ("torchalg.sparse.kernels.triangular", "PreparedTriangularSolve"),
    ("torchalg.sparse.kernels.triangular", "LibraryTriangularSolve"),
    ("torchalg.sparse.kernels.galerkin", "form_sparse_sparse"),
    ("torchalg.sparse.kernels.depth_neighborhood", "sparse_depth_neighborhood"),
    ("torchalg.sparse.preconditioners.ic0", "IC0Preconditioner"),
    ("torchalg.sparse.preconditioners.ilu", "ILUPreconditioner"),
    ("torchalg.multigrid.coarse_solve", "LUCoarseSolve"),
    ("torchalg.multigrid.coarse_solve", "PseudoInverseCoarseSolve"),
    ("torchalg.multigrid.coarse_solve", "lu_coarse_solver"),
    ("torchalg.multigrid.coarse_solve", "pseudo_inverse_coarse_solver"),
    ("torchalg.utils.spectral", "approximate_spectral_radius"),
    ("torchalg.utils.ls_interpolation", "batched_select_interpolatory_set"),
    ("torchalg.utils.pod_basis", "compute_pod_basis"),
    ("torchalg.utils.pod_basis", "_gram_basis"),
    ("torchalg.preconditioners.implementations.pod.coarsening", "PODCoarseningStrategy"),
    (
        "torchalg.preconditioners.implementations.pod.coarsening",
        "PODCoarseningStrategy.build_transfer",
    ),
]


@pytest.mark.parametrize(("module_path", "qualname"), COST_DOCUMENTED)
def test_docstring_has_cost_section(module_path: str, qualname: str) -> None:
    module = importlib.import_module(module_path)
    symbol = reduce(getattr, qualname.split("."), module)
    assert "Cost:" in (symbol.__doc__ or ""), f"{module_path}.{qualname} lacks a Cost: section"
