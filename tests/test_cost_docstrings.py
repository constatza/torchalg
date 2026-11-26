"""Every symbol whose cost drives a placement or algorithm choice documents it.

The convention puts a ``Cost:`` section before ``Args:`` in the docstring, stating flops,
bytes moved, depth, kernel launches and host synchronisations as formulas; this test keeps
the documented symbols from losing it. Every public preconditioner, smoother and coarsening
of the dense and sparse trees (and the factories that build them) must state the work
``W``, the depth ``D`` and the launch count ``L`` of each phase it has: a paragraph
starting ``Setup`` for the matrix-dependent one-off work and one starting ``Apply`` for a
single application (one sweep for a smoother), or the whole section for a symbol that only
sets up (a coarsening's ``build_transfer``, a hierarchy builder).
"""

from __future__ import annotations

import importlib
import re
from functools import reduce

import pytest

_AMG = "torchalg.sparse.preconditioners.amg"
_SPARSE = "torchalg.sparse.preconditioners"
_DENSE = "torchalg.preconditioners.implementations"
_DENSE_AMG = f"{_DENSE}.amg"

COST_DOCUMENTED = [
    ("torchalg.conjugate_gradient", "PCGSolver"),
    ("torchalg.conjugate_gradient", "ConjugateGradientSolver._iterate_step"),
    ("torchalg.utils.numerics", "stable_dot_product"),
    ("torchalg.preconditioners.base", "Identity"),
    (f"{_AMG}._compatible_relaxation", "_independent_set_of"),
    (f"{_AMG}.aggregation", "sparse_standard_aggregation"),
    (f"{_AMG}.transfer", "SparseTransferOperator"),
    (f"{_AMG}._jacobi_omega", "jacobi_omega"),
    ("torchalg.sparse.kernels.triangular", "level_schedule"),
    ("torchalg.sparse.kernels.triangular", "PreparedTriangularSolve"),
    ("torchalg.sparse.kernels.triangular", "LibraryTriangularSolve"),
    ("torchalg.sparse.kernels.galerkin", "form_sparse_sparse"),
    ("torchalg.sparse.kernels.depth_neighborhood", "sparse_depth_neighborhood"),
    ("torchalg.multigrid.coarse_solve", "LUCoarseSolve"),
    ("torchalg.multigrid.coarse_solve", "PseudoInverseCoarseSolve"),
    ("torchalg.multigrid.coarse_solve", "lu_coarse_solver"),
    ("torchalg.multigrid.coarse_solve", "pseudo_inverse_coarse_solver"),
    ("torchalg.utils.spectral", "approximate_spectral_radius"),
    ("torchalg.utils.ls_interpolation", "batched_select_interpolatory_set"),
    ("torchalg.utils.pod_basis", "compute_pod_basis"),
    ("torchalg.utils.pod_basis", "_gram_basis"),
]

SETUP_AND_APPLY = ("Setup", "Apply")
APPLY_ONLY = ("Apply",)
WHOLE_SECTION = ()

PHASED_COST_DOCUMENTED = [
    # Dense preconditioners.
    (f"{_DENSE}.jacobi", "JacobiPreconditioner", SETUP_AND_APPLY),
    (f"{_DENSE}.ic0", "IC0Preconditioner", SETUP_AND_APPLY),
    (f"{_DENSE}.ilu", "ILUPreconditioner", SETUP_AND_APPLY),
    (f"{_DENSE}.icholesky", "ICholeskyPreconditioner", SETUP_AND_APPLY),
    (f"{_DENSE}.callable", "CallablePreconditioner", APPLY_ONLY),
    (f"{_DENSE}.linear_operator", "LinearOperatorPreconditioner", APPLY_ONLY),
    (f"{_DENSE}.neural", "NeuralPreconditioner", APPLY_ONLY),
    (f"{_DENSE}.scheduled", "ScheduledPreconditioner", SETUP_AND_APPLY),
    (f"{_DENSE}.pod.variants", "pod2g_preconditioner", SETUP_AND_APPLY),
    (f"{_DENSE}.pod.coarsening", "PODCoarseningStrategy", WHOLE_SECTION),
    (f"{_DENSE}.pod.coarsening", "PODCoarseningStrategy.build_transfer", WHOLE_SECTION),
    # Dense AMG.
    (f"{_DENSE_AMG}.smoothers", "PreparedJacobi", SETUP_AND_APPLY),
    (f"{_DENSE_AMG}.smoothers", "JacobiSmoother", SETUP_AND_APPLY),
    (f"{_DENSE_AMG}.smoothers", "GaussSeidelSmoother", SETUP_AND_APPLY),
    (f"{_DENSE_AMG}._relaxation", "PreparedSymmetricGaussSeidel", APPLY_ONLY),
    (f"{_DENSE_AMG}._relaxation", "prepare_symmetric_gauss_seidel", WHOLE_SECTION),
    (f"{_DENSE_AMG}.coarsening", "AggregationCoarsening.build_transfer", WHOLE_SECTION),
    (f"{_DENSE_AMG}.coarsening", "TargetDimensionCoarsening.build_transfer", WHOLE_SECTION),
    (f"{_DENSE_AMG}.coarsening", "NeuralCoarseningStrategy.build_transfer", WHOLE_SECTION),
    (f"{_DENSE_AMG}.bootcmatch_coarsening", "BootCMatchCoarsening.build_transfer", WHOLE_SECTION),
    (f"{_DENSE_AMG}.bootcmatch", "BootCMatchPreconditioner", SETUP_AND_APPLY),
    (f"{_DENSE_AMG}.bootstrap", "BAMGCoarsening.build_transfer", WHOLE_SECTION),
    (f"{_DENSE_AMG}.bootstrap", "BootstrapAMGPreconditioner", SETUP_AND_APPLY),
    (f"{_DENSE_AMG}.adaptive", "adaptive_sa_hierarchy", WHOLE_SECTION),
    (f"{_DENSE_AMG}.adaptive", "AdaptiveSAPreconditioner", SETUP_AND_APPLY),
    (f"{_DENSE_AMG}._compatible_relaxation", "compatible_relaxation_coarsening", WHOLE_SECTION),
    (f"{_DENSE_AMG}.variants", "vcycle_amg", SETUP_AND_APPLY),
    (f"{_DENSE_AMG}.variants", "wcycle_amg", SETUP_AND_APPLY),
    # Format-free multigrid engine.
    ("torchalg.multigrid.amg", "AMGPreconditioner", SETUP_AND_APPLY),
    ("torchalg.multigrid.cycle", "VCycle", SETUP_AND_APPLY),
    ("torchalg.multigrid.cycle", "WCycle", SETUP_AND_APPLY),
    ("torchalg.multigrid.bootstrap_setup", "BootstrapSetup.run", WHOLE_SECTION),
    ("torchalg.multigrid.bootcmatch_setup", "BootCMatchSetup.run", WHOLE_SECTION),
    # Sparse preconditioners.
    (f"{_SPARSE}.jacobi", "JacobiPreconditioner", SETUP_AND_APPLY),
    (f"{_SPARSE}.ic0", "IC0Preconditioner", SETUP_AND_APPLY),
    (f"{_SPARSE}.ilu", "ILUPreconditioner", SETUP_AND_APPLY),
    (f"{_SPARSE}.icholesky", "ICholeskyPreconditioner", SETUP_AND_APPLY),
    # Sparse AMG.
    (f"{_AMG}.smoothers", "PreparedJacobi", SETUP_AND_APPLY),
    (f"{_AMG}.smoothers", "JacobiSmoother", SETUP_AND_APPLY),
    (f"{_AMG}.smoothers", "GaussSeidelSmoother", SETUP_AND_APPLY),
    (f"{_AMG}._gauss_seidel", "PreparedSymmetricGaussSeidel", APPLY_ONLY),
    (f"{_AMG}._gauss_seidel", "prepare_csr_gauss_seidel", WHOLE_SECTION),
    (f"{_AMG}.coarsening", "AggregationCoarsening.build_transfer", WHOLE_SECTION),
    (f"{_AMG}.coarsening", "TargetDimensionCoarsening.build_transfer", WHOLE_SECTION),
    (f"{_AMG}.bootcmatch_coarsening", "BootCMatchCoarsening.build_transfer", WHOLE_SECTION),
    (f"{_AMG}.bootcmatch", "BootCMatchPreconditioner", SETUP_AND_APPLY),
    (f"{_AMG}.bootstrap", "BAMGCoarsening.build_transfer", WHOLE_SECTION),
    (f"{_AMG}.bootstrap", "BootstrapAMGPreconditioner", SETUP_AND_APPLY),
    (f"{_AMG}.adaptive", "adaptive_sa_hierarchy", WHOLE_SECTION),
    (f"{_AMG}.adaptive", "AdaptiveSAPreconditioner", SETUP_AND_APPLY),
    (f"{_AMG}._compatible_relaxation", "compatible_relaxation_coarsening", WHOLE_SECTION),
    (f"{_AMG}.variants", "vcycle_amg", SETUP_AND_APPLY),
    (f"{_AMG}.variants", "wcycle_amg", SETUP_AND_APPLY),
]

# A quantity is stated as ``W = ...`` inside a literal, e.g. ``W = 4 nnz(L) k``.
_REQUIRED_QUANTITIES = ("W", "D", "L")
_PHASE_START = re.compile(r"^\s*(Setup|Apply)\b")


def _docstring_of(module_path: str, qualname: str) -> str:
    module = importlib.import_module(module_path)
    symbol = reduce(getattr, qualname.split("."), module)
    return symbol.__doc__ or ""


def _cost_section(docstring: str) -> str:
    """Return the lines of the ``Cost:`` section, up to the next section at its indent."""
    lines = docstring.splitlines()
    starts = [index for index, line in enumerate(lines) if line.strip() == "Cost:"]
    if not starts:
        return ""
    start = starts[0]
    indent = len(lines[start]) - len(lines[start].lstrip())
    section: list[str] = []
    for line in lines[start + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        section.append(line)
    return "\n".join(section)


def _phase_text(section: str, phase: str) -> str:
    """Return the paragraphs of ``section`` from a line starting ``phase`` to the next phase."""
    collected: list[str] = []
    inside = False
    for line in section.splitlines():
        match = _PHASE_START.match(line)
        if match:
            inside = match.group(1) == phase
        if inside:
            collected.append(line)
    return "\n".join(collected)


def _missing_quantities(text: str) -> list[str]:
    return [
        symbol for symbol in _REQUIRED_QUANTITIES if re.search(rf"``{symbol}\s*=", text) is None
    ]


@pytest.mark.parametrize(("module_path", "qualname"), COST_DOCUMENTED)
def test_docstring_has_cost_section(module_path: str, qualname: str) -> None:
    assert "Cost:" in _docstring_of(module_path, qualname), (
        f"{module_path}.{qualname} lacks a Cost: section"
    )


@pytest.mark.parametrize(("module_path", "qualname", "phases"), PHASED_COST_DOCUMENTED)
def test_cost_section_states_work_depth_and_launches(
    module_path: str, qualname: str, phases: tuple[str, ...]
) -> None:
    section = _cost_section(_docstring_of(module_path, qualname))
    assert section, f"{module_path}.{qualname} lacks a Cost: section"
    texts = {phase: _phase_text(section, phase) for phase in phases} or {"Cost": section}
    for phase, text in texts.items():
        assert text, f"{module_path}.{qualname}: Cost: has no {phase} paragraph"
        missing = _missing_quantities(text)
        assert not missing, f"{module_path}.{qualname}: {phase} cost lacks {missing}"
