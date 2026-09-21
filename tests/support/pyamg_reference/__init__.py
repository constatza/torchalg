"""PyAMG-backed oracles for the AMG-family preconditioners (test-only).

PyAMG is a dev dependency used purely as an independent reference: nothing
under ``src/`` imports it. The helpers here translate torchalg tensors and
hierarchies into PyAMG objects so the two implementations can be run on
*identical operators* and compared elementwise.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyamg
import scipy.sparse as sp
import torch
from pyamg.aggregation.tentative import fit_candidates
from pyamg.multilevel import MultilevelSolver
from pyamg.relaxation.smoothing import change_smoothers

from torchalg.preconditioners.implementations.amg.hierarchy import MultigridHierarchy


def to_csr(matrix: torch.Tensor) -> sp.csr_matrix:
    """Convert a dense torch matrix to a SciPy CSR matrix.

    Args:
        matrix (torch.Tensor): Dense 2-D tensor.

    Returns:
        sp.csr_matrix: CSR copy of ``matrix``.
    """
    return sp.csr_matrix(matrix.detach().cpu().numpy())


def pyamg_tentative(
    aggregate: torch.Tensor, vectors: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """PyAMG's ``fit_candidates`` for a torchalg aggregation and test-vector block.

    Args:
        aggregate (torch.Tensor): Aggregate index per node, ``-1`` if isolated.
        vectors (torch.Tensor): Test vectors, shape ``(n, m)``.

    Returns:
        tuple[np.ndarray, np.ndarray]: Dense ``(Q, R)`` with PyAMG's layout
            (``m`` columns per aggregate).
    """
    n, n_agg = aggregate.shape[0], int(aggregate.max().item()) + 1
    assigned = (aggregate >= 0).nonzero().flatten()
    agg_op = sp.csr_matrix(
        (np.ones(len(assigned)), (assigned.numpy(), aggregate[assigned].numpy())),
        shape=(n, n_agg),
    )
    q, r = fit_candidates(agg_op, vectors.numpy(), tol=0.0)
    return q.toarray(), r


def multilevel_from_hierarchy(
    hierarchy: MultigridHierarchy, omega: float, n_pre: int, n_post: int
) -> MultilevelSolver:
    """Build a PyAMG multilevel solver from torchalg's operators.

    Reuses torchalg's ``A_l`` and ``P_l`` verbatim (``R = P^T``), so only the
    cycle and smoother implementations differ between the two sides.

    Args:
        hierarchy (MultigridHierarchy): Built torchalg hierarchy.
        omega (float): Weighted-Jacobi damping.
        n_pre (int): Pre-smoothing sweeps.
        n_post (int): Post-smoothing sweeps.

    Returns:
        MultilevelSolver: PyAMG solver with weighted-Jacobi smoothing.
    """
    levels = []
    for index, level in enumerate(hierarchy.levels):
        lvl = MultilevelSolver.Level()
        lvl.A = to_csr(level.matrix)
        if level.transfer is not None:
            coarse_dim = hierarchy.levels[index + 1].matrix.shape[0]
            p = level.transfer.prolongate(torch.eye(coarse_dim, dtype=level.matrix.dtype))
            prolongation = sp.csr_matrix(p.numpy())
            lvl.P = prolongation  # ty: ignore[unresolved-attribute]
            lvl.R = prolongation.T.tocsr()  # ty: ignore[unresolved-attribute]
        levels.append(lvl)
    solver = MultilevelSolver(levels, coarse_solver="pinv")
    change_smoothers(
        solver,
        ("jacobi", {"omega": omega, "iterations": n_pre, "withrho": False}),
        ("jacobi", {"omega": omega, "iterations": n_post, "withrho": False}),
    )
    return solver


def as_torch_preconditioner(
    solver: MultilevelSolver, cycle: str, dtype: torch.dtype
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Wrap a PyAMG solver as a torch ``r -> M^{-1} r`` callable (zero initial guess).

    Args:
        solver (MultilevelSolver): PyAMG multilevel solver.
        cycle (str): ``"V"`` or ``"W"``.
        dtype (torch.dtype): Output dtype.

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: One cycle applied to a residual.
    """
    operator = solver.aspreconditioner(cycle=cycle)
    return lambda residual: torch.from_numpy(operator.matvec(residual.numpy())).to(dtype)


def pyamg_adaptive_sa(matrix: torch.Tensor, seed: int, **kwargs: object) -> MultilevelSolver:
    """PyAMG's adaptive smoothed aggregation solver, seeded for reproducibility.

    Args:
        matrix (torch.Tensor): SPD system matrix.
        seed (int): NumPy seed (PyAMG draws its random candidates from the global RNG).
        **kwargs (object): Forwarded to ``pyamg.aggregation.adaptive_sa_solver``.

    Returns:
        MultilevelSolver: The adapted multilevel solver.
    """
    np.random.seed(seed)
    solver, _ = pyamg.aggregation.adaptive_sa_solver(to_csr(matrix), **kwargs)
    return solver


def pyamg_sa_two_level(
    matrix: torch.Tensor, vectors: torch.Tensor, theta: float, omega: float, n_pre: int, n_post: int
) -> tuple[MultilevelSolver, float]:
    """PyAMG smoothed aggregation with given test vectors, plus its effective prolongator omega.

    PyAMG's Jacobi prolongator smoother uses ``omega_eff = omega / rho`` with
    ``rho ~ rho(D^-1 A)`` estimated by a short *randomly started* Arnoldi run
    (1 % tolerance), so ``omega_eff`` is not reproducible from the arguments.
    It is recovered here from the returned ``T`` (tentative) and ``P``, letting
    a torchalg strategy be built with the identical damping.

    Args:
        matrix (torch.Tensor): SPD matrix.
        vectors (torch.Tensor): Test vectors, shape ``(n, m)``.
        theta (float): Symmetric strength-of-connection threshold.
        omega (float): PyAMG's nominal prolongator omega (4/3 is the SA default).
        n_pre (int): Pre-smoothing sweeps (weighted Jacobi, fixed 0.67).
        n_post (int): Post-smoothing sweeps.

    Returns:
        tuple[MultilevelSolver, float]: Two-level solver and ``omega_eff``.
    """
    smoother = ("jacobi", {"omega": 0.67, "withrho": False})
    solver = pyamg.smoothed_aggregation_solver(
        to_csr(matrix),
        B=vectors.numpy(),
        max_levels=2,
        max_coarse=1,
        improve_candidates=None,
        keep=True,
        strength=("symmetric", {"theta": theta}),
        aggregate="standard",
        smooth=("jacobi", {"omega": omega}),
        presmoother=(smoother[0], {**smoother[1], "iterations": n_pre}),
        postsmoother=(smoother[0], {**smoother[1], "iterations": n_post}),
        coarse_solver="pinv",
    )
    level = solver.levels[0]
    dense = matrix.numpy()
    step = (dense / np.diag(dense)[:, None]) @ level.T.toarray()
    omega_eff = float((step * (level.T.toarray() - level.P.toarray())).sum() / (step * step).sum())
    return solver, omega_eff


def aggregation_operator(aggregate: torch.Tensor) -> sp.csr_matrix:
    """PyAMG-style aggregation operator ``(n_nodes, n_aggregates)`` from an index vector.

    Args:
        aggregate (torch.Tensor): Aggregate index per node, ``-1`` if isolated.

    Returns:
        sp.csr_matrix: 0/1 operator with an empty row for isolated nodes.
    """
    n_agg = max(int(aggregate.max().item()) + 1, 1)
    nodes = (aggregate >= 0).nonzero().flatten().numpy()
    return sp.csr_matrix(
        (np.ones(len(nodes)), (nodes, aggregate[nodes].numpy())), shape=(len(aggregate), n_agg)
    )


def replay_draw(seed: int) -> Callable[[int], torch.Tensor]:
    """Uniform ``[0, 1)`` draws replaying NumPy's global stream for ``np.random.seed(seed)``.

    PyAMG draws its random start vectors with ``np.random.rand``; feeding the
    same stream to torchalg makes both implementations consume identical
    numbers in identical order.

    Args:
        seed (int): Seed shared with ``np.random.seed`` on the PyAMG side.

    Returns:
        Callable[[int], torch.Tensor]: ``n -> float64`` tensor of length ``n``.
    """
    generator = np.random.RandomState(seed)
    return lambda n: torch.from_numpy(generator.rand(n, 1).ravel())


def pinned_specs(
    aggregates: tuple[torch.Tensor, ...], strengths: tuple[torch.Tensor, ...]
) -> dict[str, list[tuple[str, dict[str, sp.csr_matrix]]]]:
    """PyAMG ``aggregate=``/``strength=`` lists that pin the given aggregation and strength.

    Args:
        aggregates (tuple[torch.Tensor, ...]): Aggregate index per node, per level.
        strengths (tuple[torch.Tensor, ...]): Boolean node strength graph, per level.

    Returns:
        dict: Keyword arguments for ``adaptive_sa_solver``.
    """
    return {
        "aggregate": [("predefined", {"AggOp": aggregation_operator(a)}) for a in aggregates],
        "strength": [
            ("predefined", {"C": sp.csr_matrix(s.numpy().astype(float) + np.eye(len(s)))})
            for s in strengths
        ],
    }
