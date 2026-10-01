"""End-to-end PCG convergence test for ``BootCMatchPreconditioner`` (sparse-CSR).

Sparse-CSR sibling of
``tests/solver/preconditioners/implementations/amg/test_bootcmatch.py`` -
correctness/convergence only, same acceptance criteria.
"""

from __future__ import annotations

import torch

from torchalg import pcg
from torchalg.sparse.preconditioners.amg.bootcmatch import BootCMatchPreconditioner


def test_pcg_converges_with_bootcmatch_preconditioner(
    poisson_1d_large_dense: torch.Tensor, poisson_1d_large_rhs: torch.Tensor
) -> None:
    sparse_matrix = poisson_1d_large_dense.to_sparse_csr()
    preconditioner = BootCMatchPreconditioner(
        sparse_matrix,
        seed=3,
        max_levels=4,
        max_coarse=4,
        k_max=4,
        rho_desired=0.8,
        max_hierarchies=3,
    )

    assert not preconditioner.requires_flexible_cg
    assert len(preconditioner.hierarchies) >= 1

    solution, info = pcg(
        sparse_matrix,
        poisson_1d_large_rhs,
        preconditioner=preconditioner,
        rtol=1e-8,
        maxiter=200,
    )

    assert info.converged
    residual_norm = torch.linalg.norm(poisson_1d_large_rhs - poisson_1d_large_dense @ solution)
    assert residual_norm < 1e-6
