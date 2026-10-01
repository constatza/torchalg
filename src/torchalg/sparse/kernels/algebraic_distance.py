"""Per-edge algebraic-distance strength of connection for Bootstrap AMG, sparse-CSR sibling.

Sparse-CSR counterpart of
``preconditioners.implementations.amg._algebraic_distance.algebraic_distance``
(dense; kept unmodified for comparison - see ``docs/plan.md``'s "Correction:
dense and sparse must be separate implementations, not an internal
branch"). The dense version computes a full dense ``(n, n)`` ``cross``/
``coefficients``/``residual`` chain and only masks it by the neighborhood
*after* computing every pair - ``O(n^2 * k)`` work to produce something
that's provably sparse. This reformulates the identical [AD11] eq. 4.3
closed-form as a **gather-dot over the neighborhood's nonzero pattern
only**: for every stored edge ``(i, j)``, gather row ``i``'s
residual-corrected test vector and row ``j``'s raw test vector, and reduce
over the test-vector axis in one vectorized op - ``O(nnz(neighborhood) *
k)``, no ``(n, n, k)`` intermediate, no Python loop, no dense matmul.

``r_ij`` is a **one-sided, directional** measure - see the dense module's
own docstring - computed only at ``sparse_depth_neighborhood``'s pattern,
in the same directional, unsymmetrized form the dense function returns
(the later union-symmetrization ``BAMGCoarsening._coarse_mask`` applies to
the pruned boolean strength graph happens one step downstream of this
function, not here - this function's job is an exact parity port of
``algebraic_distance`` alone).
"""

from __future__ import annotations

import torch

from torchalg.utils.test_vector_weights import test_vector_weights

from .depth_neighborhood import sparse_depth_neighborhood
from .lsr_correction import sparse_lsr_correction
from .triangular import _expand_row_index, _require_csr

_NEAR_ZERO_ENERGY_TOL = 1e-14
"""Matches the dense sibling's tolerance - see its docstring."""

_MIN_RESIDUAL_RTOL = 1e-14
"""Matches the dense sibling's tolerance - see its docstring."""


def sparse_algebraic_distance(
    test_vectors: torch.Tensor, matrix: torch.Tensor, depth: int = 1, T: torch.Tensor | None = None
) -> torch.Tensor:
    """Pairwise caliber-one algebraic distance ``r_ij`` ([AD11] eq. 4.3), sparse CSR.

    Args:
        test_vectors (torch.Tensor): Test vectors ``V``, shape ``(n, k)``.
        matrix (torch.Tensor): Sparse CSR fine-grid matrix ``A``, shape
            ``(n, n)``.
        depth (int): Search depth ``d`` defining the neighborhood ``V_i``
            (default ``1``).
        T (torch.Tensor | None): Composite-interpolation Gram operator
            passed through to ``test_vector_weights``, shape ``(n, n)``;
            ``None`` for the ``T = I`` reduction (see that function's
            docstring).

    Returns:
        torch.Tensor: Sparse CSR algebraic-distance matrix ``r``, shape
        ``(n, n)``, stored only within the depth-``d`` neighborhood. Not
        symmetric in general.

    Raises:
        ValueError: If ``matrix`` is not sparse CSR.
    """
    _require_csr(matrix, "sparse_algebraic_distance")
    n = matrix.shape[0]
    neighborhood = sparse_depth_neighborhood(matrix, depth)
    weights = test_vector_weights(test_vectors, matrix, T=T)
    corrected = sparse_lsr_correction(test_vectors, matrix, torch.arange(n, device=matrix.device))

    row = _expand_row_index(neighborhood)
    col = neighborhood.col_indices()

    predictor_energy = (test_vectors**2 * weights).sum(dim=1)
    predictor_safe = torch.where(
        predictor_energy.abs() > _NEAR_ZERO_ENERGY_TOL,
        predictor_energy,
        torch.ones_like(predictor_energy),
    )
    target_energy = (corrected**2 * weights).sum(dim=1)

    cross = (corrected[row] * weights * test_vectors[col]).sum(dim=1)
    coefficients = cross / predictor_safe[col]
    residual = target_energy[row] - coefficients * cross
    floor = (_MIN_RESIDUAL_RTOL * target_energy[row]).clamp(min=torch.finfo(residual.dtype).tiny)
    residual = torch.maximum(residual, floor)
    values = 1.0 / residual

    return torch.sparse_csr_tensor(
        neighborhood.crow_indices(),
        neighborhood.col_indices(),
        values,
        size=(n, n),
        check_invariants=False,
    )
