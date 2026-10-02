"""Matrix sources for the dense-vs-sparse benchmark: real templates and synthetic FEM-like matrices.

Two sources:

- **Real templates**: the two FEM-derived matrices under
  ``/data/shared/mgroup-clusters/`` (anchor/validation points, not part of
  the size sweep - only two sizes exist there, 840 and 3015). Loaded once
  via ``np.loadtxt``; their structural
  stats are cached to JSON since the larger file is 136MB and reparsing it
  on every benchmark run would dominate wall-clock time for no reason.
- **Synthetic matrices**: a structured lattice Laplacian (finite-difference
  stencil on a d-dimensional grid), giving exact control over N at any
  size while staying SPD and banded like real FEM stiffness matrices. Built
  via ``scipy.sparse.kronsum`` rather than a hand-rolled mesh assembler -
  ``kronsum(A, B) == kron(A, I) + kron(I, B)`` is exactly the operator that
  extends a 1D stencil to a d-dimensional grid one axis at a time.

``AI constantinos.zip``'s ``speres8RVEstiffneesMat.txt`` files are not
loaded here: their byte size matches ``spheres-1000x.txt`` exactly, almost
certainly the same matrix under a different name/location, so only the
already-extracted top-level file is used as the large template.
# ponytail: skips an automated zip-extraction + hash-compare for what is, at
# most, a one-time curiosity; if a real second large sample turns out to be
# needed, extract and diff it by hand rather than scripting a check that
# only ever runs once.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix, diags, kronsum

REAL_TEMPLATE_DIR = Path("/data/shared/mgroup-clusters")
REAL_TEMPLATES: dict[str, Path] = {
    "rectangular-high-condition": REAL_TEMPLATE_DIR / "rectangular-high-condition.txt",
    "spheres-1000x": REAL_TEMPLATE_DIR / "spheres-1000x.txt",
}
"""Name -> path for the two real FEM-derived anchor matrices."""

STATS_CACHE_PATH = Path(__file__).with_name("results") / "template_stats.json"
"""Where structural stats for real templates are cached, keyed by template name."""

PROLONGATION_NNZ_PER_ROW = 4
"""Nonzero coarse columns per fine row in :func:`sparse_prolongation`.

Matches a real AMG/POD interpolation stencil (e.g. 2D bilinear
interpolation from the 4 surrounding coarse-grid corners) - "a handful",
not a size that scales with ``n`` or ``rank``.
"""


@dataclass(frozen=True)
class TemplateStats:
    """Structural properties of a real template matrix.

    Attributes:
        n: Matrix dimension.
        nnz: Count of entries with ``|value| > 0``.
        fill_percent: ``100 * nnz / n**2``.
        bandwidth: Max ``|i - j|`` over non-zero entries ``(i, j)``.
        symmetric: True if ``allclose(A, A.T)``.
        spd: True if a Cholesky factorization of ``A`` succeeds.
    """

    n: int
    nnz: int
    fill_percent: float
    bandwidth: int
    symmetric: bool
    spd: bool


def one_dimensional_laplacian(k: int) -> csr_matrix:
    """Build the 1D Poisson/Laplacian stencil ``[-1, 2, -1]``, size ``(k, k)``.

    Args:
        k: Number of grid points along this axis.

    Returns:
        csr_matrix: Tridiagonal SPD matrix, diag=2, off-diag=-1.
    """
    return diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(k, k), format="csr")


def lattice_laplacian(k: int, dims: int) -> csr_matrix:
    """Build a d-dimensional lattice Laplacian on a ``k`` x ... x ``k`` grid.

    ``scipy.sparse.kronsum(A, B) == kron(A, I) + kron(I, B)``: chaining it
    across ``dims`` copies of the 1D stencil extends the finite-difference
    operator one grid axis at a time, giving the standard 5-point (2D) or
    7-point (3D) stencil - SPD, banded, with the same bandwidth/connectivity
    character as a real FEM stiffness matrix, without assembling an actual
    mesh.

    Args:
        k: Grid points per axis.
        dims: Number of grid axes (2 or 3 in practice).

    Returns:
        csr_matrix: SPD matrix of size ``k**dims``.
    """
    stencil = one_dimensional_laplacian(k)
    result = stencil
    for _ in range(dims - 1):
        result = kronsum(result, stencil, format="csr")
    return result


def grid_side_for(target_n: int, dims: int) -> int:
    """Grid side length ``k`` whose ``k**dims`` is closest to ``target_n``.

    Args:
        target_n: Desired approximate matrix size.
        dims: Number of grid axes.

    Returns:
        int: ``k >= 2``, ``round(target_n ** (1/dims))`` clamped to 2.
    """
    return max(2, round(target_n ** (1.0 / dims)))


def synthetic_matrix(target_n: int, dims: int = 2) -> tuple[csr_matrix, int]:
    """Build a synthetic SPD lattice-Laplacian matrix near a target size.

    Args:
        target_n: Desired approximate matrix size (exact only when
            ``target_n**(1/dims)`` is an integer).
        dims: Number of grid axes (default 2).

    Returns:
        tuple[csr_matrix, int]: ``(matrix, actual_n)`` where ``actual_n ==
            matrix.shape[0]``, the closest achievable size to ``target_n``.
    """
    k = grid_side_for(target_n, dims)
    matrix = lattice_laplacian(k, dims)
    return matrix, matrix.shape[0]


def to_torch_dense(matrix: csr_matrix, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Convert a scipy sparse matrix to a dense torch tensor.

    Args:
        matrix: Source scipy sparse matrix.
        dtype: Target torch dtype.

    Returns:
        torch.Tensor: Dense ``(n, n)`` tensor.
    """
    return torch.from_numpy(matrix.toarray()).to(dtype)


def to_torch_sparse_csr(matrix: csr_matrix, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Convert a scipy CSR matrix to a torch sparse CSR tensor.

    Args:
        matrix: Source scipy CSR matrix.
        dtype: Target torch dtype.

    Returns:
        torch.Tensor: Sparse CSR ``(n, n)`` tensor.
    """
    csr = matrix.tocsr()
    return torch.sparse_csr_tensor(
        torch.from_numpy(csr.indptr).to(torch.int64),
        torch.from_numpy(csr.indices).to(torch.int64),
        torch.from_numpy(csr.data).to(dtype),
        size=csr.shape,
        check_invariants=False,
    )


def sparse_prolongation(
    n: int,
    rank: int,
    nnz_per_row: int = PROLONGATION_NNZ_PER_ROW,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Build a genuinely sparse ``(n, rank)`` prolongation operator ``P``.

    Real AMG/POD prolongation operators are sparse - each fine row
    interpolates from a handful of coarse columns, not from all of them.
    ``torch.randn(n, rank).to_sparse_csr()`` does *not* give that: a
    continuous Gaussian sample is essentially never exactly ``0.0``, so
    that round trip stores ~100% of entries as explicit nonzeros - dense
    data in a sparse container. This builds exactly ``nnz_per_row``
    nonzeros per row instead, via ``topk`` over per-row random keys
    (vectorized - no per-row Python loop).

    Args:
        n: Number of fine rows.
        rank: Number of coarse columns.
        nnz_per_row: Nonzero coarse columns per fine row, clamped to
            ``rank``.
        dtype: Value dtype.

    Returns:
        torch.Tensor: Sparse CSR ``(n, rank)`` tensor.
    """
    nnz_per_row = min(nnz_per_row, rank)
    col_indices = torch.rand(n, rank).topk(nnz_per_row, dim=1).indices
    row_indices = torch.arange(n).unsqueeze(1).expand(-1, nnz_per_row)
    values = torch.randn(n, nnz_per_row, dtype=dtype)
    indices = torch.stack([row_indices.reshape(-1), col_indices.reshape(-1)])
    coo = torch.sparse_coo_tensor(
        indices,
        values.reshape(-1),
        size=(n, rank),
        check_invariants=False,
    )
    return coo.coalesce().to_sparse_csr()


def load_real_template(name: str) -> torch.Tensor:
    """Load a real template matrix as a dense float64 torch tensor.

    Args:
        name: Key into :data:`REAL_TEMPLATES`.

    Returns:
        torch.Tensor: Dense ``(n, n)`` matrix, float64.

    Raises:
        KeyError: If ``name`` is not a known template.
        FileNotFoundError: If the template file is missing (e.g. the shared
            data mount isn't available on this machine).
    """
    path = REAL_TEMPLATES[name]
    if not path.exists():
        raise FileNotFoundError(f"Template matrix not found: {path}")
    return torch.from_numpy(np.loadtxt(path)).to(torch.float64)


def compute_stats(matrix: torch.Tensor) -> TemplateStats:
    """Compute structural stats for a dense matrix.

    Args:
        matrix: Dense square matrix.

    Returns:
        TemplateStats: nnz/fill/bandwidth/symmetry/SPD summary.
    """
    n = matrix.shape[0]
    nonzero = matrix.abs() > 0
    nnz = int(nonzero.sum())
    rows, cols = torch.nonzero(nonzero, as_tuple=True)
    bandwidth = int((rows - cols).abs().max()) if nnz else 0
    symmetric = bool(torch.allclose(matrix, matrix.T, atol=1e-8, rtol=1e-6))
    try:
        torch.linalg.cholesky(matrix)
        spd = True
    except torch.linalg.LinAlgError:
        spd = False
    return TemplateStats(
        n=n,
        nnz=nnz,
        fill_percent=100.0 * nnz / (n * n),
        bandwidth=bandwidth,
        symmetric=symmetric,
        spd=spd,
    )


def get_or_compute_stats(name: str) -> TemplateStats:
    """Fetch cached structural stats for a real template, computing and caching them if absent.

    Args:
        name: Key into :data:`REAL_TEMPLATES`.

    Returns:
        TemplateStats: Cached or freshly-computed stats.
    """
    cache: dict[str, dict] = {}
    if STATS_CACHE_PATH.exists():
        cache = json.loads(STATS_CACHE_PATH.read_text())
    if name in cache:
        return TemplateStats(**cache[name])

    stats = compute_stats(load_real_template(name))
    cache[name] = asdict(stats)
    STATS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_CACHE_PATH.write_text(json.dumps(cache, indent=2))
    return stats
