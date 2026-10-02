"""Operation registry: one representative, resident-only function per tensor-op family.

Every function here takes operands already placed on the target
device/format - compute cost only, no data movement. Standalone transfer
(``h2d``/``d2h``) and conversion (``to_sparse``/``to_dense``) helpers are
timed independently in ``run.py`` rather than folded into any one
operation, since they're a function of size/format alone, not of which
compute follows them.

Where torchalg already has the real kernel (IC0/ILU0 factorization,
Cholesky-solve apply), it's imported and reused directly rather than
reimplemented, so "the representative op" really is the op the algorithms
use, not a lookalike.

Some sparse/native alternatives may be unavailable on a given torch build or
device (e.g. ``torch.sparse.spsolve`` needs a cuDSS-enabled build even on
CUDA - standard wheels raise ``NotImplementedError`` on CPU and
``RuntimeError`` on CUDA without cuDSS). Callers (``run.py``) are
expected to catch ``(NotImplementedError, RuntimeError)`` around each cell
and record it as a skip rather than a crash - operations.py itself stays a
plain function registry with no try/except noise.
"""

from __future__ import annotations

import torch
from scipy import sparse as scipy_sparse
from scipy.sparse.linalg import eigsh, spilu, svds

from torchalg.preconditioners.implementations._masked_factorization import dense_ic0, dense_ilu0
from torchalg.preconditioners.implementations._triangular import cholesky_factor_solve
from torchalg.sparse.preconditioners.ic0 import sparse_ic0

# ---------------------------------------------------------------------------
# Family 1: matvec + dot/AXPY (CG inner loop)
# ---------------------------------------------------------------------------


def mv(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Matrix-vector product, dense or sparse - CG/PCG/FCG's inner-loop op.

    Args:
        matrix: Dense or sparse ``(n, n)`` tensor.
        vector: Dense ``(n,)`` tensor.

    Returns:
        torch.Tensor: ``matrix @ vector``.
    """
    return matrix @ vector


# ---------------------------------------------------------------------------
# Family 2: elementwise scale (Jacobi apply)
# ---------------------------------------------------------------------------


def extract_inv_diag(matrix: torch.Tensor) -> torch.Tensor:
    """Extract ``1 / diag(matrix)``, dense or sparse.

    Vectorized via a COO round-trip rather than a per-row Python loop: for
    a lattice Laplacian/FEM stiffness matrix the diagonal is always
    strictly positive, so no near-zero guard is needed here (unlike the
    production ``JacobiPreconditioner``, which does guard - this is a
    synthetic-benchmark simplification, not a claim that guard is
    unnecessary in general).

    Args:
        matrix: Dense or sparse ``(n, n)`` tensor.

    Returns:
        torch.Tensor: ``(n,)`` reciprocal diagonal.
    """
    if matrix.layout == torch.strided:
        return 1.0 / torch.diagonal(matrix)
    coo = matrix.to_sparse_coo().coalesce()
    idx = coo.indices()
    on_diagonal = idx[0] == idx[1]
    diag = torch.zeros(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    diag[idx[0][on_diagonal]] = coo.values()[on_diagonal]
    return 1.0 / diag


def elementwise_scale(inv_diag: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Jacobi apply: ``z = D^{-1} r``.

    Args:
        inv_diag: ``(n,)`` reciprocal diagonal, from :func:`extract_inv_diag`.
        residual: ``(n,)`` residual vector.

    Returns:
        torch.Tensor: ``(n,)`` scaled residual.
    """
    return inv_diag * residual


# ---------------------------------------------------------------------------
# Family 3: Galerkin triple product (AMG/POD coarsening) - formed vs. matrix-free
# ---------------------------------------------------------------------------


def form_galerkin_dense(prolongation: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Form ``A_coarse = P.T @ A @ P`` with both operands dense.

    Args:
        prolongation: Dense ``(n, r)`` transfer operator ``P``.
        matrix: Dense ``(n, n)`` system matrix ``A``.

    Returns:
        torch.Tensor: Dense ``(r, r)`` coarse operator.
    """
    return prolongation.T @ matrix @ prolongation


def form_galerkin_spmm(
    prolongation_dense: torch.Tensor, matrix_sparse: torch.Tensor
) -> torch.Tensor:
    """Form ``A_coarse = P.T @ A @ P`` with sparse ``A``, dense ``P`` (SpMM).

    Args:
        prolongation_dense: Dense ``(n, r)`` transfer operator ``P``, as
            torchalg currently stores it.
        matrix_sparse: Sparse ``(n, n)`` system matrix ``A``.

    Returns:
        torch.Tensor: Dense ``(r, r)`` coarse operator.
    """
    return prolongation_dense.T @ (matrix_sparse @ prolongation_dense)


def form_galerkin_spgemm(
    prolongation_sparse: torch.Tensor, matrix_sparse: torch.Tensor
) -> torch.Tensor:
    """Form ``A_coarse = P.T @ A @ P`` with both operands sparse (SpGEMM).

    Real AMG prolongation operators are themselves sparse (each fine node
    interpolates from a handful of coarse neighbors), even though torchalg
    currently stores ``P`` dense - this leg benchmarks that alternative.
    Uses COO for the transpose step: ``sparse_csr.t()`` yields a CSC
    tensor, and mixed CSC/CSR sparse-sparse matmul support isn't something
    to assume - COO sidesteps the question entirely at negligible extra
    conversion cost relative to the multiply itself.

    Args:
        prolongation_sparse: Sparse ``(n, r)`` transfer operator ``P``.
        matrix_sparse: Sparse ``(n, n)`` system matrix ``A``.

    Returns:
        torch.Tensor: Sparse ``(r, r)`` coarse operator.
    """
    p_coo = prolongation_sparse.to_sparse_coo()
    a_coo = matrix_sparse.to_sparse_coo()
    ap = torch.sparse.mm(a_coo, p_coo)
    return torch.sparse.mm(p_coo.t(), ap)


def apply_formed(coarse_matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Apply an already-formed coarse operator to a vector: ``A_coarse @ v``.

    Args:
        coarse_matrix: Dense or sparse ``(r, r)`` coarse operator.
        vector: Dense ``(r,)`` vector.

    Returns:
        torch.Tensor: ``(r,)`` result.
    """
    return coarse_matrix @ vector


def apply_matrix_free(
    prolongation: torch.Tensor, matrix: torch.Tensor, vector: torch.Tensor
) -> torch.Tensor:
    """Apply the coarse operator without ever forming it: ``P.T @ (A @ (P @ v))``.

    Algebraically identical to :func:`apply_formed` on
    :func:`form_galerkin_dense`'s result, at O(N*bandwidth) per call instead
    of paying the formation cost even once - the alternative worth
    comparing against "always form" whenever the coarse operator is reused
    only a handful of times before being discarded.

    Args:
        prolongation: Dense or sparse ``(n, r)`` transfer operator ``P``.
        matrix: Dense or sparse ``(n, n)`` system matrix ``A``.
        vector: Dense ``(r,)`` vector.

    Returns:
        torch.Tensor: ``(r,)`` result.
    """
    return prolongation.T @ (matrix @ (prolongation @ vector))


# ---------------------------------------------------------------------------
# Family 4: triangular solve (IC0/ILU/ICholesky apply - factor already computed)
# ---------------------------------------------------------------------------


def triangular_apply_dense(factor: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Apply a precomputed dense Cholesky-style factor: solve ``(L @ L.T) z = r``.

    Reuses torchalg's own ``cholesky_factor_solve`` - the actual kernel
    IC0/ICholesky use, not a lookalike.

    Args:
        factor: Dense lower-triangular factor ``L``, shape ``(n, n)``.
        residual: Dense ``(n,)`` residual vector.

    Returns:
        torch.Tensor: ``(n,)`` preconditioned residual.
    """
    return cholesky_factor_solve(factor, residual)


def triangular_apply_sparse(factor_sparse: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Apply a precomputed sparse Cholesky-style factor via ``torch.sparse.spsolve``.

    Two sparse solves (forward with ``L``, backward with ``L.T``), mirroring
    the dense two-triangular-solve apply above. ``torch.sparse.spsolve`` has
    no CPU kernel (raises ``NotImplementedError``), and on CUDA it needs
    PyTorch built against cuDSS - standard CUDA wheels aren't, so it raises
    there too (``RuntimeError``, worded "...is not supported in ROCm build,"
    which fires on plain CUDA builds without cuDSS and is misleading - it
    is not a ROCm-specific check). Callers should treat either as "this leg
    isn't available here," not a bug.

    Why not pip-install cuDSS and move on: official PyTorch CUDA wheels
    (verified by inspecting ``libtorch_cuda.so`` - no ``NEEDED`` entry and
    no ``dlopen`` reference to any ``libcudss``) have the cuDSS code path
    compiled out entirely, not merely missing an optional runtime library.
    Getting a working cuDSS-backed ``spsolve`` needs a PyTorch built from
    source with ``-DUSE_CUDSS=1`` against the cuDSS SDK - out of scope here.

    A real alternative exists for the CUDA leg: CuPy's
    ``cupyx.scipy.sparse.linalg.spsolve_triangular`` runs on cuSPARSE
    directly, no cuDSS needed, and CUDA tensors can move to/from CuPy
    zero-copy via DLPack (``cupy.from_dlpack``/``torch.from_dlpack``). Not
    wired in here: it needs its own dependency, and the DLPack conversion
    is an extra cost that would have to be timed separately from the solve
    itself (same reasoning as ``h2d``/``to_sparse`` being their own rows
    rather than folded into a compute cell) - deferred until a CUDA sparse
    solve leg is actually needed, not added speculatively.

    Args:
        factor_sparse: Sparse lower-triangular factor ``L``, CSR, shape
            ``(n, n)``.
        residual: Dense ``(n,)`` residual vector.

    Returns:
        torch.Tensor: ``(n,)`` preconditioned residual.
    """
    intermediate = torch.sparse.spsolve(factor_sparse, residual)
    upper = factor_sparse.t().to_sparse_csr()
    return torch.sparse.spsolve(upper, intermediate)


# ---------------------------------------------------------------------------
# Family 5: dense factorization (IC0/ILU setup) vs. scipy incomplete LU
# ---------------------------------------------------------------------------


def factorize_ic0(matrix: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """Dense incomplete Cholesky factorization - reuses torchalg's own kernel.

    Args:
        matrix: Dense SPD ``(n, n)`` matrix.
        threshold: Drop tolerance (see ``dense_ic0``).

    Returns:
        torch.Tensor: Dense lower-triangular factor ``L``.
    """
    return dense_ic0(matrix, threshold)


def factorize_ilu0(matrix: torch.Tensor) -> torch.Tensor:
    """Dense ILU(0) factorization - reuses torchalg's own kernel.

    Args:
        matrix: Dense ``(n, n)`` matrix.

    Returns:
        torch.Tensor: Combined dense ``L``/``U`` factor tensor.
    """
    return dense_ilu0(matrix)


def factorize_sparse_ic0(matrix: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """Sparse incomplete Cholesky factorization - reuses torchalg.sparse's own kernel.

    Args:
        matrix: Sparse CSR SPD ``(n, n)`` matrix.
        threshold: Drop tolerance (see ``sparse_ic0``).

    Returns:
        torch.Tensor: Sparse CSR lower-triangular factor ``L``.
    """
    return sparse_ic0(matrix, threshold)


def to_scipy_csc(matrix_sparse: torch.Tensor) -> scipy_sparse.csc_matrix:
    """Round-trip a torch sparse tensor to a scipy CSC matrix (host round-trip, CPU-only).

    Args:
        matrix_sparse: Sparse torch tensor (any sparse layout).

    Returns:
        scipy.sparse.csc_matrix: Equivalent matrix for scipy's sparse
            factorization/eigensolver routines, which require host memory.
    """
    coo = matrix_sparse.to_sparse_coo().coalesce().cpu()
    idx = coo.indices().numpy()
    values = coo.values().numpy()
    n = matrix_sparse.shape[0]
    return scipy_sparse.coo_matrix((values, (idx[0], idx[1])), shape=(n, n)).tocsc()


def factorize_spilu(matrix_scipy_csc: scipy_sparse.csc_matrix):
    """Sparse incomplete LU factorization - the apples-to-apples scipy analog to IC0/ILU0.

    Args:
        matrix_scipy_csc: Scipy CSC matrix, from :func:`to_scipy_csc`.

    Returns:
        scipy.sparse.linalg.SuperLU: Opaque incomplete-LU factorization
            object (scipy's own factor representation).
    """
    return spilu(matrix_scipy_csc)


# ---------------------------------------------------------------------------
# Family 6: eigendecomposition (AMG's MGE, condition-number diagnostics)
# ---------------------------------------------------------------------------


def eigh_dense(matrix: torch.Tensor) -> torch.Tensor:
    """Full dense symmetric eigendecomposition.

    Args:
        matrix: Dense symmetric ``(n, n)`` matrix.

    Returns:
        torch.Tensor: ``(n,)`` eigenvalues.
    """
    return torch.linalg.eigvalsh(matrix)


def eigh_lobpcg(matrix: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Partial-spectrum eigendecomposition via native, sparse-capable, CPU+GPU LOBPCG.

    Args:
        matrix: Dense or sparse symmetric ``(n, n)`` matrix. Sparse input is
            converted to COO first (torch's ``lobpcg`` doesn't accept CSR
            directly).
        k: Number of extreme eigenpairs to compute.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(eigenvalues, eigenvectors)``,
            shapes ``(k,)`` and ``(n, k)``.
    """
    operand = (
        matrix.to_sparse_coo() if matrix.layout in (torch.sparse_csr, torch.sparse_csc) else matrix
    )
    return torch.lobpcg(operand, k=k)


def eigh_scipy(matrix_scipy_csr: scipy_sparse.csr_matrix, k: int):
    """Partial-spectrum eigendecomposition via scipy ARPACK (secondary, CPU-only reference).

    Args:
        matrix_scipy_csr: Scipy sparse matrix, symmetric.
        k: Number of extreme eigenpairs to compute.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(eigenvalues, eigenvectors)``.
    """
    return eigsh(matrix_scipy_csr, k=k)


# ---------------------------------------------------------------------------
# Family 7: SVD (POD basis, thin snapshot matrix)
# ---------------------------------------------------------------------------


def svd_dense(snapshots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full thin SVD of a snapshot matrix - reuses torchalg POD's own call pattern.

    Args:
        snapshots: Dense ``(n_samples, n_dofs)`` snapshot matrix.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ``(u, s, vh)``.
    """
    return torch.linalg.svd(snapshots, full_matrices=False)


def svd_lowrank_native(
    snapshots: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomized truncated SVD - native, sparse-capable, CPU+GPU alternative.

    Closer to POD's actual truncated-basis use than a full SVD.

    Args:
        snapshots: Dense or sparse ``(n_samples, n_dofs)`` snapshot matrix.
        rank: Target rank.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ``(u, s, v)``.
    """
    return torch.svd_lowrank(snapshots, q=rank)


def svd_scipy(snapshots_scipy_sparse, rank: int):
    """Truncated SVD via scipy ARPACK (secondary, CPU-only reference).

    Args:
        snapshots_scipy_sparse: Scipy sparse ``(n_samples, n_dofs)`` matrix.
        rank: Target rank.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: ``(u, s, vt)``.
    """
    return svds(snapshots_scipy_sparse, k=rank)


# ---------------------------------------------------------------------------
# Standalone transfer/conversion helpers, timed independently of any compute op
# ---------------------------------------------------------------------------


def h2d(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Host-to-device copy, synchronized so the timed cost is real, not async dispatch.

    Args:
        tensor: Source tensor (any device).
        device: Target device.

    Returns:
        torch.Tensor: Copy resident on ``device``.
    """
    moved = tensor.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return moved


def d2h(tensor: torch.Tensor) -> torch.Tensor:
    """Device-to-host copy, synchronized so the timed cost is real, not async dispatch.

    Args:
        tensor: Source tensor (any device).

    Returns:
        torch.Tensor: Copy resident on CPU.
    """
    source_device = tensor.device
    moved = tensor.to("cpu")
    if source_device.type == "cuda":
        torch.cuda.synchronize(source_device)
    return moved


def to_sparse(tensor: torch.Tensor) -> torch.Tensor:
    """Dense -> sparse CSR conversion, timed as its own operation.

    Args:
        tensor: Dense tensor.

    Returns:
        torch.Tensor: Sparse CSR tensor.
    """
    return tensor.to_sparse_csr()


def to_dense(tensor: torch.Tensor) -> torch.Tensor:
    """Sparse -> dense conversion, timed as its own operation.

    Args:
        tensor: Sparse tensor (any layout).

    Returns:
        torch.Tensor: Dense (strided) tensor.
    """
    return tensor.to_dense()
