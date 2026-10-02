# Sparse implementation

`torchalg.sparse` contains the CSR-specific implementation tree. Dense
`torchalg` modules must not import from it; callers select a dense or sparse
algorithm based on the matrix representation they already own.

- `kernels/` provides reusable sparse primitives such as diagonal extraction,
  Galerkin formation, graph operations, and triangular solves.
- `preconditioners/` contains complete sparse algorithms, including AMG,
  IC(0), ICholesky, ILU, and Jacobi implementations.
- `device_policy.py` supplies internal, advisory CPU/CUDA recommendations
  derived from a particular benchmark environment. It never moves tensors and
  is not a cross-hardware performance guarantee.

Re-run the opt-in experiments in `benchmarks/sparse_dense/` when a device,
PyTorch version, or matrix family changes materially.
