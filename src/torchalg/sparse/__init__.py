"""Sparse (``torch.sparse`` CSR) package: full parallel tree to dense ``torchalg.*``.

Mirrors ``torchalg.*``'s own internal shape (see ``docs/plan.md``'s
"Correction: dense and sparse must be separate implementations, not an
internal branch"):

- ``torchalg.sparse.kernels``: primitives with no standalone correctness
  story of their own, consumed by multiple algorithms (diagonal, row-scale,
  Galerkin formation, triangular level-scheduling, strength-of-connection,
  tentative prolongation).
- ``torchalg.sparse.preconditioners``: complete algorithms with their own
  correctness/breakdown story (AMG smoothers/coarsening/aggregation/
  transfer, IC0, ILU).
- ``torchalg.sparse.device_policy``: ``recommend_device``, an advisory
  CPU/CUDA recommendation over measured operation-shape/size thresholds -
  consulted by callers and, per pipeline stage, by sparse orchestrators
  themselves; never used to silently migrate a tensor's device.

No class anywhere in dense ``torchalg.*`` may import from this package;
``tach.toml`` enforces that boundary structurally. The caller - whatever
constructs an ``AMGPreconditioner`` or a standalone preconditioner - picks
the dense or sparse sibling class based on the format of the matrix it
already has in hand, the same way a caller picks ``torch.mm`` vs.
``torch.sparse.mm`` themselves; this package deliberately exposes no
router/factory function that would pick for them.
"""

from __future__ import annotations
