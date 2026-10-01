"""Sparse-CSR AMG building blocks: smoothers, coarsening, aggregation, transfer.

``variants.{vcycle_amg,wcycle_amg}`` are the end-to-end preset factories
(wire everything in this package plus the shared ``torchalg.multigrid``
engine into a ready-to-``apply()`` ``AMGPreconditioner``); ``coarse_solve
.dense_coarse_solve`` is the coarsest-level direct-solve adapter they inject.
"""
