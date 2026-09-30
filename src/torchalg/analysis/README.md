# Analysis

`torchalg.analysis` owns optional numerical diagnostics. These routines inspect
operators but do not participate in a solve, so they stay separate from the
small, broadly depended-on primitives in `torchalg.utils`.

`spectral.py` computes exact condition numbers for dense symmetric systems and
preconditioned operators. Presentation and plotting do not belong in this
package; callers may render the returned scalar values in an application layer.
