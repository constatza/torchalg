"""Per-test-vector weighting for Bootstrap AMG, shared by the dense and sparse trees.

Promoted out of ``preconditioners.implementations.amg._algebraic_distance``
(see ``docs/plan.md``'s "Correction: dense and sparse must be separate
implementations" - same "one legitimate shared edge" precedent as
``torchalg.utils.spectral``'s ``approximate_spectral_radius``): the only
format-dependent operations here are ``matrix @ test_vectors``/``T @
test_vectors``, plain matrix-vector products that dispatch correctly
whether ``matrix``/``T`` are dense or sparse CSR, so the function itself
never needs to branch on format - only a caller choosing a dense or sparse
``matrix`` does.

References:
    - Brandt, A., Brannick, J., Kahl, K., & Livshits, I. (2015). Bootstrap
      algebraic multigrid: status report, open problems, and outlook. Numer.
      Math. Theor. Meth. Appl. 8(1). arXiv:1406.1819. Cited as [STATUS14].
"""

from __future__ import annotations

import torch

_NEAR_ZERO_ENERGY_TOL = 1e-14
"""Test-vector energies with magnitude below this are treated as 1.0 when
computing weights, protecting against division by near-zero."""


def test_vector_weights(
    test_vectors: torch.Tensor, matrix: torch.Tensor, T: torch.Tensor | None = None
) -> torch.Tensor:
    """Per-test-vector weights ``omega_kappa``.

    ``omega_kappa = <T v^(kappa), v^(kappa)> / <A v^(kappa), v^(kappa)>``.
    ``T`` is the composite-interpolation Gram operator ``P_l^H P_l``
    (``docs/bootstrap-amg.md`` Sec. 4.2); ``T = None`` uses the ``T = I``
    reduction to a pure ``A``-energy weighting, valid "on the finest level,
    or before any MGE enrichment" - i.e. whenever no composite
    interpolation has been built yet, which is every level when MGE
    (``k_e``) is disabled, and the finest level regardless.
    **TODO(bamg-fidelity, needs-check):** the ``T``-weighted generalization
    used here for ``k_e > 0`` is not confirmed against [STATUS14], which
    states ``omega_kappa`` only via ``||v||_A^2`` (no ``T`` term); see
    ``_mge.py``'s module docstring for the full status of this gap.

    Args:
        test_vectors (torch.Tensor): Test vectors ``V``, shape ``(n, k)``.
        matrix (torch.Tensor): Dense or sparse CSR matrix ``A``, shape
            ``(n, n)``.
        T (torch.Tensor | None): Composite-interpolation Gram operator,
            shape ``(n, n)``; ``None`` for the ``T = I`` reduction.

    Returns:
        torch.Tensor: Weight vector, shape ``(k,)``.
    """
    energy = (test_vectors * (matrix @ test_vectors)).sum(dim=0)
    energy_safe = torch.where(energy.abs() > _NEAR_ZERO_ENERGY_TOL, energy, torch.ones_like(energy))
    numerator = (
        (test_vectors**2).sum(dim=0)
        if T is None
        else (test_vectors * (T @ test_vectors)).sum(dim=0)
    )
    return numerator / energy_safe
