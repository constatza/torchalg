"""Sparse-native SA-AMG coarsening, sparse-CSR sibling.

Sparse-CSR counterpart of
``preconditioners.implementations.amg.coarsening.AggregationCoarsening``
(dense; kept unmodified for comparison - see ``docs/plan.md``'s "Correction:
dense and sparse must be separate implementations, not an internal branch").
Builds the strength/aggregation/prolongation chain via this package's own
sparse-native kernels end to end, never densifying the fine-grid matrix.

``TargetDimensionCoarsening`` here is the sparse-CSR sibling of the dense
``TargetDimensionCoarsening`` - same algorithm (adaptive theta search via
the shared, format-agnostic ``torchalg.utils.theta_search.adaptive_theta_scan``,
wrapping this module's own ``AggregationCoarsening``), disambiguated by
package: same public shape, same name, dense and sparse never import each
other.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from torchalg.sparse.kernels.galerkin import form_sparse_sparse
from torchalg.sparse.kernels.prolongation import sparse_piecewise_constant_prolongation
from torchalg.sparse.kernels.strength import sparse_strength_of_connection
from torchalg.sparse.preconditioners.amg._jacobi_omega import PROLONGATION_NOMINAL, jacobi_omega
from torchalg.sparse.preconditioners.amg.aggregation import (
    sparse_smoothed_prolongation,
    sparse_standard_aggregation,
)
from torchalg.sparse.preconditioners.amg.transfer import SparseTransferOperator
from torchalg.utils.theta_search import adaptive_theta_scan


class AggregationCoarsening:
    """Smoothed aggregation AMG coarsening using sparse CSR tensors (SA-AMG).

    Sparse-CSR sibling of the dense
    ``preconditioners.implementations.amg.coarsening.AggregationCoarsening``
    - same name, disambiguated by package. Implements the same five-step
    coarsening algorithm from Vanek, Mandel & Brezina (1996) as the dense
    class, entirely via sparse-native kernels (no dense ``(n, n)``
    intermediate at any step):

    1. Strength-of-connection (``sparse_strength_of_connection``).
    2. Aggregation (``sparse_standard_aggregation``).
    3. Tentative prolongation P0 (``sparse_piecewise_constant_prolongation``).
    4. Smoothed prolongation (``sparse_smoothed_prolongation``).
    5. Galerkin coarse matrix (``form_sparse_sparse``).

    Args:
        theta (float): Strength-of-connection threshold theta in (0, 1).
        omega (float | torch.Tensor | None): Jacobi damping for the
            prolongation smoother. ``None`` (default) is ``(4/3) /
            rho(D^{-1}A)``, estimated once per level matrix; a float or 0-d
            tensor fixes it.

    References:
        - Vanek, P., Mandel, J., & Brezina, M. (1996). Algebraic multigrid by
          smoothed aggregation for second and fourth order elliptic problems.
          Computing, 56(3), 179-196.
    """

    def __init__(self, theta: float = 0.25, omega: float | torch.Tensor | None = None) -> None:
        """Store the strength-of-connection threshold and smoothing damping factor.

        Args:
            theta (float): Strength-of-connection threshold theta in (0, 1).
            omega (float | torch.Tensor | None): Jacobi damping, or ``None``
                for ``(4/3) / rho``.
        """
        self._theta = theta
        self._omega = omega

    @property
    def theta(self) -> float:
        """Strength-of-connection threshold.

        Returns:
            float: The value passed at construction.
        """
        return self._theta

    @property
    def omega(self) -> float | torch.Tensor | None:
        """Prolongation-smoothing Jacobi damping.

        Returns:
            float | torch.Tensor | None: The value passed at construction.
        """
        return self._omega

    def __str__(self) -> str:
        """Human-readable structural summary.

        Returns:
            str: e.g. ``"AggregationCoarsening(theta=0.25, omega=0.67)"``.
        """
        omega = self._omega
        if isinstance(omega, torch.Tensor):
            omega = omega.item()
        omega_text = "auto" if omega is None else f"{omega:.3g}"
        return f"AggregationCoarsening(theta={self._theta:.3g}, omega={omega_text})"

    def build_transfer(self, A: torch.Tensor) -> tuple[torch.Tensor, SparseTransferOperator]:
        """Build one coarse level from a sparse CSR fine-grid matrix A.

        Args:
            A (torch.Tensor): Fine-grid matrix ``A``, shape ``(n, n)``,
                sparse CSR. Named to match the ``CoarseningStrategy``
                protocol's parameter name exactly.

        Returns:
            tuple[torch.Tensor, SparseTransferOperator]: ``(A_coarse,
                transfer)`` where ``A_coarse`` is a sparse CSR ``(n_c, n_c)``
                tensor (``form_sparse_sparse`` never densifies - see
                ``torchalg.sparse.kernels.galerkin``'s docstring). Only the
                coarsest level in a hierarchy is eventually densified, by
                the coarse solver, not here.
        """
        strength = sparse_strength_of_connection(A, self._theta)
        aggregate = sparse_standard_aggregation(strength)
        tentative = sparse_piecewise_constant_prolongation(aggregate, dtype=A.dtype)
        omega = jacobi_omega(A, PROLONGATION_NOMINAL, self._omega)
        prolongation = sparse_smoothed_prolongation(A, tentative, omega)

        coarse_matrix = form_sparse_sparse(prolongation, A)
        return coarse_matrix, SparseTransferOperator(prolongation)


@lru_cache(maxsize=1024)
def _cached_aggregation_build_transfer(
    A: torch.Tensor, theta: float, omega: float | None
) -> tuple[torch.Tensor, SparseTransferOperator]:
    """Build (and cache) one sparse `AggregationCoarsening` candidate for `(A, theta, omega)`.

    `A` hashes and compares by object identity (the default for
    `torch.Tensor`), so this only dedupes calls sharing the exact matrix
    object already resident from one search - never a false hit across two
    distinct, coincidentally-equal matrices. Backs
    `TargetDimensionCoarsening`'s `cache_candidates=True` option - the
    sparse-CSR sibling of the dense tree's own
    `_cached_aggregation_build_transfer`.
    """
    return AggregationCoarsening(theta=theta, omega=omega).build_transfer(A)


class TargetDimensionCoarsening:
    """Reaches a target coarse dimension by searching sparse `AggregationCoarsening`'s theta.

    Sparse-CSR sibling of the dense
    ``preconditioners.implementations.amg.coarsening.TargetDimensionCoarsening``
    - same name, disambiguated by package, same adaptive-theta-search
    algorithm (shared via ``torchalg.utils.theta_search.adaptive_theta_scan``,
    a dependency-free leaf with no dense/sparse assumption of its own - see
    that module's docstring). The only differences from the dense class:

    - The cheap dimension-only probe in `_search` uses
      `sparse_strength_of_connection`/`sparse_standard_aggregation` instead
      of the dense `strength_of_connection`/`standard_aggregation`. The
      aggregate tensor's ``-1``-for-isolated / max-plus-one convention for
      the realized dimension is identical between dense and sparse (see
      `sparse_standard_aggregation`'s docstring), so `realized_dimension`
      is otherwise line-for-line the same as the dense version.
    - The winning-theta full build calls this module's own sparse
      `AggregationCoarsening`, returning `SparseTransferOperator` (not
      `DenseTransferOperator`).
    - `A` is a sparse CSR tensor throughout, never densified.

    See the dense class's docstring
    (``preconditioners.implementations.amg.coarsening.TargetDimensionCoarsening``)
    for the full rationale behind exhaustive adaptive grid search over
    bisection or a black-box optimizer - realized coarse dimension is a
    non-monotonic step function of `theta` on heterogeneous matrices under
    either `standard_aggregation` implementation, dense or sparse.

    Args:
        target_coarse_dim (int): Desired realized coarse dimension.
        theta_min (float): Lower bound of the `theta` search grid.
        theta_max (float): Upper bound of the `theta` search grid.
        step (float): Coarse-pass spacing budget for `adaptive_theta_scan`.
        omega (float | None): Prolongation Jacobi-smoothing damping,
            forwarded to each candidate `AggregationCoarsening`; ``None``
            for the spectral rule ``(4/3) / rho(D^-1 A)``.
        cache_candidates (bool): If ``True``, the winning candidate's full
            `AggregationCoarsening.build_transfer` is memoized by
            `(A, theta, omega)` (object identity on `A`) via a shared
            `functools.lru_cache`, so a sibling instance searching the same
            matrix and landing on the same `theta` reuses the build instead
            of repeating it. Off by default - see the dense class's
            docstring for the same off-by-default rationale.
    """

    def __init__(
        self,
        target_coarse_dim: int,
        *,
        theta_min: float,
        theta_max: float,
        step: float,
        omega: float | None = None,
        cache_candidates: bool = False,
    ) -> None:
        """Store the target dimension and search parameters; unbuilt until `build_transfer`.

        Args:
            target_coarse_dim (int): Desired realized coarse dimension.
            theta_min (float): Lower bound of the `theta` search grid.
            theta_max (float): Upper bound of the `theta` search grid.
            step (float): Coarse-pass spacing budget for the search.
            omega (float | None): Prolongation Jacobi-smoothing damping, or
                ``None`` for the spectral rule ``(4/3) / rho(D^-1 A)``.
            cache_candidates (bool): Share the winning candidate's full
                build across instances against the same matrix - see the
                class docstring.
        """
        self._target_coarse_dim = target_coarse_dim
        self._theta_min = theta_min
        self._theta_max = theta_max
        self._step = step
        self._omega = omega
        self._cache_candidates = cache_candidates
        self._theta: float | None = None
        self._realized_coarse_dim: int | None = None

    @property
    def target_coarse_dim(self) -> int:
        """The requested coarse dimension.

        Returns:
            int: The value passed at construction.
        """
        return self._target_coarse_dim

    def realized_coarse_dim(self, A: torch.Tensor) -> int:
        """The actual coarse dimension the closest-matching theta produced.

        Triggers ``build_transfer(A)`` if the search hasn't run yet (lazy,
        cached - a second call never repeats the search).

        Args:
            A (torch.Tensor): Fine-grid matrix, sparse CSR, forwarded to
                ``build_transfer`` only if not already built.

        Returns:
            int: The realized coarse dimension.
        """
        if self._realized_coarse_dim is None:
            self.build_transfer(A)
        assert self._realized_coarse_dim is not None
        return self._realized_coarse_dim

    def __str__(self) -> str:
        """Human-readable structural summary.

        Returns:
            str: e.g. ``"TargetDimensionCoarsening(target_coarse_dim=3,
                realized_coarse_dim=3)"``, or a "not yet built" variant
                before the first ``build_transfer``/``realized_coarse_dim``
                call.
        """
        if self._realized_coarse_dim is None:
            return f"TargetDimensionCoarsening(target_coarse_dim={self._target_coarse_dim}, not yet built)"
        return (
            f"TargetDimensionCoarsening(target_coarse_dim={self._target_coarse_dim}, "
            f"realized_coarse_dim={self._realized_coarse_dim})"
        )

    def build_transfer(self, A: torch.Tensor) -> tuple[torch.Tensor, SparseTransferOperator]:
        """Search `theta`, then build the coarse level at the closest-matching value.

        The winning `theta` and its realized coarse dimension are cached as
        ``self._theta``/``self._realized_coarse_dim`` afterward, mirroring
        the dense sibling's convention.

        Args:
            A (torch.Tensor): Fine-grid matrix ``A``, shape ``(n, n)``,
                sparse CSR. Named to match the ``CoarseningStrategy``
                protocol's parameter name exactly.

        Returns:
            tuple[torch.Tensor, SparseTransferOperator]: ``(A_coarse,
                transfer)`` from the sparse `AggregationCoarsening`
                candidate whose realized coarse dimension is closest to
                ``target_coarse_dim``.
        """
        theta, a_coarse, transfer = self._search(A)
        self._theta = theta
        self._realized_coarse_dim = int(a_coarse.shape[0])
        return a_coarse, transfer

    def _search(self, A: torch.Tensor) -> tuple[float, torch.Tensor, SparseTransferOperator]:
        """Adaptively scan the `theta` grid, keeping the closest match to `target_coarse_dim`.

        Sampling uses a cheap dimension-only probe (sparse strength +
        aggregation, skipping prolongation smoothing and the Galerkin
        product) for every candidate `theta`; the full sparse
        `AggregationCoarsening.build_transfer` is called exactly once, at
        the winning `theta` - mirrors the dense sibling's `_search` split.

        Args:
            A (torch.Tensor): Fine-grid matrix, shape ``(n, n)``, sparse CSR.

        Returns:
            tuple[float, torch.Tensor, SparseTransferOperator]: ``(theta,
                A_coarse, transfer)`` for the candidate whose realized
                coarse dimension is closest to ``target_coarse_dim``.
        """

        def realized_dimension(theta: float) -> int:
            aggregate = sparse_standard_aggregation(sparse_strength_of_connection(A, theta))
            return int(aggregate.max().item()) + 1 if aggregate.numel() else 0

        samples = adaptive_theta_scan(
            self._theta_min, self._theta_max, self._step, realized_dimension
        )
        theta = min(samples, key=lambda sample: abs(sample[1] - self._target_coarse_dim))[0]
        if self._cache_candidates:
            a_coarse, transfer = _cached_aggregation_build_transfer(A, theta, self._omega)
        else:
            a_coarse, transfer = AggregationCoarsening(
                theta=theta, omega=self._omega
            ).build_transfer(A)
        return theta, a_coarse, transfer
