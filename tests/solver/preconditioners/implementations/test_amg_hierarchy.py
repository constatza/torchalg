"""Tests for the relocated test-vector helpers.

``TestBuildHierarchy`` moved to ``tests/multigrid/test_hierarchy.py``
alongside ``hierarchy.py``'s promotion into ``torchalg.multigrid`` (see
``docs/plan.md``).
"""

from __future__ import annotations

from torchalg.preconditioners.implementations.amg._test_vectors import (
    apply_jacobi_damping,
    apply_jacobi_damping_trajectory,
)
from torchalg.preconditioners.implementations.pod import weighting


class TestTestVectorHelpersRelocation:
    def test_pod_weighting_reexports_the_same_functions(self) -> None:
        assert weighting.apply_jacobi_damping is apply_jacobi_damping
        assert weighting.apply_jacobi_damping_trajectory is apply_jacobi_damping_trajectory
