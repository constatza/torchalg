"""Break-even right-hand-side count against Jacobi, as ``benchmarks.compare`` reports it."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from benchmarks.compare import break_even_counts, break_even_k, load_measurements


@pytest.mark.parametrize(
    ("setup_s", "solve_s", "jacobi_setup_s", "jacobi_solve_s", "expected"),
    [
        # 0.25 + 3 * 0.02 > 3 * 0.1 but 0.25 + 4 * 0.02 <= 4 * 0.1.
        (0.25, 0.02, 0.0, 0.1, 4),
        # Jacobi's own setup counts on its side of the comparison.
        (0.25, 0.02, 0.1, 0.1, 2),
        # Cheaper setup and solve: ahead from the first right-hand side.
        (0.0, 0.02, 0.01, 0.1, 1),
        # A solve no faster than Jacobi's never repays a larger setup.
        (0.25, 0.1, 0.0, 0.1, math.inf),
        (0.25, 0.2, 0.0, 0.1, math.inf),
        # Slower solve but a setup saving larger than the loss: ahead at k = 1.
        (0.0, 0.12, 0.5, 0.1, 1),
    ],
)
def test_break_even_k(
    setup_s: float, solve_s: float, jacobi_setup_s: float, jacobi_solve_s: float, expected: float
) -> None:
    assert break_even_k(setup_s, solve_s, jacobi_setup_s, jacobi_solve_s) == expected


def test_break_even_counts_cover_only_baselined_cells_with_both_phases(
    result_file: Path, amg_cell_label: str
) -> None:
    assert break_even_counts(load_measurements(result_file)) == {amg_cell_label: 4}
