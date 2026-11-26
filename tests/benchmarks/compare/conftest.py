"""Fixtures for the break-even report of ``benchmarks.compare``: a small recorded result file."""

from __future__ import annotations

from pathlib import Path

import pytest
from benchmarks.harness import Level, Measurement, Params, ResultSink

CELL: Params = {"n": 100, "nnz": 460, "format": "csr", "matrix": "lap2d-n100"}
UNBASELINED_CELL: Params = {"n": 64, "nnz": 1000, "format": "csr", "matrix": "box3d-n64"}


def _measurement(name: str, params: Params, median_s: float) -> Measurement:
    return Measurement(
        level=Level.ALGORITHM,
        name=name,
        params=params,
        device="cpu",
        dtype="float64",
        median_s=median_s,
        min_s=median_s,
        runs=1,
    )


@pytest.fixture
def result_file(tmp_path: Path) -> Path:
    """One run: Jacobi and AMG with both phases, IC(0) without a solve, a cell with no Jacobi."""
    sink = ResultSink()
    for name, params, median_s in (
        ("jacobi.setup", CELL, 0.0),
        ("jacobi.solve", CELL, 0.1),
        ("amg.setup", CELL, 0.25),
        ("amg.apply", CELL, 0.001),
        ("amg.solve", CELL, 0.02),
        ("ic0.setup", CELL, 0.05),
        ("amg.setup", UNBASELINED_CELL, 0.25),
        ("amg.solve", UNBASELINED_CELL, 0.02),
    ):
        sink.add(_measurement(name, params, median_s))
    path = tmp_path / "run.json"
    sink.write(path)
    return path


@pytest.fixture
def amg_cell_label() -> str:
    """Break-even label of the baselined AMG cell: its key with the phase dropped."""
    return "algorithm:amg[format=csr,matrix=lap2d-n100,n=100,nnz=460]@cpu/float64"
