"""Shared fixtures for the performance suites.

Run with ``uv run pytest benchmarks -m benchmark``. Each session writes one JSON file to
``artifacts/benchmarks/`` (override with ``TORCHALG_BENCH_OUT``); compare two runs with
``python -m benchmarks.compare before.json after.json``.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch

from benchmarks.cases import MatrixCase, default_cases
from benchmarks.harness import (
    DEFAULT_REPEATS,
    DEFAULT_WARMUPS,
    RESULTS_DIR,
    Level,
    Measurement,
    Params,
    ResultSink,
    time_callable,
)

OUTPUT_ENV = "TORCHALG_BENCH_OUT"

type Record = Callable[..., Measurement]


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every collected benchmark so the default ``-m "not benchmark"`` skips them."""
    for item in items:
        if "benchmarks" in item.path.parts:
            item.add_marker(pytest.mark.benchmark)


def _available_devices() -> list[str]:
    return ["cpu", *(["cuda"] if torch.cuda.is_available() else [])]


@pytest.fixture(scope="session")
def result_sink() -> Iterator[ResultSink]:
    sink = ResultSink()
    yield sink
    if sink.measurements:
        default = RESULTS_DIR / f"{datetime.now(UTC):%Y%m%d-%H%M%S}.json"
        sink.write(Path(os.environ.get(OUTPUT_ENV, default)))


@pytest.fixture(params=_available_devices())
def device(request: pytest.FixtureRequest) -> torch.device:
    return torch.device(request.param)


@pytest.fixture(params=[torch.float64], ids=["f64"])
def dtype(request: pytest.FixtureRequest) -> torch.dtype:
    return request.param


@pytest.fixture(params=default_cases(), ids=lambda case: case.id)
def case(request: pytest.FixtureRequest) -> MatrixCase:
    return request.param


@pytest.fixture
def dense_matrix(case: MatrixCase, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return case.dense(dtype, device)


@pytest.fixture
def csr_matrix_(case: MatrixCase, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return case.csr(dtype, device)


@pytest.fixture
def rhs(case: MatrixCase, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return case.rhs(dtype, device)


@pytest.fixture
def record(
    result_sink: ResultSink, device: torch.device, dtype: torch.dtype
) -> Callable[..., Measurement]:
    """Time ``fn`` and store the measurement; returns it so a test can also assert on ``extra``."""

    def _record(
        level: Level,
        name: str,
        params: Params,
        fn: Callable[[], object],
        *,
        repeats: int = DEFAULT_REPEATS,
        warmups: int = DEFAULT_WARMUPS,
        extra: Params | None = None,
    ) -> Measurement:
        median, minimum, runs = time_callable(fn, device=device, repeats=repeats, warmups=warmups)
        measurement = Measurement(
            level=level,
            name=name,
            params=params,
            device=device.type,
            dtype=str(dtype).removeprefix("torch."),
            median_s=median,
            min_s=minimum,
            runs=runs,
            extra=extra or {},
        )
        result_sink.add(measurement)
        return measurement

    return _record
