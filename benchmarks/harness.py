"""Timing and result recording shared by ``benchmarks/primitives`` and ``benchmarks/algorithms``.

Timing never asserts inside pytest (it flakes); suites only record a
:class:`Measurement` and assert correctness. :func:`time_callable` is the single timing
path so every number is comparable: warm-up, device synchronization on CUDA, median of
repeated runs, no GC noise.
"""

from __future__ import annotations

import gc
import json
import platform
import statistics
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from time import perf_counter

import torch

DEFAULT_REPEATS = 5
DEFAULT_WARMUPS = 1
RESULTS_DIR = Path("artifacts/benchmarks")

type ParamValue = str | int | float
type Params = Mapping[str, ParamValue]


class Level(StrEnum):
    """Which tier a measurement belongs to."""

    PRIMITIVE = "primitive"
    ALGORITHM = "algorithm"


@dataclass(frozen=True, slots=True)
class Measurement:
    """One timed quantity for one parameter point.

    ``name`` identifies the operation (``coarse_solve.apply``); ``params`` carries
    the sweep coordinates (``n``, ``variant``, ...). ``extra`` holds non-timing
    outputs such as iteration counts and residuals.
    """

    level: Level
    name: str
    params: Params
    device: str
    dtype: str
    median_s: float
    min_s: float
    runs: int
    extra: Mapping[str, ParamValue] = field(default_factory=dict)

    @property
    def key(self) -> str:
        coordinates = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.level}:{self.name}[{coordinates}]@{self.device}/{self.dtype}"


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_callable(
    fn: Callable[[], object],
    *,
    device: torch.device,
    repeats: int = DEFAULT_REPEATS,
    warmups: int = DEFAULT_WARMUPS,
) -> tuple[float, float, int]:
    """Return ``(median_s, min_s, runs)`` of ``fn`` after ``warmups`` untimed calls.

    The callable must do all the work being measured and nothing else (build inputs
    in a fixture, not here). Warm-up calls absorb one-off lazy work so steady state is
    measured; benchmark lazy first-use cost explicitly by passing ``warmups=0``.
    """
    for _ in range(warmups):
        fn()
    _synchronize(device)
    samples: list[float] = []
    gc.collect()
    gc.disable()
    try:
        for _ in range(repeats):
            _synchronize(device)
            start = perf_counter()
            fn()
            _synchronize(device)
            samples.append(perf_counter() - start)
    finally:
        gc.enable()
    return statistics.median(samples), min(samples), len(samples)


@dataclass(slots=True)
class ResultSink:
    """Collects measurements for a session and writes them as one JSON file."""

    measurements: list[Measurement] = field(default_factory=list)

    def add(self, measurement: Measurement) -> None:
        self.measurements.append(measurement)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "environment": {
                "torch": torch.__version__,
                "python": platform.python_version(),
                "cpu": platform.processor() or platform.machine(),
                "threads": torch.get_num_threads(),
                "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            "measurements": [asdict(m) for m in self.measurements],
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
