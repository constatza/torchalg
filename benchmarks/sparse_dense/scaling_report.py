"""Plot time and memory trends emitted by :mod:`benchmarks.sparse_dense.scaling`."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

DEFAULT_ARTIFACTS_DIR = Path("artifacts/benchmarks/sparse_dense")
DEFAULT_PLOTS_DIR = DEFAULT_ARTIFACTS_DIR / "scaling-plots"


@dataclass(frozen=True)
class Row:
    """One successful or skipped scaling CSV row."""

    suite: str
    algorithm: str
    format_label: str
    device_label: str
    n: int
    compile_warmup_time_s: float | None
    median_time_s: float | None
    matrix_storage_bytes: int | None
    working_set_storage_bytes: int | None
    gpu_peak_bytes: int | None
    cpu_rss_delta_bytes: int | None
    skip_reason: str | None

    @property
    def leg(self) -> str:
        """A concise representation/device legend label."""
        return f"{self.format_label}/{self.device_label}"

    @property
    def execution_peak_bytes(self) -> int | None:
        """CUDA allocator peak or approximate CPU RSS delta, where available."""
        return self.gpu_peak_bytes if self.device_label == "cuda" else self.cpu_rss_delta_bytes


def read_rows(path: Path) -> list[Row]:
    """Read scaling CSV output."""
    with path.open(newline="") as handle:
        return [
            Row(
                suite=record["suite"],
                algorithm=record["algorithm"],
                format_label=record["format"],
                device_label=record["device"],
                n=int(record["n"]),
                compile_warmup_time_s=(
                    float(record["compile_warmup_time_s"])
                    if record["compile_warmup_time_s"]
                    else None
                ),
                median_time_s=float(record["median_time_s"]) if record["median_time_s"] else None,
                matrix_storage_bytes=(
                    int(record["matrix_storage_bytes"]) if record["matrix_storage_bytes"] else None
                ),
                working_set_storage_bytes=(
                    int(record["working_set_storage_bytes"])
                    if record["working_set_storage_bytes"]
                    else None
                ),
                gpu_peak_bytes=int(record["gpu_peak_bytes"]) if record["gpu_peak_bytes"] else None,
                cpu_rss_delta_bytes=(
                    int(record["cpu_rss_delta_bytes"]) if record["cpu_rss_delta_bytes"] else None
                ),
                skip_reason=record["skip_reason"] or None,
            )
            for record in csv.DictReader(handle)
        ]


def _series(rows: list[Row], metric: str) -> dict[str, list[tuple[int, float]]]:
    """Group one algorithm's successful rows by representation/device leg."""
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        value = getattr(row, metric)
        if value is not None:
            grouped[row.leg].append((row.n, float(value)))
    for values in grouped.values():
        values.sort()
    return grouped


def _style(row: Row) -> dict[str, str]:
    """Encode format and device redundantly, independent of legend text."""
    colors = {"dense/cpu": "C0", "sparse/cpu": "C1", "dense/cuda": "C2", "sparse/cuda": "C3"}
    return {
        "color": colors[row.leg],
        "linestyle": "-" if row.format_label == "dense" else "--",
        "marker": "o" if row.device_label == "cpu" else "s",
    }


def _plot(rows: list[Row], suite: str, metric: str, label: str, output: Path) -> None:
    """Write one log-log panel per algorithm for a suite and metric."""
    algorithms = sorted({row.algorithm for row in rows})
    if not algorithms:
        return
    columns = 2
    figure, axes = plt.subplots(
        math.ceil(len(algorithms) / columns),
        columns,
        figsize=(11, 3.8 * math.ceil(len(algorithms) / columns)),
    )
    for axis, algorithm in zip(axes.flat, algorithms, strict=False):
        algorithm_rows = [row for row in rows if row.algorithm == algorithm]
        series = _series(algorithm_rows, metric)
        for leg, values in sorted(series.items()):
            ns, measurements = zip(*values, strict=True)
            style_row = next(row for row in algorithm_rows if row.leg == leg)
            axis.plot(ns, measurements, label=leg, **_style(style_row))
        axis.set_xscale("log")
        if any(
            getattr(row, metric) is not None and getattr(row, metric) > 0 for row in algorithm_rows
        ):
            axis.set_yscale("log")
        axis.set_xlabel("system dimension N")
        axis.set_ylabel(label)
        axis.set_title(algorithm.replace("_", " "))
        axis.grid(True, which="both", alpha=0.3)
        if series:
            axis.legend(fontsize=8)
        else:
            axis.text(
                0.5,
                0.5,
                "No completed cells\n(all timed out or unavailable)",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
    for axis in list(axes.flat)[len(algorithms) :]:
        axis.set_visible(False)
    figure.suptitle(f"{suite.replace('_', ' ')}: {label} scaling", y=1.01)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    """Read one scaling CSV and save time plus memory plots for both suites."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_ARTIFACTS_DIR / "scaling.csv")
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    args = parser.parse_args()
    rows = read_rows(args.input)
    for suite in ("elementary", "end_to_end"):
        suite_rows = [row for row in rows if row.suite == suite]
        _plot(
            suite_rows,
            suite,
            "compile_warmup_time_s",
            "compile + warm-up time (s)",
            args.plots_dir / f"{suite}_compile_time.png",
        )
        _plot(
            suite_rows,
            suite,
            "median_time_s",
            "median time (s)",
            args.plots_dir / f"{suite}_time.png",
        )
        _plot(
            suite_rows,
            suite,
            "matrix_storage_bytes",
            "exact A storage (bytes)",
            args.plots_dir / f"{suite}_matrix_memory.png",
        )
        _plot(
            suite_rows,
            suite,
            "working_set_storage_bytes",
            "resident working-set storage (bytes)",
            args.plots_dir / f"{suite}_working_set_memory.png",
        )
        _plot(
            suite_rows,
            suite,
            "execution_peak_bytes",
            "execution peak memory (bytes)",
            args.plots_dir / f"{suite}_peak_memory.png",
        )
    print(f"Plots written to {args.plots_dir}")


if __name__ == "__main__":
    main()
