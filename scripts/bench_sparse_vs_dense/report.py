#!/usr/bin/env python3
"""CLI: read run_benchmark.py's results CSV, print a decision table, and plot scaling curves.

Answers the actual question this benchmark exists for: which operations
should become sparse, and should they run on CPU or GPU - not just "here
are some numbers." Two outputs:

- A printed decision table per operation family: the fastest leg at the
  smallest and largest successfully-measured N, and the crossover N (or K,
  for the Galerkin form-vs-matrix-free split) where one leg overtakes
  another, if any.
- One log-log time-vs-N plot per family (one line per format/device leg,
  `skipped` cells marked rather than silently omitted), plus a time-vs-K
  chart for the Galerkin family's formed-vs-matrix-free comparison.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).with_name("results")
PLOTS_DIR = RESULTS_DIR / "plots"


@dataclass(frozen=True)
class Row:
    """One parsed CSV row."""

    family: str
    op: str
    format_label: str
    device_label: str
    n: int
    k: int | None
    median_time_s: float | None
    storage_bytes: int | None
    skip_reason: str | None

    @property
    def leg(self) -> str:
        """Short label identifying this (op, format, device) combination for plot legends."""
        return f"{self.op}/{self.format_label}/{self.device_label}"


def read_rows(csv_path: Path) -> list[Row]:
    """Parse ``run_benchmark.py``'s output CSV into typed rows.

    Args:
        csv_path: Path to the results CSV.

    Returns:
        list[Row]: One entry per grid cell, timed or skipped.
    """
    rows = []
    with csv_path.open(newline="") as handle:
        for record in csv.DictReader(handle):
            rows.append(
                Row(
                    family=record["family"],
                    op=record["op"],
                    format_label=record["format"],
                    device_label=record["device"],
                    n=int(record["n"]),
                    k=int(record["k"]) if record["k"] else None,
                    median_time_s=float(record["median_time_s"])
                    if record["median_time_s"]
                    else None,
                    storage_bytes=int(float(record["storage_bytes"]))
                    if record["storage_bytes"]
                    else None,
                    skip_reason=record["skip_reason"] or None,
                )
            )
    return rows


def find_crossover(
    series_a: list[tuple[int, float]], series_b: list[tuple[int, float]]
) -> float | None:
    """Find the x-value where series B's time first overtakes (becomes faster than) series A's.

    Both series must share the same x-values (as ``run_benchmark.py``'s
    shared ``--sizes``/``--k-values`` sweep guarantees); only x-values
    present in both are compared.

    Args:
        series_a: ``[(x, time), ...]``, sorted or not.
        series_b: ``[(x, time), ...]``, same x-domain as ``series_a``.

    Returns:
        float | None: The first ``x`` where B <= A, given A was slower (or
            equal) at the previous shared x. ``None`` if B never overtakes
            A (or they share no common x).
    """
    times_a = dict(series_a)
    times_b = dict(series_b)
    shared_x = sorted(set(times_a) & set(times_b))
    if not shared_x:
        return None

    was_a_faster = times_a[shared_x[0]] <= times_b[shared_x[0]]
    for x in shared_x[1:]:
        b_faster_now = times_b[x] <= times_a[x]
        if was_a_faster and b_faster_now:
            return x
        was_a_faster = not b_faster_now
    return None


def group_by_leg(rows: list[Row], *, k: int | None = None) -> dict[str, list[tuple[int, float]]]:
    """Group successfully-timed rows into ``{leg: [(n, time), ...]}``, sorted by N.

    Args:
        rows: Rows for a single family.
        k: If given, restricts to rows with this K (galerkin only);
            ``None`` includes all rows (every other family has no K axis).

    Returns:
        dict[str, list[tuple[int, float]]]: Per-leg, N-sorted time series.
    """
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if row.median_time_s is None:
            continue
        if k is not None and row.k != k:
            continue
        grouped[row.leg].append((row.n, row.median_time_s))
    for series in grouped.values():
        series.sort()
    return grouped


def print_decision_table(family: str, rows: list[Row]) -> None:
    """Print the fastest leg at the smallest/largest measured N, plus any pairwise crossovers.

    Args:
        family: Family name (for the header).
        rows: This family's rows (across all N/K/legs).
    """
    print(f"\n=== {family} ===")
    grouped = group_by_leg(rows, k=max((r.k for r in rows if r.k is not None), default=None))
    if not grouped:
        print("  (no successful measurements)")
        return

    all_n = sorted({n for series in grouped.values() for n, _ in series})
    smallest_n, largest_n = all_n[0], all_n[-1]
    for label, n in (("smallest N", smallest_n), ("largest N", largest_n)):
        candidates = [(leg, dict(series).get(n)) for leg, series in grouped.items()]
        candidates = [(leg, t) for leg, t in candidates if t is not None]
        if not candidates:
            continue
        fastest_leg, fastest_time = min(candidates, key=lambda pair: pair[1])
        print(f"  {label} (N={n}): fastest = {fastest_leg} ({fastest_time:.3e}s)")

    legs = sorted(grouped)
    for i, leg_a in enumerate(legs):
        for leg_b in legs[i + 1 :]:
            crossover = find_crossover(grouped[leg_a], grouped[leg_b])
            if crossover is not None:
                print(f"  crossover: {leg_b} overtakes {leg_a} at N={crossover:g}")

    skipped = {(r.op, r.format_label, r.device_label, r.skip_reason) for r in rows if r.skip_reason}
    for op, fmt, device, reason in sorted(skipped):
        skip_ns = sorted(
            {r.n for r in rows if r.op == op and r.format_label == fmt and r.skip_reason == reason}
        )
        print(f"  skipped: {op}/{fmt}/{device} ({reason}) at N >= {skip_ns[0]}")


def plot_family_scaling(family: str, rows: list[Row], output_dir: Path) -> None:
    """Save a log-log time-vs-N chart for one family, one line per leg.

    Args:
        family: Family name (used for the title/filename).
        rows: This family's rows.
        output_dir: Directory to save the PNG into.
    """
    k_for_plot = max((r.k for r in rows if r.k is not None), default=None)
    grouped = group_by_leg(rows, k=k_for_plot)
    if not grouped:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for leg, series in sorted(grouped.items()):
        ns, times = zip(*series, strict=True)
        ax.plot(ns, times, marker="o", label=leg)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("median time (s)")
    title = f"{family} scaling" + (f" (K={k_for_plot})" if k_for_plot is not None else "")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{family}_scaling.png", dpi=150)
    plt.close(fig)


def plot_galerkin_k_sweep(rows: list[Row], output_dir: Path) -> None:
    """Save a time-vs-K chart for the Galerkin formed-vs-matrix-free comparison.

    One line per (op, format) leg, one subplot per representative N (the
    smallest and largest N with data for every K) - shows the reuse-count
    break-even directly, rather than only the fixed-K scaling chart above.

    Args:
        rows: Family-3 (``galerkin_form``) rows.
        output_dir: Directory to save the PNG into.
    """
    k_rows = [r for r in rows if r.k is not None and r.median_time_s is not None]
    if not k_rows:
        return
    all_n = sorted({r.n for r in k_rows})
    representative_ns = sorted({all_n[0], all_n[-1]})

    fig, axes = plt.subplots(
        1, len(representative_ns), figsize=(6 * len(representative_ns), 5), squeeze=False
    )
    for ax, n in zip(axes[0], representative_ns, strict=True):
        by_leg: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for r in k_rows:
            if r.n == n and r.k is not None and r.median_time_s is not None:
                by_leg[f"{r.op}/{r.format_label}"].append((r.k, r.median_time_s))
        for leg, series in sorted(by_leg.items()):
            series.sort()
            ks, times = zip(*series, strict=True)
            ax.plot(ks, times, marker="o", label=leg)
        ax.set_xlabel("K (reuse count)")
        ax.set_ylabel("median time (s)")
        ax.set_title(f"N={n}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Galerkin: formed vs. matrix-free break-even")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "galerkin_form_k_sweep.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Parse CLI args, print the decision table, and generate scaling plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RESULTS_DIR / "results.csv")
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    args = parser.parse_args()

    rows = read_rows(args.input)
    by_family: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_family[row.family].append(row)

    for family in sorted(by_family):
        print_decision_table(family, by_family[family])
        plot_family_scaling(family, by_family[family], args.plots_dir)

    if "galerkin_form" in by_family:
        plot_galerkin_k_sweep(by_family["galerkin_form"], args.plots_dir)

    print(f"\nPlots written to {args.plots_dir}")


if __name__ == "__main__":
    main()
