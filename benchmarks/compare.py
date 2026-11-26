"""Compare two benchmark result files: ``python -m benchmarks.compare before.json after.json``.

Prints one row per measurement present in both files with the speed-up ratio
(``before / after``; above 1 means ``after`` is faster), then the break-even count of
every algorithm cell against Jacobi in each file: the smallest number ``k`` of
right-hand sides sharing one matrix for which ``setup + k * solve`` of the method is no
more than Jacobi's on the same matrix, format, device and dtype (``inf`` if never).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

type RawMeasurement = dict[str, object]

BASELINE = "jacobi"
SETUP = "setup"
SOLVE = "solve"


def load_measurements(path: Path) -> list[RawMeasurement]:
    return json.loads(path.read_text())["measurements"]


def _median(measurement: RawMeasurement) -> float:
    median = measurement["median_s"]
    assert isinstance(median, int | float)
    return float(median)


def _timings(measurements: list[RawMeasurement]) -> dict[str, float]:
    return {_key(m): _median(m) for m in measurements}


def _key(measurement: RawMeasurement, name: str | None = None) -> str:
    params = measurement["params"]
    assert isinstance(params, dict)
    coordinates = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
    return (
        f"{measurement['level']}:{name or measurement['name']}[{coordinates}]"
        f"@{measurement['device']}/{measurement['dtype']}"
    )


def break_even_k(
    setup_s: float, solve_s: float, jacobi_setup_s: float, jacobi_solve_s: float
) -> float:
    """Smallest ``k >= 1`` with ``setup + k * solve <= jacobi_setup + k * jacobi_solve``.

    Returns ``inf`` when no ``k`` qualifies: the method's solve is no faster than Jacobi's
    and its setup is dearer.
    """
    extra_setup = setup_s - jacobi_setup_s
    saving = jacobi_solve_s - solve_s
    if extra_setup <= 0 and saving >= 0:
        return 1
    if saving <= 0:
        return math.inf
    return max(1, math.ceil(extra_setup / saving))


def break_even_counts(measurements: list[RawMeasurement]) -> dict[str, float]:
    """Break-even ``k`` of every non-Jacobi cell with a setup, a solve and a Jacobi baseline.

    Keys are the cell's measurement key with the method name in place of
    ``<method>.<phase>``.
    """
    timings = _timings(measurements)
    counts: dict[str, float] = {}
    for measurement in measurements:
        method, _, phase = str(measurement["name"]).rpartition(".")
        if phase != SETUP or method == BASELINE:
            continue
        needed = (f"{method}.{SOLVE}", f"{BASELINE}.{SETUP}", f"{BASELINE}.{SOLVE}")
        keys = [_key(measurement, name) for name in needed]
        if not all(key in timings for key in keys):
            continue
        solve, jacobi_setup, jacobi_solve = (timings[key] for key in keys)
        counts[_key(measurement, method)] = break_even_k(
            timings[_key(measurement)], solve, jacobi_setup, jacobi_solve
        )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    before_raw, after_raw = load_measurements(args.before), load_measurements(args.after)
    before, after = _timings(before_raw), _timings(after_raw)
    shared = sorted(before.keys() & after.keys())
    width = max((len(k) for k in shared), default=0)
    print(f"{'measurement':<{width}}  {'before':>10}  {'after':>10}  {'ratio':>7}")
    for key in shared:
        ratio = before[key] / after[key] if after[key] > 0 else float("inf")
        print(f"{key:<{width}}  {before[key]:>10.6f}  {after[key]:>10.6f}  {ratio:>6.2f}x")
    for label, only in (
        ("only before", before.keys() - after.keys()),
        ("only after", after.keys() - before.keys()),
    ):
        for key in sorted(only):
            print(f"{label}: {key}")

    before_k, after_k = break_even_counts(before_raw), break_even_counts(after_raw)
    cells = sorted(before_k.keys() | after_k.keys())
    width = max((len(c) for c in cells), default=0)
    print(f"\n{'break-even k vs ' + BASELINE:<{width}}  {'before':>8}  {'after':>8}")
    for cell in cells:
        print(
            f"{cell:<{width}}  {before_k.get(cell, math.nan):>8}  {after_k.get(cell, math.nan):>8}"
        )


if __name__ == "__main__":
    main()
