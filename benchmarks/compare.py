"""Compare two benchmark result files: ``python -m benchmarks.compare before.json after.json``.

Prints one row per measurement present in both files with the speed-up ratio
(``before / after``; above 1 means ``after`` is faster).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text())
    return {_key(m): float(m["median_s"]) for m in payload["measurements"]}


def _key(measurement: dict[str, object]) -> str:
    params = measurement["params"]
    assert isinstance(params, dict)
    coordinates = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
    return (
        f"{measurement['level']}:{measurement['name']}[{coordinates}]"
        f"@{measurement['device']}/{measurement['dtype']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    before, after = _load(args.before), _load(args.after)
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


if __name__ == "__main__":
    main()
