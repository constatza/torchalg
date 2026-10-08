# Benchmarks

This directory contains reproducible, opt-in performance experiments. It is
not part of the `torchalg` runtime package and no benchmark result is treated
as a portable performance guarantee: results vary with hardware, PyTorch,
matrix structure, and dtype.

`sparse_dense/` measures dense and CSR implementations across CPU/CUDA. Its
small cross-format correctness coverage lives in
`tests/integration/sparse_dense/` and runs in the ordinary test suite; timing
and report generation never run as part of pytest.

## Running sparse-vs-dense experiments

Install the benchmark dependency group, then invoke the modules from the
repository root:

```sh
# Broad operation grid: time, storage, transfer, and conversion measurements.
uv run --group benchmark python -m benchmarks.sparse_dense.run

# Decision table and plots for the broad grid.
uv run --group benchmark python -m benchmarks.sparse_dense.report

# Focused dense-vs-CSR scaling, including preconditioner/PCG pipelines.
uv run --group benchmark python -m benchmarks.sparse_dense.scaling
uv run --group benchmark python -m benchmarks.sparse_dense.scaling_report
```

The default output directory is `artifacts/benchmarks/sparse_dense/`, which is
ignored by Git. Pass `--output`, `--input`, or `--plots-dir` to select an
experiment-specific location. Curated figures for documentation may be copied
deliberately to a documentation asset directory; raw CSVs and local plots are
not versioned.

Both runners accept bounded size and timeout options. Use `--help` before a
large run, and use a small explicit output path for a smoke test:

```sh
uv run --group benchmark python -m benchmarks.sparse_dense.run \
  --sizes 500 --output /tmp/torchalg-benchmark-smoke/results.csv
```

The broad runner records one row for each attempted or intentionally skipped
cell. Its triangular-apply family compares dense Cholesky application with
torchalg's setup-prepared, level-scheduled CSR application; factor and schedule
construction remain outside the timed repeated-apply callable. Static
exclusions avoid known impractical dense cubic operations, while runtime memory
and timeout guards remain a second line of protection for attempted cells.
