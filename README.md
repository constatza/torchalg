# torchalg

Pure-PyTorch preconditioned Conjugate Gradient solvers and a real
preconditioner library, for symmetric positive-definite (SPD) linear
systems.

![Python 3.14](https://img.shields.io/badge/python-3.14-blue)
![tests: 305 passed](https://img.shields.io/badge/tests-305_passed-brightgreen)
![code style: ruff](https://img.shields.io/badge/code_style-ruff-261230)
[![Test](https://github.com/constatza/torchalg/actions/workflows/testing.yml/badge.svg)](https://github.com/constatza/torchalg/actions/workflows/testing.yml)

## Table of contents

- [What torchalg is](#what-torchalg-is)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Capabilities](#capabilities)
  - [`pcg` vs `flexible_cg`](#pcg-vs-flexible_cg)
  - [Autograd and `differentiable`](#autograd-and-differentiable)
  - [Using a preconditioner](#using-a-preconditioner)
  - [Moving a preconditioner to a device or dtype](#moving-a-preconditioner-to-a-device-or-dtype)
  - [Running on GPU](#running-on-gpu)
  - [`TraceMode` and `IterationHistory`](#tracemode-and-iterationhistory)
  - [Comparing preconditioners](#comparing-preconditioners)
- [Correctness and testing](#correctness-and-testing)

## What torchalg is

`torchalg` solves linear systems `A x = b`. `A` is symmetric positive-definite
(SPD). `b` is the right-hand-side vector. `x` is the unknown. The method:
Conjugate Gradient, a Krylov-subspace iterative solver. The implementation:
native [PyTorch](https://pytorch.org/) tensors, `nn.Module`-based state,
device/dtype-aware, no numpy round-tripping.

It ships:

- `pcg` and `flexible_cg`: preconditioned and flexible CG solvers.
- A real preconditioner library: `Identity`, `JacobiPreconditioner`,
  `ILUPreconditioner` (dense ILU(0)), `IC0Preconditioner` (dense IC(0)),
  `ICholeskyPreconditioner`, `AMGPreconditioner` with `VCycleAMG`/`WCycleAMG`
  presets (dense smoothed-aggregation multigrid), `AdaptiveSAPreconditioner`
  (data-driven smoothed aggregation, a PyAMG port), `BootstrapAMGPreconditioner`
  (Bootstrap AMG, test-vector-driven coarsening/interpolation),
  `POD2GPreconditioner` (POD-basis two-grid), and `NeuralPreconditioner`
  (adapts a caller-supplied predictor model as a non-linear preconditioner).
- `TraceMode`/`IterationHistory` for opt-in per-iteration residual/solution
  traces.
- `run_cg_comparison`/`format_results_summary` to benchmark several
  preconditioners on the same system side by side. Comparison execution,
  result records, text presentation, and ranking live in separate modules.
- **CPU or GPU, no code changes**: the solver core has no hardcoded device —
  it runs wherever `A`/`b`/`x0` already live. Preconditioners that hold tensor
  state are `nn.Module` subclasses with registered buffers, so
  `.to(device="cuda")`/`.to(dtype=...)` genuinely moves and casts their
  internal state alongside the system tensors.

Scope: CG-family only. No GMRES, no BiCGSTAB. Use `torchalg` for SPD systems
on PyTorch-native, device/dtype-aware solvers with a real preconditioner
toolbox — not as a general sparse-linear-algebra package.

Correctness bar: scipy-CG equivalence, iteration-for-iteration.
`tests/benchmarks/exactness/test_residual_history_exactness.py` asserts `pcg`,
`flexible_cg(m_max=-1)`, and scipy's `cg` produce the *same iteration count*
and the *same residual norm at every iteration* — not just a matching final
answer.

Origin: ported from a reference implementation, `neuralls`
(sibling `dl-experiments` repository). History lives in `docs/plan.md`.
No runtime dependency on `neuralls`.

## Installation

`torchalg` is not yet published to PyPI — add it straight from GitHub with
[uv](https://docs.astral.sh/uv/):

```bash
uv add git+https://github.com/constatza/torchalg
```

Requires Python `>=3.14,<3.15` and torch `>=2.8.0,<3.0`. torchalg is pure
PyTorch and declares only that plain `torch` requirement — it never pins a
package index — so **your project chooses the torch build**:

- **NVIDIA GPU or CPU-only:** nothing more to do. PyPI's torch on Linux is the
  CUDA build, and it also runs on machines without a GPU.
- **AMD GPU (ROCm):** route torch and its Triton runtime to the PyTorch ROCm
  index in your own project:

  ```bash
  uv add torch "triton-rocm; sys_platform == 'linux'" \
      --index pytorch-rocm=https://download.pytorch.org/whl/rocm7.2
  ```

  This writes the index and a `[tool.uv.sources]` pin for both packages into
  your `pyproject.toml`, so a plain `uv sync` keeps the ROCm build from then on.
  `triton-rocm` must come from the same index as torch.
- **A library that supports both:** keep `torch` plain in your published
  dependencies and select the backend through conflicting dependency groups
  (groups are never published, so they never constrain your own consumers) —
  copy the `cuda`/`rocm` block from this repository's `pyproject.toml`.

Caveats: the ROCm index trails PyPI by about one minor torch release (2.14.1
versus 2.15.0 at the time of writing); PyPI's default CUDA build (13.2) needs an
NVIDIA driver R580 or newer. On an RX 6800 XT (gfx1030) with the rocm7.2
wheels, CSR sparse-matrix products (`csr @ x`) abort with a GPU memory access
fault inside rocSPARSE, while COO products, dense ops and the CSR triangular
solve work; see `docs/implementations/device-and-gpu.md`.

### Developing torchalg itself

`uv sync` installs PyPI's CUDA torch. On an AMD machine use the `rocm` group:

```bash
uv sync --group rocm
export UV_NO_SYNC=1   # otherwise `uv run` re-syncs back to the CUDA build
uv run python -c "import torch; print(torch.version.hip)"
```

uv has no persistent per-machine group or extra selection, so every later
`uv sync`/`uv lock` on that machine needs `--group rocm` again; `UV_NO_SYNC=1`
covers `uv run`/`uv add`. `uv sync --all-groups` fails by design (`cuda` and
`rocm` conflict).

## Quickstart

```python
import torch
from torchalg import pcg

torch.manual_seed(0)
n = 50
M = torch.randn(n, n, dtype=torch.float64)
A = M @ M.T + n * torch.eye(n, dtype=torch.float64)  # SPD
x_true = torch.randn(n, dtype=torch.float64)
b = A @ x_true

x, info = pcg(A, b)
print(f"converged={info.converged}, iterations={info.iterations}, residual={info.residual:.3e}")
```

Output:

```text
converged=True, iterations=14, residual=4.299e-07
```

## Capabilities

### `pcg` vs `flexible_cg`

`pcg`: standard preconditioned CG. Cheapest per iteration. Correct for any
fixed, linear preconditioner.

`flexible_cg`: adds explicit Gram-Schmidt reorthogonalization against
previous search directions. Required for iteration-dependent or non-linear
preconditioners (`ScheduledPreconditioner`, `NeuralPreconditioner`).

In exact arithmetic, both are identical for a fixed preconditioner
(Notay 2000) — the identity the scipy-equivalence benchmark checks.

```python
import torch
from torchalg import flexible_cg, pcg
from torchalg.preconditioners.implementations.jacobi import JacobiPreconditioner

torch.manual_seed(0)
n = 50
M = torch.randn(n, n, dtype=torch.float64)
A = M @ M.T + n * torch.eye(n, dtype=torch.float64)
b = torch.randn(n, dtype=torch.float64)

precond = JacobiPreconditioner()
precond.setup(A)
x_pcg, info_pcg = pcg(A, b, preconditioner=precond)
x_fcg, info_fcg = flexible_cg(A, b, preconditioner=precond)
print("pcg:", info_pcg.converged, info_pcg.iterations)
print("fcg:", info_fcg.converged, info_fcg.iterations)
```

Output:

```text
pcg: True 14
fcg: True 14
```

### Autograd and `differentiable`

Both solvers run under `torch.inference_mode()` by default: most callers use
an already-trained, frozen preconditioner and never backpropagate through
the solve itself, so tracking autograd by default would only build a graph
every call and immediately discard it. Pass `differentiable=True` to opt
back into normal gradient tracking, e.g. to unroll the iteration and train a
preconditioner end-to-end:

```python
x, info = pcg(A, b, preconditioner=precond, differentiable=True)
x.sum().backward()
```

### Using a preconditioner

Pass any preconditioner via the `preconditioner=` keyword.
`JacobiPreconditioner`: cheap diagonal scaling, the baseline.
`VCycleAMG`: smoothed-aggregation algebraic multigrid, V-cycle — stronger
convergence on larger, structured SPD systems:

```python
import torch
from torchalg import pcg
from torchalg.preconditioners.implementations.amg import VCycleAMG

n = 60
main = torch.full((n,), 4.0, dtype=torch.float64)
off = torch.full((n - 1,), -1.0, dtype=torch.float64)
A = torch.diag(main) + torch.diag(off, 1) + torch.diag(off, -1)  # tridiagonal SPD
b = torch.ones(n, dtype=torch.float64)

x_none, info_none = pcg(A, b)
x_amg, info_amg = pcg(A, b, preconditioner=VCycleAMG(A))
print("none:", info_none.converged, info_none.iterations)
print("amg :", info_amg.converged, info_amg.iterations)
```

Output:

```text
none: True 10
amg : True 3
```

### Moving a preconditioner to a device or dtype

Preconditioners with tensor state (`JacobiPreconditioner`, `ILUPreconditioner`,
`AMGPreconditioner`, ...) subclass `nn.Module` and register their state via
`register_buffer`. PyTorch's own `.to(...)` moves and casts every buffer at
once — no custom `torchalg` code involved:

```python
import torch
from torchalg.preconditioners.implementations.jacobi import JacobiPreconditioner

matrix = torch.diag(torch.tensor([2.0, 4.0, 1.0, 8.0], dtype=torch.float64))
precond = JacobiPreconditioner()
precond.setup(matrix)
print("before:", precond.inv_diag.dtype, precond.inv_diag.device)

precond = precond.to(dtype=torch.float32)
print("after :", precond.inv_diag.dtype, precond.inv_diag.device)

r = torch.tensor([4.0, 3.0, 2.0, 2.0], dtype=torch.float32)
print("z =", precond.apply(r))
```

Output:

```text
before: torch.float64 cpu
after : torch.float32 cpu
z = tensor([2.0000, 0.7500, 2.0000, 0.2500])
```

### Running on GPU

No separate GPU code path. The solver core has no hardcoded device — it runs
wherever `A`/`b`/`x0` already live. Move the preconditioner's buffers with
`.to(device=...)`, and the whole solve runs on GPU:

```python
import torch
from torchalg import pcg
from torchalg.preconditioners.implementations.jacobi import JacobiPreconditioner

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
n = 50
M = torch.randn(n, n, dtype=torch.float64, device=device)
A = M @ M.T + n * torch.eye(n, dtype=torch.float64, device=device)
b = torch.randn(n, dtype=torch.float64, device=device)

precond = JacobiPreconditioner().setup(A).to(device=device)
x, info = pcg(A, b, preconditioner=precond)
print(
    f"device={device}, x.device={x.device}, converged={info.converged}, iterations={info.iterations}"
)
```

Output on this machine (no CUDA device, so it ran on CPU — identical code
runs on GPU unmodified; see
`tests/solver/preconditioners/implementations/test_jacobi.py::TestJacobiIsNnModule::test_to_device_moves_the_buffer`):

```text
device=cpu, x.device=cpu, converged=True, iterations=14
```

### `TraceMode` and `IterationHistory`

Pass `trace_mode=TraceMode.FULL` to collect a per-iteration residual and
solution trace — useful for debugging or plotting convergence. Default
(`TraceMode.MINIMAL`): scalar residual norms only. `TraceMode.DISABLED`:
nothing.

```python
import torch
from torchalg import pcg
from torchalg.monitoring import TraceMode

torch.manual_seed(0)
n = 20
M = torch.randn(n, n, dtype=torch.float64)
A = M @ M.T + n * torch.eye(n, dtype=torch.float64)
b = torch.randn(n, dtype=torch.float64)

x, info = pcg(A, b, trace_mode=TraceMode.FULL)
print("iterations:", info.iterations)
print("residual_history_rel (first 3):", info.residual_history_rel[:3])
print("solution_vectors shape:", info.solution_vectors.shape)
```

Output:

```text
iterations: 12
residual_history_rel (first 3): (1.0, 0.528240813268935, 0.22923544349268415)
solution_vectors shape: torch.Size([13, 20])
```

### Comparing preconditioners

`run_cg_comparison`: runs `flexible_cg` across several preconditioners on the
same system, plus an unpreconditioned `"none"` baseline.
`format_results_summary`: renders the ranking:

```python
import torch
from torchalg import format_results_summary, run_cg_comparison
from torchalg.preconditioners.implementations.ilu import ILUPreconditioner
from torchalg.preconditioners.implementations.jacobi import JacobiPreconditioner

torch.manual_seed(0)
n = 50
M = torch.randn(n, n, dtype=torch.float64)
A = M @ M.T + n * torch.eye(n, dtype=torch.float64)
b = torch.randn(n, dtype=torch.float64)

results = run_cg_comparison(
    A,
    b,
    preconditioners={
        "jacobi": JacobiPreconditioner().setup(A),
        "ilu": ILUPreconditioner().setup(A),
    },
)
print(format_results_summary(results))
```

Output:

```text
Flexible CG results:
- jacobi             status=ok   iters= 14  rel_res=5.053e-07 (abs=4.422e-06)  exact_err=4.373e-07
- ilu                status=ok   iters=  1  rel_res=4.263e-16 (abs=3.730e-15)  exact_err=2.673e-16
- none               status=ok   iters= 14  rel_res=5.098e-07 (abs=4.461e-06)  exact_err=4.228e-07
```

## Correctness and testing

Discipline: tests before implementation. Every unit was ported only once a
failing test existed for it.

Oracle: [scipy](https://scipy.org/)'s `cg`. A `dev`-only dependency, never
imported from production code — used to check iteration counts, residual
histories, and solutions.

Boundaries: [tach](https://github.com/gauge-sh/tach) enforces the internal
dependency DAG. The solver core may depend on the `Preconditioner`
abstraction, never on concrete preconditioner classes. A violation fails the
build, not just a docstring claim.

As of this writing:

```bash
uv run pytest -q
# 305 passed, 1 skipped, 28 deselected

uv run pytest -q -m benchmark
# 28 passed, 306 deselected
```

The `benchmark` marker covers the opt-in scipy-equivalence and Notay (2000)
paper-reproduction suite — the actual mathematical-correctness gate for the
whole solver stack.
