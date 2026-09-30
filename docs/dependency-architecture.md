# Dependency Architecture

This document is the policy implemented by `tach.toml`. The policy follows
dependency inversion and the single-responsibility principle: stable,
low-level concepts do not know orchestration or concrete features, while
higher-level code depends on the narrowest suitable abstraction.

## Principles

1. Dependencies point from orchestration toward policy, records, abstractions,
   and pure functions. They never point back toward orchestration.
2. Models are immutable data. They do not import monitoring, strategies,
   solvers, preconditioners, or presentation code.
3. Protocols and abstract preconditioner interfaces do not import concrete
   implementations.
4. Solver algorithms depend on the preconditioner abstraction. Only a
   composition root may select concrete implementations.
5. A feature core does not identify or special-case a feature layered on top
   of it. In particular, AMG must not import POD; POD is an AMG client.
6. Public package facades are re-export boundaries, not dependency shortcuts.
   Internal modules import the precise module that owns a symbol. Composition
   roots may import a concrete facade when selecting implementations is their
   responsibility.
7. Presentation and recommendation policy consume result records. They do not
   invoke solvers or import preconditioners.
8. Shared code is moved downward only when its meaning is genuinely shared.
   A helper is not made generic merely to silence an architecture check.
9. Type-checking-only imports follow the same conceptual direction as runtime
   imports even when Tach does not emit a runtime edge for them.
10. Rules are declared at the smallest boundary with architectural meaning.
    Pure leaves and high-risk feature seams use file-level rules; cohesive
    facades remain package boundaries. This avoids both permissive umbrella
    rules and a brittle rule for every incidental helper.

## Layers

Dependencies may flow downward in this table. A lower layer must not depend on
a higher one.

| Layer | Modules | May depend on |
| --- | --- | --- |
| Pure primitives | `torchalg.utils.*`, `torchalg.analysis.spectral` | PyTorch and the standard library only |
| Immutable records | `torchalg.models.*` | Other explicitly listed model records |
| Monitoring and policies | `torchalg.monitoring.*`, `torchalg.strategies.*` | Models and pure primitives |
| Preconditioner abstractions | `preconditioners.base`, `preconditioners.ports` | Pure primitives when necessary |
| Concrete kernels/adapters | Individual `preconditioners.implementations.*` modules | Preconditioner abstractions, local lower-level kernels, and pure primitives |
| AMG foundation and engine | AMG protocols, transfers, hierarchy, kernels, cycle, engine | Preconditioner abstractions and lower AMG components |
| AMG/POD presets | AMG setup algorithms, variants, POD | AMG engine/foundation and local feature components |
| Solver algorithms | `base`, `conjugate_gradient` | Models, monitoring, strategies, pure primitives, and preconditioner abstractions |
| Composition | `factories` | Solver algorithms and concrete-preconditioner facade |
| Application workflows | `comparison.runner` | Composition and comparison records |
| Presentation/policy | `comparison.presentation`, `comparison.recommendations` | Comparison records only |

## Model boundaries

- `config`, `context`, `diagnostics`, and `history` are leaves.
- `result` may depend on `diagnostics`.
- `state` may depend on `history`.
- `protocols` may describe `history` and `result`, but no model imports a
  solver, monitor, strategy, or preconditioner.

## Monitoring and strategy boundaries

- Monitoring storage and trace-mode selection are leaves.
- Iteration history may compose storage, trace mode, and numerical primitives.
- Monitoring analysis may use pure energy operations.
- Norm policy may use energy operations; convergence may use norm policy.
- Direction policy may use solver state, orthogonalization policy, and stable
  numerical primitives.
- These packages must not import solver orchestration or concrete
  preconditioners.

## Preconditioner boundaries

`preconditioners.base` and `preconditioners.ports` are the stable abstraction
layer. Concrete preconditioners may depend on them; the reverse is forbidden.
Concrete implementations must not import solver state or solver strategies.

The concrete-implementation package facade exists for `factories`, the
composition root. Concrete siblings should use precise imports rather than the
facade, because a facade import silently couples them to every re-exported
implementation.

## AMG and POD boundaries

The AMG subsystem is ordered as follows:

1. Pure numerical kernels and protocols.
2. Transfer operators and hierarchy records.
3. Smoothers and cycles.
4. Generic `AMGPreconditioner` engine.
5. Coarsening algorithms and setup workflows.
6. Presets and variants.

The generic engine accepts structural abstractions. It must not use
`isinstance` checks or imports to recognize a concrete coarsening strategy.
Feature-specific display or configuration belongs to that feature's preset.

POD is a client of AMG transfer/cycle/engine abstractions. The permitted
direction is `pod -> amg`; `amg -> pod` is forbidden. POD modules import exact
AMG modules, never the broad `amg` facade.

Bootstrap-AMG kernels remain independently testable. Compatible relaxation
must not depend on the higher-level algebraic-distance feature merely to reuse
a graph helper; genuinely shared graph operations belong in a neutral kernel.

## Solver and workflow boundaries

- `base` and `conjugate_gradient` depend on `Preconditioner`, never concrete
  implementations.
- `factories` is the only solver composition root allowed to import the broad
  concrete-preconditioner facade.
- `comparison.runner` calls factories and produces comparison records.
- Comparison models have no internal dependencies. Formatting and ranking
  depend only on those models.

## Future sparse extension

If sparse implementations are added, their public and implementation boundary
is `torchalg.sparse`. Do not create a parallel `torchalg._sparse` package by
default: unlike PyTorch, this project has no compiled-extension boundary that
would justify the extra delegation layer. `torchalg.sparse.__init__` should be
a small, curated facade over implementation modules in the same package.

The intended shape is:

```text
torchalg/
├── models/                       # shared records
├── monitoring/                   # shared diagnostics
├── strategies/                   # shared mathematical policy
├── preconditioners/              # dense/default implementations
└── sparse/
    ├── __init__.py               # curated public exports
    ├── operators.py
    ├── validation.py
    ├── factories.py
    └── preconditioners/
        ├── jacobi.py
        ├── ic0.py
        ├── ilu.py
        └── amg/
```

Only representation-sensitive operations belong under `torchalg.sparse`:
sparse storage validation, index manipulation, factorization, triangular
solves, sparse Galerkin products, and sparse-specific preconditioners. Models,
convergence policy, diagnostics, and representation-independent Krylov control
flow remain shared rather than being copied into a sparse mirror.

The dependency direction is:

```text
shared models, protocols, policies, and pure utilities
                         ↑
                  torchalg.sparse
```

Future Tach rules must enforce all of the following:

- Shared modules do not import `torchalg.sparse`.
- Dense/default and sparse concrete implementations do not import each other.
- Sparse implementations depend on shared abstractions or sparse-local
  kernels, not dense implementations.
- `torchalg.sparse.__init__` may re-export sparse children, but sparse
  implementation modules import precise siblings rather than their facade.
- A sparse factory may select sparse implementations; generic solver
  orchestration continues to depend on operator and preconditioner protocols.
- Sparse operations never densify implicitly. Any dense conversion is an
  explicit public operation with documented cost and device implications.
- Representation dispatch occurs once at construction or composition time,
  never inside an iterative numerical loop.
- Sparse kernels introduce no avoidable CPU/GPU synchronization or host
  round-trips.
- SciPy remains a test oracle or an explicitly optional CPU adapter, never a
  production sparse backend.

A private `torchalg._sparse` layer becomes justified only if a real boundary
appears later, such as compiled extensions, multiple replaceable sparse
backends, or a stable public facade that must be versioned independently from
its implementation. It must not be introduced merely to imitate PyTorch's
compiled `torch._C._sparse` delegation.

## Facades

`torchalg.monitoring`, `torchalg.preconditioners.implementations`, AMG, POD,
and comparison package `__init__` files are public re-export surfaces. Their
outgoing edges are allowed only to the children they expose. Importing such a
facade from another implementation module is disallowed when a precise module
can be imported instead.

## Enforcement

`tach check` verifies forbidden edges. `tach check --exact` additionally
detects stale permissions and should pass once known violations are resolved.
Rules must not be loosened simply because an implementation currently violates
them: first decide whether the policy or the implementation owns the mistake,
document any genuine exception here, and only then update `tach.toml`.
