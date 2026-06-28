# Architecture

Aether is a **composable multiphysics world-simulation library**. It is not a
single engine with a fixed pipeline; it is a set of layers you assemble into a
world, run, and read. This document explains those layers, the one rule that
holds them together, and how the design stays extensible.

If you only want to *read or render* a finished world, skip to
[rendering.md](rendering.md). If you want the numerics, see
[physics.md](physics.md). To add your own physics, see [extending.md](extending.md).

## The layers

Aether separates seven concerns that engines usually fuse together. Each is a
crate (or small group of crates), and each can evolve without rewriting the
others:

| Concern | Crate(s) | Owns |
|---|---|---|
| Configuration / initial conditions | `cosmo` | Immutable body & system definitions |
| Spatial structure | `tessera` | Meshes, topology, partitions, coupling, geography |
| Mutable state | `pleroma` | Every field and resource the simulation evolves |
| Orchestration | `nexus` | The stage DAG and its parallel execution |
| Numerical methods | `continuum`, `tempus` | Domain-neutral solvers & time integrators |
| Physics | `aer`, `terra`, `thalassa`, `gravitas`, `lumen`, `syzygy`, `chronos` | The *logic* of each process |
| Presentation | `eidolon` | Read-only render/diagnostic/query views |

`aether` is the thin runtime facade that ties a world together; `sandbox`
assembles a concrete reference world and renders it. `utility` /
`utility_macros` are the shared foundation (math, threads, profiling,
serialization, the ID vocabulary).

## The central rule

> **Physics crates do not own global state or geometry.**

A physics module owns the *logic* of a step and nothing else. It does not hold a
mesh, it does not hold a buffer of state, and it does not reach into a global. At
world-setup it registers the field(s) it produces in `pleroma` and names the
`FieldKey`s it reads and writes. At run-time `nexus` hands it a borrowed,
scope-limited `WorldAccess` containing exactly those fields plus the read-only
geometry it needs.

This single rule is what makes the system composable:

- Two stages that touch disjoint fields are automatically parallelizable —
  `nexus` discovers that from their declarations, with no manual threading.
- A physics crate can be swapped, added, or removed without touching the others,
  because the contract between them is *data* (`FieldKey`s on a mesh), not a
  call graph.
- State has exactly one owner (`pleroma`), so there is no aliasing question to
  reason about across modules.

## The execution model

A world is **assembled**, not hard-coded:

1. `cosmo` provides initial conditions (a star, a planet, atmosphere makeup).
2. `tessera` builds the spatial domain (cube-sphere shells, radial stacks,
   partitions, cross-mesh couplers).
3. `pleroma` registers the fields and resources for that domain.
4. Physics crates register `Stage`s, each declaring its reads and writes.
5. `nexus` turns those declarations into a dependency DAG and runs
   non-conflicting stages in parallel on the shared thread pool.
6. `eidolon` reads the resulting state for rendering, file export, or geographic
   queries.

```text
cosmo ── initial conditions
  │
tessera ── meshes, partitions, couplers
  │
pleroma ── fields + resources (the only mutable state)
  │            ▲
nexus ── DAG ──┘  (borrows state to stages, runs them in parallel)
  │
physics stages (aer / terra / thalassa / gravitas / lumen / syzygy / chronos)
  │            using continuum (FVM) + tempus (integrators)
  ▼
eidolon ── render IR, VTK export, geographic query API
```

A `World::tick` advances the whole DAG by one step. `World::advance` adds the
timescale-regime logic on top (see [physics.md](physics.md#the-timescale-spectrum)).

## Dependency direction

```text
utility ─┬─ cosmo ───┐
         ├─ tessera ─┴─ pleroma ── nexus ──┬─ aer / terra / thalassa
         │                                 ├─ gravitas / lumen / syzygy / chronos
         ├─ continuum   (FVM library)      └─ (physics also use continuum + tempus)
         ├─ tempus      (integrators; depends only on utility)
         └─ eidolon     (read-only over pleroma + tessera)
                        aether + sandbox assemble everything
```

Physics crates depend on `nexus` for their state-access vocabulary. `nexus`
re-exports `pleroma::prelude::*`, so a typical physics `Cargo.toml` lists
`nexus + tessera + continuum + utility` and never `pleroma` directly. Nothing in
the simulation layer depends on a rendering engine; `eidolon` depends *on* the
simulation layer, never the reverse.

## Where extensibility comes from

Extensibility is not a plugin system bolted on top — it falls out of the
boundaries above:

- **Data-flow contracts, not call graphs.** Because stages communicate through
  named fields on meshes, a new process only needs to read the fields it
  consumes and write the fields it produces. `nexus` schedules it correctly with
  no edges added by hand.
- **Domain-neutral numerics.** `continuum` is generic over dimension, state
  size, the conservation law, and the numerical flux. A new fluid is a new
  `ConservationLaw` impl, not a new solver.
- **A stable read seam.** Consumers read through `eidolon`'s semantic `Quantity`
  vocabulary in geographic coordinates, *decoupled from the internal
  `FieldName`/`MeshType` enums*. Physics internals can churn while the public
  read contract stays put.
- **Coupling is explicit and typed.** Same-mesh dependencies are ordinary DAG
  edges; cross-mesh exchange goes through `syzygy` couplers. Either way the
  coupling is declared data, inspectable and testable.
- **Adaptivity is a versioned seam, not a special case.** Meshes can refine at
  runtime: dense `CellId` stays the hot-path key, and a `TopologyEpoch` +
  `CellRemap` (broadcast on the event bus) is how state, query, render, and
  checkpoints survive a re-mesh. See [amr.md](amr.md).

See [extending.md](extending.md) for the concrete recipes (new field, new stage,
new physics crate, new coupling, new mesh, new solver backend).

## Per-crate detail

Every crate has a `docs/overview.md` describing what it does and how it fits.
Start from the [crate map in the README](../README.md#crate-map).
