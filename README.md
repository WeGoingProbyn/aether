# Aether

Aether is a work-in-progress Rust workspace for composable multiphysics
simulation.

The project is not complete. Some systems are implemented, some crates are
still experimental, and APIs may change without notice.

## Overview

Aether is built around a small set of boundaries:

- `cosmo` describes immutable starting conditions.
- `tessera` describes where simulation happens: meshes, topology, and coupling
  between domains.
- `pleroma` owns mutable world state: fields, resources, and simulation values.
- `nexus` schedules physics stages from declared read/write dependencies.
- Physics crates transform state without owning geometry or global storage.
- `eidolon` reads state and geometry to produce presentation data.

The intent is to keep simulation configuration, spatial structure, mutable
state, numerical methods, physics, orchestration, and rendering independent
enough that each layer can evolve without turning the project into one
monolithic engine.

## Crate Map

- `aether`: top-level runtime facade for systems, worlds, and ticks
- `utility`: shared math, domain IDs, errors, graphs, and threading utilities
- `cosmo`: immutable body/system definitions and initial-condition inputs
- `tessera`: mesh geometry, topology, partitioning, and mesh coupling
- `pleroma`: typed field/resource registry for mutable simulation state
- `nexus`: dependency-aware stage scheduler and execution engine
- `continuum`: domain-neutral numerical methods and solver utilities
- `tempus`: generic time-integration kernels
- `aer`: atmospheric models and stages
- `terra`: surface/geophysical models and stages
- `gravitas`: orbital and n-body gravity stages
- `lumen`: radiative transfer stages
- `syzygy`: coupling semantics between physics modules
- `eidolon`: engine-neutral rendering and export IR
- `sandbox`: integration playground for wiring the pieces together

## Dependency and Data Topology

```text
                 ┌──────────┐
                 │ utility  │
                 └────┬─────┘
                      │
         ┌────────────┼─────────────┐
         ▼            ▼             │
    ┌─────────┐  ┌─────────┐        │
 ┌──┤ tessera │  │  cosmo  │        │
 │  └────┬────┘  └────┬────┘        │
 │       │            │             │
 │       ▼            │             │
 │  ┌─────────┐       │             │
 ├──┤ pleroma │◀──────┘             │
 │  └─────────┘                     │
 │       ▲                          │
 │       │ (nexus mutates pleroma)  │
 │       ▼                          │
 │  ┌─────────┐                     │
 │  │  nexus  │◀───┐                │
 │  └─────────┘    │                │
 │      ┌──────────┤                ▼
 │  ┌───┴─────┐ ┌──┴───┬───────┬──────────┬────────────────┐
 │  │ syzygy  │ │ aer  │ terra │ gravitas │ future physics │
 │  └─────────┘ └──┬───┴───────┴──────────┴────────────────┘
 │                 │
 │                 │
 │                 │
 │                 ▼
 │           ┌──────────┐
 ├──────────▶│continuum │   (numerical-methods library;
 │           └──────────┘    consumed by physics crates)
 │
 │  ┌──────────┐
 └─▶│ eidolon  │   (read-only viewer over pleroma + tessera)
    └────┬─────┘
         ▼
    ┌──────────┐
    │ sandbox  │   (integration rig: cosmo → pleroma → nexus → physics → eidolon)
    └──────────┘
```

## Execution Model

Aether does not run as a single hard-coded pipeline. A world is assembled from
configuration, meshes, state, and registered physics stages:

1. `cosmo` provides initial conditions.
2. `tessera` builds the spatial domain.
3. `pleroma` registers fields and resources for that domain.
4. Physics crates register stages with declared reads and writes.
5. `nexus` builds a dependency graph and runs stages in a valid order.
6. `eidolon` can read the resulting state for rendering or export.

The main architectural rule is that physics modules do not own global state or
geometry. They operate on borrowed state provided by the scheduler.

## Development

```sh
cargo run -p sandbox
```

## Maintenance Note

Aether is a personal project maintained in spare time. Issues and pull requests
are welcome, but API stability and turnaround times are not guaranteed.
