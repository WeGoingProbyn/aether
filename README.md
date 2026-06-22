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
- `aer`: atmospheric models and stages (compressible Euler, moisture, HEVI)
- `terra`: surface thermal slab and inert terrain (elevation, land/ocean/ice)
- `thalassa`: thermodynamic ocean column on a cube-sphere shell
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
 │  ┌───┴─────┐ ┌──┴───┬───────┬──────────┬───────┬───────┬────────┐
 │  │ syzygy  │ │ aer  │ terra │ thalassa │ lumen │gravitas│ future │
 │  └─────────┘ └──┬───┴───────┴──────────┴───────┴───────┴────────┘
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

## Current Capabilities

The reference world is a coupled planetary atmosphere–ocean on a cube-sphere:

- **Well-balanced compressible atmosphere.** The finite-volume scheme
  reconstructs each cell's state to shared faces along the local hydrostatic
  profile, so a fluid at rest holds `∇p = ρg` to machine precision instead of
  drifting. Gravity is an analytic radial geopotential supplied by the kernel.
- **HEVI time stepping.** Horizontally-explicit / vertically-implicit
  integration removes the vertical acoustic CFL limit that dominates thin
  shells, running through the same partitioned nexus path as the explicit
  solver (per-panel dispatch, radial columns).
- **Conservative air–sea coupling.** Evaporation, saturation, radiation, and
  sea-surface temperature exchange close the moist energy budget across meshes
  (`syzygy` interface fluxes), so the demo is stable end-to-end.
- **Geographic query layer.** `eidolon` exposes a read-only, thread-safe
  `Quantity` API (`sample_scalar` / `sample_wind` / `reduce_scalar`) over an
  interpolated snapshot, addressed in lat/lon via `tessera`'s geographic index.
- **First-class terrain.** Inert surface elevation and a land/ocean/ice mask,
  with orographic lift as the first terrain → atmosphere coupling.
- **Pluggable backends.** Explicit, implicit (matrix-free GMRES), IMEX, and
  hybrid solvers sit behind one `FvmBackend` trait; parallelism lives in nexus
  (N serial solvers over N partitions), not inside a backend.

## Development

```sh
cargo run -p sandbox
```

## Maintenance Note

Aether is a personal project maintained in spare time. Issues and pull requests
are welcome, but API stability and turnaround times are not guaranteed.
