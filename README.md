# Aether

Aether is a Rust workspace for **composable multiphysics world simulation** — a
library for growing a living planet (atmosphere, ocean, terrain, radiation,
orbits) and reading or rendering what emerges, rather than a single fixed engine.

It is a work in progress: some systems are mature, some crates are experimental,
and APIs may change without notice.

## Two ways in

Aether is designed so you can engage at the depth you need:

- **"I just want to read or render a world."** You never touch conserved-variable
  layouts or cube-sphere math. `eidolon` gives you an engine-neutral render IR, a
  reference Bevy backend, VTK export, and a thread-safe geographic **query API**
  (`sample` temperature/wind at a lat/lon, reduce over a region). Start here:
  **[docs/rendering.md](docs/rendering.md)**.

- **"I want to understand or extend the physics."** Aether is a stack of
  swappable layers held together by one rule — *physics modules own logic, never
  global state or geometry*. Start with **[docs/architecture.md](docs/architecture.md)**,
  then the numerics in **[docs/physics.md](docs/physics.md)**, and the recipes in
  **[docs/extending.md](docs/extending.md)**.

## Quick start

```sh
cargo run -p sandbox        # the reference world: terrain + ocean + moist atmosphere, rendered
cargo test --workspace      # run everything headless
```

`sandbox` is the worked example of how the pieces fit together — read
[`sandbox/docs/overview.md`](sandbox/docs/overview.md) alongside the code.

## What it does today

The reference world is a coupled planetary atmosphere–ocean–surface on a rotating
cube-sphere:

- **Well-balanced compressible atmosphere.** The finite-volume scheme
  reconstructs each cell to shared faces along the local hydrostatic profile, so
  a fluid at rest holds `∇p = ρg` to machine precision instead of drifting.
- **HEVI time stepping.** Horizontally-explicit / vertically-implicit
  integration removes the vertical acoustic CFL limit that dominates thin shells,
  running through the same partitioned scheduler path as the explicit solver.
- **Conservative air–sea coupling.** Evaporation, saturation, radiation, and
  sea-surface-temperature exchange close the moist energy budget across meshes.
- **First-class terrain.** Inert elevation + a land/ocean/ice mask + a per-cell
  albedo field, with orographic lift as the first terrain → atmosphere coupling.
- **Geographic query layer.** A read-only, thread-safe `Quantity` API over an
  interpolated snapshot, addressed in lat/lon and decoupled from internal field
  names.
- **A timescale spectrum.** A multi-rate scheduler runs the fast atmosphere and
  slow ocean on separate clocks; a climatology regime advances long game-time by
  holding the solver between bursts, so a 4X-scale consumer and a weather-scale
  consumer share one model.
- **Pluggable solver backends.** Explicit, implicit (matrix-free GMRES), IMEX,
  and hybrid sit behind one trait; parallelism lives in the scheduler, not the
  backend.

## The one rule

> Physics crates do not own global state or geometry.

A physics module registers the fields it produces, names the fields it reads, and
operates on borrowed state the scheduler hands it. State has a single owner
(`pleroma`); ordering is derived from declared reads/writes (`nexus`); space is a
shared structure (`tessera`). This is what makes worlds *assembled* rather than
hard-coded, and what keeps the system extensible. See
[docs/architecture.md](docs/architecture.md).

## Documentation map

- [docs/architecture.md](docs/architecture.md) — the layers, the central rule,
  the execution model, where extensibility comes from.
- [docs/physics.md](docs/physics.md) — finite-volume numerics, well-balanced
  atmosphere, HEVI, coupling, the timescale spectrum.
- [docs/rendering.md](docs/rendering.md) — render IR, playback, the query API,
  VTK export (the consumer's guide).
- [docs/extending.md](docs/extending.md) — recipes: new field, stage, physics
  crate, fluid, coupling, mesh, backend, render layer, cadence.
- Every crate has its own `docs/overview.md` (linked below).

## Crate map

| Crate | Role |
|---|---|
| [`aether`](aether/docs/overview.md) | Runtime facade: systems, worlds, ticks, regime-aware advance |
| [`sandbox`](sandbox/docs/overview.md) | Integration rig: assembles & renders the reference world |
| [`cosmo`](cosmo/docs/overview.md) | Immutable body/system definitions and initial conditions |
| [`tessera`](tessera/docs/overview.md) | Mesh geometry, topology, partitioning, coupling, geography |
| [`pleroma`](pleroma/docs/overview.md) | Typed field/resource registry — the only mutable state |
| [`nexus`](nexus/docs/overview.md) | Dependency-aware DAG scheduler & multi-rate execution |
| [`continuum`](continuum/docs/overview.md) | Domain-neutral finite-volume solver |
| [`tempus`](tempus/docs/overview.md) | Generic time-integration kernels |
| [`aer`](aer/docs/overview.md) | Compressible moist atmosphere (Euler, HEVI, microphysics) |
| [`terra`](terra/docs/overview.md) | Surface thermal slab and first-class terrain |
| [`thalassa`](thalassa/docs/overview.md) | Thermodynamic ocean column |
| [`gravitas`](gravitas/docs/overview.md) | Newtonian n-body gravity stage |
| [`lumen`](lumen/docs/overview.md) | Gray-atmosphere radiative transfer |
| [`syzygy`](syzygy/docs/overview.md) | Cross-mesh coupling semantics |
| [`chronos`](chronos/docs/overview.md) | Timescale spectrum: climatology + regimes |
| [`eidolon`](eidolon/docs/overview.md) | Engine-neutral render IR, query API, VTK export |
| [`utility`](utility/docs/overview.md) | Math, threads, profiler, serialization, ID vocabulary |
| [`utility_macros`](utility_macros/docs/overview.md) | Derives and `#[profile]` proc-macros |

## Dependency and data topology

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
 │  ┌───┴────┐ ┌───┴──┬──────┬────────┬──────┬────────┬─────────┐
 │  │ syzygy │ │ aer  │terra │thalassa│lumen │gravitas│ chronos │
 │  └────────┘ └──┬───┴──────┴────────┴──────┴────────┴─────────┘
 │                │ (physics consume continuum + tempus)
 │                ▼
 │           ┌──────────┐
 ├──────────▶│continuum │   (numerical-methods library)
 │           └──────────┘
 │  ┌──────────┐
 └─▶│ eidolon  │   (read-only viewer + query over pleroma + tessera)
    └────┬─────┘
         ▼
    ┌──────────┐
    │ sandbox  │   (cosmo → tessera → pleroma → nexus → physics → eidolon)
    └──────────┘
```

## Development

```sh
cargo build --workspace
cargo test --workspace
cargo fmt --all                 # 2-space indent, 80 cols, no tabs (rustfmt.toml)
cargo run -p sandbox            # the rendered demo
cargo run -p continuum --example sod_shock
```

The repo is indexed by the `codegraph` CLI for navigation (`codegraph query`,
`callers`, `impact`, `context "<task>"`).

## Maintenance note

Aether is a personal project maintained in spare time. Issues and pull requests
are welcome, but API stability and turnaround times are not guaranteed.

Licensed under Apache-2.0.
