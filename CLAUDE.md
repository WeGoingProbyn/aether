# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Build
cargo build --workspace
cargo build --workspace --release

# Test all crates
cargo test --workspace

# Test a specific crate
cargo test -p continuum
cargo test -p tessera

# Run a single integration test file (by file stem)
cargo test --test mesh_topology_invariants
cargo test --test solver_parallel_consistency
cargo test --test atm_surface_coupling

# Run a single test by name
cargo test -p continuum test_name

# Run the integration rig
cargo run -p sandbox

# Run an example
cargo run -p continuum --example sod_shock

# Format
cargo fmt --all
cargo fmt --all -- --check
```

**Formatting rules** (`rustfmt.toml`): 2-space indent, max 80 columns, no hard tabs.

## Workspace Structure

Aether is a composable multiphysics simulation workspace. The design keeps
configuration, spatial structure, mutable state, numerical methods, physics,
orchestration, and rendering as independent layers. The central rule:
**physics crates do not own global state or geometry** — they operate on
borrowed state handed to them by the scheduler.

Crates (see workspace `Cargo.toml`):

| Crate | Role |
|-------|------|
| `utility` | Shared foundation: math types, thread pool, profiler, logger, serialization, domain IDs, errors, graphs |
| `utility_macros` | Proc-macros: `Serialize`/`Deserialize`/`StateDiagnostics` derives, `#[profile]` |
| `cosmo` | Immutable body/system definitions and initial-condition inputs |
| `tessera` | Mesh geometry, topology, partitioning, cube-sphere, radial stacks, mesh coupling |
| `pleroma` | Typed field/resource registry — owns all mutable world state |
| `nexus` | Dependency-aware DAG stage scheduler and execution engine |
| `continuum` | Domain-neutral finite-volume numerical methods / solver |
| `tempus` | Generic time-integration kernels (RK, Velocity-Verlet) |
| `aer` | Atmospheric physics models and stages |
| `terra` | Surface thermal physics models and stages |
| `gravitas` | Newtonian n-body gravity stage |
| `lumen` | Radiative-transfer (gray-atmosphere) stages |
| `syzygy` | Coupling semantics between physics modules |
| `eidolon` | Engine-neutral presentation IR + Bevy backend |
| `aether` | Top-level runtime facade: systems, worlds, ticks |
| `sandbox` | Integration rig wiring the pieces together |

### Layering / dependency direction

```
utility ─┬─ cosmo ──┐
         ├─ tessera ─┴─ pleroma ── nexus ──┬─ aer / terra / gravitas / lumen / syzygy
         │                                 └─ (physics consume continuum + tempus)
         ├─ continuum   (numerical-methods library)
         ├─ tempus      (time integrators; depends only on utility)
         └─ eidolon     (read-only viewer over pleroma + tessera)
                        sandbox / aether assemble everything
```

Physics crates depend on `nexus` for their state-access vocabulary. Nexus
re-exports `pleroma::prelude::*`, so a typical physics `Cargo.toml` lists
`nexus + tessera + continuum + utility` and never `pleroma` directly.

## Architecture

### Execution model

A world is assembled rather than hard-coded into a pipeline:

1. `cosmo` provides initial conditions.
2. `tessera` builds the spatial domain (meshes, topology, partitions).
3. `pleroma` registers fields and resources for that domain.
4. Physics crates register `Stage`s with declared reads/writes.
5. `nexus` builds a dependency graph and runs non-conflicting stages in parallel.
6. `eidolon` reads the resulting state for rendering or export.

### utility

- **Math** (`maths/`): `Vector<T, D>`, `Matrix<T, R, C>`, `Quaternion<T>`.
  `Vector` wraps `Matrix<T, D, 1>`. Constructors take arrays:
  `Vec3::new([x, y, z])` — NOT separate args.
- **Thread pool** (`thread/`): work-stealing `Pool`. APIs include
  `parallel_for()`, `execute()` (task DAG), `spawn()`, plus scoped schedulers
  (`ScopedScheduler`, `ScopedTaskGraph`) used by nexus/aether.
- **Profiler** (`profiler.rs`): `#[profile]` attribute inserts a `SpanGuard` at
  fn entry; thread-local, no locking. `inline_profile!`/`end_profile!` for
  blocks. `Profiler::flush_profiler()` / `Profiler::print()`.
- **Logger** (`logger.rs`): levels Trace→Fatal, `trace!`…`fatal!` macros;
  `Logger::init(sinks, level)` once.
- **Serialization** (`serial/`): custom `Serialize`/`Deserialize` traits + derives.
  Backends: JSON and VTK XML (`XmlVtuWriter` / `XmlPvtuWriter`).
- **Domain** (`domain.rs`): the shared ID vocabulary — `WorldId`, `SystemId`,
  `CellId`, `Point<D>`, plus the keys below.

### pleroma — mutable state registry

The single owner of simulation state. Key vocabulary lives in
`pleroma::prelude` (and is re-exported by nexus):

- **Keys** (defined in `utility::domain`): `FieldKey { mesh, name }`,
  `FieldName` enum, `MeshKey(MeshType)`, `MeshType`, `ResourceKey`
  (non-mesh-bound state, e.g. `Bodies`, sun direction).
- **Access** (`core/access.rs`): `WorldAccess` is the borrowed handle stages
  receive — `read::<S>(FieldKey)`, `write::<S>(FieldKey)`,
  `resource::<R>(ResourceKey)`, `resource_mut`. `ScheduleAccess` for the planner.
- **Field storage** (`core/storage.rs`): `FieldStorage<N>` trait with
  - `SoaField<N>` (Structure of Arrays) — fast component-wise sweeps
  - `AosField<N>` (Array of Structures) — fast per-cell access
  - `LocalPartitionField<N>` — owned + ghost values for a partition;
    `gather_partition_field` / `scatter_partition_owned` move data between
    global fields and partition-local buffers.
- `exchange_ghosts` synchronises ghost layers across partitions.

`Pleroma` is the top-level handle for setup/init code.

### nexus — scheduler

- `Stage` trait: `name()`, `reads()/writes()` (`FieldKey`s),
  `resource_reads()/resource_writes()` (`ResourceKey`s), `run(StageContext)`,
  and an optional `plan()` returning a `StagePlan` (`Static` task list or a
  `Program` for dynamic, multi-task stages such as partitioned solves).
- Nexus uses the declared dependencies to build a DAG and execute
  non-conflicting stages in parallel on the `Pool`.
- `CompiledNexus` is the runnable form. `WorldConstants` (and friends like
  `AtmosphereConstants`, `RadiationConstants`) carry per-world parameters.

### tessera — geometry

- `Mesh<D>` super-trait = `CellGeometry<D> + FaceGeometry<D> + Topology`.
- `StructuredBlock<D>` — Cartesian block with precomputed centroids, volumes,
  face areas, connectivity.
- `cube_sphere.rs` / `radial_stack.rs` — curvilinear cubed-sphere shells and
  stacked radial layers for planetary atmospheres.
- **Curvilinear metrics gotcha**: a cube-sphere shell's *volume* metric must
  not be reused as a *face-area* metric. `face_sqrt_det_metric` defaults to
  `sqrt_det_metric` for Cartesian domains but must be overridden for
  curvilinear ones (otherwise characteristic lengths are off by a factor of r).
- `partition.rs` — domain decomposition into stripes with ghost layers.
- `coupling.rs` + `world_mesh.rs` (`Tessera`) — mesh coupling and the
  multi-mesh container held by a world.

### continuum — numerical methods

Generic over dimension `D`, state size `N`, `ConservationLaw<D,N>`, and
`NumericalFlux<D,N>`. It is a **dumb serial CPU solver**; parallelism is
orchestrated by nexus/aether running N solvers over N partitions (a backend
trait leaves room for other backend types later).

- `model.rs`: `ConservationLaw` (`fix_state`, `flux`, `max_wave_speed`,
  per-cell `source`) and `NumericalFlux` (`RusanovFlux`).
- `solver.rs`: `FvmSolver` drives integration; `SolverConfig` carries CFL,
  `dt_max`, `TimeIntegration` (`ForwardEuler` | `Rk2`). Internal `Scratch`
  buffers avoid hot-loop allocation — do not remove them.
- `cpu.rs`: `CpuBackend` partitioned execution.
- `boundary.rs`: `BoundaryCondition<D,N>` (`ghost_state`), `BoundaryRegistry`,
  built-ins `Transmissive` / `ReflectiveWall`. `BoundaryTag` variants include
  `Top/Bottom/Left/Right/Front/Back/Wall/Ground/Inflow/Outflow/AtmosphereEdge`.
- `output.rs`: `LawFieldSchema<D,N>` maps states to named arrays; VTK snapshot
  writing for ParaView.

### Physics crates (aer / terra / gravitas / lumen / syzygy)

All follow the same contract: own the *logic* of a step, store nothing
globally. They register their field(s) at world setup, name the `FieldKey`s
they consume/produce, and step state via `tempus` kernels.

- `aer`: compressible Euler atmosphere (`EulerAtmosphereStep`,
  `BackgroundCorrectedEuler3D`), microphysics, tracers, radiation coupling.
- `terra`: surface thermal slab — `dT/dt = NetSurfaceFlux / heat_capacity_per_area`.
- `gravitas`: `KeplerStage` integrates `BodyState<D>` (a `ResourceKey::Bodies`)
  with `tempus::VelocityVerlet`.
- `lumen`: single-band gray-atmosphere radiation; outputs
  `RadiativeHeatingTendency` / `NetSurfaceFlux`.
- `syzygy`: cross-mesh coupling — `CouplingStencil`, `ScalarInterfaceFlux`,
  `ScalarRelaxation`. Same-mesh dependencies are plain nexus DAG edges;
  cross-mesh hops go through syzygy.

### eidolon — presentation

Engine-neutral: observes state and emits owned render/diagnostic data so
simulation crates never depend on engine types.

- `ir/`: the intermediate representation — frames, layers, meshes, transforms,
  colour, an `Update` enum stream, and stable IDs.
- `extract/`: producers that read pleroma/tessera and build IR.
- `bevy/` + `backend/`: the Bevy backend (behind the `bevy` feature) that
  applies the IR/`Update` stream. `export/vtk.rs` for file output.
- Feature flags: `default = []`, `bevy = [...]`. `sandbox` pulls eidolon with
  `bevy` enabled.

## Conventions

- Edition 2024 across the workspace. (`rustfmt.toml` pins its parser edition
  separately.)
- Apache-2.0 header on each source file.
- A stale `.rmeta` cache can cause phantom "type not found" errors —
  `cargo clean --package <pkg>` fixes it.

## CodeGraph

This repo is indexed by the `codegraph` CLI (`.codegraph/codegraph.db`, git-ignored).
Useful for navigation:

```bash
codegraph status                 # index stats / freshness
codegraph query <symbol>         # find symbols
codegraph callers <symbol>       # who calls it
codegraph callees <symbol>       # what it calls
codegraph impact  <symbol>       # what a change affects
codegraph context "<task>"       # build markdown context for a task
codegraph sync                   # re-index after changes
```
