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
cargo test -p utility

# Run a single integration test file
cargo test --test mesh_topology_invariants
cargo test --test geometry_metrics_identity
cargo test --test solver_parallel_consistency

# Run a single test by name
cargo test -p continuum test_name

# Run examples
cargo run --example sod_shock

# Format
cargo fmt --all
cargo fmt --all -- --check
```

**Formatting rules** (`rustfmt.toml`): 2-space indent, max 80 columns, no hard tabs.

## Workspace Structure

Four crates: `utility`, `utility_macros` (proc-macro), `continuum`, `sandbox`.

- **`utility`** — shared foundation: math types, thread pool, profiler, logger, serialization (JSON + VTK)
- **`continuum`** — finite-volume PDE solver on structured grids
- **`sandbox`** — standalone executable for manual solver experiments (Sod shock tube)

## Architecture

### utility

**Math** (`utility/src/maths/`): `Vector<T, D>`, `Matrix<T, R, C>`, `Quaternion<T>`. `Vector` wraps `Matrix<T, D, 1>`. Constructors take arrays: `Vec3::new([x, y, z])`.

**Thread pool** (`utility/src/thread/`): Work-stealing `Pool` (default: all available cores). Key APIs: `parallel_for()`, `execute()` (DAG of tasks), `spawn()`. Use `TaskHandle` to synchronize.

**Profiler** (`utility/src/profiler/`): `#[profile]` attribute macro inserts `SpanGuard` at function entry. Thread-local state, no locking overhead. Flush with `Profiler::flush_profiler()`, print with `Profiler::print()`.

**Logger** (`utility/src/logger/`): Levels Trace→Fatal, macros `trace!`/`debug!`/`info!`/`warn!`/`error!`/`fatal!`. Initialize once with `Logger::init(sinks, level)`.

**Serialization** (`utility/src/serial/`): Custom `Serialize`/`Deserialize` traits with derive macros. Backends: JSON and VTK XML (`XmlVtuWriter` / `XmlPvtuWriter` for partitioned output).

### continuum

The solver is generic over dimension `D`, state size `N`, conservation law `L: ConservationLaw<D,N>`, and flux scheme `F: NumericalFlux<D,N>`. Currently the only implemented physics is `Euler2D` (compressible Euler, 4-component state `[ρ, ρu, ρv, E]`) with `RusanovFlux`.

**Mesh layer** (`mesh.rs`, `geometry.rs`, `topology.rs`):
- `StructuredBlock<D>` is the main concrete mesh — a Cartesian block with pre-computed centroids, volumes, face areas, and connectivity.
- `Mesh<D>` super-trait requires `CellGeometry<D> + FaceGeometry<D> + Topology`.
- `GeometryMap<D, P>` / `IdentityMap<D>` provide coordinate transforms (physical ↔ computational).
- `FaceConnection` is either `Interior { owner, neighbour }` or `Boundary { owner, tag, out_sign }`.
- `BoundaryTag` variants: `Top/Bottom/Left/Right/Front/Back/Wall/Ground/Inflow/Outflow/AtmosphereEdge`.

**Field storage** (`field.rs`):
- `FieldStorage<N>` trait with two implementations:
  - `SoaField<N>` (Structure of Arrays) — `[Vec<f64>; N]`, fast for component-wise sweeps
  - `AosField<N>` (Array of Structures) — `Vec<[f64; N]>`, fast for per-cell access
- Both support `axpy()`, `weighted_sum()` BLAS-style ops.

**Solver** (`solver.rs`):
- `FvmSolver<D, N, L, F>` drives time integration.
- `step()` — serial/reference single step.
- Partitioned CPU execution lives in `cpu.rs` via `CpuFvmRunner`, which is the layer that depends on `Pool`.
- `SolverConfig` takes CFL coefficient, max Δt, and `TimeIntegration` (`ForwardEuler` or `Rk2`).
- Internal `Scratch` buffers avoid hot-loop allocations (do not remove these).

**Domain decomposition** (`partition.rs`):
- `decompose_structured()` splits a `StructuredBlock` into 1D stripes with ghost layers.
- `PartitionMesh<D, M>` holds `local_to_global_cell` mapping and ghost descriptors.
- `Decomposition<D, M>` is the multi-partition container passed to `CpuFvmRunner::step()`.

**Boundary conditions** (`boundary.rs`):
- Implement `BoundaryCondition<D, N>` trait (`ghost_state(interior, normal) → [f64; N]`).
- Register with `BoundaryRegistry::register(tag, bc)`.
- Built-in: `Transmissive` (zero-gradient), `ReflectiveWall`.

**Output** (`output.rs`):
- `LawFieldSchema<D, N>` trait maps states to named conserved/derived arrays.
- `write_partitioned_vtu()` writes VTK snapshots for ParaView visualization.
