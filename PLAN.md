# Aether Architecture Refactor: Tessera, Pleroma, Nexus

## Context

The workspace is in a transitional state. `continuum` still owns mesh,
geometry, field storage, the FV solver, and IO; the new physics crates `aer`,
`terra`, `gravitas` (currently named `orbit/` on disk), plus orchestration
`nexus` and renderer `eidolon`, are mostly stubs (`Cargo new` boilerplate).
`cosmo` is already in good shape — immutable, pure-function seed for
solar-system bodies and atmospheres, with constants pulled into `utility`.

The goal of this refactor is six-fold:

1. **Mesh becomes a shared substrate.** Pull all mesh and geometry concerns
   out of `continuum` into a new `tessera` crate so multiple physics crates
   can build, partition, and operate on meshes without going through the
   solver.

2. **Multiple meshes, one world.** A single planet is described by several
   meshes (atmospheric shell for `aer`, ground/interior for `terra`, oceans
   later) that *know how they map onto each other* — `tessera` owns this
   coupling structure but does no physics on it.

3. **Cosmo is the immutable seed.** All static definitions and ICs originate
   in `cosmo` and flow forward immutably. Cosmo gains `Serialize`/
   `Deserialize` so a world is reproducible from a single JSON seed.

4. **Pleroma owns world state.** A new `pleroma` crate aggregates every field
   (temperature, density, velocity, …) into a typed registry bound to meshes.
   Physics crates own no field data; they only operate on borrowed pleroma
   fields.

5. **Nexus compiles physics into a DAG.** A new `nexus` crate takes stage
   descriptors with declared reads/writes, builds a schedule, and executes it
   in parallel via `utility::thread::Pool::execute(TaskGraph)`. New physics
   is added by registering a stage; nothing is hand-wired per crate.

6. **Eidolon is a pure consumer.** Eidolon takes read-only snapshots of
   pleroma, builds an engine-agnostic IR via an Update protocol, and
   translates to a Bevy backend behind a feature flag.

`sandbox` becomes the integration rig that wires the chain together.

---

## Target Crate Topology

```
                ┌──────────┐
                │ utility  │
                └────┬─────┘
                     │
        ┌────────────┼──────────────┐
        ▼            ▼              │   (utility flows to all crates)
   ┌─────────┐  ┌─────────┐         │
┌──┤ tessera │  │  cosmo  │         │
│  └────┬────┘  └────┬────┘         │
│       │            │              │
│       ▼            │              │
│  ┌─────────┐       │              │
├──┤ pleroma │◀──────┘              │
│  └────┬────┘                      │
│       │                           │
│       ▼                           │
│  ┌─────────┐  re-exports pleroma's│
│  │  nexus  │  user-facing surface │
│  └────┬────┘  (FieldKey, FieldStorage, …)
│       │                           │
│       ▼                           ▼
│  ┌─────┬───────┬──────────┬────────────────┐
│  │ aer │ terra │ gravitas │ future physics │
│  └──┬──┴───┬───┴────┬─────┴──────┬─────────┘
│     │      │        │            │
│     └──────┴────────┴────────────┘
│                     │
│                     ▼
│               ┌──────────┐
├──────────────▶│continuum │   (numerical-methods library;
│               └──────────┘    consumed by physics crates)
│
│  ┌──────────┐
└─▶│ eidolon  │   (read-only viewer over pleroma + tessera)
   └────┬─────┘
        ▼
   ┌──────────┐
   │ sandbox  │   (integration rig: cosmo → pleroma → nexus → physics → eidolon)
   └──────────┘
```

Constraints preserved by this layout:
- `tessera` knows nothing about physics or fields — pure geometry/topology.
- `pleroma` depends on `tessera` (fields are bound to mesh cell counts) and
  on `cosmo` (initial-condition seeds), and knows no physics.
- `pleroma` is split internally into `pub mod core` (vocabulary) and
  `pub(crate) mod runtime` (storage + split-borrow machinery). Only
  `core` types and the re-exported `Pleroma` registry cross the crate
  boundary; the runtime is invisible to every other crate, nexus
  included.
- `nexus` depends on `pleroma` and **re-exports `pleroma::core::*`** so
  that physics crates never need a direct `pleroma` dependency. The
  pleroma surface they touch is too small (one trait + a few index-keys
  + two view types) to justify a separate dep edge.
- `continuum` is a *numerical-methods library*, **not a `Stage` emitter**.
  It depends on `tessera` (the `Mesh` trait) and `pleroma` (the
  `FieldStorage` trait from `pleroma::core`, used as a bound on its
  solver's state arguments). It does not depend on `nexus` and knows
  nothing about stages.
- Physics crates (`aer`, `terra`, `gravitas`, future) depend on `tessera`,
  `continuum`, and `nexus` — and **not** on `pleroma` directly. Each emits
  one or more `nexus::Stage`s; inside `Stage::run` they pull typed field
  references out of `WorldAccess` (re-exported by nexus) and pass them to
  `continuum::FvmSolver::parallel_step` as the numerical step.
- `eidolon` reads `pleroma + tessera`; depends on neither nexus nor physics.
- `cosmo` stays a pure data-only crate — its only dep is `utility`. Mesh
  construction from cosmo seeds lives in `tessera::*::from_seed(...)` etc.,
  so cosmo does NOT depend on tessera.

---

## Key Abstractions

### `tessera` — mesh crate

Pure migration of the existing geometry/topology code. The contents of these
files move from `continuum/src/` to `tessera/src/`, intact:

- `geometry.rs` — `CellId`, `FaceId`, `Point<D>`, `CellGeometry<D>`,
  `FaceGeometry<D>`, `CellMetrics<D>`, `FaceMetrics<D>`, `GeometryMap<D, P>`,
  `IdentityMap<D>`.
- `topology.rs` — `Topology`, `BoundaryTag`, `FaceConnection`, `CellKind`.
- `mesh.rs` — `Mesh<D>` super-trait, `Axis`, `StructuredBlock<D>`.
- `cube_sphere.rs` — `CubeSphere`, `GnomonicPanel`, `GnomonicShellPanel`,
  `CUBE_EDGES`, panel helpers.
- `partition.rs` — `PartitionMesh<D, M>`, `Decomposition<D, M>`,
  `GhostDescriptor`, `decompose_structured`.

`continuum` keeps `field.rs`, `boundary.rs`, `solver.rs`, `model.rs`,
`output.rs` for now. `field.rs` migrates to `pleroma` in Phase 3.

### Multi-mesh & coupling (Phase 2 additions to `tessera`)

A planet's discretisation is a `LayeredPlanetMesh`, a named collection of
meshes plus an explicit coupling registry:

```rust
// tessera/src/coupling.rs (NEW)
pub trait MeshCoupler: Send + Sync {
  fn paired_face(&self, side: Side, face: FaceId) -> Option<(Side, FaceId)>;
  fn paired_cell(&self, side: Side, cell: CellId) -> Option<(Side, CellId)>;
  fn pairs(&self) -> &[FacePair];   // (Side::A face, Side::B face)
}

pub enum Side { A, B }

// tessera/src/world_mesh.rs (NEW)
pub struct MeshKey(&'static str);   // stable handle; Copy+Eq+Hash

pub struct LayeredPlanetMesh {
  meshes: HashMap<MeshKey, Arc<dyn Mesh<3>>>,
  couplers: Vec<(MeshKey, MeshKey, Box<dyn MeshCoupler>)>,
}

// tessera/src/couplers/radial_stack.rs (NEW)
pub struct RadialStackCoupler {
  // Two cube-spheres sharing the same 6 panels and angular dims, where
  // the upper mesh's bottom radial layer touches the lower mesh's top
  // radial layer. Pairings are pure index arithmetic — no interpolation.
  panel_count: usize,
  angular_dims: [usize; 2],
  lower_top_layer_idx: usize,
  upper_bottom_layer_idx: usize,
}
```

The radial-stack case (atmosphere on top of ground, ground on top of mantle)
is the common one. More general couplers (interpolation across resolution
mismatches, unstructured ↔ structured) are deferred until needed.

The coupler does *not* move data — it only describes pairings. Data movement
happens via *coupling stages* in nexus (see below).

### `pleroma` — world-state crate

Owns every field used by every physics crate. Internally split into two
modules with a one-way visibility relationship:

- **`pub mod core`** — the *vocabulary* layer. Every type that crosses
  pleroma's crate boundary as a name (in nexus signatures, physics
  `Stage` impls, eidolon snapshots) lives here. No storage internals,
  no unsafe.
- **`pub(crate) mod runtime`** — the *machinery* layer. Owns the actual
  registry, the slot storage (`UnsafeCell<Box<dyn Any>>`), and the
  split-borrow primitives. `runtime` depends on `core`; `core` does not
  depend on `runtime`.

The `runtime` module being `pub(crate)` hides it from nexus too. The
runtime's nexus-facing API surfaces only via (a) methods on `Pleroma`,
which is re-exported at the crate root, and (b) types defined in `core`
that the runtime constructs and hands out.

```rust
// pleroma/src/lib.rs
pub mod core;
pub(crate) mod runtime;

pub use runtime::Pleroma;   // top-level registry handle (for sandbox/init)
```

```rust
// pleroma/src/core/mod.rs       — public; nexus + sandbox + eidolon name these
pub use storage::{FieldStorage, SoaField, AosField, CellView};
pub use key::{FieldKey, MeshKey};
pub use access::{WorldAccess, ScheduleAccess};

// pleroma/src/core/storage.rs   (moved from continuum/src/field.rs, intact)
pub trait FieldStorage<const N: usize>: Send + Sync { … }
pub struct SoaField<const N: usize> { … }
pub struct AosField<const N: usize> { … }
pub trait CellView<const N: usize> { … }

// pleroma/src/core/key.rs
#[derive(Clone, Copy, Eq, Hash, PartialEq, Debug)]
pub struct FieldKey { pub name: &'static str, pub mesh: MeshKey }

// pleroma/src/core/access.rs
pub struct WorldAccess<'a> {
  pub(in crate::runtime) slot_view: SlotView<'a>,    // opaque to outside
}
impl<'a> WorldAccess<'a> {
  pub fn mesh_for(&self, key: MeshKey) -> Option<&Arc<dyn Mesh<3>>>;
  pub fn read<S: 'static>(&self, key: FieldKey) -> Option<&S>;
  pub fn write<S: 'static>(&mut self, key: FieldKey) -> Option<&mut S>;
}

pub struct ScheduleAccess<'a> {
  pub(in crate::runtime) inner: SplitBorrow<'a>,
}
impl<'a> ScheduleAccess<'a> {
  // Caller (nexus) guarantees the keys passed across all view_for calls
  // for this ScheduleAccess are pairwise non-overlapping.
  pub unsafe fn view_for(
    &self, reads: &[FieldKey], writes: &[FieldKey]
  ) -> WorldAccess<'a>;
}
```

```rust
// pleroma/src/runtime/mod.rs    — pub(crate); only pleroma internals see this
mod registry;
mod slot;
mod split;

pub use registry::Pleroma;        // re-exported at crate root via lib.rs

// pleroma/src/runtime/registry.rs
pub struct Pleroma {
  meshes: HashMap<MeshKey, Arc<dyn Mesh<3>>>,
  fields: HashMap<FieldKey, slot::FieldSlot>,
}

impl Pleroma {
  pub fn register_mesh(&mut self, key: MeshKey, mesh: Arc<dyn Mesh<3>>);
  pub fn register_field<S: FieldStorage<N> + 'static, const N: usize>(
    &mut self, key: FieldKey, init: S);

  // Single-stage / test direct access (safe; takes &mut self).
  pub fn read<S: 'static>(&self, key: FieldKey) -> Option<&S>;
  pub fn write<S: 'static>(&mut self, key: FieldKey) -> Option<&mut S>;

  // Nexus entry-point. Hands out a ScheduleAccess for one DAG layer;
  // nexus then calls `unsafe view_for` once per parallel stage with that
  // stage's declared reads/writes. The unsafe split is sound because the
  // schedule has already verified non-overlap at the layer level.
  pub fn schedule_access(&mut self) -> ScheduleAccess<'_>;
}
```

`Box<dyn Any>` slot storage is one valid implementation of `runtime::slot`;
the storage scheme is fully encapsulated and can be swapped for a typed-by-N
enum or const-generic registry without touching `core` or any caller.

**What lives where, summarised:**

| Item                                  | Module    | Why |
|---------------------------------------|-----------|------|
| `FieldKey`, `MeshKey`                 | core      | Named by nexus + physics |
| `FieldStorage` trait, `CellView`      | core      | Bound on continuum solver |
| `SoaField`, `AosField`                | core      | Physics may name concrete types |
| `WorldAccess<'a>`                     | core      | Stage `run` signature |
| `ScheduleAccess<'a>`                  | core      | Nexus per-layer handle |
| `Pleroma` registry struct             | runtime → re-exported at root | Sandbox needs to construct it |
| `FieldSlot`, slot allocator           | runtime   | Pure implementation detail |
| `SplitBorrow`, unsafe split internals | runtime   | Hidden from everything outside pleroma |

### `nexus` — DAG scheduler

A `Stage` declares the field keys it reads and writes; nexus builds a DAG
from these declarations and runs independent stages in parallel.

**`nexus` is the single public entry-point for physics crates.** It
re-exports the entirety of `pleroma::core` so that an `aer` or `terra`
`Cargo.toml` lists `nexus`, `tessera`, `continuum`, `utility` — and not
`pleroma`. Because pleroma is split into `pub mod core` and
`pub(crate) mod runtime`, nexus can glob-re-export core without leaking
storage internals:

```rust
// nexus/src/lib.rs
pub use pleroma::core::*;     // FieldKey, MeshKey, FieldStorage,
                              // SoaField, AosField, CellView,
                              // WorldAccess, ScheduleAccess
pub use crate::stage::*;
pub use crate::schedule::*;
```

Nexus itself depends on `pleroma` (one of the few crates that does — the
others are continuum, eidolon, sandbox), but it names only `pleroma::core`
types and `pleroma::Pleroma` (the registry, re-exported at pleroma's
crate root). It never names `pleroma::runtime`.

```rust
// nexus/src/stage.rs
pub trait Stage: Send + Sync {
  fn name(&self) -> &'static str;
  fn reads(&self) -> &[FieldKey];
  fn writes(&self) -> &[FieldKey];
  fn run(&self, ctx: StageContext<'_>) -> AetherResult<()>;
}

pub struct StageContext<'a> {
  pub world: WorldAccess<'a>,   // typed read/write into pleroma, scoped to declared keys
  pub pool: &'a Pool,            // for inner parallelism (e.g. continuum::parallel_step)
  pub dt: f64,
}

// nexus/src/schedule.rs
pub struct Schedule { … }

impl Schedule {
  pub fn add(&mut self, stage: impl Stage + 'static) -> StageId;
  pub fn before(&mut self, a: StageId, b: StageId);   // explicit ordering hint
  pub fn build(self, world: &Pleroma) -> AetherResult<CompiledSchedule>;
}

pub struct CompiledSchedule { … }   // topo-sorted, layered

impl CompiledSchedule {
  pub fn tick(&self, world: &mut Pleroma, pool: &Pool, dt: f64)
    -> AetherResult<()>;
}
```

DAG edges derived from declared reads/writes:
- **RAW**: `B` depends on `A` if `A.writes ∩ B.reads ≠ ∅`.
- **WAW**: same field can't be written by two concurrent stages.
- **`before`** hints add edges when reads/writes alone don't capture intent
  (e.g. enforcing physical ordering between unrelated fields).

`tick()` walks the DAG layer by layer, hands out non-overlapping
`StageContext` views via `Pleroma::pin_for_schedule()`, and submits
parallel-eligible groups via `Pool::execute(TaskGraph)`. The split-borrow
inside `pin_for_schedule()` is unsafe internally but sound at the schedule
layer because the schedule has verified no two simultaneous stages share
overlapping `FieldKey`s.

### Coupling stages

Cross-mesh coupling is a stage whose `reads` and `writes` reference
*different* meshes. It lives in a new dedicated `couplers/` crate
depending on `nexus + tessera + utility` — it cannot live in pleroma
because that would force a `pleroma → nexus` cycle:

```rust
// couplers/src/ghost_fill.rs (Phase 5)
pub struct GhostFillStage {
  source: FieldKey,        // e.g. ("temperature", ground_mesh)
  destination: FieldKey,   // e.g. ("ghost_temperature", atm_mesh)
  coupler: Arc<dyn MeshCoupler>,
}

impl Stage for GhostFillStage {
  // reads = [source]; writes = [destination]
  // run: iterate coupler.pairs(), copy source[lower_cell] -> destination[upper_ghost_cell]
}
```

Continuum's solver consumes the destination ghost field via a new
`PrescribedGhost` boundary condition, which slots into the existing
`BoundaryCondition` trait. No solver-side changes needed.

### `cosmo` — immutable seed (Phase 5 expansion)

Today: bodies, atmospheres, factory presets. Adds:
- `cosmo/src/discretisation.rs` — `LayeredMeshSeed` (per-layer cube-sphere
  angular dim, radial layer count, radial extent).
- `cosmo/src/scheme.rs` — IC enum (hydrostatic atmosphere, isothermal, etc.),
  per-tag BC choices, integrator params (CFL, max dt, integrator type).
- `#[derive(Serialize, Deserialize)]` across all cosmo types using
  `utility::serial`. A saved seed is one JSON file that fully reproduces
  a run.

Cosmo holds primitive seed data only — no construction logic. Mesh
construction lives in `tessera::StructuredBlock::from_seed(...)` etc., so
cosmo does not depend on tessera.

### `eidolon` — viewer (Phase 6)

Depends on `pleroma` and `tessera` only:
- `Viewer` takes `&Pleroma` and a list of `FieldKey`s of interest.
- `RenderUpdate` enum stream (Update protocol from the prior eidolon design,
  per memory).
- Backend abstraction; first concrete backend is Bevy 0.17.3 behind the
  `bevy` feature flag (using `set_parent_in_place`/`despawn` per memory).

---

## Migration Phases

### Phase 0 — Rename `orbit` → `gravitas` (trivial)

`orbit/` is a fresh stub (boilerplate `add()` test, no real code), so the
rename is mechanical. Land first, in its own commit.

1. `git mv orbit gravitas`.
2. Update `[package] name = "gravitas"` in `gravitas/Cargo.toml`.
3. Update workspace `members` in `/mnt/ssd1/dev/aether/Cargo.toml`:
   replace `"orbit"` with `"gravitas"`.

**Verification:** `cargo build --workspace` and `cargo test --workspace`
both succeed unchanged.

### Phase 1 — Extract `tessera` (mechanical, no behaviour change)

**Goal:** Move geometry/mesh/topology/partition out of continuum into a new
crate. Zero solver, field, or physics changes.

1. Create `tessera/Cargo.toml`, `tessera/src/lib.rs`. Add `tessera` to
   workspace members in `/mnt/ssd1/dev/aether/Cargo.toml`. Dependency:
   `utility = { path = "../utility" }` only.
2. Move with `git mv` into `tessera/src/`:
   - `continuum/src/geometry.rs`
   - `continuum/src/topology.rs`
   - `continuum/src/mesh.rs`
   - `continuum/src/cube_sphere.rs`
   - `continuum/src/partition.rs`
3. Update `tessera/src/lib.rs` to re-export the public surface (matching
   what continuum's lib.rs exports today for these modules).
4. Add `tessera = { path = "../tessera" }` to `continuum/Cargo.toml`.
5. Replace internal `crate::geometry::*` etc. uses across continuum with
   `tessera::*`. Files to touch (per the survey): `solver.rs`,
   `boundary.rs`, `field.rs`, `model.rs`, `output.rs`, `lib.rs`.
6. Update tests under `continuum/tests/*` whose imports change:
   - `geometry_cube_sphere.rs`
   - `geometry_metrics_identity.rs`
   - `mesh_topology_invariants.rs`
   - `solver_cube_sphere_hydrostatic.rs`
   - `solver_cube_sphere_smoke.rs`
   - `solver_hydrostatic_balance.rs`
   - `solver_parallel_consistency.rs`
7. Update `continuum/examples/sod_shock.rs` and `sandbox/src/main.rs`
   imports.

**Verification:**
- `cargo build --workspace`
- `cargo test --workspace` — all 7 continuum integration tests + cosmo tests
  pass with no diffs in numerical output
- `cargo run --example sod_shock` writes the same VTU output as before
  (compare against a stashed pre-refactor snapshot)
- `cargo fmt --all -- --check`

This phase should land as a single PR.

### Phase 2 — Multi-mesh primitives in `tessera`

**Goal:** `LayeredPlanetMesh`, `MeshCoupler` trait, `RadialStackCoupler`. No
physics consumption yet.

1. New module `tessera/src/coupling.rs` — `Side`, `FacePair`,
   `MeshCoupler` trait.
2. New module `tessera/src/world_mesh.rs` — `MeshKey`,
   `LayeredPlanetMesh` + builder.
3. New module `tessera/src/couplers/radial_stack.rs` — `RadialStackCoupler`
   using cube-sphere panel + angular index arithmetic.
4. New tests under `tessera/tests/`:
   - `coupling_radial_stack.rs` — build atmosphere + ground cube-spheres of
     identical angular topology; verify face/cell pair lookups are bijective,
     total paired-face area equals each mesh's interface area, and
     world-frame normals point in opposite directions.

No changes outside `tessera` in this phase.

### Phase 3 — Extract `pleroma` (world-state) with `core`/`runtime` split

**Goal:** Move field storage out of continuum into pleroma, **with the
`core`/`runtime` module split in place from day 1** — splitting later is
strictly more expensive than getting it right now.

1. Create `pleroma/Cargo.toml` depending on `tessera` and `utility`.
2. Lay out the module skeleton:
   - `pleroma/src/lib.rs`:
     ```rust
     pub mod core;
     pub(crate) mod runtime;
     pub use runtime::Pleroma;
     ```
   - `pleroma/src/core/{mod,storage,key,access}.rs`
   - `pleroma/src/runtime/{mod,registry,slot,split}.rs`
3. Move `continuum/src/field.rs` → `pleroma/src/core/storage.rs`. Public
   types unchanged: `FieldStorage`, `SoaField`, `AosField`, `CellView`.
4. Add `pleroma/src/core/key.rs` — `FieldKey`. Re-export `MeshKey` from
   tessera.
5. Add `pleroma/src/core/access.rs` — `WorldAccess<'a>` and
   `ScheduleAccess<'a>` types. Their fields that point into runtime are
   `pub(in crate::runtime)` so only runtime can construct them.
6. Add `pleroma/src/runtime/registry.rs` — `Pleroma` struct with
   `register_*`, safe `read`/`write`, and `schedule_access(&mut self)`.
7. Add `pleroma/src/runtime/slot.rs` — `FieldSlot` (`UnsafeCell<Box<dyn
   Any + Send + Sync>>` + a typeid tag for downcast checking).
8. Add `pleroma/src/runtime/split.rs` — `SplitBorrow` primitive used by
   `ScheduleAccess::view_for`. This is the only place `unsafe` lives.
9. Update `continuum/Cargo.toml` to depend on `pleroma`. Replace internal
   field uses with `pleroma::core::FieldStorage`. Solver retains its
   scratch buffers — they're owned, not pleroma-managed (per the existing
   CLAUDE.md note).
10. Update tests / examples / sandbox to register state in a `Pleroma`
    before invoking the solver.

**Verification:**
- All existing continuum tests pass via the new path: each test
  constructs a `Pleroma`, registers its mesh and field, and the solver
  borrows them.
- New unit tests in `pleroma/tests/`:
  - `field_register_lookup.rs` — register, retrieve, mutate via the safe
    `Pleroma::read`/`write` methods.
  - `schedule_access_disjoint.rs` — non-overlapping declared keys yield
    safe `WorldAccess` views; overlapping keys are rejected with a clear
    error.
  - `core_runtime_visibility.rs` — a doc-test or trybuild test that
    asserts `pleroma::runtime` is invisible to external crates (e.g.
    `pleroma::runtime::Pleroma` doesn't compile from a downstream crate;
    `pleroma::Pleroma` does).

### Phase 4 — Build `nexus` (scheduler + re-exports + sandbox smoke test)

**Goal:** Stand up nexus and prove it runs real workloads — without making
continuum aware of stages.

1. Replace nexus stub. Modules: `stage.rs`, `schedule.rs`, `compiled.rs`,
   plus `lib.rs` containing the pleroma re-exports listed in the
   "Key Abstractions → nexus" section.
2. Implement `Stage` trait, `Schedule` builder, `CompiledSchedule::tick`
   on top of `Pool::execute(TaskGraph)`
   (`utility/src/thread/pool.rs:68-87`).
3. DAG edges derived from `reads()`/`writes()` plus explicit `before`
   hints. Cycle detection via the existing `Graph::topological_sort` in
   `utility/src/collections/graph.rs` (already used by `Pool::execute`).
4. **Test fixtures inside nexus** (`nexus/tests/fixtures/`): an
   `IdentityStage` (reads = writes = X, run is a no-op), an accumulator
   stage that writes a counter field, and a barrier-style pair used to
   detect actual parallel execution. These let the scheduler tests stand
   without depending on any physics crate.
5. **Sandbox smoke test:** wrap the existing sod-shock loop as a thin
   `SodShockStage` *inside* `sandbox/src/main.rs` (not in continuum) that
   pulls fields out of `WorldAccess` and calls
   `continuum::FvmSolver::parallel_step` from `Stage::run`. Verify the VTU
   output matches Phase 3 byte-for-byte. This is the canonical pattern
   for "physics crate uses continuum from inside a stage" — physics crates
   in Phase 5 will mirror it.
6. **Continuum is untouched in this phase**, beyond perhaps a small
   ergonomic helper to make pulling a `Mesh` + `FieldStorage` pair from
   `WorldAccess` and feeding them to `parallel_step` less verbose. No
   `Stage` impls land in continuum.

**Verification:**
- Sandbox smoke test produces bit-identical VTU output vs Phase 3.
- New nexus tests:
  - `schedule_reads_writes_dag.rs` — overlapping writes serialise;
    independent stages run on the pool concurrently.
  - `schedule_explicit_before.rs` — `before` adds edges even when
    reads/writes don't conflict.
  - `schedule_cycle_rejection.rs` — circular dependencies surface as a
    clear `AetherError`.
  - `reexports_compile.rs` — a tiny test crate (or doc-test) that
    imports `use nexus::{FieldKey, FieldStorage, SoaField, WorldAccess};`
    with no `pleroma` dependency, to lock the re-export contract.

### Phase 5 — Cosmo seed expansion + initial physics + couplers

**Goal:** Cosmo describes the full simulation; `aer` / `terra` / `gravitas`
gain first stages (each owns its own `Stage` impls — continuum stays a
library); coupling stages move data between meshes.

Every stage in this phase follows the pattern proven by `SodShockStage` in
Phase 5: `Cargo.toml` deps are `nexus + tessera + continuum + utility`
only; `Stage::run` pulls typed field references out of `WorldAccess` and
hands them to `continuum::FvmSolver` (or solves directly when no FV step
is needed, e.g. `gravitas::KeplerStage`).

1. **Cosmo additions:**
   - `cosmo/src/discretisation.rs` — `LayeredMeshSeed` with per-layer
     cube-sphere dims and radial extents.
   - `cosmo/src/scheme.rs` — IC enum, BC choices per tag/layer, integrator
     params.
   - `#[derive(Serialize, Deserialize)]` across cosmo types using
     `utility::serial`.
   - Round-trip test: `cosmo::factory::sol() → JSON → System` deep-equal.
2. **`aer::AtmosphereSolveStage`** — atmospheric Euler step. Lives in
   `aer/src/stages/atmosphere_solve.rs`. Holds an `FvmSolver<3, 5, Euler3D,
   RusanovFlux>` from continuum; declares reads/writes against the
   atmosphere mesh's conserved field.
3. **`terra::ConductionStage`** — placeholder (no-op or trivial heat
   diffusion; real physics later). Lives in
   `terra/src/stages/conduction.rs`.
4. **`gravitas::KeplerStage`** — n-body / orbital evolution. Operates on
   cosmo body positions registered in pleroma as a non-mesh-bound resource
   (keyed by body id, see "Body/orbital state in pleroma" in
   "Decisions" below). Lives in `gravitas/src/stages/kepler.rs`.
5. **`GhostFillStage`** (coupling) — lives in a new tiny `couplers/`
   crate depending on `nexus + tessera + utility`. It cannot live in
   `pleroma` because pleroma doesn't depend on nexus (and shouldn't,
   that would be a cycle). It's a regular stage from the runtime's
   point of view: declares reads on the source mesh's field, writes on
   the destination mesh's ghost field, and uses `tessera::MeshCoupler`
   to drive the per-pair copy. Continuum gains a `PrescribedGhost` BC
   (slots into the existing `BoundaryCondition` trait) to consume those
   ghost slots — this is the only continuum change in Phase 5.
6. Compose a small end-to-end sim in sandbox: atmosphere advection →
   ground↔atm `GhostFillStage` → ground conduction. Run, write VTU,
   verify mass/energy conservation across the seam.

This phase is large and should land as multiple sub-PRs (one per stage).

### Phase 6 — Eidolon viewer + sandbox integration

**Goal:** Eidolon reads pleroma, emits `RenderUpdate`s, renders via Bevy.

1. Replace eidolon stub. Re-establish `RenderUpdate` enum, `RenderRegistry`
   ref-counting, planet model setup (per the prior eidolon design captured
   in user memory).
2. `Viewer::snapshot(&Pleroma) -> Vec<RenderUpdate>` — idempotent updates
   that bring a backend's scene up to date with current field values.
3. Bevy 0.17.3 backend (feature `bevy`): translates updates to ECS commands.
   Use `set_parent_in_place` and `despawn` per memory's correction.
4. Sandbox grows an optional `--render` flag that spawns a Bevy app and
   ticks eidolon alongside nexus.

---

## Critical files (Phase 1 first)

Files moved out of continuum:
- `/mnt/ssd1/dev/aether/continuum/src/geometry.rs`
- `/mnt/ssd1/dev/aether/continuum/src/topology.rs`
- `/mnt/ssd1/dev/aether/continuum/src/mesh.rs`
- `/mnt/ssd1/dev/aether/continuum/src/cube_sphere.rs`
- `/mnt/ssd1/dev/aether/continuum/src/partition.rs`

Files whose imports change (still in continuum):
- `/mnt/ssd1/dev/aether/continuum/src/lib.rs`
- `/mnt/ssd1/dev/aether/continuum/src/solver.rs`
- `/mnt/ssd1/dev/aether/continuum/src/boundary.rs`
- `/mnt/ssd1/dev/aether/continuum/src/field.rs`
- `/mnt/ssd1/dev/aether/continuum/src/model.rs`
- `/mnt/ssd1/dev/aether/continuum/src/output.rs`
- `/mnt/ssd1/dev/aether/continuum/Cargo.toml`

Tests and examples whose imports change:
- `/mnt/ssd1/dev/aether/continuum/tests/*.rs` (7 files listed above)
- `/mnt/ssd1/dev/aether/continuum/examples/sod_shock.rs`
- `/mnt/ssd1/dev/aether/sandbox/src/main.rs`

Workspace:
- `/mnt/ssd1/dev/aether/Cargo.toml` — add `tessera` to `members`.

---

## Reused components (do not reinvent)

- **DAG scheduler** — `utility::thread::Pool::execute(TaskGraph)`
  (`utility/src/thread/pool.rs:68-87`) and `TaskGraph::add`/`dependency`
  (`utility/src/thread/pool.rs:304-338`). Nexus's `CompiledSchedule::tick`
  builds a `TaskGraph` per layer.
- **Graph + topological sort** — `utility::collections::graph::Graph`
  (`Pool::execute` already uses its `topological_sort()`).
- **Custom Serialize/Deserialize + JSON backend** —
  `utility/src/serial/{json.rs, serialize.rs, deserialize.rs}` plus the
  derive macros in `utility_macros`. Cosmo gains derives in Phase 5.
- **Profiler / logger** — `#[profile]`, `info!`, `warn!`, etc. Stages and
  coupling steps should use them so the profiler trace stays continuous.
- **Ghost-exchange pattern** — `continuum/src/partition.rs:14-37` and
  `Decomposition::exchange_ghosts` (lines 171-205). The cross-mesh
  ghost-fill stage in Phase 5 reuses the same two-pass scatter/gather
  idiom.
- **VTK output** — `continuum/src/output.rs` + `XmlVtuWriter`/`XmlPvtuWriter`.
  After Phase 3, output stages read fields from pleroma.
- **Cosmo factory presets** — `cosmo/src/factory.rs` (sol(), earth(), …)
  remain the source of truth for body/atmosphere data; Phase 5 just adds
  serde derives and seed-expansion modules.

---

## Open items / decisions resolved

Settled by the question round in this planning session:
- **Crate names**: `tessera` (mesh), `pleroma` (world-state), `nexus`
  (scheduler — already named), `gravitas` (was `orbit`; renamed in Phase 0
  before any other work).
- **Coupling design**: implemented as nexus stages that read one mesh's
  field and write another mesh's ghost field. Solver and BCs stay
  mesh-agnostic.
- **Mutability**: stages declare `reads()`/`writes()`; nexus builds a DAG
  and runs non-conflicting stages in parallel via `Pool::execute`. Pleroma
  exposes split-borrow only to nexus.

Other settled architectural choices:
- **Continuum is a numerical-methods library, not a `Stage` emitter.**
  No `Stage` impls live in continuum. Physics crates (`aer`, `terra`,
  `gravitas`) own all stages; their `Stage::run` bodies pull fields out
  of `WorldAccess` and call `continuum::FvmSolver::parallel_step`.
  Sandbox follows the same pattern for the Phase 4 smoke test.
- **Physics crates do NOT depend on `pleroma` directly.** Their
  `Cargo.toml` lists `nexus + tessera + continuum + utility`. Nexus
  re-exports `pleroma::core::*` (i.e. all of `FieldKey`, `MeshKey`,
  `FieldStorage`, `SoaField`, `AosField`, `WorldAccess`,
  `ScheduleAccess`, `CellView`). The pleroma surface they touch is too
  small to justify a separate dep edge.
- **Continuum's pleroma dep is intentional** — continuum needs the
  `FieldStorage` trait as a bound on its solver state arguments. It
  imports `pleroma::core::FieldStorage` (note: `core`, not via nexus)
  because continuum has nothing to do with stages or scheduling.
- **Cosmo dependency**: cosmo holds primitive seed data only; mesh
  construction lives in `tessera::StructuredBlock::from_seed(...)` etc.,
  so cosmo does NOT depend on tessera.
- **FieldStorage location**: moves to `pleroma::core::storage` in Phase 3.
  Continuum gains a `pleroma` dep at that point, importing the trait via
  `pleroma::core::FieldStorage`.
- **Pleroma `core`/`runtime` split**: stood up in Phase 3 from day 1.
  `pub mod core` holds vocabulary (keys, traits, view types); `pub(crate)
  mod runtime` owns the registry, slot allocator, and unsafe
  split-borrow primitives; `pleroma::Pleroma` is a re-export of the
  runtime registry. Splitting later would force a painful cascade of
  visibility changes.
- **Body/orbital state in pleroma**: orbital body positions live in
  pleroma as a non-mesh-bound resource keyed by body id — pleroma needs a
  small "resource" channel alongside the mesh-bound field channel.

Deferred (revisit when needed):
- Interpolating couplers / unstructured couplers — only `RadialStackCoupler`
  in Phase 2.
- Whether the dedicated `couplers/` crate should be one crate covering
  all coupling stages, or split per-coupling-kind (radial-stack, …) —
  decide as more stages emerge in Phase 5.
- Naming of `LayeredPlanetMesh`, `WorldAccess` — working names; revisit at
  Phase 2/3 if better names surface.
- Whether `WorldAccess` should statically encode declared keys via type
  parameters (more compile-time safety, more boilerplate) vs. the
  runtime-checked variant proposed here. Start runtime-checked.

---

## Verification matrix

| Phase | Build               | Tests                                               | Behavioural |
|-------|---------------------|-----------------------------------------------------|-------------|
| 0     | workspace           | unchanged                                            | n/a |
| 1     | workspace           | continuum + cosmo unchanged                         | sod_shock VTU bit-identical |
| 2     | workspace           | + tessera coupling tests                            | n/a |
| 3     | workspace           | continuum tests via Pleroma; new pleroma tests      | numerics unchanged |
| 4     | workspace           | + nexus schedule tests; sandbox `SodShockStage` smoke | sod_shock VTU bit-identical to phase 3 |
| 5     | workspace           | + cosmo serde round-trip; first cross-mesh sim      | conservation across coupler seam |
| 6     | workspace + `--features bevy` | + eidolon snapshot tests                  | manual visual check |

Run after each phase:
- `cargo fmt --all -- --check`
- `cargo build --workspace`
- `cargo test --workspace`

A regression-detection harness for Phases 1, 3, and 4 is a single saved
sod_shock VTU file checked into `continuum/tests/fixtures/` with a small
helper that diffs floating-point fields with a tight tolerance — this is
the cheapest way to ensure the mechanical refactor and the world-state /
nexus rewires preserve numerical output.
