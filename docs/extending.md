# Extending aether

Aether is built to be extended without forking the engine. Because modules
communicate through *data contracts* — named fields on meshes — not a hard-coded
call graph, most additions are local. This guide gives the concrete recipes,
roughly in order of how often you'll reach for them.

Read [architecture.md](architecture.md) first for the layering and the central
rule (*physics crates do not own global state or geometry*); everything here is a
consequence of it.

## Add a new field

State lives only in `pleroma`. A field is keyed by a `FieldKey { mesh, name }`,
where `name` is a `FieldName` and `mesh` a `MeshKey` (both in
`utility::domain`). To add a quantity:

1. Add a variant to `FieldName` (it's a closed enum — that's deliberate, it keeps
   the vocabulary auditable; there are no exhaustive matches on it, so adding a
   variant is low-blast).
2. At world setup, `pleroma.register_field(key, SoaField::<N>::…)` with the right
   storage (`SoaField` for component-wise sweeps, `AosField` for per-cell
   access).
3. Producers write it, consumers read it — by naming the key in their stage.

## Add a new stage

A `Stage` (in `nexus`) is one unit of work. Implement the trait:

```rust
impl Stage for MyStage {
  fn name(&self) -> &'static str { "my_stage" }
  fn reads(&self) -> &[FieldKey]  { &self.reads }   // what it consumes
  fn writes(&self) -> &[FieldKey] { &self.writes }  // what it produces
  fn run(&mut self, ctx: StageContext<'_>) -> AetherResult<()> {
    let src: &SoaField<1> = ctx.world.fields.read(self.src)?;
    let dst: &mut SoaField<1> = ctx.world.fields.write(self.dst)?;
    // … operate on borrowed state …
    Ok(())
  }
}
```

Add it with `nexus.add(MyStage::new(...))`. **You do not wire dependencies by
hand**: `nexus` reads your `reads()`/`writes()` declarations and orders stages by
their RAW/WAR/WAW conflicts, running everything non-conflicting in parallel. Use
`nexus.before(a, b)` only to pin *physical* ordering between stages that don't
otherwise conflict.

Optional knobs:
- `resource_reads()` / `resource_writes()` for non-mesh state (a `ResourceKey`
  like the sun direction or body state).
- `subsystem()` to place the stage on a different cadence clock (see multi-rate
  below).
- `plan()` to emit a multi-task program (e.g. a partitioned solve) instead of a
  single closure.

## Add a new physics crate

Follow the contract every physics crate already follows — own the *logic*, store
nothing globally. The conventional shape (see `terra`, `thalassa`, `aer`) is a
model builder:

1. `Cargo.toml`: depend on `nexus + tessera + utility` (add `continuum`/`tempus`
   if you need the solver/integrators). Never depend on `pleroma` directly —
   `nexus` re-exports its prelude.
2. A `Model::register_fields(pleroma, mesh)` that registers your field(s).
3. A `Model::add_stages(nexus)` that adds your `Stage`(s) and returns their ids.
4. Step state through `tempus` kernels or a `continuum` solver; never hold a mesh
   or a global buffer.

Give the crate a `docs/overview.md` describing its role and how it fits — match
the existing ones.

## Add a new conservation law / fluid

`continuum` is generic over the physics. A new fluid is a new
`ConservationLaw<D, N>` implementation — define its `flux`, `max_wave_speed`,
per-cell `source`, and `fix_state` (clamp unphysical states). Pick or implement a
`NumericalFlux<D, N>` (the built-in `RusanovFlux` works for most). You get the
explicit/implicit/IMEX backends and time integration for free. If you go
implicit, remember AD through non-smooth kernels (`sqrt(0)`, `abs`, `max`)
produces NaN Jacobians — regularise and test from rest/vacuum states.

## Couple two processes

- **Same mesh:** nothing special. If your stage reads a field another writes,
  `nexus` already orders them. Coupling *is* the shared field.
- **Across meshes:** use `syzygy`. Build a `CouplingStencil` from a `tessera`
  coupler (which precomputes the cell-to-cell mapping), then move quantities with
  `ScalarInterfaceFlux`, `ScalarInterfaceDeposition`, or `ScalarRelaxation`.
  Keep the producing crate `syzygy`-free: have it take plain precomputed coupling
  data, and let the assembler (e.g. `sandbox`) build the stencil. Conserve the
  budget you transport (debit one side what you credit the other).

A proven pattern for a new coupling: precompute the static geometry at setup,
apply a relaxation/forcing with a clamp, and verify with an A/B test (coupling
on vs off) plus a sign/conservation check.

## Add a new mesh type

`tessera` owns geometry. A mesh implements `Mesh<D> = CellGeometry<D> +
FaceGeometry<D> + Topology`. Add a `MeshType` variant, build the mesh
(`StructuredBlock`, a cube-sphere shell, a radial stack, …), and register it on
the world's `Tessera`. **Curvilinear gotcha:** a cube-sphere shell's *volume*
metric must not be reused as a *face-area* metric — override
`face_sqrt_det_metric`, or characteristic lengths come out wrong by a factor of
the radius.

## Add a new solver backend

Backends sit behind the `FvmBackend` trait (`continuum`). Implement it to add a
new execution strategy (a GPU backend, a different implicit method). Parallelism
across partitions is the scheduler's job, not the backend's — a backend stays a
serial solver over one partition.

## Add a new render layer or queryable quantity

Rendering and queries are consumer-side and additive (see
[rendering.md](rendering.md)):

- A new **render layer**: add a `ScalarLayerConfig` / categorical layer to your
  `ExtractConfig`. The producer reads the field and emits it; no engine change.
- A new **queryable quantity**: add a `ScalarQuantity` variant + its mesh/channel
  binding in `eidolon::query`, and a default channel in `query_extract`. This
  keeps the public semantic vocabulary decoupled from internal `FieldName`s.

## Run on a different cadence (multi-rate)

Override `Stage::subsystem()` to put a stage on its own `SubsystemId`, and
register a cadence with `nexus.set_subsystem_cadence(id, dt)`. The multi-rate
driver subcycles each subsystem at its own dt within one outer step (e.g. a fast
atmosphere over a slow ocean). For long-horizon advance, see the climatology
regime in [`chronos/docs/overview.md`](../chronos/docs/overview.md).

## Conventions

- Edition 2024; Apache-2.0 header on each source file.
- Formatting (`rustfmt.toml`): 2-space indent, max 80 columns, no hard tabs —
  run `cargo fmt --all`.
- A stale `.rmeta` can cause phantom "type not found" errors:
  `cargo clean --package <pkg>` fixes it.
- The repo is indexed by the `codegraph` CLI for navigation (`codegraph query`,
  `callers`, `impact`, …).
