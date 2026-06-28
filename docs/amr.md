# Adaptive mesh refinement (AMR / LOD)

AMR lets a mesh add resolution where it is needed and remove it where it is not,
at runtime, while the simulation keeps running and every read-side consumer
(solver, query, render, checkpoint) stays consistent. This is roadmap step 4; the
guiding rule is unchanged — every new capability is a *seam*, never a global or a
special case.

## The core problem and the keystone

Almost everything aligns on one invariant: a **dense, stable `CellId`** in
`[0, N)`. Fields (`SoaField`/`AosField`), masks, the geo-index, the render
samples, and checkpoints all index it. AMR changes `N` and the meaning of each id
at runtime, so the keystone is an explicit identity contract:

- **`TopologyEpoch`** (`utility::domain`) — a per-mesh version stamp, bumped once
  per adapt. Consumers compare epochs to notice the cell space changed.
- **`CellRemap`** — the old↔new correspondence, with birth/death in *both*
  directions: `old → Option<new>` (None = a cell that was refined/merged away) and
  `new → NewCellSource` (`Survivor` / `Child{parent}` / `Merge{children}`), which
  is also the field-transfer rule.

Dense `CellId` stays the hot-path key; the epoch + remap (broadcast on the
`utility::events` `EventBus` as `Event::TopologyChanged`) is how everyone else
detects and survives a re-mesh.

## The refinement seam (`tessera::refine`, `tessera::adaptive`)

Responsibilities split so no mesh-agnostic logic leaks into a backend:

- The **driver** decides *what* to adapt → a desired `RefineFlags`.
- The mesh-agnostic **balancer** `balance_2to1` turns that into a 2:1-balanced
  `AdaptRequest` (it reads only `Topology` + a cell-level function).
- The **backend** (`RefinableMesh`) owns *how*: only it knows its child layout, so
  it rebuilds the mesh and computes the `CellRemap`.

`AdaptiveMesh` is the mesh-agnostic wrapper that implements `RefinableMesh` for any
base mesh providing the small `Subdividable` geometry hook (the cube-sphere is the
first backend). It holds a per-cell **refinement forest** (a quadtree per base
cell; leaves are the active cells) and, on each adapt, rebuilds the flat geometry
and topology that the engine sees.

### Hanging faces

Refinement is angular (radial layers preserved), so an interface is either
*conforming* or *hanging* (a coarse cell meets several fine cells). A hanging
interface is represented as **N ordinary `Interior` sub-faces**, each carrying the
*fine* sub-face's area and outward normal. Consequence: the finite-volume kernel
in `continuum` needs **no special case** — it already iterates interior faces —
and conservation follows from Σ(fine areas) = coarse area. This also handles
arbitrary level jumps; 2:1 balance is the driver's *accuracy* policy, not a
geometry requirement. Per-leaf geometry is read from a uniformly-refined copy of
the base mesh at the leaf's level, so no curvilinear maths lives in the wrapper.

## State remap (`pleroma`)

On adapt, `Pleroma::remap_mesh_fields` rebuilds every field on the mesh from the
`CellRemap`: survivors/children copy the parent value (piecewise-constant
prolongation — conservative for cell-averaged states), merges volume-average their
children (restriction). It is *swap-to-new*: the old buffer stays live until the
new one is built, so a remap never leaves a field half-resized.

## The adaptation driver (`aether::adapt`)

A `World` runs its registered `MeshAdapter`s at the **end-of-tick barrier** — after
a successful tick, never mid-DAG. Each adapter: evaluates its `RefinementCriterion`
→ balances → refines the mesh → remaps the fields → swaps the new mesh into the
`Tessera` with a bumped epoch → emits `TopologyChanged`. An `AdaptGovernor` bounds
cost (adapt every N ticks; cap cells changed per adapt). A refine the backend
cannot realise (e.g. one crossing a panel seam) is skipped best-effort rather than
failing the tick. Criteria are mesh- and consumer-agnostic — `GradientCriterion`
(refine on a field's jump) and `RegionRefinementCriterion` (refine a spherical cap)
ship today; a future view-dependent (camera) criterion would read an input
resource the host writes, the same seam (the event bus stays sim → consumer only).

## Consumers react

- **Render / LOD** — the eidolon producer rebuilds geometry from the current mesh
  each frame and diffs it, so a refined mesh re-emits `UpdateMeshGeometry`
  automatically. The `MeshRepresentation::Wireframe` cell-outline view makes "where
  AMR is applied" obvious — the grid densifies where the mesh refined.
- **Query** — rebuild `WorldQuery::new(&tessera, r)` on the new mesh.
- **Checkpoint** — `WorldCheckpoint` persists each adaptive mesh's epoch + forest
  leaf codes; on load the adapted mesh is reconstructed (and the field slots
  resized) *before* the stored values are loaded, so an adapted world restarts
  bit-for-bit.

## Showcase

`sandbox::build_showcase_world` makes its (inert) terrain **surface** adaptive and
attaches a `RegionRefinementCriterion` over a panel-interior cap; the surface
refines after the governor's cadence and the cell-outline wireframe shows it.
Surface refinement never conflicts with the partitioned atmosphere (it carries no
solver) and the orographic lift sites are baked at setup, so the world stays
stable across a re-mesh.

## v1 limitations

Angular refinement only (radial deferred); panel-seam-crossing refinement is
skipped; AMR runs on the serial solver path (single partition). AMR-aware
partitioning / load balancing, radial refinement, and coupler remap under AMR are
future work.
