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
(refine on a field's jump), `RegionRefinementCriterion` (refine a spherical cap)
and `FocusLodCriterion` (refine toward a host-supplied point) ship today; the
last reads an input resource the host writes, which is the seam any host-driven
criterion uses (the event bus stays sim → consumer only).

## Consumers react

- **Render / LOD** — the eidolon producer keys each mesh's geometry on the
  `TopologyEpoch` (plus a hash of its cell mask, which reads a field and so can
  change any tick). Unchanged inputs ⇒ it skips the rebuild entirely; an adapt
  bumps the epoch and re-emits `UpdateMeshGeometry` automatically. The
  `MeshRepresentation::Wireframe` cell-outline view makes "where AMR is applied"
  obvious — the grid densifies where the mesh refined.
- **Query** — rebuild `WorldQuery::new(&tessera, r)` on the new mesh.
- **Checkpoint** — `WorldCheckpoint` persists each adaptive mesh's epoch + forest
  leaf codes; on load the adapted mesh is reconstructed (and the field slots
  resized) *before* the stored values are loaded, so an adapted world restarts
  bit-for-bit.

## Focus-driven LOD (refining toward a region of interest)

Refinement can track a region of interest. `FocusLodCriterion` (aether) refines
cells whose projected size (`cell_volume^(1/3) / distance-to-focus`) is large and
coarsens those that are small, so detail follows the focus.

The focus is **the host's**, and it is an *input*:

- `RefinementFocus` lives in `utility` (shared vocab); the host sets it with
  `World::set_refinement_focus`, landing it in pleroma as the inbound
  `ResourceKey::RefinementFocus`.
- aether reads it (the criterion) and drives **tessera** refinement through the
  same barrier as any other criterion.

Note what the simulation is *not* told: nothing here is a camera. A host that
renders will usually derive the focus from its view, but "resolve detail near
here" is the same request whether it comes from a camera, a player, or a probe —
so the sim's vocabulary stays free of rendering concepts.

**The view never round-trips through the simulation.** A host that renders owns
its camera outright: it positions it directly, at render rate, *and* publishes a
focus for LOD. These are two independent consumers of one host-owned value. The
render IR carries no camera at all, deliberately — routing the view through the
sim would gate camera motion on the tick rate, which at a realistic tick cost is
visible as stutter while the view moves.

## Showcase

`sandbox::build_showcase_world` makes its terrain **surface** adaptive and drives
it with `FocusLodCriterion`; the showcase host publishes its orbit-camera eye as
a focus via `set_refinement_focus` each tick, the surface refines toward it, and
the cell-outline wireframe shows the detail tracking the view. The surface
is **coupled** to the atmosphere by orographic lift through a
`GeometricRadialCoupler`, so a re-mesh is a full coupled-AMR exercise: the barrier
rebuilds the coupler's pairings (N:M where a coarse atmosphere cell now overlaps
several fine surface cells) and the lift stage rebuilds its area-weighted gradient
sites — the atmosphere is forced by the *current* surface, not a stale snapshot.

## Coupling and the partitioned solver under AMR

AMR is no longer restricted to inert, uncoupled meshes on the serial solver:

- **Coupling survives a re-mesh.** A coupling entry carries a raw overlap `area`
  plus per-target (gather) and per-source (conservative scatter) normalised
  weights; `GeometricRadialCoupler` matches interface faces by angular footprint
  (1:1 when conforming, N:M when nested) and the adapt barrier rebuilds every
  coupler touching a re-meshed mesh, so coupling stages never read pairings with
  dead cells. Stages resolve the live coupler each run rather than caching a
  snapshot.
- **Any solver, including multi-layer.** The decomposition is mesh-type-erased
  (`Decomposition<3, dyn Mesh<3>>`), so the partitioned Euler solver runs on an
  adapted atmosphere (`decompose_panels` partitions an `AdaptiveMesh` by base
  panel); the barrier rebuilds the decomposition on adapt. Refining one cell of a
  *multi-layer* shell creates angular **and radial** hanging faces (the cell above
  the refined one stays coarse) — the footprint matcher tiles both the same way,
  and partitioned still matches serial bit-for-bit on the refined multi-layer
  mesh.

## Cost of an adapt

Re-meshing is the expensive part of AMR and it lands on the simulation tick, so it
directly bounds how often a consumer gets a fresh frame. Measure it with
`cargo run --release -p sandbox --example amr_bench`, which drives the showcase
world with an orbiting focus and prints the profiler; read `run_adapters.refine`
(the re-mesh) and `extract` (the render-batch build).

Two things dominated and are now cached or indexed:

- **Geometry oracles.** `AdaptiveMesh` reads each leaf's geometry from a uniformly
  refined copy of the base mesh, one per level present. That copy is a full shell
  build and depends only on `(base, level)`, so it is built once and carried
  across adapts rather than rebuilt every time.
- **Face matching.** Leaf faces are grouped by world footprint. Footprints live in
  one flat arena (not a `Vec` key per face) and are grouped by sorting rather than
  hashing, and the hanging-face pass indexes the unmatched faces on a grid of the
  coarse circumradius instead of comparing every leftover against every other.

Sorting also fixed a latent determinism bug: grouping used to run off `HashMap`
iteration order, so `FaceId` assignment — and therefore flux summation order —
varied between runs of the same simulation.

## v1 limitations

Angular (horizontal) refinement only — radial refinement (splitting a cell across
the shell thickness) is deferred, though multi-*layer* meshes refine fine.
Panel-seam-crossing refinement is skipped. Coupled meshes must share their base
angular grid (so interface faces nest exactly). Load balancing across the uneven
partitions a refined panel produces is future work — partitions stay panel-aligned
to keep radial columns (and the HEVI solve) intact.
