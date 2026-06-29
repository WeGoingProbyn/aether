# Roadmap

This document records the **agreed direction of travel** so it stays consistent
across work sessions. It is deliberately about *what is not built yet* and the
order we intend to build it. For what already exists, see
[architecture.md](architecture.md), [physics.md](physics.md), and each crate's
`docs/overview.md`.

The guiding principle is unchanged: every new capability is a *seam* (a named
field, a resource, a typed coupling, a stage), never a global or a special case.
Infrastructure lands before the features that depend on it.

## Where we are

The simulation and read layers are end-to-end complete: assembled worlds of
coupled moist atmosphere + ocean + terrain run stably, a semantic geographic
query API reads interpolated snapshots off-thread, and an in-DAG **diagnostics**
layer monitors finiteness and conservation drift with a configurable
`DiagnosticsPolicy`. That diagnostics work is the first piece of a *control
plane*, and building it surfaced the next gaps precisely: there is no way to roll
back a bad tick, and no way to broadcast a state change to anyone but the log.

## Direction (ordered)

The ordering is chosen so each step makes the next one safer or cheaper. Earlier
steps are smaller and unlock the later ones.

### 1. Persistence — save / load + checkpointing  (Done)

A way to snapshot and restore the full mutable world state.

- **Why first.** It is the smallest surface with the widest payoff. It closes the
  diagnostics rollback gap (`DiagnosticsPolicy::Fail` can today only freeze the
  clock and leave the field dirty — *true* freeze-at-last-good needs a
  checkpoint), enables restartable long climatology runs, and is the prerequisite
  for deterministic replay testing.
- **Where.** `pleroma` owns *all* mutable state in one registry, so it is the
  natural serialization boundary; `utility`'s `Serialize`/`Deserialize` traits +
  JSON/VTK backends already exist. The work is a state-registry round-trip, not
  new infrastructure.

### 2. Event bus — state broadcast (Done)

A deferred-dispatch event channel so stages can announce state changes and
consumers (or other stages) can react.

- **Why second.** Three half-built things are waiting on it: **instability-as-
  contract** (a `Fail` blow-up becomes a subscribable event — throttle dt,
  restore a checkpoint, flip a regime — instead of only an `Err`), **regime
  transitions** (the live↔climatology switch is a polled flag that wants to be an
  event), and **consumer reactions** (eidolon/sandbox reacting to "terrain
  registered", "regime changed", "field stale" instead of blind re-polling).
- **Design constraint.** `nexus` is a parallel DAG, so the bus must be
  *deferred-dispatch*: stages write events into a resource sink (mirroring the
  `Diagnostics` resource pattern), the bus drains and dispatches at an end-of-tick
  barrier. Anything else reintroduces the global mutable coupling the architecture
  exists to avoid.

### 3. A physics realism increment (Done)

One self-contained physics step to exercise the new infrastructure on something
real and keep the reference world advancing. Two candidates, either order:

- **Land–sea heat/moisture masking.** Today the showcase is an
  aqua-planet-with-terrain: land differs from ocean only in albedo, orography,
  and render. Evaporation and the water column run uniformly and land has no skin
  temperature. The fix is per-cell mask-weighted couplings (a new `syzygy`
  capability) or a moisture-availability surface field — the same
  "surface-property as a field" pattern the albedo seam already proved.
- **Ice model + ice-fraction field.** The `SurfaceAlbedo` seam was built
  ice-ready: a second producer blending into the albedo field by ice fraction
  gives ice-albedo feedback for free. Needs an ice-fraction field and a minimal
  ice model.

### 4. AMR / LOD — the capstone  *(v1 DONE)*

Adaptive mesh refinement / level-of-detail for the data plane. The v1
implementation is in place end-to-end — see [amr.md](amr.md) for the design.

- **Why last.** It is the heaviest item on the board and touches nearly every
  crate: `tessera` (refinement topology, hanging nodes), `pleroma` (fields must
  grow/shrink — storage is fixed-length today), `continuum` (flux at refinement
  boundaries), `nexus` (mid-run rebalancing), and `eidolon` (the IR assumes
  stable cell IDs). It wanted every other foundation under it first — checkpointing
  for refine-time snapshots, events for refinement triggers, and the already-built
  stable query layer.
- **What v1 delivers.** A mesh-agnostic refinement contract (`tessera::refine`),
  an `AdaptiveMesh` wrapper that rebuilds a conforming + hanging-node mesh from a
  per-cell refinement forest, conservative field remap (`pleroma`), an end-of-tick
  adaptation driver with criteria + a governor (`aether::adapt`), and read-side
  reactions: render-LOD (the producer re-emits geometry), the query index rebuild,
  and checkpoint persistence of the adapted topology. Dense `CellId` stays the
  hot-path key; a `TopologyEpoch` + `CellRemap` (broadcast via the event bus) is
  how consumers survive a re-mesh. The cube-sphere is the first refinable backend.
- **Coupled meshes + any solver (follow-on, DONE).** AMR is no longer limited to
  inert, uncoupled meshes on the serial solver. Coupling entries are area-weighted
  (gather / conservative-scatter), a `GeometricRadialCoupler` matches interface
  faces by angular footprint (1:1 → N:M under refinement), and the adapt barrier
  rebuilds couplers + the partitioned decomposition atomically with the mesh swap;
  coupling stages and orographic lift read the live coupler so nothing references a
  dead cell. The decomposition is mesh-type-erased, so the partitioned Euler solver
  runs on an adapted atmosphere (partitioned == serial bit-for-bit). See
  [amr.md](amr.md).
- **v1 limitations (deferred).** Angular (horizontal) refinement only; refinement
  that crosses a cube-sphere *panel seam* is skipped (best-effort); coupled meshes
  must share their base angular grid; load balancing across the uneven partitions a
  refined panel produces, multi-layer (column) refinement of a solver mesh, and
  radial refinement are future work.
- **Render-only LOD** fell out of the same topology-change path rather than being a
  separate milestone.

## Standing deferred items

Orthogonal to the sequence above; pulled in when they block something or when a
showcase needs them:

- **GPU / visual debt** — categorical per-class surface materials, true elevation
  displacement, translucent atmosphere shell. Consumer/visual work gated on a
  display; blocks nothing architectural.
- **Authoring / forcing** — prescribed external forcing (wind stress, heat flux,
  boundary nudging); everything is internally generated today.
- **Rendering-style refactor** — move colour/palette fully out of the core IR into
  a consumer style-sheet, so the IR carries information and the consumer owns art.
- **Determinism guarantees** — beyond save/load: reproducible parallel reduction
  order for bit-exact replay.
- **Event-bus hardening** — the step-2 `EventBus` ships with a single `Mutex`
  buffer, no intra-tick ordering, and no volume bound. Future work: lock-free /
  per-stage emit buffers, a per-tick cap or per-kind dedup policy, and (if a
  consumer needs it) push-style subscriber callbacks alongside the poll API.
