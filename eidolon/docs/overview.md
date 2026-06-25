# eidolon — presentation & queries

`eidolon` is the **read-only** seam between a running simulation and anything that
wants to look at it — a renderer, a file exporter, or game logic. It observes
`pleroma` state and `tessera` geometry and produces *owned* data, so simulation
crates never depend on engine types and `eidolon` never mutates the sim.

This is the crate to start with if you only care about getting data out; see
[rendering.md](../../docs/rendering.md) for the consumer-facing walkthrough.

## How it fits

`eidolon` sits at the top of the read side of the pipeline, depending on
`pleroma` + `tessera` but on no physics crate. It offers three consumption
modes — render IR, file export, and a geographic query API — over a consistent,
interpolated snapshot.

## What's inside

- **`ir/`** — the engine-neutral intermediate representation: frames, scalar /
  categorical / mask layers, meshes, transforms, stable IDs (`LayerId`,
  `RenderMeshId`), and an `Update` enum stream of changes. Deliberately
  **art-free**: it carries meaning (temperature in K; this cell is ocean), and
  palettes/glyphs are only reference-renderer hints.
- **`extract/`** — producers that read `pleroma`/`tessera` and build IR. A
  `FrameProducer` is configured by an `ExtractConfig` (which meshes and field
  layers to surface) and diff-caches between frames. `query_extract` builds the
  quantity snapshot the query API reads.
- **`playback.rs`** — `FrameInterpolator`, which buffers recent frames and serves
  values interpolated on an adaptive render clock, smoothing large physics steps.
- **`query.rs`** — the semantic query API: `WorldQuery`, `Sample<T>`
  (`Ok`/`Stale`/`Degraded`/`Unavailable`), `ScalarQuantity` / `VectorQuantity`,
  `Reduction`, addressed in geographic coordinates. The `Quantity` vocabulary is
  the **stable public read contract**, decoupled from internal `FieldName`s.
- **`bevy/` + `backend/`** — the reference Bevy backend (behind the `bevy`
  feature) that applies the IR/`Update` stream.
- **`export/vtk.rs`** — VTK XML output for ParaView and headless analysis.
- **`runtime.rs`** — `render_channel` + `spawn_runner`, the sim-thread →
  render-thread plumbing.

## The art-free principle

`eidolon` provides *information*; the consumer decides the *look*. The IR is data;
appearance (palettes, displacement exaggeration, material per class) is the
renderer's choice. The Bevy backend is a *reference* renderer, not *the*
renderer; `sandbox/src/main.rs` shows a consumer styling the data.

## See also

- Consumer walkthrough: [rendering](../../docs/rendering.md).
- The geographic index it queries through: [`tessera`](../../tessera/docs/overview.md).
