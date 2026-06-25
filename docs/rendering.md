# Reading & rendering a world

This is the entry point if you mostly care about **getting data out** of a
running world — to draw it, export it, or query it from game logic — without
learning the physics. Everything here lives in one crate, `eidolon`, which is a
**read-only** view over the simulation: it observes `pleroma` state and `tessera`
geometry and produces owned data. Simulation crates never depend on a rendering
engine, and `eidolon` never mutates the sim.

There are three ways to consume a world, from "just draw it" to "ask precise
questions":

## 1. Render it (the IR + a backend)

`eidolon` emits an **engine-neutral intermediate representation (IR)**: meshes,
per-cell sample layers, transforms, stable IDs, and an `Update` stream of
changes. A backend applies that IR. The bundled reference backend is Bevy
(`eidolon::bevy`, behind the `bevy` feature), but the IR is the contract — you
could write another.

The flow a consumer wires up:

1. Build a world (or use a `sandbox` builder).
2. Create a `FrameProducer` from an `ExtractConfig` describing *which meshes and
   fields* to surface (scalar layers, categorical layers like land/ocean/ice).
3. Each tick, call `producer.extract(...)` to get a render batch and push it down
   a channel (`eidolon::runtime::render_channel` + `spawn_runner` give you the
   sim-thread → render-thread plumbing).
4. The Bevy backend (`AetherBevyPlugin`) consumes the channel and paints.

The IR is deliberately **art-free**: it carries *meaning* (this layer is
temperature in K; this cell is ocean), not a look. Palettes and glyph hints in
the IR are only the reference renderer's defaults — your renderer is free to
ignore them and style the data however you like. `sandbox/src/main.rs` is the
worked reference: it renders terrain relief by displacing vertices with the
elevation layer, colours land/ocean/ice from the categorical layer, and draws a
translucent scattering atmosphere — all consumer-side art over art-free data.

## 2. Smooth playback (frame interpolation)

A physics step can be large (HEVI takes big stable steps), so rendering raw ticks
looks choppy. `eidolon::playback::FrameInterpolator` buffers recent frames and
serves values interpolated against an adaptive render clock, dilating when the
sim stalls and catching up on a backlog — so the picture stays smooth regardless
of the sim's step size. The Bevy backend uses this automatically.

## 3. Query it (the semantic `Quantity` API)

For game logic that needs *values*, not pixels, `eidolon::query` is a read-only,
thread-safe API addressed in **geographic coordinates** — you ask in lat/lon, not
in conserved-variable layouts or cube-sphere panel indices.

- `WorldQuery::new(&tessera, surface_radius)` builds per-mesh geographic indices
  once from the static geometry.
- `sample_scalar(&snapshot, ScalarQuantity::Temperature, GeoCoord)` returns a
  value at a point; `sample_wind` returns an east-north-up vector;
  `reduce_scalar(..., Reduction::Mean)` aggregates over a region.
- `ScalarQuantity` is a **stable semantic vocabulary** (`Temperature`, `Pressure`,
  `Humidity`, `SeaSurfaceTemperature`, `SurfaceElevation`, climatology means, …)
  decoupled from the engine's internal `FieldName` enum — so it keeps working as
  the physics internals change.

Every result is a `Sample<T>` that carries *how much to trust it*:

| Variant | Meaning |
|---|---|
| `Ok(v)` | fresh, interpolated between two frames |
| `Stale(v)` | usable, but snapped to a single frame (sim hasn't delivered a newer one) |
| `Degraded(v)` | a value was found but is non-finite/suspect — surfaced, not hidden, so a shipped game can react instead of crashing |
| `Unavailable` | the quantity isn't carried for this world, or the point is off the mesh |

Queries read an **interpolated snapshot** off the sim thread, so they are
consistent and never race the solver. This is the same idea as render
interpolation, generalised beyond rendering.

## Exporting to files (VTK)

`eidolon::export` writes VTK XML (`XmlVtuWriter` / `XmlPvtuWriter`) for ParaView
and similar tools — useful for offline analysis or headless runs with no GPU.

## What you do *not* need to know

To render or query, you never touch: conserved-variable layouts, gnomonic
cube-sphere coordinates, the stage DAG, or partitioning. Those are the
simulation layer's concern; `eidolon` is the seam that hides them. If you *do*
want to extend the physics behind the view, continue to
[physics.md](physics.md) and [extending.md](extending.md).

See [`eidolon/docs/overview.md`](../eidolon/docs/overview.md) for the module map.
