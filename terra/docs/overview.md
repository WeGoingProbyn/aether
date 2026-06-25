# terra — surface & terrain

`terra` owns the planet's surface: a thermal slab that warms and cools under the
net radiative flux, plus **first-class terrain** — elevation, a land/ocean/ice
mask, and surface albedo. As with every physics crate, it stores nothing
globally; it registers fields and adds stages.

## How it fits

`terra` registers its fields on the surface mesh and consumes the
`NetSurfaceFlux` that `lumen` deposits there. Its terrain fields feed two
consumers: `aer` (orographic lift reads the elevation slope) and `lumen`
(radiation reads the albedo field). `eidolon` renders the heightfield as relief
and the mask as material classes.

## What's inside

- **Surface thermal slab** — steps surface temperature by
  `dT/dt = NetSurfaceFlux / heat_capacity_per_area`.
- **`terrain.rs`** — `TerrainModel`: registers a static `SurfaceElevation`
  heightfield and a categorical `SurfaceType` (land/ocean/ice, encoded via
  `SurfaceClass` from `utility::domain`), generated from a
  `Fn(GeoCoord) -> TerrainSample` (e.g. `earthlike_terrain`). These are **inert**
  — set once at setup, never evolved.
- **Surface albedo** — `TerrainModel` also registers and maintains a per-cell
  `SurfaceAlbedo` field. This is the reusable "surface property as a field"
  contract: terrain writes a base albedo and any later producer (snow, ice) can
  blend into the same field, which radiation reads — making ice-albedo feedback a
  drop-in.

## Why `SurfaceClass` lives in `utility`

The class encoding is shared vocabulary so the producer (`terra`) and consumers
(`eidolon`'s query API) agree on the meaning of the numeric codes without
depending on each other. `terra` re-exports it for convenience.

## See also

- Terrain → atmosphere coupling (orographic lift):
  [`aer`](../../aer/docs/overview.md), [physics](../../docs/physics.md).
- The ocean counterpart: [`thalassa`](../../thalassa/docs/overview.md).
