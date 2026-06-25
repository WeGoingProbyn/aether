# tessera — geometry & space

`tessera` answers *where* simulation happens. It owns meshes, their topology,
how they're partitioned for parallelism, how separate meshes couple, and how to
address the world in geographic (lat/lon) coordinates. It holds **no mutable
simulation state** — only the static spatial structure that state lives on.

## How it fits

`pleroma` fields are sized and indexed by `tessera` meshes; `nexus` hands stages
read-only `tessera` geometry alongside their borrowed state; `eidolon` reads
`tessera` to build render geometry and the geographic query index. It is the
shared notion of space the whole pipeline agrees on.

## What's inside

- **`mesh.rs`** — `Mesh<D>`, the super-trait `CellGeometry<D> + FaceGeometry<D> +
  Topology` that every mesh satisfies.
- **`geometry.rs` / `topology.rs`** — cell centroids, volumes, face areas,
  normals, and connectivity.
- **`cube_sphere.rs` / `radial_stack.rs`** — curvilinear cubed-sphere shells and
  stacked radial layers, the building blocks of planetary atmospheres, oceans,
  and surfaces.
- **`partition.rs`** — domain decomposition into stripes with ghost layers, so
  `nexus` can run N serial solvers over N partitions in parallel.
- **`coupling.rs` / `world_mesh.rs`** — cross-mesh couplers and `Tessera`, the
  multi-mesh container a world holds.
- **`geo.rs` / `spatial.rs`** — `GeoCoord { lat, lon, alt }` and a planet-agnostic
  conversion to/from world Cartesian, plus `GeoIndex`, a lat/lon bucket grid for
  point and region lookups robust at the poles and panel seams. This is what lets
  `eidolon`'s query API speak geography.

## Gotcha — curvilinear metrics

On a cube-sphere shell the *volume* metric must **not** be reused as a
*face-area* metric. `face_sqrt_det_metric` defaults to `sqrt_det_metric` for
Cartesian domains but must be overridden for curvilinear ones, or characteristic
lengths come out wrong by a factor of the radius.

## See also

- Adding a mesh type: [extending](../../docs/extending.md#add-a-new-mesh-type).
- Geographic queries built on `geo`/`spatial`:
  [rendering](../../docs/rendering.md), [`eidolon`](../../eidolon/docs/overview.md).
