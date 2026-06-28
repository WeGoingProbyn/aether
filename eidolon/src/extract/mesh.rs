// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use tessera::mesh::Mesh;
use utility::domain::{BoundaryTag, CellId, FaceId, MeshKey, Point, WorldId};

use crate::ir::{
  LineMesh, MeshRepresentation, MeshSource, PointCloud, RenderGeometry,
  RenderMesh, RenderMeshId, Rgba, TriangleMesh,
};

pub fn cell_centroid_points(
  world: WorldId,
  mesh_key: MeshKey,
  mesh: &dyn Mesh<3>,
) -> RenderMesh {
  let mut cloud = PointCloud {
    positions: Vec::with_capacity(mesh.cell_count()),
    colours: Vec::with_capacity(mesh.cell_count()),
    cell_ids: Vec::with_capacity(mesh.cell_count()),
    face_ids: Vec::with_capacity(mesh.cell_count()),
  };

  for cell in (0..mesh.cell_count()).map(CellId::from) {
    cloud
      .positions
      .push(point_to_f32(&mesh.cell_world_centroid(cell)));
    cloud.colours.push(Rgba::WHITE);
    cloud.cell_ids.push(Some(cell));
    cloud.face_ids.push(None);
  }

  RenderMesh::new(
    RenderMeshId {
      world,
      mesh: mesh_key,
      representation: MeshRepresentation::Cells,
    },
    "cell centroids",
    MeshSource::TesseraMesh(mesh_key),
    RenderGeometry::Points(cloud),
  )
}

pub fn boundary_face_centroid_points(
  world: WorldId,
  mesh_key: MeshKey,
  mesh: &dyn Mesh<3>,
  tag: BoundaryTag,
) -> RenderMesh {
  face_centroid_points_from_faces(
    world,
    mesh_key,
    mesh,
    MeshRepresentation::BoundaryFaces,
    format!("boundary {tag:?} face centroids"),
    mesh.boundary_faces(tag).iter().map(|(face, _)| *face),
  )
}

pub fn all_boundary_face_centroid_points(
  world: WorldId,
  mesh_key: MeshKey,
  mesh: &dyn Mesh<3>,
) -> RenderMesh {
  let faces = mesh
    .boundary_tags()
    .flat_map(|tag| mesh.boundary_faces(tag).iter().map(|(face, _)| *face))
    .collect::<Vec<_>>();

  face_centroid_points_from_faces(
    world,
    mesh_key,
    mesh,
    MeshRepresentation::BoundaryFaces,
    "boundary face centroids",
    faces,
  )
}

pub fn all_face_centroid_points(
  world: WorldId,
  mesh_key: MeshKey,
  mesh: &dyn Mesh<3>,
) -> RenderMesh {
  face_centroid_points_from_faces(
    world,
    mesh_key,
    mesh,
    MeshRepresentation::DebugPoints,
    "face centroids",
    (0..mesh.face_count()).map(FaceId::from),
  )
}

pub fn boundary_surface_triangles(
  world: WorldId,
  mesh_key: MeshKey,
  mesh: &dyn Mesh<3>,
) -> RenderMesh {
  let faces = mesh
    .boundary_tags()
    .flat_map(|tag| {
      mesh
        .boundary_faces(tag)
        .iter()
        .map(|(face, owner)| (*face, *owner))
    })
    .collect::<Vec<_>>();

  boundary_surface_triangles_from_faces(
    world,
    mesh_key,
    mesh,
    "boundary surface",
    faces,
  )
}

/// Boundary surface, but only for cells where `keep(cell)` is true. Lets a
/// consumer render a mesh on a *subset* of the globe — e.g. the land surface on
/// terrain cells and the ocean shell on sea cells, which tile without
/// overlapping so two coincident shells never z-fight.
pub fn boundary_surface_triangles_masked(
  world: WorldId,
  mesh_key: MeshKey,
  mesh: &dyn Mesh<3>,
  keep: impl Fn(CellId) -> bool,
) -> RenderMesh {
  let faces = mesh
    .boundary_tags()
    .flat_map(|tag| {
      mesh
        .boundary_faces(tag)
        .iter()
        .map(|(face, owner)| (*face, *owner))
    })
    .filter(|(_, owner)| keep(*owner))
    .collect::<Vec<_>>();

  boundary_surface_triangles_from_faces(
    world,
    mesh_key,
    mesh,
    "boundary surface (masked)",
    faces,
  )
}

/// A wireframe of every boundary cell's outline: the four edges of each boundary
/// face drawn as line segments. On a shell this is the angular cell grid on the
/// surface, so where the mesh is adaptively refined the wireframe visibly
/// densifies — a debug view that makes "where AMR is applied" obvious.
pub fn cell_outline_lines(
  world: WorldId,
  mesh_key: MeshKey,
  mesh: &dyn Mesh<3>,
) -> RenderMesh {
  let mut lines = LineMesh::default();
  for tag in mesh.boundary_tags() {
    for &(face, _owner) in mesh.boundary_faces(tag) {
      let Some(vertices) = mesh.face_world_vertices(face) else {
        continue;
      };
      if vertices.len() != 4 {
        continue;
      }
      let base = lines.positions.len() as u32;
      lines.positions.extend(vertices.iter().map(point_to_f32));
      lines.colours.extend(std::iter::repeat_n(Rgba::WHITE, 4));
      // The four edges of the quad, closing back to the first vertex.
      for k in 0..4 {
        lines.segments.push([base + k, base + (k + 1) % 4]);
      }
    }
  }

  RenderMesh::new(
    RenderMeshId {
      world,
      mesh: mesh_key,
      representation: MeshRepresentation::Wireframe,
    },
    "cell outlines",
    MeshSource::TesseraMesh(mesh_key),
    RenderGeometry::Lines(lines),
  )
}

pub fn boundary_surface_triangles_for_tag(
  world: WorldId,
  mesh_key: MeshKey,
  mesh: &dyn Mesh<3>,
  tag: BoundaryTag,
) -> RenderMesh {
  boundary_surface_triangles_from_faces(
    world,
    mesh_key,
    mesh,
    format!("boundary {tag:?} surface"),
    mesh
      .boundary_faces(tag)
      .iter()
      .map(|(face, owner)| (*face, *owner)),
  )
}

fn boundary_surface_triangles_from_faces(
  world: WorldId,
  mesh_key: MeshKey,
  mesh: &dyn Mesh<3>,
  label: impl Into<String>,
  faces: impl IntoIterator<Item = (FaceId, CellId)>,
) -> RenderMesh {
  let mut triangles = TriangleMesh::default();

  for (face, owner) in faces {
    let Some(vertices) = mesh.face_world_vertices(face) else {
      continue;
    };
    if vertices.len() != 4 {
      continue;
    }

    let base = triangles.positions.len() as u32;
    triangles
      .positions
      .extend(vertices.iter().map(point_to_f32));
    triangles.normals.extend(
      std::iter::repeat_with(|| {
        let normal = mesh.face_area_vector(face).normalise();
        [normal[0] as f32, normal[1] as f32, normal[2] as f32]
      })
      .take(4),
    );
    triangles
      .colours
      .extend(std::iter::repeat_n(Rgba::WHITE, 4));
    triangles.indices.extend_from_slice(&[
      base,
      base + 1,
      base + 2,
      base,
      base + 2,
      base + 3,
    ]);
    triangles.cell_ids.extend([Some(owner), Some(owner)]);
    triangles.face_ids.extend([Some(face), Some(face)]);
  }

  RenderMesh::new(
    RenderMeshId {
      world,
      mesh: mesh_key,
      representation: MeshRepresentation::BoundaryFaces,
    },
    label,
    MeshSource::TesseraMesh(mesh_key),
    RenderGeometry::Triangles(triangles),
  )
}

fn face_centroid_points_from_faces(
  world: WorldId,
  mesh_key: MeshKey,
  mesh: &dyn Mesh<3>,
  representation: MeshRepresentation,
  label: impl Into<String>,
  faces: impl IntoIterator<Item = FaceId>,
) -> RenderMesh {
  let faces = faces.into_iter();
  let (lower, _) = faces.size_hint();
  let mut cloud = PointCloud {
    positions: Vec::with_capacity(lower),
    colours: Vec::with_capacity(lower),
    cell_ids: Vec::with_capacity(lower),
    face_ids: Vec::with_capacity(lower),
  };

  for face in faces {
    cloud
      .positions
      .push(point_to_f32(&mesh.face_world_centroid(face)));
    cloud.colours.push(Rgba::CYAN);
    cloud.cell_ids.push(None);
    cloud.face_ids.push(Some(face));
  }

  RenderMesh::new(
    RenderMeshId {
      world,
      mesh: mesh_key,
      representation,
    },
    label,
    MeshSource::TesseraMesh(mesh_key),
    RenderGeometry::Points(cloud),
  )
}

pub(crate) fn point_to_f32(point: &Point<3>) -> [f32; 3] {
  [point[0] as f32, point[1] as f32, point[2] as f32]
}

#[cfg(test)]
mod tests {
  use tessera::cube_sphere::CubeSphere;
  use utility::domain::{MeshKey, WorldId};

  use super::*;

  #[test]
  fn cube_sphere_cell_points_are_in_world_space() {
    let mesh = CubeSphere::new([1, 1, 1], 1.0, 2.0);
    let render_mesh = cell_centroid_points(WorldId(0), MeshKey::SURFACE, &mesh);
    let RenderGeometry::Points(points) = render_mesh.geometry else {
      panic!("expected point cloud");
    };

    let p = points.positions[0];
    let radius = (p[0] as f64).hypot(p[1] as f64).hypot(p[2] as f64);

    assert!((radius - 1.5).abs() < 1.0e-6);
  }

  #[test]
  fn cube_sphere_boundary_surface_extracts_triangles() {
    let mesh = CubeSphere::new([2, 2, 1], 1.0, 2.0);
    let render_mesh =
      boundary_surface_triangles(WorldId(0), MeshKey::SURFACE, &mesh);
    let RenderGeometry::Triangles(triangles) = render_mesh.geometry else {
      panic!("expected triangle mesh");
    };

    assert_eq!(triangles.positions.len(), 2 * 6 * 2 * 2 * 4);
    assert_eq!(triangles.triangle_count(), 2 * 6 * 2 * 2 * 2);
    assert_eq!(triangles.face_ids.len(), triangles.triangle_count());
  }

  #[test]
  fn masked_boundary_surface_is_a_disjoint_partition_of_the_full_mesh() {
    let mesh = CubeSphere::new([3, 3, 1], 1.0, 2.0);
    let full =
      match boundary_surface_triangles(WorldId(0), MeshKey::SURFACE, &mesh)
        .geometry
      {
        RenderGeometry::Triangles(t) => t.triangle_count(),
        _ => panic!("expected triangles"),
      };

    // Partition cells by index parity; the two masks must tile the full mesh.
    let even = |c: CellId| c.index() % 2 == 0;
    let count =
      |keep: &dyn Fn(CellId) -> bool| match boundary_surface_triangles_masked(
        WorldId(0),
        MeshKey::SURFACE,
        &mesh,
        keep,
      )
      .geometry
      {
        RenderGeometry::Triangles(t) => t.triangle_count(),
        _ => panic!("expected triangles"),
      };

    let evens = count(&even);
    let odds = count(&|c| !even(c));
    assert!(evens > 0 && odds > 0, "both masks should keep some cells");
    assert!(evens < full, "masking should drop faces");
    assert_eq!(evens + odds, full, "masks must tile the full mesh exactly");
  }
}
