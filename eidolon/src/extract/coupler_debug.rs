// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use tessera::{
  mesh::Mesh,
  topology::FaceConnection,
  world_mesh::{CouplerEntry, Tessera},
};
use utility::domain::{FaceId, MeshKey, Point, WorldId};

use crate::{
  extract::mesh::point_to_f32,
  ir::{
    CouplerId, LineMesh, MeshRepresentation, MeshSource, RenderGeometry,
    RenderMesh, RenderMeshId, Rgba,
  },
};

pub fn coupler_face_lines(
  world: WorldId,
  coupler_index: usize,
  tessera: &Tessera,
  entry: &CouplerEntry,
) -> Option<RenderMesh> {
  let mesh_a = tessera.mesh(entry.mesh_a())?;
  let mesh_b = tessera.mesh(entry.mesh_b())?;
  Some(coupler_face_lines_from_meshes(
    world,
    coupler_index,
    entry.mesh_a(),
    entry.mesh_b(),
    mesh_a.as_ref(),
    mesh_b.as_ref(),
    entry,
  ))
}

fn coupler_face_lines_from_meshes(
  world: WorldId,
  coupler_index: usize,
  mesh_a_key: MeshKey,
  mesh_b_key: MeshKey,
  mesh_a: &dyn tessera::mesh::Mesh<3>,
  mesh_b: &dyn tessera::mesh::Mesh<3>,
  entry: &CouplerEntry,
) -> RenderMesh {
  let pairs = entry.coupler().pairs();
  let mut lines = LineMesh {
    positions: Vec::with_capacity(pairs.len() * 2),
    segments: Vec::with_capacity(pairs.len()),
    colours: Vec::with_capacity(pairs.len() * 2),
  };

  for pair in pairs {
    let start = lines.positions.len() as u32;
    let a = coupler_line_endpoint(mesh_a, pair.a());
    let b = coupler_line_endpoint(mesh_b, pair.b());
    lines.positions.push(point_to_f32(&a));
    lines.positions.push(point_to_f32(&b));
    lines.segments.push([start, start + 1]);
    lines.colours.push(Rgba::MAGENTA);
    lines.colours.push(Rgba::YELLOW);
  }

  RenderMesh::new(
    RenderMeshId {
      world,
      mesh: mesh_a_key,
      representation: MeshRepresentation::Coupler(coupler_index),
    },
    format!("coupler {:?} -> {:?}", mesh_a_key, mesh_b_key),
    MeshSource::Coupler(CouplerId {
      world,
      mesh_a: mesh_a_key,
      mesh_b: mesh_b_key,
      index: coupler_index,
    }),
    RenderGeometry::Lines(lines),
  )
}

fn coupler_line_endpoint(mesh: &dyn Mesh<3>, face: FaceId) -> Point<3> {
  match mesh.face_connection(face) {
    FaceConnection::Boundary { owner, .. }
    | FaceConnection::Interior { owner, .. } => {
      mesh.cell_world_centroid(*owner)
    }
  }
}

#[cfg(test)]
mod tests {
  use std::sync::Arc;

  use tessera::{
    coupling::MeshCoupler,
    cube_sphere::{CubeSphere, CubeSphereShellSpec},
    mesh::Mesh,
    radial_stack::RadialStackCoupler,
    world_mesh::Tessera,
  };
  use utility::domain::{BoundaryTag, MeshKey, WorldId};

  use super::*;

  #[test]
  fn radial_coupler_lines_connect_cells_not_coincident_faces() {
    let angular_dims = [2, 2];
    let surface_layers = 2;
    let atmosphere_layers = 4;
    let surface = Arc::new(CubeSphere::shell(
      CubeSphereShellSpec::uniform(
        [angular_dims[0], angular_dims[1], surface_layers],
        0.9,
        1.0,
      )
      .with_boundaries(BoundaryTag::Ground, BoundaryTag::AtmosphereEdge),
    ));
    let atmosphere = Arc::new(CubeSphere::shell(
      CubeSphereShellSpec::uniform(
        [angular_dims[0], angular_dims[1], atmosphere_layers],
        1.0,
        1.2,
      )
      .with_boundaries(BoundaryTag::Ground, BoundaryTag::AtmosphereEdge),
    ));

    let mut tessera = Tessera::new();
    tessera.register_mesh(MeshKey::SURFACE, surface as Arc<dyn Mesh<3>>);
    tessera.register_mesh(MeshKey::ATMOSPHERE, atmosphere as Arc<dyn Mesh<3>>);
    let coupler =
      RadialStackCoupler::new(angular_dims, surface_layers, atmosphere_layers);
    let pair_count = coupler.pairs().len();
    tessera.add_coupler(MeshKey::SURFACE, MeshKey::ATMOSPHERE, coupler);

    let render_mesh =
      coupler_face_lines(WorldId(0), 0, &tessera, &tessera.couplers()[0])
        .unwrap();
    let RenderGeometry::Lines(lines) = render_mesh.geometry else {
      panic!("expected line mesh");
    };

    assert_eq!(lines.segments.len(), pair_count);
    for segment in &lines.segments {
      let a = lines.positions[segment[0] as usize];
      let b = lines.positions[segment[1] as usize];
      let distance =
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2))
          .sqrt();
      assert!(distance > 1.0e-4);
    }
  }
}
