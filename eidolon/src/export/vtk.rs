// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::{
  fs::{self, File},
  path::{Path, PathBuf},
};

use utility::{
  error::AetherResult,
  serial::{
    field::{
      FieldArray, FieldAssociation, FieldDataset, FieldDatasetWriter,
      FieldValues, UnstructuredMesh,
    },
    vtk::XmlVtuWriter,
  },
};

use crate::ir::{
  LineMesh, PointCloud, RenderFrame, RenderGeometry, RenderLayer, RenderMesh,
  RenderWorld, ScalarSamples, TriangleMesh,
};

const VTK_VERTEX: u8 = 1;
const VTK_LINE: u8 = 3;
const VTK_TRIANGLE: u8 = 5;

pub fn write_render_frame_vtu(
  frame: &RenderFrame,
  dir: impl AsRef<Path>,
) -> AetherResult<Vec<PathBuf>> {
  let dir = dir.as_ref();
  fs::create_dir_all(dir)?;
  clear_existing_vtu_files(dir)?;

  let mut written = Vec::new();
  for world in &frame.worlds {
    for (mesh_index, mesh) in world.meshes.iter().enumerate() {
      let Some(dataset) = render_mesh_dataset(world, mesh) else {
        continue;
      };

      let path = dir.join(render_mesh_filename(frame, world, mesh_index, mesh));
      let file = File::create(&path)?;
      let mut writer = XmlVtuWriter::new(file);
      writer.write_dataset(&dataset)?;
      written.push(path);
    }
  }

  Ok(written)
}

fn clear_existing_vtu_files(dir: &Path) -> AetherResult<()> {
  for entry in fs::read_dir(dir)? {
    let entry = entry?;
    let path = entry.path();
    if path.extension().is_some_and(|extension| extension == "vtu") {
      fs::remove_file(path)?;
    }
  }
  Ok(())
}

pub fn render_mesh_dataset(
  world: &RenderWorld,
  mesh: &RenderMesh,
) -> Option<FieldDataset> {
  let (unstructured, mut arrays) = match &mesh.geometry {
    RenderGeometry::Points(points) => point_cloud_dataset(points),
    RenderGeometry::Lines(lines) => line_mesh_dataset(lines),
    RenderGeometry::Triangles(triangles) => triangle_mesh_dataset(triangles),
    RenderGeometry::Packed(_) => return None,
  };

  arrays.extend(
    world
      .layers
      .iter()
      .filter_map(|layer| layer_array_for_mesh(layer, mesh)),
  );

  Some(FieldDataset {
    title: mesh.label.clone(),
    mesh: unstructured,
    arrays,
  })
}

fn point_cloud_dataset(
  points: &PointCloud,
) -> (UnstructuredMesh, Vec<FieldArray>) {
  let count = points.positions.len();
  let connectivity = (0..count as u64).collect::<Vec<_>>();
  let offsets = (1..=count as u64).collect::<Vec<_>>();
  let cell_types = vec![VTK_VERTEX; count];

  let mut arrays = Vec::new();
  if points.cell_ids.len() == count {
    arrays.push(source_cell_id_array(
      FieldAssociation::Point,
      points.cell_ids.iter().copied(),
    ));
  }
  if points.face_ids.len() == count {
    arrays.push(source_face_id_array(
      FieldAssociation::Point,
      points.face_ids.iter().copied(),
    ));
  }
  if points.colours.len() == count {
    arrays.push(colour_array(
      FieldAssociation::Point,
      points.colours.iter().map(|colour| colour.as_array()),
    ));
  }

  (
    UnstructuredMesh {
      points: flatten_positions(&points.positions),
      point_components: 3,
      connectivity,
      offsets,
      cell_types,
    },
    arrays,
  )
}

fn line_mesh_dataset(lines: &LineMesh) -> (UnstructuredMesh, Vec<FieldArray>) {
  let mut connectivity = Vec::with_capacity(lines.segments.len() * 2);
  let mut offsets = Vec::with_capacity(lines.segments.len());
  for segment in &lines.segments {
    connectivity.push(segment[0] as u64);
    connectivity.push(segment[1] as u64);
    offsets.push(connectivity.len() as u64);
  }

  let mut arrays = Vec::new();
  if lines.colours.len() == lines.positions.len() {
    arrays.push(colour_array(
      FieldAssociation::Point,
      lines.colours.iter().map(|colour| colour.as_array()),
    ));
  }

  (
    UnstructuredMesh {
      points: flatten_positions(&lines.positions),
      point_components: 3,
      connectivity,
      offsets,
      cell_types: vec![VTK_LINE; lines.segments.len()],
    },
    arrays,
  )
}

fn triangle_mesh_dataset(
  triangles: &TriangleMesh,
) -> (UnstructuredMesh, Vec<FieldArray>) {
  let triangle_count = triangles.indices.len() / 3;
  let connectivity = triangles
    .indices
    .iter()
    .copied()
    .map(u64::from)
    .collect::<Vec<_>>();
  let offsets = (1..=triangle_count)
    .map(|triangle| (triangle * 3) as u64)
    .collect::<Vec<_>>();

  let mut arrays = Vec::new();
  if triangles.cell_ids.len() == triangle_count {
    arrays.push(source_cell_id_array(
      FieldAssociation::Cell,
      triangles.cell_ids.iter().copied(),
    ));
  }
  if triangles.face_ids.len() == triangle_count {
    arrays.push(source_face_id_array(
      FieldAssociation::Cell,
      triangles.face_ids.iter().copied(),
    ));
  }
  if triangles.colours.len() == triangles.positions.len() {
    arrays.push(colour_array(
      FieldAssociation::Point,
      triangles.colours.iter().map(|colour| colour.as_array()),
    ));
  }

  (
    UnstructuredMesh {
      points: flatten_positions(&triangles.positions),
      point_components: 3,
      connectivity,
      offsets,
      cell_types: vec![VTK_TRIANGLE; triangle_count],
    },
    arrays,
  )
}

fn layer_array_for_mesh(
  layer: &RenderLayer,
  mesh: &RenderMesh,
) -> Option<FieldArray> {
  let RenderLayer::Scalar(layer) = layer else {
    return None;
  };
  if layer.target != mesh.id {
    return None;
  }

  let (association, values) =
    scalar_values_for_geometry(&mesh.geometry, &layer.samples)?;

  Some(FieldArray {
    name: layer.label.clone(),
    association,
    components: 1,
    values: FieldValues::F64(values),
  })
}

fn scalar_values_for_geometry(
  geometry: &RenderGeometry,
  samples: &ScalarSamples,
) -> Option<(FieldAssociation, Vec<f64>)> {
  match (geometry, samples) {
    (RenderGeometry::Points(points), ScalarSamples::PerCell(values)) => {
      map_point_cell_values(points, values)
        .map(|values| (FieldAssociation::Point, values))
    }
    (RenderGeometry::Points(points), ScalarSamples::PerFace(values)) => {
      map_point_face_values(points, values)
        .map(|values| (FieldAssociation::Point, values))
    }
    (RenderGeometry::Points(points), ScalarSamples::PerVertex(values))
      if values.len() == points.positions.len() =>
    {
      Some((FieldAssociation::Point, values.clone()))
    }
    (RenderGeometry::Lines(lines), ScalarSamples::PerVertex(values))
      if values.len() == lines.positions.len() =>
    {
      Some((FieldAssociation::Point, values.clone()))
    }
    (
      RenderGeometry::Triangles(triangles),
      ScalarSamples::PerVertex(values),
    ) if values.len() == triangles.positions.len() => {
      Some((FieldAssociation::Point, values.clone()))
    }
    (RenderGeometry::Triangles(triangles), ScalarSamples::PerCell(values)) => {
      map_triangle_cell_values(triangles, values)
        .map(|values| (FieldAssociation::Cell, values))
    }
    (RenderGeometry::Triangles(triangles), ScalarSamples::PerFace(values)) => {
      map_triangle_face_values(triangles, values)
        .map(|values| (FieldAssociation::Cell, values))
    }
    _ => None,
  }
}

fn map_point_cell_values(
  points: &PointCloud,
  values: &[f64],
) -> Option<Vec<f64>> {
  points
    .cell_ids
    .iter()
    .map(|id| id.and_then(|id| values.get(id.index()).copied()))
    .collect()
}

fn map_point_face_values(
  points: &PointCloud,
  values: &[f64],
) -> Option<Vec<f64>> {
  points
    .face_ids
    .iter()
    .map(|id| id.and_then(|id| values.get(id.index()).copied()))
    .collect()
}

fn map_triangle_cell_values(
  triangles: &TriangleMesh,
  values: &[f64],
) -> Option<Vec<f64>> {
  triangles
    .cell_ids
    .iter()
    .map(|id| id.and_then(|id| values.get(id.index()).copied()))
    .collect()
}

fn map_triangle_face_values(
  triangles: &TriangleMesh,
  values: &[f64],
) -> Option<Vec<f64>> {
  triangles
    .face_ids
    .iter()
    .map(|id| id.and_then(|id| values.get(id.index()).copied()))
    .collect()
}

fn flatten_positions(positions: &[[f32; 3]]) -> Vec<f64> {
  positions
    .iter()
    .flat_map(|position| position.iter().map(|&value| f64::from(value)))
    .collect()
}

fn source_cell_id_array(
  association: FieldAssociation,
  ids: impl IntoIterator<Item = Option<utility::domain::CellId>>,
) -> FieldArray {
  FieldArray {
    name: "source_cell_id".to_string(),
    association,
    components: 1,
    values: FieldValues::I64(
      ids
        .into_iter()
        .map(|id| id.map(|id| id.index() as i64).unwrap_or(-1))
        .collect(),
    ),
  }
}

fn source_face_id_array(
  association: FieldAssociation,
  ids: impl IntoIterator<Item = Option<utility::domain::FaceId>>,
) -> FieldArray {
  FieldArray {
    name: "source_face_id".to_string(),
    association,
    components: 1,
    values: FieldValues::I64(
      ids
        .into_iter()
        .map(|id| id.map(|id| id.index() as i64).unwrap_or(-1))
        .collect(),
    ),
  }
}

fn colour_array(
  association: FieldAssociation,
  colours: impl IntoIterator<Item = [f32; 4]>,
) -> FieldArray {
  FieldArray {
    name: "rgba".to_string(),
    association,
    components: 4,
    values: FieldValues::F64(
      colours
        .into_iter()
        .flat_map(|colour| colour.map(f64::from))
        .collect(),
    ),
  }
}

fn render_mesh_filename(
  frame: &RenderFrame,
  world: &RenderWorld,
  mesh_index: usize,
  mesh: &RenderMesh,
) -> String {
  format!(
    "frame_{:06}_world_{}_mesh_{:03}_{}.vtu",
    frame.frame,
    world.id.0,
    mesh_index,
    sanitize_filename(&mesh.label)
  )
}

fn sanitize_filename(label: &str) -> String {
  let mut out = String::with_capacity(label.len());
  for ch in label.chars() {
    if ch.is_ascii_alphanumeric() {
      out.push(ch.to_ascii_lowercase());
    } else if ch == '_' || ch == '-' {
      out.push(ch);
    } else if !out.ends_with('_') {
      out.push('_');
    }
  }
  out.trim_matches('_').to_string()
}

#[cfg(test)]
mod tests {
  use utility::domain::{MeshKey, WorldId};

  use crate::ir::{
    MeshRepresentation, MeshSource, PointCloud, RenderGeometry, RenderMesh,
    RenderMeshId, RenderWorld, Rgba,
  };

  use super::*;

  #[test]
  fn point_cloud_exports_as_vertex_cells() {
    let mesh = RenderMesh::new(
      RenderMeshId {
        world: WorldId(0),
        mesh: MeshKey::SURFACE,
        representation: MeshRepresentation::Cells,
      },
      "surface points",
      MeshSource::TesseraMesh(MeshKey::SURFACE),
      RenderGeometry::Points(PointCloud {
        positions: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        colours: vec![Rgba::WHITE, Rgba::BLUE],
        cell_ids: vec![Some(0usize.into()), Some(1usize.into())],
        face_ids: vec![None, None],
      }),
    );
    let mut world = RenderWorld::new(WorldId(0));
    world.meshes = vec![mesh.clone()];

    let dataset = render_mesh_dataset(&world, &mesh).unwrap();

    assert_eq!(dataset.mesh.point_count(), 2);
    assert_eq!(dataset.mesh.cell_count(), 2);
    assert_eq!(dataset.mesh.cell_types, vec![VTK_VERTEX, VTK_VERTEX]);
    assert!(dataset.find_array("source_cell_id").is_some());
    assert!(dataset.find_array("rgba").is_some());
  }
}
