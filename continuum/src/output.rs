use std::{
  collections::HashMap,
  fs::{self, File},
  path::{Path, PathBuf},
};

use crate::{
  field::FieldStorage,
  geometry::{CellGeometry, CellMetrics, FaceGeometry, FaceId, Point},
  mesh::StructuredBlock,
  model::ConservationLaw,
  partition::{Decomposition, PartitionMesh},
};

use utility::{
  error::{AetherError, AetherResult, ErrorDomain},
  serial::{
    field::{
      FieldArray, FieldAssociation, FieldDataset, FieldDatasetWriter,
      FieldValues, UnstructuredMesh, partition_debug,
    },
    vtk::{PvtuSchema, XmlPvtuWriter, XmlVtuWriter},
  },
};

/// Defines how a conservation law maps its state vectors into named export
/// arrays. This keeps format writers generic (VTK, HDF5, etc.).
pub trait LawFieldSchema<const D: usize, const N: usize>:
  ConservationLaw<D, N>
{
  fn conserved_field_names(&self) -> [&'static str; N];

  fn derived_field_names(&self) -> &'static [&'static str] {
    &[]
  }

  fn write_derived_fields(
    &self,
    _state: &[f64; N],
    _centroid: &Point<D>,
    _metrics: &CellMetrics<D>,
    _out: &mut [f64],
  ) {
  }
}

/// Builds cell-centered field arrays from a state field.
pub fn build_cell_state_arrays<const D: usize, const N: usize, L, S, M>(
  law: &L,
  state: &S,
  mesh: &M,
) -> Vec<FieldArray>
where
  L: LawFieldSchema<D, N>,
  S: FieldStorage<N>,
  M: CellGeometry<D>,
{
  let cell_count = mesh.cell_count();

  let mut arrays = Vec::new();
  let names = law.conserved_field_names();
  for (component, name) in names.iter().enumerate() {
    let mut values = vec![0.0_f64; cell_count];
    for (i, value) in values.iter_mut().enumerate() {
      let mut cell_state = [0.0; N];
      state.state_into(i.into(), &mut cell_state);
      *value = cell_state[component];
    }

    arrays.push(FieldArray {
      name: (*name).to_string(),
      association: FieldAssociation::Cell,
      components: 1,
      values: FieldValues::F64(values),
    });
  }

  let derived_names = law.derived_field_names();
  if !derived_names.is_empty() {
    let mut derived_values = vec![0.0_f64; derived_names.len() * cell_count];

    for i in 0..cell_count {
      let mut cell_state = [0.0; N];
      state.state_into(i.into(), &mut cell_state);

      let start = i * derived_names.len();
      let end = start + derived_names.len();
      law.write_derived_fields(
        &cell_state,
        mesh.cell_centroid(i.into()),
        mesh.cell_metrics(i.into()),
        &mut derived_values[start..end],
      );
    }

    for (component, name) in derived_names.iter().enumerate() {
      let mut values = vec![0.0_f64; cell_count];
      for i in 0..cell_count {
        values[i] = derived_values[i * derived_names.len() + component];
      }

      arrays.push(FieldArray {
        name: (*name).to_string(),
        association: FieldAssociation::Cell,
        components: 1,
        values: FieldValues::F64(values),
      });
    }
  }

  arrays
}

/// Converts a mesh into a dataset shell (geometry + topology, no field arrays).
pub trait MeshFieldDatasetBuilder<const D: usize> {
  fn build_dataset_shell(
    &self,
    title: impl Into<String>,
  ) -> AetherResult<FieldDataset>;
}

impl<const D: usize> MeshFieldDatasetBuilder<D> for StructuredBlock<D> {
  fn build_dataset_shell(
    &self,
    title: impl Into<String>,
  ) -> AetherResult<FieldDataset> {
    let (axis_min, axis_max) = axis_bounds_from_faces(self)?;
    let mesh =
      build_structured_unstructured_mesh(self.dims(), &axis_min, &axis_max)?;

    Ok(FieldDataset {
      title: title.into(),
      mesh,
      arrays: Vec::new(),
    })
  }
}

impl<const D: usize> MeshFieldDatasetBuilder<D>
  for PartitionMesh<D, StructuredBlock<D>>
{
  fn build_dataset_shell(
    &self,
    title: impl Into<String>,
  ) -> AetherResult<FieldDataset> {
    let global_mesh = self.mesh();
    let (axis_min, axis_max) = axis_bounds_from_faces(global_mesh)?;
    let mesh = build_partition_unstructured_mesh(
      self,
      global_mesh.dims(),
      &axis_min,
      &axis_max,
    )?;

    Ok(FieldDataset {
      title: title.into(),
      mesh,
      arrays: Vec::new(),
    })
  }
}

pub struct PartitionDebugColumns {
  pub partition_id: Vec<u32>,
  pub global_cell_id: Vec<u64>,
  pub local_cell_id: Vec<u32>,
  pub is_ghost: Vec<u8>,
  pub ghost_source_partition: Vec<i64>,
  pub ghost_source_local_cell: Vec<i64>,
}

pub fn build_partition_debug_arrays(
  columns: PartitionDebugColumns,
) -> AetherResult<Vec<FieldArray>> {
  let cell_count = columns.partition_id.len();
  let lengths = [
    columns.global_cell_id.len(),
    columns.local_cell_id.len(),
    columns.is_ghost.len(),
    columns.ghost_source_partition.len(),
    columns.ghost_source_local_cell.len(),
  ];

  if lengths.iter().any(|&len| len != cell_count) {
    return Err(AetherError::new(ErrorKind::InvalidPartitionColumns).context(
      format!(
        "partition debug columns must have identical lengths, got base={} others={:?}",
        cell_count, lengths
      ),
    ));
  }

  Ok(vec![
    FieldArray {
      name: partition_debug::PARTITION_ID.to_string(),
      association: FieldAssociation::Cell,
      components: 1,
      values: FieldValues::U32(columns.partition_id),
    },
    FieldArray {
      name: partition_debug::GLOBAL_CELL_ID.to_string(),
      association: FieldAssociation::Cell,
      components: 1,
      values: FieldValues::U64(columns.global_cell_id),
    },
    FieldArray {
      name: partition_debug::LOCAL_CELL_ID.to_string(),
      association: FieldAssociation::Cell,
      components: 1,
      values: FieldValues::U32(columns.local_cell_id),
    },
    FieldArray {
      name: partition_debug::IS_GHOST.to_string(),
      association: FieldAssociation::Cell,
      components: 1,
      values: FieldValues::U8(columns.is_ghost),
    },
    FieldArray {
      name: partition_debug::GHOST_SOURCE_PARTITION.to_string(),
      association: FieldAssociation::Cell,
      components: 1,
      values: FieldValues::I64(columns.ghost_source_partition),
    },
    FieldArray {
      name: partition_debug::GHOST_SOURCE_LOCAL_CELL.to_string(),
      association: FieldAssociation::Cell,
      components: 1,
      values: FieldValues::I64(columns.ghost_source_local_cell),
    },
  ])
}

pub fn write_partitioned_vtu<const D: usize, const N: usize, L, S>(
  decomposition: &Decomposition<D, StructuredBlock<D>>,
  states: &[S],
  law: &L,
  output_dir: impl AsRef<Path>,
  base_name: &str,
) -> AetherResult<PathBuf>
where
  L: LawFieldSchema<D, N>,
  S: FieldStorage<N>,
{
  let part_count = decomposition.partitions.len();
  if part_count == 0 {
    return Err(
      AetherError::new(ErrorKind::EmptyDecomposition)
        .context("cannot write partitioned output for an empty decomposition"),
    );
  }

  if states.len() != part_count {
    return Err(
      AetherError::new(ErrorKind::PartitionStateCountMismatch).context(
        format!(
          "state count {} does not match partition count {}",
          states.len(),
          part_count
        ),
      ),
    );
  }

  let output_dir = output_dir.as_ref();
  fs::create_dir_all(output_dir)?;

  let mut schema: Option<PvtuSchema> = None;
  let mut piece_sources = Vec::with_capacity(part_count);

  for (partition_index, (partition, state)) in decomposition
    .partitions
    .iter()
    .zip(states.iter())
    .enumerate()
  {
    if state.len() != partition.cell_count() {
      return Err(AetherError::new(ErrorKind::StateLengthMismatch).context(
        format!(
          "partition {} state length {} does not match mesh cell count {}",
          partition_index,
          state.len(),
          partition.cell_count()
        ),
      ));
    }

    let mut dataset = partition
      .build_dataset_shell(format!("{base_name} piece {partition_index}"))?;
    dataset
      .arrays
      .extend(build_cell_state_arrays(law, state, partition));
    let debug_columns =
      build_partition_debug_columns(partition_index, partition)?;
    dataset
      .arrays
      .extend(build_partition_debug_arrays(debug_columns)?);

    if schema.is_none() {
      schema = Some(PvtuSchema::from_dataset(&dataset));
    }

    let piece_name = format!("{base_name}_piece_{partition_index:04}.vtu");
    let piece_path = output_dir.join(&piece_name);
    let piece_file = File::create(&piece_path)?;
    let mut piece_writer = XmlVtuWriter::new(piece_file);
    piece_writer.write_dataset(&dataset)?;
    piece_sources.push(piece_name);
  }

  let manifest_path = output_dir.join(format!("{base_name}.pvtu"));
  let manifest_file = File::create(&manifest_path)?;
  let mut manifest_writer = XmlPvtuWriter::new(manifest_file);
  manifest_writer.write_manifest(schema.as_ref().unwrap(), &piece_sources)?;

  Ok(manifest_path)
}

fn axis_bounds_from_faces<const D: usize, M>(
  mesh: &M,
) -> AetherResult<([f64; D], [f64; D])>
where
  M: FaceGeometry<D>,
{
  if mesh.face_count() == 0 {
    return Err(
      AetherError::new(ErrorKind::InvalidMeshBounds)
        .context("cannot infer axis bounds from an empty face set"),
    );
  }

  let mut axis_min = [f64::INFINITY; D];
  let mut axis_max = [f64::NEG_INFINITY; D];

  for face in 0..mesh.face_count() {
    let centroid = mesh.face_centroid(FaceId::from(face));
    for axis in 0..D {
      axis_min[axis] = axis_min[axis].min(centroid[axis]);
      axis_max[axis] = axis_max[axis].max(centroid[axis]);
    }
  }

  for axis in 0..D {
    if !axis_min[axis].is_finite()
      || !axis_max[axis].is_finite()
      || axis_min[axis] > axis_max[axis]
    {
      return Err(AetherError::new(ErrorKind::InvalidMeshBounds).context(
        format!(
          "failed to infer axis bounds for axis {} (min={}, max={})",
          axis, axis_min[axis], axis_max[axis]
        ),
      ));
    }
  }

  Ok((axis_min, axis_max))
}

fn build_structured_unstructured_mesh<const D: usize>(
  dims: &[usize; D],
  axis_min: &[f64; D],
  axis_max: &[f64; D],
) -> AetherResult<UnstructuredMesh> {
  let cell_type = vtk_cell_type_for_dimension::<D>()?;
  let axis_coords = axis_coordinates(dims, axis_min, axis_max)?;
  let point_dims = std::array::from_fn(|axis| dims[axis] + 1);

  let point_count = product(&point_dims);
  let mut points = Vec::with_capacity(point_count * D);
  for point_flat in 0..point_count {
    let point_ijk = unravel_flat_index(&point_dims, point_flat);
    for axis in 0..D {
      points.push(axis_coords[axis][point_ijk[axis]]);
    }
  }

  let cell_count = product(dims);
  let mut connectivity = Vec::new();
  let mut offsets = Vec::with_capacity(cell_count);
  let mut cell_types = Vec::with_capacity(cell_count);

  for cell_flat in 0..cell_count {
    let cell_ijk = unravel_flat_index(dims, cell_flat);
    for vertex_ijk in cell_vertex_indices(&cell_ijk)? {
      let point_flat = flatten_index(&point_dims, &vertex_ijk);
      connectivity.push(usize_to_u64(point_flat, "point index")?);
    }

    offsets.push(usize_to_u64(connectivity.len(), "connectivity offset")?);
    cell_types.push(cell_type);
  }

  Ok(UnstructuredMesh {
    points,
    point_components: D,
    connectivity,
    offsets,
    cell_types,
  })
}

fn build_partition_unstructured_mesh<const D: usize>(
  partition: &PartitionMesh<D, StructuredBlock<D>>,
  global_dims: &[usize; D],
  axis_min: &[f64; D],
  axis_max: &[f64; D],
) -> AetherResult<UnstructuredMesh> {
  let cell_type = vtk_cell_type_for_dimension::<D>()?;
  let axis_coords = axis_coordinates(global_dims, axis_min, axis_max)?;
  let point_dims = std::array::from_fn(|axis| global_dims[axis] + 1);

  let mut point_map: HashMap<usize, u64> = HashMap::new();
  let mut points = Vec::new();
  let mut connectivity = Vec::new();
  let mut offsets = Vec::with_capacity(partition.local_cell_count());
  let mut cell_types = Vec::with_capacity(partition.local_cell_count());

  for global_cell in partition.local_to_global_cells() {
    let global_ijk = unravel_flat_index(global_dims, global_cell.index());
    for vertex_ijk in cell_vertex_indices(&global_ijk)? {
      let global_point_flat = flatten_index(&point_dims, &vertex_ijk);
      let local_point =
        if let Some(&existing) = point_map.get(&global_point_flat) {
          existing
        } else {
          let new_index = points.len() / D;
          for axis in 0..D {
            points.push(axis_coords[axis][vertex_ijk[axis]]);
          }
          let as_u64 = usize_to_u64(new_index, "partition-local point index")?;
          point_map.insert(global_point_flat, as_u64);
          as_u64
        };
      connectivity.push(local_point);
    }

    offsets.push(usize_to_u64(connectivity.len(), "connectivity offset")?);
    cell_types.push(cell_type);
  }

  Ok(UnstructuredMesh {
    points,
    point_components: D,
    connectivity,
    offsets,
    cell_types,
  })
}

fn build_partition_debug_columns<const D: usize>(
  partition_index: usize,
  partition: &PartitionMesh<D, StructuredBlock<D>>,
) -> AetherResult<PartitionDebugColumns> {
  let cell_count = partition.local_cell_count();
  let partition_id =
    vec![usize_to_u32(partition_index, "partition id")?; cell_count];

  let mut global_cell_id = Vec::with_capacity(cell_count);
  let mut local_cell_id = Vec::with_capacity(cell_count);
  let mut is_ghost = vec![0_u8; cell_count];
  let mut ghost_source_partition = vec![-1_i64; cell_count];
  let mut ghost_source_local_cell = vec![-1_i64; cell_count];

  for (local, global) in partition.local_to_global_cells().iter().enumerate() {
    global_cell_id.push(usize_to_u64(global.index(), "global cell id")?);
    local_cell_id.push(usize_to_u32(local, "local cell id")?);
    if local >= partition.num_owned() {
      is_ghost[local] = 1;
    }
  }

  for ghost in partition.ghost_cells() {
    let local = ghost.local_cell.index();
    if local >= cell_count {
      return Err(
        AetherError::new(ErrorKind::InvalidPartitionColumns).context(format!(
          "ghost descriptor local cell {} out of range 0..{}",
          local, cell_count
        )),
      );
    }

    is_ghost[local] = 1;
    ghost_source_partition[local] =
      usize_to_i64(ghost.source_partition, "ghost source partition")?;
    ghost_source_local_cell[local] =
      usize_to_i64(ghost.source_local_cell.index(), "ghost source local cell")?;
  }

  Ok(PartitionDebugColumns {
    partition_id,
    global_cell_id,
    local_cell_id,
    is_ghost,
    ghost_source_partition,
    ghost_source_local_cell,
  })
}

fn axis_coordinates<const D: usize>(
  dims: &[usize; D],
  axis_min: &[f64; D],
  axis_max: &[f64; D],
) -> AetherResult<Vec<Vec<f64>>> {
  let mut coords = Vec::with_capacity(D);
  for axis in 0..D {
    if dims[axis] == 0 {
      return Err(
        AetherError::new(ErrorKind::UnsupportedDimension)
          .context(format!("axis {} has zero cells", axis)),
      );
    }

    let spacing = (axis_max[axis] - axis_min[axis]) / dims[axis] as f64;
    let mut axis_coords = Vec::with_capacity(dims[axis] + 1);
    for i in 0..=dims[axis] {
      axis_coords.push(axis_min[axis] + i as f64 * spacing);
    }
    coords.push(axis_coords);
  }
  Ok(coords)
}

fn cell_vertex_indices<const D: usize>(
  ijk: &[usize; D],
) -> AetherResult<Vec<[usize; D]>> {
  match D {
    1 => {
      let mut a = [0; D];
      a[0] = ijk[0];
      let mut b = [0; D];
      b[0] = ijk[0] + 1;
      Ok(vec![a, b])
    }
    2 => {
      let mut v0 = [0; D];
      v0[0] = ijk[0];
      v0[1] = ijk[1];
      let mut v1 = [0; D];
      v1[0] = ijk[0] + 1;
      v1[1] = ijk[1];
      let mut v2 = [0; D];
      v2[0] = ijk[0] + 1;
      v2[1] = ijk[1] + 1;
      let mut v3 = [0; D];
      v3[0] = ijk[0];
      v3[1] = ijk[1] + 1;
      Ok(vec![v0, v1, v2, v3])
    }
    3 => {
      let mut v0 = [0; D];
      v0[0] = ijk[0];
      v0[1] = ijk[1];
      v0[2] = ijk[2];
      let mut v1 = [0; D];
      v1[0] = ijk[0] + 1;
      v1[1] = ijk[1];
      v1[2] = ijk[2];
      let mut v2 = [0; D];
      v2[0] = ijk[0] + 1;
      v2[1] = ijk[1] + 1;
      v2[2] = ijk[2];
      let mut v3 = [0; D];
      v3[0] = ijk[0];
      v3[1] = ijk[1] + 1;
      v3[2] = ijk[2];
      let mut v4 = [0; D];
      v4[0] = ijk[0];
      v4[1] = ijk[1];
      v4[2] = ijk[2] + 1;
      let mut v5 = [0; D];
      v5[0] = ijk[0] + 1;
      v5[1] = ijk[1];
      v5[2] = ijk[2] + 1;
      let mut v6 = [0; D];
      v6[0] = ijk[0] + 1;
      v6[1] = ijk[1] + 1;
      v6[2] = ijk[2] + 1;
      let mut v7 = [0; D];
      v7[0] = ijk[0];
      v7[1] = ijk[1] + 1;
      v7[2] = ijk[2] + 1;
      Ok(vec![v0, v1, v2, v3, v4, v5, v6, v7])
    }
    _ => Err(AetherError::new(ErrorKind::UnsupportedDimension).context(
      format!("structured output supports dimensions 1..=3, got {}", D),
    )),
  }
}

fn vtk_cell_type_for_dimension<const D: usize>() -> AetherResult<u8> {
  match D {
    1 => Ok(3),  // VTK_LINE
    2 => Ok(9),  // VTK_QUAD
    3 => Ok(12), // VTK_HEXAHEDRON
    _ => Err(AetherError::new(ErrorKind::UnsupportedDimension).context(
      format!("structured output supports dimensions 1..=3, got {}", D),
    )),
  }
}

fn unravel_flat_index<const D: usize>(
  dims: &[usize; D],
  mut flat: usize,
) -> [usize; D] {
  let mut ijk = [0; D];
  for axis in 0..D {
    ijk[axis] = flat % dims[axis];
    flat /= dims[axis];
  }
  ijk
}

fn flatten_index<const D: usize>(dims: &[usize; D], ijk: &[usize; D]) -> usize {
  let mut index = 0_usize;
  let mut stride = 1_usize;
  for axis in 0..D {
    index += ijk[axis] * stride;
    stride *= dims[axis];
  }
  index
}

fn product<const D: usize>(dims: &[usize; D]) -> usize {
  dims.iter().product()
}

fn usize_to_u32(value: usize, label: &str) -> AetherResult<u32> {
  u32::try_from(value).map_err(|_| {
    AetherError::new(ErrorKind::IntegerConversion)
      .context(format!("{} {} does not fit into u32", label, value))
  })
}

fn usize_to_u64(value: usize, label: &str) -> AetherResult<u64> {
  u64::try_from(value).map_err(|_| {
    AetherError::new(ErrorKind::IntegerConversion)
      .context(format!("{} {} does not fit into u64", label, value))
  })
}

fn usize_to_i64(value: usize, label: &str) -> AetherResult<i64> {
  i64::try_from(value).map_err(|_| {
    AetherError::new(ErrorKind::IntegerConversion)
      .context(format!("{} {} does not fit into i64", label, value))
  })
}

pub enum ErrorKind {
  InvalidPartitionColumns,
  UnsupportedDimension,
  PartitionStateCountMismatch,
  StateLengthMismatch,
  EmptyDecomposition,
  InvalidMeshBounds,
  IntegerConversion,
}

impl ErrorDomain for ErrorKind {
  fn domain(&self) -> &str {
    "continuum output"
  }
}

impl std::fmt::Display for ErrorKind {
  fn fmt(
    &self,
    f: &mut std::fmt::Formatter<'_>,
  ) -> Result<(), std::fmt::Error> {
    match self {
      ErrorKind::InvalidPartitionColumns => {
        write!(f, "invalid partition debug column lengths")
      }
      ErrorKind::UnsupportedDimension => {
        write!(f, "unsupported output mesh dimension")
      }
      ErrorKind::PartitionStateCountMismatch => {
        write!(f, "partition/state count mismatch")
      }
      ErrorKind::StateLengthMismatch => {
        write!(f, "partition state length mismatch")
      }
      ErrorKind::EmptyDecomposition => {
        write!(f, "cannot export an empty decomposition")
      }
      ErrorKind::InvalidMeshBounds => {
        write!(f, "failed to infer mesh axis bounds")
      }
      ErrorKind::IntegerConversion => write!(f, "integer conversion overflow"),
    }
  }
}

#[cfg(test)]
mod tests {
  use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
  };

  use crate::{
    field::AosField,
    geometry::{CellGeometry, IdentityMap},
    mesh::StructuredBlock,
    model::{Euler2D, RusanovFlux},
    partition::decompose_structured,
    solver::{FvmSolver, SolverConfig, TimeIntegration},
  };

  use super::{
    MeshFieldDatasetBuilder, build_cell_state_arrays, write_partitioned_vtu,
  };

  #[test]
  fn builds_conserved_and_derived_cell_arrays() {
    let mesh = StructuredBlock::uniform(
      [0.0, 0.0].into(),
      [1.0, 1.0],
      [2, 1],
      Box::new(IdentityMap::<2>),
    );

    let state =
      AosField::<4>::from_fn(mesh.cell_count(), |_| [1.0, 0.0, 0.0, 2.5]);

    let law = FvmSolver::new(
      SolverConfig::new(0.5, 1e-4, TimeIntegration::ForwardEuler),
      Euler2D::new(1.4),
      RusanovFlux,
    );

    let arrays = build_cell_state_arrays(law.law(), &state, &mesh);
    assert!(arrays.iter().any(|array| array.name == "rho"));
    assert!(arrays.iter().any(|array| array.name == "pressure"));
  }

  #[test]
  fn builds_structured_block_dataset_shell() {
    let mesh = StructuredBlock::uniform(
      [0.0, 0.0].into(),
      [2.0, 1.0],
      [2, 1],
      Box::new(IdentityMap::<2>),
    );

    let dataset = mesh.build_dataset_shell("structured").unwrap();
    assert_eq!(dataset.mesh.point_count(), 6);
    assert_eq!(dataset.mesh.cell_count(), 2);
    assert_eq!(dataset.mesh.connectivity.len(), 8);
    assert_eq!(dataset.mesh.cell_types, vec![9, 9]);
  }

  #[test]
  fn writes_partitioned_vtu_and_pvtu() {
    let dims = [4, 1];
    let mesh = Arc::new(StructuredBlock::uniform(
      [0.0, 0.0].into(),
      [1.0, 0.1],
      dims,
      Box::new(IdentityMap::<2>),
    ));
    let decomp = decompose_structured(Arc::clone(&mesh), dims, 2, 1);

    let states: Vec<AosField<4>> = decomp
      .partitions
      .iter()
      .map(|partition| {
        AosField::from_fn(partition.cell_count(), |local| {
          let global = partition.local_to_global(local);
          let x = mesh.cell_centroid(global)[0];
          [1.0 + x, 0.0, 0.0, 2.5]
        })
      })
      .collect();

    let unique = SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .unwrap()
      .as_nanos();
    let out_dir =
      std::env::temp_dir().join(format!("aether_vtk_export_{unique}"));

    let manifest = write_partitioned_vtu(
      &decomp,
      &states,
      &Euler2D::new(1.4),
      &out_dir,
      "step0000",
    )
    .unwrap();

    let pvtu_text = std::fs::read_to_string(&manifest).unwrap();
    assert!(pvtu_text.contains("step0000_piece_0000.vtu"));
    assert!(pvtu_text.contains("step0000_piece_0001.vtu"));

    let first_piece = out_dir.join("step0000_piece_0000.vtu");
    let piece_text = std::fs::read_to_string(first_piece).unwrap();
    assert!(piece_text.contains("vtkGhostType"));

    std::fs::remove_dir_all(out_dir).unwrap();
  }
}
