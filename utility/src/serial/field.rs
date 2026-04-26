// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use crate::{
  error::{AetherError, AetherResult, ErrorDomain},
  serial::serialize::{Serialize, Serializer},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldAssociation {
  Point,
  Cell,
}

#[derive(Clone, Debug, PartialEq)]
pub enum FieldValues {
  F64(Vec<f64>),
  U64(Vec<u64>),
  U32(Vec<u32>),
  I64(Vec<i64>),
  U8(Vec<u8>),
}

impl FieldValues {
  pub fn len(&self) -> usize {
    match self {
      FieldValues::F64(values) => values.len(),
      FieldValues::U64(values) => values.len(),
      FieldValues::U32(values) => values.len(),
      FieldValues::I64(values) => values.len(),
      FieldValues::U8(values) => values.len(),
    }
  }

  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }

  pub fn vtk_type_name(&self) -> &'static str {
    match self {
      FieldValues::F64(_) => "double",
      FieldValues::U64(_) => "unsigned_long",
      FieldValues::U32(_) => "unsigned_int",
      FieldValues::I64(_) => "long",
      FieldValues::U8(_) => "unsigned_char",
    }
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FieldArray {
  pub name: String,
  pub association: FieldAssociation,
  pub components: usize,
  pub values: FieldValues,
}

impl FieldArray {
  pub fn tuple_count(&self) -> usize {
    if self.components == 0 {
      0
    } else {
      self.values.len() / self.components
    }
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct UnstructuredMesh {
  /// Flattened point coordinates in component-major tuples.
  /// For VTK output we support 2D/3D tuples (2D is padded to z=0).
  pub points: Vec<f64>,
  pub point_components: usize,
  pub connectivity: Vec<u64>,
  pub offsets: Vec<u64>,
  pub cell_types: Vec<u8>,
}

impl UnstructuredMesh {
  pub fn point_count(&self) -> usize {
    if self.point_components == 0 {
      0
    } else {
      self.points.len() / self.point_components
    }
  }

  pub fn cell_count(&self) -> usize {
    self.offsets.len()
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FieldDataset {
  pub title: String,
  pub mesh: UnstructuredMesh,
  pub arrays: Vec<FieldArray>,
}

impl FieldDataset {
  pub fn validate(&self) -> AetherResult<()> {
    if self.mesh.point_components == 0 {
      return Err(
        AetherError::new(ErrorKind::InvalidPointComponents)
          .context("point_components must be greater than zero"),
      );
    }

    if !self
      .mesh
      .points
      .len()
      .is_multiple_of(self.mesh.point_components)
    {
      return Err(AetherError::new(ErrorKind::InvalidPointBuffer).context(
        format!(
          "points length {} is not divisible by point_components {}",
          self.mesh.points.len(),
          self.mesh.point_components
        ),
      ));
    }

    if self.mesh.cell_types.len() != self.mesh.offsets.len() {
      return Err(AetherError::new(ErrorKind::InvalidCellTypes).context(
        format!(
          "cell_types length {} does not match offsets length {}",
          self.mesh.cell_types.len(),
          self.mesh.offsets.len()
        ),
      ));
    }

    let mut previous = 0_u64;
    for (i, offset) in self.mesh.offsets.iter().enumerate() {
      if *offset < previous {
        return Err(AetherError::new(ErrorKind::InvalidConnectivity).context(
          format!(
            "offset at cell {} is {} but previous offset is {}",
            i, offset, previous
          ),
        ));
      }
      previous = *offset;
    }

    if self.mesh.offsets.last().copied().unwrap_or(0)
      != self.mesh.connectivity.len() as u64
    {
      return Err(AetherError::new(ErrorKind::InvalidConnectivity).context(
        format!(
          "last offset {} must equal connectivity length {}",
          self.mesh.offsets.last().copied().unwrap_or(0),
          self.mesh.connectivity.len(),
        ),
      ));
    }

    let point_count = self.mesh.point_count();
    let cell_count = self.mesh.cell_count();
    for array in &self.arrays {
      if array.components == 0 {
        return Err(
          AetherError::new(ErrorKind::InvalidArray)
            .context(format!("array '{}' has zero components", array.name)),
        );
      }

      if array.values.len() % array.components != 0 {
        return Err(AetherError::new(ErrorKind::InvalidArray).context(
          format!(
            "array '{}' length {} is not divisible by components {}",
            array.name,
            array.values.len(),
            array.components
          ),
        ));
      }

      let tuples = array.tuple_count();
      match array.association {
        FieldAssociation::Point if tuples != point_count => {
          return Err(AetherError::new(ErrorKind::InvalidArray).context(
            format!(
              "point array '{}' has {} tuples, expected {}",
              array.name, tuples, point_count
            ),
          ));
        }
        FieldAssociation::Cell if tuples != cell_count => {
          return Err(AetherError::new(ErrorKind::InvalidArray).context(
            format!(
              "cell array '{}' has {} tuples, expected {}",
              array.name, tuples, cell_count
            ),
          ));
        }
        _ => {}
      }
    }

    Ok(())
  }

  pub fn find_array(&self, name: &str) -> Option<&FieldArray> {
    self.arrays.iter().find(|array| array.name == name)
  }

  pub fn validate_partition_debug_arrays(&self) -> AetherResult<()> {
    let partition_id = self.find_array(partition_debug::PARTITION_ID);
    if partition_id.is_none() {
      return Ok(());
    }

    let required = [
      partition_debug::PARTITION_ID,
      partition_debug::GLOBAL_CELL_ID,
      partition_debug::LOCAL_CELL_ID,
      partition_debug::IS_GHOST,
      partition_debug::GHOST_SOURCE_PARTITION,
      partition_debug::GHOST_SOURCE_LOCAL_CELL,
    ];

    for name in required {
      let field = self.find_array(name).ok_or_else(|| {
        AetherError::new(ErrorKind::InvalidPartitionDebug)
          .context(format!("partition debug array '{}' is missing", name))
      })?;

      if field.association != FieldAssociation::Cell {
        return Err(
          AetherError::new(ErrorKind::InvalidPartitionDebug).context(format!(
            "partition debug array '{}' must be cell-associated",
            name
          )),
        );
      }

      if field.components != 1 {
        return Err(
          AetherError::new(ErrorKind::InvalidPartitionDebug).context(format!(
            "partition debug array '{}' must have one component",
            name
          )),
        );
      }
    }

    Ok(())
  }
}

pub mod partition_debug {
  pub const PARTITION_ID: &str = "partition_id";
  pub const GLOBAL_CELL_ID: &str = "global_cell_id";
  pub const LOCAL_CELL_ID: &str = "local_cell_id";
  pub const IS_GHOST: &str = "is_ghost";
  pub const GHOST_SOURCE_PARTITION: &str = "ghost_source_partition";
  pub const GHOST_SOURCE_LOCAL_CELL: &str = "ghost_source_local_cell";
}

pub trait FieldDatasetWriter {
  type Error: std::fmt::Display;

  fn format_name(&self) -> &'static str;
  fn write_dataset(
    &mut self,
    dataset: &FieldDataset,
  ) -> Result<(), Self::Error>;
}

pub trait FieldDatasetReader {
  type Error: std::fmt::Display;

  fn format_name(&self) -> &'static str;
  fn read_dataset(&mut self) -> Result<FieldDataset, Self::Error>;
}

/// Marker trait for serializers that can write field datasets to sinks while
/// still supporting the generic `Serialize` trait for metadata/manifests.
pub trait FieldSinkSerializer: FieldDatasetWriter {
  fn write_manifest<T: Serialize, S: Serializer>(
    &mut self,
    serializer: &mut S,
    value: &T,
  ) -> Result<(), S::Error> {
    value.serialize(serializer)
  }
}

impl<T: FieldDatasetWriter> FieldSinkSerializer for T {}

pub enum ErrorKind {
  InvalidPointComponents,
  InvalidPointBuffer,
  InvalidConnectivity,
  InvalidCellTypes,
  InvalidArray,
  InvalidPartitionDebug,
}

impl ErrorDomain for ErrorKind {
  fn domain(&self) -> &str {
    "field serial"
  }
}

impl std::fmt::Display for ErrorKind {
  fn fmt(
    &self,
    f: &mut std::fmt::Formatter<'_>,
  ) -> Result<(), std::fmt::Error> {
    let message = match self {
      ErrorKind::InvalidPointComponents => {
        "field dataset contains invalid point component metadata"
      }
      ErrorKind::InvalidPointBuffer => {
        "field dataset contains an invalid point coordinate buffer"
      }
      ErrorKind::InvalidConnectivity => {
        "field dataset contains invalid connectivity/offset data"
      }
      ErrorKind::InvalidCellTypes => {
        "field dataset contains invalid cell type metadata"
      }
      ErrorKind::InvalidArray => {
        "field dataset contains an invalid field array"
      }
      ErrorKind::InvalidPartitionDebug => {
        "field dataset contains invalid partition debug arrays"
      }
    };

    write!(f, "{}", message)
  }
}
