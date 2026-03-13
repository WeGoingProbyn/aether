use std::io::Write;

use crate::{
  error::{AetherError, ErrorDomain},
  serial::field::{
    FieldArray, FieldAssociation, FieldDataset, FieldDatasetWriter,
    FieldValues, partition_debug,
  },
};

pub struct LegacyVtkWriter<W: Write> {
  sink: W,
}

impl<W: Write> LegacyVtkWriter<W> {
  pub fn new(sink: W) -> LegacyVtkWriter<W> {
    LegacyVtkWriter { sink }
  }

  fn write_field_block(
    &mut self,
    association: FieldAssociation,
    dataset: &FieldDataset,
    synthetic_arrays: &[FieldArray],
  ) -> Result<(), AetherError> {
    let arrays = arrays_for_association(dataset, synthetic_arrays, association);
    if arrays.is_empty() {
      return Ok(());
    }

    let tuple_count = match association {
      FieldAssociation::Point => dataset.mesh.point_count(),
      FieldAssociation::Cell => dataset.mesh.cell_count(),
    };

    match association {
      FieldAssociation::Point => {
        writeln!(self.sink, "POINT_DATA {}", tuple_count)?;
      }
      FieldAssociation::Cell => {
        writeln!(self.sink, "CELL_DATA {}", tuple_count)?;
      }
    }

    writeln!(self.sink, "FIELD FieldData {}", arrays.len())?;
    for array in arrays {
      writeln!(
        self.sink,
        "{} {} {} {}",
        array.name,
        array.components,
        tuple_count,
        array.values.vtk_type_name()
      )?;

      match &array.values {
        FieldValues::F64(values) => {
          Self::write_chunked(&mut self.sink, values, array.components)?;
        }
        FieldValues::U64(values) => {
          Self::write_chunked(&mut self.sink, values, array.components)?;
        }
        FieldValues::U32(values) => {
          Self::write_chunked(&mut self.sink, values, array.components)?;
        }
        FieldValues::I64(values) => {
          Self::write_chunked(&mut self.sink, values, array.components)?;
        }
        FieldValues::U8(values) => {
          Self::write_chunked(&mut self.sink, values, array.components)?;
        }
      }
    }

    Ok(())
  }

  fn write_chunked<T: std::fmt::Display>(
    sink: &mut W,
    values: &[T],
    components: usize,
  ) -> Result<(), std::io::Error> {
    if components == 0 {
      return Ok(());
    }

    for tuple in values.chunks(components) {
      for (i, value) in tuple.iter().enumerate() {
        if i > 0 {
          write!(sink, " ")?;
        }
        write!(sink, "{}", value)?;
      }
      writeln!(sink)?;
    }

    Ok(())
  }
}

impl<W: Write> FieldDatasetWriter for LegacyVtkWriter<W> {
  type Error = AetherError;

  fn format_name(&self) -> &'static str {
    "vtk-legacy"
  }

  fn write_dataset(
    &mut self,
    dataset: &FieldDataset,
  ) -> Result<(), Self::Error> {
    dataset.validate()?;
    dataset.validate_partition_debug_arrays()?;
    validate_point_components(dataset)?;

    let mut synthetic_arrays = Vec::new();
    if let Some(vtk_ghost_type) = build_vtk_ghost_type_array(dataset) {
      synthetic_arrays.push(vtk_ghost_type);
    }

    writeln!(self.sink, "# vtk DataFile Version 3.0")?;
    writeln!(self.sink, "{}", dataset.title)?;
    writeln!(self.sink, "ASCII")?;
    writeln!(self.sink, "DATASET UNSTRUCTURED_GRID")?;

    let point_count = dataset.mesh.point_count();
    writeln!(self.sink, "POINTS {} double", point_count)?;
    for point in dataset.mesh.points.chunks(dataset.mesh.point_components) {
      let x = point.first().copied().unwrap_or(0.0);
      let y = point.get(1).copied().unwrap_or(0.0);
      let z = point.get(2).copied().unwrap_or(0.0);
      writeln!(self.sink, "{} {} {}", x, y, z)?;
    }

    let cell_count = dataset.mesh.cell_count();
    let legacy_size = dataset.mesh.connectivity.len() + cell_count;
    writeln!(self.sink, "CELLS {} {}", cell_count, legacy_size)?;

    let mut start = 0_usize;
    for &offset in &dataset.mesh.offsets {
      let end = usize::try_from(offset).map_err(|_| {
        AetherError::new(ErrorKind::CellOffsetOverflow)
          .context("cell offset does not fit in usize")
      })?;
      let cell = &dataset.mesh.connectivity[start..end];
      write!(self.sink, "{}", cell.len())?;
      for index in cell {
        write!(self.sink, " {}", index)?;
      }
      writeln!(self.sink)?;
      start = end;
    }

    writeln!(self.sink, "CELL_TYPES {}", dataset.mesh.cell_types.len())?;
    for cell_type in &dataset.mesh.cell_types {
      writeln!(self.sink, "{}", cell_type)?;
    }

    self.write_field_block(
      FieldAssociation::Point,
      dataset,
      &synthetic_arrays,
    )?;
    self.write_field_block(
      FieldAssociation::Cell,
      dataset,
      &synthetic_arrays,
    )?;

    Ok(())
  }
}

pub struct XmlVtuWriter<W: Write> {
  sink: W,
}

impl<W: Write> XmlVtuWriter<W> {
  pub fn new(sink: W) -> XmlVtuWriter<W> {
    XmlVtuWriter { sink }
  }

  fn write_field_data_array(
    &mut self,
    array: &FieldArray,
  ) -> Result<(), AetherError> {
    write!(
      self.sink,
      "        <DataArray type=\"{}\" Name=\"",
      vtk_xml_type_name(&array.values)
    )?;
    write_xml_attr_escaped(&mut self.sink, &array.name)?;
    writeln!(
      self.sink,
      "\" NumberOfComponents=\"{}\" format=\"ascii\">",
      array.components
    )?;

    match &array.values {
      FieldValues::F64(values) => {
        Self::write_tuples(&mut self.sink, values, array.components)?;
      }
      FieldValues::U64(values) => {
        Self::write_tuples(&mut self.sink, values, array.components)?;
      }
      FieldValues::U32(values) => {
        Self::write_tuples(&mut self.sink, values, array.components)?;
      }
      FieldValues::I64(values) => {
        Self::write_tuples(&mut self.sink, values, array.components)?;
      }
      FieldValues::U8(values) => {
        Self::write_tuples(&mut self.sink, values, array.components)?;
      }
    }

    writeln!(self.sink, "        </DataArray>")?;
    Ok(())
  }

  fn write_tuples<T: std::fmt::Display>(
    sink: &mut W,
    values: &[T],
    components: usize,
  ) -> Result<(), std::io::Error> {
    if components == 0 {
      return Ok(());
    }

    for tuple in values.chunks(components) {
      write!(sink, "          ")?;
      for (i, value) in tuple.iter().enumerate() {
        if i > 0 {
          write!(sink, " ")?;
        }
        write!(sink, "{}", value)?;
      }
      writeln!(sink)?;
    }

    Ok(())
  }

  fn write_flat<T: std::fmt::Display>(
    sink: &mut W,
    values: &[T],
    values_per_line: usize,
  ) -> Result<(), std::io::Error> {
    let width = values_per_line.max(1);
    for chunk in values.chunks(width) {
      write!(sink, "          ")?;
      for (i, value) in chunk.iter().enumerate() {
        if i > 0 {
          write!(sink, " ")?;
        }
        write!(sink, "{}", value)?;
      }
      writeln!(sink)?;
    }

    Ok(())
  }
}

impl<W: Write> FieldDatasetWriter for XmlVtuWriter<W> {
  type Error = AetherError;

  fn format_name(&self) -> &'static str {
    "vtk-xml-vtu"
  }

  fn write_dataset(
    &mut self,
    dataset: &FieldDataset,
  ) -> Result<(), Self::Error> {
    dataset.validate()?;
    dataset.validate_partition_debug_arrays()?;
    validate_point_components(dataset)?;

    let mut synthetic_arrays = Vec::new();
    if let Some(vtk_ghost_type) = build_vtk_ghost_type_array(dataset) {
      synthetic_arrays.push(vtk_ghost_type);
    }

    writeln!(self.sink, "<?xml version=\"1.0\"?>")?;
    writeln!(
      self.sink,
      "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">"
    )?;
    writeln!(self.sink, "  <UnstructuredGrid>")?;
    writeln!(
      self.sink,
      "    <Piece NumberOfPoints=\"{}\" NumberOfCells=\"{}\">",
      dataset.mesh.point_count(),
      dataset.mesh.cell_count()
    )?;

    writeln!(self.sink, "      <Points>")?;
    writeln!(
      self.sink,
      "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">"
    )?;
    for point in dataset.mesh.points.chunks(dataset.mesh.point_components) {
      let x = point.first().copied().unwrap_or(0.0);
      let y = point.get(1).copied().unwrap_or(0.0);
      let z = point.get(2).copied().unwrap_or(0.0);
      writeln!(self.sink, "          {} {} {}", x, y, z)?;
    }
    writeln!(self.sink, "        </DataArray>")?;
    writeln!(self.sink, "      </Points>")?;

    writeln!(self.sink, "      <Cells>")?;
    writeln!(
      self.sink,
      "        <DataArray type=\"UInt64\" Name=\"connectivity\" format=\"ascii\">"
    )?;
    Self::write_flat(&mut self.sink, &dataset.mesh.connectivity, 16)?;
    writeln!(self.sink, "        </DataArray>")?;

    writeln!(
      self.sink,
      "        <DataArray type=\"UInt64\" Name=\"offsets\" format=\"ascii\">"
    )?;
    Self::write_flat(&mut self.sink, &dataset.mesh.offsets, 16)?;
    writeln!(self.sink, "        </DataArray>")?;

    writeln!(
      self.sink,
      "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">"
    )?;
    Self::write_flat(&mut self.sink, &dataset.mesh.cell_types, 32)?;
    writeln!(self.sink, "        </DataArray>")?;
    writeln!(self.sink, "      </Cells>")?;

    let point_arrays = arrays_for_association(
      dataset,
      &synthetic_arrays,
      FieldAssociation::Point,
    );
    writeln!(self.sink, "      <PointData>")?;
    for array in point_arrays {
      self.write_field_data_array(array)?;
    }
    writeln!(self.sink, "      </PointData>")?;

    let cell_arrays = arrays_for_association(
      dataset,
      &synthetic_arrays,
      FieldAssociation::Cell,
    );
    writeln!(self.sink, "      <CellData>")?;
    for array in cell_arrays {
      self.write_field_data_array(array)?;
    }
    writeln!(self.sink, "      </CellData>")?;

    writeln!(self.sink, "    </Piece>")?;
    writeln!(self.sink, "  </UnstructuredGrid>")?;
    writeln!(self.sink, "</VTKFile>")?;
    Ok(())
  }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VtkArraySchema {
  pub name: String,
  pub components: usize,
  pub vtk_type: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PvtuSchema {
  pub point_arrays: Vec<VtkArraySchema>,
  pub cell_arrays: Vec<VtkArraySchema>,
}

impl PvtuSchema {
  pub fn from_dataset(dataset: &FieldDataset) -> PvtuSchema {
    let mut synthetic_arrays = Vec::new();
    if let Some(vtk_ghost_type) = build_vtk_ghost_type_array(dataset) {
      synthetic_arrays.push(vtk_ghost_type);
    }

    let point_arrays = arrays_for_association(
      dataset,
      &synthetic_arrays,
      FieldAssociation::Point,
    );
    let cell_arrays = arrays_for_association(
      dataset,
      &synthetic_arrays,
      FieldAssociation::Cell,
    );

    PvtuSchema {
      point_arrays: point_arrays
        .into_iter()
        .map(|array| VtkArraySchema {
          name: array.name.clone(),
          components: array.components,
          vtk_type: vtk_xml_type_name(&array.values),
        })
        .collect(),
      cell_arrays: cell_arrays
        .into_iter()
        .map(|array| VtkArraySchema {
          name: array.name.clone(),
          components: array.components,
          vtk_type: vtk_xml_type_name(&array.values),
        })
        .collect(),
    }
  }
}

pub struct XmlPvtuWriter<W: Write> {
  sink: W,
}

impl<W: Write> XmlPvtuWriter<W> {
  pub fn new(sink: W) -> XmlPvtuWriter<W> {
    XmlPvtuWriter { sink }
  }

  pub fn write_manifest(
    &mut self,
    schema: &PvtuSchema,
    pieces: &[String],
  ) -> Result<(), AetherError> {
    writeln!(self.sink, "<?xml version=\"1.0\"?>")?;
    writeln!(
      self.sink,
      "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">"
    )?;
    writeln!(self.sink, "  <PUnstructuredGrid GhostLevel=\"0\">")?;

    writeln!(self.sink, "    <PPoints>")?;
    writeln!(
      self.sink,
      "      <PDataArray type=\"Float64\" NumberOfComponents=\"3\"/>"
    )?;
    writeln!(self.sink, "    </PPoints>")?;

    writeln!(self.sink, "    <PCells>")?;
    writeln!(
      self.sink,
      "      <PDataArray type=\"UInt64\" Name=\"connectivity\"/>"
    )?;
    writeln!(
      self.sink,
      "      <PDataArray type=\"UInt64\" Name=\"offsets\"/>"
    )?;
    writeln!(
      self.sink,
      "      <PDataArray type=\"UInt8\" Name=\"types\"/>"
    )?;
    writeln!(self.sink, "    </PCells>")?;

    writeln!(self.sink, "    <PPointData>")?;
    for array in &schema.point_arrays {
      self.write_schema_array(array)?;
    }
    writeln!(self.sink, "    </PPointData>")?;

    writeln!(self.sink, "    <PCellData>")?;
    for array in &schema.cell_arrays {
      self.write_schema_array(array)?;
    }
    writeln!(self.sink, "    </PCellData>")?;

    for piece in pieces {
      write!(self.sink, "    <Piece Source=\"")?;
      write_xml_attr_escaped(&mut self.sink, piece)?;
      writeln!(self.sink, "\"/>")?;
    }

    writeln!(self.sink, "  </PUnstructuredGrid>")?;
    writeln!(self.sink, "</VTKFile>")?;
    Ok(())
  }

  fn write_schema_array(
    &mut self,
    array: &VtkArraySchema,
  ) -> Result<(), AetherError> {
    write!(
      self.sink,
      "      <PDataArray type=\"{}\" Name=\"",
      array.vtk_type
    )?;
    write_xml_attr_escaped(&mut self.sink, &array.name)?;
    writeln!(
      self.sink,
      "\" NumberOfComponents=\"{}\"/>",
      array.components
    )?;
    Ok(())
  }
}

fn arrays_for_association<'a>(
  dataset: &'a FieldDataset,
  synthetic_arrays: &'a [FieldArray],
  association: FieldAssociation,
) -> Vec<&'a FieldArray> {
  let mut arrays: Vec<_> = dataset
    .arrays
    .iter()
    .filter(|array| array.association == association)
    .collect();
  arrays.extend(
    synthetic_arrays
      .iter()
      .filter(|array| array.association == association),
  );
  arrays
}

fn build_vtk_ghost_type_array(dataset: &FieldDataset) -> Option<FieldArray> {
  if dataset.find_array("vtkGhostType").is_some() {
    return None;
  }

  let is_ghost = dataset.find_array(partition_debug::IS_GHOST)?;
  if is_ghost.association != FieldAssociation::Cell || is_ghost.components != 1
  {
    return None;
  }

  let values = match &is_ghost.values {
    FieldValues::U8(values) => {
      values.iter().map(|&value| to_ghost_type(value)).collect()
    }
    FieldValues::U32(values) => {
      values.iter().map(|&value| to_ghost_type(value)).collect()
    }
    FieldValues::U64(values) => {
      values.iter().map(|&value| to_ghost_type(value)).collect()
    }
    FieldValues::I64(values) => {
      values.iter().map(|&value| to_ghost_type(value)).collect()
    }
    FieldValues::F64(values) => {
      values.iter().map(|&value| to_ghost_type(value)).collect()
    }
  };

  Some(FieldArray {
    name: "vtkGhostType".to_string(),
    association: FieldAssociation::Cell,
    components: 1,
    values: FieldValues::U8(values),
  })
}

fn validate_point_components(
  dataset: &FieldDataset,
) -> Result<(), AetherError> {
  if dataset.mesh.point_components > 3 {
    return Err(
      AetherError::new(ErrorKind::UnsupportedPointDimension).context(format!(
        "vtk writers support up to 3 point components, got {}",
        dataset.mesh.point_components
      )),
    );
  }

  Ok(())
}

fn to_ghost_type<T>(value: T) -> u8
where
  T: PartialEq + PartialOrd + From<u8>,
{
  // Bit 0 marks duplicate cells for parallel partition ghosting.
  if value > T::from(0) { 1 } else { 0 }
}

fn vtk_xml_type_name(values: &FieldValues) -> &'static str {
  match values {
    FieldValues::F64(_) => "Float64",
    FieldValues::U64(_) => "UInt64",
    FieldValues::U32(_) => "UInt32",
    FieldValues::I64(_) => "Int64",
    FieldValues::U8(_) => "UInt8",
  }
}

fn write_xml_attr_escaped<W: Write>(
  sink: &mut W,
  value: &str,
) -> Result<(), std::io::Error> {
  for ch in value.chars() {
    match ch {
      '&' => write!(sink, "&amp;")?,
      '<' => write!(sink, "&lt;")?,
      '>' => write!(sink, "&gt;")?,
      '"' => write!(sink, "&quot;")?,
      '\'' => write!(sink, "&apos;")?,
      _ => write!(sink, "{}", ch)?,
    }
  }

  Ok(())
}

pub enum ErrorKind {
  UnsupportedPointDimension,
  CellOffsetOverflow,
}

impl ErrorDomain for ErrorKind {
  fn domain(&self) -> &str {
    "vtk serial"
  }
}

impl std::fmt::Display for ErrorKind {
  fn fmt(
    &self,
    f: &mut std::fmt::Formatter<'_>,
  ) -> Result<(), std::fmt::Error> {
    let message = match self {
      ErrorKind::UnsupportedPointDimension => {
        "vtk writer does not support this point dimension"
      }
      ErrorKind::CellOffsetOverflow => {
        "vtk writer encountered a cell offset overflow"
      }
    };

    write!(f, "{}", message)
  }
}

#[cfg(test)]
mod tests {
  use crate::serial::field::{
    FieldArray, FieldAssociation, FieldDataset, FieldValues, UnstructuredMesh,
    partition_debug,
  };

  use super::{
    FieldDatasetWriter, LegacyVtkWriter, PvtuSchema, XmlPvtuWriter,
    XmlVtuWriter,
  };

  #[test]
  fn writes_basic_unstructured_grid() {
    let dataset = FieldDataset {
      title: "draft".to_string(),
      mesh: UnstructuredMesh {
        points: vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        point_components: 2,
        connectivity: vec![0, 1, 2, 3],
        offsets: vec![4],
        cell_types: vec![9], // VTK_QUAD
      },
      arrays: vec![FieldArray {
        name: "rho".to_string(),
        association: FieldAssociation::Cell,
        components: 1,
        values: FieldValues::F64(vec![1.0]),
      }],
    };

    let mut output = Vec::new();
    let mut writer = LegacyVtkWriter::new(&mut output);
    writer.write_dataset(&dataset).unwrap();

    let text = String::from_utf8(output).unwrap();
    assert!(text.contains("DATASET UNSTRUCTURED_GRID"));
    assert!(text.contains("CELL_DATA 1"));
    assert!(text.contains("rho 1 1 double"));
  }

  #[test]
  fn writes_vtk_ghost_type_from_partition_debug_arrays() {
    let dataset = FieldDataset {
      title: "ghosts".to_string(),
      mesh: UnstructuredMesh {
        points: vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        point_components: 2,
        connectivity: vec![0, 1, 2, 3],
        offsets: vec![4],
        cell_types: vec![9],
      },
      arrays: vec![
        FieldArray {
          name: partition_debug::PARTITION_ID.to_string(),
          association: FieldAssociation::Cell,
          components: 1,
          values: FieldValues::U32(vec![1]),
        },
        FieldArray {
          name: partition_debug::GLOBAL_CELL_ID.to_string(),
          association: FieldAssociation::Cell,
          components: 1,
          values: FieldValues::U64(vec![10]),
        },
        FieldArray {
          name: partition_debug::LOCAL_CELL_ID.to_string(),
          association: FieldAssociation::Cell,
          components: 1,
          values: FieldValues::U32(vec![4]),
        },
        FieldArray {
          name: partition_debug::IS_GHOST.to_string(),
          association: FieldAssociation::Cell,
          components: 1,
          values: FieldValues::U8(vec![1]),
        },
        FieldArray {
          name: partition_debug::GHOST_SOURCE_PARTITION.to_string(),
          association: FieldAssociation::Cell,
          components: 1,
          values: FieldValues::I64(vec![0]),
        },
        FieldArray {
          name: partition_debug::GHOST_SOURCE_LOCAL_CELL.to_string(),
          association: FieldAssociation::Cell,
          components: 1,
          values: FieldValues::I64(vec![2]),
        },
      ],
    };

    let mut output = Vec::new();
    let mut writer = LegacyVtkWriter::new(&mut output);
    writer.write_dataset(&dataset).unwrap();

    let text = String::from_utf8(output).unwrap();
    assert!(text.contains("vtkGhostType 1 1 unsigned_char"));
  }

  #[test]
  fn rejects_partial_partition_debug_arrays() {
    let dataset = FieldDataset {
      title: "invalid".to_string(),
      mesh: UnstructuredMesh {
        points: vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        point_components: 2,
        connectivity: vec![0, 1, 2, 3],
        offsets: vec![4],
        cell_types: vec![9],
      },
      arrays: vec![FieldArray {
        name: partition_debug::PARTITION_ID.to_string(),
        association: FieldAssociation::Cell,
        components: 1,
        values: FieldValues::U32(vec![0]),
      }],
    };

    let mut output = Vec::new();
    let mut writer = LegacyVtkWriter::new(&mut output);
    assert!(writer.write_dataset(&dataset).is_err());
  }

  #[test]
  fn writes_xml_vtu_piece() {
    let dataset = FieldDataset {
      title: "xml".to_string(),
      mesh: UnstructuredMesh {
        points: vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        point_components: 2,
        connectivity: vec![0, 1, 2, 3],
        offsets: vec![4],
        cell_types: vec![9],
      },
      arrays: vec![FieldArray {
        name: "rho".to_string(),
        association: FieldAssociation::Cell,
        components: 1,
        values: FieldValues::F64(vec![1.0]),
      }],
    };

    let mut output = Vec::new();
    let mut writer = XmlVtuWriter::new(&mut output);
    writer.write_dataset(&dataset).unwrap();

    let text = String::from_utf8(output).unwrap();
    assert!(text.contains("<VTKFile type=\"UnstructuredGrid\""));
    assert!(text.contains("<DataArray type=\"Float64\" Name=\"rho\""));
  }

  #[test]
  fn writes_xml_pvtu_manifest() {
    let dataset = FieldDataset {
      title: "xml".to_string(),
      mesh: UnstructuredMesh {
        points: vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        point_components: 2,
        connectivity: vec![0, 1, 2, 3],
        offsets: vec![4],
        cell_types: vec![9],
      },
      arrays: vec![FieldArray {
        name: "rho".to_string(),
        association: FieldAssociation::Cell,
        components: 1,
        values: FieldValues::F64(vec![1.0]),
      }],
    };

    let schema = PvtuSchema::from_dataset(&dataset);
    let pieces =
      vec!["piece_0000.vtu".to_string(), "piece_0001.vtu".to_string()];

    let mut output = Vec::new();
    let mut writer = XmlPvtuWriter::new(&mut output);
    writer.write_manifest(&schema, &pieces).unwrap();

    let text = String::from_utf8(output).unwrap();
    assert!(text.contains("<VTKFile type=\"PUnstructuredGrid\""));
    assert!(text.contains("<Piece Source=\"piece_0000.vtu\"/>"));
    assert!(text.contains("<PDataArray type=\"Float64\" Name=\"rho\""));
  }
}
