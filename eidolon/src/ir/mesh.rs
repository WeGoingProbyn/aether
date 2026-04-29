// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::{
  collections::hash_map::DefaultHasher,
  fmt,
  hash::{Hash, Hasher},
  sync::Arc,
};

use utility::domain::{CellId, FaceId, MeshKey};

use crate::ir::{CouplerId, RenderMeshId, Rgba};

#[derive(Clone, Debug, PartialEq)]
pub struct RenderMesh {
  pub id: RenderMeshId,
  pub label: String,
  pub source: MeshSource,
  pub geometry: RenderGeometry,
}

impl RenderMesh {
  pub fn new(
    id: RenderMeshId,
    label: impl Into<String>,
    source: MeshSource,
    geometry: RenderGeometry,
  ) -> Self {
    Self {
      id,
      label: label.into(),
      source,
      geometry,
    }
  }
}

#[derive(Clone, Debug, PartialEq)]
pub enum MeshSource {
  TesseraMesh(MeshKey),
  Coupler(CouplerId),
  Diagnostic(&'static str),
  External(&'static str),
}

#[derive(Clone, Debug, PartialEq)]
pub enum RenderGeometry {
  Triangles(TriangleMesh),
  Lines(LineMesh),
  Points(PointCloud),
  Packed(GeometryAsset),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RenderPrimitive {
  PointList,
  LineList,
  LineStrip,
  TriangleList,
}

impl RenderPrimitive {
  pub const fn indices_per_primitive(self) -> u32 {
    match self {
      Self::PointList | Self::LineStrip => 1,
      Self::LineList => 2,
      Self::TriangleList => 3,
    }
  }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TriangleMesh {
  pub positions: Vec<[f32; 3]>,
  pub normals: Vec<[f32; 3]>,
  pub colours: Vec<Rgba>,
  pub indices: Vec<u32>,
  /// Optional source cell for each triangle.
  pub cell_ids: Vec<Option<CellId>>,
  /// Optional source face for each triangle.
  pub face_ids: Vec<Option<FaceId>>,
}

impl TriangleMesh {
  pub fn triangle_count(&self) -> usize {
    self.indices.len() / 3
  }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LineMesh {
  pub positions: Vec<[f32; 3]>,
  pub segments: Vec<[u32; 2]>,
  pub colours: Vec<Rgba>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PointCloud {
  pub positions: Vec<[f32; 3]>,
  pub colours: Vec<Rgba>,
  pub cell_ids: Vec<Option<CellId>>,
  pub face_ids: Vec<Option<FaceId>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GeometryAsset {
  pub vertex_data: Arc<[u8]>,
  pub index_data: Arc<[u8]>,
  pub vertex_layout: VertexLayoutId,
  pub index_format: IndexFormat,
  pub parts: Vec<GeometryPart>,
}

impl GeometryAsset {
  pub fn cpu_bytes(&self) -> u64 {
    (self.vertex_data.len() + self.index_data.len()) as u64
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GeometryPart {
  pub label: String,
  pub primitive: RenderPrimitive,
  pub index_start: u32,
  pub index_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum IndexFormat {
  U16,
  U32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VertexAttribute {
  pub semantic: VertexSemantic,
  pub format: VertexFormat,
  pub offset: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VertexLayout {
  attributes: Vec<VertexAttribute>,
  stride: u32,
}

impl VertexLayout {
  pub fn new(
    attributes: impl IntoIterator<Item = (VertexSemantic, VertexFormat)>,
  ) -> Self {
    let mut out = Vec::new();
    let mut offset = 0;
    for (semantic, format) in attributes {
      out.push(VertexAttribute {
        semantic,
        format,
        offset,
      });
      offset += format.size_bytes();
    }
    Self {
      attributes: out,
      stride: offset,
    }
  }

  pub fn attributes(&self) -> &[VertexAttribute] {
    &self.attributes
  }

  pub fn stride(&self) -> u32 {
    self.stride
  }

  pub fn id(&self) -> VertexLayoutId {
    let mut hasher = DefaultHasher::new();
    self.hash(&mut hasher);
    VertexLayoutId(hasher.finish())
  }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct VertexLayoutId(pub u64);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum VertexSemantic {
  Position,
  Normal,
  Colour,
  TexCoord,
  Scalar,
  CellId,
  FaceId,
  Custom(&'static str),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum VertexFormat {
  Float32,
  Float32x2,
  Float32x3,
  Float32x4,
  Uint32,
  Uint32x2,
  Uint32x3,
  Uint32x4,
}

impl VertexFormat {
  pub const fn size_bytes(self) -> u32 {
    match self {
      Self::Float32 | Self::Uint32 => 4,
      Self::Float32x2 | Self::Uint32x2 => 8,
      Self::Float32x3 | Self::Uint32x3 => 12,
      Self::Float32x4 | Self::Uint32x4 => 16,
    }
  }
}

#[derive(Clone, Debug, PartialEq)]
pub enum VertexValue {
  Float32(f32),
  Float32x2([f32; 2]),
  Float32x3([f32; 3]),
  Float32x4([f32; 4]),
  Uint32(u32),
  Uint32x2([u32; 2]),
  Uint32x3([u32; 3]),
  Uint32x4([u32; 4]),
}

impl VertexValue {
  pub const fn format(&self) -> VertexFormat {
    match self {
      Self::Float32(_) => VertexFormat::Float32,
      Self::Float32x2(_) => VertexFormat::Float32x2,
      Self::Float32x3(_) => VertexFormat::Float32x3,
      Self::Float32x4(_) => VertexFormat::Float32x4,
      Self::Uint32(_) => VertexFormat::Uint32,
      Self::Uint32x2(_) => VertexFormat::Uint32x2,
      Self::Uint32x3(_) => VertexFormat::Uint32x3,
      Self::Uint32x4(_) => VertexFormat::Uint32x4,
    }
  }

  fn write_ne_bytes(&self, out: &mut Vec<u8>) {
    match self {
      Self::Float32(value) => out.extend_from_slice(&value.to_ne_bytes()),
      Self::Float32x2(values) => push_f32s(out, values),
      Self::Float32x3(values) => push_f32s(out, values),
      Self::Float32x4(values) => push_f32s(out, values),
      Self::Uint32(value) => out.extend_from_slice(&value.to_ne_bytes()),
      Self::Uint32x2(values) => push_u32s(out, values),
      Self::Uint32x3(values) => push_u32s(out, values),
      Self::Uint32x4(values) => push_u32s(out, values),
    }
  }
}

fn push_f32s<const N: usize>(out: &mut Vec<u8>, values: &[f32; N]) {
  for value in values {
    out.extend_from_slice(&value.to_ne_bytes());
  }
}

fn push_u32s<const N: usize>(out: &mut Vec<u8>, values: &[u32; N]) {
  for value in values {
    out.extend_from_slice(&value.to_ne_bytes());
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PackedVertexBuffer {
  pub layout: VertexLayout,
  bytes: Vec<u8>,
}

impl PackedVertexBuffer {
  pub fn new(layout: VertexLayout) -> Self {
    Self {
      layout,
      bytes: Vec::new(),
    }
  }

  pub fn push_vertex(
    &mut self,
    values: &[VertexValue],
  ) -> Result<(), VertexBufferError> {
    if values.len() != self.layout.attributes.len() {
      return Err(VertexBufferError::AttributeCount {
        expected: self.layout.attributes.len(),
        got: values.len(),
      });
    }

    for (index, (attribute, value)) in
      self.layout.attributes.iter().zip(values).enumerate()
    {
      if attribute.format != value.format() {
        return Err(VertexBufferError::FormatMismatch {
          index,
          expected: attribute.format,
          got: value.format(),
        });
      }
    }

    for value in values {
      value.write_ne_bytes(&mut self.bytes);
    }
    Ok(())
  }

  pub fn as_slice(&self) -> &[u8] {
    &self.bytes
  }

  pub fn into_bytes(self) -> Vec<u8> {
    self.bytes
  }

  pub fn vertex_count(&self) -> usize {
    let stride = self.layout.stride() as usize;
    if stride == 0 {
      0
    } else {
      self.bytes.len() / stride
    }
  }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VertexBufferError {
  AttributeCount {
    expected: usize,
    got: usize,
  },
  FormatMismatch {
    index: usize,
    expected: VertexFormat,
    got: VertexFormat,
  },
}

impl fmt::Display for VertexBufferError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::AttributeCount { expected, got } => {
        write!(f, "expected {expected} vertex attributes, got {got}")
      }
      Self::FormatMismatch {
        index,
        expected,
        got,
      } => write!(
        f,
        "vertex attribute {index} format mismatch: expected {expected:?}, got {got:?}"
      ),
    }
  }
}

impl std::error::Error for VertexBufferError {}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn vertex_layout_calculates_offsets_and_id() {
    let layout = VertexLayout::new([
      (VertexSemantic::Position, VertexFormat::Float32x3),
      (VertexSemantic::Normal, VertexFormat::Float32x3),
      (VertexSemantic::Scalar, VertexFormat::Float32),
    ]);

    assert_eq!(layout.stride(), 28);
    assert_eq!(layout.attributes()[0].offset, 0);
    assert_eq!(layout.attributes()[1].offset, 12);
    assert_eq!(layout.attributes()[2].offset, 24);
    assert_eq!(layout.id(), layout.id());
  }

  #[test]
  fn packed_vertex_buffer_validates_layout() {
    let layout = VertexLayout::new([
      (VertexSemantic::Position, VertexFormat::Float32x3),
      (VertexSemantic::Colour, VertexFormat::Float32x4),
    ]);
    let mut buffer = PackedVertexBuffer::new(layout);

    buffer
      .push_vertex(&[
        VertexValue::Float32x3([1.0, 2.0, 3.0]),
        VertexValue::Float32x4([0.0, 1.0, 0.0, 1.0]),
      ])
      .unwrap();

    assert_eq!(buffer.vertex_count(), 1);
    assert_eq!(buffer.as_slice().len(), 28);
  }
}
