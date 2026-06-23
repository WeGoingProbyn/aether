// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use utility::domain::{CellId, FaceId};

#[derive(Clone, Debug, PartialEq)]
pub enum ScalarSamples {
  PerCell(Vec<f64>),
  PerFace(Vec<f64>),
  PerVertex(Vec<f64>),
  SparseCells(Vec<(CellId, f64)>),
  SparseFaces(Vec<(FaceId, f64)>),
}

impl ScalarSamples {
  pub fn len(&self) -> usize {
    match self {
      Self::PerCell(values)
      | Self::PerFace(values)
      | Self::PerVertex(values) => values.len(),
      Self::SparseCells(values) => values.len(),
      Self::SparseFaces(values) => values.len(),
    }
  }

  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }
}

#[derive(Clone, Debug, PartialEq)]
pub enum VectorSamples {
  PerCell(Vec<[f64; 3]>),
  PerFace(Vec<[f64; 3]>),
  PerVertex(Vec<[f64; 3]>),
  SparseCells(Vec<(CellId, [f64; 3])>),
  SparseFaces(Vec<(FaceId, [f64; 3])>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum MaskSamples {
  PerCell(Vec<bool>),
  PerFace(Vec<bool>),
  SparseCells(Vec<CellId>),
  SparseFaces(Vec<FaceId>),
}

/// Per-element categorical class ids (e.g. ocean / land / ice). The meaning of
/// each id is carried out-of-band by the layer's [`crate::ir::ClassSet`]; the
/// samples themselves are pure data, with no colour or material baked in.
#[derive(Clone, Debug, PartialEq)]
pub enum CategoricalSamples {
  PerCell(Vec<u32>),
  PerFace(Vec<u32>),
}

impl CategoricalSamples {
  pub fn len(&self) -> usize {
    match self {
      Self::PerCell(v) | Self::PerFace(v) => v.len(),
    }
  }

  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }
}
