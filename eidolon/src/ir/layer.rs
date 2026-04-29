// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use crate::ir::{
  DebugLayer, LayerId, LayerSource, MaskSamples, Palette, RenderMeshId,
  ScalarRange, ScalarSamples, VectorSamples,
};

#[derive(Clone, Debug, PartialEq)]
pub enum RenderLayer {
  Scalar(ScalarLayer),
  Vector(VectorLayer),
  Mask(MaskLayer),
  Debug(DebugLayer),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScalarLayer {
  pub id: LayerId,
  pub label: String,
  pub target: RenderMeshId,
  pub source: LayerSource,
  pub samples: ScalarSamples,
  pub range: Option<ScalarRange>,
  pub palette: Palette,
}

impl ScalarLayer {
  pub fn new(
    id: LayerId,
    label: impl Into<String>,
    target: RenderMeshId,
    source: LayerSource,
    samples: ScalarSamples,
  ) -> Self {
    Self {
      id,
      label: label.into(),
      target,
      source,
      samples,
      range: None,
      palette: Palette::diagnostic(),
    }
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VectorLayer {
  pub id: LayerId,
  pub label: String,
  pub target: RenderMeshId,
  pub source: LayerSource,
  pub samples: VectorSamples,
  pub glyph: VectorGlyph,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum VectorGlyph {
  Arrow,
  Line,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaskLayer {
  pub id: LayerId,
  pub label: String,
  pub target: RenderMeshId,
  pub source: LayerSource,
  pub samples: MaskSamples,
}
