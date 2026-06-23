// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use crate::ir::{
  CategoricalSamples, ClassSet, DebugLayer, LayerId, LayerSource, MaskSamples,
  Palette, RenderMeshId, ScalarRange, ScalarSamples, VectorSamples,
};

#[derive(Clone, Debug, PartialEq)]
pub enum RenderLayer {
  Scalar(ScalarLayer),
  Vector(VectorLayer),
  Mask(MaskLayer),
  Categorical(CategoricalLayer),
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

/// A per-element categorical field (e.g. surface type). Carries the class id of
/// each element plus the [`ClassSet`] describing what those ids mean. Art-free:
/// the consumer chooses how each class looks.
#[derive(Clone, Debug, PartialEq)]
pub struct CategoricalLayer {
  pub id: LayerId,
  pub label: String,
  pub target: RenderMeshId,
  pub source: LayerSource,
  pub samples: CategoricalSamples,
  pub classes: ClassSet,
}
