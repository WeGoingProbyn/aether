// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use utility::domain::{CellId, FaceId, MeshKey};

use crate::ir::{LayerId, Rgba};

#[derive(Clone, Debug, PartialEq)]
pub struct DebugLayer {
  pub id: LayerId,
  pub label: String,
  pub items: Vec<DebugItem>,
}

impl DebugLayer {
  pub fn new(id: LayerId, label: impl Into<String>) -> Self {
    Self {
      id,
      label: label.into(),
      items: Vec::new(),
    }
  }
}

#[derive(Clone, Debug, PartialEq)]
pub enum DebugItem {
  Line(DebugLine),
  Point(DebugPoint),
  FaceHighlight(FaceHighlight),
  CellHighlight(CellHighlight),
}

#[derive(Clone, Debug, PartialEq)]
pub struct DebugLine {
  pub from: [f32; 3],
  pub to: [f32; 3],
  pub colour: Rgba,
  pub label: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DebugPoint {
  pub position: [f32; 3],
  pub radius: f32,
  pub colour: Rgba,
  pub label: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FaceHighlight {
  pub mesh: MeshKey,
  pub face: FaceId,
  pub colour: Rgba,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CellHighlight {
  pub mesh: MeshKey,
  pub cell: CellId,
  pub colour: Rgba,
}
