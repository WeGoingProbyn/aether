// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use utility::domain::{CellId, FaceId, MeshKey};

use crate::ir::DiagnosticKey;

#[derive(Clone, Debug, PartialEq)]
pub struct DiagnosticLayer {
  pub id: DiagnosticKey,
  pub label: String,
  pub severity: DiagnosticSeverity,
  pub samples: DiagnosticSamples,
}

impl DiagnosticLayer {
  pub fn scalar(
    id: DiagnosticKey,
    label: impl Into<String>,
    severity: DiagnosticSeverity,
    value: f64,
  ) -> Self {
    Self {
      id,
      label: label.into(),
      severity,
      samples: DiagnosticSamples::Scalar(value),
    }
  }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DiagnosticSeverity {
  Info,
  Warning,
  Error,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DiagnosticSamples {
  Scalar(f64),
  TimeSeries(Vec<(f64, f64)>),
  PerCell { mesh: MeshKey, values: Vec<f64> },
  PerFace { mesh: MeshKey, values: Vec<f64> },
  Messages(Vec<DiagnosticMessage>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiagnosticMessage {
  pub text: String,
  pub related_mesh: Option<MeshKey>,
  pub related_cell: Option<CellId>,
  pub related_face: Option<FaceId>,
}

impl DiagnosticMessage {
  pub fn new(text: impl Into<String>) -> Self {
    Self {
      text: text.into(),
      related_mesh: None,
      related_cell: None,
      related_face: None,
    }
  }
}
