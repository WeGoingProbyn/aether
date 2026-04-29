// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use utility::domain::WorldId;

use crate::ir::{DiagnosticLayer, RenderLayer, RenderMesh};

#[derive(Clone, Debug, Default)]
pub struct RenderFrame {
  /// Monotonic renderer-facing frame number.
  pub frame: u64,
  /// Simulation time represented by this frame, in seconds.
  pub sim_time: f64,
  pub worlds: Vec<RenderWorld>,
}

#[derive(Clone, Debug)]
pub struct RenderWorld {
  pub id: WorldId,
  pub meshes: Vec<RenderMesh>,
  pub layers: Vec<RenderLayer>,
  pub diagnostics: Vec<DiagnosticLayer>,
}
