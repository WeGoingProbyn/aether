// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use utility::domain::WorldId;

use crate::ir::{DiagnosticLayer, RenderLayer, RenderMesh, Transform};

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
  /// Human-readable label, useful for backends that group entities or
  /// emit logging.
  pub label: String,
  /// Placement of this world in absolute simulation coordinates. Mesh
  /// geometry is local to this transform (centre + scale).
  pub transform: Transform,
  /// Bumped whenever `transform` changes (e.g. body orbits move).
  pub transform_epoch: u64,
  pub meshes: Vec<RenderMesh>,
  pub layers: Vec<RenderLayer>,
  pub diagnostics: Vec<DiagnosticLayer>,
}

impl RenderWorld {
  /// World with default identity transform and an empty label. Most
  /// extractors should use this as the starting point.
  pub fn new(id: WorldId) -> Self {
    Self {
      id,
      label: String::new(),
      transform: Transform::IDENTITY,
      transform_epoch: 0,
      meshes: Vec::new(),
      layers: Vec::new(),
      diagnostics: Vec::new(),
    }
  }
}
