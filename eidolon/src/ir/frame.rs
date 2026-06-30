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
  /// The view the simulation wants presented, if it owns one (e.g. when a
  /// view-dependent LOD criterion drives refinement from the camera). The
  /// backend positions its view from this; `None` leaves the backend's own
  /// camera (e.g. an interactive orbit camera) in control. Engine-neutral: it is
  /// world-space geometry, not an engine camera type.
  pub camera: Option<RenderCamera>,
}

/// A world-space view the backend should present from: where the eye is, what it
/// looks at, and which way is up. Emitted forward by the extractor when the
/// simulation owns the camera; the backend never writes back.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderCamera {
  pub position: [f64; 3],
  pub target: [f64; 3],
  pub up: [f64; 3],
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
