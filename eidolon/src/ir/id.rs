// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use utility::domain::{FieldKey, MeshKey, WorldId};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RenderMeshId {
  pub world: WorldId,
  pub mesh: MeshKey,
  pub representation: MeshRepresentation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MeshRepresentation {
  /// Cell-centred volumetric representation.
  Cells,
  /// Exterior or selected boundary faces.
  BoundaryFaces,
  /// Topology/edge view used for seam and partition inspection.
  Wireframe,
  /// Point view, usually cell or face centroids.
  DebugPoints,
  /// Coupler overlay between two meshes.
  Coupler(usize),
  /// Derived geometry produced by a diagnostic extractor.
  Diagnostic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CouplerId {
  pub world: WorldId,
  pub mesh_a: MeshKey,
  pub mesh_b: MeshKey,
  pub index: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct LayerId(pub &'static str);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct DiagnosticKey(pub &'static str);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum LayerSource {
  Field(FieldKey),
  Diagnostic(DiagnosticKey),
  Derived(&'static str),
}
