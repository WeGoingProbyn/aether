// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Stable, deterministic u64 ids — the contract a future bevy backend
//! and any other consumer (VTK, Unity port, replay tools) all rely on.

use eidolon::ir::{
  DiagnosticKey, LayerId, LayerSource, MeshHandle, MeshRepresentation,
  RenderMeshId,
};
use utility::domain::{MeshKey, WorldId};

#[test]
fn layer_id_from_static_is_deterministic_and_distinct() {
  let surface_t = LayerId::from_static("surface_temperature");
  let surface_t2 = LayerId::from_static("surface_temperature");
  let surface_p = LayerId::from_static("surface_pressure");
  assert_eq!(surface_t, surface_t2);
  assert_ne!(surface_t, surface_p);
}

#[test]
fn diagnostic_key_uses_same_hash_as_layer_id() {
  let l = LayerId::from_static("conservation_check");
  let d = DiagnosticKey::from_static("conservation_check");
  // Same hash function — useful when a layer + diagnostic share a name.
  assert_eq!(l.0, d.0);
}

#[test]
fn render_mesh_id_handle_is_stable_and_distinct_per_representation() {
  let id = RenderMeshId {
    world: WorldId(0),
    mesh: MeshKey::SURFACE,
    representation: MeshRepresentation::BoundaryFaces,
  };
  let same = RenderMeshId {
    world: WorldId(0),
    mesh: MeshKey::SURFACE,
    representation: MeshRepresentation::BoundaryFaces,
  };
  assert_eq!(id.handle(), same.handle());

  let other = RenderMeshId {
    representation: MeshRepresentation::Cells,
    ..id
  };
  assert_ne!(id.handle(), other.handle());

  // MeshHandle implements Hash, Eq, Copy — usable as a HashMap key.
  let _: MeshHandle = id.handle();
}

#[test]
fn layer_source_derived_uses_static_helper() {
  let a = LayerSource::derived_from_static("speed");
  let b = LayerSource::derived_from_static("speed");
  let c = LayerSource::derived_from_static("kinetic_energy_density");
  assert_eq!(a, b);
  assert_ne!(a, c);
}
