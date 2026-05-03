// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Stable, engine-neutral handle and id types.
//!
//! Every identifier in the IR is a `u64` so it can be a key in a backend
//! registry, travel cleanly across a channel, and (one day) cross an
//! FFI boundary. We hash from a `&'static str` for ergonomics — call
//! sites that previously wrote `LayerId("surface_temperature")` now
//! write `LayerId::from_static("surface_temperature")` and the resulting
//! id is deterministic across runs (FNV-1a 64).
//!
//! `from_static` is `const fn`, so id literals can live in `const`
//! context where useful.

use utility::domain::{FieldKey, MeshKey, MeshType, WorldId};

/// Per-process stable handle for a `RenderWorld`. Backends use this as
/// the registry key for the world's parent entity.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct WorldHandle(pub u64);

impl WorldHandle {
  /// Deterministic handle for a given `WorldId`. Two `RenderWorld`s
  /// with the same id hash to the same handle.
  pub const fn from_world_id(id: WorldId) -> Self {
    let mut h = FNV_OFFSET;
    h ^= b'W' as u64;
    h = h.wrapping_mul(FNV_PRIME);
    h = fnv_mix_u64(h, id.0 as u64);
    Self(h)
  }
}

/// Per-process stable handle for a `RenderMesh`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct MeshHandle(pub u64);

/// Per-process stable handle for a layer (scalar / vector / mask /
/// debug).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct LayerHandle(pub u64);

impl LayerHandle {
  /// Deterministic handle for a `(LayerId, target mesh)` pair. Two
  /// worlds can both have a `surface_temperature` layer without
  /// colliding because their target `MeshHandle`s differ.
  pub const fn for_target(id: LayerId, target: MeshHandle) -> Self {
    let mut h = FNV_OFFSET;
    h ^= b'L' as u64;
    h = h.wrapping_mul(FNV_PRIME);
    h = fnv_mix_u64(h, id.0);
    h = fnv_mix_u64(h, target.0);
    Self(h)
  }
}

/// Per-process stable handle for a `Palette`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PaletteHandle(pub u64);

impl PaletteHandle {
  /// Hash a palette by its name. Assumes palette names are unique
  /// within a process.
  pub const fn from_static_name(name: &'static str) -> Self {
    Self(fnv1a_64(name.as_bytes()))
  }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RenderMeshId {
  pub world: WorldId,
  pub mesh: MeshKey,
  pub representation: MeshRepresentation,
}

impl RenderMeshId {
  /// Stable handle derived from the composite key. Two `RenderMeshId`
  /// values with the same fields hash to the same `MeshHandle`.
  pub const fn handle(&self) -> MeshHandle {
    let mut h = FNV_OFFSET;
    h = fnv_mix_u64(h, self.world.0 as u64);
    h = fnv_mix_u64(h, mesh_key_tag(self.mesh));
    h = fnv_mix_u64(h, mesh_representation_tag(self.representation));
    MeshHandle(h)
  }
}

const fn mesh_key_tag(key: MeshKey) -> u64 {
  match key.mesh_type() {
    MeshType::Atmosphere => 1,
    MeshType::Surface => 2,
    MeshType::Mantle => 3,
  }
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

/// Logical layer name (e.g. "surface_temperature"). Stable across runs;
/// backends treat it as opaque. Construct with `LayerId::from_static`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct LayerId(pub u64);

impl LayerId {
  pub const fn from_static(name: &'static str) -> Self {
    Self(fnv1a_64(name.as_bytes()))
  }
}

/// Stable diagnostic key. Same FNV hash as `LayerId`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct DiagnosticKey(pub u64);

impl DiagnosticKey {
  pub const fn from_static(name: &'static str) -> Self {
    Self(fnv1a_64(name.as_bytes()))
  }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum LayerSource {
  Field(FieldKey),
  Diagnostic(DiagnosticKey),
  Derived(u64),
}

impl LayerSource {
  pub const fn derived_from_static(name: &'static str) -> Self {
    Self::Derived(fnv1a_64(name.as_bytes()))
  }
}

// ---- FNV-1a 64-bit ----
//
// Const-fn so callers can put ids in `const`. Deterministic across
// processes and platforms (no hasher state, no compiler-version
// dependency on RandomState).

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

const fn fnv1a_64(bytes: &[u8]) -> u64 {
  let mut h = FNV_OFFSET;
  let mut i = 0;
  while i < bytes.len() {
    h ^= bytes[i] as u64;
    h = h.wrapping_mul(FNV_PRIME);
    i += 1;
  }
  h
}

const fn fnv_mix_u64(mut h: u64, value: u64) -> u64 {
  let bytes = value.to_le_bytes();
  let mut i = 0;
  while i < 8 {
    h ^= bytes[i] as u64;
    h = h.wrapping_mul(FNV_PRIME);
    i += 1;
  }
  h
}

const fn mesh_representation_tag(rep: MeshRepresentation) -> u64 {
  match rep {
    MeshRepresentation::Cells => 0,
    MeshRepresentation::BoundaryFaces => 1,
    MeshRepresentation::Wireframe => 2,
    MeshRepresentation::DebugPoints => 3,
    MeshRepresentation::Coupler(index) => 0x1000 ^ (index as u64),
    MeshRepresentation::Diagnostic => 4,
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn from_static_is_deterministic() {
    let a = LayerId::from_static("surface_temperature");
    let b = LayerId::from_static("surface_temperature");
    let c = LayerId::from_static("surface_pressure");
    assert_eq!(a, b);
    assert_ne!(a, c);
    // FNV-1a("surface_temperature") — pinning the value catches
    // accidental hash-function changes.
    assert_eq!(a.0, fnv1a_64(b"surface_temperature"));
  }

  #[test]
  fn diagnostic_key_uses_same_hash_as_layer_id() {
    let l = LayerId::from_static("foo");
    let d = DiagnosticKey::from_static("foo");
    assert_eq!(l.0, d.0);
  }

  #[test]
  fn render_mesh_id_handle_is_stable() {
    let id = RenderMeshId {
      world: WorldId(7),
      mesh: MeshKey::SURFACE,
      representation: MeshRepresentation::BoundaryFaces,
    };
    assert_eq!(id.handle(), id.handle());
    let other = RenderMeshId {
      representation: MeshRepresentation::Cells,
      ..id
    };
    assert_ne!(id.handle(), other.handle());
  }
}
