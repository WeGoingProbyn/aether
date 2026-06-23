// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Update protocol — the wire shape eidolon uses to talk to a backend.
//!
//! A producer (extractor running on the sim thread) emits an
//! [`UpdateBatch`] per simulation tick. The batch carries a stream of
//! [`Update`] messages: lifecycle (`Register*` / `Free*`) and
//! per-attribute updates (`Update*Geometry`, `UpdateLayerSamples`,
//! `UpdateMeshTransform`, …).
//!
//! Consumers feed batches into a [`crate::registry::BackendRegistry`],
//! which reconstructs a snapshot for non-streaming consumers (VTK) or
//! is queried directly by streaming consumers (bevy).
//!
//! Granularity rule: lifecycle + per-attribute, *not* per-byte field
//! diffs. `UpdateMeshGeometry` replaces the whole geometry payload.
//! Common case (samples-only changed on a static mesh) costs one
//! `UpdateLayerSamples` and zero geometry uploads.

use utility::domain::WorldId;

use crate::ir::{
  CategoricalSamples, ClassSet, LayerHandle, LayerId, LayerSource, MaskSamples,
  MeshHandle, MeshSource, Palette, PaletteHandle, RenderGeometry, RenderMeshId,
  ScalarRange, ScalarSamples, Transform, VectorGlyph, VectorSamples,
  WorldHandle,
};

/// One batch of Updates emitted by the producer for a single sim tick.
#[derive(Clone, Debug, Default)]
pub struct UpdateBatch {
  pub frame: u64,
  pub sim_time: f64,
  pub updates: Vec<Update>,
}

/// Single delta in the wire protocol.
#[derive(Clone, Debug)]
pub enum Update {
  // ---- Palettes ----
  RegisterPalette {
    handle: PaletteHandle,
    palette: Palette,
  },
  FreePalette {
    handle: PaletteHandle,
  },

  // ---- Worlds ----
  RegisterWorld {
    handle: WorldHandle,
    world_id: WorldId,
    label: String,
    transform: Transform,
    transform_epoch: u64,
  },
  UpdateWorldTransform {
    handle: WorldHandle,
    transform: Transform,
    transform_epoch: u64,
  },
  FreeWorld {
    handle: WorldHandle,
  },

  // ---- Meshes ----
  RegisterMesh {
    handle: MeshHandle,
    world: WorldHandle,
    id: RenderMeshId,
    label: String,
    source: MeshSource,
    geometry: RenderGeometry,
    transform: Transform,
    geometry_epoch: u64,
    transform_epoch: u64,
  },
  UpdateMeshGeometry {
    handle: MeshHandle,
    geometry: RenderGeometry,
    epoch: u64,
  },
  UpdateMeshTransform {
    handle: MeshHandle,
    transform: Transform,
    epoch: u64,
  },
  FreeMesh {
    handle: MeshHandle,
  },

  // ---- Layers ----
  RegisterLayer {
    handle: LayerHandle,
    id: LayerId,
    label: String,
    target: MeshHandle,
    source: LayerSource,
    kind: LayerKind,
  },
  UpdateLayerSamples {
    handle: LayerHandle,
    samples: LayerSamples,
    epoch: u64,
  },
  UpdateLayerPalette {
    handle: LayerHandle,
    palette: Option<PaletteHandle>,
  },
  /// Bind / unbind which layer paints a given mesh. `None` clears the
  /// binding. Backends use this to swap which scalar field colours a
  /// surface without re-registering the layer.
  UpdateLayerBinding {
    mesh: MeshHandle,
    layer: Option<LayerHandle>,
  },
  FreeLayer {
    handle: LayerHandle,
  },

  // ---- World-scoped scalars ----
  /// Where the sun appears to be from this world's centre, in world
  /// coordinates. Backends draw a marker / set a directional light.
  UpdateSunDirection {
    world: WorldHandle,
    direction: [f64; 3],
  },

  /// Authoritative simulation time for this batch. Always emitted last
  /// in a batch; consumers read it to drive any time-dependent
  /// rendering (animation, HUDs).
  SetSimTime {
    sim_time: f64,
    frame: u64,
  },
}

/// Type-and-rendering-hint declared at layer registration. Distinct
/// from `LayerSamples` (the data) — kind is "what kind of thing am I",
/// samples is "this tick's payload".
#[derive(Clone, Debug, PartialEq)]
pub enum LayerKind {
  Scalar {
    palette: Option<PaletteHandle>,
    range: Option<ScalarRange>,
  },
  Vector {
    glyph: VectorGlyph,
  },
  Mask,
  /// A categorical field; `classes` names the class ids carried in the
  /// samples. Art-free — the consumer maps class → appearance.
  Categorical {
    classes: ClassSet,
  },
  /// Debug overlays (lines, points, highlights). Not yet routed
  /// through this protocol — debug items still ride on `RenderLayer`
  /// in the snapshot. Reserved for v0.2+.
  Debug,
}

/// Cross-cutting wire shape for `UpdateLayerSamples`. Variant must
/// match the registered `LayerKind`; the registry rejects mismatches.
#[derive(Clone, Debug, PartialEq)]
pub enum LayerSamples {
  Scalar(ScalarSamples),
  Vector(VectorSamples),
  Mask(MaskSamples),
  Categorical(CategoricalSamples),
}

impl LayerSamples {
  pub fn matches_kind(&self, kind: &LayerKind) -> bool {
    matches!(
      (self, kind),
      (LayerSamples::Scalar(_), LayerKind::Scalar { .. })
        | (LayerSamples::Vector(_), LayerKind::Vector { .. })
        | (LayerSamples::Mask(_), LayerKind::Mask)
        | (LayerSamples::Categorical(_), LayerKind::Categorical { .. })
    )
  }
}
