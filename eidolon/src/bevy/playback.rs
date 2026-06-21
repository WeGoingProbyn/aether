// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Bevy wiring for the engine-neutral [`crate::playback`] frame interpolator.
//!
//! The interpolation logic itself is backend-agnostic; this module only adapts
//! it to bevy — holding it as a resource, advancing its clock from bevy's
//! `Time`, and forcing a repaint of bound meshes each frame while interpolation
//! is active (the interpolated values change continuously, so the dirty-mesh
//! fast-path is bypassed during playback).

use bevy::prelude::*;

use crate::ir::{LayerSamples, ScalarSamples};
use crate::playback::{FrameInterpolator, SampleFrame};

use super::registry::RenderRegistry;

/// Bevy resource holding the render-side frame interpolator.
#[derive(Resource, Default)]
pub struct FrameInterpolatorResource(pub FrameInterpolator);

/// Snapshot the registry's current per-cell scalar layer samples into a
/// [`SampleFrame`] at the given simulation time — called at each frame boundary
/// (`SetSimTime`) so the interpolator always has the two latest frames.
pub fn snapshot_sample_frame(
  registry: &RenderRegistry,
  sim_time: f64,
) -> SampleFrame {
  let mut frame = SampleFrame::new(sim_time);
  for (handle, entry) in &registry.layers {
    if let Some(LayerSamples::Scalar(ScalarSamples::PerCell(values))) =
      &entry.samples
    {
      frame.insert(*handle, values.clone());
    }
  }
  frame
}

/// Advance the render clock by the real frame delta and, while interpolating,
/// mark every bound mesh dirty so `paint` re-bakes colours from the freshly
/// interpolated samples.
pub fn advance_playback_system(
  time: Res<Time>,
  mut interp: ResMut<FrameInterpolatorResource>,
  mut registry: ResMut<RenderRegistry>,
) {
  interp.0.advance(time.delta().as_secs_f64());
  if interp.0.is_primed() {
    let bound: Vec<_> = registry.bindings.keys().copied().collect();
    for mesh in bound {
      registry.dirty_meshes.insert(mesh);
    }
  }
}
