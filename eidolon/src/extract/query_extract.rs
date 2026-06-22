// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Producer wiring for the semantic query API.
//!
//! Reads the live diagnostic fields a world already maintains
//! (`aer::EulerDiagnosticsStep` writes temperature, pressure, humidity and
//! world-frame velocity each tick) into the snapshot's per-cell
//! [`MeshChannel`]s. The runner that owns the sim thread calls
//! [`extract_quantity_frame`] each tick and pushes the result into a
//! query-facing [`crate::playback::FrameInterpolator`] — a path parallel to,
//! and independent of, the render Update stream, so a headless consumer can
//! query a world without the bevy backend.

use pleroma::Pleroma;
use utility::domain::{FieldKey, FieldName, MeshKey};

use crate::extract::producer::read_scalar_component;
use crate::playback::{MeshChannel, SampleFrame};

/// Binds a snapshot quantity channel to the field (and component) that feeds it.
#[derive(Clone, Copy, Debug)]
pub struct QuantityChannel {
  pub channel: MeshChannel,
  pub field: FieldKey,
  pub component: usize,
}

impl QuantityChannel {
  pub fn new(channel: MeshChannel, field: FieldKey, component: usize) -> Self {
    Self {
      channel,
      field,
      component,
    }
  }
}

/// Build one snapshot frame's quantity channels by reading `channels` from
/// pleroma. A channel whose field is absent or the wrong shape is simply
/// omitted, so the query API reports it as
/// [`crate::query::Sample::Unavailable`] rather than fabricating data.
pub fn extract_quantity_frame(
  pleroma: &Pleroma,
  sim_time: f64,
  channels: &[QuantityChannel],
) -> SampleFrame {
  let mut frame = SampleFrame::new(sim_time);
  for c in channels {
    if let Some(values) = read_scalar_component(pleroma, c.field, c.component) {
      frame.insert_quantity(c.channel, values);
    }
  }
  frame
}

/// The standard atmosphere quantity channels for an Earth-like world: the
/// scalar primitives plus world-frame wind components, all sourced from the
/// diagnostic fields the atmosphere model already writes.
pub fn default_atmosphere_quantities() -> Vec<QuantityChannel> {
  let atm = MeshKey::ATMOSPHERE;
  let field = |name| FieldKey::new(atm, name);
  vec![
    QuantityChannel::new(
      MeshChannel::AtmosphereTemperature,
      field(FieldName::Temperature),
      0,
    ),
    QuantityChannel::new(
      MeshChannel::AtmospherePressure,
      field(FieldName::Pressure),
      0,
    ),
    QuantityChannel::new(
      MeshChannel::AtmosphereHumidity,
      field(FieldName::Humidity),
      0,
    ),
    QuantityChannel::new(
      MeshChannel::AtmosphereWindX,
      field(FieldName::VelocityX),
      0,
    ),
    QuantityChannel::new(
      MeshChannel::AtmosphereWindY,
      field(FieldName::VelocityY),
      0,
    ),
    QuantityChannel::new(
      MeshChannel::AtmosphereWindZ,
      field(FieldName::VelocityZ),
      0,
    ),
  ]
}

/// The inert terrain quantity channels on the surface mesh: static elevation
/// and the categorical surface-type code. Read once (they do not evolve), but
/// flow through the same snapshot path so the query API treats them uniformly.
pub fn default_surface_terrain_quantities() -> Vec<QuantityChannel> {
  let surface = MeshKey::SURFACE;
  vec![
    QuantityChannel::new(
      MeshChannel::SurfaceElevation,
      FieldKey::new(surface, FieldName::SurfaceElevation),
      0,
    ),
    QuantityChannel::new(
      MeshChannel::SurfaceType,
      FieldKey::new(surface, FieldName::SurfaceType),
      0,
    ),
  ]
}
