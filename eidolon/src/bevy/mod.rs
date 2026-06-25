// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Bevy backend for the eidolon Update protocol. Gated behind the
//! `bevy` cargo feature so users that only want the IR (or VTK
//! export) don't pay for the bevy version matrix.

#![cfg(feature = "bevy")]

pub mod apply;
pub mod camera;
pub mod categorical;
pub mod displace;
pub mod paint;
pub mod palette;
pub mod playback;
pub mod plugin;
pub mod registry;
pub mod sun;
pub mod transform;

pub use camera::spawn_orbit_camera;
pub use categorical::CategoricalStyle;
pub use plugin::{AetherBevyPlugin, UpdateReceiverResource};
pub use registry::{LayerEntry, MeshEntry, RenderRegistry, WorldEntry};
pub use sun::{SunDirection, SunLight, orient_sun_light_system};
