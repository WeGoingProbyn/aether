// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! `AetherBevyPlugin` — wires the eidolon Update protocol into a bevy
//! `App`. Insert the plugin together with the receiver half of a
//! `render_channel` and a worker thread (e.g. via
//! [`crate::runtime::spawn_runner`]) will feed the bevy app each tick.

use std::sync::Mutex;

use bevy::prelude::*;

use crate::runtime::UpdateReceiver;

use super::{
  apply::apply_updates_system, paint::paint_layers_system,
  registry::RenderRegistry,
};

/// Resource wrapper holding the receiver half of the update channel.
/// Drained by the apply system every `PreUpdate`.
#[derive(Resource)]
pub struct UpdateReceiverResource {
  pub receiver: Mutex<UpdateReceiver>,
}

impl UpdateReceiverResource {
  pub fn new(receiver: UpdateReceiver) -> Self {
    Self {
      receiver: Mutex::new(receiver),
    }
  }
}

pub struct AetherBevyPlugin {
  receiver: Mutex<Option<UpdateReceiver>>,
}

impl AetherBevyPlugin {
  pub fn new(receiver: UpdateReceiver) -> Self {
    Self {
      receiver: Mutex::new(Some(receiver)),
    }
  }
}

impl Plugin for AetherBevyPlugin {
  fn build(&self, app: &mut App) {
    let receiver = self
      .receiver
      .lock()
      .expect("AetherBevyPlugin receiver mutex poisoned")
      .take()
      .expect(
        "AetherBevyPlugin::build called twice on the same plugin instance",
      );
    app
      .insert_resource(UpdateReceiverResource::new(receiver))
      .init_resource::<RenderRegistry>()
      .add_systems(PreUpdate, apply_updates_system)
      .add_systems(Update, paint_layers_system);
  }
}
