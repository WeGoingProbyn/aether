// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Consumer-side artistic rendering for the showcase — atmospheric scattering.
//!
//! This deliberately lives in `sandbox`, not in aether/eidolon: *how* the world
//! looks is the consumer's decision. Eidolon supplies the data (the atmosphere
//! shell geometry, the sun direction); this module decides the look. A
//! [`RenderMode`] toggle swaps the atmosphere mesh between eidolon's debug
//! field-painting (`Debug`) and an alpha-blended scattering material
//! (`Rendered`) that lets the surface and ocean meshes show through.
//!
//! Split out so the material + plugin can be reused by other front-ends later.

use bevy::asset::embedded_asset;
use bevy::pbr::{Material, MaterialPlugin, MeshMaterial3d};
use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;

use eidolon::bevy::{RenderRegistry, SunDirection};
use eidolon::ir::{MeshRepresentation, RenderMeshId};
use utility::domain::MeshKey;

use crate::SANDBOX_WORLD_ID;

/// Which way the showcase renders the world.
#[derive(Resource, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum RenderMode {
  /// Eidolon's debug field-painting: meshes coloured by a bound scalar field.
  #[default]
  Debug,
  /// Artistic look: the atmosphere becomes a translucent scattering shell so
  /// the surface and ocean show through.
  Rendered,
}

/// The render mode currently applied to the scene (to detect transitions).
#[derive(Resource, Default)]
struct AppliedRenderMode(Option<RenderMode>);

/// Handle to the one atmosphere material instance, created at startup.
#[derive(Resource)]
struct AtmosphereAssets {
  material: Handle<AtmosphereMaterial>,
}

/// Analytic atmospheric-scattering material (see `atmosphere.wgsl`).
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct AtmosphereMaterial {
  #[uniform(0)]
  pub sky_color: LinearRgba,
  /// xyz = direction toward the sun (w unused).
  #[uniform(0)]
  pub sun_direction: Vec4,
  /// x = intensity, y = rim power, z = base alpha, w = sun-glow strength.
  #[uniform(0)]
  pub params: Vec4,
}

impl Material for AtmosphereMaterial {
  fn fragment_shader() -> ShaderRef {
    "embedded://sandbox/atmosphere.wgsl".into()
  }

  fn alpha_mode(&self) -> AlphaMode {
    AlphaMode::Blend
  }
}

/// Adds the scattering material, the [`RenderMode`] toggle, and the systems that
/// swap the atmosphere mesh's material and feed the sun direction to the shader.
pub struct ShowcaseRenderPlugin;

impl Plugin for ShowcaseRenderPlugin {
  fn build(&self, app: &mut App) {
    embedded_asset!(app, "atmosphere.wgsl");
    app
      .add_plugins(MaterialPlugin::<AtmosphereMaterial>::default())
      .init_resource::<RenderMode>()
      .init_resource::<AppliedRenderMode>()
      .add_systems(Startup, setup_atmosphere_material)
      .add_systems(
        Update,
        (toggle_render_mode, apply_render_mode, drive_sun_direction),
      );
  }
}

fn setup_atmosphere_material(
  mut commands: Commands,
  mut materials: ResMut<Assets<AtmosphereMaterial>>,
) {
  let material = materials.add(AtmosphereMaterial {
    sky_color: LinearRgba::rgb(0.3, 0.55, 1.0),
    sun_direction: Vec4::new(1.0, 0.0, 0.0, 0.0),
    // intensity, rim power, base alpha, sun-glow strength.
    params: Vec4::new(0.6, 3.0, 0.08, 1.5),
  });
  commands.insert_resource(AtmosphereAssets { material });
}

/// `Tab` toggles between the debug and rendered looks.
fn toggle_render_mode(
  keys: Res<ButtonInput<KeyCode>>,
  mut mode: ResMut<RenderMode>,
) {
  if keys.just_pressed(KeyCode::Tab) {
    *mode = match *mode {
      RenderMode::Debug => RenderMode::Rendered,
      RenderMode::Rendered => RenderMode::Debug,
    };
    info!("render mode: {:?}", *mode);
  }
}

fn atmosphere_mesh_handle() -> eidolon::ir::MeshHandle {
  RenderMeshId {
    world: SANDBOX_WORLD_ID,
    mesh: MeshKey::ATMOSPHERE,
    representation: MeshRepresentation::BoundaryFaces,
  }
  .handle()
}

/// On a mode change, swap the atmosphere entity's material. Waits until eidolon
/// has actually spawned the atmosphere mesh (it arrives with the first batch).
fn apply_render_mode(
  mode: Res<RenderMode>,
  mut applied: ResMut<AppliedRenderMode>,
  mut registry: ResMut<RenderRegistry>,
  assets: Option<Res<AtmosphereAssets>>,
  mut commands: Commands,
) {
  // Terrain relief is part of the artistic "rendered" look; the debug field
  // view stays flat so colours read cleanly. Drive this every frame (it's a
  // cheap no-op when unchanged) and independently of the atmosphere mesh, so
  // the default debug view flattens immediately on startup.
  registry.set_displacement_enabled(matches!(*mode, RenderMode::Rendered));

  let Some(assets) = assets else {
    return;
  };
  let Some(entry) = registry.meshes.get(&atmosphere_mesh_handle()) else {
    return; // atmosphere mesh not registered yet.
  };
  if applied.0 == Some(*mode) {
    return;
  }

  let mut entity = commands.entity(entry.entity);
  match *mode {
    RenderMode::Rendered => {
      entity
        .remove::<MeshMaterial3d<StandardMaterial>>()
        .insert(MeshMaterial3d(assets.material.clone()));
    }
    RenderMode::Debug => {
      entity
        .remove::<MeshMaterial3d<AtmosphereMaterial>>()
        .insert(MeshMaterial3d(entry.material_handle.clone()));
    }
  }
  applied.0 = Some(*mode);
}

/// Feed eidolon's (orbiting) sun direction into the scattering shader so the
/// forward-scatter glow tracks the sun.
fn drive_sun_direction(
  sun: Option<Res<SunDirection>>,
  assets: Option<Res<AtmosphereAssets>>,
  mut materials: ResMut<Assets<AtmosphereMaterial>>,
) {
  let (Some(sun), Some(assets)) = (sun, assets) else {
    return;
  };
  let Some(direction) = sun.direction else {
    return;
  };
  if let Some(material) = materials.get_mut(&assets.material) {
    material.sun_direction =
      Vec4::new(direction.x, direction.y, direction.z, 0.0);
  }
}
