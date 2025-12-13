use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use crate::resources::{Simulation, TemperatureTexture};

pub struct RenderingPlugin;

impl Plugin for RenderingPlugin {
  fn build(&self, app: &mut App) {
    app.add_systems(Startup, setup_temperature_view)
      .add_systems(Update, upload_temperature_to_texture);
  }
}

fn setup_temperature_view(
  mut commands: Commands,
  mut images: ResMut<Assets<Image>>,
  sim: Res<Simulation>,
) {
  // Minimal 2D camera
  commands.spawn(Camera2d::default());

  let nx = sim.solver.grid.nx as u32;
  let ny = sim.solver.grid.ny as u32;

  // RGBA8 texture: 4 bytes per pixel
  let size = Extent3d { width: nx, height: ny, depth_or_array_layers: 1 };

  let mut image = Image::new_fill(
    size,
    TextureDimension::D2,
    &[0, 0, 0, 255],
    TextureFormat::Rgba8UnormSrgb,
    Default::default(),  
  );

  // Optional: make it crisp when scaled up
  image.sampler = bevy::image::ImageSampler::nearest();

  let handle = images.add(image);

  commands.insert_resource(TemperatureTexture { handle: handle.clone() });

  // Show it as a sprite in the world
  commands.spawn((
    Sprite {
      image: handle,
      ..default()
    },
    Transform::from_scale(Vec3::splat(4.0)),
    GlobalTransform::default(),
  ));
}

fn upload_temperature_to_texture(
  sim: Res<Simulation>,
  tex: Res<TemperatureTexture>,
  mut images: ResMut<Assets<Image>>,
) {
  if !sim.is_changed() {
    return;
  }

  let Some(image) = images.get_mut(&tex.handle) else { return; };

  let nx = sim.solver.grid.nx;
  let ny = sim.solver.grid.ny;

  // 1) Find min/max for normalization (simple first pass)
  let mut tmin = f32::INFINITY;
  let mut tmax = f32::NEG_INFINITY;
  for &t in &sim.solver.temperature.field {
    tmin = tmin.min(t);
    tmax = tmax.max(t);
  }
  let inv = if (tmax - tmin).abs() < 1e-12 { 0.0 } else { 1.0 / (tmax - tmin) };

  let data = &mut image.data;

  for j in 0..ny {
    for i in 0..nx {
      let idx = j * nx + i;
      let t = sim.solver.temperature.field[idx];
      let x = ((t - tmin) * inv).clamp(0.0, 1.0);
      let v = (x * 255.0) as u8;

      let p = idx * 4;
      data.as_mut().unwrap()[p + 0] = v;   // R
      data.as_mut().unwrap()[p + 1] = v;   // G
      data.as_mut().unwrap()[p + 2] = v;   // B
      data.as_mut().unwrap()[p + 3] = 255; // A
    }
  }
}
