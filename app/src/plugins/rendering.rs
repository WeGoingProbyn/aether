use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use crate::resources::{Simulation, TemperatureTexture};

// IMPORTANT: bring the trait into scope so `.grid()` resolves on topo
use continuum::topology::Topology;

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
  commands.spawn(Camera2d::default());

  // New solver: dimensions live on the Grid inside Topology
  let g = sim.solver.topo.grid();
  let nx = g.n[0] as u32;
  let ny = g.n[1] as u32;

  let size = Extent3d { width: nx, height: ny, depth_or_array_layers: 1 };

  let mut image = Image::new_fill(
    size,
    TextureDimension::D2,
    &[0, 0, 0, 255],
    TextureFormat::Rgba8UnormSrgb,
    Default::default(),
  );

  // Crisp scaling
  image.sampler = bevy::image::ImageSampler::nearest();

  let handle = images.add(image);
  commands.insert_resource(TemperatureTexture { handle: handle.clone() });

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

  let g = sim.solver.topo.grid();
  let nx = g.n[0];
  let ny = g.n[1];

  // 1) min/max over scalar state u[idx][0]
  let mut tmin = f64::INFINITY;
  let mut tmax = f64::NEG_INFINITY;
  for u in &sim.solver.u {
    let t = u[0];
    tmin = tmin.min(t);
    tmax = tmax.max(t);
  }
  let denom = tmax - tmin;
  let inv = if denom.abs() < 1e-14 { 0.0 } else { 1.0 / denom };

  // Bevy versions differ: some have `Vec<u8>`, some `Option<Vec<u8>>`.
  // Your old code suggests `Option<Vec<u8>>`, so keep that style:
  let data = image
    .data
    .as_mut()
    .expect("Image data should be allocated");

  // Ensure buffer size matches
  let expected = nx * ny * 4;
  if data.len() != expected {
    data.resize(expected, 0);
  }

  for j in 0..ny {
    for i in 0..nx {
      let idx = j * nx + i;

      let t = sim.solver.u[idx][0];
      let mut x = ((t - tmin) * inv).clamp(0.0, 1.0);

      x = normalize(t, tmin, tmax);
      let [r, g, b] = inferno_colormap(x);

      let p = idx * 4;
      data[p + 0] = r;
      data[p + 1] = g;
      data[p + 2] = b;
      data[p + 3] = 255;
    }
  }
}

fn turbo_colormap(x: f64) -> [u8; 3] {
  // Clamp to [0, 1]
  let x = x.clamp(0.0, 1.0);

  // Polynomial approximation of Google's Turbo colormap.
  // Returns linear RGB in [0,1], then converted to u8.
  let r = 0.13572138
  + x * (4.61539260
  + x * (-42.66032258
  + x * (132.13108234
  + x * (-152.94239396
  + x * 59.28637943))));

  let g = 0.09140261
  + x * (2.19418839
  + x * (4.84296658
  + x * (-14.18503333
  + x * (4.27729857
  + x * 2.82956604))));

  let b = 0.10667330
  + x * (12.64194608
  + x * (-48.33107817
  + x * (71.94349767
  + x * (-40.38319551
  + x * 7.13559957))));

  let to_u8 = |v: f64| -> u8 { (v.clamp(0.0, 1.0) * 255.0) as u8 };
  [to_u8(r), to_u8(g), to_u8(b)]
}

pub fn viridis_colormap(x: f64) -> [u8; 3] {
  let x = x.clamp(0.0, 1.0);

  // Approx-style fit; outputs roughly sRGB in [0,1]
  let r = 0.280268 + x * (0.230519 + x * (1.010135 + x * (-1.412354 + x * 0.891495)));
  let g = 0.165368 + x * (1.160916 + x * (-1.491468 + x * (0.897369 + x * -0.226609)));
  let b = 0.476837 + x * (1.061376 + x * (-1.607240 + x * (0.683710 + x * -0.120294)));

  let to_u8 = |v: f64| -> u8 { (v.clamp(0.0, 1.0) * 255.0) as u8 };
  [to_u8(r), to_u8(g), to_u8(b)]
}

pub fn magma_colormap(x: f64) -> [u8; 3] {
  let x = x.clamp(0.0, 1.0);

  // Smooth “magma-ish” analytic approximation (hand-tuned)
  let r = (1.0 - (1.0 - x).powf(2.2)).clamp(0.0, 1.0);
  let g = (x.powf(1.7) * 0.9).clamp(0.0, 1.0);
  let b = ((x * 0.6).powf(2.2)).clamp(0.0, 1.0);

  // Add a gentle purple lift at low end
  let b = (b + (1.0 - x).powf(3.0) * 0.35).clamp(0.0, 1.0);

  let to_u8 = |v: f64| -> u8 { (v * 255.0).round() as u8 };
  [to_u8(r), to_u8(g), to_u8(b)]
}

pub fn inferno_colormap(x: f64) -> [u8; 3] {
  let x = x.clamp(0.0, 1.0);

  let r = (x.powf(0.55)).clamp(0.0, 1.0);
  let g = (x.powf(1.35)).clamp(0.0, 1.0);
  let b = ((x * 0.85).powf(3.0)).clamp(0.0, 1.0);

  // Push midtones warmer
  let g = (g * (0.7 + 0.3 * x)).clamp(0.0, 1.0);

  let to_u8 = |v: f64| -> u8 { (v * 255.0).round() as u8 };
  [to_u8(r), to_u8(g), to_u8(b)]
}

pub fn blue_white_red(x: f64) -> [u8; 3] {
  let x = x.clamp(0.0, 1.0);

  // 0 = blue, 0.5 = white, 1 = red
  let (r, g, b) = if x < 0.5 {
    let t = x / 0.5;
    // blue -> white
    (t, t, 1.0)
  } else {
    let t = (x - 0.5) / 0.5;
    // white -> red
    (1.0, 1.0 - t, 1.0 - t)
  };

  let to_u8 = |v: f64| -> u8 { (v.clamp(0.0, 1.0) * 255.0).round() as u8 };
  [to_u8(r), to_u8(g), to_u8(b)]
}

pub fn cubehelix_colormap(x: f64) -> [u8; 3] {
  let x = x.clamp(0.0, 1.0);

  // Dave Green's cubehelix-like formula
  let a = 0.5 * x * (1.0 - x);
  let angle = 2.0 * std::f64::consts::PI * (0.5 / 3.0 + 1.0 * x);

  let r = (x + a * (-0.14861 * angle.cos() + 1.78277 * angle.sin())).clamp(0.0, 1.0);
  let g = (x + a * (-0.29227 * angle.cos() - 0.90649 * angle.sin())).clamp(0.0, 1.0);
  let b = (x + a * ( 1.97294 * angle.cos() )).clamp(0.0, 1.0);

  let to_u8 = |v: f64| -> u8 { (v * 255.0).round() as u8 };
  [to_u8(r), to_u8(g), to_u8(b)]
}

pub fn plasma_colormap(x: f64) -> [u8; 3] {
  let x = x.clamp(0.0, 1.0);

  // Smooth “plasma-ish” using powered ramps
  let r = (x.powf(0.65)).clamp(0.0, 1.0);
  let g = ((x * (1.0 - x)).powf(0.35) * 1.8).clamp(0.0, 1.0);
  let b = ((1.0 - x).powf(0.9) * 0.9 + x.powf(2.5) * 0.2).clamp(0.0, 1.0);

  let to_u8 = |v: f64| -> u8 { (v * 255.0).round() as u8 };
  [to_u8(r), to_u8(g), to_u8(b)]
}

#[inline]
pub fn normalize(value: f64, min: f64, max: f64) -> f64 {
  if (max - min).abs() < 1e-12 { return 0.0; }
  ((value - min) / (max - min)).clamp(0.0, 1.0)
}
