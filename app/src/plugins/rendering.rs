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
      let x = ((t - tmin) * inv).clamp(0.0, 1.0);

      let [r, g, b] = turbo_colormap(x);

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
