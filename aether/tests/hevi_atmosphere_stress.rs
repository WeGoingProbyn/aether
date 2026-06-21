// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Stress / bisection harness for HEVI on the *demo-like* atmosphere: gravity,
//! rotation and the current-state background correction, stepped at the demo's
//! large outer dt. Isolates which ingredient (if any) breaks at large HEVI
//! steps, independent of the ocean coupling / radiation.

use std::collections::HashMap;

use aer::{AtmosphereModel, AtmosphereScheme, AtmosphereShellLayout};
use aether::{
  core::{Aether, System},
  factory::WorldFactory,
};
use cosmo::factory as cosmo_factory;
use nexus::{FieldStorage, MeshKey, SoaField, WorldId};
use utility::{
  domain::{CellId, SystemId},
  error::AetherResult,
  thread::pool::Pool,
};

struct Opts {
  scheme: AtmosphereScheme,
  rotation: bool,
  dt: f64,
  steps: usize,
}

fn density_spread(state: &SoaField<6>) -> (f64, f64) {
  let mut min = f64::INFINITY;
  let mut max = f64::NEG_INFINITY;
  for i in 0..state.len() {
    let rho = state.state(CellId::from(i))[0];
    if rho.is_finite() {
      min = min.min(rho);
      max = max.max(rho);
    }
  }
  (min, max)
}

fn run(opts: &Opts) -> AetherResult<bool> {
  let world_id = WorldId(0);
  let angular_dims = [16, 16];
  let atmosphere_radial_layers = 6;

  let mut factory =
    WorldFactory::new(world_id, cosmo_factory::earth()).with_partition_count(6);
  let constants = factory.constants();
  let shell_layout =
    AtmosphereShellLayout::new(&constants, 20_000.0, 10_000.0)?;
  factory = factory.cube_sphere_atmosphere(
    shell_layout.atmosphere_shell_spec(angular_dims, atmosphere_radial_layers),
  );
  let atmosphere_mesh =
    factory.tessera().mesh(MeshKey::ATMOSPHERE).unwrap().clone();

  let mut atmosphere_model = AtmosphereModel::new(MeshKey::ATMOSPHERE)
    .with_cfl(0.25)
    .with_scheme(opts.scheme);
  if opts.rotation {
    atmosphere_model = atmosphere_model.with_rotation();
  }
  let fields = atmosphere_model.fields();
  atmosphere_model.register_fields(
    factory.pleroma_mut(),
    atmosphere_mesh.as_ref(),
    &constants,
    shell_layout.reference_radius(),
  )?;
  atmosphere_model.add_stages(factory.nexus_mut())?;

  let world = factory.build()?;
  let mut systems = HashMap::new();
  systems.insert(SystemId(0), System::single(SystemId(0), world));
  let mut aether = Aether::new(systems, Pool::default());

  for _ in 0..opts.steps {
    aether.step(opts.dt)?;
  }

  let world = aether.world(world_id).unwrap();
  let state: &SoaField<6> = world.pleroma().read(fields.euler_state).unwrap();
  let (min, max) = density_spread(state);
  let finite_positive = (0..state.len()).all(|i| {
    let s = state.state(CellId::from(i));
    s.iter().all(|v| v.is_finite()) && s[0] > 0.0 && s[4] > 0.0
  });
  eprintln!("    final rho [{min:.3}, {max:.3}] spread {:.3}", max - min);
  Ok(finite_positive)
}

/// With well-balancing (and no background-correction hack), the bare
/// hydrostatic atmosphere — gravity, rotation, either scheme — must stay
/// physical over a long run. Before well-balancing the un-corrected scheme
/// drifted into a runaway within ~100 steps.
#[test]
fn well_balanced_atmosphere_is_stable_over_a_long_run() {
  let dt = 20.0;
  let steps = 120;

  let cases = [
    ("explicit, grav+rot", AtmosphereScheme::Explicit, true),
    ("hevi, grav only", AtmosphereScheme::Hevi, false),
    ("hevi, grav+rot", AtmosphereScheme::Hevi, true),
  ];
  for (label, scheme, rotation) in cases {
    let opts = Opts {
      scheme,
      rotation,
      dt,
      steps,
    };
    let result = run(&opts);
    match &result {
      Ok(true) => eprintln!("[OK   finite] {label}"),
      Ok(false) => eprintln!("[BAD  nonfin] {label}"),
      Err(e) => eprintln!("[ERR step    ] {label}: {e}"),
    }
    assert!(
      matches!(result, Ok(true)),
      "{label}: well-balanced atmosphere should stay physical"
    );
  }
}
