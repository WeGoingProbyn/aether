// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Proof that HEVI actually *engages* through the live, partitioned atmosphere
//! path (nexus' per-panel dispatch) and removes the vertical-CFL sub-step
//! explosion: at a large outer step with a tight sub-step cap, the explicit
//! scheme runs out of sub-steps and errors, while HEVI clears the same step.

use std::collections::HashMap;

use aer::{AtmosphereModel, AtmosphereScheme, AtmosphereShellLayout};
use aether::{
  core::{Aether, System},
  factory::WorldFactory,
};
use cosmo::factory as cosmo_factory;
use nexus::{MeshKey, WorldId};
use utility::{domain::SystemId, error::AetherResult, thread::pool::Pool};

fn run_one_step(
  scheme: AtmosphereScheme,
  dt: f64,
  max_substeps: usize,
) -> AetherResult<()> {
  let world_id = WorldId(0);
  let angular_dims = [4, 4];
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

  let atmosphere_model = AtmosphereModel::new(MeshKey::ATMOSPHERE)
    .with_cfl(0.25)
    .with_scheme(scheme)
    .with_max_substeps(max_substeps);
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
  aether.step(dt)
}

#[test]
fn hevi_engages_and_removes_substep_explosion() {
  // A 30 s outer step on a 20 km / 6-layer shell is many vertical-acoustic CFL
  // sub-steps (~dozens). With a cap of 5:
  let dt = 30.0;
  let cap = 5;

  // Explicit can't fit it → errors on hitting the sub-step cap.
  assert!(
    run_one_step(AtmosphereScheme::Explicit, dt, cap).is_err(),
    "explicit unexpectedly cleared a {dt}s step within {cap} sub-steps — \
     the vertical CFL should force far more"
  );

  // HEVI integrates the vertical acoustics implicitly per column, so the step
  // is bounded by the (huge) horizontal CFL → it clears within the cap.
  run_one_step(AtmosphereScheme::Hevi, dt, cap).expect(
    "HEVI should clear a 30s step within 5 sub-steps (it engaged and removed \
     the vertical CFL)",
  );
}
