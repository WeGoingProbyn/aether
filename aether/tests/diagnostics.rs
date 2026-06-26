// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for the runtime diagnostics layer on a fully-assembled
//! world: the in-DAG conservation monitor publishes a health report readable
//! via `World::diagnostics`, and a `Fail` policy turns a silent blow-up into a
//! surfaced `Err` while leaving the world's clocks frozen at the last good tick.

use std::collections::HashMap;

use aer::{AtmosphereModel, AtmosphereScheme, AtmosphereShellLayout};
use aether::{
  core::{Aether, System},
  factory::WorldFactory,
};
use cosmo::factory as cosmo_factory;
use nexus::{FieldKey, FieldStorage, MeshKey, SoaField, WorldId};
use utility::{
  diagnostics::DiagnosticsPolicy,
  domain::{CellId, SystemId},
  error::AetherResult,
  thread::pool::Pool,
};

/// Assemble a small, well-balanced atmosphere world with the conservation
/// monitor enabled under `policy`. Returns the runtime and the Euler-state key.
fn monitored_world(
  policy: DiagnosticsPolicy,
) -> AetherResult<(Aether, WorldId, FieldKey)> {
  let world_id = WorldId(0);
  let mut factory = WorldFactory::new(world_id, cosmo_factory::earth())
    .with_diagnostics_policy(policy);
  let constants = factory.constants();
  let shell_layout =
    AtmosphereShellLayout::new(&constants, 20_000.0, 10_000.0)?;
  factory = factory
    .cube_sphere_atmosphere(shell_layout.atmosphere_shell_spec([8, 8], 4));
  let atmosphere_mesh =
    factory.tessera().mesh(MeshKey::ATMOSPHERE).unwrap().clone();

  let atmosphere_model = AtmosphereModel::new(MeshKey::ATMOSPHERE)
    .with_cfl(0.25)
    .with_scheme(AtmosphereScheme::Explicit)
    .with_conservation_monitor(0.1);
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
  let aether = Aether::new(systems, Pool::default());
  Ok((aether, world_id, fields.euler_state))
}

#[test]
fn diagnostics_surface_a_healthy_atmosphere() {
  let (mut aether, world_id, euler_state) =
    monitored_world(DiagnosticsPolicy::Observe).unwrap();

  for _ in 0..5 {
    aether.step(20.0).unwrap();
  }

  let world = aether.world(world_id).unwrap();
  let diagnostics = world.diagnostics().expect("Diagnostics resource present");
  assert_eq!(diagnostics.policy, DiagnosticsPolicy::Observe);
  assert!(
    !diagnostics.has_non_finite(),
    "healthy run has no non-finite state"
  );

  let report = diagnostics
    .fields
    .get(&euler_state)
    .expect("monitor published a report for the Euler state");
  assert_eq!(report.non_finite_cells, 0);
  assert_eq!(
    report.conserved.len(),
    6,
    "six conserved totals for moist Euler"
  );
  assert!(
    report.conserved.iter().all(|(_, total)| total.is_finite()),
    "conserved totals are finite"
  );
  // A well-balanced atmosphere holds its conserved totals tightly.
  assert!(
    diagnostics.worst_drift() < 0.1,
    "drift {} should stay under threshold",
    diagnostics.worst_drift()
  );
}

#[test]
fn fail_policy_surfaces_a_blow_up_and_freezes_the_clock() {
  let (mut aether, world_id, euler_state) =
    monitored_world(DiagnosticsPolicy::Observe).unwrap();

  // A couple of healthy steps, then capture the clock.
  for _ in 0..2 {
    aether.step(20.0).unwrap();
  }
  let good_sim_time = aether.world(world_id).unwrap().sim_time();

  // Switch to Fail and inject a hard blow-up into the prognostic state.
  {
    let world = aether
      .system_mut(SystemId(0))
      .unwrap()
      .world_mut(world_id)
      .unwrap();
    world.set_diagnostics_policy(DiagnosticsPolicy::Fail);

    let state: &mut SoaField<6> =
      world.pleroma_mut().write(euler_state).unwrap();
    for i in 0..state.len() {
      state.write(CellId::from(i), &[f64::NAN; 6]);
    }
  }

  // The blow-up must surface as an error rather than silently advancing.
  let result = aether.step(20.0);
  assert!(
    result.is_err(),
    "Fail policy must surface the non-finite state"
  );

  // Contract: clocks do not advance past the last good tick on a failed step.
  let world = aether.world(world_id).unwrap();
  assert_eq!(
    world.sim_time(),
    good_sim_time,
    "sim clock must stay frozen at the last good tick"
  );
}
