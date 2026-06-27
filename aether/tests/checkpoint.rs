// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for the save/load checkpoint facility on a fully-assembled
//! world: a snapshot restores the integrated state and clocks bit-for-bit, a
//! reload rewinds a diverged run, the checkpoint round-trips through JSON bytes,
//! and the primitive enables rollback of a `Fail`-policy blow-up (the gap the
//! roadmap called out).

use std::collections::HashMap;

use aer::{AtmosphereModel, AtmosphereScheme, AtmosphereShellLayout};
use aether::{
  core::{Aether, System, World},
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

/// Assemble a small, well-balanced atmosphere world. Returns the runtime and the
/// Euler-state key so tests can compare prognostic state directly.
fn atmosphere_world() -> AetherResult<(Aether, WorldId, FieldKey)> {
  let world_id = WorldId(0);
  let mut factory = WorldFactory::new(world_id, cosmo_factory::earth())
    .with_diagnostics_policy(DiagnosticsPolicy::Observe);
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

fn world_mut(aether: &mut Aether, id: WorldId) -> &mut World {
  aether
    .system_mut(SystemId(0))
    .unwrap()
    .world_mut(id)
    .unwrap()
}

fn euler_snapshot(
  aether: &Aether,
  id: WorldId,
  key: FieldKey,
) -> Vec<[f64; 6]> {
  let world = aether.world(id).unwrap();
  let state: &SoaField<6> = world.pleroma().read(key).unwrap();
  (0..state.len())
    .map(|i| state.state(CellId::from(i)))
    .collect()
}

#[test]
fn checkpoint_restores_state_and_clocks_after_divergence() {
  let (mut aether, world_id, euler) = atmosphere_world().unwrap();

  // Step into a settled state, then snapshot.
  for _ in 0..4 {
    aether.step(20.0).unwrap();
  }
  let checkpoint = aether.world(world_id).unwrap().save_checkpoint().unwrap();
  let saved_time = aether.world(world_id).unwrap().sim_time();
  let saved_state = euler_snapshot(&aether, world_id, euler);

  // Keep stepping so the live world diverges from the snapshot.
  for _ in 0..5 {
    aether.step(20.0).unwrap();
  }
  assert_ne!(
    aether.world(world_id).unwrap().sim_time(),
    saved_time,
    "the live world must have advanced past the snapshot"
  );

  // Restore: clocks and every cell of prognostic state match the snapshot.
  world_mut(&mut aether, world_id)
    .load_checkpoint(&checkpoint)
    .unwrap();
  assert_eq!(aether.world(world_id).unwrap().sim_time(), saved_time);
  assert_eq!(euler_snapshot(&aether, world_id, euler), saved_state);

  // And the run resumes from the restored state.
  aether.step(20.0).unwrap();
  assert!(aether.world(world_id).unwrap().sim_time() > saved_time);
}

#[test]
fn checkpoint_round_trips_through_json_bytes() {
  let (mut aether, world_id, euler) = atmosphere_world().unwrap();
  for _ in 0..3 {
    aether.step(20.0).unwrap();
  }
  let saved_time = aether.world(world_id).unwrap().sim_time();
  let saved_state = euler_snapshot(&aether, world_id, euler);

  let mut bytes = Vec::new();
  aether
    .world(world_id)
    .unwrap()
    .save_checkpoint_to(&mut bytes)
    .unwrap();

  // Diverge, then reload purely from the serialized bytes.
  aether.step(20.0).unwrap();
  world_mut(&mut aether, world_id)
    .load_checkpoint_from(bytes.as_slice())
    .unwrap();

  assert_eq!(aether.world(world_id).unwrap().sim_time(), saved_time);
  assert_eq!(euler_snapshot(&aether, world_id, euler), saved_state);
}

#[test]
fn checkpoint_serialization_is_deterministic() {
  let (mut aether, world_id, _euler) = atmosphere_world().unwrap();
  aether.step(20.0).unwrap();

  let world = aether.world(world_id).unwrap();
  let mut first = Vec::new();
  world.save_checkpoint_to(&mut first).unwrap();
  let mut second = Vec::new();
  world.save_checkpoint_to(&mut second).unwrap();

  assert_eq!(first, second, "checkpoint bytes must be reproducible");
}

#[test]
fn checkpoint_enables_rollback_of_a_fail_blow_up() {
  let (mut aether, world_id, euler) = atmosphere_world().unwrap();
  for _ in 0..2 {
    aether.step(20.0).unwrap();
  }

  // Snapshot a healthy world, then arm Fail and inject a hard blow-up.
  let checkpoint = aether.world(world_id).unwrap().save_checkpoint().unwrap();
  let good_time = aether.world(world_id).unwrap().sim_time();
  let good_state = euler_snapshot(&aether, world_id, euler);

  {
    let world = world_mut(&mut aether, world_id);
    world.set_diagnostics_policy(DiagnosticsPolicy::Fail);
    let state: &mut SoaField<6> = world.pleroma_mut().write(euler).unwrap();
    for i in 0..state.len() {
      state.write(CellId::from(i), &[f64::NAN; 6]);
    }
  }
  assert!(aether.step(20.0).is_err(), "Fail must surface the blow-up");

  // Roll back to the last good checkpoint and resume.
  world_mut(&mut aether, world_id)
    .set_diagnostics_policy(DiagnosticsPolicy::Observe);
  world_mut(&mut aether, world_id)
    .load_checkpoint(&checkpoint)
    .unwrap();
  assert_eq!(aether.world(world_id).unwrap().sim_time(), good_time);
  assert_eq!(euler_snapshot(&aether, world_id, euler), good_state);
  aether
    .step(20.0)
    .expect("a rolled-back world resumes cleanly");
}
