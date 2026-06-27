// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for the runtime event bus: the in-DAG conservation monitor
//! broadcasts a `NonFiniteState` event (the soft counterpart to a `Fail` `Err`),
//! the world emits regime/transition events, the buffer is published on both tick
//! paths (even on a failed tick), and a consumer polling `World::events()` can
//! react — here by rolling back to a checkpoint (the instability→recovery loop the
//! roadmap described).

use std::collections::HashMap;

use aer::{AtmosphereModel, AtmosphereScheme, AtmosphereShellLayout};
use aether::{
  core::{Aether, System, World},
  factory::WorldFactory,
};
use chronos::{Regime, TransitionKind, TransitionState};
use cosmo::factory as cosmo_factory;
use nexus::{FieldKey, FieldStorage, MeshKey, SoaField, WorldId};
use utility::{
  diagnostics::DiagnosticsPolicy,
  domain::{CellId, SystemId},
  error::AetherResult,
  events::{Event, RegimeKind},
  thread::pool::Pool,
};

/// A small atmosphere world with the conservation monitor emitting events under
/// `policy`. Returns the runtime and the Euler-state key.
fn event_world(
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
    .with_conservation_monitor(0.1)
    .with_conservation_monitor_events();
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
  Ok((
    Aether::new(systems, Pool::default()),
    world_id,
    fields.euler_state,
  ))
}

fn world_mut(aether: &mut Aether, id: WorldId) -> &mut World {
  aether
    .system_mut(SystemId(0))
    .unwrap()
    .world_mut(id)
    .unwrap()
}

/// Corrupt only the water tracer (component 5, `ρq`) with NaN, leaving density /
/// momentum / energy physical. The per-tick `EulerDiagnosticsStep` only rejects
/// non-finite density/pressure, so the tick still completes — but the monitor's
/// full-state finiteness sweep catches the NaN and broadcasts it. This models a
/// blow-up the hard physicality guard misses, which is exactly what the soft
/// event is for.
fn inject_tracer_nan(aether: &mut Aether, id: WorldId, euler: FieldKey) {
  let world = world_mut(aether, id);
  let state: &mut SoaField<6> = world.pleroma_mut().write(euler).unwrap();
  for i in 0..state.len() {
    let mut s = state.state(CellId::from(i));
    s[5] = f64::NAN;
    state.write(CellId::from(i), &s);
  }
}

fn has_non_finite(events: &[Event], field: FieldKey) -> bool {
  events
    .iter()
    .any(|e| matches!(e, Event::NonFiniteState { field: f } if *f == field))
}

#[test]
fn instability_surfaces_as_a_soft_event_under_observe() {
  let (mut aether, id, euler) =
    event_world(DiagnosticsPolicy::Observe).unwrap();

  // A couple of healthy steps publish no instability events.
  for _ in 0..2 {
    aether.step(20.0).unwrap();
  }
  assert!(
    !has_non_finite(aether.world(id).unwrap().events(), euler),
    "a healthy tick emits no NonFiniteState"
  );

  // Inject a blow-up; under Observe the tick still succeeds, and the instability
  // surfaces as a pollable event.
  inject_tracer_nan(&mut aether, id, euler);
  aether.step(20.0).expect("Observe does not hard-fail");
  assert!(
    has_non_finite(aether.world(id).unwrap().events(), euler),
    "the monitor must broadcast NonFiniteState"
  );
}

#[test]
fn instability_event_publishes_even_when_the_tick_hard_fails() {
  let (mut aether, id, euler) = event_world(DiagnosticsPolicy::Fail).unwrap();
  aether.step(20.0).unwrap();

  inject_tracer_nan(&mut aether, id, euler);
  // Fail returns Err, but the event is published at the barrier regardless, so a
  // consumer handling the error can still see *why*.
  assert!(
    aether.step(20.0).is_err(),
    "Fail surfaces the blow-up as Err"
  );
  assert!(
    has_non_finite(aether.world(id).unwrap().events(), euler),
    "the NonFiniteState event publishes even on a failed tick"
  );
}

#[test]
fn regime_and_transition_changes_emit_events() {
  let (mut aether, id, _euler) =
    event_world(DiagnosticsPolicy::Observe).unwrap();

  // Emit at the world level, then step once so the barrier publishes the buffer.
  {
    let world = world_mut(&mut aether, id);
    world.set_regime(Regime::Climatology);
    world.begin_transition(TransitionState::new(
      TransitionKind::LiveToClimatology,
      100.0,
    ));
  }
  aether.step(20.0).unwrap();

  let events = aether.world(id).unwrap().events();
  assert!(
    events.iter().any(|e| matches!(
      e,
      Event::RegimeChanged {
        from: RegimeKind::Live,
        to: RegimeKind::Climatology
      }
    )),
    "set_regime emits RegimeChanged, got {events:?}"
  );
  assert!(
    events.iter().any(|e| matches!(
      e,
      Event::TransitionStarted {
        to: RegimeKind::Climatology
      }
    )),
    "begin_transition emits TransitionStarted, got {events:?}"
  );
}

#[test]
fn events_publish_on_the_direct_world_tick_path() {
  // `Aether::step` covers the fused path; here drive `World::tick` directly to
  // cover the non-fused barrier.
  let (mut aether, id, euler) =
    event_world(DiagnosticsPolicy::Observe).unwrap();
  let pool = Pool::default();

  inject_tracer_nan(&mut aether, id, euler);
  world_mut(&mut aether, id).tick(&pool, 20.0).unwrap();
  assert!(
    has_non_finite(aether.world(id).unwrap().events(), euler),
    "World::tick must also publish the event buffer"
  );
}

#[test]
fn a_consumer_reacts_to_instability_by_rolling_back() {
  let (mut aether, id, euler) =
    event_world(DiagnosticsPolicy::Observe).unwrap();
  for _ in 0..2 {
    aether.step(20.0).unwrap();
  }

  // Snapshot a healthy world, then blow it up (soft — Observe keeps the tick Ok).
  let checkpoint = aether.world(id).unwrap().save_checkpoint().unwrap();
  let good_time = aether.world(id).unwrap().sim_time();
  inject_tracer_nan(&mut aether, id, euler);
  aether.step(20.0).unwrap();

  // The consumer polls events and reacts to the instability by rolling back.
  if has_non_finite(aether.world(id).unwrap().events(), euler) {
    world_mut(&mut aether, id)
      .load_checkpoint(&checkpoint)
      .unwrap();
  }

  assert_eq!(
    aether.world(id).unwrap().sim_time(),
    good_time,
    "the world rewound to the pre-blow-up checkpoint"
  );
  aether.step(20.0).expect("and resumes cleanly");
}
