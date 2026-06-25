// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Phase 3: the climatology-stepping regime. A live world integrates by the
//! full game_dt; a climatology world runs only a bounded burst and then *holds*
//! the solver while the game clock jumps ahead — so a large game_dt costs only
//! the burst. The game clock and integrated sim time diverge by design.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use aether::core::{Aether, World};
use chronos::{
  ClimatologyNudgeStep, Regime, RegimeConfig, TransitionKind, TransitionState,
  copy_field,
};
use cosmo::factory;
use nexus::{
  FieldKey, FieldName, FieldStorage, MeshKey, Nexus, Pleroma, ResourceKey,
  SoaField, Stage, StageContext,
};
use tessera::world_mesh::Tessera;
use utility::domain::{CellId, WorldId};
use utility::error::AetherResult;
use utility::thread::pool::Pool;

const FIELD: FieldKey =
  FieldKey::new(MeshKey::ATMOSPHERE, FieldName::Temperature);

/// Integrates its own per-step dt into cell 0 and counts how often it ran, so a
/// test can read back exactly how much simulation time was integrated.
struct DtAccumulator {
  writes: [FieldKey; 1],
  runs: Arc<AtomicUsize>,
}

impl Stage for DtAccumulator {
  fn name(&self) -> &'static str {
    "dt_accumulator"
  }
  fn reads(&self) -> &[FieldKey] {
    &[]
  }
  fn writes(&self) -> &[FieldKey] {
    &self.writes
  }
  fn run(&mut self, mut ctx: StageContext<'_>) -> AetherResult<()> {
    self.runs.fetch_add(1, Ordering::Relaxed);
    let field: &mut SoaField<1> = ctx.world.fields.write(FIELD).unwrap();
    let v = field.state(CellId::from(0))[0];
    field.write(CellId::from(0), &[v + ctx.world.dt]);
    Ok(())
  }
}

fn accumulator_world(runs: &Arc<AtomicUsize>) -> World {
  let mut pleroma = Pleroma::new();
  pleroma.register_field(FIELD, SoaField::<1>::zeros(1));
  let mut nexus = Nexus::new();
  nexus.add(DtAccumulator {
    writes: [FIELD],
    runs: Arc::clone(runs),
  });
  let compiled = nexus.build(&pleroma).unwrap();
  World::new(
    WorldId(0),
    factory::earth(),
    None,
    Tessera::default(),
    pleroma,
    compiled,
  )
}

fn integrated_value(aether: &Aether) -> f64 {
  let field: &SoaField<1> = aether
    .world(WorldId(0))
    .unwrap()
    .pleroma()
    .read(FIELD)
    .unwrap();
  field.state(CellId::from(0))[0]
}

#[test]
fn live_regime_integrates_full_game_dt() {
  let runs = Arc::new(AtomicUsize::new(0));
  let mut aether = Aether::from_worlds(
    HashMap::from([(WorldId(0), accumulator_world(&runs))]),
    Pool::default(),
  );

  aether.advance(5.0).unwrap();

  let world = aether.world(WorldId(0)).unwrap();
  assert_eq!(world.regime(), Regime::Live);
  assert_eq!(world.game_clock(), 5.0);
  assert_eq!(world.sim_time(), 5.0);
  assert!((integrated_value(&aether) - 5.0).abs() < 1e-12);
}

#[test]
fn climatology_regime_bursts_then_holds() {
  let runs = Arc::new(AtomicUsize::new(0));
  let mut aether = Aether::from_worlds(
    HashMap::from([(WorldId(0), accumulator_world(&runs))]),
    Pool::default(),
  );

  {
    let world = aether
      .system_mut(utility::domain::SystemId(0))
      .unwrap()
      .world_mut(WorldId(0))
      .unwrap();
    world.set_regime(Regime::Climatology);
    world.set_regime_config(RegimeConfig::new(3, 2.0));
  }

  // Advance a game span far larger than the burst.
  let game_dt = 1000.0;
  aether.advance(game_dt).unwrap();

  let world = aether.world(WorldId(0)).unwrap();
  // The game clock advanced the full span...
  assert_eq!(world.game_clock(), game_dt);
  // ...but only the burst was integrated (cost bounded): 3 steps × 2.0 s.
  assert_eq!(runs.load(Ordering::Relaxed), 3);
  assert!(
    (world.sim_time() - 6.0).abs() < 1e-12,
    "sim_time {} should equal burst span 6.0",
    world.sim_time()
  );
  assert!(
    (integrated_value(&aether) - 6.0).abs() < 1e-12,
    "solver integrated more than the burst"
  );
  // The held span is the divergence between game and sim clocks — the saving.
  assert!((world.game_clock() - world.sim_time() - 994.0).abs() < 1e-12);
}

/// A short game_dt that the burst itself covers must not double-count: the burst
/// is capped at game_dt, and there is no negative held span.
#[test]
fn climatology_burst_is_capped_at_game_dt() {
  let runs = Arc::new(AtomicUsize::new(0));
  let mut aether = Aether::from_worlds(
    HashMap::from([(WorldId(0), accumulator_world(&runs))]),
    Pool::default(),
  );
  {
    let world = aether
      .system_mut(utility::domain::SystemId(0))
      .unwrap()
      .world_mut(WorldId(0))
      .unwrap();
    world.set_regime(Regime::Climatology);
    world.set_regime_config(RegimeConfig::new(10, 2.0));
  }

  // game_dt = 5.0 < burst span (10 × 2.0 = 20): burst should stop at 5.0.
  aether.advance(5.0).unwrap();

  let world = aether.world(WorldId(0)).unwrap();
  assert_eq!(world.game_clock(), 5.0);
  assert!((world.sim_time() - 5.0).abs() < 1e-12);
  assert!((integrated_value(&aether) - 5.0).abs() < 1e-12);
}

const OCEAN_T: FieldKey = FieldKey::new(MeshKey::OCEAN, FieldName::Temperature);
const OCEAN_MEAN: FieldKey =
  FieldKey::new(MeshKey::OCEAN, FieldName::MeanSeaSurfaceTemperature);
const DRIFT: f64 = 2.0;

/// Free dynamics stand-in on the ocean temperature field.
struct OceanDrift {
  writes: [FieldKey; 1],
}

impl Stage for OceanDrift {
  fn name(&self) -> &'static str {
    "ocean_drift"
  }
  fn reads(&self) -> &[FieldKey] {
    &[]
  }
  fn writes(&self) -> &[FieldKey] {
    &self.writes
  }
  fn run(&mut self, mut ctx: StageContext<'_>) -> AetherResult<()> {
    let field: &mut SoaField<1> = ctx.world.fields.write(OCEAN_T).unwrap();
    let v = field.state(CellId::from(0))[0];
    field.write(CellId::from(0), &[v + DRIFT]);
    Ok(())
  }
}

/// A climatology→live handoff driven through `World::advance` must spin the live
/// state up smoothly (nudge held at first, released over the window) and clear
/// itself, after which the nudge is inert and the field evolves freely.
#[test]
fn transition_spins_up_through_advance() {
  let climatology = 290.0;
  let mut pleroma = Pleroma::new();
  pleroma.register_field(OCEAN_T, SoaField::<1>::from_fn(1, |_| [250.0]));
  pleroma
    .register_field(OCEAN_MEAN, SoaField::<1>::from_fn(1, |_| [climatology]));
  pleroma.register_resource(ResourceKey::ClimateRegime, 0.0_f64);

  let mut nexus = Nexus::new();
  nexus.add(OceanDrift { writes: [OCEAN_T] });
  nexus.add(
    ClimatologyNudgeStep::new(MeshKey::OCEAN, OCEAN_T, OCEAN_MEAN, 1.0)
      .unwrap(),
  );
  let compiled = nexus.build(&pleroma).unwrap();
  // Seed the live field from the climatology: the read value is continuous.
  copy_field(&mut pleroma, OCEAN_MEAN, OCEAN_T).unwrap();

  let world = World::new(
    WorldId(0),
    factory::earth(),
    None,
    Tessera::default(),
    pleroma,
    compiled,
  );
  let mut aether =
    Aether::from_worlds(HashMap::from([(WorldId(0), world)]), Pool::default());

  aether
    .system_mut(utility::domain::SystemId(0))
    .unwrap()
    .world_mut(WorldId(0))
    .unwrap()
    .begin_transition(TransitionState::new(
      TransitionKind::ClimatologyToLive,
      5.0,
    ));

  let read = |a: &Aether| -> f64 {
    let f: &SoaField<1> = a
      .world(WorldId(0))
      .unwrap()
      .pleroma()
      .read(OCEAN_T)
      .unwrap();
    f.state(CellId::from(0))[0]
  };

  // First advance: nudge held on the climatology, barely moves (no shock).
  let mut prev = read(&aether);
  aether.advance(1.0).unwrap();
  let first = read(&aether) - prev;
  assert!(first.abs() < 1e-9, "spin-up shocked on first step: {first}");

  // Run out the window.
  for _ in 0..4 {
    aether.advance(1.0).unwrap();
  }
  assert!(
    aether.world(WorldId(0)).unwrap().transition().is_none(),
    "transition should have completed"
  );

  // Nudge now inert: a further advance moves by the full drift.
  prev = read(&aether);
  aether.advance(1.0).unwrap();
  assert!(
    (read(&aether) - prev - DRIFT).abs() < 1e-9,
    "free evolution after transition should move by DRIFT"
  );
}
