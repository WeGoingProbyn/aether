// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Phase 0B: the multirate driver. A fast subsystem (small cadence) must
//! subcycle `ceil(outer_dt / cadence)` inner steps within one outer tick,
//! while a slow subsystem (no cadence) steps exactly once. Crucially the
//! *integral* is preserved: each inner step advances by `outer_dt / n`, so
//! a stage that integrates its own dt accumulates exactly `outer_dt` over
//! the outer step regardless of how finely it is subcycled.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use nexus::{
  FieldKey, FieldName, FieldStorage, MeshKey, Nexus, Pleroma, SoaField, Stage,
  StageContext, SubsystemId, WorldConstants, WorldId,
};
use tessera::world_mesh::Tessera;
use utility::domain::CellId;
use utility::error::AetherResult;
use utility::thread::pool::Pool;

const FAST: SubsystemId = SubsystemId(1);
const SLOW: SubsystemId = SubsystemId(2);

const FAST_FIELD: FieldKey =
  FieldKey::new(MeshKey::ATMOSPHERE, FieldName::Temperature);
const SLOW_FIELD: FieldKey =
  FieldKey::new(MeshKey::OCEAN, FieldName::Temperature);

/// Integrates its own per-step dt into cell 0 of `field` and counts runs.
struct DtIntegrator {
  field: FieldKey,
  subsystem: SubsystemId,
  writes: [FieldKey; 1],
  runs: Arc<AtomicUsize>,
}

impl Stage for DtIntegrator {
  fn name(&self) -> &'static str {
    "dt_integrator"
  }
  fn reads(&self) -> &[FieldKey] {
    &[]
  }
  fn writes(&self) -> &[FieldKey] {
    &self.writes
  }
  fn subsystem(&self) -> SubsystemId {
    self.subsystem
  }
  fn run(&mut self, mut ctx: StageContext<'_>) -> AetherResult<()> {
    self.runs.fetch_add(1, Ordering::Relaxed);
    let field: &mut SoaField<1> = ctx.world.fields.write(self.field).unwrap();
    let v = field.state(CellId::from(0))[0];
    field.write(CellId::from(0), &[v + ctx.world.dt]);
    Ok(())
  }
}

fn integrator(
  field: FieldKey,
  subsystem: SubsystemId,
  runs: &Arc<AtomicUsize>,
) -> DtIntegrator {
  DtIntegrator {
    field,
    subsystem,
    writes: [field],
    runs: Arc::clone(runs),
  }
}

#[test]
fn fast_subsystem_subcycles_slow_steps_once_and_integral_is_preserved() {
  let outer_dt = 8.0;
  let fast_cadence = 2.0; // → ceil(8/2) = 4 inner steps
  let fast_runs = Arc::new(AtomicUsize::new(0));
  let slow_runs = Arc::new(AtomicUsize::new(0));

  let mut pleroma = Pleroma::new();
  pleroma.register_field(FAST_FIELD, SoaField::<1>::zeros(1));
  pleroma.register_field(SLOW_FIELD, SoaField::<1>::zeros(1));

  let mut nexus = Nexus::new();
  nexus.add(integrator(FAST_FIELD, FAST, &fast_runs));
  nexus.add(integrator(SLOW_FIELD, SLOW, &slow_runs));
  nexus.set_subsystem_cadence(FAST, fast_cadence);
  // SLOW has no cadence → steps once.

  let mut compiled = nexus.build(&pleroma).unwrap();
  assert!(compiled.is_multirate());

  compiled
    .tick(
      WorldId(0),
      &Tessera::default(),
      &WorldConstants::default(),
      &mut pleroma,
      &Pool::default(),
      outer_dt,
    )
    .unwrap();

  // Substep counts: fast ran 4×, slow ran once.
  assert_eq!(fast_runs.load(Ordering::Relaxed), 4);
  assert_eq!(slow_runs.load(Ordering::Relaxed), 1);

  // Integral preserved: both fields advanced by exactly outer_dt.
  let fast: &SoaField<1> = pleroma.read(FAST_FIELD).unwrap();
  let slow: &SoaField<1> = pleroma.read(SLOW_FIELD).unwrap();
  assert!((fast.state(CellId::from(0))[0] - outer_dt).abs() < 1e-12);
  assert!((slow.state(CellId::from(0))[0] - outer_dt).abs() < 1e-12);
}

#[test]
fn equal_cadences_reduce_to_single_step_each() {
  // A subsystem whose cadence equals (or exceeds) the outer dt steps once.
  let outer_dt = 5.0;
  let runs = Arc::new(AtomicUsize::new(0));

  let mut pleroma = Pleroma::new();
  pleroma.register_field(FAST_FIELD, SoaField::<1>::zeros(1));

  let mut nexus = Nexus::new();
  nexus.add(integrator(FAST_FIELD, FAST, &runs));
  nexus.set_subsystem_cadence(FAST, outer_dt); // cadence == outer dt

  let mut compiled = nexus.build(&pleroma).unwrap();
  compiled
    .tick(
      WorldId(0),
      &Tessera::default(),
      &WorldConstants::default(),
      &mut pleroma,
      &Pool::default(),
      outer_dt,
    )
    .unwrap();

  assert_eq!(runs.load(Ordering::Relaxed), 1);
  let fast: &SoaField<1> = pleroma.read(FAST_FIELD).unwrap();
  assert!((fast.state(CellId::from(0))[0] - outer_dt).abs() < 1e-12);
}
