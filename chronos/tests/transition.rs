// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Phase 4: regime-transition continuity. A consumer that switches between
//! reading the climatology aggregate and the live solver must see no
//! discontinuity. Proven here on a directly-prognostic scalar (ocean
//! temperature):
//!
//! - the *seed* ([`copy_field`]) makes the two fields equal at the instant of a
//!   handoff (so the read value does not jump), and
//! - the *nudge* spin-up holds the freshly-seeded live state on the climatology
//!   and releases it over the window, so dynamics ramp in smoothly instead of
//!   shocking the value right after a zoom-in.

use chronos::{
  ClimatologyNudgeStep, TransitionKind, TransitionState, copy_field,
};
use nexus::{
  FieldKey, FieldName, FieldStorage, MeshKey, Nexus, Pleroma, ResourceKey,
  SoaField, Stage, StageContext, WorldConstants, WorldId,
};
use tessera::world_mesh::Tessera;
use utility::domain::CellId;
use utility::error::AetherResult;
use utility::thread::pool::Pool;

const LIVE: FieldKey = FieldKey::new(MeshKey::OCEAN, FieldName::Temperature);
const MEAN: FieldKey =
  FieldKey::new(MeshKey::OCEAN, FieldName::MeanSeaSurfaceTemperature);

const DRIFT: f64 = 2.0;

/// Stand-in for free dynamics: pushes the live field by a fixed drift each tick,
/// so absent any nudge the live value moves by `DRIFT` per step.
struct DriftStage {
  writes: [FieldKey; 1],
}

impl Stage for DriftStage {
  fn name(&self) -> &'static str {
    "drift"
  }
  fn reads(&self) -> &[FieldKey] {
    &[]
  }
  fn writes(&self) -> &[FieldKey] {
    &self.writes
  }
  fn run(&mut self, mut ctx: StageContext<'_>) -> AetherResult<()> {
    let field: &mut SoaField<1> = ctx.world.fields.write(LIVE).unwrap();
    for cell in 0..field.len() {
      let v = field.state(CellId::from(cell))[0];
      field.write(CellId::from(cell), &[v + DRIFT]);
    }
    Ok(())
  }
}

fn cell0(pleroma: &Pleroma, key: FieldKey) -> f64 {
  let field: &SoaField<1> = pleroma.read(key).unwrap();
  field.state(CellId::from(0))[0]
}

#[test]
fn seed_makes_fields_equal_for_continuous_handoff() {
  let mut pleroma = Pleroma::new();
  pleroma.register_field(LIVE, SoaField::<1>::from_fn(3, |_| [270.0]));
  pleroma.register_field(MEAN, SoaField::<1>::from_fn(3, |_| [290.0]));

  // Zoom-out (live → climatology): seed the mean from the live state so the
  // aggregate equals the last live value shown.
  copy_field(&mut pleroma, LIVE, MEAN).unwrap();
  assert!((cell0(&pleroma, MEAN) - cell0(&pleroma, LIVE)).abs() < 1e-12);

  // Zoom-in (climatology → live): seed the live field from the climatology so
  // the live read starts exactly where the climatology left off (no jump).
  let mut p2 = Pleroma::new();
  p2.register_field(LIVE, SoaField::<1>::from_fn(3, |_| [270.0]));
  p2.register_field(MEAN, SoaField::<1>::from_fn(3, |_| [290.0]));
  copy_field(&mut p2, MEAN, LIVE).unwrap();
  assert!((cell0(&p2, LIVE) - cell0(&p2, MEAN)).abs() < 1e-12);
}

#[test]
fn climatology_to_live_spins_up_smoothly() {
  let climatology = 290.0;
  let stale_live = 250.0;

  let mut pleroma = Pleroma::new();
  pleroma.register_field(LIVE, SoaField::<1>::from_fn(1, |_| [stale_live]));
  pleroma.register_field(MEAN, SoaField::<1>::from_fn(1, |_| [climatology]));
  pleroma.register_resource(ResourceKey::ClimateRegime, 0.0_f64);

  let mut nexus = Nexus::new();
  // Free dynamics first, then the nudge corrects toward the climatology.
  nexus.add(DriftStage { writes: [LIVE] });
  nexus
    .add(ClimatologyNudgeStep::new(MeshKey::OCEAN, LIVE, MEAN, 1.0).unwrap());
  let mut compiled = nexus.build(&pleroma).unwrap();

  // Handoff: seed the live field from the climatology. The value a consumer
  // reads does not jump — it was `climatology` (the aggregate) and remains
  // `climatology` (the freshly-seeded live field).
  copy_field(&mut pleroma, MEAN, LIVE).unwrap();
  assert!((cell0(&pleroma, LIVE) - climatology).abs() < 1e-12);

  let pool = Pool::default();
  let dt = 1.0;
  let mut transition =
    TransitionState::new(TransitionKind::ClimatologyToLive, 5.0);

  let mut prev = cell0(&pleroma, LIVE);
  let mut changes = Vec::new();
  loop {
    // Drive the nudge fraction for this tick from the transition progress.
    *pleroma
      .write_resource::<f64>(ResourceKey::ClimateRegime)
      .unwrap() = transition.nudge_fraction();
    compiled
      .tick(
        WorldId(0),
        &Tessera::default(),
        &WorldConstants::default(),
        &mut pleroma,
        &pool,
        dt,
      )
      .unwrap();
    let now = cell0(&pleroma, LIVE);
    changes.push(now - prev);
    prev = now;
    if transition.advance(dt) {
      // One more free step after completion to confirm the nudge is inert.
      *pleroma
        .write_resource::<f64>(ResourceKey::ClimateRegime)
        .unwrap() = transition.nudge_fraction();
      compiled
        .tick(
          WorldId(0),
          &Tessera::default(),
          &WorldConstants::default(),
          &mut pleroma,
          &pool,
          dt,
        )
        .unwrap();
      let now = cell0(&pleroma, LIVE);
      changes.push(now - prev);
      break;
    }
  }

  // The first step barely moves: the nudge holds the seeded live state on the
  // climatology (continuity right after the handoff, no dynamics shock).
  assert!(
    changes[0].abs() < 1e-9,
    "spin-up shocked on first step: {}",
    changes[0]
  );
  // Per-step change ramps up monotonically as the nudge releases...
  for w in changes.windows(2) {
    assert!(w[1] >= w[0] - 1e-9, "spin-up not monotonic: {:?}", changes);
  }
  // ...and once the transition is complete the nudge is inert: the live field
  // evolves freely, moving by the full drift.
  assert!(
    (changes.last().unwrap() - DRIFT).abs() < 1e-9,
    "free evolution after transition should move by DRIFT, got {}",
    changes.last().unwrap()
  );
}
