// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Phase 1: inert climatology aggregation + convergence instrumentation.
//!
//! The accumulator is an EMA toward the live value. These assert that (a) it
//! converges to the mean of a stationary source, (b) the converged value is
//! invariant to how finely the subsystem is subcycled by the multirate driver
//! (the weight uses per-call dt), and (c) the convergence instrumentation
//! Phase 3 relies on actually measures settling.

use chronos::{
  ClimateQuantity, ClimatologyModel, ConvergenceSample, RegimeConfig, residual,
  settling_time, suggest_burst_steps,
};
use nexus::{
  FieldKey, FieldName, FieldStorage, MeshKey, Nexus, Pleroma, SoaField, Stage,
  StageContext, SubsystemId, WorldConstants, WorldId,
};
use tessera::world_mesh::Tessera;
use utility::domain::CellId;
use utility::error::AetherResult;
use utility::thread::pool::Pool;

const CLIMATE: SubsystemId = SubsystemId(1);

const TEMP: FieldKey =
  FieldKey::new(MeshKey::ATMOSPHERE, FieldName::Temperature);
const MEAN_TEMP: FieldKey =
  FieldKey::new(MeshKey::ATMOSPHERE, FieldName::MeanTemperature);

/// Holds a live field constant at `value` so we can study pure convergence.
struct ConstantSource {
  field: FieldKey,
  value: f64,
  writes: [FieldKey; 1],
}

impl Stage for ConstantSource {
  fn name(&self) -> &'static str {
    "constant_source"
  }
  fn reads(&self) -> &[FieldKey] {
    &[]
  }
  fn writes(&self) -> &[FieldKey] {
    &self.writes
  }
  fn run(&mut self, mut ctx: StageContext<'_>) -> AetherResult<()> {
    let field: &mut SoaField<1> = ctx.world.fields.write(self.field).unwrap();
    for cell in 0..field.len() {
      field.write(CellId::from(cell), &[self.value]);
    }
    Ok(())
  }
}

fn mean_field_values(pleroma: &Pleroma, key: FieldKey) -> Vec<f64> {
  let field: &SoaField<1> = pleroma.read(key).unwrap();
  (0..field.len())
    .map(|i| field.state(CellId::from(i))[0])
    .collect()
}

#[test]
fn ema_converges_to_stationary_source_mean() {
  let cells = 4;
  let source_value = 300.0;
  let initial_mean = 280.0;
  let tau = 10.0;

  let mut pleroma = Pleroma::new();
  pleroma
    .register_field(TEMP, SoaField::<1>::from_fn(cells, |_| [source_value]));
  // Mean deliberately starts away from the source so convergence is visible.
  pleroma.register_field(
    MEAN_TEMP,
    SoaField::<1>::from_fn(cells, |_| [initial_mean]),
  );

  let model = ClimatologyModel::new(MeshKey::ATMOSPHERE)
    .with_quantity(ClimateQuantity::Temperature)
    .with_timescale(tau)
    .with_subsystem(CLIMATE);

  let mut nexus = Nexus::new();
  nexus.add(ConstantSource {
    field: TEMP,
    value: source_value,
    writes: [TEMP],
  });
  model.add_stages(&mut nexus).unwrap();
  let mut compiled = nexus.build(&pleroma).unwrap();

  let pool = Pool::default();
  for _ in 0..400 {
    compiled
      .tick(
        WorldId(0),
        &Tessera::default(),
        &WorldConstants::default(),
        &mut pleroma,
        &pool,
        1.0,
      )
      .unwrap();
  }

  for v in mean_field_values(&pleroma, MEAN_TEMP) {
    assert!(
      (v - source_value).abs() < 1e-3,
      "mean {v} did not converge to {source_value}"
    );
  }
}

/// The EMA weight is `dt/τ` using the accumulator's *own* per-call dt, so
/// subcycling the subsystem does not rescale the climatology timescale: both a
/// coarse and a finely-subcycled run track the same analytic e-folding
/// `source − excursion·e^(−T/τ)`, and refining the substep brings the discrete
/// EMA *closer* to that continuous limit (rather than to some different fixed
/// point — which is what a dt-independent weight would wrongly do).
#[test]
fn substep_refinement_tracks_continuous_efolding() {
  let source = 300.0;
  let initial = 280.0;
  let tau = 50.0;
  let outer_dt = 4.0;
  let outer_steps = 50;

  let run = |fast_cadence: f64| -> f64 {
    let mut pleroma = Pleroma::new();
    pleroma.register_field(TEMP, SoaField::<1>::from_fn(1, |_| [source]));
    pleroma.register_field(MEAN_TEMP, SoaField::<1>::from_fn(1, |_| [initial]));

    let model = ClimatologyModel::new(MeshKey::ATMOSPHERE)
      .with_quantity(ClimateQuantity::Temperature)
      .with_timescale(tau)
      .with_subsystem(CLIMATE);

    let mut nexus = Nexus::new();
    nexus.add(ConstantSource {
      field: TEMP,
      value: source,
      writes: [TEMP],
    });
    model.add_stages(&mut nexus).unwrap();
    // Subcycle the climate subsystem at `fast_cadence` within each outer dt.
    nexus.set_subsystem_cadence(CLIMATE, fast_cadence);
    let mut compiled = nexus.build(&pleroma).unwrap();

    let pool = Pool::default();
    for _ in 0..outer_steps {
      compiled
        .tick(
          WorldId(0),
          &Tessera::default(),
          &WorldConstants::default(),
          &mut pleroma,
          &pool,
          outer_dt,
        )
        .unwrap();
    }
    mean_field_values(&pleroma, MEAN_TEMP)[0]
  };

  let total_time = outer_dt * outer_steps as f64;
  let analytic = source - (source - initial) * (-total_time / tau).exp();

  // One inner step per outer dt vs four inner steps per outer dt.
  let coarse = run(outer_dt);
  let fine = run(1.0);

  // Both stay near the continuous e-folding (timescale respected, not rescaled).
  assert!(
    (coarse - analytic).abs() < 0.1,
    "coarse {coarse} vs {analytic}"
  );
  assert!((fine - analytic).abs() < 0.1, "fine {fine} vs {analytic}");
  // Finer subcycling is a more accurate integrator of the same e-folding.
  assert!(
    (fine - analytic).abs() < (coarse - analytic).abs(),
    "refinement did not improve accuracy: coarse {coarse}, fine {fine}, \
     analytic {analytic}"
  );
}

/// The convergence instrumentation must actually measure settling: as a
/// stationary source is averaged the relative residual decreases toward zero,
/// and settling_time / suggest_burst_steps report a finite, sensible value.
#[test]
fn convergence_diagnostic_measures_settling() {
  let cells = 3;
  let dt = 1.0;
  let tau = 20.0;

  let mut pleroma = Pleroma::new();
  pleroma.register_field(TEMP, SoaField::<1>::from_fn(cells, |_| [300.0]));
  pleroma.register_field(MEAN_TEMP, SoaField::<1>::from_fn(cells, |_| [280.0]));

  let model = ClimatologyModel::new(MeshKey::ATMOSPHERE)
    .with_quantity(ClimateQuantity::Temperature)
    .with_timescale(tau);
  let mut nexus = Nexus::new();
  nexus.add(ConstantSource {
    field: TEMP,
    value: 300.0,
    writes: [TEMP],
  });
  model.add_stages(&mut nexus).unwrap();
  let mut compiled = nexus.build(&pleroma).unwrap();

  let pool = Pool::default();
  let mut trace: Vec<ConvergenceSample> = Vec::new();
  let mut elapsed = 0.0;
  for step in 1..=300 {
    let prev = mean_field_values(&pleroma, MEAN_TEMP);
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
    let curr = mean_field_values(&pleroma, MEAN_TEMP);
    elapsed += dt;
    trace.push(residual(&prev, &curr, elapsed, step));
  }

  // Residual shrinks overall (first tick moves most; last tick barely moves).
  assert!(
    trace.first().unwrap().residual_rel > trace.last().unwrap().residual_rel,
    "residual did not shrink"
  );
  assert!(
    trace.last().unwrap().residual_rel < 1e-4,
    "aggregate never settled: {}",
    trace.last().unwrap().residual_rel
  );

  let eps = 1e-3;
  let settle = settling_time(&trace, eps).expect("should settle within trace");
  assert!(settle > 0.0 && settle <= 300.0, "settling time {settle}");
  let burst = suggest_burst_steps(&trace, eps, dt);
  assert_eq!(burst, settle.ceil() as usize);

  // 1F → 3A: the measured trace, not a magic constant, sizes the regime burst.
  // The chosen burst span must cover the settling time so the climatology is
  // developed before the regime holds the solver.
  let config = RegimeConfig::new(burst, dt);
  assert!(
    config.burst_span() >= settle,
    "burst span {} does not cover settling time {settle}",
    config.burst_span()
  );
}
