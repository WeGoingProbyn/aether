// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::marker::PhantomData;

use continuum::diagnostics::{count_non_finite, integrate_conserved};
use continuum::model::MoistEuler3D;
use nexus::{
  FieldKey, FieldStorage, MeshKey, SoaField, Stage, StageContext, WorldAccess,
};
use tessera::mesh::Mesh;
use utility::{
  diagnostics::{
    ConservationQuantities, DiagnosticsPolicy, FieldReport, WorldDiagnostics,
  },
  domain::{CellId, ResourceKey},
  error::{AetherError, AetherResult},
};

use crate::{error::AerError, init::AtmosphereSpec};

/// Derives scalar diagnostics from the prognostic Euler atmosphere state.
///
/// Reads `[rho, rho_u, rho_v, rho_w, energy]` and writes temperature,
/// pressure, and world-frame velocity components. These fields are intended
/// for coupling, rendering, and diagnostics, not as independent atmosphere
/// prognostic state.
pub struct EulerDiagnosticsStep {
  mesh: MeshKey,
  state: FieldKey,
  temperature: FieldKey,
  pressure: FieldKey,
  velocity_x: FieldKey,
  velocity_y: FieldKey,
  velocity_z: FieldKey,
  humidity: FieldKey,
  reads: [FieldKey; 1],
  writes: [FieldKey; 6],
}

impl EulerDiagnosticsStep {
  pub fn new(
    mesh: MeshKey,
    state: FieldKey,
    temperature: FieldKey,
    pressure: FieldKey,
    velocity_x: FieldKey,
    velocity_y: FieldKey,
    velocity_z: FieldKey,
    humidity: FieldKey,
  ) -> AetherResult<Self> {
    validate_mesh_fields(
      mesh,
      &[
        state,
        temperature,
        pressure,
        velocity_x,
        velocity_y,
        velocity_z,
        humidity,
      ],
    )?;

    Ok(Self {
      mesh,
      state,
      temperature,
      pressure,
      velocity_x,
      velocity_y,
      velocity_z,
      humidity,
      reads: [state],
      writes: [
        temperature,
        pressure,
        velocity_x,
        velocity_y,
        velocity_z,
        humidity,
      ],
    })
  }

  pub fn mesh(&self) -> MeshKey {
    self.mesh
  }

  pub fn state(&self) -> FieldKey {
    self.state
  }

  pub fn temperature(&self) -> FieldKey {
    self.temperature
  }

  pub fn pressure(&self) -> FieldKey {
    self.pressure
  }
}

impl Stage for EulerDiagnosticsStep {
  fn name(&self) -> &'static str {
    "aer_euler_diagnostics"
  }

  fn reads(&self) -> &[FieldKey] {
    &self.reads
  }

  fn writes(&self) -> &[FieldKey] {
    &self.writes
  }

  fn run(&mut self, mut ctx: StageContext<'_>) -> AetherResult<()> {
    let mesh_cell_count = ctx
      .world
      .tessera
      .mesh(self.mesh)
      .ok_or_else(|| {
        AetherError::new(AerError::MissingMesh)
          .context(format!("{:?}", self.mesh))
      })?
      .cell_count();

    let spec = AtmosphereSpec::from_world_constants(ctx.world.constants)?;
    let diagnostics = {
      let state: &SoaField<6> =
        ctx.world.fields.read(self.state).ok_or_else(|| {
          AetherError::new(AerError::MissingReadField)
            .context(format!("{:?}", self.state))
        })?;

      if state.len() != mesh_cell_count {
        return Err(AetherError::new(AerError::FieldLengthMismatch).context(
          format!(
            "state len {}, mesh cell count {}",
            state.len(),
            mesh_cell_count
          ),
        ));
      }

      derive_diagnostics(state, spec.gamma(), spec.gas_constant())?
    };

    write_scalar_field(
      &mut ctx.world.fields,
      self.temperature,
      &diagnostics.temperature,
      mesh_cell_count,
    )?;
    write_scalar_field(
      &mut ctx.world.fields,
      self.pressure,
      &diagnostics.pressure,
      mesh_cell_count,
    )?;
    write_scalar_field(
      &mut ctx.world.fields,
      self.velocity_x,
      &diagnostics.velocity_x,
      mesh_cell_count,
    )?;
    write_scalar_field(
      &mut ctx.world.fields,
      self.velocity_y,
      &diagnostics.velocity_y,
      mesh_cell_count,
    )?;
    write_scalar_field(
      &mut ctx.world.fields,
      self.velocity_z,
      &diagnostics.velocity_z,
      mesh_cell_count,
    )?;
    write_scalar_field(
      &mut ctx.world.fields,
      self.humidity,
      &diagnostics.humidity,
      mesh_cell_count,
    )?;

    Ok(())
  }
}

struct EulerDiagnostics {
  temperature: Vec<f64>,
  pressure: Vec<f64>,
  velocity_x: Vec<f64>,
  velocity_y: Vec<f64>,
  velocity_z: Vec<f64>,
  humidity: Vec<f64>,
}

fn derive_diagnostics(
  state: &SoaField<6>,
  gamma: f64,
  gas_constant: f64,
) -> AetherResult<EulerDiagnostics> {
  let mut temperature = Vec::with_capacity(state.len());
  let mut pressure = Vec::with_capacity(state.len());
  let mut velocity_x = Vec::with_capacity(state.len());
  let mut velocity_y = Vec::with_capacity(state.len());
  let mut velocity_z = Vec::with_capacity(state.len());
  let mut humidity = Vec::with_capacity(state.len());

  for i in 0..state.len() {
    let cell = CellId::from(i);
    let s = state.state(cell);
    let rho = s[0];
    if !rho.is_finite() || rho <= 0.0 {
      return Err(
        AetherError::new(AerError::InvalidAtmosphereState)
          .context(format!("cell {} density {}", i, rho)),
      );
    }

    let inv_rho = 1.0 / rho;
    let u = s[1] * inv_rho;
    let v = s[2] * inv_rho;
    let w = s[3] * inv_rho;
    let kinetic = 0.5 * rho * (u * u + v * v + w * w);
    let p = (gamma - 1.0) * (s[4] - kinetic);
    let t = p / (rho * gas_constant);
    if !p.is_finite() || p <= 0.0 || !t.is_finite() || t <= 0.0 {
      return Err(
        AetherError::new(AerError::InvalidAtmosphereState)
          .context(format!("cell {} pressure {}, temperature {}", i, p, t)),
      );
    }
    // Specific humidity q = ρq / ρ (water-vapour mass fraction).
    let q = s[5] * inv_rho;

    temperature.push(t);
    pressure.push(p);
    velocity_x.push(u);
    velocity_y.push(v);
    velocity_z.push(w);
    humidity.push(q);
  }

  Ok(EulerDiagnostics {
    temperature,
    pressure,
    velocity_x,
    velocity_y,
    velocity_z,
    humidity,
  })
}

fn write_scalar_field(
  fields: &mut WorldAccess<'_>,
  key: FieldKey,
  values: &[f64],
  mesh_cell_count: usize,
) -> AetherResult<()> {
  let field: &mut SoaField<1> = fields.write(key).ok_or_else(|| {
    AetherError::new(AerError::MissingWriteField).context(format!("{:?}", key))
  })?;
  if field.len() != values.len() || field.len() != mesh_cell_count {
    return Err(AetherError::new(AerError::FieldLengthMismatch).context(
      format!(
        "field {:?} len {}, values len {}, mesh cell count {}",
        key,
        field.len(),
        values.len(),
        mesh_cell_count
      ),
    ));
  }

  for (cell, value) in values.iter().enumerate() {
    field.write(CellId::from(cell), &[*value]);
  }
  Ok(())
}

fn validate_mesh_fields(
  mesh: MeshKey,
  fields: &[FieldKey],
) -> AetherResult<()> {
  if fields.iter().all(|field| field.mesh() == mesh) {
    Ok(())
  } else {
    Err(
      AetherError::new(AerError::FieldMeshMismatch)
        .context(format!("mesh {:?}, fields {:?}", mesh, fields)),
    )
  }
}

/// Default conservation-drift tolerance (relative) before [`Warn`] logs. A
/// well-balanced atmosphere should hold its conserved totals to far tighter
/// than this over a run; the value is exposed on the builder so it is never a
/// hidden magic constant.
///
/// [`Warn`]: DiagnosticsPolicy::Warn
pub const DEFAULT_DRIFT_THRESHOLD: f64 = 1.0e-2;

/// In-DAG runtime health monitor for a conserved-law field.
///
/// Each tick it sweeps the field for non-finite (NaN/Inf) cells and
/// volume-integrates the law's declared conserved totals, tracking their drift
/// away from a settled-state baseline. Findings are merged into the
/// [`WorldDiagnostics`] held in [`ResourceKey::Diagnostics`]; the active
/// [`DiagnosticsPolicy`] (read from that same resource) decides whether to
/// merely observe, `warn!`, or fail the tick.
///
/// Generic over the state size `N` and the law type `L` (which contributes
/// only its associated `CONSERVED_QUANTITIES` — no instance is constructed).
/// The atmosphere binds it via [`AtmosphereConservationMonitor`].
///
/// **Scheduling.** The monitor must observe the *post-step* state, so it
/// declares a read on the state `FieldKey` (`reads = [state]`). nexus orders a
/// reader after every writer of the same field, which is what places this
/// stage after the dynamics solver — the dependency is on the field, not on
/// the shared `Diagnostics` resource.
///
/// **Fail semantics.** By the time the monitor runs, the solver has already
/// mutated the state in place. A `Fail` therefore does not roll back: it
/// surfaces the blow-up on the exact tick that produced it by returning `Err`.
/// `World::tick` advances its clocks only after a successful tick, so on `Err`
/// the clocks stay frozen at the last good tick while the field holds the bad
/// values. True freeze-at-last-good would need a state checkpoint and is out of
/// scope here.
pub struct ConservationMonitorStage<const N: usize, L> {
  mesh: MeshKey,
  state: FieldKey,
  /// Settled-state conserved totals captured after warm-up; `None` until then.
  baseline: Option<Vec<f64>>,
  ticks_seen: u64,
  warmup_ticks: u64,
  drift_threshold: f64,
  reads: [FieldKey; 1],
  resource_writes: [ResourceKey; 1],
  /// `fn() -> L` so the marker is unconditionally `Send + Sync` regardless of
  /// `L`; the law is only ever used at the type level.
  _law: PhantomData<fn() -> L>,
}

impl<const N: usize, L> ConservationMonitorStage<N, L>
where
  L: ConservationQuantities<N>,
{
  /// Monitor `state` (living on `mesh`) with the default drift threshold and a
  /// one-tick warm-up before the baseline is captured.
  pub fn new(mesh: MeshKey, state: FieldKey) -> Self {
    Self {
      mesh,
      state,
      baseline: None,
      ticks_seen: 0,
      warmup_ticks: 1,
      drift_threshold: DEFAULT_DRIFT_THRESHOLD,
      reads: [state],
      resource_writes: [ResourceKey::Diagnostics],
      _law: PhantomData,
    }
  }

  /// Set the relative conservation-drift tolerance.
  pub fn with_drift_threshold(mut self, threshold: f64) -> Self {
    self.drift_threshold = threshold;
    self
  }

  /// Set how many ticks to skip before capturing the conservation baseline, so
  /// the reference is a settled state rather than an initial-condition
  /// transient.
  pub fn with_warmup_ticks(mut self, warmup_ticks: u64) -> Self {
    self.warmup_ticks = warmup_ticks;
    self
  }
}

impl<const N: usize, L> Stage for ConservationMonitorStage<N, L>
where
  L: ConservationQuantities<N> + Send + Sync + 'static,
{
  fn name(&self) -> &'static str {
    "aer_conservation_monitor"
  }

  fn reads(&self) -> &[FieldKey] {
    &self.reads
  }

  fn writes(&self) -> &[FieldKey] {
    &[]
  }

  fn resource_writes(&self) -> &[ResourceKey] {
    &self.resource_writes
  }

  fn run(&mut self, mut ctx: StageContext<'_>) -> AetherResult<()> {
    // Read the policy once, up front: a single read means a mid-tick policy
    // change (under any future parallel scheduling) can't be observed
    // inconsistently within this stage.
    let policy = ctx
      .world
      .fields
      .resource::<WorldDiagnostics>(ResourceKey::Diagnostics)
      .map(|d| d.policy)
      .unwrap_or_default();
    if policy == DiagnosticsPolicy::Off {
      return Ok(());
    }

    let mesh: &dyn Mesh<3> = ctx
      .world
      .tessera
      .mesh(self.mesh)
      .ok_or_else(|| {
        AetherError::new(AerError::MissingMesh)
          .context(format!("{:?}", self.mesh))
      })?
      .as_ref();

    let field: &SoaField<N> =
      ctx.world.fields.read(self.state).ok_or_else(|| {
        AetherError::new(AerError::MissingReadField)
          .context(format!("{:?}", self.state))
      })?;

    let non_finite = count_non_finite(field);
    let totals = integrate_conserved::<N, _, _, L>(mesh, field);

    // Baseline / drift. Capture the reference only once we are past warm-up and
    // the state is finite, so it is never a transient or a born-NaN tick.
    self.ticks_seen += 1;
    let mut max_relative_drift = 0.0_f64;
    match &self.baseline {
      Some(baseline) => {
        // Relative drift is only meaningful for conserved quantities with a
        // non-trivial total. Some (e.g. momentum of an atmosphere starting
        // from rest) baseline to ~0, where relative drift against zero is
        // ill-defined and would dominate spuriously — skip those, judged
        // against the largest baseline magnitude as the system scale.
        let scale = baseline.iter().fold(0.0_f64, |m, b| m.max(b.abs()));
        let floor = 1.0e-9 * scale;
        for ((_, total), base) in totals.iter().zip(baseline.iter()) {
          if base.abs() <= floor {
            continue;
          }
          let drift = (total - base).abs() / base.abs();
          max_relative_drift = max_relative_drift.max(drift);
        }
      }
      None if self.ticks_seen > self.warmup_ticks && non_finite == 0 => {
        self.baseline = Some(totals.iter().map(|(_, t)| *t).collect());
      }
      None => {}
    }
    let drift_exceeded =
      self.baseline.is_some() && max_relative_drift > self.drift_threshold;

    let report = FieldReport {
      non_finite_cells: non_finite,
      conserved: totals,
      max_relative_drift,
    };

    // Publish into the shared report (keyed by this field, so independent
    // monitors coexist). The immutable `field` borrow has ended above.
    if let Some(diagnostics) = ctx
      .world
      .fields
      .resource_mut::<WorldDiagnostics>(ResourceKey::Diagnostics)
    {
      diagnostics.merge_field(self.state, report);
    }

    // Enforce the policy. `Fail` only fails on non-finite state (a hard
    // blow-up); drift is a softer signal that warns but never fails.
    if non_finite > 0 {
      match policy {
        DiagnosticsPolicy::Fail => {
          return Err(AetherError::new(AerError::NonFiniteState).context(
            format!("{:?}: {} non-finite cells", self.state, non_finite),
          ));
        }
        DiagnosticsPolicy::Warn => {
          utility::warn!(
            "conservation monitor: {} non-finite cells in {:?}",
            non_finite,
            self.state
          );
        }
        _ => {}
      }
    } else if drift_exceeded
      && matches!(policy, DiagnosticsPolicy::Warn | DiagnosticsPolicy::Fail)
    {
      utility::warn!(
        "conservation monitor: {:?} drift {:.3e} exceeds threshold {:.3e}",
        self.state,
        max_relative_drift,
        self.drift_threshold
      );
    }

    Ok(())
  }
}

/// The atmosphere binding of [`ConservationMonitorStage`]: the moist-Euler
/// state (`N = 6`, [`MoistEuler3D`]'s six conserved components).
pub type AtmosphereConservationMonitor =
  ConservationMonitorStage<6, MoistEuler3D>;

#[cfg(test)]
mod tests {
  use std::sync::Arc;

  use nexus::{
    AtmosphereConstants, FieldName, Nexus, Pleroma, WorldConstants, WorldId,
  };
  use tessera::{
    geometry::IdentityMap,
    mesh::{Mesh, StructuredBlock},
    world_mesh::Tessera,
  };
  use utility::thread::pool::Pool;

  use super::*;

  const STATE: FieldKey =
    FieldKey::new(MeshKey::ATMOSPHERE, FieldName::EulerState);
  const TEMPERATURE: FieldKey =
    FieldKey::new(MeshKey::ATMOSPHERE, FieldName::Temperature);
  const PRESSURE: FieldKey =
    FieldKey::new(MeshKey::ATMOSPHERE, FieldName::Pressure);
  const VELOCITY_X: FieldKey =
    FieldKey::new(MeshKey::ATMOSPHERE, FieldName::VelocityX);
  const VELOCITY_Y: FieldKey =
    FieldKey::new(MeshKey::ATMOSPHERE, FieldName::VelocityY);
  const VELOCITY_Z: FieldKey =
    FieldKey::new(MeshKey::ATMOSPHERE, FieldName::VelocityZ);
  const HUMIDITY: FieldKey =
    FieldKey::new(MeshKey::ATMOSPHERE, FieldName::Humidity);

  fn constants() -> WorldConstants {
    WorldConstants {
      mass: 1.0,
      radius: 1.0,
      surface_gravity: 0.0,
      atmosphere: Some(AtmosphereConstants {
        reference_temperature: 300.0,
        reference_pressure: 120_000.0,
        gamma: 1.4,
        gas_constant: 300.0,
        molar_mass: 1.0,
        albedo: None,
        angular_velocity: 0.0,
        axial_tilt: 0.0,
      }),
      radiation: None,
    }
  }

  #[test]
  fn diagnostics_derive_temperature_pressure_and_velocity() {
    let mut tessera = Tessera::new();
    let mesh = Arc::new(StructuredBlock::uniform(
      [0.0; 3].into(),
      [1.0; 3],
      [1, 1, 1],
      Box::new(IdentityMap::<3>),
    ));
    let mesh_for_registry: Arc<dyn Mesh<3>> = mesh;
    tessera.register_mesh(MeshKey::ATMOSPHERE, mesh_for_registry);

    let rho = 2.0;
    let velocity = [3.0, 4.0, 5.0];
    let pressure = 120_000.0;
    let gamma = 1.4;
    let kinetic = 0.5
      * rho
      * (velocity[0] * velocity[0]
        + velocity[1] * velocity[1]
        + velocity[2] * velocity[2]);
    let energy = pressure / (gamma - 1.0) + kinetic;

    let humidity_q = 0.012;
    let mut pleroma = Pleroma::new();
    pleroma.register_field(
      STATE,
      SoaField::<6>::from_fn(1, |_| {
        [
          rho,
          rho * velocity[0],
          rho * velocity[1],
          rho * velocity[2],
          energy,
          rho * humidity_q,
        ]
      }),
    );
    pleroma.register_field(TEMPERATURE, SoaField::<1>::zeros(1));
    pleroma.register_field(PRESSURE, SoaField::<1>::zeros(1));
    pleroma.register_field(VELOCITY_X, SoaField::<1>::zeros(1));
    pleroma.register_field(VELOCITY_Y, SoaField::<1>::zeros(1));
    pleroma.register_field(VELOCITY_Z, SoaField::<1>::zeros(1));
    pleroma.register_field(HUMIDITY, SoaField::<1>::zeros(1));

    let mut nexus = Nexus::new();
    nexus.add(
      EulerDiagnosticsStep::new(
        MeshKey::ATMOSPHERE,
        STATE,
        TEMPERATURE,
        PRESSURE,
        VELOCITY_X,
        VELOCITY_Y,
        VELOCITY_Z,
        HUMIDITY,
      )
      .unwrap(),
    );
    let mut compiled = nexus.build(&pleroma).unwrap();
    compiled
      .tick(
        WorldId(0),
        &tessera,
        &constants(),
        &mut pleroma,
        &Pool::default(),
        1.0,
      )
      .unwrap();

    let temperature: &SoaField<1> = pleroma.read(TEMPERATURE).unwrap();
    let pressure_field: &SoaField<1> = pleroma.read(PRESSURE).unwrap();
    let u: &SoaField<1> = pleroma.read(VELOCITY_X).unwrap();
    let v: &SoaField<1> = pleroma.read(VELOCITY_Y).unwrap();
    let w: &SoaField<1> = pleroma.read(VELOCITY_Z).unwrap();
    let q: &SoaField<1> = pleroma.read(HUMIDITY).unwrap();

    assert_eq!(temperature.state(CellId::from(0))[0], 200.0);
    assert_eq!(pressure_field.state(CellId::from(0))[0], pressure);
    assert_eq!(u.state(CellId::from(0))[0], velocity[0]);
    assert_eq!(v.state(CellId::from(0))[0], velocity[1]);
    assert_eq!(w.state(CellId::from(0))[0], velocity[2]);
    assert_eq!(q.state(CellId::from(0))[0], humidity_q);
  }
}
