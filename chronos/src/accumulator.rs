// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use nexus::{
  FieldKey, FieldStorage, MeshKey, SoaField, Stage, StageContext, SubsystemId,
};
use utility::{
  domain::CellId,
  error::{AetherError, AetherResult},
};

use crate::error::ChronosError;

/// Accumulates a slowly-varying time-mean (climatology) of one live scalar
/// field via an exponential moving average toward the live value:
///
/// `meanₙ = meanₙ₋₁ + (liveₙ − meanₙ₋₁) · clamp(dt / τ, 0, 1)`
///
/// with `τ` the climatology timescale. The EMA needs only one stored field per
/// aggregate (unlike a block mean, which needs a running sum + weight) and is
/// naturally slowly-varying. The weight uses the stage's *own* per-call `dt`,
/// so the converged mean is invariant to how finely the subsystem is
/// subcycled by the multirate driver.
///
/// This stage is inert with respect to physics: it only reads a live field and
/// writes its companion mean field. It carries no feedback into the live state.
pub struct ClimatologyAccumulatorStep {
  source: FieldKey,
  mean: FieldKey,
  timescale: f64,
  subsystem: SubsystemId,
  reads: [FieldKey; 2],
  writes: [FieldKey; 1],
}

impl ClimatologyAccumulatorStep {
  pub fn new(
    mesh: MeshKey,
    source: FieldKey,
    mean: FieldKey,
    timescale: f64,
    subsystem: SubsystemId,
  ) -> AetherResult<Self> {
    if source.mesh() != mesh || mean.mesh() != mesh {
      return Err(AetherError::new(ChronosError::FieldMeshMismatch).context(
        format!("mesh {:?}, source {:?}, mean {:?}", mesh, source, mean),
      ));
    }
    if !timescale.is_finite() || timescale <= 0.0 {
      return Err(
        AetherError::new(ChronosError::InvalidTimeScale)
          .context(format!("timescale {}", timescale)),
      );
    }
    Ok(Self {
      source,
      mean,
      timescale,
      subsystem,
      reads: [source, mean],
      writes: [mean],
    })
  }

  pub fn source(&self) -> FieldKey {
    self.source
  }

  pub fn mean(&self) -> FieldKey {
    self.mean
  }
}

impl Stage for ClimatologyAccumulatorStep {
  fn name(&self) -> &'static str {
    "chronos_climatology_accumulator"
  }

  fn reads(&self) -> &[FieldKey] {
    &self.reads
  }

  fn writes(&self) -> &[FieldKey] {
    &self.writes
  }

  fn subsystem(&self) -> SubsystemId {
    self.subsystem
  }

  fn run(&mut self, mut ctx: StageContext<'_>) -> AetherResult<()> {
    let dt = ctx.world.dt;
    if !dt.is_finite() || dt <= 0.0 {
      return Err(
        AetherError::new(ChronosError::InvalidTimeStep)
          .context(format!("dt {}", dt)),
      );
    }
    // EMA weight: a fraction of the way to the live value per unit dt, capped
    // at 1 so a single step never overshoots even when dt ≥ τ.
    let weight = (dt / self.timescale).clamp(0.0, 1.0);

    let live: Vec<f64> = {
      let field: &SoaField<1> =
        ctx.world.fields.read(self.source).ok_or_else(|| {
          AetherError::new(ChronosError::MissingReadField)
            .context(format!("{:?}", self.source))
        })?;
      field.component(0).as_ref().to_vec()
    };

    let mean: &mut SoaField<1> =
      ctx.world.fields.write(self.mean).ok_or_else(|| {
        AetherError::new(ChronosError::MissingWriteField)
          .context(format!("{:?}", self.mean))
      })?;
    if mean.len() != live.len() {
      return Err(AetherError::new(ChronosError::FieldLengthMismatch).context(
        format!("source {} cells, mean {} cells", live.len(), mean.len()),
      ));
    }

    for (cell, &target) in live.iter().enumerate() {
      let id = CellId::from(cell);
      let prev = mean.state(id)[0];
      let updated = prev + (target - prev) * weight;
      if !updated.is_finite() {
        return Err(
          AetherError::new(ChronosError::NonFiniteAggregate)
            .context(format!("cell {} mean {}", cell, updated)),
        );
      }
      mean.write(id, &[updated]);
    }
    Ok(())
  }
}
