// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Regime-transition continuity: the mechanism that lets a consumer switch
//! between reading a climatology aggregate and the live solver without seeing a
//! discontinuity.
//!
//! Two pieces:
//!
//! - [`copy_field`] — the *seed* used at the instant of a handoff. On zoom-out
//!   (live→climatology) the climatology mean is seeded from the current live
//!   state, so the aggregate is continuous with the last live value shown. On
//!   zoom-in (climatology→live) the live field is seeded from the climatology,
//!   so the live read starts exactly where the climatology left off.
//! - [`ClimatologyNudgeStep`] — a *spin-up* relaxation on the live side. After
//!   a zoom-in seed, free dynamics would yank the (climatologically-initialised)
//!   live state around; the nudge holds it on the climatology at first and
//!   releases it over the transition window, so the spin-up is smooth rather
//!   than a shock. It reads the current relaxation fraction from
//!   [`ResourceKey::ClimateRegime`] (absent ⇒ inert), so the stage is harmless
//!   to leave in a world that is not transitioning.
//!
//! Scope: this is the safe, directly-prognostic-scalar form (e.g. ocean
//! temperature). Spinning up the conserved Euler state via a derived quantity
//! (temperature → internal energy, holding density/momentum) is a documented
//! follow-on so it cannot disturb the well-balanced hydrostatic state.

use nexus::{
  FieldKey, FieldStorage, MeshKey, Pleroma, ResourceKey, SoaField, Stage,
  StageContext,
};
use utility::{
  domain::CellId,
  error::{AetherError, AetherResult},
};

use crate::error::ChronosError;

/// Copy a scalar field cell-for-cell into another on the same mesh. Used to seed
/// a target (climatology mean or live field) from a source at a handoff so the
/// two are equal at the instant the consumer switches which one it reads.
pub fn copy_field(
  pleroma: &mut Pleroma,
  from: FieldKey,
  to: FieldKey,
) -> AetherResult<()> {
  let values: Vec<f64> = {
    let src: &SoaField<1> = pleroma.read(from).ok_or_else(|| {
      AetherError::new(ChronosError::MissingReadField)
        .context(format!("{:?}", from))
    })?;
    (0..src.len())
      .map(|i| src.state(CellId::from(i))[0])
      .collect()
  };
  let dst: &mut SoaField<1> = pleroma.write(to).ok_or_else(|| {
    AetherError::new(ChronosError::MissingWriteField)
      .context(format!("{:?}", to))
  })?;
  if dst.len() != values.len() {
    return Err(AetherError::new(ChronosError::FieldLengthMismatch).context(
      format!("from {} cells, to {} cells", values.len(), dst.len()),
    ));
  }
  for (cell, value) in values.into_iter().enumerate() {
    dst.write(CellId::from(cell), &[value]);
  }
  Ok(())
}

/// Relaxes a live scalar field toward its climatology mean during a transition:
/// `live += (mean − live) · base_strength · fraction`, where `fraction` is read
/// from [`ResourceKey::ClimateRegime`] each tick (a missing resource ⇒ fraction
/// 0 ⇒ no-op). The driver ramps the fraction from 1 down to 0 over the window,
/// so the live state is held on the climatology right after a zoom-in seed and
/// released to free evolution as the transition completes.
pub struct ClimatologyNudgeStep {
  source: FieldKey,
  mean: FieldKey,
  base_strength: f64,
  reads: [FieldKey; 2],
  writes: [FieldKey; 1],
  resource_reads: [ResourceKey; 1],
}

impl ClimatologyNudgeStep {
  pub fn new(
    mesh: MeshKey,
    source: FieldKey,
    mean: FieldKey,
    base_strength: f64,
  ) -> AetherResult<Self> {
    if source.mesh() != mesh || mean.mesh() != mesh {
      return Err(AetherError::new(ChronosError::FieldMeshMismatch).context(
        format!("mesh {:?}, source {:?}, mean {:?}", mesh, source, mean),
      ));
    }
    Ok(Self {
      source,
      mean,
      base_strength,
      reads: [mean, source],
      writes: [source],
      resource_reads: [ResourceKey::ClimateRegime],
    })
  }
}

impl Stage for ClimatologyNudgeStep {
  fn name(&self) -> &'static str {
    "chronos_climatology_nudge"
  }

  fn reads(&self) -> &[FieldKey] {
    &self.reads
  }

  fn writes(&self) -> &[FieldKey] {
    &self.writes
  }

  fn resource_reads(&self) -> &[ResourceKey] {
    &self.resource_reads
  }

  fn run(&mut self, mut ctx: StageContext<'_>) -> AetherResult<()> {
    // No transition in progress (resource absent or zero) ⇒ inert.
    let fraction = ctx
      .world
      .fields
      .resource::<f64>(ResourceKey::ClimateRegime)
      .copied()
      .unwrap_or(0.0);
    let coeff = (self.base_strength * fraction).clamp(0.0, 1.0);
    if coeff == 0.0 {
      return Ok(());
    }

    let mean: Vec<f64> = {
      let field: &SoaField<1> =
        ctx.world.fields.read(self.mean).ok_or_else(|| {
          AetherError::new(ChronosError::MissingReadField)
            .context(format!("{:?}", self.mean))
        })?;
      field.component(0).as_ref().to_vec()
    };

    let live: &mut SoaField<1> =
      ctx.world.fields.write(self.source).ok_or_else(|| {
        AetherError::new(ChronosError::MissingWriteField)
          .context(format!("{:?}", self.source))
      })?;
    if live.len() != mean.len() {
      return Err(AetherError::new(ChronosError::FieldLengthMismatch));
    }

    for (cell, &target) in mean.iter().enumerate() {
      let id = CellId::from(cell);
      let prev = live.state(id)[0];
      let updated = prev + (target - prev) * coeff;
      if !updated.is_finite() {
        return Err(
          AetherError::new(ChronosError::NonFiniteAggregate)
            .context(format!("cell {} live {}", cell, updated)),
        );
      }
      live.write(id, &[updated]);
    }
    Ok(())
  }
}
