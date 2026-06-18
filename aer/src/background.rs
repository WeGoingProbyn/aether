// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use continuum::model::{ConservationLaw, MoistEuler3D};
use tessera::geometry::CellMetrics;
use utility::domain::{CellId, Point};

/// MoistEuler3D with a fixed per-cell source correction.
///
/// Aer uses this to cancel the discrete residual of a chosen background state.
/// This is not a strictly well-balanced finite-volume formulation; it only
/// removes the residual of the captured background under the current flux,
/// source and boundary discretisation.
pub struct BackgroundCorrectedMoistEuler3D {
  base: MoistEuler3D,
  source_correction: Vec<[f64; 6]>,
}

impl BackgroundCorrectedMoistEuler3D {
  pub fn new(base: MoistEuler3D, source_correction: Vec<[f64; 6]>) -> Self {
    Self {
      base,
      source_correction,
    }
  }

  pub fn correction(&self) -> &[[f64; 6]] {
    &self.source_correction
  }
}

impl ConservationLaw<3, 6> for BackgroundCorrectedMoistEuler3D {
  fn fix_state(&self, state: &mut [f64; 6]) {
    self.base.fix_state(state);
  }

  fn flux(&self, state: &[f64; 6]) -> [[f64; 6]; 3] {
    self.base.flux(state)
  }

  fn max_wave_speed(&self, state: &[f64; 6]) -> f64 {
    self.base.max_wave_speed(state)
  }

  fn source(
    &self,
    state: &[f64; 6],
    cell: CellId,
    centroid: &Point<3>,
    metrics: &CellMetrics<3>,
  ) -> [f64; 6] {
    let mut source = self.base.source(state, cell, centroid, metrics);
    if let Some(correction) = self.source_correction.get(cell.index()) {
      for i in 0..6 {
        source[i] += correction[i];
      }
    }
    source
  }
}
