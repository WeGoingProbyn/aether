// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use continuum::model::{ConservationLaw, Euler3D};
use tessera::geometry::CellMetrics;
use utility::domain::{CellId, Point};

/// Euler3D with a fixed per-cell source correction.
///
/// Aer uses this to cancel the discrete residual of a chosen background state.
/// This is not a strictly well-balanced finite-volume formulation; it only
/// removes the residual of the captured background under the current flux,
/// source and boundary discretisation.
pub struct BackgroundCorrectedEuler3D {
  base: Euler3D,
  source_correction: Vec<[f64; 5]>,
}

impl BackgroundCorrectedEuler3D {
  pub fn new(base: Euler3D, source_correction: Vec<[f64; 5]>) -> Self {
    Self {
      base,
      source_correction,
    }
  }

  pub fn correction(&self) -> &[[f64; 5]] {
    &self.source_correction
  }
}

impl ConservationLaw<3, 5> for BackgroundCorrectedEuler3D {
  fn fix_state(&self, state: &mut [f64; 5]) {
    self.base.fix_state(state);
  }

  fn flux(&self, state: &[f64; 5]) -> [[f64; 5]; 3] {
    self.base.flux(state)
  }

  fn max_wave_speed(&self, state: &[f64; 5]) -> f64 {
    self.base.max_wave_speed(state)
  }

  fn source(
    &self,
    state: &[f64; 5],
    cell: CellId,
    centroid: &Point<3>,
    metrics: &CellMetrics<3>,
  ) -> [f64; 5] {
    let mut source = self.base.source(state, cell, centroid, metrics);
    if let Some(correction) = self.source_correction.get(cell.index()) {
      for i in 0..5 {
        source[i] += correction[i];
      }
    }
    source
  }
}
