// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use continuum::model::{ConservationLaw, MoistEuler3D, Scalar};
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

  fn flux<T: Scalar>(&self, state: &[T; 6]) -> [[T; 6]; 3] {
    self.base.flux(state)
  }

  fn max_wave_speed<T: Scalar>(&self, state: &[T; 6]) -> T {
    self.base.max_wave_speed(state)
  }

  fn source<T: Scalar>(
    &self,
    state: &[T; 6],
    cell: CellId,
    centroid: &Point<3>,
    metrics: &CellMetrics<3>,
  ) -> [T; 6] {
    let mut source = self.base.source(state, cell, centroid, metrics);
    if let Some(correction) = self.source_correction.get(cell.index()) {
      for i in 0..6 {
        source[i] = source[i] + correction[i];
      }
    }
    source
  }

  // Forward the IMEX/HEVI operator split to the base moist-Euler law so a
  // vertically-implicit (HEVI) backend sees the acoustic split (the background
  // correction is an explicit per-cell source, so it stays in the RHS only).
  fn implicit_flux<T: Scalar>(&self, state: &[T; 6]) -> [[T; 6]; 3] {
    self.base.implicit_flux(state)
  }

  fn acoustic_speed<T: Scalar>(&self, state: &[T; 6]) -> T {
    self.base.acoustic_speed(state)
  }

  fn explicit_wave_speed<T: Scalar>(&self, state: &[T; 6]) -> T {
    self.base.explicit_wave_speed(state)
  }
}
