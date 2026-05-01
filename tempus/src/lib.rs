// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Generic time-integration kernels.
//!
//! `tempus` deliberately has no dependency on Nexus, Pleroma, Tessera, or any
//! physics crate. Physics crates own their state layout and call these kernels
//! from their own stages/models.

pub mod ode;
pub mod integrator;

#[cfg(test)]
mod tests {
  use super::*;

  struct Exponential;

  impl OdeSystem for Exponential {
    fn dimension(&self) -> usize {
      1
    }

    fn rhs(&self, _t: f64, y: &[f64], dy: &mut [f64]) {
      dy[0] = y[0];
    }
  }

  struct HarmonicOscillator;

  impl SecondOrderSystem for HarmonicOscillator {
    fn degrees_of_freedom(&self) -> usize {
      1
    }

    fn acceleration(&self, _t: f64, q: &[f64], _v: &[f64], a: &mut [f64]) {
      a[0] = -q[0];
    }
  }

  #[test]
  fn rk4_integrates_exponential_growth() {
    let mut stepper = Rk4::new();
    let system = Exponential;
    let mut y = [1.0];
    let dt = 0.01;
    for i in 0..100 {
      stepper.step(&system, i as f64 * dt, &mut y, dt);
    }
    assert!((y[0] - std::f64::consts::E).abs() < 1.0e-8);
  }

  #[test]
  fn velocity_verlet_keeps_harmonic_oscillator_energy_bounded() {
    let mut stepper = VelocityVerlet::new();
    let system = HarmonicOscillator;
    let mut q = [1.0];
    let mut v = [0.0];
    let initial_energy = 0.5 * q[0] * q[0] + 0.5 * v[0] * v[0];
    let dt = 0.01;
    for i in 0..10_000 {
      stepper.step(&system, i as f64 * dt, &mut q, &mut v, dt);
    }
    let final_energy = 0.5 * q[0] * q[0] + 0.5 * v[0] * v[0];
    assert!((final_energy - initial_energy).abs() < 1.0e-4);
  }
}
