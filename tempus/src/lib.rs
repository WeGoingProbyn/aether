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
  use utility::maths::vector::Vector;

use super::ode::*;
  use super::integrator::*;

  struct Exponential;

  impl OdeSystem<1> for Exponential {
    fn rhs(&self, _t: f64, y: &[Vector<f64, 1>], dy: &mut [Vector<f64, 1>]) {
      dy[0] = y[0];
    }
  }

  struct HarmonicOscillator;

  impl SecondOrderSystem<1> for HarmonicOscillator {
    fn acceleration(&self, _t: f64, q: &[Vector<f64, 1>], _v: &[Vector<f64, 1>], a: &mut [Vector<f64, 1>]) {
      a[0][0] = -q[0][0];
    }
  }

  #[test]
  fn rk4_integrates_exponential_growth() {
    let mut stepper = Rk4::new();
    let system = Exponential;
    let mut y = [[1.0].into()];
    let dt = 0.01;
    for i in 0..100 {
      stepper.step(&system, i as f64 * dt, &mut y, dt);
    }
    println!("{}", (y[0][0] - std::f64::consts::E).abs());
    assert!((y[0][0] - std::f64::consts::E).abs() < 1.0e-8);
  }

  #[test]
  fn velocity_verlet_keeps_harmonic_oscillator_energy_bounded() {
    let mut stepper = VelocityVerlet::new();
    let system = HarmonicOscillator;
    let mut q: [Vector<f64, 1>; 1] = [[1.0].into()];
    let mut v: [Vector<f64, 1>; 1] = [[0.0].into()];
    let initial_energy = 0.5 * q[0][0] * q[0][0] + 0.5 * v[0][0] * v[0][0];
    let dt = 0.01;
    for i in 0..10_000 {
      stepper.step(&system, i as f64 * dt, &mut q, &mut v, dt);
    }
    let final_energy = 0.5 * q[0][0] * q[0][0] + 0.5 * v[0][0] * v[0][0];
    assert!((final_energy - initial_energy).abs() < 1.0e-4);
  }
}
