// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Generic time-integration kernels.
//!
//! `tempus` deliberately has no dependency on Nexus, Pleroma, Tessera, or any
//! physics crate. Physics crates own their state layout and call these kernels
//! from their own stages/models.

/// First-order ODE system in the form `dy/dt = f(t, y)`.
pub trait OdeSystem {
  fn dimension(&self) -> usize;
  fn rhs(&self, t: f64, y: &[f64], dy: &mut [f64]);
}

/// Mutable one-step integrator for first-order ODE systems.
pub trait OdeStepper<S: OdeSystem> {
  fn step(&mut self, system: &S, t: f64, y: &mut [f64], dt: f64);
}

#[derive(Clone, Debug, Default)]
pub struct ForwardEuler {
  rhs: Vec<f64>,
}

impl ForwardEuler {
  pub fn new() -> Self {
    Self::default()
  }
}

impl<S: OdeSystem> OdeStepper<S> for ForwardEuler {
  fn step(&mut self, system: &S, t: f64, y: &mut [f64], dt: f64) {
    assert_eq!(y.len(), system.dimension());
    self.rhs.resize(y.len(), 0.0);
    system.rhs(t, y, &mut self.rhs);
    for (value, rhs) in y.iter_mut().zip(&self.rhs) {
      *value += dt * rhs;
    }
  }
}

#[derive(Clone, Debug, Default)]
pub struct Rk4 {
  k1: Vec<f64>,
  k2: Vec<f64>,
  k3: Vec<f64>,
  k4: Vec<f64>,
  scratch: Vec<f64>,
}

impl Rk4 {
  pub fn new() -> Self {
    Self::default()
  }
}

impl<S: OdeSystem> OdeStepper<S> for Rk4 {
  fn step(&mut self, system: &S, t: f64, y: &mut [f64], dt: f64) {
    let n = system.dimension();
    assert_eq!(y.len(), n);
    self.k1.resize(n, 0.0);
    self.k2.resize(n, 0.0);
    self.k3.resize(n, 0.0);
    self.k4.resize(n, 0.0);
    self.scratch.resize(n, 0.0);

    system.rhs(t, y, &mut self.k1);

    for i in 0..n {
      self.scratch[i] = y[i] + 0.5 * dt * self.k1[i];
    }
    system.rhs(t + 0.5 * dt, &self.scratch, &mut self.k2);

    for i in 0..n {
      self.scratch[i] = y[i] + 0.5 * dt * self.k2[i];
    }
    system.rhs(t + 0.5 * dt, &self.scratch, &mut self.k3);

    for i in 0..n {
      self.scratch[i] = y[i] + dt * self.k3[i];
    }
    system.rhs(t + dt, &self.scratch, &mut self.k4);

    for (i, value) in y.iter_mut().enumerate() {
      *value += dt / 6.0
        * (self.k1[i] + 2.0 * self.k2[i] + 2.0 * self.k3[i] + self.k4[i]);
    }
  }
}

/// Second-order system in the form `d²q/dt² = a(t, q, v)`.
pub trait SecondOrderSystem {
  fn degrees_of_freedom(&self) -> usize;
  fn acceleration(&self, t: f64, q: &[f64], v: &[f64], a: &mut [f64]);
}

/// Velocity-Verlet integration for second-order systems.
///
/// This is symplectic for acceleration fields that do not depend on velocity,
/// which is the common path for point-mass Newtonian gravity. If acceleration
/// depends on velocity, the `v_half` estimate is used for the second
/// acceleration evaluation and the method should be treated as an explicit
/// approximation rather than strictly symplectic.
#[derive(Clone, Debug, Default)]
pub struct VelocityVerlet {
  a0: Vec<f64>,
  a1: Vec<f64>,
  v_half: Vec<f64>,
}

impl VelocityVerlet {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn step<S: SecondOrderSystem>(
    &mut self,
    system: &S,
    t: f64,
    q: &mut [f64],
    v: &mut [f64],
    dt: f64,
  ) {
    let n = system.degrees_of_freedom();
    assert_eq!(q.len(), n);
    assert_eq!(v.len(), n);
    self.a0.resize(n, 0.0);
    self.a1.resize(n, 0.0);
    self.v_half.resize(n, 0.0);

    system.acceleration(t, q, v, &mut self.a0);
    for i in 0..n {
      self.v_half[i] = v[i] + 0.5 * dt * self.a0[i];
      q[i] += dt * self.v_half[i];
    }

    system.acceleration(t + dt, q, &self.v_half, &mut self.a1);
    for (i, velocity) in v.iter_mut().enumerate() {
      *velocity = self.v_half[i] + 0.5 * dt * self.a1[i];
    }
  }
}

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
