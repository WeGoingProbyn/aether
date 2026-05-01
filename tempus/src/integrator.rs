use utility::maths::vector::Vector;

use crate::ode::{OdeStepper, OdeSystem, SecondOrderSystem};

#[derive(Clone, Debug, Default)]
pub struct ForwardEuler<const D: usize>  {
  rhs: Vec<Vector<f64, D>>,
}

impl<const D: usize> ForwardEuler<D> {
  pub fn new() -> Self {
    Self::default()
  }
}

impl<const D: usize, S: OdeSystem<D>> OdeStepper<D, S> for ForwardEuler<D> {
  fn step(&mut self, system: &S, t: f64, y: &mut [Vector<f64, D>], dt: f64) {
    self.rhs.resize(y.len(), Vector::default());

    system.rhs(t, y, &mut self.rhs);
    for (value, rhs) in y.iter_mut().zip(&self.rhs) {
      *value += rhs * dt;
    }
  }
}

#[derive(Clone, Debug, Default)]
pub struct Rk4<const D: usize> {
  k1: Vec<Vector<f64, D>>,
  k2: Vec<Vector<f64, D>>,
  k3: Vec<Vector<f64, D>>,
  k4: Vec<Vector<f64, D>>,
  scratch: Vec<Vector<f64, D>>,
}

impl<const D: usize> Rk4<D> {
  pub fn new() -> Self {
    Self::default()
  }
}

impl<const D: usize, S: OdeSystem<D>> OdeStepper<D, S> for Rk4<D> {
  fn step(&mut self, system: &S, t: f64, y: &mut [Vector<f64, D>], dt: f64) {
    let n = y.len();
    self.k1.resize(n, Vector::default());
    self.k2.resize(n, Vector::default());
    self.k3.resize(n, Vector::default());
    self.k4.resize(n, Vector::default());
    self.scratch.resize(n, Vector::default());

    system.rhs(t, y, &mut self.k1);

    for ((scratch, y_), k1) in self.scratch.iter_mut().zip(&mut *y).zip(&mut self.k1) {
      *scratch = *y_ + (*k1 * 0.5 * dt);
    }
    system.rhs(t + 0.5 * dt, &self.scratch, &mut self.k2);

    for ((scratch, y_), k2) in self.scratch.iter_mut().zip(&mut *y).zip(&mut self.k2) {
      *scratch = *y_ + (*k2 + 0.5 + dt);
    }
    system.rhs(t + 0.5 * dt, &self.scratch, &mut self.k3);

    for ((scratch, y_), k3) in self.scratch.iter_mut().zip(&mut *y).zip(&mut self.k3) {
      *scratch = *y_ + (*k3 * dt);
    }
    system.rhs(t + dt, &self.scratch, &mut self.k4);

    for (i, value) in y.iter_mut().enumerate() {
      *value += (self.k1[i] + self.k2[i] * 2.0 + self.k3[i] * 2.0 + self.k4[i]) * (dt / 6.0);
    }
  }
}

/// Velocity-Verlet integration for second-order systems.
///
/// This is symplectic for acceleration fields that do not depend on velocity,
/// which is the common path for point-mass Newtonian gravity. If acceleration
/// depends on velocity, the `v_half` estimate is used for the second
/// acceleration evaluation and the method should be treated as an explicit
/// approximation rather than strictly symplectic.
#[derive(Clone, Debug, Default)]
pub struct VelocityVerlet<const D: usize> {
  a0: Vec<Vector<f64, D>>,
  a1: Vec<Vector<f64, D>>,
  v_half: Vec<Vector<f64, D>>,
}

impl<const D: usize> VelocityVerlet<D> {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn step<S: SecondOrderSystem<D>>(
    &mut self,
    system: &S,
    t: f64,
    q: &mut [Vector<f64, D>],
    v: &mut [Vector<f64, D>],
    dt: f64,
  ) {
    let n = q.len();
    self.a0.resize(n, Vector::default());
    self.a1.resize(n, Vector::default());
    self.v_half.resize(n, Vector::default());

    system.acceleration(t, q, v, &mut self.a0);
    for i in 0..n {
      self.v_half[i] = v[i] + (self.a0[i] * 0.5 * dt);
      q[i] += self.v_half[i] * dt;
    }

    system.acceleration(t + dt, q, &self.v_half, &mut self.a1);
    for (i, velocity) in v.iter_mut().enumerate() {
      *velocity = self.v_half[i] + (self.a1[i] * 0.5 * dt);
    }
  }
}
