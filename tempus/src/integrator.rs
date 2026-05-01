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

impl<const D: usize, S: OdeSystem> OdeStepper<S> for ForwardEuler<D> {
  fn step(&mut self, system: &S, t: f64, y: &mut [Vector<f64, D>], dt: f64) {
    assert_eq!(y.len(), system.dimension());
    self.rhs.resize(y.len(), 0.0);
    system.rhs(t, y, &mut self.rhs);
    for (value, rhs) in y.iter_mut().zip(&self.rhs) {
      *value += dt * rhs;
    }
  }
}

#[derive(Clone, Debug, Default)]
pub struct Rk4<const D: usize> {
  k1: Vec<f64>,
  k2: Vec<f64>,
  k3: Vec<f64>,
  k4: Vec<f64>,
  scratch: Vec<f64>,
}

impl<const D: usize> Rk4<D> {
  pub fn new() -> Self {
    Self::default()
  }
}

impl<const D: usize, S: OdeSystem> OdeStepper<D, S> for Rk4<D> {
  fn step(&mut self, system: &S, t: f64, y: &mut [Vector<f64. D>], dt: f64) {
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
