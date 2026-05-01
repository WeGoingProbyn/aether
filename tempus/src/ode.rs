use utility::maths::vector::Vector;

/// First-order ODE system in the form `dy/dt = f(t, y)`.
pub trait OdeSystem<const D: usize> {
  fn rhs(&self, t: f64, y: &[Vector<f64, D>], dy: &mut [Vector<f64, D>]);
}

/// Mutable one-step integrator for first-order ODE systems.
pub trait OdeStepper<const D: usize, S: OdeSystem<D>> {
  fn step(&mut self, system: &S, t: f64, y: &mut [Vector<f64, D>], dt: f64);
}

/// Second-order system in the form `d²q/dt² = a(t, q, v)`.
pub trait SecondOrderSystem<const D: usize>  {
  fn acceleration(&self, t: f64, q: &[Vector<f64, D>], v: &[Vector<f64, D>], a: &mut [Vector<f64, D>]);
}
