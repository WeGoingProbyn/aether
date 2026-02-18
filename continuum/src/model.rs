use crate::geometry::{CellMetrics, Point};
use utility::{maths::vector::Vector, profile};

pub trait ConservationLaw<const D: usize, const N: usize>: Send + Sync {
  fn fix_state(&self, state: &mut [f64; N]);
  fn flux(&self, state: &[f64; N]) -> [[f64; N]; D];
  fn max_wave_speed(&self, state: &[f64; N]) -> f64;
  fn source(
    &self,
    state: &[f64; N],
    centroid: &Point<D>,
    metrics: &CellMetrics<D>,
  ) -> [f64; N];
}

pub trait NumericalFlux<const D: usize, const N: usize>: Send + Sync {
  fn compute(
    &self,
    law: &dyn ConservationLaw<D, N>,
    left: &[f64; N],
    right: &[f64; N],
    normal: &Vector<f64, D>,
  ) -> [f64; N];
}

pub struct RusanovFlux;

impl<const D: usize, const N: usize> NumericalFlux<D, N> for RusanovFlux {
  #[profile]
  fn compute(
    &self,
    law: &dyn ConservationLaw<D, N>,
    left: &[f64; N],
    right: &[f64; N],
    normal: &Vector<f64, D>,
  ) -> [f64; N] {
    let fl = law.flux(left);
    let fr = law.flux(right);
    let s_max = law.max_wave_speed(left).max(law.max_wave_speed(right));

    let mut result = [0.0; N];
    for i in 0..N {
      let mut fn_avg = 0.0;
      for d in 0..D {
        fn_avg += 0.5 * (fl[d][i] + fr[d][i]) * normal[d];
      }
      result[i] = fn_avg - 0.5 * s_max * (right[i] - left[i]);
    }
    result
  }
}

pub struct Euler2D {
  gamma: f64, // Ratio of specific heats
}

impl Euler2D {
  pub fn new(gamma: f64) -> Euler2D {
    Euler2D { gamma }
  }

  pub fn pressure(&self, state: &[f64; 4]) -> f64 {
    let rho = state[0];
    let u = state[1] / rho;
    let v = state[2] / rho;
    (self.gamma - 1.0) * (state[3] - 0.5 * rho * (u * u + v * v))
  }
}

impl ConservationLaw<2, 4> for Euler2D {
  fn flux(&self, state: &[f64; 4]) -> [[f64; 4]; 2] {
    let rho = state[0];

    let u = state[1] / rho;
    let v = state[2] / rho;
    let p = self.pressure(state);

    let fx = [
      state[1],           // rho * u
      state[1] * u + p,   // rho * u^2 + p
      state[1] * v,       // rho * u * v
      (state[3] + p) * u, // (E + p) * u
    ];
    let fy = [
      state[2],           // rho * v
      state[2] * u,       // rho * v * u
      state[2] * v + p,   // rho * v^2 + p
      (state[3] + p) * v, // (E + p) * v
    ];
    [fx, fy]
  }

  fn max_wave_speed(&self, state: &[f64; 4]) -> f64 {
    let rho = state[0];
    let u = state[1] / rho;
    let v = state[2] / rho;
    let p = self.pressure(state);
    let c = (self.gamma * p / rho).sqrt();
    (u * u + v * v).sqrt() + c
  }

  fn source(
    &self,
    _state: &[f64; 4],
    _centroid: &Point<2>,
    _metrics: &CellMetrics<2>,
  ) -> [f64; 4] {
    [0.0; 4] // no source terms for basic Euler
  }

  #[profile]
  fn fix_state(&self, state: &mut [f64; 4]) {
    let floor = 1e-8;
    if state[0] < floor {
      state[0] = floor;
    }
    let rho = state[0];
    let u = state[1] / rho;
    let v = state[2] / rho;
    let ke = 0.5 * rho * (u * u + v * v);
    if state[3] - ke < floor {
      state[3] = ke + floor;
    }
  }
}
