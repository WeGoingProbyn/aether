// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Rosenbrock(-Wanner) integration schemes expressed as plain tableau data.
//!
//! A scheme is just coefficients, so swapping integrators (tinkering toward
//! ROS2 / RODAS / IMEX) is a data change, not a plumbing change. The backend
//! ([`super::backend`]) interprets any [`RosenbrockTableau`] uniformly. For an
//! autonomous residual `R(U) = dU/dt`, a constant-γ Rosenbrock-W step with
//! step size `h` solves, for each stage `i`:
//!
//! ```text
//!   (I/(γ·h) − J)·kᵢ = R(U + Σ_{j<i} aᵢⱼ·kⱼ) + (1/h)·Σ_{j<i} cᵢⱼ·kⱼ
//! ```
//!
//! and updates `U_new = U + Σ_i mᵢ·kᵢ`, with `J` linearized once at `U` (the
//! "W" property — an approximate/frozen Jacobian is admissible).

/// A constant-γ Rosenbrock-W tableau. `a` and `c` are strictly-lower-triangular
/// (row `i` holds entries `j < i`); `m` are the update weights.
#[derive(Clone, Debug)]
pub struct RosenbrockTableau {
  pub name: &'static str,
  pub stages: usize,
  pub gamma: f64,
  /// `a[i][j]` for `j < i` — stage-state combination weights.
  pub a: Vec<Vec<f64>>,
  /// `c[i][j]` for `j < i` — stage-RHS combination weights (divided by `h`).
  pub c: Vec<Vec<f64>>,
  /// `m[i]` — final update weights.
  pub m: Vec<f64>,
  pub order: usize,
}

impl RosenbrockTableau {
  /// Linearly-implicit (Rosenbrock-)Euler: one stage, one GMRES solve, γ = 1.
  /// `(I/h − J)·k = R(U)`, `U_new = U + k`. Unconditionally stable, order 1 —
  /// the default and the simplest thing to validate.
  pub fn lin_implicit_euler() -> Self {
    Self {
      name: "lin_implicit_euler",
      stages: 1,
      gamma: 1.0,
      a: vec![vec![]],
      c: vec![vec![]],
      m: vec![1.0],
      order: 1,
    }
  }

  /// ROS2: the 2-stage, second-order, L-stable Rosenbrock of Verwer et al.,
  /// γ = 1 + 1/√2. Demonstrates that a higher-order scheme drops in by data
  /// alone; validated by an order-2 convergence test.
  pub fn ros2() -> Self {
    let gamma = 1.0 + 1.0 / 2.0_f64.sqrt();
    Self {
      name: "ros2",
      stages: 2,
      gamma,
      a: vec![vec![], vec![1.0 / gamma]],
      c: vec![vec![], vec![-2.0 / gamma]],
      m: vec![3.0 / (2.0 * gamma), 1.0 / (2.0 * gamma)],
      order: 2,
    }
  }
}

/// Named registry of the built-in schemes, so a backend / demo can pick an
/// integrator by string and we can add schemes without touching call sites.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RosenbrockScheme {
  LinearlyImplicitEuler,
  Ros2,
}

impl RosenbrockScheme {
  pub fn tableau(self) -> RosenbrockTableau {
    match self {
      RosenbrockScheme::LinearlyImplicitEuler => {
        RosenbrockTableau::lin_implicit_euler()
      }
      RosenbrockScheme::Ros2 => RosenbrockTableau::ros2(),
    }
  }

  pub fn from_name(name: &str) -> Option<Self> {
    match name {
      "lin_implicit_euler" => Some(Self::LinearlyImplicitEuler),
      "ros2" => Some(Self::Ros2),
      _ => None,
    }
  }
}

impl Default for RosenbrockScheme {
  fn default() -> Self {
    Self::LinearlyImplicitEuler
  }
}
