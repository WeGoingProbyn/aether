// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! The implicit finite-volume backend: a Rosenbrock integrator that linearizes
//! the residual and solves each stage with matrix-free GMRES. It implements the
//! existing [`FvmBackend`] trait, so it is a drop-in alternative to the
//! explicit [`CpuBackend`](crate::cpu::CpuBackend) — the win is that callers
//! may pass a `dt` far larger than the acoustic CFL limit and stay stable.

use num_dual::Dual64;
use pleroma::core::storage::FieldStorage;
use tessera::mesh::Mesh;

use crate::{
  boundary::BoundaryRegistry,
  implicit::{
    ad::{
      AutoDiffJacobian, BlockJacobi, PreconditionerKind,
      face_adjacency_colouring,
    },
    gmres::{self, GmresConfig},
    jacobian::{
      FiniteDifferenceJacobian, JacobianStrategy, ResidualEval, ShiftedOperator,
    },
    linalg::{flatten, unflatten},
    schemes::RosenbrockTableau,
  },
  kernel,
  model::{ConservationLaw, NumericalFlux},
  solver::{FvmBackend, SolverConfig},
};

/// Outcome of one implicit step, surfaced for the Phase-3 hybrid controller.
#[derive(Clone, Copy, Debug)]
pub struct ImplicitStepReport {
  /// Every stage solve reached the GMRES tolerance.
  pub converged: bool,
  /// Total GMRES matrix-vector products across all stages.
  pub gmres_iters: usize,
  /// Final-stage GMRES residual norm (diagnostic).
  pub residual: f64,
}

/// Which operator a [`LawResidual`] evaluates: the full residual `R(U)`, or
/// just the IMEX implicit part `R_im(U)`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum OperatorPart {
  Full,
  Implicit,
}

/// A [`ResidualEval`] that evaluates a conservation law's residual over a mesh —
/// either the full `R(U)` ([`kernel::compute_residual_into`]) or, for IMEX, the
/// implicit part `R_im(U)` ([`kernel::compute_residual_implicit`]). Holds a flux
/// accumulator so repeated evaluations (every JVP, every stage) allocate nothing.
struct LawResidual<'a, const D: usize, const N: usize, L, F, M: ?Sized> {
  law: &'a L,
  flux: &'a F,
  mesh: &'a M,
  bcs: &'a BoundaryRegistry<D, N>,
  part: OperatorPart,
  accum: Vec<[f64; N]>,
  // Dual scratch for exact AD Jacobian-vector products.
  dual_state: Vec<[Dual64; N]>,
  dual_accum: Vec<[Dual64; N]>,
  dual_out: Vec<[Dual64; N]>,
}

impl<'a, const D: usize, const N: usize, L, F, M: ?Sized>
  LawResidual<'a, D, N, L, F, M>
{
  fn new(
    law: &'a L,
    flux: &'a F,
    mesh: &'a M,
    bcs: &'a BoundaryRegistry<D, N>,
    part: OperatorPart,
  ) -> Self {
    Self {
      law,
      flux,
      mesh,
      bcs,
      part,
      accum: Vec::new(),
      dual_state: Vec::new(),
      dual_accum: Vec::new(),
      dual_out: Vec::new(),
    }
  }
}

impl<const D: usize, const N: usize, L, F, M> ResidualEval<N>
  for LawResidual<'_, D, N, L, F, M>
where
  L: ConservationLaw<D, N>,
  F: NumericalFlux<D, N>,
  M: Mesh<D> + ?Sized,
{
  fn eval(&mut self, state: &[[f64; N]], out: &mut [[f64; N]]) {
    match self.part {
      OperatorPart::Full => kernel::compute_residual_into(
        self.law,
        self.flux,
        state,
        &mut self.accum,
        out,
        self.mesh,
        self.bcs,
      ),
      OperatorPart::Implicit => kernel::compute_residual_implicit(
        self.law,
        self.flux,
        state,
        &mut self.accum,
        out,
        self.mesh,
        self.bcs,
      ),
    }
  }

  fn dual_jvp(&mut self, u: &[[f64; N]], v: &[f64], out: &mut [f64]) -> bool {
    let cells = u.len();
    if self.dual_state.len() != cells {
      self.dual_state.resize(cells, [Dual64::from(0.0); N]);
      self.dual_out.resize(cells, [Dual64::from(0.0); N]);
    }
    // Seed each input with its real value and the direction in the dual part:
    // U_ij + ε·v_ij. One dual residual sweep then carries ∂R/∂U·v in `.eps`.
    for i in 0..cells {
      for c in 0..N {
        self.dual_state[i][c] = Dual64::new(u[i][c], v[i * N + c]);
      }
    }
    match self.part {
      OperatorPart::Full => kernel::compute_residual_generic(
        self.law,
        self.flux,
        &self.dual_state,
        &mut self.dual_accum,
        &mut self.dual_out,
        self.mesh,
        self.bcs,
      ),
      OperatorPart::Implicit => kernel::compute_residual_implicit(
        self.law,
        self.flux,
        &self.dual_state,
        &mut self.dual_accum,
        &mut self.dual_out,
        self.mesh,
        self.bcs,
      ),
    }
    for i in 0..cells {
      for c in 0..N {
        out[i * N + c] = self.dual_out[i][c].eps;
      }
    }
    true
  }
}

/// Advance `u` (a per-cell state cache) by one Rosenbrock step of size `dt`.
///
/// Mesh-, law- and `FieldStorage`-agnostic: it speaks only [`ResidualEval`] +
/// [`JacobianStrategy`] + [`Preconditioner`], so the scheme math is testable in
/// isolation and the IMEX split (later) reuses it verbatim by passing a
/// different residual. Two residual evaluators are required — `res_jvp` is
/// captured by the linear operator for Jacobian-vector products, `res_rhs`
/// builds each stage's right-hand side.
pub fn rosenbrock_step<const N: usize>(
  tableau: &RosenbrockTableau,
  res_jvp: &mut dyn ResidualEval<N>,
  res_rhs: &mut dyn ResidualEval<N>,
  jacobian: &mut dyn JacobianStrategy<N>,
  precond: &mut PreconditionerKind<N>,
  gmres_cfg: &GmresConfig,
  dt: f64,
  u: &mut [[f64; N]],
) -> ImplicitStepReport {
  let cells = u.len();
  let dim = cells * N;
  let stages = tableau.stages;

  // Base state and base residual `R(U)` — the Jacobian is linearized here and
  // reused for every stage (the Rosenbrock-W property).
  let base = u.to_vec();
  let mut r0 = vec![[0.0; N]; cells];
  res_rhs.eval(&base, &mut r0);

  let gamma_dt = tableau.gamma * dt;
  // (Re)linearize the preconditioner at the base state before the stage solves
  // (a no-op for the identity preconditioner). Built from `res_jvp` — the
  // operator's Jacobian residual — so under IMEX it preconditions `I/(γ·dt) −
  // J_im`, not the full Jacobian.
  precond.rebuild(res_jvp, &base, gamma_dt);
  let mut op = ShiftedOperator::new(res_jvp, jacobian, &base, &r0, gamma_dt);

  let mut k: Vec<Vec<[f64; N]>> = vec![vec![[0.0; N]; cells]; stages];
  let mut u_stage = vec![[0.0; N]; cells];
  let mut rhs = vec![[0.0; N]; cells];
  let mut b = Vec::with_capacity(dim);
  let mut k_flat = vec![0.0; dim];

  let mut converged = true;
  let mut gmres_iters = 0;
  let mut last_residual = 0.0;

  for i in 0..stages {
    // Stage state: U + Σ_{j<i} a_ij k_j.
    u_stage.copy_from_slice(&base);
    for j in 0..i {
      let a = tableau.a[i][j];
      if a != 0.0 {
        for cell in 0..cells {
          for d in 0..N {
            u_stage[cell][d] += a * k[j][cell][d];
          }
        }
      }
    }

    // Stage RHS: R(U_stage) + (1/dt) Σ_{j<i} c_ij k_j.
    res_rhs.eval(&u_stage, &mut rhs);
    for j in 0..i {
      let cc = tableau.c[i][j] / dt;
      if cc != 0.0 {
        for cell in 0..cells {
          for d in 0..N {
            rhs[cell][d] += cc * k[j][cell][d];
          }
        }
      }
    }

    // Solve (I/(γ·dt) − J) k_i = rhs.
    flatten(&rhs, &mut b);
    k_flat.iter_mut().for_each(|x| *x = 0.0);
    let result =
      gmres::solve(&mut op, &mut *precond, &b, &mut k_flat, gmres_cfg);
    converged &= result.converged;
    gmres_iters += result.iterations;
    last_residual = result.residual;
    unflatten(&k_flat, &mut k[i]);
  }

  // U_new = U + Σ_i m_i k_i.
  u.copy_from_slice(&base);
  for i in 0..stages {
    let m = tableau.m[i];
    for cell in 0..cells {
      for d in 0..N {
        u[cell][d] += m * k[i][cell][d];
      }
    }
  }

  ImplicitStepReport {
    converged,
    gmres_iters,
    residual: last_residual,
  }
}

/// Implicit finite-volume backend. Holds the scheme tableau, GMRES tuning, a
/// swappable Jacobian strategy (finite-difference by default; AD later) and a
/// preconditioner (identity hook now; block-Jacobi later).
pub struct ImplicitBackend<const N: usize> {
  scheme: RosenbrockTableau,
  gmres_cfg: GmresConfig,
  jacobian: Box<dyn JacobianStrategy<N>>,
  precond: PreconditionerKind<N>,
  /// When set, lazily build a block-Jacobi preconditioner once the mesh is
  /// known (its colouring is derived from the mesh adjacency).
  want_block_jacobi: bool,
  /// IMEX mode: linearize only the law's implicit operator `R_im` (the RHS
  /// stays the full residual), so the stiff acoustic part is implicit and the
  /// advective remainder is explicit.
  imex: bool,
  state_cache: Vec<[f64; N]>,
  last_report: Option<ImplicitStepReport>,
}

impl<const N: usize> ImplicitBackend<N> {
  /// Build a backend for the given scheme with finite-difference Jacobians and
  /// no preconditioning — the Phase-0/1 default.
  pub fn new(scheme: RosenbrockTableau) -> Self {
    Self {
      scheme,
      gmres_cfg: GmresConfig::default(),
      jacobian: Box::new(FiniteDifferenceJacobian::<N>::new()),
      precond: PreconditionerKind::Identity,
      want_block_jacobi: false,
      imex: false,
      state_cache: Vec::new(),
      last_report: None,
    }
  }

  /// Enable IMEX stepping: only the law's implicit operator is linearized, so
  /// the acoustic terms are implicit and advection is explicit. Pair with a
  /// law that overrides `implicit_flux` (e.g. `MoistEuler3D`); a law that does
  /// not opt in has an empty implicit operator and IMEX reduces to explicit.
  /// Use the linearly-implicit-Euler scheme (the first-order IMEX-Euler).
  pub fn with_imex(mut self) -> Self {
    self.imex = true;
    self
  }

  pub fn with_gmres_config(mut self, cfg: GmresConfig) -> Self {
    self.gmres_cfg = cfg;
    self
  }

  /// Swap the Jacobian strategy (e.g. the AD strategy in Phase 2).
  pub fn with_jacobian(
    mut self,
    jacobian: Box<dyn JacobianStrategy<N>>,
  ) -> Self {
    self.jacobian = jacobian;
    self
  }

  /// Use exact automatic-differentiation Jacobian-vector products.
  pub fn with_auto_diff(mut self) -> Self {
    self.jacobian = Box::new(AutoDiffJacobian::new());
    self
  }

  /// Enable the AD block-Jacobi preconditioner. The colouring is built from the
  /// mesh on first use, and the blocks are re-factored each step.
  pub fn with_block_jacobi(mut self) -> Self {
    self.want_block_jacobi = true;
    self
  }

  /// The most recent step's solver report, for diagnostics / the hybrid
  /// controller.
  pub fn last_report(&self) -> Option<ImplicitStepReport> {
    self.last_report
  }

  /// Build the block-Jacobi colouring from the mesh the first time it is seen
  /// (or if the cell count changes).
  fn ensure_preconditioner<const D: usize, M>(&mut self, mesh: &M)
  where
    M: Mesh<D> + ?Sized,
  {
    if !self.want_block_jacobi {
      return;
    }
    let cells = mesh.cell_count();
    let stale = match &self.precond {
      PreconditionerKind::BlockJacobi(bj) => bj.cell_count() != cells,
      PreconditionerKind::Identity => true,
    };
    if stale {
      let (colour, n) = face_adjacency_colouring(cells, mesh.interior_faces());
      self.precond =
        PreconditionerKind::BlockJacobi(BlockJacobi::new(colour, n));
    }
  }

  fn advance<const D: usize, L, F, S, M>(
    &mut self,
    law: &L,
    flux: &F,
    dt: f64,
    state: &mut S,
    mesh: &M,
    bcs: &BoundaryRegistry<D, N>,
  ) where
    L: ConservationLaw<D, N>,
    F: NumericalFlux<D, N>,
    S: FieldStorage<N>,
    M: Mesh<D> + ?Sized,
  {
    self.ensure_preconditioner(mesh);
    kernel::gather_state_cache(state, mesh, &mut self.state_cache);

    // The RHS is always the full residual; under IMEX only the operator's
    // Jacobian residual is the implicit part (`R = R_ex + R_im`).
    let jvp_part = if self.imex {
      OperatorPart::Implicit
    } else {
      OperatorPart::Full
    };
    let mut res_jvp = LawResidual::new(law, flux, mesh, bcs, jvp_part);
    let mut res_rhs =
      LawResidual::new(law, flux, mesh, bcs, OperatorPart::Full);

    let report = rosenbrock_step(
      &self.scheme,
      &mut res_jvp,
      &mut res_rhs,
      &mut *self.jacobian,
      &mut self.precond,
      &self.gmres_cfg,
      dt,
      &mut self.state_cache,
    );
    self.last_report = Some(report);

    // Scatter back, then enforce positivity / physical bounds.
    for (i, s) in self.state_cache.iter_mut().enumerate() {
      law.fix_state(s);
      state.write(utility::domain::CellId::from(i), s);
    }
  }
}

impl<const D: usize, const N: usize, L, F> FvmBackend<D, N, L, F>
  for ImplicitBackend<N>
where
  L: ConservationLaw<D, N>,
  F: NumericalFlux<D, N>,
{
  fn step<S, M>(
    &mut self,
    config: &SolverConfig,
    law: &L,
    flux: &F,
    state: &mut S,
    residual: &mut S,
    mesh: &M,
    bcs: &BoundaryRegistry<D, N>,
  ) -> f64
  where
    S: FieldStorage<N>,
    M: Mesh<D> + ?Sized,
  {
    let _ = residual;
    kernel::gather_state_cache(state, mesh, &mut self.state_cache);
    // IMEX is stable to the advective CFL; full-implicit can take any dt but
    // here matches the explicit one when no target is given.
    let dt = if self.imex {
      kernel::compute_explicit_dt_from_cache(
        config,
        law,
        &self.state_cache,
        mesh,
      )
    } else {
      kernel::compute_dt_from_cache(config, law, &self.state_cache, mesh)
    };
    self.advance(law, flux, dt, state, mesh, bcs);
    dt
  }

  fn step_with_dt<S, M>(
    &mut self,
    config: &SolverConfig,
    law: &L,
    flux: &F,
    dt: f64,
    state: &mut S,
    residual: &mut S,
    mesh: &M,
    bcs: &BoundaryRegistry<D, N>,
  ) where
    S: FieldStorage<N>,
    M: Mesh<D> + ?Sized,
  {
    let _ = (config, residual);
    self.advance(law, flux, dt, state, mesh, bcs);
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::boundary::ReflectiveWall;
  use crate::implicit::ad::AutoDiffJacobian;
  use crate::implicit::schemes::RosenbrockScheme;
  use crate::model::{MoistEuler3D, RusanovFlux};
  use tessera::geometry::{CellGeometry, IdentityMap};
  use tessera::mesh::StructuredBlock;
  use utility::domain::{BoundaryTag, CellId};

  #[test]
  fn ad_jvp_matches_fd_jvp_on_moist_euler() {
    // Small 3D box with a non-trivial moist-atmosphere state.
    let mesh = StructuredBlock::uniform(
      [0.0, 0.0, 0.0].into(),
      [1.0, 1.0, 1.0],
      [4, 4, 2],
      Box::new(IdentityMap::<3>),
    );
    let cells = mesh.cell_count();
    let law = MoistEuler3D::with_gravity(1.4, [0.0, 0.0, -9.81])
      .with_rotation([0.0, 0.0, 7.0e-3]);
    let flux = RusanovFlux;
    let mut bcs = BoundaryRegistry::<3, 6>::new();
    for tag in [
      BoundaryTag::Left,
      BoundaryTag::Right,
      BoundaryTag::Bottom,
      BoundaryTag::Top,
      BoundaryTag::Front,
      BoundaryTag::Back,
    ] {
      bcs.register(tag, ReflectiveWall);
    }

    // A smooth, physical state (ρ, ρu, …, E, ρq).
    let mut u = vec![[0.0; 6]; cells];
    for (i, cell) in u.iter_mut().enumerate() {
      let c = mesh.cell_centroid(CellId::from(i));
      let rho = 1.0 + 0.2 * (c[0] * 3.0).sin();
      let p = 1.0 + 0.1 * (c[1] * 2.0).cos();
      let (mu, mv, mw) = (0.1 * rho, -0.05 * rho, 0.02 * rho);
      let ke = 0.5 / rho * (mu * mu + mv * mv + mw * mw);
      *cell = [rho, mu, mv, mw, p / 0.4 + ke, 0.01 * rho];
    }

    // A non-trivial direction.
    let mut v = vec![0.0; cells * 6];
    for (k, vk) in v.iter_mut().enumerate() {
      *vk = ((k as f64) * 0.7).sin() * 0.01;
    }

    let mut res =
      LawResidual::new(&law, &flux, &mesh, &bcs, OperatorPart::Full);
    let mut r0 = vec![[0.0; 6]; cells];
    res.eval(&u, &mut r0);

    let mut jv_fd = vec![0.0; cells * 6];
    FiniteDifferenceJacobian::<6>::new().jvp(&mut res, &u, &r0, &v, &mut jv_fd);

    let mut jv_ad = vec![0.0; cells * 6];
    AutoDiffJacobian.jvp(&mut res, &u, &r0, &v, &mut jv_ad);

    // ‖J·v_AD − J·v_FD‖ / ‖J·v_AD‖ within finite-difference truncation.
    let mut num = 0.0;
    let mut den = 0.0;
    for k in 0..cells * 6 {
      num += (jv_ad[k] - jv_fd[k]).powi(2);
      den += jv_ad[k].powi(2);
    }
    let rel = (num / den).sqrt();
    assert!(den > 0.0, "AD JVP was identically zero");
    assert!(rel < 1e-5, "AD vs FD relative mismatch {rel} (want < 1e-5)");
  }

  #[test]
  fn block_jacobi_cuts_gmres_iterations() {
    use crate::solver::{FvmSolver, SolverConfig, TimeIntegration};
    use pleroma::core::storage::{CellView, SoaField};

    let mesh = StructuredBlock::uniform(
      [0.0, 0.0, 0.0].into(),
      [1.0, 1.0, 1.0],
      [6, 6, 3],
      Box::new(IdentityMap::<3>),
    );
    let cells = mesh.cell_count();
    let gamma = 1.4;
    let mut bcs = BoundaryRegistry::<3, 6>::new();
    for tag in [
      BoundaryTag::Left,
      BoundaryTag::Right,
      BoundaryTag::Bottom,
      BoundaryTag::Top,
      BoundaryTag::Front,
      BoundaryTag::Back,
    ] {
      bcs.register(tag, ReflectiveWall);
    }

    let make_state = || {
      SoaField::<6>::from_fn(cells, |cell| {
        let c = mesh.cell_centroid(cell);
        let rho = 1.0 + 0.15 * (c[0] * 3.0).sin();
        let p = 1.0 + 0.1 * (c[1] * 2.0).cos();
        [rho, 0.0, 0.0, 0.0, p / (gamma - 1.0), 0.01 * rho]
      })
    };

    // Explicit CFL step, then a much larger implicit step.
    let cfl_dt = FvmSolver::new(
      SolverConfig::new(0.5, 1.0, TimeIntegration::ForwardEuler),
      MoistEuler3D::new(gamma),
      RusanovFlux,
    )
    .compute_dt(&make_state(), &mesh);
    let dt = 5.0 * cfl_dt;
    let config = SolverConfig::new(0.5, 1.0, TimeIntegration::ForwardEuler);
    let gmres = GmresConfig {
      restart: 40,
      max_restarts: 10,
      rel_tol: 1e-9,
      abs_tol: 1e-12,
    };

    // Run one implicit step and return (converged, gmres_iters, final state).
    let run = |backend: &mut ImplicitBackend<6>| {
      let mut state = make_state();
      let mut residual = SoaField::<6>::zeros(cells);
      backend.step_with_dt(
        &config,
        &MoistEuler3D::with_gravity(gamma, [0.0, 0.0, -9.81]),
        &RusanovFlux,
        dt,
        &mut state,
        &mut residual,
        &mesh,
        &bcs,
      );
      let r = backend.last_report().unwrap();
      (r.converged, r.gmres_iters, state)
    };

    let scheme = RosenbrockScheme::LinearlyImplicitEuler.tableau();
    let (id_conv, _id_iters, id_state) = run(
      &mut ImplicitBackend::<6>::new(scheme.clone())
        .with_auto_diff()
        .with_gmres_config(gmres),
    );
    let (bj_conv, _bj_iters, bj_state) = run(
      &mut ImplicitBackend::<6>::new(scheme)
        .with_auto_diff()
        .with_block_jacobi()
        .with_gmres_config(gmres),
    );

    // Correctness: the block-Jacobi-preconditioned solve must converge and
    // reach the same step as the unpreconditioned solve. (On a pure-acoustic
    // box the stiffness is in the off-diagonal coupling, so block-diagonal
    // preconditioning is correct but not a big iteration win here — see the
    // source-stiff test in `ad` for where it pays off.)
    assert!(id_conv && bj_conv, "a solve did not converge");
    let mut max_diff = 0.0_f64;
    for i in 0..cells {
      let a = id_state.state(CellId::from(i));
      let b = bj_state.state(CellId::from(i));
      for k in 0..6 {
        max_diff = max_diff.max((a.as_state()[k] - b.as_state()[k]).abs());
      }
    }
    assert!(
      max_diff < 1e-6,
      "block-Jacobi vs identity step differ by {max_diff}"
    );
  }

  /// Synthetic residual `R(U) = λ·U` per cell (decoupled, N=1) — a stiff linear
  /// ODE with exact solution `U(t) = U₀·e^{λt}`, used to validate scheme order
  /// without a mesh.
  struct LinearOde {
    lambda: f64,
  }

  impl ResidualEval<1> for LinearOde {
    fn eval(&mut self, state: &[[f64; 1]], out: &mut [[f64; 1]]) {
      for i in 0..state.len() {
        out[i][0] = self.lambda * state[i][0];
      }
    }
  }

  fn integrate(
    scheme: RosenbrockScheme,
    lambda: f64,
    t_end: f64,
    n: usize,
  ) -> f64 {
    let tableau = scheme.tableau();
    let dt = t_end / n as f64;
    let mut u = [[1.0]];
    let mut fd = FiniteDifferenceJacobian::<1>::new();
    let mut precond = PreconditionerKind::<1>::Identity;
    let cfg = GmresConfig {
      restart: 1,
      max_restarts: 2,
      rel_tol: 1e-12,
      abs_tol: 1e-14,
    };
    for _ in 0..n {
      let mut jvp = LinearOde { lambda };
      let mut rhs = LinearOde { lambda };
      rosenbrock_step(
        &tableau,
        &mut jvp,
        &mut rhs,
        &mut fd,
        &mut precond,
        &cfg,
        dt,
        &mut u,
      );
    }
    u[0][0]
  }

  #[test]
  fn lin_implicit_euler_is_first_order() {
    let lambda = -2.0_f64;
    let t = 1.0_f64;
    let exact = (lambda * t).exp();
    let e_coarse =
      (integrate(RosenbrockScheme::LinearlyImplicitEuler, lambda, t, 20)
        - exact)
        .abs();
    let e_fine =
      (integrate(RosenbrockScheme::LinearlyImplicitEuler, lambda, t, 40)
        - exact)
        .abs();
    // Halving dt should roughly halve a first-order error.
    let ratio = e_coarse / e_fine;
    assert!(ratio > 1.7 && ratio < 2.4, "order-1 ratio was {ratio}");
  }

  #[test]
  fn ros2_is_second_order() {
    let lambda = -2.0_f64;
    let t = 1.0_f64;
    let exact = (lambda * t).exp();
    let e_coarse =
      (integrate(RosenbrockScheme::Ros2, lambda, t, 20) - exact).abs();
    let e_fine =
      (integrate(RosenbrockScheme::Ros2, lambda, t, 40) - exact).abs();
    // Halving dt should cut a second-order error by ~4×.
    let ratio = e_coarse / e_fine;
    assert!(ratio > 3.3, "order-2 ratio was {ratio} (want ~4)");
  }

  #[test]
  fn implicit_euler_unconditionally_stable_on_stiff_mode() {
    // A very stiff decay with a huge step: explicit Euler would blow up
    // (|1 + λdt| ≫ 1), backward/lin-implicit Euler must stay bounded → 0.
    let lambda = -1000.0_f64;
    let result =
      integrate(RosenbrockScheme::LinearlyImplicitEuler, lambda, 1.0, 5);
    assert!(result.abs() < 1.0, "stiff step diverged: {result}");
    assert!(result > 0.0, "decay should stay positive: {result}");
  }
}
