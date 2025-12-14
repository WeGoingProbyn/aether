use crate::geometry::{Geometry, VecN};
use crate::solver::fv::{FiniteVolumeSolver, Model, State};
use crate::solver::gmres::{gmres_left_precond, GmresConfig, LeftPreconditioner, LinearOperator};
use crate::topology::{Neighbor, Topology};

#[derive(Debug, Clone)]
pub struct NewtonConfig {
  pub max_iters: usize,
  pub tol: f64,        // relative (to initial residual)
  pub fd_epsilon: f64, // finite-difference epsilon for Jv
}

#[derive(Debug, Clone)]
pub struct ImplicitConfig {
  pub newton: NewtonConfig,
  pub gmres: GmresConfig,
}

#[derive(Debug, Clone)]
pub enum ImplicitStepError {
  NewtonNoConverge { iters: usize, rel_res: f64 },
  GmresFailed(String),
}

fn pack<const NV: usize>(u: &[State<NV>], out: &mut [f64]) {
  let mut k = 0usize;
  for cell in u {
    for m in 0..NV {
      out[k] = cell[m];
      k += 1;
    }
  }
}

fn unpack_add<const NV: usize>(x: &[f64], u: &mut [State<NV>], alpha: f64) {
  let mut k = 0usize;
  for cell in u {
    for m in 0..NV {
      cell[m] += alpha * x[k];
      k += 1;
    }
  }
}

fn vec_norm(v: &[f64]) -> f64 {
  let mut s = 0.0;
  for &a in v {
    s += a * a;
  }
  s.sqrt()
}

fn dot<const D: usize>(a: VecN<D>, b: VecN<D>) -> f64 {
  let mut s = 0.0;
  for i in 0..D {
    s += a[i] * b[i];
  }
  s
}
fn norm<const D: usize>(v: VecN<D>) -> f64 {
  dot::<D>(v, v).sqrt()
}

/// Diagonal (Jacobi) left preconditioner: y = M^{-1} x
#[derive(Debug, Clone)]
pub struct DiagonalPreconditioner {
  inv_diag: Vec<f64>,
}
impl DiagonalPreconditioner {
  pub fn new(inv_diag: Vec<f64>) -> Self {
    Self { inv_diag }
  }
}
impl LeftPreconditioner for DiagonalPreconditioner {
  fn dim(&self) -> usize {
    self.inv_diag.len()
  }
  fn apply_inv(&self, x: &[f64], y: &mut [f64]) {
    for i in 0..x.len() {
      y[i] = self.inv_diag[i] * x[i];
    }
  }
}

/// Approximate diagonal of BE Jacobian: diag(J) ≈ 1 + dt * diag(|L'|)
fn build_jacobi_precond<Topo, Geom, M, const D: usize, const NV: usize>(
  solver: &FiniteVolumeSolver<Topo, Geom, M, D, NV>,
  u: &[State<NV>],
  dt: f64,
  t: f64,
) -> DiagonalPreconditioner
where
  Topo: Topology<D>,
  Geom: Geometry<D>,
  M: Model<D, NV>,
{
  let grid = solver.topo.grid();
  let ncell = u.len();
  let n = ncell * NV;

  let mut diag_l = vec![0.0; n];

  for cell in grid.iter_cells() {
    let i = grid.cell_linear(cell);
    let ui = u[i];

    for face in solver.topo.faces_of_cell(cell) {
      let s = solver.geom.face_area_vector(face);
      let area = norm::<D>(s);
      if area == 0.0 {
        continue;
      }

      let mut n_unit = [0.0; D];
      for d in 0..D {
        n_unit[d] = s[d] / area;
      }

      let neighbor = solver.topo.neighbor_across(face);
      let ur = match neighbor {
        Neighbor::Cell(nc) => {
          let j = grid.cell_linear(nc);
          u[j]
        }
        Neighbor::Boundary(bid) => {
          let xf = solver.geom.face_center_x(face);
          solver.model.boundary_state(bid, &ui, xf, t)
        }
      };

      // Rusanov dissipation coefficient ~ 0.5 * a
      let a = solver
        .model
        .max_wave_speed(&ui, n_unit)
        .max(solver.model.max_wave_speed(&ur, n_unit));

      // dist_n estimate for diffusion coefficient
      let x_l = solver.geom.cell_center_x(face.cell);
      let x_r = match neighbor {
        Neighbor::Cell(nc) => solver.geom.cell_center_x(nc),
        Neighbor::Boundary(_) => {
          let xf = solver.geom.face_center_x(face);
          let mut xr = [0.0; D];
          for d in 0..D {
            xr[d] = 2.0 * xf[d] - x_l[d];
          }
          xr
        }
      };
      let mut dx = [0.0; D];
      for d in 0..D {
        dx[d] = x_r[d] - x_l[d];
      }
      let dist_n = dot::<D>(dx, n_unit).abs().max(1e-12);

      // diffusion lipschitz coefficient c (componentwise)
      let c = solver.model.diffusion_lipschitz_coeff(&ui, &ur, n_unit, dist_n, t);

      // accumulate diagonal magnitude for this cell dof
      // |∂(flux)/∂U_i| ~ area * (0.5*a + c)
      for m in 0..NV {
        let idx = i * NV + m;
        diag_l[idx] += area * (0.5 * a + c[m].max(0.0));
      }
    }

    // divide by volume (because L includes 1/vol)
    let vol = solver.geom.cell_volume(cell);
    for m in 0..NV {
      diag_l[i * NV + m] /= vol;
    }
  }

  // Backward Euler Jacobian diagonal approx: 1 + dt*diag_l
  let mut inv_diag = vec![0.0; n];
  for k in 0..n {
    let d = 1.0 + dt * diag_l[k];
    inv_diag[k] = 1.0 / d.max(1e-12);
  }

  DiagonalPreconditioner::new(inv_diag)
}

/// Matrix-free Jacobian of G(U) = U - Uold - dt*L(U)
struct JacobianFreeOp<'a, Topo, Geom, M, const D: usize, const NV: usize>
where
    Topo: Topology<D>,
    Geom: Geometry<D>,
    M: Model<D, NV>,
  {
    solver: &'a FiniteVolumeSolver<Topo, Geom, M, D, NV>,
    dt: f64,
    u_base: &'a [State<NV>],
    l_base_flat: &'a [f64],

    // scratch
    u_pert: Vec<State<NV>>,
    l_pert: Vec<State<NV>>,
    l_pert_flat: Vec<f64>,

    eps: f64,
    t: f64,
  }

impl<'a, Topo, Geom, M, const D: usize, const NV: usize> JacobianFreeOp<'a, Topo, Geom, M, D, NV>
where
  Topo: Topology<D>,
  Geom: Geometry<D>,
  M: Model<D, NV>,
{
  fn new(
    solver: &'a FiniteVolumeSolver<Topo, Geom, M, D, NV>,
    u_base: &'a [State<NV>],
    l_base_flat: &'a [f64],
    dt: f64,
    eps: f64,
    t: f64,
  ) -> Self {
    let ncell = u_base.len();
    Self {
      solver,
      dt,
      u_base,
      l_base_flat,
      u_pert: vec![[0.0; NV]; ncell],
      l_pert: vec![[0.0; NV]; ncell],
      l_pert_flat: vec![0.0; ncell * NV],
      eps,
      t,
    }
  }

  fn dim(&self) -> usize {
    self.u_base.len() * NV
  }

  fn apply(&mut self, x: &[f64], y: &mut [f64]) {
    let ncell = self.u_base.len();
    let eps = self.eps;

    self.u_pert.copy_from_slice(self.u_base);
    unpack_add::<NV>(x, &mut self.u_pert, eps);

    self.solver
      .compute_l(&self.u_pert, self.t, &mut self.l_pert);
    pack::<NV>(&self.l_pert, &mut self.l_pert_flat);

    // y = x - dt * (L(u+eps x)-L(u))/eps
    for i in 0..(ncell * NV) {
      let jl_v = (self.l_pert_flat[i] - self.l_base_flat[i]) / eps;
      y[i] = x[i] - self.dt * jl_v;
    }
  }
}

/// Adapter so GMRES can call our mutable operator
struct OpAdapter<'a, T>(std::cell::RefCell<&'a mut T>);
impl<'a, Topo, Geom, M, const D: usize, const NV: usize> LinearOperator
  for OpAdapter<'a, JacobianFreeOp<'a, Topo, Geom, M, D, NV>>
where
  Topo: Topology<D>,
  Geom: Geometry<D>,
  M: Model<D, NV>,
{
  fn dim(&self) -> usize {
    self.0.borrow().dim()
  }
  fn apply(&self, x: &[f64], y: &mut [f64]) {
    self.0.borrow_mut().apply(x, y);
  }
}

/// Backward Euler step using Newton–(preconditioned) GMRES.
/// Returns Err on failure and does NOT mutate solver state on failure.
pub fn step_implicit_backward_euler<Topo, Geom, M, const D: usize, const NV: usize>(
  solver: &mut FiniteVolumeSolver<Topo, Geom, M, D, NV>,
  dt: f64,
  cfg: ImplicitConfig,
) -> Result<(), ImplicitStepError>
where
  Topo: Topology<D>,
  Geom: Geometry<D>,
  M: Model<D, NV>,
{
  let ncell = solver.u.len();
  let n = ncell * NV;

  let u_old = solver.u.clone();
  let mut u = solver.u.clone();

  let mut l_base = vec![[0.0; NV]; ncell];
  let mut l_base_flat = vec![0.0; n];
  let mut g = vec![0.0; n];
  let mut rhs = vec![0.0; n];
  let mut delta = vec![0.0; n];

  let mut uold_flat = vec![0.0; n];
  pack::<NV>(&u_old, &mut uold_flat);

  let mut u_flat = vec![0.0; n];

  let mut g0_norm: Option<f64> = None;
  let mut last_rel = f64::INFINITY;

  for _newton_it in 0..cfg.newton.max_iters {
    solver.compute_l(&u, solver.time, &mut l_base);
    pack::<NV>(&l_base, &mut l_base_flat);
    pack::<NV>(&u, &mut u_flat);

    for i in 0..n {
      g[i] = u_flat[i] - uold_flat[i] - dt * l_base_flat[i];
      rhs[i] = -g[i];
    }

    let gnorm = vec_norm(&g);
    let refn = g0_norm.get_or_insert(gnorm.max(1e-30));
    last_rel = gnorm / *refn;

    if last_rel <= cfg.newton.tol {
      solver.u = u;
      solver.time += dt;
      return Ok(());
    }

    // Build left preconditioner from current u
    let precond = build_jacobi_precond::<Topo, Geom, M, D, NV>(solver, &u, dt, solver.time);

    // Jacobian-free operator
    let mut op = JacobianFreeOp::new(
      solver,
      &u,
      &l_base_flat,
      dt,
      cfg.newton.fd_epsilon,
      solver.time,
    );

    let adapter = OpAdapter(std::cell::RefCell::new(&mut op));

    delta.fill(0.0);
    let gm = gmres_left_precond(&adapter, &precond, &rhs, &mut delta, &cfg.gmres);
    if let Err(e) = gm {
      return Err(ImplicitStepError::GmresFailed(e));
    }

    // Update Newton iterate
    unpack_add::<NV>(&delta, &mut u, 1.0);
  }

  Err(ImplicitStepError::NewtonNoConverge {
    iters: cfg.newton.max_iters,
    rel_res: last_rel,
  })
}

