use crate::solver::fv::{FiniteVolumeSolver, Model, State};
use crate::solver::gmres::{gmres, GmresConfig, LinearOperator};
use crate::geometry::Geometry;
use crate::topology::Topology;

#[derive(Debug, Clone)]
pub struct NewtonConfig {
    pub max_iters: usize,    // e.g. 10
    pub tol: f64,            // e.g. 1e-8 (relative)
    pub fd_epsilon: f64,     // e.g. 1e-6
}

#[derive(Debug, Clone)]
pub struct ImplicitConfig {
    pub newton: NewtonConfig,
    pub gmres: GmresConfig,
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
    // u += alpha * x (x is flat)
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
    for &a in v { s += a * a; }
    s.sqrt()
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

    // current Newton point:
    u_base: &'a [State<NV>],

    // cached L(U_base) flattened:
    l_base_flat: &'a [f64],

    // scratch (owned outside, passed by &mut through apply via interior mutability substitute):
    // we’ll store mutable scratch in a RefCell-like pattern by making apply take &mut self.
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
}

impl<'a, Topo, Geom, M, const D: usize, const NV: usize> LinearOperator
    for JacobianFreeOp<'a, Topo, Geom, M, D, NV>
where
    Topo: Topology<D>,
    Geom: Geometry<D>,
    M: Model<D, NV>,
{
    fn dim(&self) -> usize {
        self.u_base.len() * NV
    }

    fn apply(&self, _x: &[f64], _y: &mut [f64]) {
        unreachable!("This impl is for &mut self below");
    }
}

/// We need a mutable apply (scratch), so wrap it:
struct JacobianFreeOpMut<'a, Topo, Geom, M, const D: usize, const NV: usize>
where
    Topo: Topology<D>,
    Geom: Geometry<D>,
    M: Model<D, NV>,
{
    inner: JacobianFreeOp<'a, Topo, Geom, M, D, NV>,
}

impl<'a, Topo, Geom, M, const D: usize, const NV: usize> JacobianFreeOpMut<'a, Topo, Geom, M, D, NV>
where
    Topo: Topology<D>,
    Geom: Geometry<D>,
    M: Model<D, NV>,
{
    fn dim(&self) -> usize { self.inner.dim() }

    fn apply(&mut self, x: &[f64], y: &mut [f64]) {
        let ncell = self.inner.u_base.len();
        let eps = self.inner.eps;

        // u_pert = u_base + eps * x
        self.inner.u_pert.copy_from_slice(self.inner.u_base);
        unpack_add::<NV>(x, &mut self.inner.u_pert, eps);

        // L(u_pert)
        self.inner.solver.compute_l(&self.inner.u_pert, self.inner.t, &mut self.inner.l_pert);
        pack::<NV>(&self.inner.l_pert, &mut self.inner.l_pert_flat);

        // y = x - dt * (L(u_pert) - L(u_base))/eps
        for i in 0..(ncell * NV) {
            let jl_v = (self.inner.l_pert_flat[i] - self.inner.l_base_flat[i]) / eps;
            y[i] = x[i] - self.inner.dt * jl_v;
        }
    }
}

/// Backward Euler step using Newton–GMRES.
pub fn step_implicit_backward_euler<Topo, Geom, M, const D: usize, const NV: usize>(
    solver: &mut FiniteVolumeSolver<Topo, Geom, M, D, NV>,
    dt: f64,
    cfg: ImplicitConfig,
)
where
    Topo: Topology<D>,
    Geom: Geometry<D>,
    M: Model<D, NV>,
{
    let ncell = solver.u.len();
    let n = ncell * NV;

    let u_old = solver.u.clone();
    let mut u = solver.u.clone(); // Newton iterate (initial guess: old)

    let mut l_base = vec![[0.0; NV]; ncell];
    let mut l_base_flat = vec![0.0; n];
    let mut g = vec![0.0; n];      // G(U)
    let mut rhs = vec![0.0; n];    // -G(U)
    let mut delta = vec![0.0; n];  // Newton step

    let uold_flat = {
        let mut tmp = vec![0.0; n];
        pack::<NV>(&u_old, &mut tmp);
        tmp
    };

    let mut u_flat = vec![0.0; n];

    let mut g0_norm: Option<f64> = None;

    for newton_it in 0..cfg.newton.max_iters {
        // L(u)
        solver.compute_l(&u, solver.time, &mut l_base);
        pack::<NV>(&l_base, &mut l_base_flat);
        pack::<NV>(&u, &mut u_flat);

        // G(U) = U - Uold - dt*L(U)
        for i in 0..n {
            g[i] = u_flat[i] - uold_flat[i] - dt * l_base_flat[i];
            rhs[i] = -g[i];
        }

        let gnorm = vec_norm(&g);
        let refn = g0_norm.get_or_insert(gnorm.max(1e-30));
        let rel = gnorm / *refn;

        // Converged?
        if rel <= cfg.newton.tol {
            solver.u = u;
            solver.time += dt;
            return;
        }

        // Solve J_G * delta = -G with GMRES (matrix-free)
        delta.fill(0.0);

        let mut op = JacobianFreeOpMut {
            inner: JacobianFreeOp::new(
                solver,
                &u,
                &l_base_flat,
                dt,
                cfg.newton.fd_epsilon,
                solver.time, // time in L(U,t)
            ),
        };

        // Wrap mut op in a tiny adapter for gmres expecting &dyn LinearOperator:
        // We'll implement a local vtable object:
        struct OpAdapter<'a, T>(std::cell::RefCell<&'a mut T>);
        impl<'a, Topo, Geom, M, const D: usize, const NV: usize> LinearOperator
            for OpAdapter<'a, JacobianFreeOpMut<'a, Topo, Geom, M, D, NV>>
        where
            Topo: Topology<D>,
            Geom: Geometry<D>,
            M: Model<D, NV>,
        {
            fn dim(&self) -> usize { self.0.borrow().dim() }
            fn apply(&self, x: &[f64], y: &mut [f64]) {
                self.0.borrow_mut().apply(x, y);
            }
        }

        let adapter = OpAdapter(std::cell::RefCell::new(&mut op));

        let gmres_res = gmres(&adapter, &rhs, &mut delta, &cfg.gmres);
        if let Err(e) = gmres_res {
            // If GMRES fails, bail out (you can choose to fall back to explicit, reduce dt, etc.)
            panic!("Newton-GMRES failed at iter {}: {}", newton_it, e);
        }

        // Update: U <- U + delta
        unpack_add::<NV>(&delta, &mut u, 1.0);
    }

    panic!("Implicit step failed to converge within max Newton iterations");
}
