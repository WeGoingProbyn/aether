use crate::geometry::{Geometry, VecN};
use crate::grid::grid::{CellId, FaceId};
use crate::topology::{BoundaryId, Neighbor, Topology};

pub type State<const NV: usize> = [f64; NV];
pub type Flux<const D: usize, const NV: usize> = [[f64; D]; NV];

fn dot<const D: usize>(a: VecN<D>, b: VecN<D>) -> f64 {
    let mut s = 0.0;
    for i in 0..D { s += a[i] * b[i]; }
    s
}
fn norm<const D: usize>(v: VecN<D>) -> f64 { dot::<D>(v, v).sqrt() }

fn zero_state<const NV: usize>() -> State<NV> { [0.0; NV] }

fn add_scaled_state<const NV: usize>(a: &mut State<NV>, b: &State<NV>, s: f64) {
    for m in 0..NV { a[m] += s * b[m]; }
}

pub trait Model<const D: usize, const NV: usize> {
    fn flux(&self, u: &State<NV>) -> Flux<D, NV>;
    fn max_wave_speed(&self, u: &State<NV>, n_unit: VecN<D>) -> f64;

    fn boundary_state(
        &self,
        boundary: BoundaryId,
        interior: &State<NV>,
        x_face: VecN<D>,
        t: f64,
    ) -> State<NV>;

    fn source(&self, _u: &State<NV>, _x_cell: VecN<D>, _t: f64) -> State<NV> {
        zero_state::<NV>()
    }

    /// Optional: diffusive normal flux per unit area (component-wise F_diff · n).
    /// `dist_n` is the distance between left/right sample points along n.
    fn diffusive_flux_unit_normal(
        &self,
        _ul: &State<NV>,
        _ur: &State<NV>,
        _n_unit: VecN<D>,
        _dist_n: f64,
        _t: f64,
    ) -> State<NV> {
        zero_state::<NV>()
    }
}

/// Choose explicit vs implicit at construction.
#[derive(Debug, Clone)]
pub enum TimeIntegrator {
    ExplicitEuler,
    ImplicitBackwardEuler(crate::solver::implicit::ImplicitConfig),
}

/// Main solver object: owns topology, geometry, model, state, integrator choice.
#[derive(Debug, Clone)]
pub struct FiniteVolumeSolver<Topo, Geom, M, const D: usize, const NV: usize>
where
    Topo: Topology<D>,
    Geom: Geometry<D>,
    M: Model<D, NV>,
{
    pub topo: Topo,
    pub geom: Geom,
    pub model: M,

    pub u: Vec<State<NV>>,
    pub time: f64,

    pub integrator: TimeIntegrator,
}

impl<Topo, Geom, M, const D: usize, const NV: usize> FiniteVolumeSolver<Topo, Geom, M, D, NV>
where
    Topo: Topology<D>,
    Geom: Geometry<D>,
    M: Model<D, NV>,
{
    pub fn new(topo: Topo, geom: Geom, model: M, integrator: TimeIntegrator) -> Self {
        let n = topo.grid().cell_count();
        Self {
            topo,
            geom,
            model,
            u: vec![[0.0; NV]; n],
            time: 0.0,
            integrator,
        }
    }

    pub fn initialize_with<F>(&mut self, mut f: F)
    where
        F: FnMut(VecN<D>) -> State<NV>,
    {
        let grid = self.topo.grid();
        for cell in grid.iter_cells() {
            let idx = grid.cell_linear(cell);
            let x = self.geom.cell_center_x(cell);
            self.u[idx] = f(x);
        }
    }

    /// Public step entry point. Delegates to explicit or implicit module.
    pub fn step(&mut self, dt: f64) {
        match &self.integrator {
            TimeIntegrator::ExplicitEuler => {
                crate::solver::explicit::step_explicit(self, dt);
            }
            TimeIntegrator::ImplicitBackwardEuler(cfg) => {
                crate::solver::implicit::step_implicit_backward_euler(self, dt, cfg.clone());
            }
        }
    }

    /// Compute L(U) into `out` such that dU/dt = L(U).
    /// This is the shared FV spatial operator used by both explicit and implicit.
    pub fn compute_l(&self, u_in: &[State<NV>], t: f64, out: &mut [State<NV>]) {
        let grid = self.topo.grid();
        debug_assert_eq!(u_in.len(), grid.cell_count());
        debug_assert_eq!(out.len(), grid.cell_count());

        // zero out
        for i in 0..out.len() { out[i] = [0.0; NV]; }

        for cell in grid.iter_cells() {
            let i = grid.cell_linear(cell);
            let ui = u_in[i];

            // Source term (cell-averaged) adds directly to dU/dt
            let x_cell = self.geom.cell_center_x(cell);
            let src = self.model.source(&ui, x_cell, t);
            add_scaled_state::<NV>(&mut out[i], &src, 1.0);

            // Flux divergence
            for face in self.topo.faces_of_cell(cell) {
                let s = self.geom.face_area_vector(face); // n*A outward
                let area = norm::<D>(s);
                if area == 0.0 { continue; }

                let mut n_unit = [0.0; D];
                for d in 0..D { n_unit[d] = s[d] / area; }

                let ul = ui;

                let neighbor = self.topo.neighbor_across(face);
                let ur = match neighbor {
                    Neighbor::Cell(nc) => {
                        let j = grid.cell_linear(nc);
                        u_in[j]
                    }
                    Neighbor::Boundary(bid) => {
                        let xf = self.geom.face_center_x(face);
                        self.model.boundary_state(bid, &ul, xf, t)
                    }
                };

                // --- Advective numerical flux: Rusanov on F·n ---
                let fl = self.model.flux(&ul);
                let fr = self.model.flux(&ur);

                let mut fn_l = [0.0; NV];
                let mut fn_r = [0.0; NV];
                for m in 0..NV {
                    fn_l[m] = dot::<D>(fl[m], n_unit);
                    fn_r[m] = dot::<D>(fr[m], n_unit);
                }

                let a = self.model.max_wave_speed(&ul, n_unit).max(self.model.max_wave_speed(&ur, n_unit));

                // fhat_n is flux per unit area through the face (already dotted with n)
                let mut fhat_n = [0.0; NV];
                for m in 0..NV {
                    fhat_n[m] = 0.5 * (fn_l[m] + fn_r[m]) - 0.5 * a * (ur[m] - ul[m]);
                }

                // --- Optional diffusion: add (F_diff · n) ---
                // approximate normal distance using cell centers
                let x_l = self.geom.cell_center_x(face.cell);
                let x_r = match neighbor {
                    Neighbor::Cell(nc) => self.geom.cell_center_x(nc),
                    Neighbor::Boundary(_) => {
                        // mirror across face center (reasonable first approximation)
                        let xf = self.geom.face_center_x(face);
                        let mut xr = [0.0; D];
                        for d in 0..D { xr[d] = 2.0 * xf[d] - x_l[d]; }
                        xr
                    }
                };
                let mut dx = [0.0; D];
                for d in 0..D { dx[d] = x_r[d] - x_l[d]; }
                let dist_n = dot::<D>(dx, n_unit).abs().max(1e-12);

                let f_diff_n = self.model.diffusive_flux_unit_normal(&ul, &ur, n_unit, dist_n, t);
                for m in 0..NV { fhat_n[m] += f_diff_n[m]; }

                // Integrated flux = fhat_n * area
                for m in 0..NV {
                    out[i][m] -= (fhat_n[m] * area);
                }
            }

            // divide by physical volume
            let vol = self.geom.cell_volume(cell);
            let inv_vol = 1.0 / vol;
            for m in 0..NV { out[i][m] *= inv_vol; }
        }
    }
}
