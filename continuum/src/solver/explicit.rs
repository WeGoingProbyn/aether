use crate::solver::fv::{FiniteVolumeSolver, Model};
use crate::geometry::Geometry;
use crate::topology::Topology;

pub fn step_explicit<Topo, Geom, M, const D: usize, const NV: usize>(
    solver: &mut FiniteVolumeSolver<Topo, Geom, M, D, NV>,
    dt: f64,
)
where
    Topo: Topology<D>,
    Geom: Geometry<D>,
    M: Model<D, NV>,
{
    let n = solver.u.len();
    let mut l = vec![[0.0; NV]; n];
    solver.compute_l(&solver.u, solver.time, &mut l);

    for i in 0..n {
        for m in 0..NV {
            solver.u[i][m] += dt * l[i][m];
        }
    }
    solver.time += dt;
}
