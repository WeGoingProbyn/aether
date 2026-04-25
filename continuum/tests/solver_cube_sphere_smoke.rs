//! Smoke test: drive `FvmSolver::step` on a `CubeSphere` mesh starting from
//! a constant state at rest. Verifies that the mesh is wired into the solver
//! correctly — mass and energy fluxes vanish identically at every face, so ρ
//! and E should remain *exactly* unchanged on every cell. Momentum should
//! stay near zero globally (pressure × normal sums to zero over the closed
//! sphere by panel-pair symmetry); per-cell momentum drift is allowed because
//! the world-frame area vectors are midpoint approximations on a curved grid.

use continuum::{
  boundary::{BoundaryRegistry, Transmissive},
  cube_sphere::CubeSphere,
  field::{FieldStorage, SoaField},
  geometry::{CellGeometry, CellId, CellMetrics, Point},
  model::{ConservationLaw, RusanovFlux},
  solver::{FvmSolver, SolverConfig, TimeIntegration},
  topology::BoundaryTag,
};

/// Minimal 3D compressible Euler (γ-law gas). State = [ρ, ρu, ρv, ρw, E].
/// Lives here, not in the library, because the user hasn't yet committed to
/// a specific atmospheric law shape — this is just enough to smoke-test the
/// mesh/solver wiring.
struct Euler3D {
  gamma: f64,
}

impl Euler3D {
  fn pressure(&self, state: &[f64; 5]) -> f64 {
    let rho = state[0];
    let inv_rho = 1.0 / rho;
    let ke = 0.5 * inv_rho
      * (state[1].powi(2) + state[2].powi(2) + state[3].powi(2));
    (self.gamma - 1.0) * (state[4] - ke)
  }
}

impl ConservationLaw<3, 5> for Euler3D {
  fn flux(&self, state: &[f64; 5]) -> [[f64; 5]; 3] {
    let rho = state[0];
    let u = state[1] / rho;
    let v = state[2] / rho;
    let w = state[3] / rho;
    let p = self.pressure(state);
    let h = state[4] + p;

    let fx = [state[1], state[1] * u + p, state[1] * v, state[1] * w, h * u];
    let fy = [state[2], state[2] * u, state[2] * v + p, state[2] * w, h * v];
    let fz = [state[3], state[3] * u, state[3] * v, state[3] * w + p, h * w];
    [fx, fy, fz]
  }

  fn max_wave_speed(&self, state: &[f64; 5]) -> f64 {
    let rho = state[0];
    let u = state[1] / rho;
    let v = state[2] / rho;
    let w = state[3] / rho;
    let p = self.pressure(state);
    let c = (self.gamma * p / rho).sqrt();
    (u * u + v * v + w * w).sqrt() + c
  }

  fn source(
    &self,
    _: &[f64; 5],
    _: &Point<3>,
    _: &CellMetrics<3>,
  ) -> [f64; 5] {
    [0.0; 5]
  }

  fn fix_state(&self, state: &mut [f64; 5]) {
    let floor = 1e-8;
    if state[0] < floor {
      state[0] = floor;
    }
    let rho = state[0];
    let ke = 0.5 / rho
      * (state[1].powi(2) + state[2].powi(2) + state[3].powi(2));
    if state[4] - ke < floor {
      state[4] = ke + floor;
    }
  }
}

/// Sum component `c` of state weighted by physical cell volume.
fn integrate_component(
  state: &SoaField<5>,
  mesh: &CubeSphere,
  c: usize,
) -> f64 {
  let mut total = 0.0;
  let mut s = [0.0; 5];
  for i in 0..mesh.cell_count() {
    let cell = CellId::from(i);
    state.state_into(cell, &mut s);
    let phys_vol = mesh.cell_volume(cell) * mesh.cell_metrics(cell).sqrt_metric;
    total += s[c] * phys_vol;
  }
  total
}

#[test]
fn cube_sphere_constant_state_at_rest() {
  let dims = [4, 4, 2];
  let mesh = CubeSphere::new(dims, 1.0, 2.0);

  let gamma = 1.4;
  let rho0 = 1.0;
  let p0 = 1.0;
  let e0 = p0 / (gamma - 1.0); // u=v=w=0 → E = p/(γ-1)
  let initial: [f64; 5] = [rho0, 0.0, 0.0, 0.0, e0];

  let n_cells = mesh.cell_count();
  let mut state = SoaField::<5>::from_fn(n_cells, |_| initial);
  let mut residual = SoaField::<5>::zeros(n_cells);

  let mut bcs = BoundaryRegistry::<3, 5>::new();
  bcs.register(BoundaryTag::Ground, Transmissive);
  bcs.register(BoundaryTag::AtmosphereEdge, Transmissive);

  let mut solver = FvmSolver::new(
    SolverConfig::new(0.5, 0.1, TimeIntegration::ForwardEuler),
    Euler3D { gamma },
    RusanovFlux,
  );

  let mass_before = integrate_component(&state, &mesh, 0);
  let energy_before = integrate_component(&state, &mesh, 4);

  let dt = solver.step(&mut state, &mut residual, &mesh, &bcs);
  assert!(dt > 0.0 && dt.is_finite(), "dt should be positive and finite, got {}", dt);

  // ----- Per-cell invariants: ρ and E exactly unchanged -----
  // Mass flux is ρu·n; with u = 0 everywhere, F_ρ = 0 at every face on every
  // panel. Same for energy flux (E+p)u·n. Per-cell ρ and E should not move
  // by even one ulp.
  let mut s = [0.0; 5];
  for i in 0..n_cells {
    let cell = CellId::from(i);
    state.state_into(cell, &mut s);
    let drho = (s[0] - rho0).abs();
    let de = (s[4] - e0).abs();
    assert!(
      drho < 1e-13,
      "cell {}: ρ drifted by {} (expected exact)", i, drho
    );
    assert!(
      de < 1e-13,
      "cell {}: E drifted by {} (expected exact)", i, de
    );
  }

  // ----- Global invariants: total mass and energy unchanged -----
  let mass_after = integrate_component(&state, &mesh, 0);
  let energy_after = integrate_component(&state, &mesh, 4);
  let mass_drift = (mass_after - mass_before).abs() / mass_before;
  let energy_drift = (energy_after - energy_before).abs() / energy_before;
  assert!(mass_drift < 1e-13, "Σρ drift = {}", mass_drift);
  assert!(energy_drift < 1e-13, "ΣE drift = {}", energy_drift);

  // ----- Global momentum: net force on a closed sphere is zero -----
  // Each Ground/AtmosphereEdge face contributes -p·n_out·A·sqrt_metric to
  // the residual. By panel-pair (XP/XN, YP/YN, ZP/ZN) symmetry of the cube
  // sphere, the boundary normals cancel in each component, so the global
  // sum should be ≈ 0. Per-cell momentum can be non-zero (curvature gives a
  // non-vanishing discrete pressure gradient) but the integral should not.
  for c in 1..=3 {
    let p_total = integrate_component(&state, &mesh, c);
    assert!(
      p_total.abs() < 1e-10 * mass_before,
      "Σρu_{} = {} (mass = {})", c, p_total, mass_before
    );
  }
}
