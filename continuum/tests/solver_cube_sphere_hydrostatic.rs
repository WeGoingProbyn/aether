//! Validates the per-cell radial gravity machinery on the cube sphere:
//!   * `CubeSphere::cell_world_centroid` returns sane world coordinates.
//!   * `CubeSphere::radial_gravity_field` produces a vector of correct length
//!     with the right inward-radial direction and magnitude.
//!   * `Euler3D::with_per_cell_gravity` indexes into the field correctly,
//!     producing a momentum source that points toward the planet centre.
//!   * One step from a uniform state at rest leaves ρ unchanged (mass flux
//!     is still zero everywhere) and gives every cell inward momentum.
//!
//! Long-term hydrostatic stability is *not* tested here — that requires a
//! well-balanced pressure-gradient discretisation that the current Rusanov
//! flux doesn't implement. Without it, even an exact hydrostatic state on a
//! sphere drifts because the discrete `∮ p·n dA` over a curved cell doesn't
//! identically cancel the gravity source.

use continuum::{
  boundary::{BoundaryRegistry, ReflectiveWall},
  cube_sphere::CubeSphere,
  field::{FieldStorage, SoaField},
  geometry::{CellGeometry, CellId},
  model::{Euler3D, RusanovFlux},
  solver::{FvmSolver, SolverConfig, TimeIntegration},
  topology::BoundaryTag,
};

#[test]
fn radial_gravity_field_is_inward_radial_with_correct_magnitude() {
  let mesh = CubeSphere::new([4, 4, 4], 1.0, 2.0);
  let g = 9.81;
  let field = mesh.radial_gravity_field(g);

  assert_eq!(field.len(), mesh.cell_count());

  for (i, gv) in field.iter().enumerate() {
    let mag = (gv[0].powi(2) + gv[1].powi(2) + gv[2].powi(2)).sqrt();
    assert!(
      (mag - g).abs() < 1e-10,
      "cell {}: |gravity| = {} (expected {})",
      i,
      mag,
      g
    );

    // Should point opposite to the world centroid (inward radial).
    let centroid = mesh.cell_world_centroid(CellId::from(i));
    let r =
      (centroid[0].powi(2) + centroid[1].powi(2) + centroid[2].powi(2)).sqrt();
    let r_hat = [centroid[0] / r, centroid[1] / r, centroid[2] / r];
    let dot = gv[0] * r_hat[0] + gv[1] * r_hat[1] + gv[2] * r_hat[2];
    assert!(
      (dot + g).abs() < 1e-10,
      "cell {}: gravity·r̂ = {} (expected -{})",
      i,
      dot,
      g
    );
  }
}

#[test]
fn one_step_with_radial_gravity_drives_inward_momentum() {
  // Uniform state at rest + radial gravity ⇒ after one step every cell
  // should gain momentum pointing toward the planet centre. ρ should stay
  // exactly unchanged (mass flux is still zero with u = 0).
  let dims = [4, 4, 4];
  let r_inner = 1.0;
  let r_outer = 2.0;
  let mesh = CubeSphere::new(dims, r_inner, r_outer);
  let n_cells = mesh.cell_count();

  let gamma = 1.4;
  let rho0 = 1.0;
  let p0 = 1.0;
  let e0 = p0 / (gamma - 1.0);
  let initial: [f64; 5] = [rho0, 0.0, 0.0, 0.0, e0];
  let mut state = SoaField::<5>::from_fn(n_cells, |_| initial);
  let mut residual = SoaField::<5>::zeros(n_cells);

  let g = 1.0;
  let gravity = mesh.radial_gravity_field(g);

  let mut bcs = BoundaryRegistry::<3, 5>::new();
  bcs.register(BoundaryTag::Ground, ReflectiveWall);
  bcs.register(BoundaryTag::AtmosphereEdge, ReflectiveWall);

  let mut solver = FvmSolver::new(
    SolverConfig::new(0.1, 1.0, TimeIntegration::ForwardEuler),
    Euler3D::with_per_cell_gravity(gamma, gravity),
    RusanovFlux,
  );

  let dt = solver.step(&mut state, &mut residual, &mesh, &bcs);
  assert!(dt > 0.0 && dt.is_finite());

  let mut s = [0.0; 5];
  for i in 0..n_cells {
    let cell = CellId::from(i);
    state.state_into(cell, &mut s);

    // ρ should be exactly unchanged (mass flux ρ·u·n = 0 when u = 0).
    let drho = (s[0] - rho0).abs();
    assert!(
      drho < 1e-13,
      "cell {} ρ drifted by {} from initial {}",
      i,
      drho,
      rho0
    );

    // Momentum should point inward: dot with -r̂ should be positive.
    let centroid = mesh.cell_world_centroid(cell);
    let r =
      (centroid[0].powi(2) + centroid[1].powi(2) + centroid[2].powi(2)).sqrt();
    let inward_dot =
      -(s[1] * centroid[0] + s[2] * centroid[1] + s[3] * centroid[2]) / r;
    assert!(
      inward_dot > 0.0,
      "cell {}: momentum has no inward component (inward·ρu = {})",
      i,
      inward_dot
    );
  }
}
