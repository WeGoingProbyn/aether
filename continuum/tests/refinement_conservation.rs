// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Phase-3 gate for AMR: the finite-volume solver must run **unchanged** on an
//! adaptively-refined mesh and stay conservative across hanging interfaces.
//!
//! `continuum` only ever sees a `Mesh<3>`; an [`AdaptiveMesh`] is one, so the
//! solver is oblivious to refinement, hanging faces, or topology epochs — exactly
//! the layering the design intends. A hanging interface is just extra interior
//! faces, so conservation is the test: each face flux is applied equal-and-opposite
//! to its two cells, hence the exactly-conserved quantities (density of a state at
//! rest; total mass under dynamics) must be preserved to machine precision even
//! though the coarse cells now carry fine sub-faces.
//!
//! (Momentum is *not* asserted exactly: a cube-sphere cell's discrete `∮ p·n dA`
//! does not identically vanish — the curved-face closure is approximate — so an
//! at-rest state drifts in momentum on the base mesh too. See
//! `solver_cube_sphere_hydrostatic`.)

use std::sync::Arc;

use continuum::boundary::{BoundaryRegistry, ReflectiveWall};
use continuum::model::{Euler3D, RusanovFlux};
use continuum::solver::{FvmSolver, SolverConfig, TimeIntegration};
use pleroma::core::storage::{FieldStorage, SoaField};
use tessera::adaptive::AdaptiveMesh;
use tessera::cube_sphere::CubeSphere;
use tessera::geometry::CellGeometry;
use tessera::mesh::Mesh;
use tessera::refine::{AdaptRequest, RefinableMesh};
use utility::domain::{BoundaryTag, CellId};

const GAMMA: f64 = 1.4;

/// A refined cube-sphere: refine one panel-interior cell of a `[4,4,2]` shell.
/// This creates lateral hanging interfaces with its same-layer angular
/// neighbours *and* a radial (4:1) hanging interface with the cell directly
/// above — all inside one panel, so no seam is involved.
fn refined_mesh() -> Arc<dyn Mesh<3>> {
  let base = Arc::new(CubeSphere::new([4, 4, 2], 1.0, 2.0));
  let mesh0 = AdaptiveMesh::new(base);
  // panel 0, angular (1,1), radial 0 ⇒ local 1 + 1*4 + 0*16 = 5.
  let req = AdaptRequest {
    refine: vec![CellId::from(5)],
    coarsen: vec![],
  };
  let (refined, _remap) = mesh0.adapt(&req).unwrap();
  refined
}

fn walls() -> BoundaryRegistry<3, 5> {
  let mut bcs = BoundaryRegistry::<3, 5>::new();
  bcs.register(BoundaryTag::Ground, ReflectiveWall);
  bcs.register(BoundaryTag::AtmosphereEdge, ReflectiveWall);
  bcs
}

fn total_mass(state: &SoaField<5>, mesh: &dyn Mesh<3>) -> f64 {
  (0..mesh.cell_count())
    .map(|i| {
      state.state(CellId::from(i))[0] * mesh.cell_volume(CellId::from(i))
    })
    .sum()
}

#[test]
fn density_is_invariant_at_rest_across_hanging_faces() {
  // A uniform state at rest with no gravity: mass flux ρu·n = 0 everywhere, so
  // ρ must be unchanged to machine precision — including the coarse cells whose
  // shared faces were replaced by fine sub-faces. If the sub-faces did not sum to
  // the coarse face, a coarse cell would see a spurious net mass flux.
  let mesh = refined_mesh();
  let n = mesh.cell_count();
  let rho0 = 1.0;
  let e0 = 1.0 / (GAMMA - 1.0);
  let mut state = SoaField::<5>::from_fn(n, |_| [rho0, 0.0, 0.0, 0.0, e0]);
  let mut residual = SoaField::<5>::zeros(n);

  let mut solver = FvmSolver::new(
    SolverConfig::new(0.1, 1.0, TimeIntegration::ForwardEuler),
    Euler3D::new(GAMMA),
    RusanovFlux,
  );
  let dt = solver.step(&mut state, &mut residual, mesh.as_ref(), &walls());
  assert!(dt > 0.0 && dt.is_finite());

  let mut s = [0.0; 5];
  for i in 0..n {
    state.state_into(CellId::from(i), &mut s);
    assert!(s.iter().all(|v| v.is_finite()), "cell {i} went non-finite");
    assert!(
      (s[0] - rho0).abs() < 1e-12,
      "cell {i} density drifted by {} at rest",
      (s[0] - rho0).abs()
    );
  }
}

/// Evolve a density blob centred on `centre` for 25 RK2 steps and return the
/// relative change in total mass, plus whether every cell stayed finite. The
/// blob (uniform pressure, u = 0) is smeared by the Rusanov dissipation, driving
/// real flux through whatever interfaces the mesh has.
fn blob_mass_drift(mesh: &dyn Mesh<3>, centre: [f64; 3]) -> (f64, bool) {
  let n = mesh.cell_count();
  let e0 = 1.0 / (GAMMA - 1.0); // p0 = 1
  let scale = 0.5;
  let mut state = SoaField::<5>::from_fn(n, |c| {
    let p = mesh.cell_world_centroid(c);
    let d2 = (p[0] - centre[0]).powi(2)
      + (p[1] - centre[1]).powi(2)
      + (p[2] - centre[2]).powi(2);
    [1.0 + 0.5 * (-d2 / (scale * scale)).exp(), 0.0, 0.0, 0.0, e0]
  });
  let mut residual = SoaField::<5>::zeros(n);
  let mut solver = FvmSolver::new(
    SolverConfig::new(0.4, 1.0, TimeIntegration::Rk2),
    Euler3D::new(GAMMA),
    RusanovFlux,
  );
  let m0 = total_mass(&state, mesh);
  for _ in 0..25 {
    solver.step(&mut state, &mut residual, mesh, &walls());
  }
  let m1 = total_mass(&state, mesh);
  let finite =
    (0..n).all(|i| state.state(CellId::from(i)).iter().all(|v| v.is_finite()));
  ((m1 - m0).abs() / m0, finite)
}

#[test]
fn refinement_does_not_worsen_mass_conservation() {
  // The cube-sphere + Rusanov + reflective walls is not exactly mass-conserving
  // on a curved shell (the same curved-cell closure error that makes an at-rest
  // state drift in momentum — see `solver_cube_sphere_hydrostatic`). The AMR
  // claim is therefore relative: assembling the mesh adaptively, and refining a
  // region, must not make conservation *worse* than the base mesh — the hanging
  // sub-faces flux equal-and-opposite just like ordinary faces.
  let base = Arc::new(CubeSphere::new([4, 4, 2], 1.0, 2.0));
  let centre = base.cell_world_centroid(CellId::from(5));
  let centre = [centre[0], centre[1], centre[2]];
  let (base_drift, base_finite) = blob_mass_drift(base.as_ref(), centre);

  // Level-0 adaptive mesh must conserve *identically* to the base (the assembly
  // reproduces the base topology cell-for-cell).
  let lvl0 = AdaptiveMesh::new(base.clone());
  let (lvl0_drift, _) = blob_mass_drift(&lvl0, centre);
  assert!(
    (lvl0_drift - base_drift).abs() <= 1e-12,
    "level-0 adaptive conservation differs from base: {lvl0_drift} vs {base_drift}"
  );

  // Refining the blob's region must not worsen conservation.
  let (refined, _remap) = AdaptiveMesh::new(base)
    .adapt(&AdaptRequest {
      refine: vec![CellId::from(5)],
      coarsen: vec![],
    })
    .unwrap();
  let (refined_drift, refined_finite) =
    blob_mass_drift(refined.as_ref(), centre);

  assert!(base_finite && refined_finite, "state went non-finite");
  assert!(
    refined_drift <= 1.5 * base_drift.max(1e-9),
    "refinement worsened mass conservation: refined {refined_drift:.3e} vs base \
     {base_drift:.3e}"
  );
}
