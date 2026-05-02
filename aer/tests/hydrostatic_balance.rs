// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Cosmo-seeded hydrostatic IC should remain hydrostatic for a few
//! ticks when the dynamics step uses `BackgroundCorrectionMode::CurrentState`
//! (which captures the IC's discrete residual as a fixed source). This
//! is the smallest end-to-end check that aer's MVP wiring is sane.

use std::sync::Arc;

use aer::{AtmosphereModel, AtmosphereShellLayout};
use aether::core::world_constants_from_seed;
use cosmo::factory as cosmo_factory;
use nexus::{FieldStorage, MeshKey, Nexus, Pleroma, SoaField, WorldId};
use tessera::{
  cube_sphere::CubeSphere, geometry::CellGeometry, mesh::Mesh,
  world_mesh::Tessera,
};
use utility::{domain::CellId, thread::pool::Pool};

#[test]
fn hydrostatic_state_remains_hydrostatic_under_background_correction() {
  let seed = cosmo_factory::earth();
  let primary = cosmo_factory::sun();
  let constants = world_constants_from_seed(&seed, Some(&primary));

  let layout = AtmosphereShellLayout::new(&constants, 20_000.0, 10_000.0)
    .expect("shell layout for earth-ish world");
  let atmosphere_spec = layout.atmosphere_shell_spec([2, 2], 4);
  let mesh = Arc::new(CubeSphere::shell(atmosphere_spec));
  let mut tessera = Tessera::new();
  let mesh_dyn: Arc<dyn Mesh<3>> = mesh.clone();
  tessera.register_mesh(MeshKey::ATMOSPHERE, mesh_dyn);

  let model = AtmosphereModel::new(MeshKey::ATMOSPHERE)
    .with_cfl(0.25)
    .with_current_state_background_correction();
  let fields = model.fields();

  let mut pleroma = Pleroma::new();
  model
    .register_fields(
      &mut pleroma,
      mesh.as_ref(),
      &constants,
      layout.reference_radius(),
    )
    .unwrap();

  // Snapshot the IC so we can compare every conserved component
  // post-ticks. Tolerances are generous because we just need to know
  // the solver isn't blowing up — the background correction should
  // hold the state to single-precision-ish drift.
  let initial: Vec<[f64; 5]> = {
    let state: &SoaField<5> = pleroma.read(fields.euler_state).unwrap();
    (0..mesh.cell_count())
      .map(|i| {
        let s = state.state(CellId::from(i));
        [s[0], s[1], s[2], s[3], s[4]]
      })
      .collect()
  };

  let mut nexus = Nexus::new();
  model.add_stages(&mut nexus).unwrap();
  let mut compiled = nexus.build(&pleroma).unwrap();

  for _ in 0..5 {
    compiled
      .tick(
        WorldId(0),
        &tessera,
        &constants,
        &mut pleroma,
        &Pool::default(),
        1.0e-2,
      )
      .unwrap();
  }

  let final_state: &SoaField<5> = pleroma.read(fields.euler_state).unwrap();
  for (i, before) in initial.iter().enumerate() {
    let after = final_state.state(CellId::from(i));
    let rho_before = before[0];
    let rho_after = after[0];
    assert!(
      rho_after.is_finite() && rho_after > 0.0,
      "cell {i}: density blew up to {rho_after}"
    );
    let rho_drift = (rho_after - rho_before).abs() / rho_before;
    assert!(
      rho_drift < 1.0e-6,
      "cell {i}: density drifted by {rho_drift} (before {rho_before}, \
       after {rho_after})"
    );
    let energy_before = before[4];
    let energy_after = after[4];
    let energy_drift = (energy_after - energy_before).abs() / energy_before;
    assert!(
      energy_drift < 1.0e-6,
      "cell {i}: energy drifted by {energy_drift}"
    );
  }
}
