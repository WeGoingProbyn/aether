// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! End-to-end check that aer's `EulerDiagnosticsStep` recovers the
//! cosmo-supplied reference temperature and pressure from the
//! hydrostatic IC after one tick.

use std::sync::Arc;

use aer::AtmosphereModel;
use aether::core::world_constants_from_seed;
use cosmo::factory as cosmo_factory;
use nexus::{FieldStorage, MeshKey, Nexus, Pleroma, SoaField, WorldId};
use tessera::{
  cube_sphere::{CubeSphere, CubeSphereShellSpec},
  geometry::CellGeometry,
  mesh::Mesh,
  world_mesh::Tessera,
};
use utility::{domain::CellId, thread::pool::Pool};

#[test]
fn diagnostics_recover_reference_temperature_and_pressure_at_surface() {
  let seed = cosmo_factory::earth();
  let primary = cosmo_factory::sun();
  let constants = world_constants_from_seed(&seed, Some(&primary));
  let atmosphere = constants.atmosphere.expect("earth has an atmosphere");

  // A thin near-surface shell so every cell sits very close to the
  // reference radius and the hydrostatic profile is essentially flat —
  // diagnostics should match the IC reference values to a tight
  // tolerance after one tick.
  let inner = constants.radius;
  let outer = constants.radius + 1.0;
  let mesh = Arc::new(CubeSphere::shell(CubeSphereShellSpec::uniform(
    [2, 2, 1],
    inner,
    outer,
  )));
  let mut tessera = Tessera::new();
  let mesh_dyn: Arc<dyn Mesh<3>> = mesh.clone();
  tessera.register_mesh(MeshKey::ATMOSPHERE, mesh_dyn);

  let model = AtmosphereModel::new(MeshKey::ATMOSPHERE).with_cfl(0.25);
  let fields = model.fields();

  let mut pleroma = Pleroma::new();
  model
    .register_fields(&mut pleroma, mesh.as_ref(), &constants, inner)
    .unwrap();

  let mut nexus = Nexus::new();
  model.add_stages(&mut nexus).unwrap();
  let mut compiled = nexus.build(&pleroma).unwrap();
  compiled
    .tick(
      WorldId(0),
      &tessera,
      &constants,
      &mut pleroma,
      &Pool::default(),
      1.0e-3,
    )
    .unwrap();

  let temperature: &SoaField<1> = pleroma.read(fields.temperature).unwrap();
  let pressure: &SoaField<1> = pleroma.read(fields.pressure).unwrap();

  for i in 0..mesh.cell_count() {
    let cell = CellId::from(i);
    let t = temperature.state(cell)[0];
    let p = pressure.state(cell)[0];
    assert!(
      (t - atmosphere.reference_temperature).abs()
        / atmosphere.reference_temperature
        < 1.0e-3,
      "cell {i}: T={t}, expected {}",
      atmosphere.reference_temperature
    );
    assert!(
      (p - atmosphere.reference_pressure).abs() / atmosphere.reference_pressure
        < 1.0e-2,
      "cell {i}: P={p}, expected {}",
      atmosphere.reference_pressure
    );
  }
}
