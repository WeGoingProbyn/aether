// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! End-to-end test for `SurfaceEnergyBalanceStep`. With a constant net
//! flux Q (W/m²), the surface temperature must drift linearly:
//!
//! `T(t) = T0 + Q * t / heat_capacity_per_area`
//!
//! The Stefan-Boltzmann *equilibrium* check (T → ((Q + …)/εσ)^¼) is
//! lumen's job — terra's stage is a pure forward-Euler thermal slab.

use std::sync::Arc;

use nexus::{
  FieldStorage, MeshKey, Nexus, Pleroma, SoaField, WorldConstants, WorldId,
};
use terra::SurfaceThermalModel;
use tessera::{
  cube_sphere::{CubeSphere, CubeSphereShellSpec},
  geometry::CellGeometry,
  mesh::Mesh,
  world_mesh::Tessera,
};
use utility::{domain::CellId, thread::pool::Pool};

#[test]
fn constant_flux_drives_linear_surface_temperature_growth() {
  let mesh = Arc::new(CubeSphere::shell(CubeSphereShellSpec::uniform(
    [2, 2, 1],
    0.99,
    1.0,
  )));
  let mut tessera = Tessera::new();
  let mesh_dyn: Arc<dyn Mesh<3>> = mesh.clone();
  tessera.register_mesh(MeshKey::SURFACE, mesh_dyn);

  let initial_temperature = 280.0;
  let heat_capacity_per_area = 1.0e6;
  let net_flux = 1000.0;

  let model = SurfaceThermalModel::new(MeshKey::SURFACE)
    .with_initial_temperature(initial_temperature)
    .with_heat_capacity_per_area(heat_capacity_per_area);
  let fields = model.fields();

  let mut pleroma = Pleroma::new();
  model.register_fields(&mut pleroma, mesh.as_ref()).unwrap();
  pleroma.register_field(
    fields.net_flux,
    SoaField::<1>::from_fn(mesh.cell_count(), |_| [net_flux]),
  );

  let mut nexus = Nexus::new();
  model.add_stages(&mut nexus).unwrap();
  let mut compiled = nexus.build(&pleroma).unwrap();

  let dt = 100.0;
  let n_steps = 5;
  for _ in 0..n_steps {
    compiled
      .tick(
        WorldId(0),
        &tessera,
        &WorldConstants::default(),
        &mut pleroma,
        &Pool::default(),
        dt,
      )
      .unwrap();
  }

  let total_time = dt * n_steps as f64;
  let expected =
    initial_temperature + net_flux * total_time / heat_capacity_per_area;
  let temperature: &SoaField<1> = pleroma.read(fields.temperature).unwrap();
  for i in 0..mesh.cell_count() {
    let t = temperature.state(CellId::from(i))[0];
    assert!(
      (t - expected).abs() < 1.0e-9,
      "cell {i}: expected {expected}, got {t}"
    );
  }
}

#[test]
fn negative_flux_cools_surface() {
  let mesh = Arc::new(CubeSphere::shell(CubeSphereShellSpec::uniform(
    [2, 2, 1],
    0.99,
    1.0,
  )));
  let mut tessera = Tessera::new();
  let mesh_dyn: Arc<dyn Mesh<3>> = mesh.clone();
  tessera.register_mesh(MeshKey::SURFACE, mesh_dyn);

  let model = SurfaceThermalModel::new(MeshKey::SURFACE)
    .with_initial_temperature(300.0)
    .with_heat_capacity_per_area(1.0e6);
  let fields = model.fields();

  let mut pleroma = Pleroma::new();
  model.register_fields(&mut pleroma, mesh.as_ref()).unwrap();
  pleroma.register_field(
    fields.net_flux,
    SoaField::<1>::from_fn(mesh.cell_count(), |_| [-500.0]),
  );

  let mut nexus = Nexus::new();
  model.add_stages(&mut nexus).unwrap();
  let mut compiled = nexus.build(&pleroma).unwrap();
  compiled
    .tick(
      WorldId(0),
      &tessera,
      &WorldConstants::default(),
      &mut pleroma,
      &Pool::default(),
      100.0,
    )
    .unwrap();

  let temperature: &SoaField<1> = pleroma.read(fields.temperature).unwrap();
  for i in 0..mesh.cell_count() {
    let t = temperature.state(CellId::from(i))[0];
    assert!(t < 300.0, "cell {i} expected cooling, got {t}");
  }
}
