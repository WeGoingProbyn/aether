// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Phase 2A: the thermodynamic ocean column. Two invariants:
//! a positive net surface flux warms the sea-surface layer, and vertical
//! diffusion redistributes heat while conserving a column's total heat
//! content under insulated (zero-flux) boundaries.

use std::sync::Arc;

use nexus::{FieldStorage, Nexus, Pleroma, SoaField, WorldConstants, WorldId};
use tessera::{
  cube_sphere::{CubeSphere, CubeSphereShellSpec},
  geometry::CellGeometry,
  world_mesh::Tessera,
};
use thalassa::{OceanColumnLayout, OceanModel};
use utility::domain::CellId;
use utility::thread::pool::Pool;

const ANGULAR: [usize; 2] = [2, 2];
const LAYERS: usize = 4;

fn ocean_mesh() -> Arc<CubeSphere> {
  // Inner radius 1000, outer 1200 — 4 layers. Geometry scale is irrelevant
  // to the column thermodynamics (it uses the configured layer thickness).
  Arc::new(CubeSphere::shell(CubeSphereShellSpec::uniform(
    [ANGULAR[0], ANGULAR[1], LAYERS],
    1000.0,
    1200.0,
  )))
}

fn layout() -> OceanColumnLayout {
  OceanColumnLayout::cube_sphere(ANGULAR, LAYERS)
}

#[test]
fn positive_surface_flux_warms_sea_surface_layer() {
  let mesh = ocean_mesh();
  let mut tessera = Tessera::new();
  tessera.register_mesh(nexus::MeshKey::OCEAN, mesh.clone());

  let model = OceanModel::new(nexus::MeshKey::OCEAN, layout())
    .with_initial_temperature(290.0)
    .with_layer_thickness(50.0);
  let fields = model.fields();

  let mut pleroma = Pleroma::new();
  model.register_fields(&mut pleroma, mesh.as_ref()).unwrap();
  // Strong uniform downward flux into every column's surface cell.
  pleroma.register_field(
    fields.net_flux,
    SoaField::<1>::from_fn(mesh.cell_count(), |_| [500.0]),
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
      3600.0,
    )
    .unwrap();

  let temperature: &SoaField<1> = pleroma.read(fields.temperature).unwrap();
  let stride = layout().radial_stride();
  let surface = layout().surface_layer();
  // Surface cell of panel 0, column 0.
  let surface_cell = surface * stride;
  let bottom_cell = 0;
  assert!(
    temperature.state(CellId::from(surface_cell))[0] > 290.0,
    "sea surface should warm under positive flux"
  );
  assert!(
    temperature.state(CellId::from(bottom_cell))[0] <= 290.0 + 1e-9,
    "deep layer shouldn't warm faster than the surface in one step"
  );
}

#[test]
fn vertical_diffusion_conserves_column_heat() {
  let mesh = ocean_mesh();
  let mut tessera = Tessera::new();
  tessera.register_mesh(nexus::MeshKey::OCEAN, mesh.clone());

  let model = OceanModel::new(nexus::MeshKey::OCEAN, layout())
    .with_layer_thickness(50.0)
    .with_vertical_diffusivity(0.5);
  let fields = model.fields();

  let layout = layout();
  let stride = layout.radial_stride();

  let mut pleroma = Pleroma::new();
  // Vertical gradient: warm surface, cold deep — per column, T = 280 + 5·k.
  pleroma.register_field(
    fields.temperature,
    SoaField::<1>::from_fn(mesh.cell_count(), |cell| {
      let local = cell.index() % layout.cells_per_panel();
      let k = local / stride;
      [280.0 + 5.0 * k as f64]
    }),
  );
  // No surface flux — pure redistribution, insulated boundaries.
  pleroma
    .register_field(fields.net_flux, SoaField::<1>::zeros(mesh.cell_count()));

  let column_heat = |p: &Pleroma| -> f64 {
    let t: &SoaField<1> = p.read(fields.temperature).unwrap();
    (0..LAYERS)
      .map(|k| t.state(CellId::from(k * stride))[0])
      .sum()
  };
  let before = column_heat(&pleroma);

  let mut nexus = Nexus::new();
  model.add_stages(&mut nexus).unwrap();
  let mut compiled = nexus.build(&pleroma).unwrap();
  for _ in 0..50 {
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
  }

  let after = column_heat(&pleroma);
  assert!(
    (after - before).abs() < 1e-6,
    "insulated column heat must be conserved: before {before}, after {after}"
  );

  // Gradient should have shrunk: surface cooler, deep warmer than start.
  let t: &SoaField<1> = pleroma.read(fields.temperature).unwrap();
  let surface = t.state(CellId::from((LAYERS - 1) * stride))[0];
  let bottom = t.state(CellId::from(0))[0];
  assert!(surface < 280.0 + 5.0 * (LAYERS - 1) as f64);
  assert!(bottom > 280.0);
  assert!(surface > bottom, "surface should stay warmer than deep");
}
