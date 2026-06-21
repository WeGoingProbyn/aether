// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Bisect which coupling stage drives the coupled mass/energy runaway. HEVI
//! only (explicit is bit-identical). Reports total atmosphere mass growth.
//!
//! `cargo test -p sandbox --release --test ocean_world_coupling_bisect -- --ignored --nocapture`

use aer::AtmosphereScheme;
use nexus::{FieldStorage, MeshKey, SoaField};
use sandbox::{
  OceanWorldCoupling, SANDBOX_WORLD_ID, build_ocean_world_configured,
};
use utility::domain::{CellId, FieldKey, FieldName};

fn total_mass(aether: &aether::core::Aether) -> Option<f64> {
  let world = aether.world(SANDBOX_WORLD_ID).unwrap();
  let key = FieldKey::new(MeshKey::ATMOSPHERE, FieldName::EulerState);
  let state = world.pleroma().read::<SoaField<6>>(key)?;
  let mut m = 0.0;
  for i in 0..state.len() {
    let rho = state.state(CellId::from(i))[0];
    if !rho.is_finite() {
      return None;
    }
    m += rho;
  }
  Some(m)
}

fn run(label: &str, coupling: OceanWorldCoupling) {
  let (mut aether, _) =
    build_ocean_world_configured(AtmosphereScheme::Hevi, coupling).unwrap();
  let m0 = total_mass(&aether).unwrap();
  for step in 1..=40 {
    if aether.step(20.0).is_err() || total_mass(&aether).is_none() {
      eprintln!("[{label}] step {step}: non-physical");
      return;
    }
    if step % 10 == 0 {
      let m = total_mass(&aether).unwrap();
      eprintln!(
        "[{label}] step {step:3}: mass {m:.1} ({:+.1}%)",
        100.0 * (m - m0) / m0
      );
    }
  }
  let m = total_mass(&aether).unwrap();
  eprintln!(
    "[{label}] survived 40 steps: mass {:+.1}%",
    100.0 * (m - m0) / m0
  );
}

#[test]
#[ignore = "diagnostic: builds the coupled world per config; slow"]
fn bisect_coupling_runaway() {
  let off = OceanWorldCoupling {
    radiation: false,
    evaporation: false,
    saturation: false,
  };
  run("none", off);
  run("radiation only", OceanWorldCoupling { radiation: true, ..off });
  run("evaporation only", OceanWorldCoupling { evaporation: true, ..off });
  run(
    "evap+saturation",
    OceanWorldCoupling {
      evaporation: true,
      saturation: true,
      ..off
    },
  );
  run("full", OceanWorldCoupling::default());
}
