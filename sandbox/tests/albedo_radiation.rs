// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Pillar 3 coupling: terrain (and, later, ice) albedo → radiation, end-to-end.
//!
//! The surface short-wave albedo is a per-cell field derived from the surface
//! type; radiation reads it per cell. We build two identical worlds — one with
//! a uniformly dark (ocean) surface, one uniformly bright (ice) — and confirm
//! the bright surface absorbs less shortwave, so its net surface radiative flux
//! is lower everywhere the sun reaches (and identical on the night side).

use nexus::{FieldStorage, SoaField};
use sandbox::{SANDBOX_WORLD_ID, build_albedo_world};
use terra::{SurfaceClass, TerrainSample};
use utility::domain::{CellId, FieldKey, FieldName, MeshKey};

fn net_surface_flux(aether: &aether::core::Aether) -> Vec<f64> {
  let world = aether.world(SANDBOX_WORLD_ID).unwrap();
  let field: &SoaField<1> = world
    .pleroma()
    .read(FieldKey::new(MeshKey::SURFACE, FieldName::NetSurfaceFlux))
    .unwrap();
  (0..field.len())
    .map(|i| field.state(CellId::from(i))[0])
    .collect()
}

#[test]
fn surface_albedo_modulates_absorbed_shortwave() {
  // Uniformly dark ocean vs uniformly bright ice — same world otherwise.
  let ocean = |_| TerrainSample {
    elevation: 0.0,
    class: SurfaceClass::Ocean,
  };
  let ice = |_| TerrainSample {
    elevation: 0.0,
    class: SurfaceClass::Ice,
  };
  let (mut dark, _) = build_albedo_world(ocean).unwrap();
  let (mut bright, _) = build_albedo_world(ice).unwrap();
  dark.step(1.0).unwrap();
  bright.step(1.0).unwrap();

  let dark_flux = net_surface_flux(&dark);
  let bright_flux = net_surface_flux(&bright);
  assert_eq!(dark_flux.len(), bright_flux.len());

  assert!(
    dark_flux.iter().chain(&bright_flux).all(|x| x.is_finite()),
    "radiation produced a non-finite net flux"
  );

  // The albedo difference must change the flux somewhere (the sunlit side).
  let differing = dark_flux
    .iter()
    .zip(&bright_flux)
    .filter(|(d, b)| (*d - *b).abs() > 1e-6)
    .count();
  assert!(differing > 0, "per-cell albedo had no effect on radiation");

  // Brighter surface ⇒ less absorbed shortwave ⇒ lower (or equal, on the night
  // side) net flux at every cell, and strictly lower in aggregate.
  for (d, b) in dark_flux.iter().zip(&bright_flux) {
    assert!(
      *b <= *d + 1e-9,
      "ice net flux {b} exceeded ocean net flux {d}"
    );
  }
  let sum_dark: f64 = dark_flux.iter().sum();
  let sum_bright: f64 = bright_flux.iter().sum();
  assert!(
    sum_bright < sum_dark,
    "bright surface should absorb less: ice sum {sum_bright} vs ocean {sum_dark}"
  );
}
