//! Mixture equation-of-state tests for `Atmosphere`. Reference values come
//! from textbook ideal-gas mixing rules; mole fractions are mole-fraction-
//! weighted into molar mass and DOF, then γ = 1 + 2/f, R = R_universal/M.

use std::collections::HashMap;

use cosmo::body::{Atmosphere, Species};

const R_UNIVERSAL: f64 = 8.314462618;

#[test]
fn pure_diatomic_at_room_temperature_gives_gamma_seven_fifths() {
  // N₂ at 288 K: 5 DOF (3 trans + 2 rot, vibration frozen out at this T).
  // γ = 1 + 2/5 = 1.4. Standard result.
  let mut comp = HashMap::new();
  comp.insert(Species::Nitrogen, 1.0);
  let atm = Atmosphere::new(comp, None);
  let p = atm.properties(288.0);
  assert!((p.gamma - 1.4).abs() < 1e-12, "γ = {}", p.gamma);
  assert!((p.molar_mass - 0.028).abs() < 1e-12, "M = {}", p.molar_mass);
  let r_expected = R_UNIVERSAL / 0.028;
  assert!((p.gas_constant - r_expected).abs() < 1e-9, "R = {}", p.gas_constant);
}

#[test]
fn pure_monatomic_helium_gives_gamma_five_thirds() {
  // He: 3 DOF (translation only). γ = 1 + 2/3 = 5/3 ≈ 1.667.
  let mut comp = HashMap::new();
  comp.insert(Species::Helium, 1.0);
  let atm = Atmosphere::new(comp, None);
  let p = atm.properties(288.0);
  assert!((p.gamma - 5.0 / 3.0).abs() < 1e-12, "γ = {}", p.gamma);
}

#[test]
fn earth_air_gives_realistic_gamma_and_specific_gas_constant() {
  // 78% N₂, 21% O₂, 1% trace (ignored — auto-normalisation pulls the active
  // 99% up to 1.0). Real earth air at 288 K: γ ≈ 1.4, R_specific ≈ 287 J/(kg·K).
  let mut comp = HashMap::new();
  comp.insert(Species::Nitrogen, 0.78);
  comp.insert(Species::Oxygen, 0.21);
  let atm = Atmosphere::new(comp, Some(0.3));
  let p = atm.properties(288.0);

  // Both N₂ and O₂ are 5-DOF at 288 K, so γ comes out exactly 1.4.
  assert!((p.gamma - 1.4).abs() < 1e-12, "γ = {}", p.gamma);

  // M ≈ 0.78·0.028 + 0.21·0.032 = 0.02856, normalised over 0.99 → 0.02885.
  // R_specific = 8.314 / 0.02885 ≈ 288 J/(kg·K) — close to the literature 287
  // (the small gap is the Argon we don't model).
  assert!(
    (p.gas_constant - 288.0).abs() < 1.5,
    "R_specific = {}", p.gas_constant
  );
}

#[test]
fn co2_atmosphere_gamma_drops_at_higher_temperature() {
  // CO₂ has temperature-dependent DOF. At 250 K it's 6 (no vibration); at
  // 1000 K it picks up extra modes. γ should shrink monotonically with T.
  let mut comp = HashMap::new();
  comp.insert(Species::CarbonDioxide, 1.0);
  let atm = Atmosphere::new(comp, None);

  let g_cold = atm.properties(250.0).gamma; // 6 DOF → γ = 1 + 2/6 ≈ 1.333
  let g_warm = atm.properties(500.0).gamma; // 7.5 DOF → γ ≈ 1.267
  let g_hot = atm.properties(1000.0).gamma; // 9 DOF → γ ≈ 1.222

  assert!(g_cold > g_warm, "{} should exceed {}", g_cold, g_warm);
  assert!(g_warm > g_hot, "{} should exceed {}", g_warm, g_hot);
  assert!((g_cold - 4.0 / 3.0).abs() < 1e-12);
  assert!((g_hot - 11.0 / 9.0).abs() < 1e-12);
}

#[test]
fn unnormalised_composition_auto_normalises_in_properties() {
  // Caller passes mole ratios that sum to 2.0 instead of 1.0. The result
  // must equal the same composition rescaled to sum to 1.0.
  let mut a = HashMap::new();
  a.insert(Species::Nitrogen, 0.78);
  a.insert(Species::Oxygen, 0.22);
  let atm_norm = Atmosphere::new(a, None);

  let mut b = HashMap::new();
  b.insert(Species::Nitrogen, 1.56);
  b.insert(Species::Oxygen, 0.44);
  let atm_double = Atmosphere::new(b, None);

  let p1 = atm_norm.properties(288.0);
  let p2 = atm_double.properties(288.0);
  assert!((p1.molar_mass - p2.molar_mass).abs() < 1e-15);
  assert!((p1.gamma - p2.gamma).abs() < 1e-15);
  assert!((p1.gas_constant - p2.gas_constant).abs() < 1e-9);
}

#[test]
fn validate_elements_accepts_complete_or_partial_compositions() {
  let mut full = HashMap::new();
  full.insert(Species::Nitrogen, 0.78);
  full.insert(Species::Oxygen, 0.22);
  assert!(Atmosphere::new(full, None).validate_elements());

  let mut partial = HashMap::new();
  partial.insert(Species::Nitrogen, 0.5);
  assert!(Atmosphere::new(partial, None).validate_elements());

  let mut over = HashMap::new();
  over.insert(Species::Nitrogen, 1.2);
  assert!(!Atmosphere::new(over, None).validate_elements());
}

#[test]
fn normalise_components_makes_sum_unity() {
  let mut comp = HashMap::new();
  comp.insert(Species::Nitrogen, 1.5);
  comp.insert(Species::Oxygen, 0.5);
  let mut atm = Atmosphere::new(comp, None);
  atm.normalise_components();
  let sum: f64 = atm.composition.values().sum();
  assert!((sum - 1.0).abs() < 1e-15);
}
