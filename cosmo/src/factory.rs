// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Hand-built presets for the Sol system. Each planet factory returns a
//! `CelestialBody` placed on the +x axis at its orbital radius, moving in +y
//! at its mean orbital velocity (a circular-orbit approximation — eccentricity
//! is ignored for game purposes).
//!
//! Atmospheric mole fractions reflect the dominant species the simulator
//! models; trace species (Argon, methane on rocky planets, etc.) are dropped
//! and the active fraction is auto-normalised by `Atmosphere::properties`.

use std::collections::HashMap;

use utility::{
  constants::{angular_velocity, orbital_velocity},
  consts::*,
  maths::vector::Vector,
};

use crate::{
  body::{Atmosphere, Species},
  kind::{BodyKind, CelestialBody, GasGiant, RockyBody, Star},
  system::System,
};

/// Place a body on +x at `orbit_radius`, moving in +y at the circular orbital
/// velocity around a primary of mass `primary_mass`.
fn circular_orbit_state(
  primary_mass: f64,
  orbit_radius: f64,
) -> (Vector<f64, 3>, Vector<f64, 3>) {
  let v = orbital_velocity(primary_mass, orbit_radius);
  ([orbit_radius, 0.0, 0.0].into(), [0.0, v, 0.0].into())
}

fn atmosphere_from(
  species: &[(Species, f64)],
  albedo: Option<f64>,
) -> Atmosphere {
  let mut comp = HashMap::new();
  for &(s, x) in species {
    comp.insert(s, x);
  }
  Atmosphere::new(comp, albedo)
}

pub fn sun() -> CelestialBody {
  CelestialBody::new(
    SOLAR_MASS,
    SOLAR_RADIUS,
    [0.0, 0.0, 0.0].into(),
    [0.0, 0.0, 0.0].into(),
    BodyKind::Star(Star {
      surface_temperature: SOLAR_SURFACE_TEMP,
      core_temperature: SOLAR_CORE_TEMP,
    }),
  )
}

pub fn mercury() -> CelestialBody {
  let (pos, vel) = circular_orbit_state(SOLAR_MASS, MERCURY_ORBIT);
  CelestialBody::new(
    MERCURY_MASS,
    MERCURY_RADIUS,
    pos,
    vel,
    BodyKind::RockyBody(RockyBody {
      surface_temperature: MERCURY_SURFACE_T,
      surface_pressure: MERCURY_SURFACE_P,
      angular_velocity: angular_velocity(MERCURY_DAY),
      axial_tilt: MERCURY_AXIAL_TILT.to_radians(),
      atmosphere: None,
    }),
  )
}

pub fn venus() -> CelestialBody {
  let (pos, vel) = circular_orbit_state(SOLAR_MASS, VENUS_ORBIT);
  let atm = atmosphere_from(
    &[(Species::CarbonDioxide, 0.965), (Species::Nitrogen, 0.035)],
    Some(VENUS_ALBEDO),
  );
  CelestialBody::new(
    VENUS_MASS,
    VENUS_RADIUS,
    pos,
    vel,
    BodyKind::RockyBody(RockyBody {
      surface_temperature: VENUS_SURFACE_T,
      surface_pressure: VENUS_SURFACE_P,
      angular_velocity: angular_velocity(VENUS_DAY),
      axial_tilt: VENUS_AXIAL_TILT.to_radians(),
      atmosphere: Some(atm),
    }),
  )
}

pub fn earth() -> CelestialBody {
  let (pos, vel) = circular_orbit_state(SOLAR_MASS, EARTH_ORBIT);
  let atm = atmosphere_from(
    &[(Species::Nitrogen, 0.78), (Species::Oxygen, 0.21)],
    Some(EARTH_ALBEDO),
  );
  CelestialBody::new(
    EARTH_MASS,
    EARTH_RADIUS,
    pos,
    vel,
    BodyKind::RockyBody(RockyBody {
      surface_temperature: EARTH_SURFACE_T,
      surface_pressure: EARTH_SURFACE_P,
      angular_velocity: angular_velocity(EARTH_DAY),
      axial_tilt: EARTH_AXIAL_TILT.to_radians(),
      atmosphere: Some(atm),
    }),
  )
}

pub fn mars() -> CelestialBody {
  let (pos, vel) = circular_orbit_state(SOLAR_MASS, MARS_ORBIT);
  let atm = atmosphere_from(
    &[
      (Species::CarbonDioxide, 0.96),
      (Species::Nitrogen, 0.019),
      (Species::Oxygen, 0.0015),
    ],
    Some(MARS_ALBEDO),
  );
  CelestialBody::new(
    MARS_MASS,
    MARS_RADIUS,
    pos,
    vel,
    BodyKind::RockyBody(RockyBody {
      surface_temperature: MARS_SURFACE_T,
      surface_pressure: MARS_SURFACE_P,
      angular_velocity: angular_velocity(MARS_DAY),
      axial_tilt: MARS_AXIAL_TILT.to_radians(),
      atmosphere: Some(atm),
    }),
  )
}

pub fn jupiter() -> CelestialBody {
  let (pos, vel) = circular_orbit_state(SOLAR_MASS, JUPITER_ORBIT);
  let atm = atmosphere_from(
    &[
      (Species::Hydrogen, 0.89),
      (Species::Helium, 0.10),
      (Species::Methane, 0.003),
    ],
    Some(0.34),
  );
  CelestialBody::new(
    JUPITER_MASS,
    JUPITER_RADIUS,
    pos,
    vel,
    BodyKind::GasGiant(GasGiant {
      reference_temperature: JUPITER_SURFACE_T,
      reference_pressure: NOMINAL_GAS_PRESS,
      angular_velocity: angular_velocity(JUPITER_DAY),
      axial_tilt: JUPITER_AXIAL_TILT.to_radians(),
      heat_factor: JUPITER_HEAT_FACTOR,
      atmosphere: atm,
    }),
  )
}

pub fn saturn() -> CelestialBody {
  let (pos, vel) = circular_orbit_state(SOLAR_MASS, SATURN_ORBIT);
  let atm = atmosphere_from(
    &[
      (Species::Hydrogen, 0.96),
      (Species::Helium, 0.03),
      (Species::Methane, 0.0045),
    ],
    Some(0.34),
  );
  CelestialBody::new(
    SATURN_MASS,
    SATURN_RADIUS,
    pos,
    vel,
    BodyKind::GasGiant(GasGiant {
      reference_temperature: SATURN_SURFACE_T,
      reference_pressure: NOMINAL_GAS_PRESS,
      angular_velocity: angular_velocity(SATURN_DAY),
      axial_tilt: SATURN_AXIAL_TILT.to_radians(),
      heat_factor: SATURN_HEAT_FACTOR,
      atmosphere: atm,
    }),
  )
}

pub fn neptune() -> CelestialBody {
  let (pos, vel) = circular_orbit_state(SOLAR_MASS, NEPTUNE_ORBIT);
  let atm = atmosphere_from(
    &[
      (Species::Hydrogen, 0.80),
      (Species::Helium, 0.19),
      (Species::Methane, 0.015),
    ],
    Some(0.29),
  );
  CelestialBody::new(
    NEPTUNE_MASS,
    NEPTUNE_RADIUS,
    pos,
    vel,
    BodyKind::GasGiant(GasGiant {
      reference_temperature: NEPTUNE_SURFACE_T,
      reference_pressure: NOMINAL_GAS_PRESS,
      angular_velocity: angular_velocity(NEPTUNE_DAY),
      axial_tilt: NEPTUNE_AXIAL_TILT.to_radians(),
      heat_factor: NEPTUNE_HEAT_FACTOR,
      atmosphere: atm,
    }),
  )
}

/// Sol system: Sun + the seven planets currently parameterised in
/// `utility::constants`. Bodies are ordered by orbital radius.
pub fn sol() -> System {
  System::new(vec![
    sun(),
    mercury(),
    venus(),
    earth(),
    mars(),
    jupiter(),
    saturn(),
    neptune(),
  ])
}
