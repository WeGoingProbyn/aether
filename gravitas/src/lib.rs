// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use tempus::{SecondOrderSystem, VelocityVerlet};

pub const NEWTON_G: f64 = 6.67430e-11;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointMassBody {
  mass: f64,
  position: [f64; 3],
  velocity: [f64; 3],
}

impl PointMassBody {
  pub fn new(mass: f64, position: [f64; 3], velocity: [f64; 3]) -> Self {
    Self {
      mass,
      position,
      velocity,
    }
  }

  pub fn mass(&self) -> f64 {
    self.mass
  }

  pub fn position(&self) -> [f64; 3] {
    self.position
  }

  pub fn velocity(&self) -> [f64; 3] {
    self.velocity
  }
}

#[derive(Clone, Debug)]
pub struct NBodyGravity {
  masses: Vec<f64>,
  gravitational_constant: f64,
  softening_length: f64,
}

impl NBodyGravity {
  pub fn new(masses: Vec<f64>) -> Self {
    Self {
      masses,
      gravitational_constant: NEWTON_G,
      softening_length: 0.0,
    }
  }

  pub fn with_gravitational_constant(mut self, g: f64) -> Self {
    self.gravitational_constant = g;
    self
  }

  pub fn with_softening_length(mut self, softening_length: f64) -> Self {
    self.softening_length = softening_length;
    self
  }

  pub fn masses(&self) -> &[f64] {
    &self.masses
  }

  pub fn gravitational_constant(&self) -> f64 {
    self.gravitational_constant
  }

  pub fn softening_length(&self) -> f64 {
    self.softening_length
  }
}

impl SecondOrderSystem for NBodyGravity {
  fn degrees_of_freedom(&self) -> usize {
    3 * self.masses.len()
  }

  fn acceleration(&self, _t: f64, q: &[f64], _v: &[f64], a: &mut [f64]) {
    let n = self.masses.len();
    assert_eq!(q.len(), 3 * n);
    assert_eq!(a.len(), 3 * n);
    a.fill(0.0);

    let eps2 = self.softening_length * self.softening_length;
    for i in 0..n {
      let ix = 3 * i;
      for j in (i + 1)..n {
        let jx = 3 * j;
        let dx = q[jx] - q[ix];
        let dy = q[jx + 1] - q[ix + 1];
        let dz = q[jx + 2] - q[ix + 2];
        let r2 = dx * dx + dy * dy + dz * dz + eps2;
        let inv_r = 1.0 / r2.sqrt();
        let inv_r3 = inv_r * inv_r * inv_r;
        let scale_i = self.gravitational_constant * self.masses[j] * inv_r3;
        let scale_j = self.gravitational_constant * self.masses[i] * inv_r3;

        a[ix] += scale_i * dx;
        a[ix + 1] += scale_i * dy;
        a[ix + 2] += scale_i * dz;

        a[jx] -= scale_j * dx;
        a[jx + 1] -= scale_j * dy;
        a[jx + 2] -= scale_j * dz;
      }
    }
  }
}

#[derive(Clone, Debug)]
pub struct NBodySimulation {
  gravity: NBodyGravity,
  positions: Vec<f64>,
  velocities: Vec<f64>,
  stepper: VelocityVerlet,
  time: f64,
}

impl NBodySimulation {
  pub fn new(bodies: impl IntoIterator<Item = PointMassBody>) -> Self {
    let bodies: Vec<PointMassBody> = bodies.into_iter().collect();
    let masses = bodies.iter().map(PointMassBody::mass).collect();
    let mut positions = Vec::with_capacity(3 * bodies.len());
    let mut velocities = Vec::with_capacity(3 * bodies.len());
    for body in bodies {
      positions.extend(body.position);
      velocities.extend(body.velocity);
    }

    Self {
      gravity: NBodyGravity::new(masses),
      positions,
      velocities,
      stepper: VelocityVerlet::new(),
      time: 0.0,
    }
  }

  pub fn with_gravitational_constant(mut self, g: f64) -> Self {
    self.gravity = self.gravity.with_gravitational_constant(g);
    self
  }

  pub fn with_softening_length(mut self, softening_length: f64) -> Self {
    self.gravity = self.gravity.with_softening_length(softening_length);
    self
  }

  pub fn time(&self) -> f64 {
    self.time
  }

  pub fn body_count(&self) -> usize {
    self.gravity.masses.len()
  }

  pub fn gravity(&self) -> &NBodyGravity {
    &self.gravity
  }

  pub fn positions_flat(&self) -> &[f64] {
    &self.positions
  }

  pub fn velocities_flat(&self) -> &[f64] {
    &self.velocities
  }

  pub fn position(&self, index: usize) -> [f64; 3] {
    let offset = 3 * index;
    [
      self.positions[offset],
      self.positions[offset + 1],
      self.positions[offset + 2],
    ]
  }

  pub fn velocity(&self, index: usize) -> [f64; 3] {
    let offset = 3 * index;
    [
      self.velocities[offset],
      self.velocities[offset + 1],
      self.velocities[offset + 2],
    ]
  }

  pub fn body(&self, index: usize) -> PointMassBody {
    PointMassBody::new(
      self.gravity.masses[index],
      self.position(index),
      self.velocity(index),
    )
  }

  pub fn step(&mut self, dt: f64) {
    self.stepper.step(
      &self.gravity,
      self.time,
      &mut self.positions,
      &mut self.velocities,
      dt,
    );
    self.time += dt;
  }

  pub fn total_energy(&self) -> f64 {
    kinetic_energy(self.gravity.masses(), &self.velocities)
      + potential_energy(
        self.gravity.masses(),
        &self.positions,
        self.gravity.gravitational_constant(),
        self.gravity.softening_length(),
      )
  }

  pub fn center_of_mass(&self) -> [f64; 3] {
    let total_mass: f64 = self.gravity.masses.iter().sum();
    let mut out = [0.0; 3];
    if total_mass <= 0.0 {
      return out;
    }

    for (i, mass) in self.gravity.masses.iter().enumerate() {
      let offset = 3 * i;
      out[0] += mass * self.positions[offset];
      out[1] += mass * self.positions[offset + 1];
      out[2] += mass * self.positions[offset + 2];
    }
    out[0] /= total_mass;
    out[1] /= total_mass;
    out[2] /= total_mass;
    out
  }
}

fn kinetic_energy(masses: &[f64], velocities: &[f64]) -> f64 {
  masses
    .iter()
    .enumerate()
    .map(|(i, mass)| {
      let offset = 3 * i;
      let v2 = velocities[offset] * velocities[offset]
        + velocities[offset + 1] * velocities[offset + 1]
        + velocities[offset + 2] * velocities[offset + 2];
      0.5 * mass * v2
    })
    .sum()
}

fn potential_energy(
  masses: &[f64],
  positions: &[f64],
  g: f64,
  softening_length: f64,
) -> f64 {
  let mut energy = 0.0;
  let eps2 = softening_length * softening_length;
  for i in 0..masses.len() {
    let ix = 3 * i;
    for j in (i + 1)..masses.len() {
      let jx = 3 * j;
      let dx = positions[jx] - positions[ix];
      let dy = positions[jx + 1] - positions[ix + 1];
      let dz = positions[jx + 2] - positions[ix + 2];
      let r = (dx * dx + dy * dy + dz * dz + eps2).sqrt();
      energy -= g * masses[i] * masses[j] / r;
    }
  }
  energy
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn equal_mass_binary_preserves_center_of_mass_and_energy() {
    let g: f64 = 1.0;
    let mass: f64 = 1.0;
    let separation: f64 = 2.0;
    let speed = (g * mass / (2.0 * separation)).sqrt();
    let mut simulation = NBodySimulation::new([
      PointMassBody::new(mass, [-1.0, 0.0, 0.0], [0.0, -speed, 0.0]),
      PointMassBody::new(mass, [1.0, 0.0, 0.0], [0.0, speed, 0.0]),
    ])
    .with_gravitational_constant(g);

    let initial_energy = simulation.total_energy();
    for _ in 0..10_000 {
      simulation.step(0.001);
    }

    let center = simulation.center_of_mass();
    assert!(center[0].abs() < 1.0e-12);
    assert!(center[1].abs() < 1.0e-12);
    assert!(center[2].abs() < 1.0e-12);
    let relative_energy_error =
      ((simulation.total_energy() - initial_energy) / initial_energy).abs();
    assert!(relative_energy_error < 1.0e-6);
  }

  #[test]
  fn earth_like_circular_orbit_remains_near_initial_radius() {
    let solar_mass = 1.988_47e30;
    let earth_mass = 5.972_2e24;
    let radius = 1.495_978_707e11;
    let speed = (NEWTON_G * solar_mass / radius).sqrt();
    let mut simulation = NBodySimulation::new([
      PointMassBody::new(solar_mass, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
      PointMassBody::new(earth_mass, [radius, 0.0, 0.0], [0.0, speed, 0.0]),
    ]);

    let day = 86_400.0;
    for _ in 0..365 {
      simulation.step(day);
    }

    let earth = simulation.body(1);
    let position = earth.position();
    let final_radius =
      (position[0] * position[0] + position[1] * position[1]).sqrt();
    assert!(((final_radius - radius) / radius).abs() < 5.0e-4);
  }
}
