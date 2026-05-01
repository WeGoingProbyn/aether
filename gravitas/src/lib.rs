// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Newtonian n-body gravity as a nexus stage.
//!
//! Body state lives in pleroma as `ResourceKey::Bodies` (a `BodyState<D>`);
//! `KeplerStage` is the integration stage that pulls it via `WorldAccess`
//! and steps it forward with `tempus::VelocityVerlet`.

use nexus::{FieldKey, ResourceKey, Stage, StageContext};
use tempus::integrator::VelocityVerlet;
use tempus::ode::SecondOrderSystem;
use utility::constants::NEWTON_G;
use utility::error::{AetherError, AetherResult, ErrorDomain};
use utility::maths::vector::Vector;

/// One Newtonian point mass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointMassBody<const D: usize> {
  mass: f64,
  position: Vector<f64, D>,
  velocity: Vector<f64, D>,
}

impl<const D: usize> PointMassBody<D> {
  pub fn new(
    mass: f64,
    position: Vector<f64, D>,
    velocity: Vector<f64, D>,
  ) -> Self {
    Self {
      mass,
      position,
      velocity,
    }
  }

  pub fn mass(&self) -> f64 {
    self.mass
  }

  pub fn position(&self) -> &Vector<f64, D> {
    &self.position
  }

  pub fn velocity(&self) -> &Vector<f64, D> {
    &self.velocity
  }
}

/// Mutable body state owned by pleroma as `ResourceKey::Bodies`.
///
/// `KeplerStage` declares this as a `resource_writes` and updates positions /
/// velocities in place each tick. Other stages (e.g. lumen reading sun
/// direction off the first body) declare it as `resource_reads`.
#[derive(Clone, Debug)]
pub struct BodyState<const D: usize> {
  masses: Vec<f64>,
  positions: Vec<Vector<f64, D>>,
  velocities: Vec<Vector<f64, D>>,
  time: f64,
}

impl<const D: usize> BodyState<D> {
  pub fn from_bodies(
    bodies: impl IntoIterator<Item = PointMassBody<D>>,
  ) -> Self {
    let bodies: Vec<PointMassBody<D>> = bodies.into_iter().collect();
    let masses = bodies.iter().map(PointMassBody::mass).collect();
    let mut positions = Vec::with_capacity(bodies.len());
    let mut velocities = Vec::with_capacity(bodies.len());
    for body in bodies {
      positions.push(body.position);
      velocities.push(body.velocity);
    }

    Self {
      masses,
      positions,
      velocities,
      time: 0.0,
    }
  }

  pub fn body_count(&self) -> usize {
    self.masses.len()
  }

  pub fn masses(&self) -> &[f64] {
    &self.masses
  }

  pub fn positions(&self) -> &[Vector<f64, D>] {
    &self.positions
  }

  pub fn velocities(&self) -> &[Vector<f64, D>] {
    &self.velocities
  }

  pub fn position(&self, index: usize) -> &Vector<f64, D> {
    &self.positions[index]
  }

  pub fn velocity(&self, index: usize) -> &Vector<f64, D> {
    &self.velocities[index]
  }

  pub fn body(&self, index: usize) -> PointMassBody<D> {
    PointMassBody::new(
      self.masses[index],
      self.positions[index],
      self.velocities[index],
    )
  }

  pub fn time(&self) -> f64 {
    self.time
  }

  pub fn center_of_mass(&self) -> Vector<f64, D> {
    let total_mass: f64 = self.masses.iter().sum();
    let mut out = Vector::<f64, D>::default();
    if total_mass <= 0.0 {
      return out;
    }
    for (i, mass) in self.masses.iter().enumerate() {
      out += self.positions[i] * mass;
    }
    out /= total_mass;
    out
  }
}

/// Newtonian gravitational law parameters. Implements
/// `tempus::SecondOrderSystem` so it can be plugged into a velocity-Verlet
/// integrator. Held inside a `KeplerStage`; the masses here are the
/// authoritative copy used by the integrator's acceleration term.
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

  pub fn total_energy<const D: usize>(&self, state: &BodyState<D>) -> f64 {
    kinetic_energy(&self.masses, &state.velocities)
      + potential_energy(
        &self.masses,
        &state.positions,
        self.gravitational_constant,
        self.softening_length,
      )
  }
}

impl<const D: usize> SecondOrderSystem<D> for NBodyGravity {
  fn acceleration(
    &self,
    _t: f64,
    q: &[Vector<f64, D>],
    _v: &[Vector<f64, D>],
    a: &mut [Vector<f64, D>],
  ) {
    let n = self.masses.len();
    assert_eq!(q.len(), n);
    assert_eq!(a.len(), n);
    a.fill(Vector::default());

    let eps2 = self.softening_length * self.softening_length;
    for i in 0..n {
      for j in (i + 1)..n {
        let d = q[j] - q[i];
        let r2 = d.powi(2).sum() + eps2;
        let inv_r = r2.powf(-1.5);
        let scale_i = self.gravitational_constant * self.masses[j] * inv_r;
        let scale_j = self.gravitational_constant * self.masses[i] * inv_r;

        a[i] += d * scale_i;
        a[j] -= d * scale_j;
      }
    }
  }
}

/// Velocity-Verlet integration stage for `BodyState<D>` under
/// Newtonian gravity. Declares `ResourceKey::Bodies` as a write; reads
/// nothing.
pub struct KeplerStage<const D: usize> {
  gravity: NBodyGravity,
  stepper: VelocityVerlet<D>,
  resource_writes: [ResourceKey; 1],
}

impl<const D: usize> KeplerStage<D> {
  pub fn new(gravity: NBodyGravity) -> Self {
    Self {
      gravity,
      stepper: VelocityVerlet::new(),
      resource_writes: [ResourceKey::Bodies],
    }
  }

  pub fn gravity(&self) -> &NBodyGravity {
    &self.gravity
  }

  /// Step a `BodyState` directly, bypassing nexus. Useful for tests and
  /// for stages that compose gravity inside a larger integrator.
  pub fn step_in_place(&mut self, state: &mut BodyState<D>, dt: f64) {
    self.stepper.step(
      &self.gravity,
      state.time,
      &mut state.positions,
      &mut state.velocities,
      dt,
    );
    state.time += dt;
  }
}

impl<const D: usize> Stage for KeplerStage<D>
where
  BodyState<D>: 'static,
{
  fn name(&self) -> &'static str {
    "gravitas_kepler"
  }

  fn reads(&self) -> &[FieldKey] {
    &[]
  }

  fn writes(&self) -> &[FieldKey] {
    &[]
  }

  fn resource_writes(&self) -> &[ResourceKey] {
    &self.resource_writes
  }

  fn run(&mut self, mut ctx: StageContext<'_>) -> AetherResult<()> {
    let dt = ctx.world.dt;
    let bodies: &mut BodyState<D> = ctx
      .world
      .fields
      .resource_mut(ResourceKey::Bodies)
      .ok_or_else(|| AetherError::new(GravitasError::MissingBodies))?;
    self.stepper.step(
      &self.gravity,
      bodies.time,
      &mut bodies.positions,
      &mut bodies.velocities,
      dt,
    );
    bodies.time += dt;
    Ok(())
  }
}

#[derive(Debug)]
pub enum GravitasError {
  MissingBodies,
}

impl ErrorDomain for GravitasError {
  fn domain(&self) -> &str {
    "gravitas"
  }
}

impl std::fmt::Display for GravitasError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      GravitasError::MissingBodies => write!(
        f,
        "ResourceKey::Bodies not registered or wrong type (expected BodyState)"
      ),
    }
  }
}

fn kinetic_energy<const D: usize>(
  masses: &[f64],
  velocities: &[Vector<f64, D>],
) -> f64 {
  masses
    .iter()
    .enumerate()
    .map(|(i, mass)| {
      let v2 = velocities[i].powi(2);
      0.5 * mass * v2.sum()
    })
    .sum()
}

fn potential_energy<const D: usize>(
  masses: &[f64],
  positions: &[Vector<f64, D>],
  g: f64,
  softening_length: f64,
) -> f64 {
  let mut energy = 0.0;
  let eps2 = softening_length * softening_length;
  for i in 0..masses.len() {
    for j in (i + 1)..masses.len() {
      let d = positions[j] - positions[i];
      let r = (d.powi(2).sum() + eps2).sqrt();
      energy -= g * masses[i] * masses[j] / r;
    }
  }
  energy
}

#[cfg(test)]
mod tests {
  use super::*;

  fn binary_state() -> (BodyState<3>, NBodyGravity) {
    let g: f64 = 1.0;
    let mass: f64 = 1.0;
    let separation: f64 = 2.0;
    let speed = (g * mass / (2.0 * separation)).sqrt();
    let state = BodyState::from_bodies([
      PointMassBody::new(
        mass,
        [-1.0, 0.0, 0.0].into(),
        [0.0, -speed, 0.0].into(),
      ),
      PointMassBody::new(
        mass,
        [1.0, 0.0, 0.0].into(),
        [0.0, speed, 0.0].into(),
      ),
    ]);
    let gravity =
      NBodyGravity::new(state.masses().to_vec()).with_gravitational_constant(g);
    (state, gravity)
  }

  #[test]
  fn equal_mass_binary_preserves_center_of_mass_and_energy() {
    let (mut state, gravity) = binary_state();
    let initial_energy = gravity.total_energy(&state);
    let mut stage = KeplerStage::new(gravity.clone());
    for _ in 0..10_000 {
      stage.step_in_place(&mut state, 0.001);
    }

    let center = state.center_of_mass();
    assert!(center[0].abs() < 1.0e-12);
    assert!(center[1].abs() < 1.0e-12);
    assert!(center[2].abs() < 1.0e-12);
    let relative_energy_error =
      ((gravity.total_energy(&state) - initial_energy) / initial_energy).abs();
    assert!(relative_energy_error < 1.0e-6);
  }

  #[test]
  fn earth_like_circular_orbit_remains_near_initial_radius() {
    let solar_mass = 1.988_47e30;
    let earth_mass = 5.972_2e24;
    let radius = 1.495_978_707e11;
    let speed = (NEWTON_G * solar_mass / radius).sqrt();
    let state = BodyState::from_bodies([
      PointMassBody::new(
        solar_mass,
        [0.0, 0.0, 0.0].into(),
        [0.0, 0.0, 0.0].into(),
      ),
      PointMassBody::new(
        earth_mass,
        [radius, 0.0, 0.0].into(),
        [0.0, speed, 0.0].into(),
      ),
    ]);
    let gravity = NBodyGravity::new(state.masses().to_vec());
    let mut stage = KeplerStage::new(gravity);
    let mut state = state;

    let day = 86_400.0;
    for _ in 0..365 {
      stage.step_in_place(&mut state, day);
    }

    let earth = state.body(1);
    let position = earth.position();
    let final_radius =
      (position[0] * position[0] + position[1] * position[1]).sqrt();
    assert!(((final_radius - radius) / radius).abs() < 5.0e-4);
  }
}
