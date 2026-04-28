// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use pleroma::prelude::WorldAccess;
use tessera::world_mesh::Tessera;
use utility::{
  domain::{FieldKey, WorldId},
  error::AetherResult,
  thread::pool::Pool,
};

/// One unit of physics work inside a `Nexus`. Stages declare which fields
/// they read and write; nexus uses those declarations to build a DAG and run
/// non-conflicting stages in parallel.
pub trait Stage: Send + Sync {
  fn name(&self) -> &'static str;
  fn reads(&self) -> &[FieldKey];
  fn writes(&self) -> &[FieldKey];
  fn run(&mut self, ctx: StageContext<'_>) -> AetherResult<()>;
}

pub struct WorldView<'a> {
  /// Stable ID of the body/world being ticked.
  pub world_id: WorldId,
  /// Read-only geometry, topology, couplers, and partition metadata.
  pub tessera: &'a Tessera,
  /// Typed read/write into pleroma, scoped to the keys the stage declared.
  pub fields: WorldAccess<'a>,
  /// For inner parallelism (e.g. `continuum::FvmSolver::parallel_step`).
  pub pool: &'a Pool,
  /// Time step picked by the integration driver.
  pub dt: f64,
}

pub struct StageContext<'a> {
  /// Stage-scoped world view assembled by nexus from tessera + pleroma.
  pub world: WorldView<'a>,
}
