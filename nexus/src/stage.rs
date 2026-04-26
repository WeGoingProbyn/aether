// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use pleroma::prelude::WorldAccess;
use utility::{domain::FieldKey, error::AetherResult, thread::pool::Pool};

/// One unit of physics work inside a `Schedule`. Stages declare which fields
/// they read and write; nexus uses those declarations to build a DAG and run
/// non-conflicting stages in parallel.
pub trait Stage: Send + Sync {
  fn name(&self) -> &'static str;
  fn reads(&self) -> &[FieldKey];
  fn writes(&self) -> &[FieldKey];
  fn run(&self, ctx: StageContext<'_>) -> AetherResult<()>;
}

pub struct StageContext<'a> {
  /// Typed read/write into pleroma, scoped to the keys the stage declared.
  pub world: WorldAccess<'a>,
  /// For inner parallelism (e.g. `continuum::FvmSolver::parallel_step`).
  pub pool: &'a Pool,
  /// Time step picked by the integration driver.
  pub dt: f64,
}
