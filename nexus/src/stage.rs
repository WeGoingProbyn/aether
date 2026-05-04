// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use pleroma::prelude::WorldAccess;
use tessera::world_mesh::Tessera;
use utility::{
  domain::{FieldKey, ResourceKey, WorldId},
  error::AetherResult,
  thread::pool::{Pool, ScopedScheduler},
};

use crate::constants::WorldConstants;

/// One unit of physics work inside a `Nexus`. Stages declare which fields
/// and resources they read and write; nexus uses those declarations to
/// build a DAG and run non-conflicting stages in parallel.
pub trait Stage: Send + Sync {
  fn name(&self) -> &'static str;
  fn reads(&self) -> &[FieldKey];
  fn writes(&self) -> &[FieldKey];

  /// Non-mesh-bound resources this stage reads (e.g. body state, sun
  /// direction). Default is empty.
  fn resource_reads(&self) -> &[ResourceKey] {
    &[]
  }

  /// Non-mesh-bound resources this stage writes. Default is empty.
  fn resource_writes(&self) -> &[ResourceKey] {
    &[]
  }

  fn run(&mut self, ctx: StageContext<'_>) -> AetherResult<()>;

  fn plan<'a>(
    &'a mut self,
    ctx: StageContext<'a>,
  ) -> AetherResult<StagePlan<'a>> {
    let name = self.name();
    Ok(StagePlan::single(name, move || self.run(ctx)))
  }
}

pub enum StagePlan<'a> {
  Static(Vec<StagePlanTask<'a>>),
  Program(Box<dyn StageProgram<'a> + Send + 'a>),
}

impl<'a> StagePlan<'a> {
  pub fn single(
    name: &'static str,
    task: impl FnOnce() -> AetherResult<()> + Send + 'a,
  ) -> Self {
    Self::Static(vec![StagePlanTask {
      name,
      task: Box::new(task),
      predecessors: Vec::new(),
    }])
  }

  pub fn from_tasks(tasks: Vec<StagePlanTask<'a>>) -> Self {
    Self::Static(tasks)
  }

  pub fn program(program: impl StageProgram<'a> + Send + 'a) -> Self {
    Self::Program(Box::new(program))
  }
}

pub trait StageProgram<'a>: Send {
  fn execute(
    self: Box<Self>,
    scheduler: &mut ScopedScheduler,
  ) -> AetherResult<()>;
}

impl<'a, F> StageProgram<'a> for F
where
  F: FnOnce(&mut ScopedScheduler) -> AetherResult<()> + Send + 'a,
{
  fn execute(
    self: Box<Self>,
    scheduler: &mut ScopedScheduler,
  ) -> AetherResult<()> {
    self(scheduler)
  }
}

impl<'a> From<Vec<StagePlanTask<'a>>> for StagePlan<'a> {
  fn from(tasks: Vec<StagePlanTask<'a>>) -> Self {
    StagePlan::Static(tasks)
  }
}

impl<'a> From<StagePlanTask<'a>> for StagePlan<'a> {
  fn from(task: StagePlanTask<'a>) -> Self {
    StagePlan::Static(vec![task])
  }
}

impl<'a> StagePlanTask<'a> {
  pub fn new(
    name: &'static str,
    task: impl FnOnce() -> AetherResult<()> + Send + 'a,
  ) -> Self {
    StagePlanTask {
      name,
      task: Box::new(task),
      predecessors: Vec::new(),
    }
  }

  pub fn after(mut self, predecessors: Vec<usize>) -> Self {
    self.predecessors = predecessors;
    self
  }
}

pub struct StagePlanTask<'a> {
  pub name: &'static str,
  pub task: Box<dyn FnOnce() -> AetherResult<()> + Send + 'a>,
  pub predecessors: Vec<usize>,
}

pub struct WorldView<'a> {
  /// Stable ID of the body/world being ticked.
  pub world_id: WorldId,
  /// Read-only geometry, topology, couplers, and partition metadata.
  pub tessera: &'a Tessera,
  /// Immutable per-world constants derived during world setup.
  pub constants: &'a WorldConstants,
  /// Typed read/write into pleroma, scoped to the keys the stage declared.
  pub fields: WorldAccess<'a>,
  /// Shared worker pool for stages that need scheduling context.
  pub pool: &'a Pool,
  /// Time step picked by the integration driver.
  pub dt: f64,
  /// Static decomposition hint declared by the owning world.
  pub partition_count: usize,
}

pub struct StageContext<'a> {
  /// Stage-scoped world view assembled by nexus from tessera + pleroma.
  pub world: WorldView<'a>,
}
