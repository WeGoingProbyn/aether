// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Nexus construction and execution.
//!
//! `Nexus::build` turns a list of stages plus their declared reads/writes
//! into a layered DAG. Edges fall out of three relationships:
//!
//! - **RAW** (read-after-write): stage `B` reads what stage `A` writes.
//! - **WAR** (write-after-read): stage `B` writes what stage `A` reads.
//! - **WAW** (write-after-write): both write the same field.
//!
//! Whenever two stages have any of those conflicts, the earlier-added stage
//! runs first. Callers can pin extra ordering with `Nexus::before`. Cycles
//! (whether from the data flow or from contradictory `before` hints) are
//! surfaced as errors during `build`.
//!
//! `CompiledNexus::tick` builds scheduler-owned tasks from the resulting
//! layers. Within a layer all stages have pairwise-disjoint conflicts, and
//! layer barriers preserve the same ordering semantics as the original
//! per-layer dispatch path.

use pleroma::Pleroma;
use pleroma::prelude::{FieldKey, ResourceKey};
use tessera::world_mesh::Tessera;
use utility::collections::graph::Graph;
use utility::domain::WorldId;
use utility::error::{AetherError, AetherResult, ErrorDomain};
use utility::thread::pool::{Pool, ScopedTaskGraph};
use utility::{end_profile, inline_profile};

use crate::{
  constants::WorldConstants,
  stage::{
    Stage, StageContext, StagePlan, StagePlanTask, StageProgram, WorldView,
  },
};

/// Stable identifier for a stage inside one `Nexus`. Returned by
/// `Nexus::add` and used for `before` ordering hints.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct StageId(pub(crate) usize);

impl StageId {
  /// Position of this stage in its `Nexus` — also its index in the
  /// flattened topological order returned by `CompiledNexus::topo_order`.
  pub fn index(&self) -> usize {
    self.0
  }
}

#[derive(Default)]
pub struct Nexus {
  stages: Vec<Box<dyn Stage>>,
  ordering_hints: Vec<(StageId, StageId)>,
}

impl Nexus {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn add(&mut self, stage: impl Stage + 'static) -> StageId {
    let id = StageId(self.stages.len());
    self.stages.push(Box::new(stage));
    id
  }

  /// Force `b` to run after `a` even when their declared reads/writes
  /// don't conflict. Useful for physical ordering between independent
  /// fields (e.g. integrate orbits before rotating the atmosphere).
  pub fn before(&mut self, a: StageId, b: StageId) {
    self.ordering_hints.push((a, b));
  }

  pub fn build(self, _world: &Pleroma) -> AetherResult<CompiledNexus> {
    let n = self.stages.len();

    // Build a graph keyed by stage index. Node data is unit — the StageId
    // is the node id (0..n).
    let mut graph: Graph<()> = Graph::new();
    for _ in 0..n {
      graph.add_node(());
    }

    // Data-flow edges: any conflict between i < j adds edge i → j.
    for i in 0..n {
      for j in (i + 1)..n {
        let a = self.stages[i].as_ref();
        let b = self.stages[j].as_ref();
        if has_conflict(a, b) {
          graph.add_edge(i, j)?;
        }
      }
    }

    // Explicit `before` hints. Caller-supplied edges that don't follow
    // add-order are allowed and will form cycles only if combined with a
    // contradictory data-flow conflict (caught below by topological_sort).
    for (a, b) in &self.ordering_hints {
      if a.0 >= n || b.0 >= n {
        return Err(AetherError::new(NexusError::UnknownStage).context(
          format!(
            "before hint {:?} -> {:?} references stage out of range (n = {})",
            a, b, n
          ),
        ));
      }
      graph.add_edge(a.0, b.0)?;
    }

    // Cycle check via topological sort (the result also gives a sequential
    // execution order if the caller wants one).
    let topo_order = graph.topological_sort()?;

    let layers = build_layers(&graph, n)?;

    Ok(CompiledNexus {
      stages: self.stages,
      layers,
      topo_order,
    })
  }
}

fn has_conflict(a: &dyn Stage, b: &dyn Stage) -> bool {
  // Field RAW/WAR/WAW.
  intersects(a.writes(), b.reads())
    || intersects(a.reads(), b.writes())
    || intersects(a.writes(), b.writes())
    // Resource RAW/WAR/WAW — same shape as fields.
    || resource_intersects(a.resource_writes(), b.resource_reads())
    || resource_intersects(a.resource_reads(), b.resource_writes())
    || resource_intersects(a.resource_writes(), b.resource_writes())
}

fn intersects(a: &[FieldKey], b: &[FieldKey]) -> bool {
  a.iter().any(|k| b.contains(k))
}

fn resource_intersects(a: &[ResourceKey], b: &[ResourceKey]) -> bool {
  a.iter().any(|k| b.contains(k))
}

/// Kahn's algorithm with layer tracking. Each iteration peels off all
/// zero-indegree nodes into the current layer; ties within a layer mean
/// "could run in parallel".
fn build_layers(
  graph: &Graph<()>,
  n: usize,
) -> AetherResult<Vec<Vec<StageId>>> {
  let mut indegree: Vec<usize> =
    (0..n).map(|i| graph.incoming_edges(i).len()).collect();

  let mut layers: Vec<Vec<StageId>> = Vec::new();
  let mut current: Vec<usize> = (0..n).filter(|&i| indegree[i] == 0).collect();

  let mut placed = 0usize;
  while !current.is_empty() {
    let mut next: Vec<usize> = Vec::new();
    for &node in &current {
      if let Some(edges) = graph.outgoing_edges(node) {
        for edge in edges {
          indegree[edge.target] -= 1;
          if indegree[edge.target] == 0 {
            next.push(edge.target);
          }
        }
      }
    }
    placed += current.len();
    layers.push(current.into_iter().map(StageId).collect());
    current = next;
  }

  if placed != n {
    return Err(
      AetherError::new(NexusError::Cycle)
        .context("nexus contains a cycle (unexpected after topo sort)"),
    );
  }

  Ok(layers)
}

pub struct CompiledNexus {
  stages: Vec<Box<dyn Stage>>,
  layers: Vec<Vec<StageId>>,
  topo_order: Vec<usize>,
}

pub struct StageTask<'a> {
  pub name: &'static str,
  pub task: ScheduledStageTask<'a>,
  pub predecessors: Vec<usize>,
}

pub enum ScheduledStageTask<'a> {
  Worker(Box<dyn FnOnce() -> AetherResult<()> + Send + 'a>),
  Program(Box<dyn StageProgram<'a> + Send + 'a>),
}

impl CompiledNexus {
  pub fn stage_count(&self) -> usize {
    self.stages.len()
  }

  pub fn layer_count(&self) -> usize {
    self.layers.len()
  }

  pub fn layers(&self) -> &[Vec<StageId>] {
    &self.layers
  }

  /// Sequential topological order — useful for serial execution and tests.
  pub fn topo_order(&self) -> &[usize] {
    &self.topo_order
  }

  pub fn tick(
    &mut self,
    world_id: WorldId,
    tessera: &Tessera,
    constants: &WorldConstants,
    pleroma: &mut Pleroma,
    pool: &Pool,
    dt: f64,
  ) -> AetherResult<()> {
    self.tick_with_partition_count(
      world_id, tessera, constants, pleroma, pool, dt, 1,
    )
  }

  pub fn tick_with_partition_count(
    &mut self,
    world_id: WorldId,
    tessera: &Tessera,
    constants: &WorldConstants,
    pleroma: &mut Pleroma,
    pool: &Pool,
    dt: f64,
    partition_count: usize,
  ) -> AetherResult<()> {
    let tasks = self.build_tick_tasks(
      world_id,
      tessera,
      constants,
      pleroma,
      pool,
      dt,
      partition_count,
    )?;
    let mut graph = ScopedTaskGraph::new();
    let mut node_ids = Vec::with_capacity(tasks.len());

    for task in tasks {
      let StageTask {
        name,
        task,
        predecessors,
      } = task;
      let node = match task {
        ScheduledStageTask::Worker(task) => graph.add(task),
        ScheduledStageTask::Program(program) => {
          graph.add_scheduler(move |scheduler| {
            inline_profile!(name);
            let result = program.execute(scheduler);
            end_profile!(name);
            result
          })
        }
      };
      for predecessor in predecessors {
        graph.dependency(node, node_ids[predecessor])?;
      }
      node_ids.push(node);
    }

    pool.execute_scoped(graph)
  }

  pub fn build_tick_tasks<'a>(
    &'a mut self,
    world_id: WorldId,
    tessera: &'a Tessera,
    constants: &'a WorldConstants,
    pleroma: &'a mut Pleroma,
    pool: &'a Pool,
    dt: f64,
    partition_count: usize,
  ) -> AetherResult<Vec<StageTask<'a>>> {
    inline_profile!("nexus.tick.layer_build_tasks");

    let access = pleroma.schedule_access();
    let layers = self.layers.clone();
    let mut tasks = Vec::with_capacity(self.stages.len());
    let stages = self.stages.as_mut_ptr();
    let mut previous_layer_terminals: Vec<usize> = Vec::new();

    for layer in layers {
      let mut current_layer_terminals = Vec::new();

      for stage_id in layer {
        let stage = unsafe { &mut *stages.add(stage_id.0) };
        let reads = stage.reads().to_vec();
        let writes = stage.writes().to_vec();
        let resource_reads = stage.resource_reads().to_vec();
        let resource_writes = stage.resource_writes().to_vec();

        // SAFETY: the scoped task graph gives the same execution guarantee
        // the previous layer-by-layer dispatcher did: stages in the same layer
        // are conflict-free, and later layers cannot start until all terminal
        // tasks in the previous layer have completed. `WorldAccess` values are
        // created up front, but typed field/resource references are only
        // materialized inside the scheduled task closures.
        let view = unsafe {
          access.view_for(&reads, &writes, &resource_reads, &resource_writes)
        };

        let ctx = StageContext {
          world: WorldView {
            world_id,
            tessera,
            constants,
            fields: view,
            pool,
            dt,
            partition_count,
          },
        };

        let stage_name = stage.name();
        match stage.plan(ctx)? {
          StagePlan::Static(planned_tasks) => {
            let base = tasks.len();
            let local_count = planned_tasks.len();
            let mut has_local_dependents = vec![false; local_count];
            for planned in &planned_tasks {
              for &predecessor in &planned.predecessors {
                if predecessor >= local_count {
                  return Err(
                    AetherError::new(NexusError::InvalidStagePlan).context(
                      format!(
                        "stage {} references local predecessor {} with only {} tasks",
                        planned.name, predecessor, local_count
                      ),
                    ),
                  );
                }
                has_local_dependents[predecessor] = true;
              }
            }

            for (local_index, planned) in planned_tasks.into_iter().enumerate()
            {
              let StagePlanTask {
                name,
                task,
                predecessors,
              } = planned;
              let mut resolved_predecessors = predecessors
                .into_iter()
                .map(|idx| base + idx)
                .collect::<Vec<_>>();
              if resolved_predecessors.is_empty() {
                resolved_predecessors
                  .extend(previous_layer_terminals.iter().copied());
              }

              tasks.push(StageTask {
                name,
                task: ScheduledStageTask::Worker(Box::new(move || {
                  inline_profile!(name);
                  let result = task();
                  end_profile!(name);
                  result
                })),
                predecessors: resolved_predecessors,
              });

              if !has_local_dependents[local_index] {
                current_layer_terminals.push(base + local_index);
              }
            }
          }
          StagePlan::Program(program) => {
            let index = tasks.len();
            tasks.push(StageTask {
              name: stage_name,
              task: ScheduledStageTask::Program(program),
              predecessors: previous_layer_terminals.clone(),
            });
            current_layer_terminals.push(index);
          }
        }
      }

      previous_layer_terminals = current_layer_terminals;
    }

    end_profile!("nexus.tick.layer_build_tasks");
    Ok(tasks)
  }
}

#[derive(Debug)]
pub enum NexusError {
  UnknownStage,
  Cycle,
  InvalidStagePlan,
}

impl ErrorDomain for NexusError {
  fn domain(&self) -> &str {
    "nexus"
  }
}

impl std::fmt::Display for NexusError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      NexusError::UnknownStage => {
        write!(f, "ordering hint references a stage that wasn't added")
      }
      NexusError::Cycle => write!(f, "nexus has a cycle"),
      NexusError::InvalidStagePlan => {
        write!(f, "stage returned an invalid execution plan")
      }
    }
  }
}
