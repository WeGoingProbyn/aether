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
//! `CompiledNexus::tick` walks the resulting layers; within a layer all
//! stages have pairwise-disjoint conflicts, so they fan out across the
//! thread pool via the `pleroma::ScheduleAccess` split-borrow.

use std::sync::{Arc, Mutex};

use pleroma::Pleroma;
use pleroma::prelude::FieldKey;
use tessera::world_mesh::Tessera;
use utility::collections::graph::Graph;
use utility::domain::WorldId;
use utility::error::{AetherError, AetherResult, ErrorDomain};
use utility::thread::pool::Pool;

use crate::{
  constants::WorldConstants,
  stage::{Stage, StageContext, WorldView},
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
  intersects(a.writes(), b.reads())  // RAW: a writes, b reads
    || intersects(a.reads(), b.writes())  // WAR: a reads, b overwrites
    || intersects(a.writes(), b.writes()) // WAW: both write same key
}

fn intersects(a: &[FieldKey], b: &[FieldKey]) -> bool {
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
    for layer in &self.layers {
      let access = pleroma.schedule_access();
      let error_slot: Arc<Mutex<Option<AetherError>>> =
        Arc::new(Mutex::new(None));

      let mut tasks: Vec<Box<dyn FnOnce() + Send + '_>> =
        Vec::with_capacity(layer.len());

      for (idx, stage) in self.stages.iter_mut().enumerate() {
        if !layer.iter().any(|stage_id| stage_id.0 == idx) {
          continue;
        }

        let reads = stage.reads().to_vec();
        let writes = stage.writes().to_vec();

        // SAFETY: `build` placed these stages in the same layer only when
        // none of them have any RAW/WAR/WAW conflict with each other. The
        // declared reads/writes are therefore pairwise disjoint across the
        // views we hand out here, which is exactly the precondition
        // `ScheduleAccess::view_for` requires.
        let view = unsafe { access.view_for(&reads, &writes) };

        let ctx = StageContext {
          world: WorldView {
            world_id,
            tessera,
            constants,
            fields: view,
            pool,
            dt,
          },
        };
        let err_slot = Arc::clone(&error_slot);

        tasks.push(Box::new(move || {
          if let Err(e) = stage.run(ctx) {
            let mut guard = err_slot.lock().unwrap();
            if guard.is_none() {
              *guard = Some(e);
            }
          }
        }));
      }

      pool.dispatch(tasks);

      //drop(access);

      let err = error_slot.lock().unwrap().take();
      if let Some(e) = err {
        return Err(e);
      }
    }
    Ok(())
  }
}

#[derive(Debug)]
pub enum NexusError {
  UnknownStage,
  Cycle,
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
    }
  }
}
