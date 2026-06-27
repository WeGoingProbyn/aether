// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Deferred-dispatch event bus — the runtime's broadcast channel.
//!
//! In-DAG stages and the `World` [`emit`](EventBus::emit) [`Event`]s during a
//! tick; the buffer is rotated (published) at the single-threaded end-of-tick
//! barrier and read back via a poll API (`World::events()`). This is the
//! announce/react seam: a stage broadcasts a state change without knowing who
//! listens, and consumers (or other stages, reading last tick's batch) react.
//!
//! The vocabulary here is law-agnostic and dependency-free (mirroring
//! [`crate::diagnostics`]), so any producer or consumer shares it without
//! depending on each other.
//!
//! # Guarantees and limitations
//!
//! - **Cross-tick order is preserved**: tick N's batch is fully published before
//!   tick N+1's.
//! - **No intra-tick ordering**: a tick's events are pushed by parallel stages in
//!   lock-acquisition order, so within one tick they are an *unordered set* — do
//!   not rely on causal ordering between two events of the same tick.
//! - **No volume bound**: a tick may emit any number of events (there is no cap or
//!   dedup in this version). Emitters are expected to be sparing — e.g. one event
//!   per monitored field per tick, not per cell.
//! - The single `Mutex` is a contention point only if many stages emit heavily in
//!   one tick; the [`emit`](EventBus::emit)/[`publish`](EventBus::publish) API is
//!   the seam, so the internal structure can change without touching callers.

use std::sync::{Arc, Mutex};

use crate::domain::{CellRemap, FieldKey, MeshKey, TopologyEpoch};

/// A time-advance regime, as a dependency-free vocabulary value. Mirrors the
/// `chronos::Regime` states without `utility` depending on `chronos`; the runtime
/// maps between the two at the emit site.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum RegimeKind {
  Live,
  Climatology,
}

/// A broadcast state-change. Law-agnostic and extensible: new variants are added
/// here as new producers appear, without changing the bus.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
  /// A monitored field went non-finite (NaN/Inf) this tick. The soft counterpart
  /// to a `DiagnosticsPolicy::Fail` `Err`: published whether or not the tick also
  /// hard-failed, so a consumer can react (throttle, roll back, flip regime).
  NonFiniteState { field: FieldKey },
  /// A monitored field's conserved totals drifted past the monitor's threshold.
  ConservationDrift { field: FieldKey, drift: f64 },
  /// The world's advance regime changed (e.g. via `World::set_regime`).
  RegimeChanged { from: RegimeKind, to: RegimeKind },
  /// A live↔climatology handoff began, targeting `to`.
  TransitionStarted { to: RegimeKind },
  /// A live↔climatology handoff completed, settling into `to`.
  TransitionCompleted { to: RegimeKind },
  /// A mesh's topology was adapted (refined / coarsened) at the end-of-tick
  /// barrier. Carries the new [`TopologyEpoch`] and the old→new [`CellRemap`], so
  /// consumers (the query index, the render producer, checkpointing) can rebuild
  /// or re-initialise against the new dense `CellId` space. The remap is shared
  /// (`Arc`) so publishing and cloning the batch stays cheap even when the mesh
  /// is large.
  TopologyChanged {
    mesh: MeshKey,
    epoch: TopologyEpoch,
    remap: Arc<CellRemap>,
  },
}

/// The event channel held as a pleroma resource. Double-buffered: stages emit
/// into `pending` (under a lock, via a shared `&EventBus`, so parallel emitters
/// are sound), and the world rotates `pending` into `published` at the end-of-tick
/// barrier. Consumers and stages read `published` (last tick's batch).
#[derive(Default)]
pub struct EventBus {
  pending: Mutex<Vec<Event>>,
  published: Vec<Event>,
}

impl EventBus {
  pub fn new() -> Self {
    Self::default()
  }

  /// Emit an event into this tick's pending buffer. Takes `&self` (interior
  /// mutability) so an in-DAG stage holding a *shared* `&EventBus` from
  /// `resource_reads` can emit; the lock serialises concurrent pushes.
  pub fn emit(&self, event: Event) {
    self
      .pending
      .lock()
      .expect("event bus lock poisoned")
      .push(event);
  }

  /// Last tick's published events (the set a consumer or stage reads).
  pub fn published(&self) -> &[Event] {
    &self.published
  }

  /// Rotate this tick's pending events into the published buffer. Called once,
  /// single-threaded, at the end-of-tick barrier (the world holds `&mut`).
  pub fn publish(&mut self) {
    let drained = std::mem::take(
      &mut *self.pending.lock().expect("event bus lock poisoned"),
    );
    self.published = drained;
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::domain::{FieldName, MeshKey};

  fn temp() -> FieldKey {
    FieldKey::new(MeshKey::ATMOSPHERE, FieldName::Temperature)
  }

  #[test]
  fn emitted_events_are_invisible_until_published() {
    let mut bus = EventBus::new();
    bus.emit(Event::NonFiniteState { field: temp() });
    // Not visible until the barrier rotates the buffer.
    assert!(bus.published().is_empty());
    bus.publish();
    assert_eq!(bus.published(), &[Event::NonFiniteState { field: temp() }]);
  }

  #[test]
  fn publish_clears_pending_so_the_next_tick_starts_empty() {
    let mut bus = EventBus::new();
    bus.emit(Event::RegimeChanged {
      from: RegimeKind::Live,
      to: RegimeKind::Climatology,
    });
    bus.publish();
    assert_eq!(bus.published().len(), 1);
    // A tick that emits nothing publishes an empty batch.
    bus.publish();
    assert!(bus.published().is_empty());
  }

  #[test]
  fn shared_ref_can_emit_via_interior_mutability() {
    let bus = EventBus::new();
    let shared: &EventBus = &bus;
    shared.emit(Event::ConservationDrift {
      field: temp(),
      drift: 0.5,
    });
    // Still pending (no &mut publish yet), but the emit through `&self` worked.
    assert!(bus.published().is_empty());
  }
}
