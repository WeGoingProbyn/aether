// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Background-thread plumbing between a producer and a backend.
//!
//! Two pieces:
//!
//! - **Bounded SPSC channel.** [`render_channel`] returns a
//!   [`UpdateSender`] / [`UpdateReceiver`] pair. The sender uses
//!   `try_send` so a slow consumer applies back-pressure; on overflow
//!   it falls back to blocking `send`. Coalescing happens
//!   *receiver-side* via [`UpdateReceiver::drain_coalesced`] — by the
//!   time the bevy main thread reads, stale `Update*` attribute
//!   updates are squashed but every lifecycle (`Register*` / `Free*`)
//!   is preserved in order.
//!
//! - **Runner.** [`spawn_runner`] takes an arbitrary tick closure and
//!   spawns a worker thread that loops `tick → send → tick → …` until
//!   the returned [`RunnerHandle`] is dropped or `shutdown` is
//!   signalled. Eidolon stays free of the aether crate; the sandbox
//!   wires a real `Aether` + `FrameProducer` into the closure.

use std::{
  sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
  },
  thread::{self, JoinHandle},
  time::Duration,
};

use crossbeam_channel::{Receiver, Sender, TrySendError};
use utility::error::{AetherError, AetherResult, ErrorKind};

use crate::ir::{LayerHandle, MeshHandle, Update, UpdateBatch, WorldHandle};

/// Sender half of the producer→backend channel.
#[derive(Clone, Debug)]
pub struct UpdateSender {
  inner: Sender<UpdateBatch>,
}

/// Receiver half. Keeps no internal state — `drain_coalesced` does the
/// merging on demand.
#[derive(Debug)]
pub struct UpdateReceiver {
  inner: Receiver<UpdateBatch>,
}

/// Build a bounded SPSC channel sized for `capacity` pending batches.
pub fn render_channel(capacity: usize) -> (UpdateSender, UpdateReceiver) {
  let (tx, rx) = crossbeam_channel::bounded(capacity.max(1));
  (UpdateSender { inner: tx }, UpdateReceiver { inner: rx })
}

impl UpdateSender {
  /// Try to enqueue without blocking. On overflow, fall back to a
  /// blocking send so the producer applies back-pressure rather than
  /// silently dropping. Returns `Err` only if the receiver hung up.
  pub fn send(&self, batch: UpdateBatch) -> Result<(), UpdateBatch> {
    match self.inner.try_send(batch) {
      Ok(()) => Ok(()),
      Err(TrySendError::Full(batch)) => {
        self.inner.send(batch).map_err(|e| e.into_inner())
      }
      Err(TrySendError::Disconnected(batch)) => Err(batch),
    }
  }
}

impl UpdateReceiver {
  /// Pull every available batch and merge them into one. Returns
  /// `None` if the channel is empty.
  ///
  /// Merge rules:
  /// - `Register*` / `Free*` — keep all, in original order.
  /// - `UpdateMeshGeometry` for the same mesh — keep last only.
  /// - `UpdateMeshTransform` for the same mesh — keep last only.
  /// - `UpdateWorldTransform` for the same world — keep last only.
  /// - `UpdateLayerSamples` for the same layer — keep last only.
  /// - `UpdateLayerPalette` for the same layer — keep last only.
  /// - `UpdateLayerBinding` for the same mesh — keep last only.
  /// - `UpdateSunDirection` for the same world — keep last only.
  /// - `SetCamera` — keep last (latest view wins).
  /// - `SetSimTime` — keep last (latest sim time wins).
  pub fn drain_coalesced(&self) -> Option<UpdateBatch> {
    let mut first = self.inner.try_recv().ok()?;
    while let Ok(next) = self.inner.try_recv() {
      first.frame = next.frame;
      first.sim_time = next.sim_time;
      first.updates.extend(next.updates);
    }
    Some(coalesce_batch(first))
  }

  /// Block up to `timeout` waiting for a batch. Useful in tests and
  /// for backends that want a hard shutdown deadline.
  pub fn recv_timeout(&self, timeout: Duration) -> Option<UpdateBatch> {
    self.inner.recv_timeout(timeout).ok()
  }
}

/// Coalesce a single (already-concatenated) batch in place.
///
/// Walks back-to-front. The last attribute update for each
/// `(handle, kind)` wins; everything else (every Register/Free, the
/// final `SetSimTime`) is preserved.
fn coalesce_batch(batch: UpdateBatch) -> UpdateBatch {
  use std::collections::HashSet;

  let UpdateBatch {
    frame,
    sim_time,
    updates,
  } = batch;

  let mut seen_mesh_geometry: HashSet<MeshHandle> = HashSet::new();
  let mut seen_mesh_transform: HashSet<MeshHandle> = HashSet::new();
  let mut seen_world_transform: HashSet<WorldHandle> = HashSet::new();
  let mut seen_layer_samples: HashSet<LayerHandle> = HashSet::new();
  let mut seen_layer_palette: HashSet<LayerHandle> = HashSet::new();
  let mut seen_layer_binding: HashSet<MeshHandle> = HashSet::new();
  let mut seen_sun: HashSet<WorldHandle> = HashSet::new();
  let mut seen_camera = false;
  let mut seen_set_sim_time = false;

  let mut reversed: Vec<Update> = Vec::with_capacity(updates.len());
  for update in updates.into_iter().rev() {
    match &update {
      Update::UpdateMeshGeometry { handle, .. } => {
        if !seen_mesh_geometry.insert(*handle) {
          continue;
        }
      }
      Update::UpdateMeshTransform { handle, .. } => {
        if !seen_mesh_transform.insert(*handle) {
          continue;
        }
      }
      Update::UpdateWorldTransform { handle, .. } => {
        if !seen_world_transform.insert(*handle) {
          continue;
        }
      }
      Update::UpdateLayerSamples { handle, .. } => {
        if !seen_layer_samples.insert(*handle) {
          continue;
        }
      }
      Update::UpdateLayerPalette { handle, .. } => {
        if !seen_layer_palette.insert(*handle) {
          continue;
        }
      }
      Update::UpdateLayerBinding { mesh, .. } => {
        if !seen_layer_binding.insert(*mesh) {
          continue;
        }
      }
      Update::UpdateSunDirection { world, .. } => {
        if !seen_sun.insert(*world) {
          continue;
        }
      }
      Update::SetCamera { .. } => {
        if seen_camera {
          continue;
        }
        seen_camera = true;
      }
      Update::SetSimTime { .. } => {
        if seen_set_sim_time {
          continue;
        }
        seen_set_sim_time = true;
      }
      // Lifecycle: keep all, preserve original order.
      _ => {}
    }
    reversed.push(update);
  }
  reversed.reverse();

  UpdateBatch {
    frame,
    sim_time,
    updates: reversed,
  }
}

/// Handle to a spawned runner thread. Drop or call
/// [`shutdown_and_join`] to stop the loop and wait for the worker to
/// exit.
pub struct RunnerHandle {
  shutdown: Arc<AtomicBool>,
  join: Option<JoinHandle<AetherResult<()>>>,
}

impl RunnerHandle {
  pub fn shutdown(&self) {
    self.shutdown.store(true, Ordering::Release);
  }

  pub fn shutdown_and_join(mut self) -> AetherResult<()> {
    self.shutdown();
    if let Some(handle) = self.join.take() {
      match handle.join() {
        Ok(result) => result,
        Err(_) => Err(
          AetherError::new(ErrorKind::Unknown)
            .context("runner thread panicked"),
        ),
      }
    } else {
      Ok(())
    }
  }
}

impl Drop for RunnerHandle {
  fn drop(&mut self) {
    self.shutdown.store(true, Ordering::Release);
    if let Some(handle) = self.join.take() {
      let _ = handle.join();
    }
  }
}

/// Spawn a worker that calls `tick_fn` in a loop, sending each
/// produced batch over the channel. The closure receives a borrow of
/// the shutdown flag so it can early-exit between sub-steps.
///
/// The worker exits cleanly when `tick_fn` returns `Ok(None)` or when
/// the shutdown flag is set.
pub fn spawn_runner<F>(sender: UpdateSender, tick_fn: F) -> RunnerHandle
where
  F: FnMut(&Arc<AtomicBool>) -> AetherResult<Option<UpdateBatch>>
    + Send
    + 'static,
{
  let shutdown = Arc::new(AtomicBool::new(false));
  let shutdown_for_thread = Arc::clone(&shutdown);
  let join = thread::spawn(move || -> AetherResult<()> {
    let mut tick_fn = tick_fn;
    while !shutdown_for_thread.load(Ordering::Acquire) {
      match tick_fn(&shutdown_for_thread)? {
        Some(batch) => {
          // If the receiver is gone we silently exit. The runner is a
          // pure side-effect; producing batches with no consumer is
          // pointless.
          if sender.send(batch).is_err() {
            break;
          }
        }
        None => break,
      }
    }
    Ok(())
  });
  RunnerHandle {
    shutdown,
    join: Some(join),
  }
}
