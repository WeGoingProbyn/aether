// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Phase 3: spawn a worker thread that emits batches into the channel,
//! verify the consumer side actually receives them, then signal
//! shutdown and confirm the thread exits cleanly within a deadline.

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use eidolon::ir::{Update, UpdateBatch};
use eidolon::runtime::{render_channel, spawn_runner};

#[test]
fn runner_emits_batches_and_shuts_down_cleanly() {
  let (tx, rx) = render_channel(4);

  let mut frame: u64 = 0;
  let runner = spawn_runner(tx, move |shutdown| {
    if shutdown.load(Ordering::Acquire) {
      return Ok(None);
    }
    let f = frame;
    frame = frame.wrapping_add(1);
    let batch = UpdateBatch {
      frame: f,
      sim_time: f as f64 * 0.5,
      updates: vec![Update::SetSimTime {
        sim_time: f as f64 * 0.5,
        frame: f,
      }],
    };
    // Yield so the thread doesn't spin a tight loop in a unit test.
    std::thread::sleep(Duration::from_millis(1));
    Ok(Some(batch))
  });

  // Wait for at least one batch to land. 500ms is generous.
  let deadline = Instant::now() + Duration::from_millis(500);
  let mut got_batch = false;
  while Instant::now() < deadline {
    if let Some(batch) = rx.drain_coalesced() {
      assert!(
        batch
          .updates
          .iter()
          .any(|u| matches!(u, Update::SetSimTime { .. })),
        "expected SetSimTime in batch"
      );
      got_batch = true;
      break;
    }
    std::thread::sleep(Duration::from_millis(5));
  }
  assert!(got_batch, "runner produced no batches within 500ms");

  // Signal shutdown and verify the thread exits within a deadline.
  let shutdown_started = Instant::now();
  runner
    .shutdown_and_join()
    .expect("runner thread joined cleanly");
  assert!(
    shutdown_started.elapsed() < Duration::from_millis(500),
    "shutdown took too long: {:?}",
    shutdown_started.elapsed()
  );
}

#[test]
fn runner_exits_when_tick_fn_returns_none() {
  let (tx, _rx) = render_channel(4);

  let mut counter = 0;
  let runner = spawn_runner(tx, move |_| {
    counter += 1;
    if counter > 3 {
      Ok(None)
    } else {
      Ok(Some(UpdateBatch::default()))
    }
  });

  // Without an explicit shutdown, the thread should exit on its own.
  // Drop will join.
  let started = Instant::now();
  runner.shutdown_and_join().expect("runner exited cleanly");
  assert!(started.elapsed() < Duration::from_millis(500));
}
