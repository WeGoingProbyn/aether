// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::{
  sync::{Mutex, atomic::Ordering},
  time::Duration,
};

use crate::{
  error::Unpoison,
  profiler::Profiler,
  thread::{pool::Context, task::Job},
};

pub struct Queue {
  queue: Mutex<VecDeque<Job>>,
  index: usize,
}

impl Queue {
  pub(crate) fn new(index: usize) -> Queue {
    Queue {
      queue: Mutex::new(VecDeque::new()),
      index,
    }
  }

  pub(crate) fn push(&self, job: Job) {
    self.queue.lock().unpoison().push_back(job);
  }

  pub(crate) fn pop_back(&self) -> Option<Job> {
    self.queue.lock().unpoison().pop_back()
  }

  pub(crate) fn pop_front(&self) -> Option<Job> {
    self.queue.lock().unpoison().pop_front()
  }

  pub(crate) fn worker_loop(&self, context: &Context) {
    loop {
      // do we need to shutdown?
      if context.shutdown.load(Ordering::Relaxed) {
        for job in self.queue.lock().unpoison().drain(..) {
          job()
        }
        Profiler::flush_local();
        return;
      }

      // Drain our own queue first: FILO
      if let Some(job) = self.pop_back() {
        //let _span = SpanGuard::new("worker::execute", "thread");
        job();
        continue;
      }

      // Drain other worker queues next: FIFO
      let mut stolen = false;
      for worker in &context.workers {
        if self.index == worker.index {
          continue;
        }

        if let Some(job) = worker.pop_front() {
          //let _span = SpanGuard::new("worker::steal", "thread");
          job();
          stolen = true;
          break;
        }
      }
      if stolen {
        continue;
      }

      // Wait 1ms before trying again
      // let _span = SpanGuard::new("worker::barrier", "thread");
      let guard = context.global_mutex.lock().unpoison();
      let _ = context
        .global_barrier
        .wait_timeout(guard, Duration::from_millis(1));
    }
  }
}
