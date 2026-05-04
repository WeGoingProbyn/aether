// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use crate::error::ErrorDomain;

pub mod pool;
pub mod task;
pub mod worker;

pub enum ErrorKind {
  ThreadPoolShutdown,
  ThreadPoolPanic,
  ReductionEmpty,
  ReductionInputOutOfRange,
  ReductionMissingInput,
  ReductionNotReady,
}

impl ErrorDomain for ErrorKind {
  fn domain(&self) -> &str {
    "thread"
  }
}

impl std::fmt::Display for ErrorKind {
  fn fmt(
    &self,
    f: &mut std::fmt::Formatter<'_>,
  ) -> Result<(), std::fmt::Error> {
    let string = match self {
      ErrorKind::ThreadPoolPanic => {
        "a task inside the threading pool has panicked"
      }
      ErrorKind::ThreadPoolShutdown => {
        "a task has been submitted to a dropped pool"
      }
      ErrorKind::ReductionEmpty => {
        "a scheduler reduction was created without input slots"
      }
      ErrorKind::ReductionInputOutOfRange => {
        "a scheduler reduction input index is out of range"
      }
      ErrorKind::ReductionMissingInput => {
        "a scheduler reduction was run before all inputs were written"
      }
      ErrorKind::ReductionNotReady => {
        "a scheduler reduction value was read before the reduction ran"
      }
    };

    write!(f, "{}", string)?;
    Ok(())
  }
}

#[cfg(test)]
mod test {
  use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
  };
  use std::time::Duration;
  use utility::error::{AetherError, ErrorDomain, Unpoison};
  use utility::thread::pool::{
    Pool, ScopedReduction, ScopedTaskGraph, TaskGraph,
  };

  #[test]
  fn spawn_wait() {
    let pool = Pool::new(2).unwrap();
    let flag = Arc::new(AtomicBool::new(false));
    let f = Arc::clone(&flag);
    let handle = pool.spawn(move || f.store(true, Ordering::Release));
    handle.signal().wait();
    assert!(flag.load(Ordering::Acquire));
  }

  #[test]
  fn spawn_multiple() {
    let pool = Pool::new(4).unwrap();
    let counter = Arc::new(AtomicUsize::new(0));
    let mut handles = vec![];

    for _ in 0..100 {
      let c = Arc::clone(&counter);
      handles.push(pool.spawn(move || {
        c.fetch_add(1, Ordering::Relaxed);
      }));
    }

    for h in handles {
      h.signal().wait();
    }
    assert_eq!(counter.load(Ordering::Relaxed), 100);
  }

  #[test]
  fn linear_graph() {
    let pool = Pool::new(2).unwrap();
    let order = Arc::new(Mutex::new(Vec::new()));
    let mut graph = TaskGraph::new();
    let (o1, o2, o3) =
      (Arc::clone(&order), Arc::clone(&order), Arc::clone(&order));
    let a = graph.add(move || o1.lock().unpoison().push('A'));
    let b = graph.add(move || o2.lock().unpoison().push('B'));
    let c = graph.add(move || o3.lock().unpoison().push('C'));
    graph.dependency(b, a).unwrap();
    graph.dependency(c, b).unwrap();
    pool.execute(graph).unwrap();
    assert_eq!(*order.lock().unpoison(), vec!['A', 'B', 'C']);
  }

  #[test]
  fn fan_out_fan_in() {
    let pool = Pool::new(4).unwrap();
    let order = Arc::new(Mutex::new(Vec::new()));
    let mut graph = TaskGraph::new();
    let (o1, o2, o3, o4) = (
      Arc::clone(&order),
      Arc::clone(&order),
      Arc::clone(&order),
      Arc::clone(&order),
    );

    let a = graph.add(move || o1.lock().unpoison().push('A'));
    let b = graph.add(move || o2.lock().unpoison().push('B'));
    let c = graph.add(move || o3.lock().unpoison().push('C'));
    let d = graph.add(move || o4.lock().unpoison().push('D'));

    graph.dependency(b, a).unwrap();
    graph.dependency(c, a).unwrap();
    graph.dependency(d, b).unwrap();
    graph.dependency(d, c).unwrap();
    pool.execute(graph).unwrap();
    let result = order.lock().unpoison();

    assert_eq!(result[0], 'A'); // A first
    assert_eq!(result[3], 'D'); // D last
    assert!(result[1..3].contains(&'B')); // B and C in middle, any order
    assert!(result[1..3].contains(&'C'));
  }

  #[test]
  fn scoped_graph_accepts_borrowed_captures() {
    let pool = Pool::new(2).unwrap();
    let input = vec![1, 2, 3, 4];
    let mut observed = 0usize;

    let mut graph = ScopedTaskGraph::new();
    graph.add(|| {
      observed = input.iter().sum();
      Ok(())
    });

    pool.execute_scoped(graph).unwrap();
    assert_eq!(observed, 10);
  }

  #[test]
  fn scoped_graph_respects_dependencies() {
    let pool = Pool::new(2).unwrap();
    let order = Arc::new(Mutex::new(Vec::new()));
    let mut graph = ScopedTaskGraph::new();
    let (o1, o2, o3) =
      (Arc::clone(&order), Arc::clone(&order), Arc::clone(&order));
    let a = graph.add(move || {
      o1.lock().unpoison().push('A');
      Ok(())
    });
    let b = graph.add(move || {
      o2.lock().unpoison().push('B');
      Ok(())
    });
    let c = graph.add(move || {
      o3.lock().unpoison().push('C');
      Ok(())
    });
    graph.dependency(b, a).unwrap();
    graph.dependency(c, b).unwrap();

    pool.execute_scoped(graph).unwrap();
    assert_eq!(*order.lock().unpoison(), vec!['A', 'B', 'C']);
  }

  #[derive(Debug)]
  enum ScopedTestError {
    Failed,
  }

  impl ErrorDomain for ScopedTestError {
    fn domain(&self) -> &str {
      "scoped-test"
    }
  }

  impl std::fmt::Display for ScopedTestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
      write!(f, "scoped graph test error")
    }
  }

  #[test]
  fn scoped_graph_error_skips_dependents_but_runs_independent_tasks() {
    let pool = Pool::new(2).unwrap();
    let failed_child_ran = Arc::new(AtomicBool::new(false));
    let independent_ran = Arc::new(AtomicBool::new(false));

    let mut graph = ScopedTaskGraph::new();
    let root = graph.add(|| Err(AetherError::new(ScopedTestError::Failed)));
    let child_flag = Arc::clone(&failed_child_ran);
    let child = graph.add(move || {
      child_flag.store(true, Ordering::Release);
      Ok(())
    });
    let independent_flag = Arc::clone(&independent_ran);
    graph.add(move || {
      independent_flag.store(true, Ordering::Release);
      Ok(())
    });
    graph.dependency(child, root).unwrap();

    let result = pool.execute_scoped(graph);

    assert!(result.is_err());
    assert!(!failed_child_ran.load(Ordering::Acquire));
    assert!(independent_ran.load(Ordering::Acquire));
  }

  #[test]
  fn scoped_reduction_runs_after_inputs_and_before_consumers() {
    let pool = Pool::new(4).unwrap();
    let reduction = ScopedReduction::new(3);
    let observed = Arc::new(Mutex::new(None));
    let mut graph = ScopedTaskGraph::new();

    let mut inputs = Vec::new();
    for (index, dt) in [0.4, 0.15, 0.25].into_iter().enumerate() {
      let reduction = reduction.clone();
      inputs.push(graph.add(move || {
        reduction.write_input(index, dt)?;
        Ok(())
      }));
    }

    let reducer_handle = reduction.clone();
    let reducer = graph.add(move || {
      reducer_handle.reduce_min()?;
      Ok(())
    });
    for input in inputs {
      graph.dependency(reducer, input).unwrap();
    }

    let consumer_handle = reduction.clone();
    let observed_handle = Arc::clone(&observed);
    let consumer = graph.add(move || {
      *observed_handle.lock().unpoison() = Some(consumer_handle.value()?);
      Ok(())
    });
    graph.dependency(consumer, reducer).unwrap();

    pool.execute_scoped(graph).unwrap();

    assert_eq!(*observed.lock().unpoison(), Some(0.15));
  }

  #[test]
  fn scoped_reduction_missing_input_errors_and_skips_consumers() {
    let pool = Pool::new(2).unwrap();
    let reduction = ScopedReduction::new(2);
    let consumer_ran = Arc::new(AtomicBool::new(false));
    let mut graph = ScopedTaskGraph::new();

    let input_handle = reduction.clone();
    let input = graph.add(move || {
      input_handle.write_input(0, 1.0)?;
      Ok(())
    });

    let reducer_handle = reduction.clone();
    let reducer = graph.add(move || {
      reducer_handle.reduce_min()?;
      Ok(())
    });
    graph.dependency(reducer, input).unwrap();

    let consumer_handle = Arc::clone(&consumer_ran);
    let consumer = graph.add(move || {
      consumer_handle.store(true, Ordering::Release);
      Ok(())
    });
    graph.dependency(consumer, reducer).unwrap();

    let result = pool.execute_scoped(graph);

    assert!(result.is_err());
    assert!(!consumer_ran.load(Ordering::Acquire));
  }

  #[test]
  fn scoped_scheduler_nodes_run_on_caller_thread() {
    let pool = Pool::new(2).unwrap();
    let caller = std::thread::current().id();
    let observed = Arc::new(Mutex::new(None));
    let mut graph = ScopedTaskGraph::new();

    let observed_handle = Arc::clone(&observed);
    graph.add_scheduler(move |_| {
      *observed_handle.lock().unpoison() = Some(std::thread::current().id());
      Ok(())
    });

    pool.execute_scoped(graph).unwrap();

    assert_eq!(*observed.lock().unpoison(), Some(caller));
  }

  #[test]
  fn scoped_scheduler_node_can_run_child_wave() {
    let pool = Pool::new(2).unwrap();
    let counter = Arc::new(AtomicUsize::new(0));
    let mut graph = ScopedTaskGraph::new();

    let counter_handle = Arc::clone(&counter);
    graph.add_scheduler(move |scheduler| {
      let mut wave = ScopedTaskGraph::new();
      for _ in 0..8 {
        let counter_handle = Arc::clone(&counter_handle);
        wave.add(move || {
          counter_handle.fetch_add(1, Ordering::AcqRel);
          Ok(())
        });
      }
      scheduler.run(wave)
    });

    pool.execute_scoped(graph).unwrap();

    assert_eq!(counter.load(Ordering::Acquire), 8);
  }

  #[test]
  fn scoped_scheduler_map_and_reduce_min_cover_common_program_steps() {
    let pool = Pool::new(2).unwrap();
    let reduction = ScopedReduction::new(3);
    let observed = Arc::new(Mutex::new(None));
    let mut graph = ScopedTaskGraph::new();

    let reduction_handle = reduction.clone();
    let observed_handle = Arc::clone(&observed);
    graph.add_scheduler(move |scheduler| {
      scheduler.map(3, |index| {
        reduction_handle.write_input(index, [0.3, 0.1, 0.2][index])
      })?;
      let dt = scheduler.reduce_min(&reduction_handle)?;
      *observed_handle.lock().unpoison() = Some(dt);
      Ok(())
    });

    pool.execute_scoped(graph).unwrap();

    assert_eq!(*observed.lock().unpoison(), Some(0.1));
  }

  #[test]
  fn scoped_graph_cycle_detection() {
    let mut graph = ScopedTaskGraph::new();
    let a = graph.add(|| Ok(()));
    let b = graph.add(|| Ok(()));
    graph.dependency(a, b).unwrap();
    graph.dependency(b, a).unwrap();
    let pool = Pool::new(2).unwrap();
    assert!(pool.execute_scoped(graph).is_err());
  }

  #[test]
  fn cylce_detection() {
    let mut graph = TaskGraph::new();
    let a = graph.add(|| {});
    let b = graph.add(|| {});
    graph.dependency(a, b).unwrap();
    graph.dependency(b, a).unwrap();
    let pool = Pool::new(2).unwrap();
    assert!(pool.execute(graph).is_err());
  }

  #[test]
  fn parallel_for_every_element_processed() {
    let pool = Pool::new(4).unwrap();
    let mut data: Vec<u32> = (0..1000).collect();
    pool.parallel_for(&mut data, 64, |chunk| {
      for x in chunk {
        *x *= 2;
      }
    });
    for (i, val) in data.iter().enumerate() {
      assert_eq!(*val, (i as u32) * 2);
    }
  }

  #[test]
  fn drop_doesnt_hang() {
    let pool = Pool::new(4).unwrap();
    pool.spawn(|| std::thread::sleep(Duration::from_millis(10)));
    drop(pool);
  }

  #[test]
  fn work_stealing() {
    let pool = Pool::new(4).unwrap();
    let counter = Arc::new(AtomicUsize::new(0));
    // Push 100 jobs directly to worker 0
    for _ in 0..100 {
      let c = Arc::clone(&counter);
      pool.context.workers[0].push(Box::new(move || {
        c.fetch_add(1, Ordering::Relaxed);
      }));
    }
    pool.context.global_barrier.notify_all();
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(counter.load(Ordering::Relaxed), 100);
  }

  fn panic_task() {
    panic!("intentional panic for dispatch test");
  }

  fn noop_task() {}

  #[test]
  fn dispatch_panic_propagates_without_deadlock() {
    let pool = Pool::new(2).unwrap();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
      pool.dispatch(vec![panic_task as fn(), noop_task as fn()]);
    }));

    assert!(result.is_err());

    // Pool should remain usable after panic propagation.
    let flag = Arc::new(AtomicBool::new(false));
    let f = Arc::clone(&flag);
    let handle = pool.spawn(move || f.store(true, Ordering::Release));
    handle.signal().wait();
    assert!(flag.load(Ordering::Acquire));
  }

  #[test]
  fn parallel_for_panic_propagates_without_deadlock() {
    let pool = Pool::new(4).unwrap();
    let mut data: Vec<u32> = (0..128).collect();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
      pool.parallel_for(&mut data, 16, |_| {
        panic!("intentional panic for parallel_for test");
      });
    }));

    assert!(result.is_err());

    // Pool should remain usable after panic propagation.
    let counter = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&counter);
    let handle = pool.spawn(move || {
      c.fetch_add(1, Ordering::Relaxed);
    });
    handle.signal().wait();
    assert_eq!(counter.load(Ordering::Relaxed), 1);
  }
}
