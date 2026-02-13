use crate::error::ErrorDomain;

pub mod task;
pub mod pool;
pub mod worker;

pub enum ErrorKind {
  ThreadPoolShutdown,
  ThreadPoolPanic,
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
      ErrorKind::ThreadPoolPanic => "a task inside the threading pool has panicked",
      ErrorKind::ThreadPoolShutdown => "a task has been submitted to a dropped pool",
    };

    write!(f, "{}", string)?;
    Ok(())
  }
}

#[cfg(test)]
mod test {
  use utility::error::Unpoison;
  use utility::thread::pool::{Pool, TaskGraph};
  use std::sync::{atomic::{AtomicBool, AtomicUsize, Ordering}, Arc, Mutex};
  use std::time::Duration;

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
      handles.push(pool.spawn(move || { c.fetch_add(1, Ordering::Relaxed); }));
    }

    for h in handles { h.signal().wait(); }
    assert_eq!(counter.load(Ordering::Relaxed), 100);
  }

  #[test]
  fn linear_graph() {
    let pool = Pool::new(2).unwrap();
    let order = Arc::new(Mutex::new(Vec::new()));
    let mut graph = TaskGraph::new();
    let (o1, o2, o3) = (Arc::clone(&order), Arc::clone(&order), Arc::clone(&order));
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
      Arc::clone(&order)
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

    assert_eq!(result[0], 'A');          // A first
    assert_eq!(result[3], 'D');          // D last
    assert!(result[1..3].contains(&'B')); // B and C in middle, any order
    assert!(result[1..3].contains(&'C'));
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
      for x in chunk { *x *= 2; }
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
}
