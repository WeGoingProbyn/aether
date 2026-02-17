use std::{any::Any, panic::{self, AssertUnwindSafe}, sync::{atomic::{AtomicBool, AtomicUsize, Ordering}, Arc, Condvar, Mutex}, thread::JoinHandle};

use crate::{collections::graph::Graph, error::{AetherResult, Unpoison}, profiler::Profiler, thread::{task::{Job, TaskHandle}, worker::Queue}};

pub struct Context {
  pub(crate) workers: Vec<Arc<Queue>>,
  pub(crate) shutdown: AtomicBool,
  pub(crate) global_barrier: Condvar,
  pub(crate) global_mutex: Mutex<()>,
}

pub struct Pool {
  pub(crate) context: Arc<Context>,
  next_worker: AtomicUsize,
  handles: Vec<JoinHandle<()>>,
}

impl Default for Pool {
  fn default() -> Self {
    let n: usize = unsafe {
      // This will never not be safe as 1usize can never be non zero or negative
      std::thread::available_parallelism()
        .unwrap_or(std::num::NonZero::new_unchecked(1usize))
        .get() 
    };
    Pool::new(n).unwrap()
  }
}

impl Pool {
  pub fn spawn<F>(&self, f: F) -> TaskHandle 
  where 
    F: FnOnce() + Send + 'static,
  {
    let (handle, signal) = TaskHandle::new();
    let job: Job = Box::new(move || {
      f();
      signal.complete();
    });
    self.submit(job);
    handle 
  }

  fn submit(&self, job: Job) {
    let next = self.next_worker.fetch_add(1, Ordering::Relaxed) % self.context.workers.len();

    self.context.workers[next].push(job);
    self.context.global_barrier.notify_one();
  }

  pub fn execute(&self, graph: TaskGraph) -> AetherResult<()> {
    let _ = graph.inner.topological_sort()?;

    let exec: Arc<GraphExecution> = Arc::new(graph.into());
    let ctx = Arc::clone(&self.context);

    // Enqueue roots
    exec.remaining_deps.iter().enumerate().for_each(|(id, dep)| {
      if dep.load(Ordering::Acquire) == 0 {
        enqueue_graph_task(id, &exec, &ctx);
      }
    });

    exec.wait();
    Ok(())
  }

  pub fn parallel_for<T, F>(&self, data: &mut [T], chunk_size: usize, f: F)
  where
    T: Send + 'static,
    F: Fn(&mut [T]) + Send + Sync,
  {
    if data.is_empty() { return; }

    let remaining = Arc::new(AtomicUsize::new(0));
    let done_mutex = Arc::new(Mutex::new(()));
    let done_condvar = Arc::new(Condvar::new());
    let panic_payload = Arc::new(Mutex::new(None::<Box<dyn Any + Send + 'static>>));

    // we block until all jobs complete, so `f` and `data`
    // outlive all submitted jobs. The transmute erases the lifetime
    // bound but the blocking guarantee makes it sound.

    #[allow(clippy::type_complexity)]
    let f: Arc<dyn Fn(&mut [T]) + Send + Sync + 'static> = unsafe {
      std::mem::transmute(Arc::new(f) as Arc<dyn Fn(&mut [T]) + Send + Sync>)
    };

    let chunks: Vec<&mut [T]> = data.chunks_mut(chunk_size).collect();
    remaining.store(chunks.len(), Ordering::Release);

    for chunk in chunks {
      let f = Arc::clone(&f);
      let remaining = Arc::clone(&remaining);
      let done_mutex = Arc::clone(&done_mutex);
      let done_condvar = Arc::clone(&done_condvar);
      let panic_payload = Arc::clone(&panic_payload);

      let ptr = chunk.as_mut_ptr() as usize;
      let len = chunk.len();

      self.submit(Box::new(move || {
        // This will only truly be unsafe is chunks
        // overalp, which we've made sure that they don't
        let chunk = unsafe { std::slice::from_raw_parts_mut(ptr as *mut T, len) };
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
          f(chunk);
        }));

        if let Err(payload) = result {
          record_panic(&panic_payload, payload);
        }

        if remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
          let _guard = done_mutex.lock().unpoison();
          done_condvar.notify_all();
        }
      }));
    }

    let mut guard = done_mutex.lock().unpoison();
    while remaining.load(Ordering::Acquire) > 0 {
      guard = done_condvar.wait(guard).unpoison();
    }

    if let Some(payload) = panic_payload.lock().unpoison().take() {
      panic::resume_unwind(payload);
    }
  }

  pub fn new(n: usize) -> AetherResult<Pool> {
    let mut queues = Vec::with_capacity(n);
    for i in 0..n {
      queues.push(Arc::new(Queue::new(i)));
    }

    let context = Arc::new(Context {
      workers: queues,
      shutdown: AtomicBool::new(false),
      global_barrier: Condvar::new(),
      global_mutex: Mutex::new(()),
    });

    let mut handles = Vec::with_capacity(n);
    for i in 0..n {
      let ctx = Arc::clone(&context);
      let handle = std::thread::Builder::new()
        .name(format!("aether-worker-{}", i))
        .spawn(move || {
          ctx.workers[i].worker_loop(&ctx);
        })?;
      handles.push(handle);
    }

    Ok(Pool {
      context,
      handles,
      next_worker: AtomicUsize::new(0),
    })
  }

  pub fn flush_profiler(&self) {
    let barrier = Arc::new((AtomicUsize::new(self.context.workers.len()), Mutex::new(()), Condvar::new()));

    for worker in &self.context.workers {
      let b = Arc::clone(&barrier);
      worker.push(Box::new(move || {
        Profiler::flush_local();
        if b.0.fetch_sub(1, Ordering::AcqRel) == 1 {
          let _guard = b.1.lock().unpoison();
          b.2.notify_all();
        }
      }));
    }
    self.context.global_barrier.notify_all();

    let mut guard = barrier.1.lock().unpoison();
    while barrier.0.load(Ordering::Acquire) > 0 {
      guard = barrier.2.wait(guard).unpoison();
    }
  }

  pub fn dispatch<F>(&self, tasks: Vec<F>)
  where
    F: FnOnce() + Send,
  {
    if tasks.is_empty() { return; }
    let remaining = Arc::new(AtomicUsize::new(tasks.len()));
    let done_mutex = Arc::new(Mutex::new(()));
    let done_condvar = Arc::new(Condvar::new());
    let panic_payload = Arc::new(Mutex::new(None::<Box<dyn Any + Send + 'static>>));

    for task in tasks {
      let remaining = Arc::clone(&remaining);
      let done_mutex = Arc::clone(&done_mutex);
      let done_condvar = Arc::clone(&done_condvar);
      let panic_payload = Arc::clone(&panic_payload);

      // Safe: we block below, so task's captures outlive the job
      let task: Box<dyn FnOnce() + Send + 'static> = unsafe {
        std::mem::transmute(Box::new(task) as Box<dyn FnOnce() + Send>)
      };
      self.submit(Box::new(move || {
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
          task();
        }));
        if let Err(payload) = result {
          record_panic(&panic_payload, payload);
        }

        if remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
          let _guard = done_mutex.lock().unpoison();
          done_condvar.notify_all();
        }
      }));
    }

    let mut guard = done_mutex.lock().unpoison();
    while remaining.load(Ordering::Acquire) > 0 {
      guard = done_condvar.wait(guard).unpoison();
    }

    if let Some(payload) = panic_payload.lock().unpoison().take() {
      panic::resume_unwind(payload);
    }
  }
}

impl Drop for Pool {
  fn drop(&mut self) {
    self.context.shutdown.store(true, Ordering::Release);
    self.context.global_barrier.notify_all();

    for handle in self.handles.drain(..) {
      let _ = handle.join();
    }
  }
}

fn enqueue_graph_task(id: usize, exec: &Arc<GraphExecution>, ctx: &Arc<Context>) {
  let task = exec.tasks.lock().unpoison()[id].take();
  let exec = Arc::clone(exec);
  let ctx1 = Arc::clone(ctx);

  let job: Job = Box::new(move || {
    if let Some(task) = task {
      task()
    }

    for &dep_id in &exec.dependents[id] {
      if exec.remaining_deps[dep_id].fetch_sub(1, Ordering::AcqRel) == 1 {
        enqueue_graph_task(dep_id, &exec, &ctx1);
      }
    }

    if exec.remaining_total.fetch_sub(1, Ordering::AcqRel) == 1 {
      let _guard = exec.done_mutex.lock().unpoison();
      exec.done_condvar.notify_all();
    }
  });

  // Submit to a worker queue
  let worker_idx = id % ctx.workers.len();
  ctx.workers[worker_idx].push(job);
  ctx.global_barrier.notify_one();
}

pub struct TaskGraph {
  inner: Graph<TaskNode>,
}

impl Default for TaskGraph {
  fn default() -> Self {
    TaskGraph { 
      inner: 
      Graph::new()
    }
  }
}

impl TaskGraph {
  pub fn new() -> TaskGraph {
    TaskGraph::default()
  }

  pub fn add<F>(&mut self, f: F) -> usize
where
    F: FnOnce() + Send + 'static,
  {
    self.inner.add_node(TaskNode {
      //name,
      task: Some(Box::new(f)),
    })
  }

  pub fn dependency(&mut self, task: usize, depends_on: usize) -> AetherResult<()> {
    self.inner.add_edge(depends_on, task) // edge goes dep -> task
  }
}

pub(crate) struct TaskNode {
  //name: &'static str,
  task: Option<Job>,
}

struct GraphExecution {
  tasks: Mutex<Vec<Option<Job>>>,
  //names: Vec<&'static str>,             // for profiler spans
  dependents: Vec<Vec<usize>>,          // who to notify on completion
  remaining_deps: Vec<AtomicUsize>,     // per-task dep counter
  remaining_total: AtomicUsize,         // how many tasks left
  done_mutex: Mutex<()>,
  done_condvar: Condvar,
}

impl From<TaskGraph> for GraphExecution {
  fn from(graph: TaskGraph) -> Self {
    let count = graph.inner.next_node_id();
    //let mut names = Vec::with_capacity(count);
    let mut dependents = vec![Vec::new(); count];
    let mut remaining_deps = Vec::with_capacity(count);

    dependents.iter_mut().enumerate().for_each(|(id, dep)| {
      //let node = graph.inner.get_node(id).unwrap();
      //names.push(node.data.name);
      remaining_deps.push(
        AtomicUsize::new(
          graph.inner.incoming_edges(id).len()
      ));

      if let Some(edges) = graph.inner.outgoing_edges(id) {
        for edge in edges {
          dep.push(edge.target);
        }
      }
    });

    let mut tasks: Vec<Option<Job>> = (0..count).map(|_| None).collect();
    for node in graph.inner.into_nodes() {
      tasks[node.id] = node.data.task;
    }

    GraphExecution {
      tasks: Mutex::new(tasks),
      //names,
      dependents,
      remaining_deps,
      remaining_total: AtomicUsize::new(count),
      done_mutex: Mutex::new(()),
      done_condvar: Condvar::new(),
    }
  }
}

impl GraphExecution {
  fn wait(&self) {
    let mut guard = self.done_mutex.lock().unpoison();
    while self.remaining_total.load(Ordering::Acquire) > 0 {
      guard = self.done_condvar.wait(guard).unpoison();
    }
  }
}

fn record_panic(
  slot: &Arc<Mutex<Option<Box<dyn Any + Send + 'static>>>>,
  payload: Box<dyn Any + Send + 'static>,
) {
  let mut first = slot.lock().unpoison();
  if first.is_none() {
    *first = Some(payload);
  }
}


