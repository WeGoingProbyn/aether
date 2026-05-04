// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::{
  any::Any,
  collections::VecDeque,
  panic::{self, AssertUnwindSafe},
  sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
  },
  thread::JoinHandle,
};

use crate::{
  collections::graph::Graph,
  error::{AetherError, AetherResult, Unpoison},
  profiler::Profiler,
  thread::{
    ErrorKind,
    task::{Job, TaskHandle},
    worker::Queue,
  },
};

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
    let next = self.next_worker.fetch_add(1, Ordering::Relaxed)
      % self.context.workers.len();

    self.context.workers[next].push(job);
    self.context.global_barrier.notify_one();
  }

  pub fn execute(&self, graph: TaskGraph) -> AetherResult<()> {
    let _ = graph.inner.topological_sort()?;

    let exec: Arc<GraphExecution> = Arc::new(graph.into());
    let ctx = Arc::clone(&self.context);

    // Enqueue roots
    exec
      .remaining_deps
      .iter()
      .enumerate()
      .for_each(|(id, dep)| {
        if dep.load(Ordering::Acquire) == 0 {
          enqueue_graph_task(id, &exec, &ctx);
        }
      });

    exec.wait();
    Ok(())
  }

  pub fn execute_scoped<'a>(
    &self,
    graph: ScopedTaskGraph<'a>,
  ) -> AetherResult<()> {
    execute_scoped_graph(graph, self.context.workers.len().max(1))
  }

  pub fn parallel_for<T, F>(&self, data: &mut [T], chunk_size: usize, f: F)
  where
    T: Send + 'static,
    F: Fn(&mut [T]) + Send + Sync,
  {
    if data.is_empty() {
      return;
    }

    let remaining = Arc::new(AtomicUsize::new(0));
    let done_mutex = Arc::new(Mutex::new(()));
    let done_condvar = Arc::new(Condvar::new());
    let panic_payload =
      Arc::new(Mutex::new(None::<Box<dyn Any + Send + 'static>>));

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
        let chunk =
          unsafe { std::slice::from_raw_parts_mut(ptr as *mut T, len) };
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
    let barrier = Arc::new((
      AtomicUsize::new(self.context.workers.len()),
      Mutex::new(()),
      Condvar::new(),
    ));

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
    if tasks.is_empty() {
      return;
    }
    let remaining = Arc::new(AtomicUsize::new(tasks.len()));
    let done_mutex = Arc::new(Mutex::new(()));
    let done_condvar = Arc::new(Condvar::new());
    let panic_payload =
      Arc::new(Mutex::new(None::<Box<dyn Any + Send + 'static>>));

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

fn execute_scoped_graph<'a>(
  graph: ScopedTaskGraph<'a>,
  worker_count: usize,
) -> AetherResult<()> {
  let _ = graph.inner.topological_sort()?;

  let exec = ScopedGraphExecution::from(graph);
  if exec.remaining_total == 0 {
    return Ok(());
  }

  let shared = (Mutex::new(exec), Condvar::new());
  let panic_payload = Mutex::new(None::<Box<dyn Any + Send + 'static>>);

  std::thread::scope(|scope| {
    for _ in 0..worker_count {
      scope.spawn(|| {
        scoped_worker_loop(&shared, &panic_payload);
      });
    }
    let mut scheduler = ScopedScheduler::new(worker_count);
    scoped_scheduler_loop(&shared, &panic_payload, &mut scheduler);
  });

  if let Some(payload) = panic_payload.lock().unpoison().take() {
    panic::resume_unwind(payload);
  }

  shared
    .0
    .lock()
    .unpoison()
    .first_error
    .take()
    .map_or(Ok(()), Err)
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

fn enqueue_graph_task(
  id: usize,
  exec: &Arc<GraphExecution>,
  ctx: &Arc<Context>,
) {
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

pub struct ScopedTaskGraph<'a> {
  inner: Graph<ScopedTaskNode<'a>>,
}

pub struct ScopedScheduler {
  worker_count: usize,
}

impl ScopedScheduler {
  fn new(worker_count: usize) -> Self {
    Self { worker_count }
  }

  pub fn run<'a>(&mut self, graph: ScopedTaskGraph<'a>) -> AetherResult<()> {
    execute_scoped_graph(graph, self.worker_count)
  }

  pub fn map<'a, F>(&mut self, count: usize, f: F) -> AetherResult<()>
  where
    F: Fn(usize) -> AetherResult<()> + Send + Sync + 'a,
  {
    let mut graph = ScopedTaskGraph::new();
    let f = &f;
    for index in 0..count {
      graph.add(move || f(index));
    }
    self.run(graph)
  }

  pub fn reduce_min(
    &mut self,
    reduction: &ScopedReduction<f64>,
  ) -> AetherResult<f64> {
    reduction.reduce_min()
  }
}

pub struct ScopedReduction<T> {
  inner: Arc<ScopedReductionInner<T>>,
}

struct ScopedReductionInner<T> {
  inputs: Vec<Mutex<Option<T>>>,
  value: Mutex<Option<T>>,
}

impl<T> Clone for ScopedReduction<T> {
  fn clone(&self) -> Self {
    Self {
      inner: Arc::clone(&self.inner),
    }
  }
}

impl<T> ScopedReduction<T>
where
  T: Send,
{
  pub fn new(input_count: usize) -> Self {
    ScopedReduction {
      inner: Arc::new(ScopedReductionInner {
        inputs: (0..input_count).map(|_| Mutex::new(None)).collect(),
        value: Mutex::new(None),
      }),
    }
  }

  pub fn input_count(&self) -> usize {
    self.inner.inputs.len()
  }

  pub fn write_input(&self, index: usize, value: T) -> AetherResult<()> {
    let Some(input) = self.inner.inputs.get(index) else {
      return Err(
        AetherError::new(ErrorKind::ReductionInputOutOfRange).context(format!(
          "reduction input {} is out of range for {} inputs",
          index,
          self.input_count()
        )),
      );
    };

    let mut guard = input.lock().unpoison();
    debug_assert!(
      guard.is_none(),
      "scheduler reduction input slot was written more than once",
    );
    *guard = Some(value);
    Ok(())
  }

  pub fn reduce<F>(&self, mut reduce: F) -> AetherResult<T>
  where
    T: Clone,
    F: FnMut(T, T) -> T,
  {
    if self.inner.inputs.is_empty() {
      return Err(AetherError::new(ErrorKind::ReductionEmpty));
    }

    let mut values = Vec::with_capacity(self.inner.inputs.len());
    for (index, input) in self.inner.inputs.iter().enumerate() {
      let Some(value) = input.lock().unpoison().take() else {
        return Err(
          AetherError::new(ErrorKind::ReductionMissingInput)
            .context(format!("reduction input {} was not written", index)),
        );
      };
      values.push(value);
    }

    let mut values = values.into_iter();
    let mut accumulated = values.next().unwrap();
    for value in values {
      accumulated = reduce(accumulated, value);
    }

    *self.inner.value.lock().unpoison() = Some(accumulated.clone());
    Ok(accumulated)
  }

  pub fn value(&self) -> AetherResult<T>
  where
    T: Clone,
  {
    self
      .inner
      .value
      .lock()
      .unpoison()
      .clone()
      .ok_or_else(|| AetherError::new(ErrorKind::ReductionNotReady))
  }

  pub fn clear(&self) {
    for input in &self.inner.inputs {
      *input.lock().unpoison() = None;
    }
    *self.inner.value.lock().unpoison() = None;
  }
}

impl ScopedReduction<f64> {
  pub fn reduce_min(&self) -> AetherResult<f64> {
    self.reduce(f64::min)
  }
}

impl Default for ScopedTaskGraph<'_> {
  fn default() -> Self {
    ScopedTaskGraph {
      inner: Graph::new(),
    }
  }
}

impl<'a> ScopedTaskGraph<'a> {
  pub fn new() -> ScopedTaskGraph<'a> {
    ScopedTaskGraph::default()
  }

  pub fn add<F>(&mut self, f: F) -> usize
  where
    F: FnOnce() -> AetherResult<()> + Send + 'a,
  {
    self.inner.add_node(ScopedTaskNode {
      task: ScopedTaskEntry::Worker(Some(Box::new(f))),
    })
  }

  pub fn add_scheduler<F>(&mut self, f: F) -> usize
  where
    F: FnOnce(&mut ScopedScheduler) -> AetherResult<()> + Send + 'a,
  {
    self.inner.add_node(ScopedTaskNode {
      task: ScopedTaskEntry::Scheduler(Some(Box::new(f))),
    })
  }

  pub fn dependency(
    &mut self,
    task: usize,
    depends_on: usize,
  ) -> AetherResult<()> {
    self.inner.add_edge(depends_on, task)
  }
}

type ScopedJob<'a> = Box<dyn FnOnce() -> AetherResult<()> + Send + 'a>;
type ScopedSchedulerJob<'a> =
  Box<dyn FnOnce(&mut ScopedScheduler) -> AetherResult<()> + Send + 'a>;

enum ScopedTaskEntry<'a> {
  Worker(Option<ScopedJob<'a>>),
  Scheduler(Option<ScopedSchedulerJob<'a>>),
}

impl ScopedTaskEntry<'_> {
  fn is_scheduler(&self) -> bool {
    matches!(self, ScopedTaskEntry::Scheduler(_))
  }
}

struct ScopedTaskNode<'a> {
  task: ScopedTaskEntry<'a>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ScopedNodeState {
  Pending,
  Queued,
  Running,
  Complete,
  Failed,
  Skipped,
}

struct ScopedGraphExecution<'a> {
  tasks: Vec<ScopedTaskEntry<'a>>,
  dependents: Vec<Vec<usize>>,
  remaining_deps: Vec<usize>,
  state: Vec<ScopedNodeState>,
  ready_workers: VecDeque<usize>,
  ready_scheduler: VecDeque<usize>,
  remaining_total: usize,
  first_error: Option<AetherError>,
}

impl<'a> From<ScopedTaskGraph<'a>> for ScopedGraphExecution<'a> {
  fn from(graph: ScopedTaskGraph<'a>) -> Self {
    let count = graph.inner.next_node_id();
    let mut dependents = vec![Vec::new(); count];
    let mut remaining_deps = Vec::with_capacity(count);
    let mut state = vec![ScopedNodeState::Pending; count];
    let mut ready_workers = VecDeque::new();
    let mut ready_scheduler = VecDeque::new();
    let mut roots = Vec::new();

    dependents.iter_mut().enumerate().for_each(|(id, dep)| {
      let deps = graph.inner.incoming_edges(id).len();
      remaining_deps.push(deps);
      if deps == 0 {
        state[id] = ScopedNodeState::Queued;
        roots.push(id);
      }

      if let Some(edges) = graph.inner.outgoing_edges(id) {
        for edge in edges {
          dep.push(edge.target);
        }
      }
    });

    let mut tasks: Vec<ScopedTaskEntry<'a>> =
      (0..count).map(|_| ScopedTaskEntry::Worker(None)).collect();
    for node in graph.inner.into_nodes() {
      tasks[node.id] = node.data.task;
    }

    for id in roots {
      if tasks[id].is_scheduler() {
        ready_scheduler.push_back(id);
      } else {
        ready_workers.push_back(id);
      }
    }

    ScopedGraphExecution {
      tasks,
      dependents,
      remaining_deps,
      state,
      ready_workers,
      ready_scheduler,
      remaining_total: count,
      first_error: None,
    }
  }
}

fn scoped_worker_loop<'a>(
  shared: &(Mutex<ScopedGraphExecution<'a>>, Condvar),
  panic_payload: &Mutex<Option<Box<dyn Any + Send + 'static>>>,
) {
  loop {
    let (id, task) = {
      let mut exec = shared.0.lock().unpoison();
      'next_task: loop {
        while let Some(id) = exec.ready_workers.pop_front() {
          if exec.state[id] == ScopedNodeState::Queued {
            exec.state[id] = ScopedNodeState::Running;
            let task = match &mut exec.tasks[id] {
              ScopedTaskEntry::Worker(task) => task.take(),
              ScopedTaskEntry::Scheduler(_) => unreachable!(),
            };
            break 'next_task (id, task);
          }
        }

        if exec.remaining_total == 0 {
          drop(exec);
          Profiler::flush_local();
          return;
        }

        exec = shared.1.wait(exec).unpoison();
      }
    };

    let result = match task {
      Some(task) => panic::catch_unwind(AssertUnwindSafe(task)),
      None => Ok(Ok(())),
    };

    let mut exec = shared.0.lock().unpoison();
    match result {
      Ok(Ok(())) => complete_scoped_node(&mut exec, id),
      Ok(Err(error)) => fail_scoped_node(&mut exec, id, error),
      Err(payload) => {
        record_scoped_panic(panic_payload, payload);
        skip_scoped_node_dependents(&mut exec, id);
        exec.state[id] = ScopedNodeState::Failed;
        exec.remaining_total -= 1;
      }
    }
    shared.1.notify_all();
  }
}

fn scoped_scheduler_loop<'a>(
  shared: &(Mutex<ScopedGraphExecution<'a>>, Condvar),
  panic_payload: &Mutex<Option<Box<dyn Any + Send + 'static>>>,
  scheduler: &mut ScopedScheduler,
) {
  loop {
    let (id, task) = {
      let mut exec = shared.0.lock().unpoison();
      'next_task: loop {
        while let Some(id) = exec.ready_scheduler.pop_front() {
          if exec.state[id] == ScopedNodeState::Queued {
            exec.state[id] = ScopedNodeState::Running;
            let task = match &mut exec.tasks[id] {
              ScopedTaskEntry::Scheduler(task) => task.take(),
              ScopedTaskEntry::Worker(_) => unreachable!(),
            };
            break 'next_task (id, task);
          }
        }

        if exec.remaining_total == 0 {
          return;
        }

        exec = shared.1.wait(exec).unpoison();
      }
    };

    let result = match task {
      Some(task) => panic::catch_unwind(AssertUnwindSafe(|| task(scheduler))),
      None => Ok(Ok(())),
    };

    let mut exec = shared.0.lock().unpoison();
    match result {
      Ok(Ok(())) => complete_scoped_node(&mut exec, id),
      Ok(Err(error)) => fail_scoped_node(&mut exec, id, error),
      Err(payload) => {
        record_scoped_panic(panic_payload, payload);
        skip_scoped_node_dependents(&mut exec, id);
        exec.state[id] = ScopedNodeState::Failed;
        exec.remaining_total -= 1;
      }
    }
    shared.1.notify_all();
  }
}

fn complete_scoped_node(exec: &mut ScopedGraphExecution<'_>, id: usize) {
  exec.state[id] = ScopedNodeState::Complete;
  exec.remaining_total -= 1;

  for dep_id in exec.dependents[id].clone() {
    if exec.state[dep_id] != ScopedNodeState::Pending {
      continue;
    }
    exec.remaining_deps[dep_id] -= 1;
    if exec.remaining_deps[dep_id] == 0 {
      exec.state[dep_id] = ScopedNodeState::Queued;
      if exec.tasks[dep_id].is_scheduler() {
        exec.ready_scheduler.push_back(dep_id);
      } else {
        exec.ready_workers.push_back(dep_id);
      }
    }
  }
}

fn fail_scoped_node(
  exec: &mut ScopedGraphExecution<'_>,
  id: usize,
  error: AetherError,
) {
  if exec.first_error.is_none() {
    exec.first_error = Some(error);
  }
  skip_scoped_node_dependents(exec, id);
  exec.state[id] = ScopedNodeState::Failed;
  exec.remaining_total -= 1;
}

fn skip_scoped_node_dependents(exec: &mut ScopedGraphExecution<'_>, id: usize) {
  for dep_id in exec.dependents[id].clone() {
    skip_scoped_node(exec, dep_id);
  }
}

fn skip_scoped_node(exec: &mut ScopedGraphExecution<'_>, id: usize) {
  match exec.state[id] {
    ScopedNodeState::Pending | ScopedNodeState::Queued => {
      exec.state[id] = ScopedNodeState::Skipped;
      exec.remaining_total -= 1;
      skip_scoped_node_dependents(exec, id);
    }
    ScopedNodeState::Running
    | ScopedNodeState::Complete
    | ScopedNodeState::Failed
    | ScopedNodeState::Skipped => {}
  }
}

fn record_scoped_panic(
  slot: &Mutex<Option<Box<dyn Any + Send + 'static>>>,
  payload: Box<dyn Any + Send + 'static>,
) {
  let mut first = slot.lock().unpoison();
  if first.is_none() {
    *first = Some(payload);
  }
}

impl Default for TaskGraph {
  fn default() -> Self {
    TaskGraph {
      inner: Graph::new(),
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

  pub fn dependency(
    &mut self,
    task: usize,
    depends_on: usize,
  ) -> AetherResult<()> {
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
  dependents: Vec<Vec<usize>>, // who to notify on completion
  remaining_deps: Vec<AtomicUsize>, // per-task dep counter
  remaining_total: AtomicUsize, // how many tasks left
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
      remaining_deps
        .push(AtomicUsize::new(graph.inner.incoming_edges(id).len()));

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
