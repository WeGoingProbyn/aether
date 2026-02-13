use std::sync::{Condvar, Mutex, Arc};

pub type Job = Box<dyn FnOnce() + Send>;

#[derive(Clone)]
pub struct TaskHandle {
  inner: Arc<TaskSignal>,
}

impl TaskHandle {
  pub(crate) fn new() -> (TaskHandle, Arc<TaskSignal>) {
    let inner = Arc::new(TaskSignal::new());
    let handle = TaskHandle {
      inner: inner.clone(),
    };

    (handle, inner)
  }

  pub fn signal(&self) -> Arc<TaskSignal> {
    self.inner.clone()
  }
}

pub struct TaskSignal {
  done: Mutex<bool>,
  barrier: Condvar,
}

impl TaskSignal {
  pub(crate) fn new() -> TaskSignal {
    TaskSignal {
      done: Mutex::new(false),
      barrier: Condvar::new(),
    }
  }

  pub fn wait(&self) {
    let mut done = self.done.lock().unwrap();
    while !*done {
      done = self.barrier.wait(done).unwrap();
    }
  }

  pub(crate) fn complete(&self) {
    let mut done = self.done.lock().unwrap();
    *done = true;
    self.barrier.notify_all();
  }
}

