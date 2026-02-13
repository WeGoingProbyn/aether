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
