// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::{
  sync::atomic::{AtomicU8, Ordering},
  sync::{Mutex, OnceLock},
  time::{SystemTime, UNIX_EPOCH},
};

use crate::error::Unpoison;

static LOGGER: OnceLock<Logger> = OnceLock::new();
static MAX_LEVEL: AtomicU8 = AtomicU8::new(Level::Info as u8);

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub enum Level {
  Trace = 0,
  Debug,
  Info,
  Warn,
  Error,
  Fatal,
}

impl std::fmt::Display for Level {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let string = match self {
      Level::Trace => "\x1b[2mTrace",
      Level::Debug => "\x1b[36mDebug",
      Level::Info => "\x1b[32mInfo ",
      Level::Warn => "\x1b[33mWarn ",
      Level::Error => "\x1b[31mError",
      Level::Fatal => "\x1b[1;31mFatal",
    };

    write!(f, "{}\x1b[0m", string)?;
    Ok(())
  }
}

impl Level {
  fn display_len(&self) -> usize {
    // length without ANSI colour codes
    match self {
      Level::Trace => "Trace".len(),
      Level::Debug => "Debug".len(),
      Level::Info => "Info ".len(),
      Level::Warn => "Warn ".len(),
      Level::Error => "Error".len(),
      Level::Fatal => "Fatal".len(),
    }
  }
}

#[derive(Clone)]
pub enum Value {
  I64(i64),
  U64(u64),
  F64(f64),
  Bool(bool),
  Str(String),
}

impl From<i64> for Value {
  fn from(v: i64) -> Self {
    Value::I64(v)
  }
}

impl From<u64> for Value {
  fn from(v: u64) -> Self {
    Value::U64(v)
  }
}

impl From<f64> for Value {
  fn from(v: f64) -> Self {
    Value::F64(v)
  }
}

impl From<bool> for Value {
  fn from(v: bool) -> Self {
    Value::Bool(v)
  }
}

impl From<String> for Value {
  fn from(v: String) -> Self {
    Value::Str(v)
  }
}

impl From<&str> for Value {
  fn from(v: &str) -> Self {
    Value::Str(v.to_string())
  }
}

impl std::fmt::Display for Value {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Value::Str(v) => write!(f, "{}", v)?,
      Value::I64(v) => write!(f, "{}", v)?,
      Value::U64(v) => write!(f, "{}", v)?,
      Value::F64(v) => write!(f, "{}", v)?,
      Value::Bool(v) => write!(f, "{}", v)?,
    }

    Ok(())
  }
}

#[derive(Clone)]
pub struct Field {
  pub key: &'static str,
  pub value: Value,
}

#[derive(Clone)]
pub struct Record {
  pub ts: SystemTime,
  pub meta: MetaData,
  pub message: String,
  pub fields: Vec<Field>,
}

impl std::fmt::Display for Record {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let prefix1 = format!(
      "[{}] [{}] ",
      Logger::iso_timestamp(self.ts),
      std::thread::current().name().unwrap_or("main")
    );
    let mut prefix2 = format!("{}:{}", self.meta.target, self.meta.line);
    let pad = " ".repeat(
      prefix1.len() + prefix2.len() + self.meta.level.display_len() + 1,
    );
    prefix2.insert_str(0, &format!("{} ", self.meta.level));
    prefix2.insert_str(0, &prefix1);
    let lines: Vec<&str> = self.message.lines().collect();
    let last = lines.len().saturating_sub(1);

    for (i, line) in lines.iter().enumerate() {
      let arrow = if last == 0 {
        "──>"
      } else if i == 0 {
        "┬─>"
      } else if i < last {
        "├─>"
      } else {
        "└─>"
      };

      if i == 0 {
        writeln!(f, "{} {} {}", prefix2, arrow, line)?;
      } else {
        writeln!(f, "{} {} {}", pad, arrow, line)?;
      }
    }
    Ok(())
  }
}

#[derive(Clone)]
pub struct MetaData {
  pub line: u32,
  pub level: Level,
  pub file: &'static str,
  pub target: &'static str,
}

#[derive(Default)]
pub struct Logger {
  sinks: Vec<Box<dyn Sink>>,
}

impl Logger {
  pub fn push(&self, record: Record) {
    let last = self.sinks.len() - 1;
    for (i, sink) in self.sinks.iter().enumerate() {
      if i == last {
        sink.write(record);
        return;
      }
      sink.write(record.clone());
    }
  }

  pub fn flush(&self) {
    self.sinks.iter().for_each(|s| s.flush());
  }

  pub fn init(sinks: Vec<Box<dyn Sink>>, max_level: Level) {
    let _ = LOGGER.set(Logger { sinks });

    MAX_LEVEL.store(max_level as u8, Ordering::Relaxed);
  }

  pub fn enabled(level: Level) -> bool {
    MAX_LEVEL.load(Ordering::Relaxed) <= (level as u8)
  }

  pub fn submit(record: Record) {
    if let Some(logger) = LOGGER.get() {
      logger.push(record);
    }
  }

  fn iso_timestamp(ts: SystemTime) -> String {
    let dur = ts.duration_since(UNIX_EPOCH).unwrap();
    let secs = dur.as_secs();

    let days = secs / 86400;
    let time = secs % 86400;
    let h = time / 3600;
    let m = (time % 3600) / 60;
    let s = time % 60;

    // days since 1970-01-01 → date (civil calendar)
    let (y, mo, d) = Logger::days_to_date(days);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
  }

  fn days_to_date(mut days: u64) -> (u64, u64, u64) {
    let mut y = 1970;
    loop {
      let year_days = if Logger::is_leap(y) { 366 } else { 365 };
      if days < year_days {
        break;
      }
      days -= year_days;
      y += 1;
    }

    let month_days = if Logger::is_leap(y) {
      [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
      [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut m = 0;
    for &md in &month_days {
      if days < md {
        break;
      }
      days -= md;
      m += 1;
    }

    (y, m + 1, days + 1)
  }

  fn is_leap(y: u64) -> bool {
    (y.is_multiple_of(4) && !y.is_multiple_of(100)) || y.is_multiple_of(400)
  }
}

pub trait Sink: Send + Sync {
  fn write(&self, record: Record);
  fn flush(&self);
}

pub struct StdSink<T>
where
  T: std::io::Write + Send + Sync,
{
  writer: Mutex<T>,
  buffer: Mutex<Vec<Record>>,
  buffer_capacity: usize,
}

impl<T> StdSink<T>
where
  T: std::io::Write + Send + Sync,
{
  pub fn new(sink: T) -> StdSink<T> {
    StdSink {
      writer: Mutex::new(sink),
      buffer: Mutex::new(vec![]),
      buffer_capacity: 8,
    }
  }

  pub fn capacity(self, cap: usize) -> StdSink<T> {
    StdSink {
      writer: self.writer,
      buffer: self.buffer,
      buffer_capacity: cap,
    }
  }
}

impl<T> Sink for StdSink<T>
where
  T: std::io::Write + Send + Sync,
{
  fn write(&self, record: Record) {
    let immediate = record.meta.level >= Level::Error;
    let len = {
      let mut buffer = self.buffer.lock().unpoison();
      buffer.push(record);
      buffer.len()
    };

    if immediate || len >= self.buffer_capacity {
      self.flush();
    }
  }

  fn flush(&self) {
    let mut buffer = self.buffer.lock().unpoison();
    for record in buffer.drain(..) {
      write!(self.writer.lock().unpoison(), "{}", record).unwrap();
    }
  }
}

pub struct LogWriter {
  level: Level,
}

impl LogWriter {
  pub fn new(level: Level) -> LogWriter {
    LogWriter { level }
  }
}

impl std::io::Write for LogWriter {
  fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
    let msg = std::str::from_utf8(buf).unwrap_or("<invalid utf8>");
    utility::log!(self.level, "{}", msg);
    Ok(buf.len())
  }

  fn flush(&mut self) -> std::io::Result<()> {
    Ok(())
  }
}

#[macro_export]
macro_rules! log {
($lvl:expr, $fmt:literal $(, $arg:expr)* $(; $k:ident = $v:expr)* $(,)?) => {{
    if $crate::logger::Logger::enabled($lvl) {
      let fields = Vec::new();
      $(
        fields.push($crate::logger::Field {
          key: stringify!($k),
          value: $crate::logger::Value::from($v),
        });
      )*

      $crate::logger::Logger::submit($crate::logger::Record {
        ts: std::time::SystemTime::now(),
        meta: $crate::logger::MetaData {
          level: $lvl,
          target: module_path!(),
          file: file!(),
          line: line!(),
        },

        message: format!($fmt $(, $arg)*),
        fields,
      });
    }

    if $lvl == $crate::logger::Level::Fatal { panic!() }
  }};
}

#[macro_export]
macro_rules! info { ($($t:tt)*) => { $crate::log!($crate::logger::Level::Info,  $($t)*) }; }
#[macro_export]
macro_rules! warn { ($($t:tt)*) => { $crate::log!($crate::logger::Level::Warn,  $($t)*) }; }
#[macro_export]
macro_rules! error { ($($t:tt)*) => { $crate::log!($crate::logger::Level::Error, $($t)*) }; }
#[macro_export]
macro_rules! debug { ($($t:tt)*) => { $crate::log!($crate::logger::Level::Debug, $($t)*) }; }
#[macro_export]
macro_rules! trace { ($($t:tt)*) => { $crate::log!($crate::logger::Level::Trace, $($t)*) }; }
#[macro_export]
macro_rules! fatal { ($($t:tt)*) => { $crate::log!($crate::logger::Level::Fatal, $($t)*) }; }
