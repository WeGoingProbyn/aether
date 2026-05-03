// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::{
  cell::RefCell,
  collections::HashMap,
  sync::{
    Mutex, OnceLock,
    atomic::{AtomicBool, Ordering},
  },
  time::Instant,
};

use crate::debug;

static PROFILER: OnceLock<Profiler> = OnceLock::new();
static ENABLED: AtomicBool = AtomicBool::new(true);

struct SpanStats {
  total_us: u64,
  count: u64,
}

struct OpenSpan {
  name: &'static str,
  start: Instant,
}

struct ThreadState {
  stack: Vec<OpenSpan>,
  stats: HashMap<&'static str, SpanStats>,
}

thread_local! {
  static STATE: RefCell<ThreadState> = RefCell::new(
    ThreadState {
      stack: Vec::with_capacity(64),
      stats: HashMap::new(),
    })
}

pub struct Profiler {
  stats: Mutex<HashMap<&'static str, SpanStats>>,
}

pub struct SpanGuard {
  name: &'static str,
}

impl SpanGuard {
  pub fn new(name: &'static str, _category: &'static str) -> SpanGuard {
    Profiler::start_span(name);
    SpanGuard { name }
  }
}

impl Drop for SpanGuard {
  fn drop(&mut self) {
    Profiler::end_span(self.name);
  }
}

impl Profiler {
  pub fn init() {
    let _ = PROFILER.set(Profiler {
      stats: Mutex::new(HashMap::new()),
    });
  }

  pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
  }

  pub fn start_span(name: &'static str) {
    if Profiler::enabled() {
      STATE.with(|s| {
        let mut borrow = s.borrow_mut();
        borrow.stack.push(OpenSpan {
          name,
          start: Instant::now(),
        });
      })
    }
  }

  pub fn end_span(name: &'static str) {
    if Profiler::enabled() {
      let end = Instant::now();
      let mismatch = STATE.with(|s| {
        let mut borrow = s.borrow_mut();
        let Some(span) = borrow.stack.pop() else {
          return Some((name, "<empty>"));
        };
        let mismatch = (span.name != name).then_some((name, span.name));
        let duration = end.duration_since(span.start).as_micros() as u64;
        let entry = borrow.stats.entry(span.name).or_insert(SpanStats {
          total_us: 0,
          count: 0,
        });
        entry.total_us += duration;
        entry.count += 1;
        mismatch
      });

      if let Some((expected, actual)) = mismatch {
        debug!(
          "profiler span mismatch: ended {}, but top span was {}",
          expected, actual
        );
      }
    }
  }

  pub fn print(writer: &mut impl std::io::Write) {
    let profiler = PROFILER.get().unwrap();
    Profiler::flush(writer);
    debug!("{}", profiler);
  }

  pub fn flush(_writer: &mut impl std::io::Write) {
    Profiler::flush_local();
  }

  pub fn flush_local() {
    let profiler = match PROFILER.get() {
      Some(p) => p,
      None => return,
    };
    STATE.with(|s| {
      let mut borrow = s.borrow_mut();
      let mut global = profiler.stats.lock().unwrap();

      for (name, local) in borrow.stats.drain() {
        let entry = global.entry(name).or_insert(SpanStats {
          total_us: 0,
          count: 0,
        });
        entry.total_us += local.total_us;
        entry.count += local.count;
      }
    });
  }
}

impl std::fmt::Display for Profiler {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let stats = self.stats.lock().unwrap();

    let mut agg: Vec<(&str, u64, u64)> = stats
      .iter()
      .map(|(&name, s)| (name, s.total_us, s.count))
      .collect();

    agg.sort_by(|a, b| (b.1 / b.2).cmp(&(a.1 / a.2)));

    let max_name = agg.iter().map(|(n, _, _)| n.len()).max().unwrap_or(0);
    let max_avg = agg.iter().map(|(_, d, c)| d / c).max().unwrap_or(1);
    let bar_width: u64 = 32;

    writeln!(f, "── Profiler Diagnostics ──")?;
    for (name, total, count) in &agg {
      let avg = total / count;
      let filled = if max_avg > 0 {
        (avg * bar_width) / max_avg
      } else {
        0
      };
      let bar: String = "█".repeat(filled as usize);
      let pad = " ".repeat((bar_width - filled) as usize);

      writeln!(
        f,
        "{:<width$}  {}{}  {}  ×{}",
        name,
        bar,
        pad,
        format_duration(avg),
        count,
        width = max_name,
      )?;
    }
    Ok(())
  }
}

fn format_duration(us: u64) -> String {
  if us >= 1_000_000 {
    format!("{:.1}s", us as f64 / 1_000_000.0)
  } else if us >= 1_000 {
    format!("{:.1}ms", us as f64 / 1_000.0)
  } else {
    format!("{}µs", us)
  }
}
