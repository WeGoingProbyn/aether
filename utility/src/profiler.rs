use std::{
  cell::RefCell, sync::{Mutex, OnceLock, atomic::{AtomicBool, Ordering}}, time::Instant,
};

use crate::debug;

static PROFILER: OnceLock<Profiler> = OnceLock::new();
static ENABLED: AtomicBool = AtomicBool::new(true);

struct SpanEvent {
  name: &'static str,
  category: &'static str,
  start: u64,
  duration: u64,
  thread_id: u64,
}

struct OpenSpan {
  name: &'static str,
  category: &'static str,
  start: Instant,
}

struct ThreadState {
  stack: Vec<OpenSpan>,
  complete: Vec<SpanEvent>,
}

thread_local! {
  static STATE: RefCell<ThreadState> = RefCell::new(
    ThreadState { 
      stack: Vec::with_capacity(64),
      complete: Vec::with_capacity(4096), 
    })
}

pub struct Profiler {
  epoch: Instant,
  events: Mutex<Vec<SpanEvent>>,
}

pub struct SpanGuard;

impl SpanGuard {
  pub fn new(name: &'static str, category: &'static str) -> SpanGuard {
    if Profiler::enabled() {
      STATE.with(|s| {
        let mut borrow = s.borrow_mut(); 
        borrow.stack.push(OpenSpan {
          name,
          category,
          start: Instant::now(),
        });
      })
    }

    SpanGuard
  }
}

impl Drop for SpanGuard {
  fn drop(&mut self) {
    if Profiler::enabled() {
      let end = Instant::now();
      STATE.with(|s|{
        let mut borrow = s.borrow_mut();
        if let Some(span) = borrow.stack.pop() {
          let profiler = PROFILER.get().unwrap();
          borrow.complete.push(SpanEvent { 
            name: span.name, 
            category: span.category, 
            start: span.start.duration_since(profiler.epoch).as_micros() as u64, 
            duration: end.duration_since(span.start).as_micros() as u64,
            thread_id: 0, 
          });

          if borrow.complete.len() >= 4096 {
            let mut events = profiler.events.lock().unwrap();
            events.extend(borrow.complete.drain(..));
          }
        }
      })
    }
  }
}

impl Profiler {
  pub fn init() {
    let _ = PROFILER.set(Profiler {
      epoch: Instant::now(),
      events: Mutex::new(Vec::with_capacity(8192)),
    });
  }

  pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
  }

  pub fn print() {
    let profiler = PROFILER.get().unwrap();
    Profiler::flush(&mut std::io::stdout());
    debug!("{}", profiler);
  }

  pub fn flush(writer: &mut dyn std::io::Write) {
    let profiler = match PROFILER.get() {
      Some(p) => p,
      None => return,
    };

    STATE.with(|s| {
      let mut borrow = s.borrow_mut();
      let mut events = profiler.events.lock().unwrap();

      events.extend(borrow.complete.drain(..));
    });

    let events = profiler.events.lock().unwrap();
    let pid = std::process::id();

    let _ = write!(writer, "{{\"traceEvents\":[");
    for (i, ev) in events.iter().enumerate() {
      if i > 0 { let _ = write!(writer, ","); }
      let _ = write!(
        writer,
        "{{\"name\":\"{}\",\"cat\":\"{}\",\"ph\":\"X\",\"ts\":{},\"dur\":{},\"pid\":{},\"tid\":{}}}",
        ev.name, ev.category, ev.start, ev.duration, pid, ev.thread_id,
      );
    }
    let _ = write!(writer, "]}}\n\n");
  }
}

impl std::fmt::Display for Profiler {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let events = self.events.lock().unwrap();

    // aggregate: name -> (total_duration_us, count)
    let mut agg: Vec<(&str, u64, u64)> = Vec::new();
    for ev in events.iter() {
      if let Some(entry) = agg.iter_mut().find(|(n, _, _)| *n == ev.name) {
        entry.1 += ev.duration;
        entry.2 += 1;
      } else {
        agg.push((ev.name, ev.duration, 1));
      }
    }

    // sort by avg duration descending
    agg.sort_by(|a, b| (b.1 / b.2).cmp(&(a.1 / a.2)));

    let max_name = agg.iter().map(|(n, _, _)| n.len()).max().unwrap_or(0);
    let max_avg = agg.iter().map(|(_, d, c)| d / c).max().unwrap_or(1);
    let bar_width: u64 = 32;

    writeln!(f, "── Profiler Diagnostics ──")?;
    for (name, total, count) in &agg {
      let avg = total / count;
      let filled = if max_avg > 0 { (avg * bar_width) / max_avg } else { 0 };
      let bar: String = "█".repeat(filled as usize);
      let pad = " ".repeat((bar_width - filled) as usize);

      writeln!(f, "{:<width$}  {}{}  {}  ×{}",
        name, bar, pad, format_duration(avg), count,
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
