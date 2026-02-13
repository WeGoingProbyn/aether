use utility::error::{
  AetherError, AetherResult, ErrorDomain 
};

use utility::logger::{Logger, StdSink, Level};
use utility::maths::matrix::Matrix;
use utility::maths::quaternion::Quaternion;
use utility::maths::vector::Vector;

use utility::profiler::Profiler;
use utility::thread::pool::{Pool, TaskGraph};
use utility::{debug, error, info, trace, warn};
use utility::profile;

#[profile]
fn testing() {
  for _ in 0..1000 {
    let mut _this = 0;
    _this += 1;
  }
}

#[profile]
fn testing2() {
  for _ in 0..100 {
    for _ in 0..100 {
      let mut _this = 0;
      _this += 1;
    }
  }
}

#[profile]
fn testing3() {
  for _ in 0..100 {
    for _ in 0..100 {
      for _ in 0..100 {
        let mut _this = 0;
        _this += 1;
      }
    }
  }
}

fn main() -> AetherResult<()> {
  Logger::init(
    vec![
      Box::new(StdSink::new(std::io::stdout()).capacity(1)),
    ], 
    Level::Trace
  );

  Profiler::init();

  // Create pool with one worker per core
  let pool = Pool::default();

  let handle = pool.spawn(|| {
    testing3();
  });
  handle.signal().wait(); // block until done

  let mut graph = TaskGraph::new();

  info!("submitting broadphase");
  let broadphase  = graph.add("broadphase", || {
    testing2();
    info!("completed broadphase")
  });
  info!("submitting narrowphase a");
  let narrow1 = graph.add("narrowphase_a", || {
    testing();
    info!("completed narrowphase a")
  });
  info!("submitting narrowphase b");
  let narrow2 = graph.add("narrowphase_b", || {
    testing();
    info!("completed narrowphase b")
  });
  info!("submitting solver");
  let solver = graph.add("solver", || {
    testing2();
    info!("completed solver")
  });
  info!("submitting integrate");
  let integrate = graph.add("integrate", || {
    testing3();
    info!("completed integrate")
  });

  graph.dependency(narrow1, broadphase)?;
  graph.dependency(narrow2, broadphase)?;
  graph.dependency(solver, narrow1)?;
  graph.dependency(solver, narrow2)?;
  graph.dependency(integrate, solver)?;

  pool.execute(graph)?;

  pool.flush_profiler();
  // drop(pool);
  Profiler::print(&mut std::io::stdout());

  Ok(())
}
