use utility::error::AetherResult;

use utility::logger::{Level, Logger, StdSink};
//use utility::maths::quaternion::Quaternion;
//use utility::maths::matrix::Matrix;
//use utility::maths::vector::Vector;
//use utility::logger::LogWriter;

use utility::thread::pool::{Pool, TaskGraph};
use utility::profiler::Profiler;
use utility::profile;
use utility::info;

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

  let pool = Pool::default();

  let handle = pool.spawn(|| {
    testing3();
  });
  handle.signal().wait();

  let mut graph = TaskGraph::new();

  info!("submitting broadphase");
  let broadphase  = graph.add(|| {
    testing2();
    info!("completed broadphase")
  });
  info!("submitting narrowphase a");
  let narrow1 = graph.add(|| {
    testing();
    info!("completed narrowphase a")
  });
  info!("submitting narrowphase b");
  let narrow2 = graph.add(|| {
    testing();
    info!("completed narrowphase b")
  });
  info!("submitting solver");
  let solver = graph.add(|| {
    testing2();
    info!("completed solver")
  });
  info!("submitting integrate");
  let integrate = graph.add(|| {
    testing3();
    info!("completed integrate")
  });

  graph.dependency(narrow1, broadphase)?;
  graph.dependency(narrow2, broadphase)?;
  graph.dependency(solver, narrow1)?;
  graph.dependency(solver, narrow2)?;
  graph.dependency(integrate, solver)?;

  pool.execute(graph)?;

  let mut data = [1f32; 100];
  pool.parallel_for(&mut data, 25, |chunk| {
    for thing in chunk {
      *thing *= 2.0;
    }
  });

  info!("{:?}", data);

  pool.flush_profiler();
  //Profiler::print(&mut LogWriter::new(Level::Debug));

  Ok(())
}







