use utility::error::{
  AetherError, AetherResult, ErrorDomain, UtilityErrorKind
};

use utility::logger::{Logger, StdSink, Level};
use utility::maths::matrix::Matrix;
use utility::maths::quaternion::Quaternion;
use utility::maths::vector::Vector;

use utility::profiler::Profiler;
use utility::serial::deserialize::Deserialize;
use utility::serial::json::{JsonDeserializer, JsonSerializer};
use utility::{Deserialize, debug, error, info, trace, warn};

use utility::profile;
use utility::Serialize;
use utility::serial::serialize::Serialize;

#[profile]
fn testing() {
  for _ in 0..1000 {
    let mut this = 0;
    this += 1;
  }
}

#[profile]
fn testing2() {
  for _ in 0..100 {
    for _ in 0..100 {
      let mut this = 0;
      this += 1;
    }
  }
}

#[profile]
fn testing3() {
  for _ in 0..100 {
    for _ in 0..100 {
      for _ in 0..100 {
        let mut this = 0;
        this += 1;
      }
    }
  }
}

#[derive(Serialize, Deserialize, Debug)]
struct Testingtest {
  thing: f64,
  another: u32,
  athing: Vec<Another>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Another {
  thing2: f64,
  another2: u32,
  athing2: Vec<u8>,
}

impl Testingtest {
  fn new() -> Testingtest {
    Testingtest {
      thing: 10f64,
      another: 45u32,
      athing: vec![Another::new(); 3],
    }
  }
}

impl Another {
  fn new() -> Another {
    Another {
      thing2: 10f64,
      another2: 45u32,
      athing2: vec![1u8, 2u8, 3u8],
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

  let mat: Matrix<f32, 3, 3> =
    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]].into();

  let identity = Matrix::<f32, 3, 3>::identity(1.0);

  let vec: Vector<f32, 3> = [0.0, 0.0, 1.0].into();
  let vec2: Vector<f32, 3> = [1.0, 0.0, 0.0].into();

  let quat: Quaternion<f32> = [1.0, 0.0, 1.0, 0.0].into();
  let cross = vec.cross(&vec2);
  trace!("{:?}", mat);
  info!("{:?}", mat);
  warn!("{:?}", cross);
  debug!("{:?}", quat);
  error!("{:?}", identity);

  testing();
  testing2();
  testing2();
  testing3();

  let string: String = "{\"thing\":10,\"another\":45,\"athing\":[{\"thing2\":10,\"another2\":45,\"athing2\":[1,2,3]},{\"thing2\":10,\"another2\":45,\"athing2\":[1,2,3]},{\"thing2\":10,\"another2\":45,\"athing2\":[1,2,3]}]}".into();
  let mut ds = JsonDeserializer::new(std::io::Cursor::new(string));
  let test = Testingtest::deserialize(&mut ds)?;
  info!("{:?}", test);
  
  let test = Testingtest::new();
  let mut buf = Vec::new();
  test.serialize(&mut JsonSerializer::new(&mut buf))?;
  let json = String::from_utf8(buf).unwrap();
  info!("{}", json); 

  Profiler::print();

  Err(
    AetherError::new(ErrorDomain::Utility(UtilityErrorKind::Unknown))
      .context("idfk")
      .context("another one?!")
      .parent(
        AetherError::new(ErrorDomain::Utility(UtilityErrorKind::Unknown))
          .context("irdfk")
          .parent(
            AetherError::new(ErrorDomain::Utility(UtilityErrorKind::Unknown))
              .context("Something really bad went whoopsie sorry about that")
              .parent(std::io::Error::from(std::io::ErrorKind::UnexpectedEof)),
          ),
      ),
  )
}
