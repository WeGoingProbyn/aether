use crate::kind::{BodyKind, CelestialBody};

/// A set of celestial bodies sharing a coordinate frame — typically one star
/// at the origin with planets orbiting around it. Multi-star systems are
/// allowed; `star()` returns the first one found.
#[derive(Clone, Debug)]
pub struct System {
  pub bodies: Vec<CelestialBody>,
}

impl System {
  pub fn new(bodies: Vec<CelestialBody>) -> Self {
    System { bodies }
  }

  /// First body whose kind is `Star`. Returns `None` for rogue-planet systems.
  pub fn star(&self) -> Option<&CelestialBody> {
    self
      .bodies
      .iter()
      .find(|b| matches!(b.kind(), BodyKind::Star(_)))
  }

  /// Iterator over non-stellar bodies.
  pub fn planets(&self) -> impl Iterator<Item = &CelestialBody> + '_ {
    self
      .bodies
      .iter()
      .filter(|b| !matches!(b.kind(), BodyKind::Star(_)))
  }
}
