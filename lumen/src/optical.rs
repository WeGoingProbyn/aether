// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Geometry helpers for radiative transfer. Pure functions over centroid
//! positions and sun direction — no state, no field access.

use utility::domain::Point;

/// Cosine of the local solar zenith angle, clamped to `[0, 1]`.
///
/// `centroid` is a world-frame position; the local zenith is taken as the
/// outward radial direction from the planet's centre. The cosine is
/// clamped at zero so the night side returns 0 (no direct illumination).
///
/// `sun_direction` should be the unit vector pointing *from the planet
/// toward the sun*. If you store the sun's absolute position instead,
/// normalise the difference before calling.
pub fn zenith_cosine(centroid: &Point<3>, sun_direction: &[f64; 3]) -> f64 {
  let r = (centroid[0] * centroid[0]
    + centroid[1] * centroid[1]
    + centroid[2] * centroid[2])
    .sqrt();
  if r <= 0.0 {
    return 0.0;
  }
  let cos = (centroid[0] * sun_direction[0]
    + centroid[1] * sun_direction[1]
    + centroid[2] * sun_direction[2])
    / r;
  cos.max(0.0)
}

/// Normalise a 3-vector. Returns `[0; 3]` if the input has zero length.
pub fn normalise(v: &[f64; 3]) -> [f64; 3] {
  let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
  if len <= 0.0 {
    [0.0, 0.0, 0.0]
  } else {
    [v[0] / len, v[1] / len, v[2] / len]
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use utility::maths::vector::Vector;

  fn p(x: f64, y: f64, z: f64) -> Point<3> {
    Vector::from([x, y, z])
  }

  #[test]
  fn sub_solar_point_returns_one() {
    let centroid = p(1.0, 0.0, 0.0);
    let sun = [1.0, 0.0, 0.0];
    assert!((zenith_cosine(&centroid, &sun) - 1.0).abs() < 1e-12);
  }

  #[test]
  fn anti_solar_point_returns_zero() {
    let centroid = p(-1.0, 0.0, 0.0);
    let sun = [1.0, 0.0, 0.0];
    assert_eq!(zenith_cosine(&centroid, &sun), 0.0);
  }

  #[test]
  fn terminator_returns_zero() {
    let centroid = p(0.0, 1.0, 0.0);
    let sun = [1.0, 0.0, 0.0];
    assert!(zenith_cosine(&centroid, &sun).abs() < 1e-12);
  }

  #[test]
  fn forty_five_degree_returns_root_half() {
    let centroid = p(1.0, 1.0, 0.0);
    let sun = [1.0, 0.0, 0.0];
    let cos = zenith_cosine(&centroid, &sun);
    assert!((cos - (0.5_f64).sqrt()).abs() < 1e-12);
  }

  #[test]
  fn normalise_unit_vector() {
    let n = normalise(&[3.0, 0.0, 4.0]);
    assert!((n[0] - 0.6).abs() < 1e-12);
    assert_eq!(n[1], 0.0);
    assert!((n[2] - 0.8).abs() < 1e-12);
  }
}
