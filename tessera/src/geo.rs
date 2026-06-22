// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Geographic coordinates and their mapping to/from world Cartesian space.
//!
//! This is the consumer-facing coordinate vocabulary for the semantic query
//! API: a library consumer thinks in latitude / longitude / altitude, never in
//! the cube-sphere's gnomonic `(ξ, η, r)` computational coordinates. The
//! conversions here are pure spherical geometry and are *planet-agnostic* — the
//! caller supplies the body's surface radius, so nothing here bakes in Earth.
//!
//! Convention: a right-handed world frame with `+z` as the north polar axis.
//! This matches the cube-sphere's `ZP` / `ZN` panels being the polar caps (see
//! [`crate::cube_sphere::panel_axes`]). Longitude is measured from the `+x`
//! axis toward `+y` (eastward).
//!
//! - latitude  φ ∈ [-π/2, +π/2], +π/2 at the north pole (`+z`)
//! - longitude λ ∈ (-π, +π], measured eastward from `+x`
//! - altitude  h: metres above the supplied surface radius (may be negative)

use std::f64::consts::PI;

use utility::domain::Point;

/// A point on or above a planetary sphere, in geographic coordinates.
///
/// Latitude and longitude are stored in **radians**; altitude is in **metres**
/// above the surface radius passed to the conversion functions. Use
/// [`GeoCoord::from_degrees`] / [`GeoCoord::latitude_deg`] at the API edge where
/// degrees are more ergonomic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeoCoord {
  /// Latitude in radians, +π/2 at the north pole.
  pub lat: f64,
  /// Longitude in radians, eastward from the +x axis, wrapped to (-π, π].
  pub lon: f64,
  /// Altitude in metres above the surface radius.
  pub alt: f64,
}

impl GeoCoord {
  /// Construct from radians. Longitude is wrapped to (-π, π]; latitude is
  /// clamped to [-π/2, π/2].
  pub fn from_radians(lat: f64, lon: f64, alt: f64) -> Self {
    Self {
      lat: lat.clamp(-PI / 2.0, PI / 2.0),
      lon: wrap_longitude(lon),
      alt,
    }
  }

  /// Construct from degrees. Convenient at the consumer-facing API edge.
  pub fn from_degrees(lat_deg: f64, lon_deg: f64, alt: f64) -> Self {
    Self::from_radians(lat_deg.to_radians(), lon_deg.to_radians(), alt)
  }

  /// Latitude in degrees.
  pub fn latitude_deg(&self) -> f64 {
    self.lat.to_degrees()
  }

  /// Longitude in degrees.
  pub fn longitude_deg(&self) -> f64 {
    self.lon.to_degrees()
  }

  /// Map to a world-space Cartesian point, given the body's surface radius.
  /// The radial distance is `surface_radius + alt`.
  pub fn to_world(&self, surface_radius: f64) -> Point<3> {
    let r = surface_radius + self.alt;
    let (sin_lat, cos_lat) = self.lat.sin_cos();
    let (sin_lon, cos_lon) = self.lon.sin_cos();
    [r * cos_lat * cos_lon, r * cos_lat * sin_lon, r * sin_lat].into()
  }

  /// Recover geographic coordinates from a world-space Cartesian point, given
  /// the body's surface radius. At the poles longitude is degenerate and
  /// returns 0, but the round-tripped *position* is exact.
  pub fn from_world(p: &Point<3>, surface_radius: f64) -> Self {
    let r = p.magnitude();
    let lat = if r > 0.0 {
      (p[2] / r).clamp(-1.0, 1.0).asin()
    } else {
      0.0
    };
    let lon = p[1].atan2(p[0]);
    Self {
      lat,
      lon: wrap_longitude(lon),
      alt: r - surface_radius,
    }
  }
}

/// Wrap a longitude into (-π, π].
fn wrap_longitude(lon: f64) -> f64 {
  let two_pi = 2.0 * PI;
  let mut wrapped = lon % two_pi;
  if wrapped <= -PI {
    wrapped += two_pi;
  } else if wrapped > PI {
    wrapped -= two_pi;
  }
  wrapped
}

#[cfg(test)]
mod tests {
  use super::*;

  const R: f64 = 6.371e6;
  const EPS: f64 = 1e-6;

  fn assert_world_close(a: &Point<3>, b: &Point<3>) {
    for k in 0..3 {
      assert!(
        (a[k] - b[k]).abs() <= EPS * R.max(1.0),
        "component {k}: {} vs {}",
        a[k],
        b[k]
      );
    }
  }

  #[test]
  fn equator_prime_meridian_maps_to_plus_x() {
    let g = GeoCoord::from_degrees(0.0, 0.0, 0.0);
    let w = g.to_world(R);
    assert_world_close(&w, &[R, 0.0, 0.0].into());
  }

  #[test]
  fn ninety_east_maps_to_plus_y() {
    let g = GeoCoord::from_degrees(0.0, 90.0, 0.0);
    let w = g.to_world(R);
    assert_world_close(&w, &[0.0, R, 0.0].into());
  }

  #[test]
  fn north_pole_maps_to_plus_z() {
    let g = GeoCoord::from_degrees(90.0, 123.0, 0.0);
    let w = g.to_world(R);
    assert_world_close(&w, &[0.0, 0.0, R].into());
  }

  #[test]
  fn altitude_extends_radius() {
    let g = GeoCoord::from_degrees(0.0, 0.0, 1000.0);
    let w = g.to_world(R);
    assert!((w.magnitude() - (R + 1000.0)).abs() <= EPS * R);
  }

  #[test]
  fn round_trip_general_points() {
    for &(lat, lon, alt) in &[
      (12.3, 45.6, 0.0),
      (-67.8, -120.0, 5000.0),
      (0.0, 179.9, -200.0),
      (-89.0, 30.0, 0.0),
    ] {
      let g = GeoCoord::from_degrees(lat, lon, alt);
      let back = GeoCoord::from_world(&g.to_world(R), R);
      assert!(
        (back.lat - g.lat).abs() <= EPS,
        "lat {} vs {}",
        back.lat,
        g.lat
      );
      assert!(
        (back.lon - g.lon).abs() <= EPS,
        "lon {} vs {}",
        back.lon,
        g.lon
      );
      assert!((back.alt - g.alt).abs() <= EPS * R, "alt");
    }
  }

  #[test]
  fn poles_round_trip_position_even_if_longitude_degenerate() {
    // Longitude is undefined at the pole, but the position must round-trip.
    let g = GeoCoord::from_degrees(90.0, 77.0, 0.0);
    let w = g.to_world(R);
    let back = GeoCoord::from_world(&w, R).to_world(R);
    assert_world_close(&w, &back);
  }

  #[test]
  fn dateline_wraps_to_pi() {
    let g = GeoCoord::from_degrees(0.0, 180.0, 0.0);
    assert!((g.lon - PI).abs() <= EPS, "lon {}", g.lon);
    let g2 = GeoCoord::from_degrees(0.0, -180.0, 0.0);
    assert!((g2.lon - PI).abs() <= EPS, "lon {}", g2.lon);
  }
}
