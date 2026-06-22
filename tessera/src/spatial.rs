// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Geographic spatial index over a mesh's cells.
//!
//! Backs the semantic query API: it maps a [`GeoCoord`] (or raw world point) to
//! the cell containing/nearest it, and collects every cell falling inside a
//! [`GeoBounds`] region. It is deliberately **range-capable from the start** —
//! point lookups and region queries share one structure — so the climatology
//! layer's regional aggregates do not force an index rewrite later.
//!
//! The index reads only [`CellGeometry::cell_world_centroid`] and
//! [`CellGeometry::cell_count`], so it is agnostic to the mesh's internal cell
//! ordering or whether it is a cube-sphere at all. Cells are bucketed by the
//! latitude/longitude of their centroid; the radial (altitude) dimension is
//! resolved naturally by nearest-3D-centroid within a horizontal bucket, so a
//! stacked atmosphere mesh needs no special handling.

use std::f64::consts::PI;

use crate::geo::GeoCoord;
use crate::geometry::CellGeometry;
use utility::domain::{CellId, Point};

/// A geographic region: latitude/longitude/altitude ranges. Longitude ranges
/// may cross the antimeridian — if `lon_min > lon_max` the band is taken to wrap
/// through ±π (e.g. `lon_min = 170°`, `lon_max = -170°` is the 20°-wide band
/// straddling the dateline).
#[derive(Clone, Copy, Debug)]
pub struct GeoBounds {
  pub lat_min: f64,
  pub lat_max: f64,
  pub lon_min: f64,
  pub lon_max: f64,
  pub alt_min: f64,
  pub alt_max: f64,
}

impl GeoBounds {
  /// Build from degrees, with an inclusive altitude band in metres.
  pub fn from_degrees(
    lat: (f64, f64),
    lon: (f64, f64),
    alt: (f64, f64),
  ) -> Self {
    Self {
      lat_min: lat.0.to_radians(),
      lat_max: lat.1.to_radians(),
      lon_min: lon.0.to_radians(),
      lon_max: lon.1.to_radians(),
      alt_min: alt.0,
      alt_max: alt.1,
    }
  }

  /// Does this region contain the given geographic coordinate?
  pub fn contains(&self, g: &GeoCoord) -> bool {
    if g.lat < self.lat_min || g.lat > self.lat_max {
      return false;
    }
    if g.alt < self.alt_min || g.alt > self.alt_max {
      return false;
    }
    self.contains_lon(g.lon)
  }

  fn contains_lon(&self, lon: f64) -> bool {
    if self.lon_min <= self.lon_max {
      lon >= self.lon_min && lon <= self.lon_max
    } else {
      // Wraps through the antimeridian.
      lon >= self.lon_min || lon <= self.lon_max
    }
  }
}

/// A latitude/longitude bucket grid over a mesh's cell centroids.
pub struct GeoIndex {
  surface_radius: f64,
  n_lat: usize,
  n_lon: usize,
  /// Row-major `n_lat * n_lon` buckets; each holds `(cell, world_centroid)`.
  bins: Vec<Vec<(CellId, Point<3>)>>,
}

impl GeoIndex {
  /// Build an index over `mesh`. `surface_radius` is the body's mean surface
  /// radius (used to convert centroids to geographic coordinates). Bucket
  /// resolution is chosen automatically from the cell count; use
  /// [`GeoIndex::with_resolution`] to override.
  pub fn build<M>(mesh: &M, surface_radius: f64) -> Self
  where
    M: CellGeometry<3> + ?Sized,
  {
    // Aim for a handful of cells per bucket so a small neighbourhood search
    // around a query reliably contains the nearest centroid.
    let cells = mesh.cell_count().max(1);
    let n = ((cells as f64 / 8.0).sqrt().round() as usize).clamp(4, 256);
    Self::with_resolution(mesh, surface_radius, n, 2 * n)
  }

  /// Build with an explicit latitude/longitude bucket resolution.
  pub fn with_resolution<M>(
    mesh: &M,
    surface_radius: f64,
    n_lat: usize,
    n_lon: usize,
  ) -> Self
  where
    M: CellGeometry<3> + ?Sized,
  {
    let n_lat = n_lat.max(1);
    let n_lon = n_lon.max(1);
    let mut bins: Vec<Vec<(CellId, Point<3>)>> =
      vec![Vec::new(); n_lat * n_lon];
    for i in 0..mesh.cell_count() {
      let cell = CellId::from(i);
      let centroid = mesh.cell_world_centroid(cell);
      let g = GeoCoord::from_world(&centroid, surface_radius);
      let (ilat, ilon) = bin_of(g.lat, g.lon, n_lat, n_lon);
      bins[ilat * n_lon + ilon].push((cell, centroid));
    }
    Self {
      surface_radius,
      n_lat,
      n_lon,
      bins,
    }
  }

  /// The body radius this index was built with.
  pub fn surface_radius(&self) -> f64 {
    self.surface_radius
  }

  /// Cell whose centroid is nearest the given geographic coordinate, or `None`
  /// if the mesh has no cells.
  pub fn locate(&self, g: &GeoCoord) -> Option<CellId> {
    self.nearest_world(&g.to_world(self.surface_radius))
  }

  /// Cell whose centroid is nearest the given world-space point.
  pub fn nearest_world(&self, p: &Point<3>) -> Option<CellId> {
    let g = GeoCoord::from_world(p, self.surface_radius);
    let (ilat, ilon) = bin_of(g.lat, g.lon, self.n_lat, self.n_lon);

    let mut best: Option<(f64, CellId)> = None;
    let mut consider = |bin: &Vec<(CellId, Point<3>)>| {
      for (cell, c) in bin {
        let d = dist2(p, c);
        if best.map_or(true, |(bd, _)| d < bd) {
          best = Some((d, *cell));
        }
      }
    };

    // Longitude collapses near the poles, so the nearest centroid can sit in a
    // far-apart longitude bucket of the same (or adjacent) latitude row. When
    // the query is in a polar row, scan those rows in full.
    let polar = ilat == 0 || ilat + 1 >= self.n_lat;
    if polar {
      let lo = ilat.saturating_sub(1);
      let hi = (ilat + 1).min(self.n_lat - 1);
      for row in lo..=hi {
        for col in 0..self.n_lon {
          consider(&self.bins[row * self.n_lon + col]);
        }
      }
    }

    // Expanding Chebyshev rings: search outward until the first non-empty ring,
    // then two further rings to absorb non-uniform cube-sphere cell spacing.
    let max_ring = self.n_lat.max(self.n_lon);
    let mut first_hit: Option<usize> = None;
    for ring in 0..=max_ring {
      let mut found = false;
      for (row, col) in self.ring_bins(ilat, ilon, ring) {
        let bin = &self.bins[row * self.n_lon + col];
        if !bin.is_empty() {
          found = true;
        }
        consider(bin);
      }
      if found && first_hit.is_none() {
        first_hit = Some(ring);
      }
      if let Some(fh) = first_hit {
        if ring >= fh + 2 {
          break;
        }
      }
    }

    best.map(|(_, cell)| cell)
  }

  /// All cells whose centroid falls inside `bounds`.
  pub fn cells_in(&self, bounds: &GeoBounds) -> Vec<CellId> {
    let mut out = Vec::new();
    for bin in &self.bins {
      for (cell, c) in bin {
        let g = GeoCoord::from_world(c, self.surface_radius);
        if bounds.contains(&g) {
          out.push(*cell);
        }
      }
    }
    out
  }

  /// Bucket coordinates at exactly Chebyshev distance `ring` from
  /// `(ilat, ilon)`. Latitude is clamped to the grid; longitude wraps.
  fn ring_bins(
    &self,
    ilat: usize,
    ilon: usize,
    ring: usize,
  ) -> Vec<(usize, usize)> {
    let r = ring as isize;
    let mut out = Vec::new();
    let push = |dlat: isize, dlon: isize, out: &mut Vec<(usize, usize)>| {
      let row = ilat as isize + dlat;
      if row < 0 || row >= self.n_lat as isize {
        return;
      }
      let col = (ilon as isize + dlon).rem_euclid(self.n_lon as isize) as usize;
      out.push((row as usize, col));
    };
    if ring == 0 {
      push(0, 0, &mut out);
      return out;
    }
    for dlon in -r..=r {
      push(-r, dlon, &mut out);
      push(r, dlon, &mut out);
    }
    for dlat in (-r + 1)..r {
      push(dlat, -r, &mut out);
      push(dlat, r, &mut out);
    }
    out
  }
}

/// Latitude/longitude bucket of a coordinate.
fn bin_of(lat: f64, lon: f64, n_lat: usize, n_lon: usize) -> (usize, usize) {
  let flat = ((lat + PI / 2.0) / PI * n_lat as f64).floor() as isize;
  let flon = ((lon + PI) / (2.0 * PI) * n_lon as f64).floor() as isize;
  let ilat = flat.clamp(0, n_lat as isize - 1) as usize;
  let ilon = flon.rem_euclid(n_lon as isize) as usize;
  (ilat, ilon)
}

fn dist2(a: &Point<3>, b: &Point<3>) -> f64 {
  let dx = a[0] - b[0];
  let dy = a[1] - b[1];
  let dz = a[2] - b[2];
  dx * dx + dy * dy + dz * dz
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::cube_sphere::CubeSphere;
  use crate::geometry::CellGeometry;

  const R_INNER: f64 = 6.371e6;
  const R_OUTER: f64 = 6.391e6;
  const SURFACE: f64 = R_INNER;

  fn shell() -> CubeSphere {
    // 16x16 angular per panel, 4 radial layers.
    CubeSphere::new([16, 16, 4], R_INNER, R_OUTER)
  }

  #[test]
  fn locate_returns_the_containing_cell_centroid() {
    let mesh = shell();
    let index = GeoIndex::build(&mesh, SURFACE);
    // For every cell, its own centroid's geo position must locate back to it.
    for i in 0..mesh.cell_count() {
      let cell = CellId::from(i);
      let centroid = mesh.cell_world_centroid(cell);
      let g = GeoCoord::from_world(&centroid, SURFACE);
      let found = index.locate(&g).expect("non-empty mesh");
      assert_eq!(
        found.index(),
        i,
        "centroid of cell {i} located cell {} instead",
        found.index()
      );
    }
  }

  #[test]
  fn locate_is_robust_at_poles_and_panel_boundaries() {
    let mesh = shell();
    let index = GeoIndex::build(&mesh, SURFACE);
    // Sample points including the poles and the gnomonic panel seams.
    let probes = [
      GeoCoord::from_degrees(89.9, 0.0, 0.0),
      GeoCoord::from_degrees(-89.9, 137.0, 0.0),
      GeoCoord::from_degrees(90.0, 0.0, 0.0),
      GeoCoord::from_degrees(44.9, 44.9, 5000.0), // near a cube corner
      GeoCoord::from_degrees(0.0, 180.0, 0.0),    // antimeridian
      GeoCoord::from_degrees(0.0, 45.0, 10000.0), // panel seam, top layer
    ];
    for g in probes {
      let found = index.locate(&g).expect("non-empty mesh");
      // Brute-force ground truth: globally nearest centroid.
      let target = g.to_world(SURFACE);
      let mut best = (f64::INFINITY, usize::MAX);
      for i in 0..mesh.cell_count() {
        let c = mesh.cell_world_centroid(CellId::from(i));
        let d = dist2(&target, &c);
        if d < best.0 {
          best = (d, i);
        }
      }
      assert_eq!(
        found.index(),
        best.1,
        "probe {g:?}: index found {} but nearest is {}",
        found.index(),
        best.1
      );
    }
  }

  #[test]
  fn range_query_matches_brute_force() {
    let mesh = shell();
    let index = GeoIndex::build(&mesh, SURFACE);
    let bounds = GeoBounds::from_degrees(
      (-10.0, 30.0),
      (20.0, 80.0),
      (-1.0, R_OUTER - R_INNER + 1.0),
    );
    let mut from_index = index.cells_in(&bounds);
    from_index.sort_by_key(|c| c.index());

    let mut brute: Vec<CellId> = (0..mesh.cell_count())
      .map(CellId::from)
      .filter(|&c| {
        let g = GeoCoord::from_world(&mesh.cell_world_centroid(c), SURFACE);
        bounds.contains(&g)
      })
      .collect();
    brute.sort_by_key(|c| c.index());

    assert_eq!(from_index, brute);
    assert!(!brute.is_empty(), "expected some cells in the region");
  }

  #[test]
  fn range_query_handles_antimeridian_wrap() {
    let mesh = shell();
    let index = GeoIndex::build(&mesh, SURFACE);
    // 20-degree band straddling the dateline: lon_min > lon_max.
    let bounds = GeoBounds::from_degrees(
      (-5.0, 5.0),
      (170.0, -170.0),
      (-1.0, R_OUTER - R_INNER + 1.0),
    );
    let mut from_index = index.cells_in(&bounds);
    from_index.sort_by_key(|c| c.index());
    let mut brute: Vec<CellId> = (0..mesh.cell_count())
      .map(CellId::from)
      .filter(|&c| {
        let g = GeoCoord::from_world(&mesh.cell_world_centroid(c), SURFACE);
        bounds.contains(&g)
      })
      .collect();
    brute.sort_by_key(|c| c.index());
    assert_eq!(from_index, brute);
    assert!(!brute.is_empty(), "expected cells straddling the dateline");
  }
}
