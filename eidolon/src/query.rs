// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Semantic query API: the consumer-facing read seam over a world snapshot.
//!
//! A library consumer (a civilisation sim, a survival game, …) asks for
//! *meaning* in geographic coordinates — "the wind at this lat/lon", "mean
//! surface temperature over this region" — and never touches conserved-variable
//! layouts or the cube-sphere's gnomonic coordinates. The public [`Quantity`]
//! vocabulary is deliberately decoupled from the engine's internal field names,
//! so it stays stable as the physics evolves.
//!
//! Queries read the interpolated [`crate::playback::FrameInterpolator`]
//! snapshot, not live simulation state, so they are thread-safe and consistent
//! while the solver runs on another thread. Every result is a [`Sample`], which
//! carries the value together with how much to trust it.

use crate::playback::{FrameInterpolator, MeshChannel};
use std::collections::HashMap;
use tessera::geo::GeoCoord;
use tessera::spatial::{GeoBounds, GeoIndex};
use tessera::world_mesh::Tessera;
use utility::domain::{CellId, MeshType};
use utility::maths::vector::Vector;

/// Outcome of a world query. The value is always present except for
/// [`Sample::Unavailable`]; the variant tells the consumer how much to trust it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sample<T> {
  /// A fresh value, interpolated between two distinct snapshot frames.
  Ok(T),
  /// A usable value, but served by snapping to a single frame (the sim has not
  /// delivered a newer frame to interpolate toward — e.g. it stalled).
  Stale(T),
  /// A value was found but is non-finite or otherwise physically suspect; the
  /// simulation may be degrading. Surfaced rather than hidden so a shipped game
  /// can react instead of crashing.
  Degraded(T),
  /// No value: the quantity is not carried for this world, or the coordinate is
  /// off the relevant mesh.
  Unavailable,
}

impl<T> Sample<T> {
  /// The contained value, if any.
  pub fn value(self) -> Option<T> {
    match self {
      Sample::Ok(v) | Sample::Stale(v) | Sample::Degraded(v) => Some(v),
      Sample::Unavailable => None,
    }
  }

  /// True only for [`Sample::Ok`].
  pub fn is_ok(&self) -> bool {
    matches!(self, Sample::Ok(_))
  }

  /// True for [`Sample::Unavailable`].
  pub fn is_unavailable(&self) -> bool {
    matches!(self, Sample::Unavailable)
  }

  /// Transform the contained value, preserving the quality tag.
  pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Sample<U> {
    match self {
      Sample::Ok(v) => Sample::Ok(f(v)),
      Sample::Stale(v) => Sample::Stale(f(v)),
      Sample::Degraded(v) => Sample::Degraded(f(v)),
      Sample::Unavailable => Sample::Unavailable,
    }
  }
}

/// Scalar physical quantities a consumer can sample. Decoupled from the
/// engine's internal `FieldName` enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarQuantity {
  /// Air temperature (K).
  Temperature,
  /// Air pressure (Pa).
  Pressure,
  /// Specific humidity (kg/kg).
  Humidity,
  /// Sea-surface temperature (K).
  SeaSurfaceTemperature,
}

/// Vector physical quantities. Returned in a local east-north-up (ENU) frame at
/// the query point, which is the meaningful form for gameplay ("trade winds").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VectorQuantity {
  /// Air velocity (m/s), returned as `[east, north, up]`.
  Wind,
}

/// How to collapse a region of scalar samples into one value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Reduction {
  Mean,
  Min,
  Max,
}

impl ScalarQuantity {
  /// The mesh this quantity lives on and the snapshot channel carrying it.
  fn binding(self) -> (MeshType, MeshChannel) {
    match self {
      ScalarQuantity::Temperature => {
        (MeshType::Atmosphere, MeshChannel::AtmosphereTemperature)
      }
      ScalarQuantity::Pressure => {
        (MeshType::Atmosphere, MeshChannel::AtmospherePressure)
      }
      ScalarQuantity::Humidity => {
        (MeshType::Atmosphere, MeshChannel::AtmosphereHumidity)
      }
      ScalarQuantity::SeaSurfaceTemperature => {
        (MeshType::Ocean, MeshChannel::SeaSurfaceTemperature)
      }
    }
  }
}

/// A geographic, semantic, read-only view over a world. Owns a per-mesh spatial
/// index (built once from the static mesh geometry) and is queried against an
/// interpolated snapshot supplied at call time.
pub struct WorldQuery {
  surface_radius: f64,
  indices: HashMap<MeshType, GeoIndex>,
}

impl WorldQuery {
  /// Build the query view from a world's meshes. `surface_radius` is the body's
  /// mean surface radius, used for the geographic ↔ world coordinate mapping.
  pub fn new(tessera: &Tessera, surface_radius: f64) -> Self {
    let mut indices = HashMap::new();
    for (key, mesh) in tessera.meshes() {
      let index = GeoIndex::build(mesh.as_ref(), surface_radius);
      indices.insert(key.mesh_type(), index);
    }
    Self {
      surface_radius,
      indices,
    }
  }

  /// The body radius this view was built with.
  pub fn surface_radius(&self) -> f64 {
    self.surface_radius
  }

  /// The cell of `mesh` nearest a geographic coordinate, if that mesh exists.
  pub fn locate(&self, mesh: MeshType, at: GeoCoord) -> Option<CellId> {
    self.indices.get(&mesh).and_then(|index| index.locate(&at))
  }

  /// Sample a scalar quantity at a geographic coordinate.
  pub fn sample_scalar(
    &self,
    snapshot: &FrameInterpolator,
    quantity: ScalarQuantity,
    at: GeoCoord,
  ) -> Sample<f64> {
    let (mesh, channel) = quantity.binding();
    let Some(index) = self.indices.get(&mesh) else {
      return Sample::Unavailable;
    };
    let Some(cell) = index.locate(&at) else {
      return Sample::Unavailable;
    };
    let Some(values) = snapshot.quantity(channel) else {
      return Sample::Unavailable;
    };
    match values.get(cell.index()) {
      Some(&v) => classify(v, snapshot.is_interpolating()),
      None => Sample::Unavailable,
    }
  }

  /// Sample wind at a geographic coordinate, returned as `[east, north, up]`
  /// (m/s) in the local tangent frame.
  pub fn sample_wind(
    &self,
    snapshot: &FrameInterpolator,
    at: GeoCoord,
  ) -> Sample<[f64; 3]> {
    let Some(index) = self.indices.get(&MeshType::Atmosphere) else {
      return Sample::Unavailable;
    };
    let Some(cell) = index.locate(&at) else {
      return Sample::Unavailable;
    };
    let (Some(vx), Some(vy), Some(vz)) = (
      snapshot.quantity(MeshChannel::AtmosphereWindX),
      snapshot.quantity(MeshChannel::AtmosphereWindY),
      snapshot.quantity(MeshChannel::AtmosphereWindZ),
    ) else {
      return Sample::Unavailable;
    };
    let i = cell.index();
    let (Some(&x), Some(&y), Some(&z)) = (vx.get(i), vy.get(i), vz.get(i))
    else {
      return Sample::Unavailable;
    };
    let world: Vector<f64, 3> = [x, y, z].into();
    let enu = world_to_enu(&world, &at);
    let finite = enu.iter().all(|c| c.is_finite());
    if !finite {
      Sample::Degraded(enu)
    } else if snapshot.is_interpolating() {
      Sample::Ok(enu)
    } else {
      Sample::Stale(enu)
    }
  }

  /// Reduce a scalar quantity over a geographic region.
  pub fn reduce_scalar(
    &self,
    snapshot: &FrameInterpolator,
    quantity: ScalarQuantity,
    bounds: &GeoBounds,
    reduction: Reduction,
  ) -> Sample<f64> {
    let (mesh, channel) = quantity.binding();
    let Some(index) = self.indices.get(&mesh) else {
      return Sample::Unavailable;
    };
    let Some(values) = snapshot.quantity(channel) else {
      return Sample::Unavailable;
    };
    let mut acc: Option<f64> = None;
    let mut count = 0u64;
    let mut saw_nonfinite = false;
    for cell in index.cells_in(bounds) {
      let Some(&v) = values.get(cell.index()) else {
        continue;
      };
      if !v.is_finite() {
        saw_nonfinite = true;
        continue;
      }
      count += 1;
      acc = Some(match (reduction, acc) {
        (_, None) => v,
        (Reduction::Mean, Some(a)) => a + v,
        (Reduction::Min, Some(a)) => a.min(v),
        (Reduction::Max, Some(a)) => a.max(v),
      });
    }
    let Some(a) = acc else {
      // No finite samples in the region.
      return if saw_nonfinite {
        Sample::Degraded(f64::NAN)
      } else {
        Sample::Unavailable
      };
    };
    let result = match reduction {
      Reduction::Mean => a / count as f64,
      _ => a,
    };
    if saw_nonfinite {
      Sample::Degraded(result)
    } else if snapshot.is_interpolating() {
      Sample::Ok(result)
    } else {
      Sample::Stale(result)
    }
  }
}

/// Tag a scalar value by finiteness and interpolation state.
fn classify(v: f64, interpolating: bool) -> Sample<f64> {
  if !v.is_finite() {
    Sample::Degraded(v)
  } else if interpolating {
    Sample::Ok(v)
  } else {
    Sample::Stale(v)
  }
}

/// Project a world-frame vector into the local east-north-up frame at `at`.
fn world_to_enu(world: &Vector<f64, 3>, at: &GeoCoord) -> [f64; 3] {
  let (sin_lat, cos_lat) = at.lat.sin_cos();
  let (sin_lon, cos_lon) = at.lon.sin_cos();
  let east: Vector<f64, 3> = [-sin_lon, cos_lon, 0.0].into();
  let north: Vector<f64, 3> =
    [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat].into();
  let up: Vector<f64, 3> =
    [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat].into();
  [world.dot(&east), world.dot(&north), world.dot(&up)]
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::playback::SampleFrame;
  use std::sync::Arc;
  use tessera::cube_sphere::CubeSphere;
  use tessera::geometry::CellGeometry;
  use tessera::mesh::Mesh;
  use utility::domain::MeshKey;

  const R_INNER: f64 = 6.371e6;
  const R_OUTER: f64 = 6.391e6;
  const SURFACE: f64 = R_INNER;

  fn world_with_atmosphere() -> (Tessera, Arc<CubeSphere>) {
    let mesh = Arc::new(CubeSphere::new([16, 16, 4], R_INNER, R_OUTER));
    let mut tessera = Tessera::new();
    tessera
      .register_mesh(MeshKey::ATMOSPHERE, mesh.clone() as Arc<dyn Mesh<3>>);
    (tessera, mesh)
  }

  /// A snapshot whose temperature channel equals each cell's latitude in
  /// degrees, so a query result is analytically predictable.
  fn primed_snapshot(mesh: &CubeSphere) -> FrameInterpolator {
    let build = |sim_time: f64| {
      let mut frame = SampleFrame::new(sim_time);
      let temps: Vec<f64> = (0..mesh.cell_count())
        .map(|i| {
          let g = GeoCoord::from_world(
            &mesh.cell_world_centroid(CellId::from(i)),
            SURFACE,
          );
          g.latitude_deg()
        })
        .collect();
      frame.insert_quantity(MeshChannel::AtmosphereTemperature, temps);
      frame
    };
    let mut fi = FrameInterpolator::new();
    fi.push(build(0.0));
    fi.push(build(1.0));
    // Land the clock between the two frames so results are Ok, not Stale.
    fi.advance(0.5);
    fi
  }

  #[test]
  fn scalar_sample_locates_and_reads_the_right_cell() {
    let (tessera, mesh) = world_with_atmosphere();
    let query = WorldQuery::new(&tessera, SURFACE);
    let snapshot = primed_snapshot(&mesh);

    let at = GeoCoord::from_degrees(20.0, 50.0, 5000.0);
    let result =
      query.sample_scalar(&snapshot, ScalarQuantity::Temperature, at);
    // The channel encodes latitude-in-degrees, so the sampled value must match
    // the latitude of the located cell's centroid (within one cell's span).
    let cell = query.indices[&MeshType::Atmosphere].locate(&at).unwrap();
    let truth = GeoCoord::from_world(&mesh.cell_world_centroid(cell), SURFACE)
      .latitude_deg();
    assert!(matches!(result, Sample::Ok(_) | Sample::Stale(_)));
    assert!((result.value().unwrap() - truth).abs() < 1e-9);
  }

  #[test]
  fn unknown_quantity_or_offmesh_is_unavailable() {
    let (tessera, mesh) = world_with_atmosphere();
    let query = WorldQuery::new(&tessera, SURFACE);
    let snapshot = primed_snapshot(&mesh);
    // Humidity channel was never inserted into the snapshot.
    let r = query.sample_scalar(
      &snapshot,
      ScalarQuantity::Humidity,
      GeoCoord::from_degrees(0.0, 0.0, 0.0),
    );
    assert_eq!(r, Sample::Unavailable);
    // No ocean mesh registered.
    let r2 = query.sample_scalar(
      &snapshot,
      ScalarQuantity::SeaSurfaceTemperature,
      GeoCoord::from_degrees(0.0, 0.0, 0.0),
    );
    assert_eq!(r2, Sample::Unavailable);
  }

  #[test]
  fn degraded_value_is_surfaced_not_hidden() {
    let (tessera, mesh) = world_with_atmosphere();
    let query = WorldQuery::new(&tessera, SURFACE);
    let mut frame = SampleFrame::new(0.0);
    let mut temps = vec![300.0; mesh.cell_count()];
    let at = GeoCoord::from_degrees(0.0, 0.0, 0.0);
    let cell = query.indices[&MeshType::Atmosphere].locate(&at).unwrap();
    temps[cell.index()] = f64::NAN;
    frame.insert_quantity(MeshChannel::AtmosphereTemperature, temps);
    let mut fi = FrameInterpolator::new();
    fi.push(frame);
    let r = query.sample_scalar(&fi, ScalarQuantity::Temperature, at);
    assert!(matches!(r, Sample::Degraded(_)));
  }

  #[test]
  fn world_frame_wind_projects_to_correct_enu() {
    let (tessera, mesh) = world_with_atmosphere();
    let query = WorldQuery::new(&tessera, SURFACE);
    // A constant world-frame velocity of 10 m/s along +z (the north polar
    // axis). Its ENU decomposition at latitude φ is exact and independent of
    // which cell is located: east = 0, north = 10·cosφ, up = 10·sinφ.
    let n = mesh.cell_count();
    let mut frame = SampleFrame::new(0.0);
    frame.insert_quantity(MeshChannel::AtmosphereWindX, vec![0.0; n]);
    frame.insert_quantity(MeshChannel::AtmosphereWindY, vec![0.0; n]);
    frame.insert_quantity(MeshChannel::AtmosphereWindZ, vec![10.0; n]);
    let mut fi = FrameInterpolator::new();
    fi.push(frame);

    let lat_deg = 15.0;
    let at = GeoCoord::from_degrees(lat_deg, 65.0, 5000.0);
    let wind = query.sample_wind(&fi, at).value().unwrap();
    let phi = lat_deg.to_radians();
    assert!(wind[0].abs() < 1e-9, "east {}", wind[0]);
    assert!(
      (wind[1] - 10.0 * phi.cos()).abs() < 1e-9,
      "north {}",
      wind[1]
    );
    assert!((wind[2] - 10.0 * phi.sin()).abs() < 1e-9, "up {}", wind[2]);
  }

  #[test]
  fn regional_mean_matches_brute_force() {
    let (tessera, mesh) = world_with_atmosphere();
    let query = WorldQuery::new(&tessera, SURFACE);
    let snapshot = primed_snapshot(&mesh);
    let bounds = GeoBounds::from_degrees(
      (-20.0, 20.0),
      (-30.0, 30.0),
      (-1.0, R_OUTER - R_INNER + 1.0),
    );
    let mean = query.reduce_scalar(
      &snapshot,
      ScalarQuantity::Temperature,
      &bounds,
      Reduction::Mean,
    );

    // Brute-force: the channel is latitude-deg, so mean over in-region cells.
    let cells = query.indices[&MeshType::Atmosphere].cells_in(&bounds);
    let truth: f64 = cells
      .iter()
      .map(|&c| {
        GeoCoord::from_world(&mesh.cell_world_centroid(c), SURFACE)
          .latitude_deg()
      })
      .sum::<f64>()
      / cells.len() as f64;
    assert!((mean.value().unwrap() - truth).abs() < 1e-9);
  }
}
