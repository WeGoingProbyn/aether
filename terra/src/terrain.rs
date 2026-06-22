// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Inert terrain state: a static surface heightfield and a categorical
//! land / ocean / ice classification over a surface mesh.
//!
//! This is the first, deliberately *inert* slice of Pillar 3 — terrain becomes
//! real, queryable, renderable state with **no physics coupling yet**. Like the
//! rest of terra, it owns only the *logic* of populating the fields at world
//! setup; the storage lives in pleroma. Couplings (orographic lift, albedo,
//! drainage) are wired in later, one at a time, so any terrain-induced solver
//! instability stays isolated.

use nexus::{FieldKey, FieldName, MeshKey, Pleroma, SoaField};
use tessera::geo::GeoCoord;
use tessera::mesh::Mesh;
use utility::{
  domain::{CellId, SurfaceClass},
  error::AetherResult,
};

/// One cell's terrain: elevation in metres relative to the mean surface radius
/// (positive = land above datum, negative = basin/ocean depth) and its class.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainSample {
  pub elevation: f64,
  pub class: SurfaceClass,
}

/// The `FieldKey`s terrain occupies on its mesh.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TerrainFields {
  pub elevation: FieldKey,
  pub surface_type: FieldKey,
}

impl TerrainFields {
  pub const fn for_mesh(mesh: MeshKey) -> Self {
    Self {
      elevation: FieldKey::new(mesh, FieldName::SurfaceElevation),
      surface_type: FieldKey::new(mesh, FieldName::SurfaceType),
    }
  }
}

/// Registers and initialises the inert terrain fields on a surface mesh.
pub struct TerrainModel {
  mesh: MeshKey,
  fields: TerrainFields,
  surface_radius: f64,
}

impl TerrainModel {
  /// `surface_radius` is the body's mean surface radius, used to turn each
  /// cell's world centroid into the geographic coordinate handed to the
  /// generator.
  pub fn new(mesh: MeshKey, surface_radius: f64) -> Self {
    Self {
      mesh,
      fields: TerrainFields::for_mesh(mesh),
      surface_radius,
    }
  }

  pub fn mesh(&self) -> MeshKey {
    self.mesh
  }

  pub fn fields(&self) -> TerrainFields {
    self.fields
  }

  /// Populate elevation and surface-type from a geographic generator and
  /// register them in pleroma. No stages are added — terrain is static.
  pub fn register_fields<M, G>(
    &self,
    pleroma: &mut Pleroma,
    mesh: &M,
    generator: G,
  ) -> AetherResult<()>
  where
    M: Mesh<3> + ?Sized,
    G: Fn(GeoCoord) -> TerrainSample,
  {
    let n = mesh.cell_count();
    let samples: Vec<TerrainSample> = (0..n)
      .map(|i| {
        let pos = mesh.cell_world_centroid(CellId::from(i));
        let geo = GeoCoord::from_world(&pos, self.surface_radius);
        generator(geo)
      })
      .collect();

    let elevation =
      SoaField::<1>::from_fn(n, |i| [samples[i.index()].elevation]);
    let surface_type =
      SoaField::<1>::from_fn(n, |i| [samples[i.index()].class.code()]);
    pleroma.register_field(self.fields.elevation, elevation);
    pleroma.register_field(self.fields.surface_type, surface_type);
    Ok(())
  }
}

/// A simple deterministic Earth-like generator, useful as a default and for
/// tests. Ice poleward of ~70°; a smooth low-order spherical-harmonic-ish
/// function carves rough continents (positive ⇒ land) from ocean basins
/// (negative ⇒ depth). No randomness, so worlds are reproducible.
pub fn earthlike_terrain(geo: GeoCoord) -> TerrainSample {
  let polar = 70_f64.to_radians();
  if geo.lat.abs() > polar {
    return TerrainSample {
      elevation: 50.0,
      class: SurfaceClass::Ice,
    };
  }
  // Continent field in roughly [-1, 1].
  let c = 0.5 * (2.0 * geo.lon).sin() + 0.5 * (3.0 * geo.lat).cos();
  if c > 0.0 {
    TerrainSample {
      elevation: c * 2000.0,
      class: SurfaceClass::Land,
    }
  } else {
    TerrainSample {
      elevation: c * 4000.0,
      class: SurfaceClass::Ocean,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use nexus::FieldStorage;
  use std::sync::Arc;
  use tessera::cube_sphere::CubeSphere;
  use tessera::geometry::CellGeometry;

  const R_INNER: f64 = 6.371e6;
  const R_OUTER: f64 = 6.381e6;

  #[test]
  fn registers_finite_terrain_with_valid_classes() {
    let mesh = Arc::new(CubeSphere::new([12, 12, 1], R_INNER, R_OUTER));
    let mut pleroma = Pleroma::new();
    let model = TerrainModel::new(MeshKey::SURFACE, R_INNER);
    model
      .register_fields(&mut pleroma, mesh.as_ref(), earthlike_terrain)
      .unwrap();

    let elevation: &SoaField<1> =
      pleroma.read(model.fields().elevation).unwrap();
    let surface_type: &SoaField<1> =
      pleroma.read(model.fields().surface_type).unwrap();
    assert_eq!(elevation.len(), mesh.cell_count());

    let mut saw_land = false;
    let mut saw_ocean = false;
    for i in 0..mesh.cell_count() {
      let cell = CellId::from(i);
      let e = elevation.state(cell)[0];
      assert!(e.is_finite(), "elevation {e} not finite");
      // Code must round-trip to a real class.
      let code = surface_type.state(cell)[0];
      match SurfaceClass::from_code(code) {
        SurfaceClass::Land => {
          saw_land = true;
          assert!(e >= 0.0, "land should be at/above datum, got {e}");
        }
        SurfaceClass::Ocean => {
          saw_ocean = true;
          assert!(e <= 0.0, "ocean should be at/below datum, got {e}");
        }
        SurfaceClass::Ice => assert!(e.is_finite()),
      }
    }
    assert!(
      saw_land && saw_ocean,
      "generator should produce land and ocean"
    );
  }

  #[test]
  fn surface_class_code_round_trips() {
    for class in [SurfaceClass::Ocean, SurfaceClass::Land, SurfaceClass::Ice] {
      assert_eq!(SurfaceClass::from_code(class.code()), class);
    }
    // Interpolated / noisy codes snap to the nearest class.
    assert_eq!(SurfaceClass::from_code(0.9), SurfaceClass::Land);
    assert_eq!(SurfaceClass::from_code(2.4), SurfaceClass::Ice);
  }
}
