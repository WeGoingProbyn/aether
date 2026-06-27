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

use nexus::{
  FieldKey, FieldName, FieldStorage, MeshKey, Nexus, Pleroma, SoaField, Stage,
  StageContext, StageId,
};
use tessera::geo::GeoCoord;
use tessera::mesh::Mesh;
use utility::{
  domain::{CellId, SurfaceClass},
  error::{AetherError, AetherResult},
};

use crate::TerraError;

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
  pub albedo: FieldKey,
}

impl TerrainFields {
  pub const fn for_mesh(mesh: MeshKey) -> Self {
    Self {
      elevation: FieldKey::new(mesh, FieldName::SurfaceElevation),
      surface_type: FieldKey::new(mesh, FieldName::SurfaceType),
      albedo: FieldKey::new(mesh, FieldName::SurfaceAlbedo),
    }
  }
}

/// Per-class short-wave albedo — the *base layer* of the composable
/// surface-albedo contract. Radiation never reads this directly; it reads the
/// per-cell `SurfaceAlbedo` field, which a producer fills from this table (and
/// which a future ice / snow model blends on top of).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlbedoTable {
  pub ocean: f64,
  pub land: f64,
  pub ice: f64,
}

impl Default for AlbedoTable {
  fn default() -> Self {
    // Earth-ish: open ocean is dark, vegetated/bare land mid, ice bright.
    Self {
      ocean: 0.06,
      land: 0.30,
      ice: 0.60,
    }
  }
}

impl AlbedoTable {
  pub fn albedo_for(&self, class: SurfaceClass) -> f64 {
    match class {
      SurfaceClass::Ocean => self.ocean,
      SurfaceClass::Land => self.land,
      SurfaceClass::Ice => self.ice,
    }
  }
}

/// Per-class open-water fraction — the base layer of the composable
/// [`FieldName::MoistureAvailability`] contract that gates air–sea evaporation.
/// `1` = open water (full evaporation), `0` = dry. Like [`AlbedoTable`], a
/// consumer (the evaporation stage) reads the per-cell field, not this table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AvailabilityTable {
  pub ocean: f64,
  pub land: f64,
  pub ice: f64,
}

impl Default for AvailabilityTable {
  fn default() -> Self {
    // Open ocean evaporates fully; land and (frozen) ice do not. Sublimation
    // over ice is ignored in v1.
    Self {
      ocean: 1.0,
      land: 0.0,
      ice: 0.0,
    }
  }
}

impl AvailabilityTable {
  pub fn availability_for(&self, class: SurfaceClass) -> f64 {
    match class {
      SurfaceClass::Ocean => self.ocean,
      SurfaceClass::Land => self.land,
      SurfaceClass::Ice => self.ice,
    }
  }
}

/// Register and initialise the inert [`FieldName::MoistureAvailability`] field on
/// `mesh` from a geographic land/sea classifier, using the default
/// [`AvailabilityTable`]. The value at each cell is the open-water fraction of the
/// surface class directly below it (`generator(geo).class`), so registering this on
/// the **atmosphere** mesh gives the evaporation stage a same-mesh moisture gate —
/// the vertical projection of the land/sea mask. Returns the field's key.
///
/// This is the "surface property as a field" producer for moisture, mirroring the
/// terrain albedo path; inert/static in v1, but the field seam supports a future
/// dynamic (soil-moisture) producer.
pub fn register_moisture_availability<M, G>(
  pleroma: &mut Pleroma,
  mesh_key: MeshKey,
  mesh: &M,
  surface_radius: f64,
  generator: G,
) -> FieldKey
where
  M: Mesh<3> + ?Sized,
  G: Fn(GeoCoord) -> TerrainSample,
{
  register_moisture_availability_with_table(
    pleroma,
    mesh_key,
    mesh,
    surface_radius,
    AvailabilityTable::default(),
    generator,
  )
}

/// As [`register_moisture_availability`] but with an explicit [`AvailabilityTable`].
pub fn register_moisture_availability_with_table<M, G>(
  pleroma: &mut Pleroma,
  mesh_key: MeshKey,
  mesh: &M,
  surface_radius: f64,
  table: AvailabilityTable,
  generator: G,
) -> FieldKey
where
  M: Mesh<3> + ?Sized,
  G: Fn(GeoCoord) -> TerrainSample,
{
  let n = mesh.cell_count();
  let availability: Vec<f64> = (0..n)
    .map(|i| {
      let pos = mesh.cell_world_centroid(CellId::from(i));
      let geo = GeoCoord::from_world(&pos, surface_radius);
      table.availability_for(generator(geo).class)
    })
    .collect();
  let key = FieldKey::new(mesh_key, FieldName::MoistureAvailability);
  pleroma.register_field(
    key,
    SoaField::<1>::from_fn(n, |i| [availability[i.index()]]),
  );
  key
}

/// Registers and initialises the inert terrain fields on a surface mesh,
/// including the base per-cell surface albedo.
pub struct TerrainModel {
  mesh: MeshKey,
  fields: TerrainFields,
  surface_radius: f64,
  albedo_table: AlbedoTable,
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
      albedo_table: AlbedoTable::default(),
    }
  }

  pub fn with_albedo_table(mut self, table: AlbedoTable) -> Self {
    self.albedo_table = table;
    self
  }

  pub fn mesh(&self) -> MeshKey {
    self.mesh
  }

  pub fn fields(&self) -> TerrainFields {
    self.fields
  }

  pub fn albedo_table(&self) -> AlbedoTable {
    self.albedo_table
  }

  /// Populate elevation, surface-type, and the base surface albedo from a
  /// geographic generator and register them in pleroma. The albedo is
  /// initialised here (from the surface class) so it is immediately valid as
  /// inert data; for a *dynamic* surface (sea ice that forms and melts) add
  /// [`TerrainModel::add_stages`] so the base is re-derived each tick before a
  /// future ice producer blends on top.
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
    let albedo = SoaField::<1>::from_fn(n, |i| {
      [self.albedo_table.albedo_for(samples[i.index()].class)]
    });
    pleroma.register_field(self.fields.elevation, elevation);
    pleroma.register_field(self.fields.surface_type, surface_type);
    pleroma.register_field(self.fields.albedo, albedo);
    Ok(())
  }

  /// Add the composable base-albedo producer: a stage that re-derives the
  /// per-cell `SurfaceAlbedo` from the (possibly changing) `SurfaceType` each
  /// tick. Optional for a purely static terrain — the field is already set by
  /// [`TerrainModel::register_fields`] — but it is the seam dynamic producers
  /// build on (ice / snow blend after it).
  pub fn add_stages(&self, nexus: &mut Nexus) -> StageId {
    nexus.add(SurfaceAlbedoStep::new(self.mesh, self.albedo_table))
  }
}

/// Composable base-albedo producer: writes the per-cell `SurfaceAlbedo` from the
/// categorical `SurfaceType` via an [`AlbedoTable`]. It overwrites (establishes
/// the base layer); later producers in the DAG blend ice / snow on top.
pub struct SurfaceAlbedoStep {
  surface_type: FieldKey,
  albedo: FieldKey,
  table: AlbedoTable,
  reads: [FieldKey; 1],
  writes: [FieldKey; 1],
}

impl SurfaceAlbedoStep {
  pub fn new(mesh: MeshKey, table: AlbedoTable) -> Self {
    let surface_type = FieldKey::new(mesh, FieldName::SurfaceType);
    let albedo = FieldKey::new(mesh, FieldName::SurfaceAlbedo);
    Self {
      surface_type,
      albedo,
      table,
      reads: [surface_type],
      writes: [albedo],
    }
  }
}

impl Stage for SurfaceAlbedoStep {
  fn name(&self) -> &'static str {
    "terra_surface_albedo"
  }

  fn reads(&self) -> &[FieldKey] {
    &self.reads
  }

  fn writes(&self) -> &[FieldKey] {
    &self.writes
  }

  fn run(&mut self, mut ctx: StageContext<'_>) -> AetherResult<()> {
    let values: Vec<f64> = {
      let surface_type: &SoaField<1> =
        ctx.world.fields.read(self.surface_type).ok_or_else(|| {
          AetherError::new(TerraError::MissingReadField)
            .context(format!("{:?}", self.surface_type))
        })?;
      (0..surface_type.len())
        .map(|i| {
          let code = surface_type.state(CellId::from(i))[0];
          self.table.albedo_for(SurfaceClass::from_code(code))
        })
        .collect()
    };

    let albedo: &mut SoaField<1> =
      ctx.world.fields.write(self.albedo).ok_or_else(|| {
        AetherError::new(TerraError::MissingWriteField)
          .context(format!("{:?}", self.albedo))
      })?;
    for (i, &a) in values.iter().enumerate() {
      albedo.write(CellId::from(i), &[a]);
    }
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
  fn moisture_availability_matches_surface_class() {
    let mesh = Arc::new(CubeSphere::new([12, 12, 1], R_INNER, R_OUTER));
    let mut pleroma = Pleroma::new();
    let key = register_moisture_availability(
      &mut pleroma,
      MeshKey::ATMOSPHERE,
      mesh.as_ref(),
      R_INNER,
      earthlike_terrain,
    );

    let field: &SoaField<1> = pleroma.read(key).unwrap();
    let table = AvailabilityTable::default();
    assert_eq!(field.len(), mesh.cell_count());

    let mut saw_ocean = false;
    let mut saw_dry = false;
    for i in 0..mesh.cell_count() {
      let cell = CellId::from(i);
      let geo = GeoCoord::from_world(&mesh.cell_world_centroid(cell), R_INNER);
      let class = earthlike_terrain(geo).class;
      let a = field.state(cell)[0];
      assert!(
        a.is_finite() && (0.0..=1.0).contains(&a),
        "availability {a}"
      );
      assert!((a - table.availability_for(class)).abs() < 1e-12);
      match class {
        SurfaceClass::Ocean => {
          saw_ocean = true;
          assert_eq!(a, 1.0);
        }
        SurfaceClass::Land | SurfaceClass::Ice => {
          saw_dry = true;
          assert_eq!(a, 0.0);
        }
      }
    }
    assert!(
      saw_ocean && saw_dry,
      "generator should yield ocean and dry cells"
    );
  }

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
  fn register_fields_sets_base_albedo_from_class() {
    let mesh = Arc::new(CubeSphere::new([12, 12, 1], R_INNER, R_OUTER));
    let mut pleroma = Pleroma::new();
    let table = AlbedoTable::default();
    let model = TerrainModel::new(MeshKey::SURFACE, R_INNER);
    model
      .register_fields(&mut pleroma, mesh.as_ref(), earthlike_terrain)
      .unwrap();

    let surface_type: &SoaField<1> =
      pleroma.read(model.fields().surface_type).unwrap();
    let albedo: &SoaField<1> = pleroma.read(model.fields().albedo).unwrap();
    for i in 0..mesh.cell_count() {
      let cell = CellId::from(i);
      let class = SurfaceClass::from_code(surface_type.state(cell)[0]);
      let a = albedo.state(cell)[0];
      assert!((a - table.albedo_for(class)).abs() < 1e-12);
      assert!((0.0..=1.0).contains(&a), "albedo {a} out of range");
    }
  }

  #[test]
  fn albedo_table_orders_ocean_land_ice() {
    let t = AlbedoTable::default();
    assert!(
      t.albedo_for(SurfaceClass::Ocean) < t.albedo_for(SurfaceClass::Land)
    );
    assert!(t.albedo_for(SurfaceClass::Land) < t.albedo_for(SurfaceClass::Ice));
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
