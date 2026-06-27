use crate::maths::vector::Vector;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum BoundaryTag {
  Top,
  Left,
  Right,
  Bottom,
  Front,
  Back,

  Wall,
  Ground,
  Inflow,
  Outflow,
  AtmosphereEdge,
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Copy, Hash)]
pub enum Axis {
  X = 0,
  Y = 1,
  Z = 2,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct CellId(usize);

impl CellId {
  pub fn index(&self) -> usize {
    self.0
  }
}

impl From<usize> for CellId {
  fn from(value: usize) -> Self {
    CellId(value)
  }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct FaceId(usize);

impl FaceId {
  pub fn index(&self) -> usize {
    self.0
  }
}

impl From<usize> for FaceId {
  fn from(value: usize) -> Self {
    FaceId(value)
  }
}

pub type Point<const D: usize> = Vector<f64, D>;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct WorldId(pub usize);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SystemId(pub usize);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FieldName {
  Temperature,
  TemperatureTendency,
  EulerState,
  VelocityX,
  VelocityY,
  VelocityZ,
  Pressure,
  Humidity,
  RadiativeHeatingTendency,
  NetSurfaceFlux,
  /// Upward water-vapour mass flux (kg/m²/s) from the sea surface into the
  /// atmosphere — the source side of the hydrological cycle.
  EvaporationFlux,
  /// Downward liquid-water mass flux (kg/m²/s) condensing out of the
  /// atmosphere and returning to the surface/ocean.
  PrecipitationFlux,
  /// Sea-surface temperature (K) mapped onto the atmosphere mesh from the
  /// ocean's top layer, so the air–sea evaporation flux can read it as a
  /// field on its own mesh.
  SeaSurfaceTemperature,
  /// Static surface elevation (m) relative to the body's mean surface radius:
  /// positive is land above the datum, negative is ocean floor / basin depth.
  /// Inert terrain data — set once at world setup, not evolved by physics.
  SurfaceElevation,
  /// Static categorical surface classification (ocean / land / ice), stored as
  /// a numeric code (see `terra::SurfaceClass`). Inert terrain data.
  SurfaceType,
  /// Per-cell short-wave surface albedo (0..1). The reusable contract for
  /// surface brightness: derived from the surface type / coverage by one or
  /// more producers (terrain base, then ice / snow), and read by radiation.
  SurfaceAlbedo,
  /// Slowly-varying time-mean (climatology) of air temperature (K). Written by
  /// the chronos climatology accumulator, not by physics — the aggregate field
  /// a slow consumer reads while game-time advances by holding the live state.
  MeanTemperature,
  /// Slowly-varying time-mean (climatology) of air pressure (Pa).
  MeanPressure,
  /// Slowly-varying time-mean (climatology) of specific humidity (kg/kg).
  MeanHumidity,
  /// Slowly-varying time-mean (climatology) of sea-surface temperature (K).
  MeanSeaSurfaceTemperature,
}

/// Categorical surface classification — the semantic meaning of the numeric
/// codes stored in the [`FieldName::SurfaceType`] field. Lives here in the
/// shared vocabulary so both the producer (terra) and consumers (the query
/// API) agree on the encoding without depending on each other.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum SurfaceClass {
  Ocean,
  Land,
  Ice,
}

impl SurfaceClass {
  /// The numeric code stored in the `SurfaceType` field.
  pub fn code(self) -> f64 {
    match self {
      SurfaceClass::Ocean => 0.0,
      SurfaceClass::Land => 1.0,
      SurfaceClass::Ice => 2.0,
    }
  }

  /// Recover a class from a stored (possibly interpolated) code. Values round
  /// to the nearest class; anything unrecognised falls back to Ocean.
  pub fn from_code(code: f64) -> Self {
    match code.round() as i64 {
      1 => SurfaceClass::Land,
      2 => SurfaceClass::Ice,
      _ => SurfaceClass::Ocean,
    }
  }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum MeshType {
  Atmosphere,
  Surface,
  Mantle,
  /// Liquid water column beneath the surface skin (a radial stack of ocean
  /// layers). Owns sea temperature / heat content; the top layer is the
  /// sea surface that exchanges heat and water with the atmosphere.
  Ocean,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct MeshKey(MeshType);

impl MeshKey {
  pub const ATMOSPHERE: MeshKey = MeshKey(MeshType::Atmosphere);
  pub const SURFACE: MeshKey = MeshKey(MeshType::Surface);
  pub const MANTLE: MeshKey = MeshKey(MeshType::Mantle);
  pub const OCEAN: MeshKey = MeshKey(MeshType::Ocean);

  pub const fn new(mesh_type: MeshType) -> Self {
    MeshKey(mesh_type)
  }

  pub const fn mesh_type(self) -> MeshType {
    self.0
  }
}

/// Identifier for a typed singleton in pleroma that isn't bound to a mesh.
/// Used for things like orbital body state, sun direction, planetary spin —
/// data physics stages need but that doesn't live per-cell on a mesh.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ResourceKey {
  Bodies,
  SunPosition,
  PlanetSpin,
  /// Climatology-regime transition state read by the chronos nudge stage: the
  /// current relaxation fraction (0 = inert) pulling live fields toward their
  /// climatology means during a live↔climatology handoff. Absent means no
  /// transition is in progress (the nudge is a no-op).
  ClimateRegime,
  /// Aggregate runtime health report (`WorldDiagnostics`): per-field finiteness
  /// and conservation-drift findings published by in-DAG monitor stages and
  /// read at the `World` level. Also carries the active `DiagnosticsPolicy` so
  /// monitor stages know whether to merely observe, warn, or fail the tick.
  Diagnostics,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FieldKey {
  mesh: MeshKey,
  name: FieldName,
}

impl FieldKey {
  pub const fn new(mesh: MeshKey, name: FieldName) -> Self {
    Self { mesh, name }
  }

  pub const fn mesh(self) -> MeshKey {
    self.mesh
  }

  pub const fn name(self) -> FieldName {
    self.name
  }
}
