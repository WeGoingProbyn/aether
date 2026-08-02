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

/// Monotonic version stamp for a mesh's topology. Bumped once each time the mesh
/// is adapted (refined / coarsened), so a consumer can tell that the dense
/// [`CellId`] space has changed underneath it. Base meshes start at
/// [`TopologyEpoch::ZERO`]. The keystone of the AMR identity contract: dense
/// `CellId` stays the hot-path key, and the epoch (plus a [`CellRemap`]) is how
/// everyone else detects and survives a re-mesh.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct TopologyEpoch(pub u64);

impl TopologyEpoch {
  pub const ZERO: TopologyEpoch = TopologyEpoch(0);

  /// The next epoch after this one.
  pub fn next(self) -> Self {
    TopologyEpoch(self.0 + 1)
  }
}

/// Where a cell in the *new* (post-adapt) mesh draws its initial state from when
/// fields are remapped across a topology change. The variant selects the
/// state-transfer rule, so a single per-new-cell value drives the whole remap:
///
/// - [`Survivor`](NewCellSource::Survivor): unchanged cell — copy the old value.
/// - [`Child`](NewCellSource::Child): a freshly-created refinement child —
///   *prolong* by inheriting the parent's value (piecewise-constant, which is
///   conservative for cell-averaged states).
/// - [`Merge`](NewCellSource::Merge): a coarsened cell replacing several old
///   children — *restrict* by volume-averaging the children.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NewCellSource {
  Survivor(CellId),
  Child { parent: CellId },
  Merge { children: Vec<CellId> },
}

/// The old↔new [`CellId`] correspondence produced by one mesh adapt, encoding
/// **birth and death explicitly in both directions** so consumers can both
/// follow surviving cells and correctly initialise new ones:
///
/// - `old → new` ([`image_of`](CellRemap::image_of)): `None` ⇒ the old cell
///   *died* (it was refined or merged away and has no single image).
/// - `new → source` ([`source_of`](CellRemap::source_of)): a [`NewCellSource`]
///   that is also the field-remap rule and answers "where did this new cell come
///   from" (a survivor has an old self; a child/merge does not).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellRemap {
  old_epoch: TopologyEpoch,
  new_epoch: TopologyEpoch,
  old_to_new: Vec<Option<CellId>>,
  new_sources: Vec<NewCellSource>,
}

impl CellRemap {
  /// Build a remap. `old_to_new` is indexed by old [`CellId`] (length = old cell
  /// count); `new_sources` is indexed by new [`CellId`] (length = new cell
  /// count). Debug builds assert the two directions reference in-range ids.
  pub fn new(
    old_epoch: TopologyEpoch,
    new_epoch: TopologyEpoch,
    old_to_new: Vec<Option<CellId>>,
    new_sources: Vec<NewCellSource>,
  ) -> Self {
    let old_count = old_to_new.len();
    let new_count = new_sources.len();
    if cfg!(debug_assertions) {
      for image in old_to_new.iter().flatten() {
        debug_assert!(image.index() < new_count, "remap image out of range");
      }
      for src in &new_sources {
        match src {
          NewCellSource::Survivor(c) | NewCellSource::Child { parent: c } => {
            debug_assert!(c.index() < old_count, "remap source out of range");
          }
          NewCellSource::Merge { children } => {
            for c in children {
              debug_assert!(
                c.index() < old_count,
                "remap merge child out of range"
              );
            }
          }
        }
      }
    }
    Self {
      old_epoch,
      new_epoch,
      old_to_new,
      new_sources,
    }
  }

  /// An identity remap of `count` cells: every cell survives to itself and the
  /// epoch is unchanged. The verified no-op used as the "no adaptation" case.
  pub fn identity(epoch: TopologyEpoch, count: usize) -> Self {
    Self {
      old_epoch: epoch,
      new_epoch: epoch,
      old_to_new: (0..count).map(|i| Some(CellId::from(i))).collect(),
      new_sources: (0..count)
        .map(|i| NewCellSource::Survivor(CellId::from(i)))
        .collect(),
    }
  }

  pub fn old_epoch(&self) -> TopologyEpoch {
    self.old_epoch
  }

  pub fn new_epoch(&self) -> TopologyEpoch {
    self.new_epoch
  }

  pub fn old_count(&self) -> usize {
    self.old_to_new.len()
  }

  pub fn new_count(&self) -> usize {
    self.new_sources.len()
  }

  /// The new-mesh image of an old cell, or `None` if it died.
  pub fn image_of(&self, old: CellId) -> Option<CellId> {
    self.old_to_new.get(old.index()).copied().flatten()
  }

  /// The source rule for a new cell (panics if out of range — a new cell id must
  /// be valid for this remap).
  pub fn source_of(&self, new: CellId) -> &NewCellSource {
    &self.new_sources[new.index()]
  }

  /// Whether a new cell has no single old self (a refinement child or a merge),
  /// so consumers know it needs initialising rather than carrying over.
  pub fn is_newborn(&self, new: CellId) -> bool {
    !matches!(self.source_of(new), NewCellSource::Survivor(_))
  }

  /// Old cells that died (no new image).
  pub fn died(&self) -> impl Iterator<Item = CellId> + '_ {
    self
      .old_to_new
      .iter()
      .enumerate()
      .filter_map(|(i, image)| image.is_none().then(|| CellId::from(i)))
  }

  /// New cells that were born this adapt (children / merges).
  pub fn born(&self) -> impl Iterator<Item = CellId> + '_ {
    self.new_sources.iter().enumerate().filter_map(|(i, src)| {
      (!matches!(src, NewCellSource::Survivor(_))).then(|| CellId::from(i))
    })
  }

  /// The per-new-cell source rules, in new-[`CellId`] order.
  pub fn new_sources(&self) -> &[NewCellSource] {
    &self.new_sources
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
  /// Per-cell open-water fraction (0..1) gating air–sea moisture exchange:
  /// `1` = open ocean (full evaporation), `0` = dry land. The reusable contract
  /// for the moisture half of land–sea masking — derived from the surface
  /// land/sea class and read by the evaporation stage. Inert/static in v1.
  MoistureAvailability,
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
  /// Deferred-dispatch event channel (`EventBus`): in-DAG stages and the `World`
  /// emit `Event`s during a tick; the buffer is published at the end-of-tick
  /// barrier and read via `World::events()`. Ephemeral / not checkpointed.
  Events,
  /// Adaptive-refinement region of interest ([`RefinementFocus`]): the host
  /// writes a world-space point it wants resolved finely; a focus-driven
  /// refinement criterion reads it to refine nearby and coarsen far away. An
  /// *inbound* resource (host→sim), the mirror of the outbound
  /// [`ResourceKey::Events`]. Ephemeral / not checkpointed.
  RefinementFocus,
}

/// A world-space point the host wants resolved finely, stored as the inbound
/// [`ResourceKey::RefinementFocus`] resource.
///
/// Deliberately *not* a camera. A host that renders will usually derive this
/// from its view, but the simulation has no business knowing that: all it needs
/// is "resolve detail near here", which is the same request whether it comes
/// from a camera, a player position, a probe, or a scripted region of interest.
/// Keeping the sim's vocabulary free of rendering concepts is what lets the
/// renderer stay a pure downstream consumer.
///
/// Lives in `utility` (the shared-type discipline used by [`SurfaceClass`]) so
/// aether can read it without depending on any particular host. Only a
/// world-space position is needed for distance-based level-of-detail; richer
/// data (a direction, a frustum, a falloff radius) can be added without
/// changing the seam.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RefinementFocus {
  /// World-space point to resolve finely.
  pub position: [f64; 3],
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
