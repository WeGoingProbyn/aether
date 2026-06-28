use std::collections::HashMap;
use std::sync::Arc;

use chronos::{Regime, RegimeConfig, TransitionKind, TransitionState};
use cosmo::kind::{BodyKind, CelestialBody};
use nexus::{
  AtmosphereConstants, CompiledNexus, RadiationConstants, ResourceKey,
  ScheduledStageTask, StageTask, WorldConstants, WorldId,
};
use pleroma::Pleroma;
use pleroma::prelude::PleromaCheckpoint;
use tessera::adaptive::AdaptiveMesh;
use tessera::geometry::CellGeometry;
use tessera::refine::balance_2to1;
use tessera::world_mesh::Tessera;
use utility::{
  constants::solar_flux,
  diagnostics::{DiagnosticsPolicy, WorldDiagnostics},
  domain::{CellId, MeshKey, MeshType, SystemId, TopologyEpoch},
  error::{AetherError, AetherResult, ErrorDomain},
  events::{Event, EventBus, RegimeKind},
  profile,
  serial::deserialize::Deserialize,
  serial::json::{JsonDeserializer, JsonSerializer},
  serial::serialize::Serialize,
  thread::pool::{Pool, ScopedTaskGraph},
};

/// Default surface long-wave emissivity for rocky bodies. Cosmo doesn't
/// carry an emissivity yet, so we use a single conservative value rather
/// than per-body data.
const DEFAULT_SURFACE_EMISSIVITY: f64 = 0.95;
/// Fallback surface short-wave albedo when cosmo has no atmosphere /
/// albedo for the body (e.g. airless rocky body). Earth-like neutral.
const DEFAULT_SURFACE_ALBEDO: f64 = 0.30;

/// Runtime state for one simulated body.
pub struct World {
  id: WorldId,
  seed: CelestialBody,
  primary: Option<CelestialBody>,
  constants: WorldConstants,
  tessera: Tessera,
  pleroma: Pleroma,
  nexus: CompiledNexus,
  partition_count: usize,
  /// Index into `BodyState<3>::positions` (a `ResourceKey::Bodies`
  /// resource on the system-level pleroma) identifying which orbital
  /// body this world tracks. `None` means the world is fixed at the
  /// origin. The eidolon producer uses this hint to read the world's
  /// centre per-tick.
  body_index: Option<usize>,
  /// Time-advance regime (live vs climatology burst-then-hold). Defaults to
  /// [`Regime::Live`], which reproduces the original `step`-driven behaviour.
  regime: Regime,
  /// Parameters for the climatology regime's burst-then-hold advance.
  regime_config: RegimeConfig,
  /// Game time advanced so far (s). In the live regime this equals `sim_time`;
  /// in the climatology regime it runs ahead, since held spans advance the game
  /// clock without integrating the solver.
  game_clock: f64,
  /// Simulation time actually integrated by the solver so far (s).
  sim_time: f64,
  /// An in-progress live↔climatology handoff, if any. While set, each
  /// integrated tick publishes the transition's relaxation fraction to
  /// `ResourceKey::ClimateRegime` so a chronos nudge stage can spin the live
  /// state up from / down to the climatology smoothly. Cleared when complete.
  transition: Option<TransitionState>,
  /// Per-mesh adaptive-refinement controllers, run at the end-of-tick barrier.
  /// Empty ⇒ no AMR (the default). See [`crate::adapt`].
  adapters: Vec<crate::adapt::MeshAdapter>,
  /// Successful ticks integrated so far — the adapt governors' cadence clock.
  tick_count: u64,
}

/// The serialised advance state of a transition handoff (the chronos
/// `TransitionState` rendered as primitives, since the serial derives only cover
/// named structs of serialisable fields).
#[derive(utility::Serialize, utility::Deserialize)]
pub struct TransitionRecord {
  /// `0 = ClimatologyToLive`, `1 = LiveToClimatology`.
  pub kind: u32,
  pub progress: f64,
  pub window: f64,
}

/// The adapted topology of one AMR mesh: its [`MeshType`] code, current epoch,
/// and the refinement forest's leaf codes. Persisted so a reload reconstructs the
/// *adapted* mesh before the (adapted-size) fields are loaded into it.
#[derive(utility::Serialize, utility::Deserialize)]
pub struct RefinementRecord {
  pub mesh: u32,
  pub epoch: u64,
  pub leaf_bases: Vec<u64>,
  pub leaf_paths: Vec<Vec<u32>>,
}

/// A full, restartable snapshot of one [`World`]: the integrated clocks and
/// advance mode, the adapted mesh topology of any AMR meshes, plus the entire
/// pleroma state. Reload an *identically assembled* world from it (same base
/// meshes + adapters, same registered fields) and resume bit-for-bit. The
/// compiled DAG and derived/transient resources are rebuilt by world assembly.
#[derive(utility::Serialize, utility::Deserialize)]
pub struct WorldCheckpoint {
  pub sim_time: f64,
  pub game_clock: f64,
  /// `0 = Live`, `1 = Climatology`.
  pub regime: u32,
  pub transition: Option<TransitionRecord>,
  /// Adapted topology per AMR mesh (empty for a non-adaptive world).
  pub refinements: Vec<RefinementRecord>,
  pub pleroma: PleromaCheckpoint,
}

#[derive(Debug)]
enum CheckpointError {
  UnknownRegime(u32),
  UnknownTransitionKind(u32),
  UnknownMeshType(u32),
}

impl ErrorDomain for CheckpointError {
  fn domain(&self) -> &str {
    "aether checkpoint"
  }
}

impl std::fmt::Display for CheckpointError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      CheckpointError::UnknownRegime(c) => {
        write!(f, "checkpoint has unknown regime code {c}")
      }
      CheckpointError::UnknownTransitionKind(c) => {
        write!(f, "checkpoint has unknown transition-kind code {c}")
      }
      CheckpointError::UnknownMeshType(c) => {
        write!(f, "checkpoint has unknown mesh-type code {c}")
      }
    }
  }
}

fn regime_code(regime: Regime) -> u32 {
  match regime {
    Regime::Live => 0,
    Regime::Climatology => 1,
  }
}

fn mesh_type_code(mesh_type: MeshType) -> u32 {
  match mesh_type {
    MeshType::Atmosphere => 0,
    MeshType::Surface => 1,
    MeshType::Mantle => 2,
    MeshType::Ocean => 3,
  }
}

fn mesh_type_from_code(code: u32) -> AetherResult<MeshType> {
  match code {
    0 => Ok(MeshType::Atmosphere),
    1 => Ok(MeshType::Surface),
    2 => Ok(MeshType::Mantle),
    3 => Ok(MeshType::Ocean),
    other => Err(AetherError::new(CheckpointError::UnknownMeshType(other))),
  }
}

/// Map the chronos advance regime onto the dependency-free event vocabulary, so
/// `utility::events` need not depend on `chronos`.
fn regime_kind(regime: Regime) -> RegimeKind {
  match regime {
    Regime::Live => RegimeKind::Live,
    Regime::Climatology => RegimeKind::Climatology,
  }
}

/// The settled regime a transition is heading toward.
fn transition_target(kind: TransitionKind) -> RegimeKind {
  match kind {
    TransitionKind::ClimatologyToLive => RegimeKind::Live,
    TransitionKind::LiveToClimatology => RegimeKind::Climatology,
  }
}

fn regime_from_code(code: u32) -> AetherResult<Regime> {
  match code {
    0 => Ok(Regime::Live),
    1 => Ok(Regime::Climatology),
    other => Err(AetherError::new(CheckpointError::UnknownRegime(other))),
  }
}

fn transition_kind_code(kind: TransitionKind) -> u32 {
  match kind {
    TransitionKind::ClimatologyToLive => 0,
    TransitionKind::LiveToClimatology => 1,
  }
}

fn transition_kind_from_code(code: u32) -> AetherResult<TransitionKind> {
  match code {
    0 => Ok(TransitionKind::ClimatologyToLive),
    1 => Ok(TransitionKind::LiveToClimatology),
    other => Err(AetherError::new(CheckpointError::UnknownTransitionKind(
      other,
    ))),
  }
}

impl World {
  pub fn new(
    id: WorldId,
    seed: CelestialBody,
    primary: Option<CelestialBody>,
    tessera: Tessera,
    pleroma: Pleroma,
    nexus: CompiledNexus,
  ) -> Self {
    Self::with_body_index(id, seed, primary, tessera, pleroma, nexus, None)
  }

  pub fn with_body_index(
    id: WorldId,
    seed: CelestialBody,
    primary: Option<CelestialBody>,
    tessera: Tessera,
    pleroma: Pleroma,
    nexus: CompiledNexus,
    body_index: Option<usize>,
  ) -> Self {
    let constants = world_constants_from_seed(&seed, primary.as_ref());
    Self {
      id,
      seed,
      primary,
      constants,
      tessera,
      pleroma,
      nexus,
      partition_count: 1,
      body_index,
      regime: Regime::default(),
      regime_config: RegimeConfig::default(),
      game_clock: 0.0,
      sim_time: 0.0,
      transition: None,
      adapters: Vec::new(),
      tick_count: 0,
    }
  }

  /// Register an adaptive-refinement controller for one mesh. The world drives it
  /// at the end-of-tick barrier (see [`crate::adapt`]); without any adapter the
  /// mesh never changes. The `mesh` must be the same [`AdaptiveMesh`] that was
  /// registered into the world's `Tessera` for `mesh_key`.
  pub fn add_adapter(&mut self, adapter: crate::adapt::MeshAdapter) {
    self.adapters.push(adapter);
  }

  pub fn body_index(&self) -> Option<usize> {
    self.body_index
  }

  pub fn set_body_index(&mut self, body_index: Option<usize>) {
    self.body_index = body_index;
  }

  pub fn partition_count(&self) -> usize {
    self.partition_count
  }

  pub fn set_partition_count(&mut self, partition_count: usize) {
    self.partition_count = partition_count.max(1);
  }

  pub fn primary(&self) -> Option<&CelestialBody> {
    self.primary.as_ref()
  }

  pub fn id(&self) -> WorldId {
    self.id
  }

  pub fn seed(&self) -> &CelestialBody {
    &self.seed
  }

  pub fn constants(&self) -> &WorldConstants {
    &self.constants
  }

  pub fn tessera(&self) -> &Tessera {
    &self.tessera
  }

  pub fn tessera_mut(&mut self) -> &mut Tessera {
    &mut self.tessera
  }

  pub fn pleroma(&self) -> &Pleroma {
    &self.pleroma
  }

  pub fn pleroma_mut(&mut self) -> &mut Pleroma {
    &mut self.pleroma
  }

  pub fn runtime_parts_mut(&mut self) -> (&Tessera, &mut Pleroma) {
    (&self.tessera, &mut self.pleroma)
  }

  /// The latest aggregate runtime-diagnostics report (per-field finiteness and
  /// conservation drift) published by in-DAG monitor stages. Present for worlds
  /// built via `WorldFactory`; `None` if no `Diagnostics` resource was
  /// registered.
  pub fn diagnostics(&self) -> Option<&WorldDiagnostics> {
    self
      .pleroma
      .read_resource::<WorldDiagnostics>(ResourceKey::Diagnostics)
  }

  /// Change the runtime-diagnostics enforcement policy. Takes effect on the
  /// next tick; a no-op if the world has no `Diagnostics` resource.
  pub fn set_diagnostics_policy(&mut self, policy: DiagnosticsPolicy) {
    if let Some(diagnostics) = self
      .pleroma
      .write_resource::<WorldDiagnostics>(ResourceKey::Diagnostics)
    {
      diagnostics.policy = policy;
    }
  }

  /// The events published at the last end-of-tick barrier — the broadcast batch
  /// a consumer polls (mirrors [`World::diagnostics`]). Within a tick these are an
  /// unordered set (see `utility::events`); empty if the world has no `Events`
  /// resource. Stages read the same batch via the `Events` resource.
  pub fn events(&self) -> &[Event] {
    self
      .pleroma
      .read_resource::<EventBus>(ResourceKey::Events)
      .map(EventBus::published)
      .unwrap_or(&[])
  }

  /// Emit an event into this tick's pending buffer (interior mutability, so a
  /// `&self` world-level emitter works). A no-op if the world has no `Events`
  /// resource. The event becomes visible via [`events`](World::events) after the
  /// next end-of-tick barrier.
  fn emit_event(&self, event: Event) {
    if let Some(bus) =
      self.pleroma.read_resource::<EventBus>(ResourceKey::Events)
    {
      bus.emit(event);
    }
  }

  /// Rotate the event buffer at the end-of-tick barrier: this tick's pending
  /// emissions become the published batch. A no-op without an `Events` resource.
  fn publish_events(&mut self) {
    if let Some(bus) =
      self.pleroma.write_resource::<EventBus>(ResourceKey::Events)
    {
      bus.publish();
    }
  }

  /// Whether this world's nexus advances more than one subsystem clock —
  /// i.e. the multirate driver subcycles it and it cannot be fused into the
  /// shared multi-world scheduler graph.
  pub fn is_multirate(&self) -> bool {
    self.nexus.is_multirate()
  }

  /// Current time-advance regime.
  pub fn regime(&self) -> Regime {
    self.regime
  }

  pub fn set_regime(&mut self, regime: Regime) {
    if regime != self.regime {
      self.emit_event(Event::RegimeChanged {
        from: regime_kind(self.regime),
        to: regime_kind(regime),
      });
    }
    self.regime = regime;
  }

  pub fn regime_config(&self) -> RegimeConfig {
    self.regime_config
  }

  pub fn set_regime_config(&mut self, config: RegimeConfig) {
    self.regime_config = config;
  }

  /// Game time advanced so far (s). Runs ahead of [`World::sim_time`] in the
  /// climatology regime, where held spans advance the clock without integrating.
  pub fn game_clock(&self) -> f64 {
    self.game_clock
  }

  /// Simulation time actually integrated by the solver so far (s).
  pub fn sim_time(&self) -> f64 {
    self.sim_time
  }

  /// Advance both clocks by `dt` without ticking — used by the fused step path,
  /// which integrates the solver via `build_tick_tasks` rather than `tick`.
  pub(crate) fn advance_clocks(&mut self, dt: f64) {
    self.sim_time += dt;
    self.game_clock += dt;
  }

  /// The live↔climatology handoff in progress, if any.
  pub fn transition(&self) -> Option<TransitionState> {
    self.transition
  }

  /// Begin a live↔climatology handoff. While it runs, integrated ticks publish
  /// its relaxation fraction to `ResourceKey::ClimateRegime` so a chronos nudge
  /// stage (if the world has one) blends the live state across the switch.
  pub fn begin_transition(&mut self, transition: TransitionState) {
    self.emit_event(Event::TransitionStarted {
      to: transition_target(transition.kind),
    });
    self.transition = Some(transition);
  }

  /// Capture a full, restartable [`WorldCheckpoint`]: the integrated clocks,
  /// advance mode, and the entire pleroma state. Fails if any field is
  /// non-finite (a blown-up world is not checkpointed).
  pub fn save_checkpoint(&self) -> AetherResult<WorldCheckpoint> {
    Ok(WorldCheckpoint {
      sim_time: self.sim_time,
      game_clock: self.game_clock,
      regime: regime_code(self.regime),
      transition: self.transition.map(|t| TransitionRecord {
        kind: transition_kind_code(t.kind),
        progress: t.progress,
        window: t.window,
      }),
      refinements: self
        .adapters
        .iter()
        .map(|a| {
          let (leaf_bases, leaf_paths) = a.mesh.forest_leaf_codes();
          RefinementRecord {
            mesh: mesh_type_code(a.mesh_key.mesh_type()),
            epoch: self.tessera.topology_epoch(a.mesh_key).0,
            leaf_bases,
            leaf_paths,
          }
        })
        .collect(),
      pleroma: self.pleroma.save()?,
    })
  }

  /// Restore from a [`WorldCheckpoint`] into this already-assembled world: the
  /// pleroma schema must match (same registered fields/resources of the same
  /// type and size) or a clear error is returned. The clocks and advance mode are
  /// rewound to the snapshot and the run can resume.
  pub fn load_checkpoint(
    &mut self,
    checkpoint: &WorldCheckpoint,
  ) -> AetherResult<()> {
    // Validate the advance mode before mutating any state, so a corrupt header
    // fails cleanly rather than half-applying.
    let regime = regime_from_code(checkpoint.regime)?;
    let transition = match &checkpoint.transition {
      Some(record) => {
        let mut state = TransitionState::new(
          transition_kind_from_code(record.kind)?,
          record.window,
        );
        state.progress = record.progress;
        Some(state)
      }
      None => None,
    };
    // Reconstruct the adapted mesh topology *before* loading the (adapted-size)
    // fields, so the pleroma schema (cell counts) matches. Each record rebuilds
    // its adapter's `AdaptiveMesh` from the saved leaf codes, swaps it into the
    // tessera, and bumps the epoch.
    for record in &checkpoint.refinements {
      let key = MeshKey::new(mesh_type_from_code(record.mesh)?);
      let Some(idx) = self.adapters.iter().position(|a| a.mesh_key == key)
      else {
        continue;
      };
      let base = self.adapters[idx].mesh.base().clone();
      let restored = Arc::new(AdaptiveMesh::restore(
        base,
        &record.leaf_bases,
        &record.leaf_paths,
        TopologyEpoch(record.epoch),
      )?);
      let adapted_count = restored.cell_count();
      self.tessera.register_mesh(key, restored.clone());
      self
        .tessera
        .set_topology_epoch(key, TopologyEpoch(record.epoch));
      self.adapters[idx].mesh = restored;
      // Grow/shrink the field slots to the reconstructed topology so the
      // (adapted-size) checkpoint values load cleanly over them.
      self.pleroma.resize_mesh_fields(key, adapted_count);
    }
    self.pleroma.load(&checkpoint.pleroma)?;
    self.sim_time = checkpoint.sim_time;
    self.game_clock = checkpoint.game_clock;
    self.regime = regime;
    self.transition = transition;
    Ok(())
  }

  /// Write a checkpoint to `writer` as JSON, reusing the serialization backend.
  pub fn save_checkpoint_to<W: std::io::Write>(
    &self,
    writer: W,
  ) -> AetherResult<()> {
    let checkpoint = self.save_checkpoint()?;
    let mut serializer = JsonSerializer::new(writer);
    checkpoint.serialize(&mut serializer)
  }

  /// Read a checkpoint from `reader` (JSON) and restore it into this world.
  pub fn load_checkpoint_from<R: std::io::Read>(
    &mut self,
    reader: R,
  ) -> AetherResult<()> {
    let mut deserializer = JsonDeserializer::new(reader);
    let checkpoint = WorldCheckpoint::deserialize(&mut deserializer)?;
    self.load_checkpoint(&checkpoint)
  }

  /// Integrate one tick of `dt`, publishing the current transition relaxation
  /// fraction beforehand and advancing the transition afterwards. Worlds with
  /// no transition and no `ClimateRegime` resource behave exactly like `tick`.
  fn tick_in_transition(&mut self, pool: &Pool, dt: f64) -> AetherResult<()> {
    // Publish the current relaxation fraction (0 when no transition is active,
    // so the nudge is inert once a handoff completes). Worlds without the
    // resource simply skip this.
    let fraction = self.transition.map(|t| t.nudge_fraction()).unwrap_or(0.0);
    if let Some(slot) = self
      .pleroma
      .write_resource::<f64>(ResourceKey::ClimateRegime)
    {
      *slot = fraction;
    }
    self.tick(pool, dt)?;
    if let Some(transition) = self.transition.as_mut() {
      if transition.advance(dt) {
        let settled = transition_target(transition.kind);
        self.transition = None;
        self.emit_event(Event::TransitionCompleted { to: settled });
      }
    }
    Ok(())
  }

  pub fn tick(&mut self, pool: &Pool, dt: f64) -> AetherResult<()> {
    let result = self.nexus.tick_with_partition_count(
      self.id,
      &self.tessera,
      &self.constants,
      &mut self.pleroma,
      pool,
      dt,
      self.partition_count,
    );
    // Adapt at the end-of-tick barrier — never mid-DAG — and only on success, so
    // no stage is holding state borrows and a failed tick freezes the mesh too.
    // Runs before `publish_events` so its `TopologyChanged` joins this tick's
    // batch. Mesh adaptation is bounded by each adapter's governor.
    let adapt_result = if result.is_ok() {
      self.tick_count = self.tick_count.wrapping_add(1);
      self.run_adapters()
    } else {
      Ok(())
    };
    // Publish the event buffer at the barrier regardless of success, so a stage
    // that emitted (e.g. a `NonFiniteState` before a `Fail` `Err`) is still
    // visible via `events()` to a consumer handling the error.
    self.publish_events();
    result?;
    adapt_result?;
    // Clocks only advance on a successful tick (preserving the Fail-freeze
    // contract from the diagnostics layer).
    self.sim_time += dt;
    self.game_clock += dt;
    Ok(())
  }

  /// Run every registered adapter whose governor fires this tick: evaluate its
  /// criterion, balance the request, refine the mesh, conservatively remap the
  /// mesh's fields, swap in the new mesh with a bumped epoch, and emit
  /// [`Event::TopologyChanged`]. The serial solver path picks up the new mesh on
  /// the next tick (AMR v1 runs on a single partition).
  #[profile]
  fn run_adapters(&mut self) -> AetherResult<()> {
    for i in 0..self.adapters.len() {
      let governor = self.adapters[i].governor;
      if !governor.fires_on(self.tick_count) {
        continue;
      }
      let mesh = self.adapters[i].mesh.clone();
      let key = self.adapters[i].mesh_key;

      let desired = self.adapters[i]
        .criterion
        .evaluate(mesh.as_ref(), &self.pleroma)?;
      let request = governor.cap(desired);
      if request.refine.is_empty() && request.coarsen.is_empty() {
        continue;
      }
      let balanced = balance_2to1(
        mesh.as_ref(),
        mesh.cell_count(),
        |c| mesh.level_of(c),
        &request,
      );

      // Pre-adapt volumes drive the conservative coarsening average.
      let old_volumes: Vec<f64> = (0..mesh.cell_count())
        .map(|c| mesh.cell_volume(CellId::from(c)))
        .collect();
      // Best-effort: a request the backend can't realise (e.g. one that would
      // cross a cube-sphere panel seam — the v1 limitation) is skipped this tick,
      // leaving the mesh unchanged, rather than failing the whole simulation.
      let Ok((new_mesh, remap)) = mesh.refine(&balanced) else {
        continue;
      };
      let epoch = remap.new_epoch();

      // Remap state into the new cell layout, then swap the mesh in and bump the
      // epoch (build-then-swap: state is intact until the remap completes).
      self.pleroma.remap_mesh_fields(key, &remap, &old_volumes);
      let new_mesh = Arc::new(new_mesh);
      self.tessera.register_mesh(key, new_mesh.clone());
      self.tessera.set_topology_epoch(key, epoch);

      // Broadcast for the read-side consumers (query / render / checkpoint).
      if let Some(bus) =
        self.pleroma.read_resource::<EventBus>(ResourceKey::Events)
      {
        bus.emit(Event::TopologyChanged {
          mesh: key,
          epoch,
          remap: Arc::new(remap),
        });
      }
      self.adapters[i].mesh = new_mesh;
    }
    Ok(())
  }

  /// Advance game-time by `game_dt`, honouring the world's [`Regime`].
  ///
  /// In [`Regime::Live`] this integrates the world by `game_dt` (one outer
  /// tick, internally subcycled by the multirate driver). In
  /// [`Regime::Climatology`] it runs `burst_steps` short live steps to refresh
  /// the climatology aggregates, then advances the game clock by the remainder
  /// of `game_dt` while *holding* the Euler state — so a large `game_dt` costs
  /// only the burst, not a full integration. The game clock and integrated sim
  /// time diverge by design in the climatology regime; that is the cost saving.
  pub fn advance(&mut self, pool: &Pool, game_dt: f64) -> AetherResult<()> {
    match self.regime {
      Regime::Live => self.tick_in_transition(pool, game_dt),
      Regime::Climatology => {
        let cfg = self.regime_config;
        let mut integrated = 0.0;
        for _ in 0..cfg.burst_steps {
          // The burst integrates the solver (advancing both clocks); cap it at
          // game_dt so a burst never overshoots the requested span.
          if integrated >= game_dt {
            break;
          }
          let dt = cfg.burst_dt.min(game_dt - integrated);
          self.tick_in_transition(pool, dt)?;
          integrated += dt;
        }
        // Hold: advance the game clock over the remaining span without
        // integrating the Euler state. Consumers read the climatology here.
        let held = (game_dt - integrated).max(0.0);
        self.game_clock += held;
        Ok(())
      }
    }
  }

  pub fn build_tick_tasks<'a>(
    &'a mut self,
    pool: &'a Pool,
    dt: f64,
  ) -> AetherResult<Vec<StageTask<'a>>> {
    self.nexus.build_tick_tasks(
      self.id,
      &self.tessera,
      &self.constants,
      &mut self.pleroma,
      pool,
      dt,
      self.partition_count,
    )
  }
}

pub fn world_constants_from_seed(
  seed: &CelestialBody,
  primary: Option<&CelestialBody>,
) -> WorldConstants {
  let atmosphere = atmosphere_constants_from_seed(seed);
  WorldConstants {
    mass: seed.mass(),
    radius: seed.radius(),
    surface_gravity: seed.surface_gravity(),
    radiation: radiation_constants_from_seed(seed, primary, atmosphere),
    atmosphere,
  }
}

fn radiation_constants_from_seed(
  seed: &CelestialBody,
  primary: Option<&CelestialBody>,
  atmosphere: Option<AtmosphereConstants>,
) -> Option<RadiationConstants> {
  let primary = primary?;
  let luminosity = primary.luminosity()?;
  let position = seed.position();
  let distance =
    (position[0].powi(2) + position[1].powi(2) + position[2].powi(2)).sqrt();
  if !distance.is_finite() || distance <= 0.0 {
    return None;
  }
  let solar_irradiance = solar_flux(luminosity, distance);
  let surface_albedo = atmosphere
    .and_then(|a| a.albedo)
    .unwrap_or(DEFAULT_SURFACE_ALBEDO);
  Some(RadiationConstants {
    solar_irradiance,
    surface_albedo,
    surface_emissivity: DEFAULT_SURFACE_EMISSIVITY,
  })
}

fn atmosphere_constants_from_seed(
  seed: &CelestialBody,
) -> Option<AtmosphereConstants> {
  match seed.kind() {
    BodyKind::RockyBody(body) => {
      let atmosphere = body.atmosphere.as_ref()?;
      let properties = atmosphere.properties(body.surface_temperature);
      Some(AtmosphereConstants {
        reference_temperature: body.surface_temperature,
        reference_pressure: body.surface_pressure,
        gamma: properties.gamma,
        gas_constant: properties.gas_constant,
        molar_mass: properties.molar_mass,
        albedo: atmosphere.albedo,
        angular_velocity: body.angular_velocity,
        axial_tilt: body.axial_tilt,
      })
    }
    BodyKind::GasGiant(body) => {
      let properties = body.atmosphere.properties(body.reference_temperature);
      Some(AtmosphereConstants {
        reference_temperature: body.reference_temperature,
        reference_pressure: body.reference_pressure,
        gamma: properties.gamma,
        gas_constant: properties.gas_constant,
        molar_mass: properties.molar_mass,
        albedo: body.atmosphere.albedo,
        angular_velocity: body.angular_velocity,
        axial_tilt: body.axial_tilt,
      })
    }
    BodyKind::Star(_) => None,
  }
}

/// Runtime state for one generated star/planet system.
///
/// System-level physics such as N-body gravity belongs here. World-local
/// physics still lives inside each `World` and is ticked after the system
/// layer.
pub struct System {
  id: SystemId,
  worlds: HashMap<WorldId, World>,
}

impl System {
  pub fn new(id: SystemId, worlds: HashMap<WorldId, World>) -> Self {
    Self { id, worlds }
  }

  pub fn single(id: SystemId, world: World) -> Self {
    let mut worlds = HashMap::new();
    worlds.insert(world.id(), world);
    Self::new(id, worlds)
  }

  pub fn id(&self) -> SystemId {
    self.id
  }

  pub fn insert_world(&mut self, world: World) -> Option<World> {
    self.worlds.insert(world.id(), world)
  }

  pub fn world(&self, id: WorldId) -> Option<&World> {
    self.worlds.get(&id)
  }

  pub fn world_mut(&mut self, id: WorldId) -> Option<&mut World> {
    self.worlds.get_mut(&id)
  }

  pub fn worlds(&self) -> impl Iterator<Item = &World> {
    self.worlds.values()
  }

  pub fn worlds_mut(&mut self) -> impl Iterator<Item = &mut World> {
    self.worlds.values_mut()
  }

  pub fn tick(&mut self, pool: &Pool, dt: f64) -> AetherResult<()> {
    // Future system-level nexus/resources run here before world-local physics.
    for world in self.worlds.values_mut() {
      world.tick(pool, dt)?;
    }
    Ok(())
  }
}

pub struct Aether {
  systems: HashMap<SystemId, System>,
  pool: Pool,
}

impl Aether {
  pub fn new(systems: HashMap<SystemId, System>, pool: Pool) -> Self {
    Self { systems, pool }
  }

  pub fn from_worlds(worlds: HashMap<WorldId, World>, pool: Pool) -> Self {
    let mut systems = HashMap::new();
    systems.insert(SystemId(0), System::new(SystemId(0), worlds));
    Self::new(systems, pool)
  }

  #[profile("aether.step")]
  pub fn step(&mut self, dt: f64) -> AetherResult<()> {
    let systems = &mut self.systems;
    let pool = &self.pool;

    // Multirate worlds subcycle their subsystems internally and can't be
    // fused into the shared cross-world graph, so they tick individually.
    // When no world is multirate this collapses to the original single
    // fused-graph path, preserving behaviour and cross-world parallelism.
    let any_multirate = systems
      .values()
      .flat_map(System::worlds)
      .any(World::is_multirate);
    if any_multirate {
      for system in systems.values_mut() {
        system.tick(pool, dt)?;
      }
      return Ok(());
    }

    let mut graph = ScopedTaskGraph::new();

    for system in systems.values_mut() {
      for world in system.worlds_mut() {
        let tasks = world.build_tick_tasks(pool, dt)?;
        let mut node_ids = Vec::with_capacity(tasks.len());
        for task in tasks {
          let StageTask {
            name,
            task,
            predecessors,
          } = task;
          let node = match task {
            ScheduledStageTask::Worker(task) => graph.add(task),
            ScheduledStageTask::Program(program) => {
              graph.add_scheduler(move |scheduler| {
                utility::inline_profile!(name);
                let result = program.execute(scheduler);
                utility::end_profile!(name);
                result
              })
            }
          };
          for predecessor in predecessors {
            graph.dependency(node, node_ids[predecessor])?;
          }
          node_ids.push(node);
        }
      }
    }

    let result = pool.execute_scoped(graph);
    // The fused path bypasses `World::tick`, so do its barrier work here: rotate
    // the event buffer **regardless of success** (so a stage that emitted before a
    // failing tick is still visible to `events()`), and advance the clocks only on
    // a successful tick (preserving the Fail-freeze contract).
    for system in self.systems.values_mut() {
      for world in system.worlds_mut() {
        world.publish_events();
        if result.is_ok() {
          world.advance_clocks(dt);
        }
      }
    }
    result
  }

  /// Advance game-time by `game_dt`, honouring each world's [`Regime`]. Live
  /// worlds integrate by `game_dt`; climatology worlds burst-then-hold (see
  /// [`World::advance`]). Unlike [`Aether::step`] this drives worlds
  /// individually rather than fusing them into one cross-world graph, since the
  /// per-world burst-then-hold cadence is not a single shared dt.
  pub fn advance(&mut self, game_dt: f64) -> AetherResult<()> {
    let pool = &self.pool;
    for system in self.systems.values_mut() {
      for world in system.worlds_mut() {
        world.advance(pool, game_dt)?;
      }
    }
    Ok(())
  }

  pub fn system(&self, id: SystemId) -> Option<&System> {
    self.systems.get(&id)
  }

  pub fn system_mut(&mut self, id: SystemId) -> Option<&mut System> {
    self.systems.get_mut(&id)
  }

  pub fn systems(&self) -> impl Iterator<Item = &System> {
    self.systems.values()
  }

  pub fn systems_mut(&mut self) -> impl Iterator<Item = &mut System> {
    self.systems.values_mut()
  }

  pub fn insert_system(&mut self, system: System) -> Option<System> {
    self.systems.insert(system.id(), system)
  }

  pub fn world(&self, id: WorldId) -> Option<&World> {
    self.systems.values().find_map(|system| system.world(id))
  }

  pub fn world_in_system(
    &self,
    system_id: SystemId,
    world_id: WorldId,
  ) -> Option<&World> {
    self.system(system_id)?.world(world_id)
  }

  pub fn worlds(&self) -> impl Iterator<Item = &World> {
    self.systems.values().flat_map(System::worlds)
  }
}

#[cfg(test)]
mod tests {
  use cosmo::factory;
  use nexus::{
    CellView, FieldKey, FieldName, FieldStorage, MeshKey, Nexus, SoaField,
    Stage, StageContext,
  };
  use tessera::world_mesh::Tessera;
  use utility::constants::{EARTH_ORBIT, SOLAR_LUMIN};
  use utility::domain::CellId;

  use super::*;

  const TEST_FIELD: FieldKey =
    FieldKey::new(MeshKey::SURFACE, FieldName::Temperature);

  struct WriteWorldId {
    writes: [FieldKey; 1],
  }

  impl WriteWorldId {
    fn new() -> Self {
      Self {
        writes: [TEST_FIELD],
      }
    }
  }

  impl Stage for WriteWorldId {
    fn name(&self) -> &'static str {
      "write_world_id"
    }

    fn reads(&self) -> &[FieldKey] {
      &[]
    }

    fn writes(&self) -> &[FieldKey] {
      &self.writes
    }

    fn run(&mut self, mut ctx: StageContext<'_>) -> AetherResult<()> {
      let field: &mut SoaField<1> = ctx.world.fields.write(TEST_FIELD).unwrap();
      field.write(
        CellId::from(0),
        &[ctx.world.world_id.0 as f64 + ctx.world.dt],
      );
      Ok(())
    }
  }

  struct WritePartitionCount {
    writes: [FieldKey; 1],
  }

  impl WritePartitionCount {
    fn new() -> Self {
      Self {
        writes: [TEST_FIELD],
      }
    }
  }

  impl Stage for WritePartitionCount {
    fn name(&self) -> &'static str {
      "write_partition_count"
    }

    fn reads(&self) -> &[FieldKey] {
      &[]
    }

    fn writes(&self) -> &[FieldKey] {
      &self.writes
    }

    fn run(&mut self, mut ctx: StageContext<'_>) -> AetherResult<()> {
      let field: &mut SoaField<1> = ctx.world.fields.write(TEST_FIELD).unwrap();
      field.write(CellId::from(0), &[ctx.world.partition_count as f64]);
      Ok(())
    }
  }

  fn scheduled_test_world(id: WorldId) -> World {
    let mut pleroma = Pleroma::new();
    pleroma.register_field(TEST_FIELD, SoaField::<1>::zeros(1));

    let mut nexus = Nexus::new();
    nexus.add(WriteWorldId::new());
    let compiled = nexus.build(&pleroma).unwrap();

    World::new(
      id,
      factory::earth(),
      None,
      Tessera::default(),
      pleroma,
      compiled,
    )
  }

  fn partition_count_test_world(id: WorldId, partition_count: usize) -> World {
    let mut pleroma = Pleroma::new();
    pleroma.register_field(TEST_FIELD, SoaField::<1>::zeros(1));

    let mut nexus = Nexus::new();
    nexus.add(WritePartitionCount::new());
    let compiled = nexus.build(&pleroma).unwrap();

    let mut world = World::new(
      id,
      factory::earth(),
      None,
      Tessera::default(),
      pleroma,
      compiled,
    );
    world.set_partition_count(partition_count);
    world
  }

  #[test]
  fn adaptation_refines_a_high_gradient_region_and_emits_topology_changed() {
    use crate::adapt::{AdaptGovernor, GradientCriterion, MeshAdapter};
    use tessera::adaptive::AdaptiveMesh;
    use tessera::cube_sphere::CubeSphere;
    use tessera::geometry::CellGeometry;
    use utility::events::{Event, EventBus};
    use utility::thread::pool::Pool;

    let key = MeshKey::ATMOSPHERE;
    let field = FieldKey::new(key, FieldName::Temperature);

    let base = Arc::new(CubeSphere::new([8, 8, 2], 1.0, 2.0));
    let amesh = Arc::new(AdaptiveMesh::new(base));
    let n0 = amesh.cell_count();

    // A sharp spike deep in panel 0's interior (angular (4,4), radial 0 ⇒ local
    // 4 + 4*8 = 36), zero elsewhere — so every high-gradient cell is interior to
    // the panel (no seam), which v1 refinement supports.
    let spike = 36usize;
    let mut pleroma = Pleroma::new();
    pleroma.register_field(
      field,
      SoaField::<1>::from_fn(n0, |c| {
        if c.index() == spike { [1.0] } else { [0.0] }
      }),
    );
    pleroma.register_resource(ResourceKey::Events, EventBus::new());

    let mut tessera = Tessera::default();
    tessera.register_mesh(key, amesh.clone());

    let compiled = Nexus::new().build(&pleroma).unwrap();
    let mut world = World::new(
      WorldId(0),
      factory::earth(),
      None,
      tessera,
      pleroma,
      compiled,
    );
    world.add_adapter(MeshAdapter::new(
      key,
      amesh,
      // Refine where the field jumps by > 0.5; never coarsen; cap depth at 1.
      Box::new(GradientCriterion::<1>::new(field, 0, 0.5, -1.0, 1)),
      AdaptGovernor::new(1, usize::MAX, usize::MAX),
    ));

    world.tick(&Pool::default(), 1.0).unwrap();

    // The spike region refined: more cells, epoch bumped to 1.
    let new_count = world.tessera.mesh(key).unwrap().cell_count();
    assert!(new_count > n0, "mesh did not refine: {n0} -> {new_count}");
    assert_eq!(world.tessera.topology_epoch(key).0, 1);

    // The field was conservatively remapped onto the new layout and stays finite.
    let remapped: &SoaField<1> = world.pleroma.read(field).unwrap();
    assert_eq!(remapped.len(), new_count);
    assert!(
      (0..new_count).all(|i| remapped.state(CellId::from(i))[0].is_finite())
    );

    // The topology change was broadcast for the read-side consumers.
    assert!(
      world.events().iter().any(|e| matches!(
        e,
        Event::TopologyChanged { mesh, epoch, .. }
          if *mesh == key && epoch.0 == 1
      )),
      "no TopologyChanged event: {:?}",
      world.events()
    );
  }

  /// A fresh, level-0 adaptive world (ATMOSPHERE) with a single-cell spike field
  /// and a gradient adapter — assembled identically each call so a checkpoint can
  /// round-trip between two instances.
  fn adaptive_spike_world() -> (World, FieldKey, MeshKey) {
    use crate::adapt::{AdaptGovernor, GradientCriterion, MeshAdapter};
    use tessera::cube_sphere::CubeSphere;
    use utility::events::EventBus;

    let key = MeshKey::ATMOSPHERE;
    let field = FieldKey::new(key, FieldName::Temperature);
    let amesh = Arc::new(AdaptiveMesh::new(Arc::new(CubeSphere::new(
      [8, 8, 2],
      1.0,
      2.0,
    ))));
    let n0 = amesh.cell_count();

    let mut pleroma = Pleroma::new();
    pleroma.register_field(
      field,
      SoaField::<1>::from_fn(
        n0,
        |c| {
          if c.index() == 36 { [1.0] } else { [0.0] }
        },
      ),
    );
    pleroma.register_resource(ResourceKey::Events, EventBus::new());

    let mut tessera = Tessera::default();
    tessera.register_mesh(key, amesh.clone());
    let compiled = Nexus::new().build(&pleroma).unwrap();
    let mut world = World::new(
      WorldId(0),
      factory::earth(),
      None,
      tessera,
      pleroma,
      compiled,
    );
    world.add_adapter(MeshAdapter::new(
      key,
      amesh,
      Box::new(GradientCriterion::<1>::new(field, 0, 0.5, -1.0, 1)),
      AdaptGovernor::new(1, usize::MAX, usize::MAX),
    ));
    (world, field, key)
  }

  #[test]
  fn checkpoint_round_trips_an_adapted_world() {
    use tessera::geometry::CellGeometry;
    use utility::thread::pool::Pool;
    let pool = Pool::default();

    // World A: adapt once, then snapshot the adapted topology + state.
    let (mut a, field, key) = adaptive_spike_world();
    let n0 = a.tessera.mesh(key).unwrap().cell_count();
    a.tick(&pool, 1.0).unwrap();
    let adapted = a.tessera.mesh(key).unwrap().cell_count();
    assert!(adapted > n0, "world A did not refine");
    let checkpoint = a.save_checkpoint().unwrap();
    let a_values: Vec<f64> = {
      let f: &SoaField<1> = a.pleroma.read(field).unwrap();
      (0..adapted).map(|i| f.state(CellId::from(i))[0]).collect()
    };

    // World B: fresh level-0 assembly; load must reconstruct the adapted mesh and
    // restore the adapted-size state bit-for-bit.
    let (mut b, _, _) = adaptive_spike_world();
    assert_eq!(b.tessera.mesh(key).unwrap().cell_count(), n0);
    b.load_checkpoint(&checkpoint).unwrap();

    assert_eq!(
      b.tessera.mesh(key).unwrap().cell_count(),
      adapted,
      "reconstructed mesh has the wrong cell count"
    );
    assert_eq!(b.tessera.topology_epoch(key).0, 1);
    let b_field: &SoaField<1> = b.pleroma.read(field).unwrap();
    assert_eq!(b_field.len(), adapted);
    for i in 0..adapted {
      assert_eq!(
        b_field.state(CellId::from(i))[0],
        a_values[i],
        "restored cell {i} differs from the saved world"
      );
    }
  }

  #[test]
  fn world_constants_without_primary_have_no_radiation_block() {
    let constants = world_constants_from_seed(&factory::earth(), None);
    assert!(constants.radiation.is_none());
  }

  #[test]
  fn world_constants_derive_solar_irradiance_from_primary() {
    let earth = factory::earth();
    let sun = factory::sun();
    let constants = world_constants_from_seed(&earth, Some(&sun));
    let radiation = constants.radiation.expect("radiation derived");

    // Cosmo computes the star's luminosity from Stefan-Boltzmann on
    // (radius, surface temperature) rather than reading SOLAR_LUMIN
    // directly, so the two values agree only to ~0.5%.
    let expected_irradiance =
      SOLAR_LUMIN / (4.0 * std::f64::consts::PI * EARTH_ORBIT.powi(2));
    let relative_error = (radiation.solar_irradiance - expected_irradiance)
      .abs()
      / expected_irradiance;
    assert!(relative_error < 0.01, "got {}", radiation.solar_irradiance);
    assert_eq!(radiation.surface_albedo, 0.30);
    assert_eq!(radiation.surface_emissivity, DEFAULT_SURFACE_EMISSIVITY);
  }

  #[test]
  fn airless_body_falls_back_to_default_albedo() {
    let mercury = factory::mercury();
    let sun = factory::sun();
    let constants = world_constants_from_seed(&mercury, Some(&sun));
    let radiation = constants.radiation.expect("radiation derived");
    assert_eq!(radiation.surface_albedo, DEFAULT_SURFACE_ALBEDO);
  }

  #[test]
  fn aether_step_ticks_multiple_worlds_from_one_scheduler_graph() {
    let worlds = (0..4)
      .map(|id| {
        let world_id = WorldId(id);
        (world_id, scheduled_test_world(world_id))
      })
      .collect();
    let mut aether = Aether::from_worlds(worlds, Pool::new(2).unwrap());

    aether.step(3.0).unwrap();

    for id in 0..4 {
      let world_id = WorldId(id);
      let world = aether.world(world_id).unwrap();
      let field: &SoaField<1> = world.pleroma().read(TEST_FIELD).unwrap();
      assert_eq!(field.state(CellId::from(0)).as_state(), &[id as f64 + 3.0]);
    }
  }

  #[test]
  fn aether_step_passes_world_partition_count_to_stages() {
    let world_id = WorldId(0);
    let mut worlds = HashMap::new();
    worlds.insert(world_id, partition_count_test_world(world_id, 6));
    let mut aether = Aether::from_worlds(worlds, Pool::new(2).unwrap());

    aether.step(1.0).unwrap();

    let world = aether.world(world_id).unwrap();
    let field: &SoaField<1> = world.pleroma().read(TEST_FIELD).unwrap();
    assert_eq!(field.state(CellId::from(0)).as_state(), &[6.0]);
  }
}
