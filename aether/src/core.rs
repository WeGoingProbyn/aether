use std::collections::HashMap;

use cosmo::kind::{BodyKind, CelestialBody};
use nexus::{
  AtmosphereConstants, CompiledNexus, RadiationConstants, ScheduledStageTask,
  StageTask, WorldConstants, WorldId,
};
use pleroma::Pleroma;
use tessera::world_mesh::Tessera;
use utility::{
  constants::solar_flux,
  domain::SystemId,
  error::AetherResult,
  profile,
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
    }
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

  /// Whether this world's nexus advances more than one subsystem clock —
  /// i.e. the multirate driver subcycles it and it cannot be fused into the
  /// shared multi-world scheduler graph.
  pub fn is_multirate(&self) -> bool {
    self.nexus.is_multirate()
  }

  pub fn tick(&mut self, pool: &Pool, dt: f64) -> AetherResult<()> {
    self.nexus.tick_with_partition_count(
      self.id,
      &self.tessera,
      &self.constants,
      &mut self.pleroma,
      pool,
      dt,
      self.partition_count,
    )
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

    pool.execute_scoped(graph)
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
