// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, sync::Arc};

use cosmo::kind::CelestialBody;
use nexus::{
  FieldKey, FieldStorage, MeshKey, Nexus, Stage, StageId, WorldConstants,
  WorldId,
};
use pleroma::Pleroma;
use syzygy::ScalarInterfaceFlux;
use tessera::{
  coupling::MeshCoupler,
  cube_sphere::{CubeSphere, CubeSphereShellSpec},
  mesh::Mesh,
  partition::decompose_cube_sphere_panels,
  radial_stack::RadialStackCoupler,
  world_mesh::{DecompositionKey, Tessera},
};
use utility::{
  diagnostics::{DiagnosticsPolicy, WorldDiagnostics},
  domain::ResourceKey,
  error::{AetherError, AetherResult, ErrorDomain},
};

use crate::core::{World, world_constants_from_seed};

/// Top-level builder for a single simulated world.
///
/// The factory is allowed to know about seed/catalogue data and crate
/// composition. Physics-specific setup should still live in the physics crate
/// that owns it; the factory wires those meshes, fields and stages together.
pub struct WorldFactory {
  world_id: WorldId,
  seed: CelestialBody,
  primary: Option<CelestialBody>,
  tessera: Tessera,
  pleroma: Pleroma,
  nexus: Nexus,
  cube_sphere_shells: HashMap<MeshKey, CubeSphereShellSpec>,
  body_index: Option<usize>,
  partition_count: usize,
  /// Initial runtime-diagnostics policy seeded into the `Diagnostics` resource.
  diagnostics_policy: DiagnosticsPolicy,
}

impl WorldFactory {
  pub fn new(world_id: WorldId, seed: CelestialBody) -> Self {
    Self {
      world_id,
      seed,
      primary: None,
      tessera: Tessera::new(),
      pleroma: Pleroma::new(),
      nexus: Nexus::new(),
      cube_sphere_shells: HashMap::new(),
      body_index: None,
      partition_count: 1,
      diagnostics_policy: DiagnosticsPolicy::default(),
    }
  }

  /// Set the initial runtime-diagnostics enforcement policy for the world.
  /// Seeds the `Diagnostics` resource that monitor stages read; can be changed
  /// later via [`World::set_diagnostics_policy`].
  pub fn with_diagnostics_policy(mut self, policy: DiagnosticsPolicy) -> Self {
    self.diagnostics_policy = policy;
    self
  }

  /// Hint that this world's centre is tracked by index `index` in the
  /// system-level `BodyState<3>::positions`. The eidolon producer uses
  /// this to emit `UpdateWorldTransform` from gravitas-driven body
  /// motion without eidolon depending on the gravitas crate.
  pub fn with_body_index(mut self, index: usize) -> Self {
    self.body_index = Some(index);
    self
  }

  pub fn with_partition_count(mut self, partition_count: usize) -> Self {
    self.partition_count = partition_count.max(1);
    self
  }

  pub fn set_body_index(&mut self, index: Option<usize>) {
    self.body_index = index;
  }

  pub fn with_primary(mut self, primary: CelestialBody) -> Self {
    self.primary = Some(primary);
    self
  }

  pub fn set_primary(&mut self, primary: CelestialBody) {
    self.primary = Some(primary);
  }

  pub fn world_id(&self) -> WorldId {
    self.world_id
  }

  pub fn seed(&self) -> &CelestialBody {
    &self.seed
  }

  pub fn primary(&self) -> Option<&CelestialBody> {
    self.primary.as_ref()
  }

  pub fn constants(&self) -> WorldConstants {
    world_constants_from_seed(&self.seed, self.primary.as_ref())
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

  pub fn nexus_mut(&mut self) -> &mut Nexus {
    &mut self.nexus
  }

  pub fn register_mesh(
    &mut self,
    key: MeshKey,
    mesh: Arc<dyn Mesh<3>>,
  ) -> Option<Arc<dyn Mesh<3>>> {
    self.cube_sphere_shells.remove(&key);
    self.tessera.register_mesh(key, mesh)
  }

  pub fn with_mesh(mut self, key: MeshKey, mesh: Arc<dyn Mesh<3>>) -> Self {
    self.register_mesh(key, mesh);
    self
  }

  pub fn cube_sphere_shell(
    &mut self,
    key: MeshKey,
    spec: CubeSphereShellSpec,
  ) -> Option<Arc<dyn Mesh<3>>> {
    self.cube_sphere_shells.insert(key, spec.clone());
    let mesh = Arc::new(CubeSphere::shell(spec));
    let previous = self.tessera.register_mesh(key, mesh.clone());
    self.tessera.register_decomposition(
      key,
      DecompositionKey::DEFAULT,
      decompose_cube_sphere_panels(mesh),
    );
    previous
  }

  pub fn with_cube_sphere_shell(
    mut self,
    key: MeshKey,
    spec: CubeSphereShellSpec,
  ) -> Self {
    self.cube_sphere_shell(key, spec);
    self
  }

  pub fn cube_sphere_surface(self, spec: CubeSphereShellSpec) -> Self {
    self.with_cube_sphere_shell(MeshKey::SURFACE, spec)
  }

  pub fn cube_sphere_atmosphere(self, spec: CubeSphereShellSpec) -> Self {
    self.with_cube_sphere_shell(MeshKey::ATMOSPHERE, spec)
  }

  pub fn cube_sphere_mantle(self, spec: CubeSphereShellSpec) -> Self {
    self.with_cube_sphere_shell(MeshKey::MANTLE, spec)
  }

  pub fn cube_sphere_ocean(self, spec: CubeSphereShellSpec) -> Self {
    self.with_cube_sphere_shell(MeshKey::OCEAN, spec)
  }

  pub fn add_coupler(
    &mut self,
    mesh_a: MeshKey,
    mesh_b: MeshKey,
    coupler: impl MeshCoupler + 'static,
  ) -> usize {
    self.tessera.add_coupler(mesh_a, mesh_b, coupler)
  }

  pub fn add_radial_stack_coupler(
    &mut self,
    lower: MeshKey,
    upper: MeshKey,
  ) -> AetherResult<usize> {
    let lower_spec = self.cube_sphere_shell_spec(lower)?;
    let upper_spec = self.cube_sphere_shell_spec(upper)?;
    if lower_spec.angular_dims != upper_spec.angular_dims {
      return Err(
        AetherError::new(WorldFactoryError::IncompatibleShells).context(
          format!(
            "{:?} angular dims {:?}, {:?} angular dims {:?}",
            lower, lower_spec.angular_dims, upper, upper_spec.angular_dims
          ),
        ),
      );
    }

    Ok(self.tessera.add_coupler(
      lower,
      upper,
      RadialStackCoupler::new(
        lower_spec.angular_dims,
        lower_spec.radial_layers(),
        upper_spec.radial_layers(),
      ),
    ))
  }

  pub fn radial_surface_atmosphere_coupler(mut self) -> AetherResult<Self> {
    self.add_radial_stack_coupler(MeshKey::SURFACE, MeshKey::ATMOSPHERE)?;
    Ok(self)
  }

  pub fn add_scalar_interface_flux(
    &mut self,
    coupler_index: usize,
    source: FieldKey,
    target: FieldKey,
    tendency: FieldKey,
    conductance: f64,
  ) -> AetherResult<StageId> {
    let stage = ScalarInterfaceFlux::from_coupler(
      &self.tessera,
      coupler_index,
      source,
      target,
      tendency,
      conductance,
    )?;
    Ok(self.add_stage(stage))
  }

  pub fn with_scalar_interface_flux(
    mut self,
    coupler_index: usize,
    source: FieldKey,
    target: FieldKey,
    tendency: FieldKey,
    conductance: f64,
  ) -> AetherResult<Self> {
    self.add_scalar_interface_flux(
      coupler_index,
      source,
      target,
      tendency,
      conductance,
    )?;
    Ok(self)
  }

  pub fn register_field<S, const N: usize>(&mut self, key: FieldKey, init: S)
  where
    S: FieldStorage<N> + 'static,
  {
    self.pleroma.register_field(key, init);
  }

  pub fn with_field<S, const N: usize>(mut self, key: FieldKey, init: S) -> Self
  where
    S: FieldStorage<N> + 'static,
  {
    self.register_field(key, init);
    self
  }

  pub fn add_stage(&mut self, stage: impl Stage + 'static) -> StageId {
    self.nexus.add(stage)
  }

  pub fn with_stage(mut self, stage: impl Stage + 'static) -> Self {
    self.add_stage(stage);
    self
  }

  pub fn before(&mut self, a: StageId, b: StageId) {
    self.nexus.before(a, b);
  }

  pub fn build(mut self) -> AetherResult<World> {
    // Always present so `World::diagnostics` works even with no monitor stage;
    // monitor stages merge into this aggregate report and read its policy.
    self.pleroma.register_resource(
      ResourceKey::Diagnostics,
      WorldDiagnostics::with_policy(self.diagnostics_policy),
    );
    let compiled_nexus = self.nexus.build(&self.pleroma)?;
    let mut world = World::with_body_index(
      self.world_id,
      self.seed,
      self.primary,
      self.tessera,
      self.pleroma,
      compiled_nexus,
      self.body_index,
    );
    world.set_partition_count(self.partition_count);
    Ok(world)
  }

  fn cube_sphere_shell_spec(
    &self,
    key: MeshKey,
  ) -> AetherResult<&CubeSphereShellSpec> {
    self.cube_sphere_shells.get(&key).ok_or_else(|| {
      AetherError::new(WorldFactoryError::MissingCubeSphereShell)
        .context(format!("{:?}", key))
    })
  }
}

#[derive(Debug)]
pub enum WorldFactoryError {
  MissingCubeSphereShell,
  IncompatibleShells,
}

impl ErrorDomain for WorldFactoryError {
  fn domain(&self) -> &str {
    "aether_factory"
  }
}

impl std::fmt::Display for WorldFactoryError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      WorldFactoryError::MissingCubeSphereShell => {
        write!(f, "mesh was not registered from a cube-sphere shell spec")
      }
      WorldFactoryError::IncompatibleShells => {
        write!(f, "cube-sphere shells cannot be coupled directly")
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use cosmo::factory;
  use nexus::{FieldName, SoaField};
  use tessera::cube_sphere::CubeSphereShellSpec;
  use utility::thread::pool::Pool;

  use super::*;

  #[test]
  fn factory_builds_world_with_registered_cube_sphere_meshes() {
    let world = WorldFactory::new(WorldId(7), factory::earth())
      .cube_sphere_surface(CubeSphereShellSpec::uniform([2, 2, 1], 0.9, 1.0))
      .cube_sphere_atmosphere(CubeSphereShellSpec::uniform([2, 2, 2], 1.0, 1.2))
      .build()
      .unwrap();

    assert_eq!(world.id(), WorldId(7));
    assert!(world.tessera().contains_mesh(MeshKey::SURFACE));
    assert!(world.tessera().contains_mesh(MeshKey::ATMOSPHERE));
    assert!(
      world
        .tessera()
        .contains_decomposition(MeshKey::SURFACE, DecompositionKey::DEFAULT)
    );
    assert!(
      world
        .tessera()
        .contains_decomposition(MeshKey::ATMOSPHERE, DecompositionKey::DEFAULT)
    );
    assert_eq!(world.tessera().couplers().len(), 0);
  }

  #[test]
  fn factory_sets_static_partition_count() {
    let world = WorldFactory::new(WorldId(7), factory::earth())
      .with_partition_count(6)
      .build()
      .unwrap();

    assert_eq!(world.partition_count(), 6);
  }

  #[test]
  fn factory_adds_radial_stack_coupler_from_shell_specs() {
    let world = WorldFactory::new(WorldId(0), factory::earth())
      .cube_sphere_surface(CubeSphereShellSpec::uniform([4, 4, 2], 0.9, 1.0))
      .cube_sphere_atmosphere(CubeSphereShellSpec::uniform([4, 4, 3], 1.0, 1.2))
      .radial_surface_atmosphere_coupler()
      .unwrap()
      .build()
      .unwrap();

    let view = world.tessera().coupler_view(0).unwrap();
    assert_eq!(view.mesh_a(), MeshKey::SURFACE);
    assert_eq!(view.mesh_b(), MeshKey::ATMOSPHERE);
    assert_eq!(view.pair_count(), 6 * 4 * 4);
  }

  #[test]
  fn factory_rejects_radial_coupler_for_mismatched_shell_dims() {
    let result = WorldFactory::new(WorldId(0), factory::earth())
      .cube_sphere_surface(CubeSphereShellSpec::uniform([4, 4, 2], 0.9, 1.0))
      .cube_sphere_atmosphere(CubeSphereShellSpec::uniform([5, 5, 3], 1.0, 1.2))
      .radial_surface_atmosphere_coupler();

    assert!(result.is_err());
  }

  #[test]
  fn factory_adds_scalar_interface_flux_stage() {
    let source = FieldKey::new(MeshKey::SURFACE, FieldName::Temperature);
    let target = FieldKey::new(MeshKey::ATMOSPHERE, FieldName::Temperature);
    let tendency =
      FieldKey::new(MeshKey::ATMOSPHERE, FieldName::TemperatureTendency);

    let mut factory = WorldFactory::new(WorldId(0), factory::earth())
      .cube_sphere_surface(CubeSphereShellSpec::uniform([2, 2, 1], 0.9, 1.0))
      .cube_sphere_atmosphere(CubeSphereShellSpec::uniform(
        [2, 2, 1],
        1.0,
        1.2,
      ));
    let coupler = factory
      .add_radial_stack_coupler(MeshKey::SURFACE, MeshKey::ATMOSPHERE)
      .unwrap();

    factory.register_field(source, SoaField::<1>::from_fn(24, |_| [300.0]));
    factory.register_field(target, SoaField::<1>::from_fn(24, |_| [250.0]));
    factory.register_field(tendency, SoaField::<1>::zeros(24));
    let stage_id = factory
      .add_scalar_interface_flux(coupler, source, target, tendency, 0.5)
      .unwrap();
    assert_eq!(stage_id.index(), 0);

    let mut world = factory.build().unwrap();
    world.tick(&Pool::default(), 1.0).unwrap();

    let tendency_field: &SoaField<1> = world.pleroma().read(tendency).unwrap();
    assert!(
      tendency_field
        .component(0)
        .as_ref()
        .iter()
        .any(|value| *value > 0.0),
      "warm surface should produce positive atmosphere-side tendencies"
    );
  }
}
