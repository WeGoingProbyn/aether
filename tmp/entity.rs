use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use crate::maths::matrix::Matrix;
use crate::maths::transformation::Transformation;

use super::materials::Materials;
use super::mesh::Mesh;
use super::uniforms::Uniforms;

pub trait IcsAsset
where
  Self: Sized,
{
  fn uid(arc: &Arc<Self>) -> usize {
    Arc::as_ptr(arc) as usize
  }
}

/// A struct representing an entity in a world space.
///
/// # Migration Note
///
/// This struct is being phased out in favor of ECS components:
/// - **Mesh data**: Use `MeshComponent` in ECS
/// - **Materials**: Stored in `MeshComponent.materials`
/// - **Transforms**: Use `InstanceTransforms` in ECS
/// - **Uniforms**: Use `InstanceUniforms` in ECS
///
/// Entity remains available for build-time operations and legacy compatibility.
/// For new code, prefer working with ECS components directly via `EcsWorld`.
///
/// # Example (Legacy)
/// ```ignore
/// let entity = EntityBuilder::new().build(&key, resources)?;
/// ```
///
/// # Example (ECS - Preferred)
/// ```ignore
/// let id = ecs.spawn();
/// ecs.insert_mesh(id, MeshComponent::new(mesh, materials));
/// ecs.insert_transforms(id, InstanceTransforms::single(transform));
/// ```
pub struct Entity {
  /// The number of instances of this entity.
  instances: usize,
  /// The mesh associated with this entity, stored as a reference-counted pointer.
  mesh: Arc<Mesh>,
  /// The uniforms for this entity, protected by a mutex for safe concurrent access.
  uniforms: Mutex<Uniforms>,
  /// The materials used by this entity, protected by a mutex for safe concurrent access.
  materials: Mutex<Materials>,
  /// A vector of root transformations, each protected by a mutex.
  root_transforms: Vec<Mutex<Transformation>>,
}

impl Hash for Entity {
  /// Computes a hash for the entity based on its mesh.
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.mesh.hash(state)
  }
}

impl Entity {
  /// Creates a new `Entity`.
  ///
  /// # Arguments
  ///
  /// * `instances` - The number of instances of this entity.
  /// * `mesh` - A shared reference to the mesh.
  /// * `uniforms` - The uniform data associated with the entity.
  /// * `materials` - The materials applied to the entity.
  /// * `root_transforms` - A vector of optional root transformations.
  ///   If `None`, a default transformation is used.
  ///
  /// # Returns
  ///
  /// A new `Entity` instance with initialized values.
  pub fn new(
    instances: usize,
    mesh: Arc<Mesh>,
    uniforms: Uniforms,
    materials: Materials,
    root_transforms: Vec<Option<Transformation>>,
  ) -> Entity {
    let root_transforms = root_transforms
      .into_iter()
      .map(|t| Mutex::new(t.unwrap_or(Transformation::new())))
      .collect::<Vec<_>>();
    Entity {
      instances,
      mesh,
      materials: Mutex::new(materials),
      uniforms: Mutex::new(uniforms),
      root_transforms,
    }
  }

  /// Returns the number of instances of this entity.
  pub fn instances(&self) -> usize {
    self.instances
  }

  /// Returns a reference to the mesh associated with this entity.
  pub fn mesh(&self) -> &Arc<Mesh> {
    &self.mesh
  }

  /// Returns a reference to the mutex protecting the entity's uniforms.
  pub fn uniforms(&self) -> &Mutex<Uniforms> {
    &self.uniforms
  }

  /// Returns a reference to the mutex protecting the entity's materials.
  pub fn materials(&self) -> &Mutex<Materials> {
    &self.materials
  }

  /// Returns a reference to the mutex-protected transformation of a specific instance.
  ///
  /// # Arguments
  ///
  /// * `instance` - The index of the instance.
  ///
  /// # Panics
  ///
  /// Panics if the `instance` index is out of bounds.
  pub fn root_transformation(&self, instance: usize) -> &Mutex<Transformation> {
    &self.root_transforms[instance]
  }

  /// Retrieves the model matrix from the root transformation of a specific instance.
  ///
  /// # Arguments
  ///
  /// * `instance` - The index of the instance.
  ///
  /// # Returns
  ///
  /// A 4x4 transformation matrix representing the root transformation of the instance.
  ///
  /// # Panics
  ///
  /// Panics if the `instance` index is out of bounds.
  pub fn root_transform(&self, instance: usize) -> Matrix<4, 4, f32> {
    self.root_transforms[instance].lock().unwrap().transform()
  }
}
impl IcsAsset for Entity {}
