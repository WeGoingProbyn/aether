//! Scene management for the ICS engine.
//!
//! A Scene is a view into the ECS filtered by SceneId. It provides
//! query methods to access entities, cameras, and lights for a specific scene.
//!
//! # Architecture
//!
//! Scene does not own entity data - the ECS owns all data. Scene holds:
//! - A reference to the shared ECS world
//! - A SceneId to filter queries
//! - Projection settings for rendering
//! - Animation system for the scene
//!
//! # Example
//!
//! ```ignore
//! let scene = IcsScene::new(scene_id, ecs.clone());
//!
//! // Query entities
//! for (id, mesh, transforms) in scene.query_meshes() {
//!     // ...
//! }
//!
//! // Get active camera
//! if let Some(camera) = scene.active_camera() {
//!     // ...
//! }
//! ```

use std::{collections::HashMap, sync::Arc};

use crate::{
  debugger::IcsError,
  ecs::{EcsWorld, EntityId, InstanceTransforms, MeshComponent, SceneId, SharedEcsWorld},
  maths::{camera::Camera, matrix::Matrix, projection::Projection, transformation::Transformation},
  ICS_WARN,
};

use super::{
  animation::AnimationSystem, hierarchy::HierarchyNode, lights::Light, pipeline::PipelineMapKey,
};

/// A scene is a view into the ECS filtered by SceneId.
///
/// Scene provides query methods to access entities, cameras, and lights
/// that belong to this scene. All data is owned by the ECS.
#[derive(Clone)]
pub struct IcsScene {
  /// The scene identifier for filtering ECS queries.
  scene_id: SceneId,
  /// Shared reference to the ECS world.
  ecs: SharedEcsWorld,
  /// Projection settings for rendering.
  projection: Projection,
  /// Index of the active camera (among cameras in this scene).
  active_camera_index: usize,
  /// Animation system for this scene.
  animations: AnimationSystem,
  /// Mapping from legacy pipeline entity keys to ECS entity IDs.
  legacy_entity_map: HashMap<PipelineMapKey, EntityId>,
}

impl IcsScene {
  /// Creates a new scene with the given ID and ECS reference.
  pub fn new(scene_id: SceneId, ecs: SharedEcsWorld) -> IcsScene {
    IcsScene {
      scene_id,
      ecs,
      projection: Projection::default(),
      active_camera_index: 0,
      animations: AnimationSystem::new(),
      legacy_entity_map: HashMap::new(),
    }
  }

  /// Creates a new empty scene (for backwards compatibility during migration).
  ///
  /// This creates a scene with no ECS reference. Use `new()` instead when possible.
  pub fn empty() -> IcsScene {
    use std::sync::RwLock;
    IcsScene {
      scene_id: SceneId(0),
      ecs: Arc::new(RwLock::new(EcsWorld::new())),
      projection: Projection::default(),
      active_camera_index: 0,
      animations: AnimationSystem::new(),
      legacy_entity_map: HashMap::new(),
    }
  }

  /// Registers a mapping from a legacy pipeline entity key to an ECS entity ID.
  pub fn register_legacy_entity(&mut self, legacy_key: PipelineMapKey, entity_id: EntityId) {
    self.legacy_entity_map.insert(legacy_key, entity_id);
  }

  /// Registers multiple legacy-to-ECS entity mappings.
  pub fn register_legacy_entities(&mut self, mappings: HashMap<PipelineMapKey, EntityId>) {
    self.legacy_entity_map.extend(mappings);
  }

  /// Returns the scene ID.
  pub fn scene_id(&self) -> SceneId {
    self.scene_id
  }

  /// Returns a reference to the shared ECS world.
  pub fn ecs(&self) -> &SharedEcsWorld {
    &self.ecs
  }

  // =========================================================================
  // Mesh entity queries
  // =========================================================================

  /// Queries all mesh entities in this scene.
  ///
  /// Returns an iterator over (EntityId, MeshComponent, InstanceTransforms).
  pub fn query_meshes(&self) -> Vec<(EntityId, MeshComponent, InstanceTransforms)> {
    let ecs = self.ecs.read().unwrap();
    ecs
      .iter_scene_meshes(self.scene_id)
      .map(|(id, mesh, transforms)| (id, mesh.clone(), transforms.clone()))
      .collect()
  }

  /// Returns the number of mesh entities in this scene.
  pub fn mesh_count(&self) -> usize {
    let ecs = self.ecs.read().unwrap();
    ecs.iter_scene_meshes(self.scene_id).count()
  }

  // =========================================================================
  // Camera queries
  // =========================================================================

  /// Queries all cameras in this scene.
  pub fn query_cameras(&self) -> Vec<(EntityId, Camera)> {
    let ecs = self.ecs.read().unwrap();
    ecs
      .iter_scene_cameras(self.scene_id)
      .map(|(id, cam)| (id, cam.clone()))
      .collect()
  }

  /// Returns the cameras in this scene (for backwards compatibility).
  pub fn cameras(&self) -> Vec<Camera> {
    self
      .query_cameras()
      .into_iter()
      .map(|(_, cam)| cam)
      .collect()
  }

  /// Returns the active camera for this scene.
  pub fn active_camera(&self) -> Option<Camera> {
    let cameras = self.query_cameras();
    cameras
      .get(self.active_camera_index)
      .map(|(_, cam)| cam.clone())
  }

  /// Returns the EntityId of the active camera.
  pub fn active_camera_id(&self) -> Option<EntityId> {
    let cameras = self.query_cameras();
    cameras.get(self.active_camera_index).map(|(id, _)| *id)
  }

  /// Sets the active camera by index.
  pub fn set_active_camera(&mut self, index: usize) {
    self.active_camera_index = index;
  }

  /// Returns the index of the currently active camera.
  pub fn current_camera(&self) -> usize {
    self.active_camera_index
  }

  /// Gets a mutable reference to a camera in the ECS.
  pub fn camera_mut(&self, id: EntityId) -> Option<Camera> {
    let ecs = self.ecs.read().unwrap();
    ecs.camera(id).cloned()
  }

  /// Updates a camera in the ECS.
  pub fn update_camera(&self, id: EntityId, camera: Camera) {
    let mut ecs = self.ecs.write().unwrap();
    ecs.insert_camera(id, camera);
  }

  // =========================================================================
  // Light queries
  // =========================================================================

  /// Queries all lights in this scene.
  pub fn query_lights(&self) -> Vec<(EntityId, Light)> {
    let ecs = self.ecs.read().unwrap();
    ecs
      .iter_scene_lights(self.scene_id)
      .map(|(id, light)| (id, light.clone()))
      .collect()
  }

  /// Returns all lights in this scene (for backwards compatibility).
  pub fn lights_vec(&self) -> Vec<Light> {
    self
      .query_lights()
      .into_iter()
      .map(|(_, light)| light)
      .collect()
  }

  // =========================================================================
  // Projection
  // =========================================================================

  /// Returns a reference to the scene's projection.
  pub fn projection(&self) -> &Projection {
    &self.projection
  }

  /// Returns a mutable reference to the scene's projection.
  pub fn projection_mut(&mut self) -> &mut Projection {
    &mut self.projection
  }

  /// Sets the scene's projection to a new value.
  pub fn set_projection(&mut self, proj: Projection) {
    self.projection = proj;
  }

  // =========================================================================
  // Animation
  // =========================================================================

  /// Returns a reference to the scene's animation system.
  pub fn animations(&self) -> &AnimationSystem {
    &self.animations
  }

  /// Returns a mutable reference to the scene's animation system.
  pub fn animations_mut(&mut self) -> &mut AnimationSystem {
    &mut self.animations
  }

  /// Updates all animation players and propagates transforms.
  pub fn update_animations(&mut self, total_time: f32) -> Result<(), IcsError> {
    // Update animation players
    for player in self.animations.players_mut() {
      player.update_with_total_time(total_time)?;
    }

    // Propagate transforms for all mesh entities
    self.propagate_all_transforms();

    Ok(())
  }

  // =========================================================================
  // Transform propagation
  // =========================================================================

  /// Propagates transforms for all mesh entities in the scene.
  ///
  /// This updates the hierarchy node transforms to contain local hierarchy
  /// transforms only (parent-child relationships). Root instance transforms
  /// are applied separately during uniform sync to support multi-instance.
  pub fn propagate_all_transforms(&self) {
    let meshes = self.query_meshes();

    for (_entity_id, mesh_comp, _transforms) in meshes {
      // Propagate with identity - stores hierarchy-local transforms only
      // Per-instance root transforms are applied in sync_mvp_uniforms
      self.propagate_hierarchy_transforms(&mesh_comp);
    }
  }

  /// Propagates transforms through the mesh hierarchy using identity root.
  ///
  /// This stores the cumulative local transform (parent * child chain) in each
  /// node's `propagated_transform`. The per-instance root transform is multiplied
  /// in during uniform sync, allowing multi-instance entities to work correctly.
  fn propagate_hierarchy_transforms(&self, mesh_comp: &MeshComponent) {
    use crate::maths::matrix::Matrix;
    let identity = Matrix::<4, 4, f32>::identity();

    let mut hierarchy = match mesh_comp.mesh.hierarchy().lock() {
      Ok(h) => h,
      Err(err) => {
        ICS_WARN!("Scene: failed to lock hierarchy: {}", err);
        return;
      }
    };

    for root in &mut hierarchy.root_nodes {
      if let Some(transform) = root.relative_transform {
        root.propagated_transform = transform.transform();
      } else if let Some(matrix) = root.relative_matrix {
        root.propagated_transform = matrix;
      } else {
        root.propagated_transform = identity;
      }
      root.propagate_transform(identity);
    }
  }

  // =========================================================================
  // MVP Uniform Synchronization
  // =========================================================================

  /// Synchronizes MVP (Model-View-Projection) uniforms for all entities.
  ///
  /// This method updates the model, view, and projection matrices in each
  /// entity's uniforms based on:
  /// - Model: from each node's `propagated_transform`
  /// - View: from the active camera's view matrix
  /// - Proj: from the scene's projection matrix
  ///
  /// Call this after `propagate_all_transforms()` to ensure uniforms are
  /// ready for rendering.
  ///
  /// # Arguments
  /// * `pipelines` - The render pipelines containing entity references
  pub fn sync_mvp_uniforms(
    &mut self,
    pipelines: &[std::sync::Arc<std::sync::Mutex<super::pipeline::IcsPipeline>>],
  ) {
    use crate::structures::uniforms::{ConcreteUniform, Pushable};

    // Get view matrix from active camera
    let mut updated_uniforms = 0usize;
    let view_matrix = if let Some(mut camera) = self.active_camera() {
      crate::ICS_TRACE!(
        "Scene: MVP sync camera pos={:?} look_at={:?}",
        camera.position,
        camera.look_at
      );
      camera.view_matrix()
    } else {
      crate::ICS_TRACE!("Scene: MVP sync has no active camera");
      crate::maths::matrix::Matrix::<4, 4, f32>::identity()
    };

    // Get projection matrix
    let proj_matrix = self.projection.projection_matrix();
    crate::ICS_TRACE!(
      "Scene: MVP sync projection fov={} aspect={} near={} far={}",
      self.projection.fov,
      self.projection.aspect_ratio,
      self.projection.near,
      self.projection.far
    );

    // Iterate through all pipelines and update MVP uniforms
    for pipeline_handle in pipelines {
      let pipeline = match pipeline_handle.try_lock() {
        Ok(pipeline) => pipeline,
        Err(std::sync::TryLockError::WouldBlock) => {
          crate::ICS_TRACE!("Scene: skipping MVP sync for busy pipeline");
          continue;
        }
        Err(std::sync::TryLockError::Poisoned(err)) => {
          crate::ICS_WARN!("Scene: pipeline lock poisoned during MVP sync: {}", err);
          continue;
        }
      };

      for (entity_key, nodes) in pipeline.parts() {
        let Some(entity) = pipeline.entities().get(entity_key) else {
          continue;
        };

        let live_transforms = {
          let hierarchy = match entity.mesh().hierarchy().try_lock() {
            Ok(hierarchy) => hierarchy,
            Err(std::sync::TryLockError::WouldBlock) => {
              crate::ICS_TRACE!(
                "Scene: skipping MVP sync for busy hierarchy (entity={:?})",
                entity_key
              );
              continue;
            }
            Err(std::sync::TryLockError::Poisoned(err)) => {
              crate::ICS_WARN!(
                "Scene: hierarchy lock poisoned during MVP sync (entity={:?}): {}",
                entity_key,
                err
              );
              continue;
            }
          };
          let mut map = std::collections::HashMap::new();
          for node in hierarchy.iter() {
            map.insert(node.name().clone(), node.propagated_transform);
          }
          map
        };

        let ecs_transforms = self.legacy_entity_map.get(entity_key).and_then(|id| {
          let ecs = match self.ecs.read() {
            Ok(ecs) => ecs,
            Err(err) => {
              crate::ICS_WARN!(
                "Scene: ECS lock poisoned during MVP sync (entity={:?}): {}",
                entity_key,
                err
              );
              return None;
            }
          };
          ecs.transforms(*id).cloned()
        });

        let uniforms = match entity.uniforms().try_lock() {
          Ok(uniforms) => uniforms,
          Err(std::sync::TryLockError::WouldBlock) => {
            crate::ICS_TRACE!(
              "Scene: skipping MVP sync for busy uniforms (entity={:?})",
              entity_key
            );
            continue;
          }
          Err(std::sync::TryLockError::Poisoned(err)) => {
            crate::ICS_WARN!(
              "Scene: uniforms lock poisoned during MVP sync (entity={:?}): {}",
              entity_key,
              err
            );
            continue;
          }
        };

        for node in nodes {
          // Get hierarchy-local transform from node's propagated transform
          // This does NOT include the instance's root transform yet
          let hierarchy_local = match live_transforms.get(node.name()) {
            Some(mat) => *mat,
            None => {
              crate::ICS_TRACE!(
                "Scene: missing live node '{}' for MVP sync (entity={:?})",
                node.name(),
                entity_key
              );
              node.propagated_transform
            }
          };

          // Get the MVP uniform for this node
          let Some(mvp_uniforms) =
            uniforms.uniforms(node.name(), &ConcreteUniform::ModelViewProject)
          else {
            continue;
          };

          // Update each MVP uniform instance with per-instance root transform
          for (instance_idx, uniform_guard) in mvp_uniforms.iter().enumerate() {
            let mut uniform = match uniform_guard.try_lock() {
              Ok(uniform) => uniform,
              Err(std::sync::TryLockError::WouldBlock) => {
                crate::ICS_TRACE!(
                  "Scene: skipping MVP sync for busy uniform (entity={:?}, node={}, instance={})",
                  entity_key,
                  node.name(),
                  instance_idx
                );
                continue;
              }
              Err(std::sync::TryLockError::Poisoned(err)) => {
                crate::ICS_WARN!(
                                    "Scene: uniform lock poisoned during MVP sync (entity={:?}, node={}, instance={}): {}",
                                    entity_key,
                                    node.name(),
                                    instance_idx,
                                    err
                                );
                continue;
              }
            };

            // Compute model matrix: root_transform[instance] * hierarchy_local
            // This allows each instance to have its own world position
            let model_matrix = ecs_transforms
              .as_ref()
              .and_then(|transforms| transforms.root_transforms.get(instance_idx))
              .map(|transform| transform.transform() * hierarchy_local)
              .unwrap_or_else(|| {
                if instance_idx < entity.instances() {
                  entity.root_transform(instance_idx) * hierarchy_local
                } else {
                  // Fallback for mismatched instance counts
                  hierarchy_local
                }
              });

            // Push model matrix (index 0) - use column-major for GLSL
            let _ = uniform.push(0, &Pushable::Mat4(model_matrix.as_slice_column_major()));
            // Push view matrix (index 1)
            let _ = uniform.push(1, &Pushable::Mat4(view_matrix.as_slice_column_major()));
            // Push projection matrix (index 2)
            let _ = uniform.push(2, &Pushable::Mat4(proj_matrix.as_slice_column_major()));

            uniform.needs_update = true;
            updated_uniforms += 1;
          }
        }
      }
    }

    crate::ICS_TRACE!("Scene: MVP sync updated {} uniforms", updated_uniforms);
  }

  // =========================================================================
  // Render extraction
  // =========================================================================

  /// Extracts a RenderView for this scene.
  ///
  /// This is a convenience method that calls EcsWorld::extract_render_view
  /// with this scene's ID.
  pub fn extract_render_view(&self) -> crate::ecs::render::RenderView {
    let ecs = self.ecs.read().unwrap();
    ecs.extract_render_view(self.scene_id, None)
  }

  /// Extracts a RenderView for this scene with a specific render layer.
  pub fn extract_render_view_for_layer(
    &self,
    layer: crate::ecs::RenderLayer,
  ) -> crate::ecs::render::RenderView {
    let ecs = self.ecs.read().unwrap();
    ecs.extract_render_view(self.scene_id, Some(layer))
  }

  // =========================================================================
  // Node access (for compatibility)
  // =========================================================================

  /// Searches for a node in a mesh and applies a closure if found.
  pub fn with_mesh_node<F>(
    &self,
    mesh_comp: &MeshComponent,
    node_name: &str,
    f: &mut F,
  ) -> Result<(), IcsError>
  where
    F: FnMut(&mut HierarchyNode),
  {
    let mut hierarchy = mesh_comp.mesh.hierarchy().lock().map_err(|e| {
      crate::ICS_ERROR!(
          why: format!("Scene: failed to lock hierarchy: {}", e),
          fix: "Ensure no panic poisoned the mesh hierarchy mutex"
      )
    })?;

    if hierarchy.find_node_mut(node_name, f) {
      Ok(())
    } else {
      Err(crate::ICS_ERROR!(
          why: "Scene: Could not find node within mesh hierarchy",
          fix: "Ensure you're referencing the correct node name"
      ))
    }
  }
}

impl Default for IcsScene {
  fn default() -> Self {
    Self::empty()
  }
}

/// Represents a single node within a SceneGraph (legacy, kept for compatibility).
#[derive(Debug, Clone)]
pub struct SceneNode {
  /// Unique identifier for the node.
  pub id: u32,
  /// Optional parent node ID.
  pub parent: Option<u32>,
  /// List of child nodes.
  pub children: Vec<Box<SceneNode>>,
  /// The relative transform for this node.
  pub relative_transform: Transformation,
  /// Cached world transformation.
  pub propagated_transform: Matrix<4, 4, f32>,
}

impl SceneNode {
  /// Creates a new `Node`.
  pub fn new(id: u32, parent: Option<u32>, relative_transform: Transformation) -> SceneNode {
    SceneNode {
      id,
      parent,
      children: Vec::new(),
      relative_transform,
      propagated_transform: Matrix::<4, 4, f32>::new(),
    }
  }

  pub fn transform_mut(&mut self) -> &mut Transformation {
    &mut self.relative_transform
  }

  /// Adds a child node to this node.
  pub fn add_child(&mut self, child: Box<SceneNode>) -> Result<(), IcsError> {
    if self.contains_node(child.id) {
      return Err(crate::ICS_ERROR!(
          why: "SceneNode: Trying to add parent as a child to itself",
          fix: "Don't create cycles in scene graph"
      ));
    }
    let mut child = *child;
    child.parent = Some(self.id);
    self.children.push(Box::new(child));
    Ok(())
  }

  fn contains_node(&self, id: u32) -> bool {
    if self.id == id {
      return true;
    }
    for child in &self.children {
      if child.contains_node(id) {
        return true;
      }
    }
    false
  }

  /// Updates the propagated transformations of this node and its children.
  pub fn propagate_transform(&mut self, parent_transform: Matrix<4, 4, f32>) {
    self.propagated_transform = parent_transform * self.relative_transform.transform();
    for child in self.children.iter_mut() {
      child.propagate_transform(self.propagated_transform);
    }
  }

  /// Traverses the node and its children, applying a function to each node.
  pub fn traverse<F>(&self, func: &F)
  where
    F: Fn(&SceneNode),
  {
    func(self);
    for child in &self.children {
      child.traverse(func);
    }
  }
}

// Legacy types kept for migration compatibility
#[derive(Debug, Hash, Eq, PartialEq, Copy, Clone)]
pub struct SceneMapKey(pub usize);
