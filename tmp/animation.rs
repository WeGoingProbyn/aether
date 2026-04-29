//! Animation system for the ICS engine.
//!
//! # ECS Integration
//!
//! The animation system works with both legacy `Entity` and ECS `MeshComponent`.
//! Since both share the same `Arc<Mesh>`, animation updates affect both systems.
//!
//! For ECS-based scenes:
//! 1. AnimationPlayer still uses `Arc<Entity>` internally to access the mesh hierarchy
//! 2. The `entity_id` field tracks the corresponding ECS EntityId
//! 3. Hierarchy modifications are shared via `Arc<Mesh>`
//!
//! # Example
//! ```ignore
//! // Create animation player with ECS tracking
//! let player = AnimationPlayer::new(clip, entity)?
//!     .with_entity_id(entity_id);
//!
//! // Add to scene's animation system
//! scene.animations_mut().add_player(player);
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::debugger::IcsError;
use crate::ecs::EntityId;
use crate::maths::quaternion::Quaternion;
use crate::maths::transformation::Transformation;
use crate::memory::memory::Memory;
use crate::structures::entity::Entity;
use crate::structures::hierarchy::{HierarchyNode, NodeId};
use crate::utility::file_io::file_to_string;
use crate::utility::json::{Gltf, GltfAccessor, JsonParser};
use crate::{ICS_ERROR, ICS_WARN};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AnimationInterpolation {
  Linear,
  Step,
  CubicSpline,
}

impl AnimationInterpolation {
  fn from_gltf(value: &str) -> Result<Self, IcsError> {
    match value.to_uppercase().as_str() {
      "LINEAR" => Ok(AnimationInterpolation::Linear),
      "STEP" => Ok(AnimationInterpolation::Step),
      "CUBICSPLINE" => Ok(AnimationInterpolation::CubicSpline),
      _ => Err(ICS_ERROR!(
        why: format!("Animation: Unsupported interpolation '{}'", value),
        fix: "Use LINEAR, STEP, or CUBICSPLINE"
      )),
    }
  }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AnimationPath {
  Translation,
  Rotation,
  Scale,
  Weights,
}

impl AnimationPath {
  fn from_gltf(value: &str) -> Result<Self, IcsError> {
    match value.to_lowercase().as_str() {
      "translation" => Ok(AnimationPath::Translation),
      "rotation" => Ok(AnimationPath::Rotation),
      "scale" => Ok(AnimationPath::Scale),
      "weights" => Ok(AnimationPath::Weights),
      _ => Err(ICS_ERROR!(
        why: format!("Animation: Unsupported target path '{}'", value),
        fix: "Use translation, rotation, scale, or weights"
      )),
    }
  }
}

#[derive(Debug, Clone)]
pub struct AnimationChannel {
  sampler: usize,
  target_node: usize,
  path: AnimationPath,
}

#[derive(Debug, Clone)]
enum AnimationSamplerOutput {
  Vec3(Vec<[f32; 3]>),
  Vec4(Vec<[f32; 4]>),
}

#[derive(Debug, Clone)]
pub struct AnimationSampler {
  input: Vec<f32>,
  output: Option<AnimationSamplerOutput>,
  interpolation: AnimationInterpolation,
}

impl AnimationSampler {
  fn new(input: Vec<f32>, interpolation: AnimationInterpolation) -> Self {
    AnimationSampler {
      input,
      output: None,
      interpolation,
    }
  }

  fn duration(&self) -> f32 {
    self.input.last().copied().unwrap_or(0.0)
  }

  fn ensure_output(
    &mut self,
    gltf: &Gltf,
    accessor_index: usize,
    path: AnimationPath,
  ) -> Result<(), IcsError> {
    match path {
      AnimationPath::Translation | AnimationPath::Scale => match &self.output {
        Some(AnimationSamplerOutput::Vec3(_)) => Ok(()),
        Some(AnimationSamplerOutput::Vec4(_)) => Err(ICS_ERROR!(
          why: "Animation: Sampler output type does not match channel path",
          fix: "Ensure sampler output accessor type matches channel path"
        )),
        None => {
          let values = read_accessor_vec3(gltf, accessor_index)?;
          let expected = match self.interpolation {
            AnimationInterpolation::CubicSpline => self.input.len() * 3,
            _ => self.input.len(),
          };
          if values.len() != expected {
            return Err(ICS_ERROR!(
              why: "Animation: Sampler output count does not match input count",
              fix: "Ensure output accessor count matches input accessor count"
            ));
          }
          self.output = Some(AnimationSamplerOutput::Vec3(values));
          Ok(())
        }
      },
      AnimationPath::Rotation => match &self.output {
        Some(AnimationSamplerOutput::Vec4(_)) => Ok(()),
        Some(AnimationSamplerOutput::Vec3(_)) => Err(ICS_ERROR!(
          why: "Animation: Sampler output type does not match channel path",
          fix: "Ensure sampler output accessor type matches channel path"
        )),
        None => {
          let values = read_accessor_vec4(gltf, accessor_index)?;
          let expected = match self.interpolation {
            AnimationInterpolation::CubicSpline => self.input.len() * 3,
            _ => self.input.len(),
          };
          if values.len() != expected {
            return Err(ICS_ERROR!(
              why: "Animation: Sampler output count does not match input count",
              fix: "Ensure output accessor count matches input accessor count"
            ));
          }
          self.output = Some(AnimationSamplerOutput::Vec4(values));
          Ok(())
        }
      },
      AnimationPath::Weights => Err(ICS_ERROR!(
        why: "Animation: Weights channels are not implemented yet",
        fix: "Skip weights channels until skinning support lands"
      )),
    }
  }

  fn sample_vec3(&self, time: f32) -> Result<[f32; 3], IcsError> {
    let output = match &self.output {
      Some(AnimationSamplerOutput::Vec3(values)) => values,
      Some(AnimationSamplerOutput::Vec4(_)) => {
        return Err(ICS_ERROR!(
          why: "Animation: Sampler output type mismatch for vec3 sampling",
          fix: "Ensure channel path matches sampler output type"
        ))
      }
      None => {
        return Err(ICS_ERROR!(
          why: "Animation: Sampler output not initialized",
          fix: "Ensure the sampler output accessor is loaded"
        ))
      }
    };

    if output.is_empty() {
      return Err(ICS_ERROR!(
        why: "Animation: Sampler output is empty",
        fix: "Ensure the output accessor has data"
      ));
    }

    let (i0, i1, t) = sample_indices(&self.input, time);
    let v0 = output[i0];
    let v1 = output[i1];

    match self.interpolation {
      AnimationInterpolation::Step => Ok(v0),
      AnimationInterpolation::Linear => Ok(lerp_vec3(v0, v1, t)),
      AnimationInterpolation::CubicSpline => Err(ICS_ERROR!(
        why: "Animation: CUBICSPLINE interpolation not implemented",
        fix: "Use LINEAR/STEP or add cubic spline support"
      )),
    }
  }

  fn sample_vec4(&self, time: f32) -> Result<[f32; 4], IcsError> {
    let output = match &self.output {
      Some(AnimationSamplerOutput::Vec4(values)) => values,
      Some(AnimationSamplerOutput::Vec3(_)) => {
        return Err(ICS_ERROR!(
          why: "Animation: Sampler output type mismatch for vec4 sampling",
          fix: "Ensure channel path matches sampler output type"
        ))
      }
      None => {
        return Err(ICS_ERROR!(
          why: "Animation: Sampler output not initialized",
          fix: "Ensure the sampler output accessor is loaded"
        ))
      }
    };

    if output.is_empty() {
      return Err(ICS_ERROR!(
        why: "Animation: Sampler output is empty",
        fix: "Ensure the output accessor has data"
      ));
    }

    let (i0, i1, t) = sample_indices(&self.input, time);
    let v0 = output[i0];
    let v1 = output[i1];

    match self.interpolation {
      AnimationInterpolation::Step => Ok(v0),
      AnimationInterpolation::Linear => Ok(nlerp_quat(v0, v1, t)),
      AnimationInterpolation::CubicSpline => Err(ICS_ERROR!(
        why: "Animation: CUBICSPLINE interpolation not implemented",
        fix: "Use LINEAR/STEP or add cubic spline support"
      )),
    }
  }
}

#[derive(Debug, Clone)]
pub struct AnimationClip {
  name: Option<String>,
  channels: Vec<AnimationChannel>,
  samplers: Vec<AnimationSampler>,
  duration: f32,
  node_names: Vec<Vec<String>>,
}

impl AnimationClip {
  pub fn from_gltf(gltf: &Gltf) -> Result<Vec<AnimationClip>, IcsError> {
    let node_names = build_node_name_map(gltf);
    let mut clips = Vec::new();

    for animation in &gltf.animations {
      let mut samplers = Vec::with_capacity(animation.samplers.len());
      for sampler in &animation.samplers {
        let input = read_accessor_f32(gltf, sampler.input)?;
        let interpolation = AnimationInterpolation::from_gltf(&sampler.interpolation)?;
        samplers.push(AnimationSampler::new(input, interpolation));
      }

      let mut channels = Vec::new();
      for channel in &animation.channels {
        let path = AnimationPath::from_gltf(&channel.target.path)?;
        if path == AnimationPath::Weights {
          ICS_WARN!("Animation: Skipping weights channel (skinning not implemented).");
          continue;
        }

        let sampler_index = channel.sampler;
        if sampler_index >= samplers.len() {
          return Err(ICS_ERROR!(
            why: "Animation: Channel references invalid sampler index",
            fix: "Ensure channel sampler indices are valid"
          ));
        }
        let output_accessor = animation.samplers[sampler_index].output;
        samplers[sampler_index].ensure_output(gltf, output_accessor, path)?;

        channels.push(AnimationChannel {
          sampler: sampler_index,
          target_node: channel.target.node,
          path,
        });
      }

      let duration = samplers
        .iter()
        .map(AnimationSampler::duration)
        .fold(0.0, f32::max);

      clips.push(AnimationClip {
        name: animation.name.clone(),
        channels,
        samplers,
        duration,
        node_names: node_names.clone(),
      });
    }

    Ok(clips)
  }

  pub fn from_gltf_path(path: &str) -> Result<Vec<AnimationClip>, IcsError> {
    let gltf = load_gltf_from_path(path)?;
    AnimationClip::from_gltf(&gltf)
  }

  pub fn duration(&self) -> f32 {
    self.duration
  }

  pub fn name(&self) -> Option<&str> {
    self.name.as_deref()
  }
}

#[derive(Debug, Clone, Copy)]
pub enum PlaybackMode {
  Loop,
  Clamp,
}

#[derive(Clone)]
/// Plays an animation clip on an entity.
///
/// The player samples animation channels and applies transformations to
/// mesh hierarchy nodes. Works with both legacy Entity and ECS MeshComponent
/// since they share the same `Arc<Mesh>`.
pub struct AnimationPlayer {
  clip: Arc<AnimationClip>,
  entity: Arc<Entity>,
  /// ECS EntityId for tracking (optional, for ECS integration).
  entity_id: Option<EntityId>,
  bindings: HashMap<usize, Vec<NodeId>>,
  mode: PlaybackMode,
  speed: f32,
  local_time: f32,
  last_total_time: Option<f32>,
}

impl AnimationPlayer {
  pub fn new(clip: Arc<AnimationClip>, entity: Arc<Entity>) -> Result<Self, IcsError> {
    let mut bindings: HashMap<usize, Vec<NodeId>> = HashMap::new();
    {
      let hierarchy = entity.mesh().hierarchy().lock().map_err(|e| {
        ICS_ERROR!(
          why: format!("Animation: failed to lock hierarchy: {}", e),
          fix: "Ensure no panic poisoned the entity hierarchy mutex"
        )
      })?;

      for channel in &clip.channels {
        let Some(names) = clip.node_names.get(channel.target_node) else {
          ICS_WARN!(
            "Animation: Channel targets missing node index {}",
            channel.target_node
          );
          continue;
        };
        for name in names {
          let found = hierarchy.find_node(name, &|_| {});
          if !found {
            ICS_WARN!("Animation: Node '{}' not found in entity hierarchy", name);
            continue;
          }
          let entry = bindings.entry(channel.target_node).or_default();
          let node_id = NodeId::new(name.clone());
          if !entry.iter().any(|existing| existing == &node_id) {
            entry.push(node_id);
          }
        }
      }
    }

    if bindings.is_empty() {
      ICS_WARN!("Animation: No bindings were created for this player.");
    }

    Ok(AnimationPlayer {
      clip,
      entity,
      entity_id: None,
      bindings,
      mode: PlaybackMode::Loop,
      speed: 1.0,
      local_time: 0.0,
      last_total_time: None,
    })
  }

  /// Sets the ECS EntityId for this player.
  ///
  /// This enables tracking which ECS entity this animation affects.
  /// Use this when creating players for ECS-based scenes.
  pub fn with_entity_id(mut self, id: EntityId) -> Self {
    self.entity_id = Some(id);
    self
  }

  /// Returns the ECS EntityId if set.
  pub fn entity_id(&self) -> Option<EntityId> {
    self.entity_id
  }

  /// Sets the ECS EntityId.
  pub fn set_entity_id(&mut self, id: EntityId) {
    self.entity_id = Some(id);
  }

  pub fn entity(&self) -> &Arc<Entity> {
    &self.entity
  }

  pub fn set_playback_mode(&mut self, mode: PlaybackMode) {
    self.mode = mode;
  }

  pub fn set_speed(&mut self, speed: f32) {
    self.speed = speed;
  }

  pub fn local_time(&self) -> f32 {
    self.local_time
  }

  pub fn update_with_delta(&mut self, delta_t: f32) -> Result<(), IcsError> {
    let delta_t = delta_t.max(0.0);
    self.local_time = self.advance_time(self.local_time + delta_t * self.speed);
    self.apply()
  }

  pub fn update_with_total_time(&mut self, total_time: f32) -> Result<(), IcsError> {
    let delta_t = match self.last_total_time {
      Some(last) => total_time - last,
      None => 0.0,
    };
    self.last_total_time = Some(total_time);
    self.update_with_delta(delta_t)
  }

  fn advance_time(&self, time: f32) -> f32 {
    let duration = self.clip.duration();
    if duration <= 0.0 {
      return 0.0;
    }
    match self.mode {
      PlaybackMode::Loop => time.rem_euclid(duration),
      PlaybackMode::Clamp => time.clamp(0.0, duration),
    }
  }

  fn apply(&mut self) -> Result<(), IcsError> {
    let time = self.local_time;
    let mut hierarchy = self.entity.mesh().hierarchy().lock().map_err(|e| {
      ICS_ERROR!(
        why: format!("Animation: failed to lock hierarchy: {}", e),
        fix: "Ensure no panic poisoned the entity hierarchy mutex"
      )
    })?;
    for channel in &self.clip.channels {
      let Some(node_ids) = self.bindings.get(&channel.target_node) else {
        continue;
      };
      let sampler = self.clip.samplers.get(channel.sampler).ok_or_else(|| {
        ICS_ERROR!(
          why: "Animation: Channel references invalid sampler index",
          fix: "Ensure channel sampler indices are valid"
        )
      })?;

      for node_id in node_ids {
        let apply_channel = |node: &mut HierarchyNode| -> Result<(), IcsError> {
          let Some(transform) = ensure_trs(node) else {
            ICS_WARN!(
              "Animation: failed to ensure transform for node '{}' on channel",
              node.name()
            );
            return Ok(());
          };
          match channel.path {
            AnimationPath::Translation => {
              let value = sampler.sample_vec3(time)?;
              transform.set_translation(value);
            }
            AnimationPath::Scale => {
              let value = sampler.sample_vec3(time)?;
              transform.set_scale(value);
            }
            AnimationPath::Rotation => {
              let value = sampler.sample_vec4(time)?;
              let quat = Quaternion::from([value[3], value[0], value[1], value[2]]);
              transform.set_rotation(quat.to_euler_angles());
            }
            AnimationPath::Weights => {}
          }
          Ok(())
        };

        let mut result = Ok(());
        let mut handler = |node: &mut HierarchyNode| {
          result = apply_channel(node);
        };
        if hierarchy.find_node_mut(node_id.as_str(), &mut handler) {
          result?;
        }
      }
    }
    Ok(())
  }
}

#[derive(Default, Clone)]
pub struct AnimationSystem {
  players: Vec<AnimationPlayer>,
}

impl AnimationSystem {
  pub fn new() -> Self {
    AnimationSystem {
      players: Vec::new(),
    }
  }

  pub fn add_player(&mut self, player: AnimationPlayer) {
    self.players.push(player);
  }

  pub fn players(&self) -> &Vec<AnimationPlayer> {
    &self.players
  }

  pub fn players_mut(&mut self) -> &mut Vec<AnimationPlayer> {
    &mut self.players
  }
}

fn ensure_trs(node: &mut HierarchyNode) -> Option<&mut Transformation> {
  if node.relative_transform.is_none() {
    let transform = if let Some(matrix) = node.relative_matrix {
      Transformation::from_matrix(matrix)
    } else {
      Transformation::new()
    };
    node.relative_transform = Some(transform);
    node.relative_matrix = None;
  }
  node.relative_transform.as_mut()
}

fn lerp_vec3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
  [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ]
}

fn nlerp_quat(a: [f32; 4], mut b: [f32; 4], t: f32) -> [f32; 4] {
  let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  if dot < 0.0 {
    b = [-b[0], -b[1], -b[2], -b[3]];
  }
  let mut out = [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
    a[3] + (b[3] - a[3]) * t,
  ];
  let len = (out[0] * out[0] + out[1] * out[1] + out[2] * out[2] + out[3] * out[3]).sqrt();
  if len > 0.0 {
    out = [out[0] / len, out[1] / len, out[2] / len, out[3] / len];
  }
  out
}

fn sample_indices(input: &[f32], time: f32) -> (usize, usize, f32) {
  if input.len() <= 1 {
    return (0, 0, 0.0);
  }
  if time <= input[0] {
    return (0, 0, 0.0);
  }
  let last = input.len() - 1;
  if time >= input[last] {
    return (last, last, 0.0);
  }

  match input.binary_search_by(|probe| {
    probe
      .partial_cmp(&time)
      .unwrap_or(std::cmp::Ordering::Greater)
  }) {
    Ok(index) => (index, index, 0.0),
    Err(next) => {
      let prev = next - 1;
      let t0 = input[prev];
      let t1 = input[next];
      let denom = t1 - t0;
      let t = if denom.abs() > f32::EPSILON {
        (time - t0) / denom
      } else {
        0.0
      };
      (prev, next, t)
    }
  }
}

fn build_node_name_map(gltf: &Gltf) -> Vec<Vec<String>> {
  let mut parts_in_use: HashMap<String, usize> = HashMap::new();
  let mut map: Vec<Vec<String>> = vec![Vec::new(); gltf.nodes.len()];

  for (node_idx, node) in gltf.nodes.iter().enumerate() {
    let node_name = node
      .name
      .clone()
      .unwrap_or_else(|| format!("node_{}", node_idx));

    if let Some(mesh_idx) = node.mesh {
      let mesh = &gltf.meshes[mesh_idx];
      for _ in &mesh.primitives {
        let unique_name = if let Some(count) = parts_in_use.get_mut(&node_name) {
          *count += 1;
          format!("{}_{}", node_name, count)
        } else {
          parts_in_use.insert(node_name.clone(), 0);
          format!("{}_{}", node_name, 0)
        };
        map[node_idx].push(unique_name);
      }
    } else {
      map[node_idx].push(node_name);
    }
  }

  map
}

fn load_gltf_from_path(path: &str) -> Result<Gltf, IcsError> {
  let base = Path::new(path)
    .parent()
    .and_then(|p| p.to_str())
    .ok_or_else(|| {
      ICS_ERROR!(
        why: "Animation: Could not determine base path for glTF",
        fix: "Use a path that includes a parent directory"
      )
    })?;
  let json_str = file_to_string(path)?;
  let mut parser = JsonParser::new(&json_str);
  let json_value = parser.parse()?;
  Gltf::from_json(json_value, format!("{}/", base))
}

fn read_accessor_f32(gltf: &Gltf, accessor_index: usize) -> Result<Vec<f32>, IcsError> {
  let accessor = &gltf.accessors[accessor_index];
  if accessor.bytes_type != "SCALAR" {
    return Err(ICS_ERROR!(
      why: "Animation: Accessor type is not SCALAR",
      fix: "Ensure input accessor is SCALAR"
    ));
  }
  if accessor.component_type != 5126 {
    return Err(ICS_ERROR!(
      why: "Animation: Accessor component type is not FLOAT",
      fix: "Ensure input accessor uses FLOAT component type"
    ));
  }
  read_accessor_data::<f32>(gltf, accessor)
}

fn read_accessor_vec3(gltf: &Gltf, accessor_index: usize) -> Result<Vec<[f32; 3]>, IcsError> {
  let accessor = &gltf.accessors[accessor_index];
  if accessor.bytes_type != "VEC3" {
    return Err(ICS_ERROR!(
      why: "Animation: Accessor type is not VEC3",
      fix: "Ensure output accessor is VEC3"
    ));
  }
  if accessor.component_type != 5126 {
    return Err(ICS_ERROR!(
      why: "Animation: Accessor component type is not FLOAT",
      fix: "Ensure output accessor uses FLOAT component type"
    ));
  }
  read_accessor_data::<[f32; 3]>(gltf, accessor)
}

fn read_accessor_vec4(gltf: &Gltf, accessor_index: usize) -> Result<Vec<[f32; 4]>, IcsError> {
  let accessor = &gltf.accessors[accessor_index];
  if accessor.bytes_type != "VEC4" {
    return Err(ICS_ERROR!(
      why: "Animation: Accessor type is not VEC4",
      fix: "Ensure output accessor is VEC4"
    ));
  }
  if accessor.component_type != 5126 {
    return Err(ICS_ERROR!(
      why: "Animation: Accessor component type is not FLOAT",
      fix: "Ensure output accessor uses FLOAT component type"
    ));
  }
  read_accessor_data::<[f32; 4]>(gltf, accessor)
}

fn read_accessor_data<T: Clone>(gltf: &Gltf, accessor: &GltfAccessor) -> Result<Vec<T>, IcsError> {
  let buffer_view_index = accessor.buffer_view.ok_or_else(|| {
    ICS_ERROR!(
      why: "Animation: Accessor has no buffer view",
      fix: "Ensure the accessor references a buffer view"
    )
  })?;
  let buffer_view = &gltf.buffer_views[buffer_view_index];
  let blob = &gltf.blobs[buffer_view.buffer];

  let offset = buffer_view.byte_offset + accessor.byte_offset;
  let length = accessor.count * accessor_element_size(accessor)?;
  if offset + length > blob.get_ref().len() {
    return Err(ICS_ERROR!(
      why: "Animation: Accessor data exceeds buffer length",
      fix: "Check buffer view and accessor offsets/lengths"
    ));
  }

  let u8_data = blob.get_ref()[offset..offset + length].to_vec();
  let data = Memory::bytes_as::<T>(&u8_data).to_vec();
  Ok(data)
}

fn accessor_element_size(accessor: &GltfAccessor) -> Result<usize, IcsError> {
  let component_size = match accessor.component_type {
    5120 | 5121 => 1,
    5122 | 5123 => 2,
    5125 | 5126 => 4,
    _ => {
      return Err(ICS_ERROR!(
        why: "Animation: Unknown accessor component type",
        fix: "Use a valid glTF component type"
      ))
    }
  };
  let num_components = match accessor.bytes_type.as_str() {
    "SCALAR" => 1,
    "VEC2" => 2,
    "VEC3" => 3,
    "VEC4" => 4,
    "MAT2" => 4,
    "MAT3" => 9,
    "MAT4" => 16,
    _ => {
      return Err(ICS_ERROR!(
        why: "Animation: Unknown accessor type",
        fix: "Use a valid glTF accessor type"
      ))
    }
  };
  Ok(component_size * num_components)
}
