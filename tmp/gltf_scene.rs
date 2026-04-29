use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::debugger::IcsError;
use crate::maths::camera::{CameraType, FirstPersonCamera};
use crate::maths::matrix::Matrix;
use crate::maths::quaternion::Quaternion;
use crate::maths::transformation::Transformation;
use crate::maths::vector::Vector;
use crate::structures::lights::{Light, LightOwner, LightType};
use crate::structures::materials::MaterialPipelineType;
use crate::structures::textures::Texture;
use crate::utility::file_io::file_to_string;
use crate::utility::json::{Gltf, GltfNode, JsonParser};
use crate::{ICS_ERROR, ICS_WARN};

use crate::maths::geometry::GltfGeometry;

#[derive(Debug, Clone)]
pub struct GltfSceneLight {
  pub light_type: LightType,
  pub position: [f32; 3],
  pub color: [f32; 3],
  pub intensity: f32,
}

impl GltfSceneLight {
  pub fn to_light(&self, pipeline_type: MaterialPipelineType) -> Light {
    let mut light = Light::new(
      self.light_type,
      LightOwner::Scene,
      pipeline_type,
      self.position,
      1,
    );
    light.set_color_intensity(self.color, self.intensity);
    light
  }
}

/// Camera extracted from a glTF scene with world-space transform applied.
#[derive(Debug, Clone)]
pub struct GltfSceneCamera {
  /// Name from the glTF camera (if any)
  pub name: Option<String>,
  /// World-space position
  pub position: [f32; 3],
  /// Forward direction (derived from node rotation)
  pub direction: [f32; 3],
  /// Up vector (derived from node rotation)
  pub up: [f32; 3],
  /// Vertical field of view in degrees (for perspective cameras)
  pub fov_degrees: Option<f32>,
  /// Near clipping plane
  pub znear: f32,
  /// Far clipping plane (None for infinite)
  pub zfar: Option<f32>,
  /// Whether this is an orthographic camera
  pub is_orthographic: bool,
}

/// Settings for configuring cameras loaded from glTF.
#[derive(Debug, Clone)]
pub struct CameraSettings {
  /// The controller type to use for all cameras
  pub controller_type: CameraType,
  /// Movement speed in units per second
  pub move_speed: f32,
  /// Mouse sensitivity multiplier
  pub mouse_sensitivity: f32,
}

impl Default for CameraSettings {
  fn default() -> Self {
    Self {
      controller_type: CameraType::FirstPerson,
      move_speed: 5.0,
      mouse_sensitivity: 0.002,
    }
  }
}

impl CameraSettings {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn with_move_speed(mut self, speed: f32) -> Self {
    self.move_speed = speed;
    self
  }

  pub fn with_mouse_sensitivity(mut self, sensitivity: f32) -> Self {
    self.mouse_sensitivity = sensitivity;
    self
  }

  pub fn with_controller_type(mut self, controller_type: CameraType) -> Self {
    self.controller_type = controller_type;
    self
  }
}

impl GltfSceneCamera {
  /// Creates a FirstPersonCamera from this glTF camera data.
  pub fn to_first_person_camera(&self, settings: &CameraSettings) -> FirstPersonCamera {
    // Calculate look-at point from position and direction
    let look_at = [
      self.position[0] + self.direction[0],
      self.position[1] + self.direction[1],
      self.position[2] + self.direction[2],
    ];

    let mut camera = FirstPersonCamera::from_position_target(self.position, look_at);
    camera.move_speed = settings.move_speed;
    camera.mouse_sensitivity = settings.mouse_sensitivity;
    camera
  }
}

#[derive(Debug)]
pub struct GltfScene {
  pub gltf: Gltf,
  pub geometry: GltfGeometry,
  pub lights: Vec<GltfSceneLight>,
  pub cameras: Vec<GltfSceneCamera>,
  /// Pre-loaded textures keyed by their file path.
  /// Populated during async loading to avoid main-thread I/O.
  pub preloaded_textures: HashMap<String, Arc<Texture>>,
}

impl GltfScene {
  /// Creates a GltfScene from a file path (synchronous I/O).
  /// Prefer `from_parts` with async loading for better performance.
  pub fn from_gltf(path: &str) -> Result<Self, IcsError> {
    let base = Path::new(path)
      .parent()
      .and_then(|p| p.to_str())
      .ok_or_else(|| {
        ICS_ERROR!(
          why: "GltfScene: Could not determine base path for glTF",
          fix: "Use a path that includes a parent directory"
        )
      })?;
    let json_str = file_to_string(path)?;
    let mut parser = JsonParser::new(&json_str);
    let json_value = parser.parse()?;
    let gltf = Gltf::from_json(json_value, format!("{}/", base))?;
    let geometry = GltfGeometry::from_gltf_data(&gltf)?;
    let lights = collect_scene_lights(&gltf);
    let cameras = collect_scene_cameras(&gltf);
    Ok(GltfScene {
      gltf,
      geometry,
      lights,
      cameras,
      preloaded_textures: HashMap::new(),
    })
  }

  /// Creates a GltfScene from pre-loaded parts (for async loading).
  pub fn from_parts(gltf: Gltf, geometry: GltfGeometry) -> Result<Self, IcsError> {
    let lights = collect_scene_lights(&gltf);
    let cameras = collect_scene_cameras(&gltf);
    Ok(GltfScene {
      gltf,
      geometry,
      lights,
      cameras,
      preloaded_textures: HashMap::new(),
    })
  }

  /// Creates a GltfScene from pre-loaded parts with textures (for async loading).
  pub fn from_parts_with_textures(
    gltf: Gltf,
    geometry: GltfGeometry,
    preloaded_textures: HashMap<String, Arc<Texture>>,
  ) -> Result<Self, IcsError> {
    let lights = collect_scene_lights(&gltf);
    let cameras = collect_scene_cameras(&gltf);
    Ok(GltfScene {
      gltf,
      geometry,
      lights,
      cameras,
      preloaded_textures,
    })
  }

  /// Returns a pre-loaded texture by path, if available.
  pub fn get_preloaded_texture(&self, path: &str) -> Option<Arc<Texture>> {
    self.preloaded_textures.get(path).cloned()
  }

  /// Returns true if textures were pre-loaded.
  pub fn has_preloaded_textures(&self) -> bool {
    !self.preloaded_textures.is_empty()
  }

  pub fn lights_or_default(&self) -> Vec<GltfSceneLight> {
    if self.lights.is_empty() {
      vec![default_sun_light()]
    } else {
      self.lights.clone()
    }
  }

  /// Returns the cameras from the glTF, or an empty vector if none exist.
  pub fn cameras(&self) -> &[GltfSceneCamera] {
    &self.cameras
  }

  /// Returns true if the glTF contains at least one camera.
  pub fn has_cameras(&self) -> bool {
    !self.cameras.is_empty()
  }
}

fn default_sun_light() -> GltfSceneLight {
  GltfSceneLight {
    light_type: LightType::Directional,
    position: [0.0, 100.0, 0.0],
    color: [1.0, 0.98, 0.92],
    intensity: 5.0,
  }
}

fn collect_scene_lights(gltf: &Gltf) -> Vec<GltfSceneLight> {
  let mut lights = Vec::new();
  if gltf.lights.is_empty() {
    return lights;
  }

  for (node_idx, node) in gltf.nodes.iter().enumerate() {
    let Some(light_idx) = node.light else {
      continue;
    };
    let Some(light) = gltf.lights.get(light_idx) else {
      ICS_WARN!(
        "GltfScene: Node references missing light index {}",
        light_idx
      );
      continue;
    };
    let world = node_world_transform(gltf, node_idx);
    let position = Transformation::translation_from_model_matrix(&world);
    let light_type = map_light_type(&light.light_type);
    lights.push(GltfSceneLight {
      light_type,
      position,
      color: light.color,
      intensity: light.intensity,
    });
  }

  lights
}

fn map_light_type(light_type: &str) -> LightType {
  match light_type {
    "directional" => LightType::Directional,
    "spot" => LightType::Spotlight,
    _ => LightType::Point,
  }
}

fn collect_scene_cameras(gltf: &Gltf) -> Vec<GltfSceneCamera> {
  let mut cameras = Vec::new();
  if gltf.cameras.is_empty() {
    return cameras;
  }

  for (node_idx, node) in gltf.nodes.iter().enumerate() {
    let Some(camera_idx) = node.camera else {
      continue;
    };
    let Some(camera) = gltf.cameras.get(camera_idx) else {
      ICS_WARN!(
        "GltfScene: Node references missing camera index {}",
        camera_idx
      );
      continue;
    };

    let world = node_world_transform(gltf, node_idx);
    let position = Transformation::translation_from_model_matrix(&world);

    // Extract rotation from world transform to get direction and up vectors
    // Default camera looks down -Z with Y up in glTF
    // Apply the rotation part of the world matrix to transform the default vectors

    // Transform default forward vector (0, 0, -1) by the rotation part
    let direction = Vector::<3, f32>::from([
      -world[0][2], // -Z component transformed by first row
      -world[1][2], // -Z component transformed by second row
      -world[2][2], // -Z component transformed by third row
    ])
    .normalise();

    // Transform default up vector (0, 1, 0) by the rotation part
    let up = Vector::<3, f32>::from([
      world[0][1], // Y component transformed by first row
      world[1][1], // Y component transformed by second row
      world[2][1], // Y component transformed by third row
    ])
    .normalise();

    // Extract camera properties
    let (fov_degrees, znear, zfar, is_orthographic) = if let Some(persp) = &camera.perspective {
      let fov_deg = persp.yfov.to_degrees();
      (Some(fov_deg), persp.znear, persp.zfar, false)
    } else if let Some(ortho) = &camera.orthographic {
      (None, ortho.znear, Some(ortho.zfar), true)
    } else {
      // Default perspective values
      (Some(45.0), 0.1, Some(1000.0), false)
    };

    cameras.push(GltfSceneCamera {
      name: camera.name.clone(),
      position,
      direction: [direction[0], direction[1], direction[2]],
      up: [up[0], up[1], up[2]],
      fov_degrees,
      znear,
      zfar,
      is_orthographic,
    });
  }

  cameras
}

fn node_world_transform(gltf: &Gltf, node_idx: usize) -> Matrix<4, 4, f32> {
  let mut chain = Vec::new();
  let mut current = Some(node_idx);
  while let Some(idx) = current {
    chain.push(idx);
    current = gltf.node_parents.get(idx).and_then(|parent| *parent);
  }
  chain.reverse();

  let mut world = Matrix::<4, 4, f32>::identity();
  for idx in chain {
    let local = node_local_transform(&gltf.nodes[idx]);
    world = world * local;
  }
  world
}

fn node_local_transform(node: &GltfNode) -> Matrix<4, 4, f32> {
  if !node.matrix.is_empty() {
    if node.matrix.len() == 16 {
      let model_matrix = Matrix::<4, 4, f32>::from_array([
        Vector::from([
          node.matrix[0],
          node.matrix[1],
          node.matrix[2],
          node.matrix[3],
        ]),
        Vector::from([
          node.matrix[4],
          node.matrix[5],
          node.matrix[6],
          node.matrix[7],
        ]),
        Vector::from([
          node.matrix[8],
          node.matrix[9],
          node.matrix[10],
          node.matrix[11],
        ]),
        Vector::from([
          node.matrix[12],
          node.matrix[13],
          node.matrix[14],
          node.matrix[15],
        ]),
      ]);
      return model_matrix.transpose();
    }
  }

  let translation = if node.translation.len() == 3 {
    [
      node.translation[0],
      node.translation[1],
      node.translation[2],
    ]
  } else {
    [0.0, 0.0, 0.0]
  };

  let scale = if node.scale.len() == 3 {
    [node.scale[0], node.scale[1], node.scale[2]]
  } else {
    [1.0, 1.0, 1.0]
  };

  let rotation = if node.rotation.len() == 4 {
    let quat = Quaternion::from([
      node.rotation[3],
      node.rotation[0],
      node.rotation[1],
      node.rotation[2],
    ]);
    quat.to_euler_angles()
  } else {
    [0.0, 0.0, 0.0]
  };

  let transform = Transformation::from(scale, rotation, translation);
  transform.transform()
}
