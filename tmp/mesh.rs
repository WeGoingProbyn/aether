use std::{
  collections::hash_map::DefaultHasher,
  collections::HashMap,
  hash::{Hash, Hasher},
  sync::{Arc, Mutex},
};

use super::{
  entity::IcsAsset,
  hierarchy::{Hierarchy, NodeId},
  indices::{Indices, Primitive},
  pipeline::PipelineAttribute,
  textures::Textures,
  uniforms::{Pushable, UniformType},
  vertices::{Layout, VertexBuffer, Vertices},
};

use crate::{
  assets::GpuId,
  debugger::IcsError,
  maths::geometry::{Geometry, GltfGeometry},
  memory::memory::Memory,
  structures::animation::AnimationClip,
  ICS_ERROR,
};

#[derive(Debug)]
pub struct Mesh {
  indices: Indices,
  vertices: Vertices,
  textures: Option<Textures>,
  hierarchy: Mutex<Hierarchy>,
  known_materials: HashMap<NodeId, Vec<(String, Pushable, UniformType)>>,
  animations: Vec<AnimationClip>,
}

impl Mesh {
  /// Creates a new `Mesh` instance.
  ///
  /// # Arguments
  /// * `indices` - A collection of indices for the mesh.
  /// * `vertices` - A collection of vertices for the mesh.
  /// * `hierarchy` - The hierarchical structure of the mesh, wrapped in a `Mutex`.
  /// * `textures` - An optional collection of textures applied to the mesh.
  /// * `known_materials` - A `HashMap` mapping node names to material data, including variable names, pushable values, and uniform types.
  /// * `animations` - A list of animation clips associated with the mesh.
  ///
  /// # Returns
  /// A new `Mesh` instance with the provided data.
  pub fn new(
    indices: Indices,
    vertices: Vertices,
    hierarchy: Hierarchy,
    textures: Option<Textures>,
    known_materials: HashMap<NodeId, Vec<(String, Pushable, UniformType)>>,
    animations: Vec<AnimationClip>,
  ) -> Mesh {
    Mesh {
      indices,
      vertices,
      hierarchy: Mutex::new(hierarchy),
      textures,
      known_materials: known_materials.clone(),
      animations,
    }
  }

  /// Returns a reference to the `indices` of the mesh.
  pub fn indices(&self) -> &Indices {
    &self.indices
  }

  /// Returns a reference to the `vertices` of the mesh.
  pub fn vertices(&self) -> &Vertices {
    &self.vertices
  }

  /// Returns a reference to the `Mutex<Hierarchy>` of the mesh, which contains the hierarchical structure.
  pub fn hierarchy(&self) -> &Mutex<Hierarchy> {
    &self.hierarchy
  }

  /// Returns a reference to the `textures` of the mesh.
  pub fn textures(&self) -> &Option<Textures> {
    &self.textures
  }

  /// Returns a reference to the animation clips associated with this mesh.
  pub fn animations(&self) -> &Vec<AnimationClip> {
    &self.animations
  }

  /// Returns a `HashMap` of known materials for a specific node.
  ///
  /// # Arguments
  /// * `node_name` - The identifier of the node for which to retrieve the material data.
  ///
  /// # Returns
  /// A `HashMap` mapping material variable names to their corresponding `Pushable` and `UniformType` values.
  ///
  /// # Example
  /// ```rust
  /// let known_materials = mesh.known_materials("node_name");
  /// ```
  ///
  /// # Possible Errors
  /// Does not return errors, but will return an empty `HashMap` if no materials are found for the given node name.
  pub fn known_materials(&self, node_name: &NodeId) -> HashMap<String, (Pushable, UniformType)> {
    let mut known = HashMap::new();
    if let Some(materials) = self.known_materials.get(node_name) {
      for (var_name, data, ty) in materials {
        known.insert(var_name.clone(), (*data, *ty));
      }
    }
    known
  }
}

impl Hash for Mesh {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.hierarchy().lock().unwrap().hash(state)
  }
}

impl IcsAsset for Mesh {}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct MeshLayoutId(pub u64);

impl MeshLayoutId {
  pub fn from_layout(layout: &Layout) -> Self {
    let mut hasher = DefaultHasher::new();
    layout.hash(&mut hasher);
    MeshLayoutId(hasher.finish())
  }
}

#[derive(Clone, Debug)]
pub struct SubmeshRange {
  pub node: NodeId,
  pub index_start: u32,
  pub index_count: u32,
  pub primitive: Primitive,
}

#[derive(Clone, Debug)]
pub struct MeshAsset {
  pub vertex_data: Arc<[u8]>,
  pub index_data: Arc<[u8]>,
  pub vertex_layout: MeshLayoutId,
  pub submeshes: Vec<SubmeshRange>,
}

impl MeshAsset {
  pub fn from_gltf_path(path: &str) -> Result<Self, IcsError> {
    let geometry = GltfGeometry::from_gltf(path)?;
    Self::from_geometry(&geometry)
  }

  pub fn from_geometry(geometry: &impl Geometry) -> Result<Self, IcsError> {
    let vertices = geometry.vertex_attributes();
    let layout = Self::build_layout(&vertices)?;
    let mut vertex_buffer = VertexBuffer::new(layout.clone());
    vertex_buffer.build_from(&vertices)?;
    let vertex_data: Arc<[u8]> = vertex_buffer.as_slice().to_vec().into();

    let vertex_layout = MeshLayoutId::from_layout(&layout);

    let mut index_values: Vec<u32> = Vec::new();
    let mut submeshes = Vec::new();
    let mut cursor: u32 = 0;
    for (node_name, _parent, primitive, indices) in geometry.index_parts() {
      let (Some(primitive), Some(indices)) = (primitive, indices) else {
        continue;
      };
      let start = cursor;
      let count = indices.len() as u32;
      index_values.extend_from_slice(&indices);
      submeshes.push(SubmeshRange {
        node: NodeId::new(node_name),
        index_start: start,
        index_count: count,
        primitive,
      });
      cursor = cursor.saturating_add(count);
    }
    let index_bytes = Memory::slice_as_bytes(&index_values, std::mem::size_of::<u32>()).to_vec();

    Ok(MeshAsset {
      vertex_data,
      index_data: index_bytes.into(),
      vertex_layout,
      submeshes,
    })
  }

  pub fn cpu_bytes(&self) -> u64 {
    self.vertex_data.len() as u64 + self.index_data.len() as u64
  }

  fn build_layout(vertices: &Vertices) -> Result<Layout, IcsError> {
    let mut names = Vec::new();
    let mut types = Vec::new();
    for attr in Self::canonical_attribute_order() {
      if vertices.available_attributes().contains(&attr) {
        let name = Self::attribute_key(attr);
        let attr_type = vertices.get_attribute_type(name).ok_or_else(|| {
          ICS_ERROR!(
            why: "MeshAsset: Missing attribute type for vertex layout",
            fix: "Ensure vertices include the requested attribute"
          )
        })?;
        names.push(name.to_string());
        types.push(attr_type.clone());
      }
    }
    Layout::new(&names, &types)
  }

  fn canonical_attribute_order() -> [PipelineAttribute; 6] {
    [
      PipelineAttribute::Position,
      PipelineAttribute::Normal,
      PipelineAttribute::Texture,
      PipelineAttribute::Colour,
      PipelineAttribute::Tangent,
      PipelineAttribute::BitTangent,
    ]
  }

  fn attribute_key(attr: PipelineAttribute) -> &'static str {
    match attr {
      PipelineAttribute::Colour => "colour",
      PipelineAttribute::Normal => "normal",
      PipelineAttribute::Texture => "texture",
      PipelineAttribute::Tangent => "tangent",
      PipelineAttribute::Position => "position",
      PipelineAttribute::BitTangent => "bit_tangent",
    }
  }
}

impl IcsAsset for MeshAsset {}

#[derive(Debug, Clone, Copy)]
pub struct MeshVertexBuffer;

#[derive(Debug, Clone, Copy)]
pub struct MeshIndexBuffer;

#[derive(Clone, Debug)]
pub struct MeshGpu {
  pub vertex: GpuId<MeshVertexBuffer>,
  pub index: GpuId<MeshIndexBuffer>,
  pub vertex_layout: MeshLayoutId,
  pub submeshes: Vec<SubmeshRange>,
}

impl IcsAsset for MeshGpu {}
