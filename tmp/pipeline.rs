use std::{
  collections::{hash_map::DefaultHasher, HashMap},
  hash::{Hash, Hasher},
  sync::{Arc, Mutex},
};

use crate::ICS_ERROR;
use crate::{debugger::IcsError, ICS_WARN};

use super::{
  entity::{Entity, IcsAsset},
  hierarchy::{HierarchyNode, NodeId},
  indices::IndicesPart,
  shaders::Shaders,
  textures::TextureType,
  uniforms::{ConcreteUniform, ShaderStage, UniformLayout, UniformType},
  vertices::{Layout, VertexBuffer},
};

// Each VulkanBuffer holds a hashmap describing
// each object within the big buffer, an asset
// must implement IcsAsset trait and hold an Arc<T>
// to uniquely identify it
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub struct BufferMapKey {
  pub entity: usize,
  pub pipeline: usize,
  pub uniform: Option<usize>,
}

impl BufferMapKey {
  /// Creates a new `BufferMapKey` instance.
  ///
  /// # Arguments
  /// * `pipeline_key` - A unique identifier for the pipeline associated with the buffer.
  /// * `entity_key` - A unique identifier for the entity within the pipeline.
  /// * `uniform_key` - An optional unique identifier for a uniform associated with the buffer.
  ///
  /// # Returns
  /// A new `BufferMapKey` instance with the provided keys.
  pub fn new(pipeline_key: usize, entity_key: usize, uniform_key: Option<usize>) -> Self {
    BufferMapKey {
      entity: entity_key,
      pipeline: pipeline_key,
      uniform: uniform_key,
    }
  }

  // Retrieve the IcsAsset uid of the underlying entity in this map key
  pub fn to_entity_key(&self) -> usize {
    self.entity
  }

  /// Retrieves the optional uniform key from this buffer map key.
  pub fn to_uniform_key(&self) -> Option<usize> {
    self.uniform
  }
}

impl Hash for BufferMapKey {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.entity.hash(state);
    self.pipeline.hash(state);
    if let Some(uni) = self.uniform {
      uni.hash(state);
    };
  }
}

#[derive(Debug, Hash, Eq, PartialEq, Copy, Clone)]
pub struct PipelineMapKey(pub usize);

impl PipelineMapKey {
  /// Creates a new `PipelineMapKey` instance.
  ///
  /// # Arguments
  /// * `key` - A unique identifier for the pipeline.
  ///
  /// # Returns
  /// A new `PipelineMapKey` instance with the provided key.
  pub fn new(key: usize) -> Self {
    PipelineMapKey(key)
  }
}

#[derive(Eq, PartialEq, Hash, Copy, Clone, Debug, PartialOrd, Ord)]
pub enum PipelineAttribute {
  Position = 0,
  Normal,
  Texture,
  Colour,
  Tangent,
  BitTangent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageAccess {
  ReadOnly,
  WriteOnly,
  ReadWrite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageOffsetMode {
  None,
  Dynamic,
  PushConstant,
  RangeBuffer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineCullMode {
  None,
  Front,
  Back,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineTopology {
  TriangleList,
  TriangleStrip,
  LineList,
  LineStrip,
  PointList,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelinePolygonMode {
  Fill,
  Line,
  Point,
}

/// Blend factor for color blending operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BlendFactor {
  Zero,
  #[default]
  One,
  SrcColor,
  OneMinusSrcColor,
  DstColor,
  OneMinusDstColor,
  SrcAlpha,
  OneMinusSrcAlpha,
  DstAlpha,
  OneMinusDstAlpha,
  ConstantColor,
  OneMinusConstantColor,
  ConstantAlpha,
  OneMinusConstantAlpha,
  SrcAlphaSaturate,
}

/// Blend operation for combining source and destination colors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BlendOp {
  #[default]
  Add,
  Subtract,
  ReverseSubtract,
  Min,
  Max,
}

/// Depth comparison function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DepthCompareOp {
  Never,
  Less,
  Equal,
  LessOrEqual,
  Greater,
  NotEqual,
  GreaterOrEqual,
  #[default]
  Always,
}

/// Stencil operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum StencilOp {
  #[default]
  Keep,
  Zero,
  Replace,
  IncrementAndClamp,
  DecrementAndClamp,
  Invert,
  IncrementAndWrap,
  DecrementAndWrap,
}

/// Color write mask flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColorMask {
  pub r: bool,
  pub g: bool,
  pub b: bool,
  pub a: bool,
}

impl Default for ColorMask {
  fn default() -> Self {
    Self {
      r: true,
      g: true,
      b: true,
      a: true,
    }
  }
}

impl ColorMask {
  pub const ALL: Self = Self {
    r: true,
    g: true,
    b: true,
    a: true,
  };
  pub const NONE: Self = Self {
    r: false,
    g: false,
    b: false,
    a: false,
  };
  pub const RGB: Self = Self {
    r: true,
    g: true,
    b: true,
    a: false,
  };
}

/// Blend state configuration for a single color attachment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlendState {
  pub enabled: bool,
  pub src_color_factor: BlendFactor,
  pub dst_color_factor: BlendFactor,
  pub color_op: BlendOp,
  pub src_alpha_factor: BlendFactor,
  pub dst_alpha_factor: BlendFactor,
  pub alpha_op: BlendOp,
  pub color_mask: ColorMask,
}

impl Default for BlendState {
  fn default() -> Self {
    Self {
      enabled: false,
      src_color_factor: BlendFactor::One,
      dst_color_factor: BlendFactor::Zero,
      color_op: BlendOp::Add,
      src_alpha_factor: BlendFactor::One,
      dst_alpha_factor: BlendFactor::Zero,
      alpha_op: BlendOp::Add,
      color_mask: ColorMask::ALL,
    }
  }
}

impl BlendState {
  /// Standard alpha blending: srcColor * srcAlpha + dstColor * (1 - srcAlpha)
  pub fn alpha_blend() -> Self {
    Self {
      enabled: true,
      src_color_factor: BlendFactor::SrcAlpha,
      dst_color_factor: BlendFactor::OneMinusSrcAlpha,
      color_op: BlendOp::Add,
      src_alpha_factor: BlendFactor::One,
      dst_alpha_factor: BlendFactor::OneMinusSrcAlpha,
      alpha_op: BlendOp::Add,
      color_mask: ColorMask::ALL,
    }
  }

  /// Additive blending: srcColor + dstColor
  pub fn additive() -> Self {
    Self {
      enabled: true,
      src_color_factor: BlendFactor::One,
      dst_color_factor: BlendFactor::One,
      color_op: BlendOp::Add,
      src_alpha_factor: BlendFactor::One,
      dst_alpha_factor: BlendFactor::One,
      alpha_op: BlendOp::Add,
      color_mask: ColorMask::ALL,
    }
  }

  /// Premultiplied alpha blending
  pub fn premultiplied_alpha() -> Self {
    Self {
      enabled: true,
      src_color_factor: BlendFactor::One,
      dst_color_factor: BlendFactor::OneMinusSrcAlpha,
      color_op: BlendOp::Add,
      src_alpha_factor: BlendFactor::One,
      dst_alpha_factor: BlendFactor::OneMinusSrcAlpha,
      alpha_op: BlendOp::Add,
      color_mask: ColorMask::ALL,
    }
  }
}

/// Stencil test configuration for front/back faces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct StencilState {
  pub fail_op: StencilOp,
  pub pass_op: StencilOp,
  pub depth_fail_op: StencilOp,
  pub compare_op: DepthCompareOp,
  pub compare_mask: u32,
  pub write_mask: u32,
  pub reference: u32,
}

/// Complete pipeline state descriptor for configuring graphics pipeline state.
///
/// This struct captures all configurable pipeline state that can be overridden
/// from material or shader metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineStateDescriptor {
  /// Depth test enabled.
  pub depth_test: bool,
  /// Depth write enabled.
  pub depth_write: bool,
  /// Depth comparison operation.
  pub depth_compare_op: DepthCompareOp,
  /// Cull mode.
  pub cull_mode: PipelineCullMode,
  /// Primitive topology.
  pub topology: PipelineTopology,
  /// Polygon mode.
  pub polygon_mode: PipelinePolygonMode,
  /// Front face winding order (true = counter-clockwise).
  pub front_face_ccw: bool,
  /// Blend state for color attachments.
  pub blend_state: BlendState,
  /// Optional per-attachment blend states (empty = use blend_state).
  pub blend_attachments: Vec<BlendState>,
  /// Stencil test enabled.
  pub stencil_test: bool,
  /// Stencil state for front faces.
  pub stencil_front: StencilState,
  /// Stencil state for back faces.
  pub stencil_back: StencilState,
  /// Use dynamic viewport/scissor.
  pub dynamic_viewport_scissor: bool,
}

impl Default for PipelineStateDescriptor {
  fn default() -> Self {
    Self {
      depth_test: true,
      depth_write: true,
      depth_compare_op: DepthCompareOp::Less,
      cull_mode: PipelineCullMode::Back,
      topology: PipelineTopology::TriangleList,
      polygon_mode: PipelinePolygonMode::Fill,
      front_face_ccw: true,
      blend_state: BlendState::default(),
      blend_attachments: Vec::new(),
      stencil_test: false,
      stencil_front: StencilState::default(),
      stencil_back: StencilState::default(),
      dynamic_viewport_scissor: true,
    }
  }
}

impl PipelineStateDescriptor {
  /// Creates a new descriptor with default values (opaque rendering).
  pub fn opaque() -> Self {
    Self::default()
  }

  /// Creates a descriptor for transparent rendering with alpha blending.
  pub fn transparent() -> Self {
    Self {
      depth_write: false,
      blend_state: BlendState::alpha_blend(),
      ..Self::default()
    }
  }

  /// Creates a descriptor for skybox rendering (no depth write, no cull).
  pub fn skybox() -> Self {
    Self {
      depth_test: false,
      depth_write: false,
      cull_mode: PipelineCullMode::None,
      ..Self::default()
    }
  }

  /// Creates a descriptor for shadow map rendering (depth only).
  pub fn shadow_map() -> Self {
    Self {
      depth_test: true,
      depth_write: true,
      depth_compare_op: DepthCompareOp::Less,
      blend_state: BlendState {
        color_mask: ColorMask::NONE,
        ..Default::default()
      },
      ..Self::default()
    }
  }

  /// Builder method: set depth test.
  pub fn with_depth_test(mut self, enabled: bool) -> Self {
    self.depth_test = enabled;
    self
  }

  /// Builder method: set depth write.
  pub fn with_depth_write(mut self, enabled: bool) -> Self {
    self.depth_write = enabled;
    self
  }

  /// Builder method: set depth compare operation.
  pub fn with_depth_compare_op(mut self, op: DepthCompareOp) -> Self {
    self.depth_compare_op = op;
    self
  }

  /// Builder method: set cull mode.
  pub fn with_cull_mode(mut self, mode: PipelineCullMode) -> Self {
    self.cull_mode = mode;
    self
  }

  /// Builder method: set topology.
  pub fn with_topology(mut self, topology: PipelineTopology) -> Self {
    self.topology = topology;
    self
  }

  /// Builder method: set polygon mode.
  pub fn with_polygon_mode(mut self, mode: PipelinePolygonMode) -> Self {
    self.polygon_mode = mode;
    self
  }

  /// Builder method: set blend state.
  pub fn with_blend_state(mut self, state: BlendState) -> Self {
    self.blend_state = state;
    self
  }

  /// Builder method: set per-attachment blend states.
  pub fn with_blend_attachments(mut self, attachments: Vec<BlendState>) -> Self {
    self.blend_attachments = attachments;
    self
  }

  /// Builder method: set stencil test enabled.
  pub fn with_stencil_test(mut self, enabled: bool) -> Self {
    self.stencil_test = enabled;
    self
  }

  /// Builder method: set dynamic viewport/scissor.
  pub fn with_dynamic_viewport_scissor(mut self, enabled: bool) -> Self {
    self.dynamic_viewport_scissor = enabled;
    self
  }

  /// Generates a stable hash for this descriptor (for cache keys).
  pub fn state_hash(&self) -> u64 {
    let mut hasher = DefaultHasher::new();
    self.depth_test.hash(&mut hasher);
    self.depth_write.hash(&mut hasher);
    self.depth_compare_op.hash(&mut hasher);
    self.cull_mode.hash(&mut hasher);
    self.topology.hash(&mut hasher);
    self.polygon_mode.hash(&mut hasher);
    self.front_face_ccw.hash(&mut hasher);
    self.blend_state.hash(&mut hasher);
    self.blend_attachments.hash(&mut hasher);
    self.stencil_test.hash(&mut hasher);
    self.stencil_front.hash(&mut hasher);
    self.stencil_back.hash(&mut hasher);
    self.dynamic_viewport_scissor.hash(&mut hasher);
    hasher.finish()
  }
}

#[derive(Debug, Clone)]
pub struct StorageBufferLayout {
  pub name: String,
  pub binding: usize,
  pub access: StorageAccess,
  pub offset_mode: StorageOffsetMode,
}

impl PipelineAttribute {
  /// Converts the `PipelineAttribute` enum variant to its corresponding uniform type.
  pub fn to_uniform_type(&self) -> UniformType {
    match self {
      PipelineAttribute::Position => UniformType::Vec3,
      PipelineAttribute::BitTangent => UniformType::Vec4,
      PipelineAttribute::Tangent => UniformType::Vec4,
      PipelineAttribute::Normal => UniformType::Vec3,
      PipelineAttribute::Colour => UniformType::Vec3,
      PipelineAttribute::Texture => UniformType::Vec2,
    }
  }
}

#[derive(Clone)]
pub struct IcsPipeline {
  pub shaders: Shaders,
  pub vertex_layout: Layout,
  /// Legacy entity storage.
  ///
  /// **Deprecated**: Use `populate_from_draw_calls()` with ECS render extraction instead.
  /// This field is retained for backwards compatibility during migration.
  pub entities: HashMap<PipelineMapKey, Arc<Entity>>,
  pub parts: HashMap<PipelineMapKey, Vec<HierarchyNode>>,
  pub attribute_require: Vec<PipelineAttribute>,
  pub target_subpass: u32,
  pub depth_test: bool,
  pub depth_write: bool,
  pub cull_mode: PipelineCullMode,
  /// Full pipeline state descriptor (optional, for advanced state configuration).
  pub state_descriptor: Option<PipelineStateDescriptor>,

  // (resource_name, binding_index, _)
  pub textures_layout: HashMap<ShaderStage, Vec<(String, usize, TextureType)>>,
  pub uniform_layouts: HashMap<ShaderStage, Vec<(ConcreteUniform, usize, usize, UniformLayout)>>,
  pub storage_layouts: HashMap<ShaderStage, Vec<StorageBufferLayout>>,
}

impl IcsAsset for IcsPipeline {}
impl IcsAsset for Mutex<IcsPipeline> {}

impl IcsPipeline {
  /// Creates a new `IcsPipeline` instance.
  ///
  /// # Arguments
  /// * `shaders` - The collection of shaders for this pipeline.
  /// * `vertex_layout` - The layout defining the vertex structure for this pipeline.
  ///
  /// # Returns
  /// A new `IcsPipeline` instance with the provided shaders and vertex layout.
  pub fn new(shaders: Shaders, vertex_layout: Layout) -> IcsPipeline {
    IcsPipeline {
      shaders,
      vertex_layout,
      parts: HashMap::new(),
      uniform_layouts: HashMap::new(),
      textures_layout: HashMap::new(),
      storage_layouts: HashMap::new(),
      entities: HashMap::new(),
      attribute_require: Vec::new(),
      target_subpass: 0,
      depth_test: true,
      depth_write: true,
      cull_mode: PipelineCullMode::Back,
      state_descriptor: None,
    }
  }

  /// Applies state from a PipelineStateDescriptor.
  ///
  /// This sets depth_test, depth_write, and cull_mode from the descriptor,
  /// and stores the full descriptor for backends that support advanced state.
  pub fn apply_state_descriptor(&mut self, descriptor: PipelineStateDescriptor) {
    self.depth_test = descriptor.depth_test;
    self.depth_write = descriptor.depth_write;
    self.cull_mode = descriptor.cull_mode;
    self.state_descriptor = Some(descriptor);
  }

  /// Returns the state descriptor if set.
  pub fn state_descriptor(&self) -> Option<&PipelineStateDescriptor> {
    self.state_descriptor.as_ref()
  }

  /// Returns a reference to the hierarchy nodes associated with the pipeline.
  /// This does not respect the origin for nodes and can include nodes from any
  /// number of different mesh objects.
  pub fn parts(&self) -> &HashMap<PipelineMapKey, Vec<HierarchyNode>> {
    &self.parts
  }

  /// Returns a reference to the entities associated with the pipeline.
  pub fn entities(&self) -> &HashMap<PipelineMapKey, Arc<Entity>> {
    &self.entities
  }

  /// Returns a reference to the shaders associated with the pipeline.
  pub fn shaders(&self) -> &Shaders {
    &self.shaders
  }

  /// Returns the total number of nodes across all parts of the pipeline.
  pub fn total_nodes(&self) -> usize {
    self
      .parts
      .iter()
      .map(|(_, per_entity_parts)| per_entity_parts.len())
      .sum()
  }

  pub fn set_target_subpass(&mut self, subpass: u32) {
    self.target_subpass = subpass;
  }

  pub fn target_subpass(&self) -> u32 {
    self.target_subpass
  }

  pub fn set_depth_test(&mut self, enabled: bool) {
    self.depth_test = enabled;
  }

  pub fn depth_test(&self) -> bool {
    self.depth_test
  }

  pub fn set_depth_write(&mut self, enabled: bool) {
    self.depth_write = enabled;
  }

  pub fn depth_write(&self) -> bool {
    self.depth_write
  }

  pub fn set_cull_mode(&mut self, mode: PipelineCullMode) {
    self.cull_mode = mode;
  }

  pub fn cull_mode(&self) -> PipelineCullMode {
    self.cull_mode
  }

  pub fn interface_hash(&self) -> u64 {
    let mut hasher = DefaultHasher::new();
    for shader in &self.shaders {
      shader.stage().hash(&mut hasher);
      shader.interface_hash().hash(&mut hasher);
    }
    hasher.finish()
  }

  pub fn cache_key(&self, pipeline_uid: usize) -> usize {
    let mut hasher = DefaultHasher::new();
    pipeline_uid.hash(&mut hasher);
    self.interface_hash().hash(&mut hasher);
    hasher.finish() as usize
  }

  /// Generates a comprehensive cache key that includes all pipeline state.
  ///
  /// This key is stable and deterministic - it will produce the same hash
  /// for pipelines with identical configuration. The key format includes:
  /// - Shader interface hash (ordered by stage)
  /// - Vertex layout hash
  /// - Required attributes (sorted for stability)
  /// - Full pipeline state (depth, blend, stencil, cull)
  ///
  /// # Arguments
  /// * `pipeline_uid` - Unique identifier for this pipeline instance.
  ///
  /// # Returns
  /// A 64-bit hash suitable for use as a cache key.
  pub fn full_cache_key(&self, pipeline_uid: usize) -> u64 {
    let mut hasher = DefaultHasher::new();

    // 1. Pipeline UID
    pipeline_uid.hash(&mut hasher);

    // 2. Shader interface (ordered by stage for stability)
    let mut shader_hashes: Vec<_> = (&self.shaders)
      .into_iter()
      .map(|s| (s.stage() as u8, s.interface_hash()))
      .collect();
    shader_hashes.sort_by_key(|(stage, _)| *stage);
    for (stage, hash) in shader_hashes {
      stage.hash(&mut hasher);
      hash.hash(&mut hasher);
    }

    // 3. Vertex layout (size/stride)
    self.vertex_layout.size().hash(&mut hasher);

    // 4. Required attributes (sorted for stability)
    let mut attrs: Vec<_> = self.attribute_require.iter().map(|a| *a as u8).collect();
    attrs.sort();
    attrs.hash(&mut hasher);

    // 5. Pipeline state
    self.depth_test.hash(&mut hasher);
    self.depth_write.hash(&mut hasher);
    self.cull_mode.hash(&mut hasher);
    self.target_subpass.hash(&mut hasher);

    // 6. State descriptor if present
    if let Some(ref desc) = self.state_descriptor {
      desc.state_hash().hash(&mut hasher);
    }

    hasher.finish()
  }

  /// Returns a human-readable cache key description for debugging.
  ///
  /// This is useful for understanding what contributes to the cache key
  /// when debugging pipeline caching issues.
  pub fn cache_key_description(&self) -> String {
    let mut parts = Vec::new();

    parts.push(format!("interface={:016x}", self.interface_hash()));
    parts.push(format!("stride={}", self.vertex_layout.size()));

    let attrs: Vec<_> = self
      .attribute_require
      .iter()
      .map(|a| format!("{:?}", a))
      .collect();
    if !attrs.is_empty() {
      parts.push(format!("attrs=[{}]", attrs.join(",")));
    }

    parts.push(format!("depth_test={}", self.depth_test));
    parts.push(format!("depth_write={}", self.depth_write));
    parts.push(format!("cull={:?}", self.cull_mode));
    parts.push(format!("subpass={}", self.target_subpass));

    if let Some(ref desc) = self.state_descriptor {
      parts.push(format!("state_hash={:016x}", desc.state_hash()));
    }

    parts.join("|")
  }

  /// Adds required pipeline attributes to the pipeline.
  ///
  /// # Arguments
  /// * `requires` - A slice of `PipelineAttribute` values to be required for this pipeline.
  pub fn require_attributes(&mut self, requires: &[PipelineAttribute]) {
    for require in requires {
      self.attribute_require.push(*require);
    }
  }

  /// Pushes a new part for a specific entity to the pipeline.
  ///
  /// # Arguments
  /// * `entity` - The entity to which the part belongs.
  /// * `part` - The `HierarchyNode` representing the part to be added.
  pub fn push_part_for(&mut self, entity: &Arc<Entity>, part: HierarchyNode) {
    let key = PipelineMapKey::new(IcsAsset::uid(entity));
    if let Some(vec) = self.parts.get_mut(&key) {
      vec.push(part);
    } else {
      let parts = vec![part; 1];
      self.parts.insert(key, parts);
      self.entities.insert(key, entity.clone());
    };
  }

  /// Builds buffers for the pipeline to be uploaded to GPU and returns a memory map.
  /// Note: this is an expensive function
  ///
  /// # Arguments
  /// * `pipeline_arc` - A reference to the `Arc<Mutex<IcsPipeline>>` representing the pipeline.
  ///
  /// # Returns
  /// A `HashMap` mapping `BufferMapKey` to a tuple of `VertexBuffer` and a `HashMap` of `IndicesPart`s.
  pub fn build_gpu_buffers(
    &mut self,
    pipeline_arc: &Arc<Mutex<IcsPipeline>>,
  ) -> Result<HashMap<BufferMapKey, (VertexBuffer, HashMap<NodeId, IndicesPart>)>, IcsError> {
    let mut memory_map = HashMap::new();
    let pipeline_key = self.cache_key(IcsAsset::uid(pipeline_arc));
    let vertices = VertexBuffer::new(self.vertex_layout.clone());
    for (entity_pipeline_key, per_entity_parts) in self.parts.iter() {
      let entity = self.entities.get(entity_pipeline_key).ok_or_else(|| {
        ICS_ERROR!(
          why: "Pipeline: Missing entity for key while building GPU buffers",
          fix: "Ensure entities are registered before building buffers"
        )
      })?;
      let entity_key = IcsAsset::uid(entity);
      let mesh = entity.mesh();
      let rtrn = vertices
        .build_subset_from(mesh.vertices(), mesh.indices(), &per_entity_parts)
        .map_err(|e| {
          ICS_ERROR!(
            why: "Pipeline: Failed to build vertex subset for entity",
            fix: "Validate mesh data and pipeline parts",
            src: e
          )
        })?;
      memory_map.insert(BufferMapKey::new(pipeline_key, entity_key, None), rtrn);
    }
    Ok(memory_map)
  }

  /// Returns a reference to the uniform layouts for the pipeline.
  pub fn uniform_layouts(
    &self,
  ) -> &HashMap<ShaderStage, Vec<(ConcreteUniform, usize, usize, UniformLayout)>> {
    &self.uniform_layouts
  }

  /// Returns a reference to the textures layouts for the pipeline.
  pub fn textures_layout(&self) -> &HashMap<ShaderStage, Vec<(String, usize, TextureType)>> {
    &self.textures_layout
  }

  pub fn storage_layouts(&self) -> &HashMap<ShaderStage, Vec<StorageBufferLayout>> {
    &self.storage_layouts
  }

  /// Returns the total number of uniforms in the pipeline.
  pub fn number_uniforms(&self) -> usize {
    self.uniform_layouts.iter().map(|(_, vec)| vec.len()).sum()
  }

  /// Returns the total number of textures in the pipeline.
  pub fn number_textures(&self) -> usize {
    self.textures_layout.iter().map(|(_, vec)| vec.len()).sum()
  }

  pub fn number_storage_buffers(&self) -> usize {
    self.storage_layouts.iter().map(|(_, vec)| vec.len()).sum()
  }

  /// Returns a reference to the vertex layout for the pipeline.
  pub fn vertex_layout(&self) -> &Layout {
    &self.vertex_layout
  }

  /// Populates this pipeline's parts from a RenderView.
  ///
  /// This is the ECS-based approach to pipeline population. Instead of
  /// iterating over entities, we receive pre-extracted draw calls from
  /// the ECS render extraction.
  ///
  /// Note: This only populates parts for draw calls matching this pipeline's
  /// material type. Call this for each pipeline with the filtered draw calls.
  pub fn populate_from_draw_calls(&mut self, draw_calls: &[crate::ecs::render::DrawCall]) {
    self.parts.clear();

    for call in draw_calls {
      let key = PipelineMapKey::new(call.entity_id.0 as usize);
      self.parts.entry(key).or_default().push(call.node.clone());
    }
  }

  /// Clears all parts from this pipeline.
  pub fn clear_parts(&mut self) {
    self.parts.clear();
  }
}

#[derive(Clone)]
pub struct IcsComputePipeline {
  pub shaders: Shaders,
  pub uniform_layouts: HashMap<ShaderStage, Vec<(ConcreteUniform, usize, usize, UniformLayout)>>,
  pub textures_layout: HashMap<ShaderStage, Vec<(String, usize, TextureType)>>,
  pub storage_layouts: HashMap<ShaderStage, Vec<StorageBufferLayout>>,
}

impl IcsAsset for IcsComputePipeline {}
impl IcsAsset for Mutex<IcsComputePipeline> {}

impl IcsComputePipeline {
  /// Creates a new compute pipeline from a shader collection.
  ///
  /// Expects exactly one compute shader in the collection.
  pub fn new(shaders: Shaders) -> Self {
    let mut compute_count = 0usize;
    let mut shader_count = 0usize;
    for shader in shaders.data().iter() {
      shader_count += 1;
      if shader.stage() == ShaderStage::Compute {
        compute_count += 1;
      }
    }
    if shader_count != 1 || compute_count != 1 {
      ICS_WARN!(
        "expected exactly one compute shader (got {} shaders, {} compute)",
        shader_count,
        compute_count
      );
    }
    Self {
      shaders,
      uniform_layouts: HashMap::new(),
      textures_layout: HashMap::new(),
      storage_layouts: HashMap::new(),
    }
  }

  pub fn shaders(&self) -> &Shaders {
    &self.shaders
  }

  pub fn uniform_layouts(
    &self,
  ) -> &HashMap<ShaderStage, Vec<(ConcreteUniform, usize, usize, UniformLayout)>> {
    &self.uniform_layouts
  }

  pub fn textures_layout(&self) -> &HashMap<ShaderStage, Vec<(String, usize, TextureType)>> {
    &self.textures_layout
  }

  pub fn storage_layouts(&self) -> &HashMap<ShaderStage, Vec<StorageBufferLayout>> {
    &self.storage_layouts
  }

  pub fn interface_hash(&self) -> u64 {
    let mut hasher = DefaultHasher::new();
    let mut shader_hashes: Vec<_> = self
      .shaders
      .data()
      .iter()
      .map(|s| (s.stage() as u8, s.interface_hash()))
      .collect();
    shader_hashes.sort_by_key(|(stage, _)| *stage);
    for (stage, hash) in shader_hashes {
      stage.hash(&mut hasher);
      hash.hash(&mut hasher);
    }
    hasher.finish()
  }

  pub fn cache_key(&self, pipeline_uid: usize) -> u64 {
    let mut hasher = DefaultHasher::new();
    pipeline_uid.hash(&mut hasher);
    self.interface_hash().hash(&mut hasher);
    hasher.finish()
  }
}
