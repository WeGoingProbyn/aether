use std::fmt;
use std::sync::Arc;

use super::entity::IcsAsset;
use super::uniforms::ShaderStage;

/// Enum representing different shader platforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
  OpenGL, // liar
  Vulkan,
  DirectX11, // liar
  DirectX12, // liar
  UnknownPlatform,
}

/// Enum representing different types of shaders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderType {
  Pixel,
  Vertex,
  Compute,
  Geometry,
  TessellationHull,
  TessellationDomain,
  UnknownShader,
}

/// Represents an element within the shader layout.
#[derive(Debug, Clone)]
pub struct ShaderLayoutElement {
  id: usize,
  shader_type: ShaderType,
}

impl ShaderLayoutElement {
  /// Creates a new `ShaderLayoutElement` with the specified id and shader type.
  ///
  /// # Arguments
  /// * `id` - The unique identifier for the element.
  /// * `shader_type` - The type of shader associated with this element.
  ///
  /// # Returns
  /// A new `ShaderLayoutElement` instance.
  pub fn new(id: usize, shader_type: ShaderType) -> Self {
    ShaderLayoutElement { id, shader_type }
  }

  /// Returns the id of the shader layout element.
  pub fn id(&self) -> usize {
    self.id
  }

  /// Returns the shader type of the layout element.
  pub fn get_type(&self) -> ShaderType {
    self.shader_type
  }
}

impl fmt::Display for ShaderLayoutElement {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(
      f,
      "Element {{ id: {}, type: {} }}",
      self.id, self.shader_type
    )
  }
}

/// Represents the layout of shaders, including the platform and shader types.
#[derive(Debug, Clone)]
pub struct ShaderLayout {
  platform: Platform,
  structure: Vec<ShaderLayoutElement>,
}

impl ShaderLayout {
  /// Creates a new `ShaderLayout` with the specified platform.
  ///
  /// # Arguments
  /// * `platform` - The target platform for this shader layout.
  ///
  /// # Returns
  /// A new `ShaderLayout` instance with an empty structure.
  pub fn new(platform: Platform) -> Self {
    ShaderLayout {
      platform,
      structure: Vec::new(),
    }
  }

  /// Sets the platform for the shader layout.
  ///
  /// # Arguments
  /// * `platform` - The new platform to set.
  pub fn set_platform(&mut self, platform: Platform) {
    self.platform = platform;
  }

  /// Adds a shader element to the layout structure.
  ///
  /// # Arguments
  /// * `shader_type` - The type of shader to add to the layout.
  pub fn push(&mut self, shader_type: ShaderType) {
    let id = self.structure.len();
    self
      .structure
      .push(ShaderLayoutElement::new(id, shader_type));
  }

  /// Returns a reference to the structure of the shader layout.
  ///
  /// # Returns
  /// A reference to the `Vec<ShaderLayoutElement>`, representing the layout's structure.
  pub fn structure(&self) -> &Vec<ShaderLayoutElement> {
    &self.structure
  }

  /// Returns the platform of the shader layout.
  ///
  /// # Returns
  /// The `Platform` of the shader layout.
  pub fn platform(&self) -> Platform {
    self.platform
  }
}

impl fmt::Display for ShaderLayout {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    writeln!(f, "ShaderLayout (Platform: {})", self.platform)?;
    for element in &self.structure {
      writeln!(f, "  {}", element)?;
    }
    Ok(())
  }
}

/// Represents a shader, containing its code.
#[derive(Debug, Clone)]
pub struct Shader {
  name: String,
  code: String,
  entry: String,
  bytes: Option<Vec<u8>>,
  stage: ShaderStage,
  interface_hash: u64,
}

impl IcsAsset for Shader {}

impl Shader {
  /// Creates a new `Shader` instance with the specified details.
  ///
  /// # Arguments
  /// * `name` - The name of the shader.
  /// * `code` - The source code of the shader.
  /// * `entry` - The entry function for the shader.
  /// * `stage` - The stage of the shader (e.g., vertex, fragment, etc.).
  /// * `spirv` - Optional SPIR-V bytecode for the shader.
  ///
  /// # Returns
  /// A new `Shader` instance.
  pub fn new(
    name: &str,
    code: String,
    entry: &str,
    stage: ShaderStage,
    spirv: Option<Vec<u8>>,
  ) -> Shader {
    Shader {
      name: name.to_string(),
      code,
      stage,
      entry: entry.to_string(),
      bytes: spirv,
      interface_hash: 0,
    }
  }

  /// Returns the name of the shader.
  pub fn name(&self) -> String {
    self.name.clone()
  }

  /// Returns the stage of the shader.
  pub fn stage(&self) -> ShaderStage {
    self.stage
  }

  /// Returns the readable code of the shader.
  pub fn code(&self) -> &str {
    &self.code
  }

  /// Returns the entry function name for this shader.
  pub fn entry(&self) -> &str {
    &self.entry
  }

  /// Returns the optional bytecode (SPIR-V) associated with the shader.
  pub fn bytes(&self) -> &Option<Vec<u8>> {
    &self.bytes
  }

  pub fn interface_hash(&self) -> u64 {
    self.interface_hash
  }

  pub fn set_interface_hash(&mut self, hash: u64) {
    self.interface_hash = hash;
  }
}

impl fmt::Display for Shader {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    writeln!(f, "Shader Code:\n{}", self.code)
  }
}

/// Manages a collection of shaders and their associated layout.
#[derive(Debug, Clone)]
pub struct ShaderBuffer {
  layout: ShaderLayout,
  shaders: Vec<Arc<Shader>>,
}

impl ShaderBuffer {
  /// Creates a new `ShaderBuffer` with the given layout.
  ///
  /// # Arguments
  /// * `layout` - The `ShaderLayout` for the buffer.
  ///
  /// # Returns
  /// A new `ShaderBuffer` instance.
  pub fn new(layout: ShaderLayout) -> Self {
    ShaderBuffer {
      layout,
      shaders: Vec::new(),
    }
  }

  /// Returns a reference to the layout of the shader buffer.
  pub fn layout(&self) -> &ShaderLayout {
    &self.layout
  }

  /// Adds a shader to the buffer.
  ///
  /// # Arguments
  /// * `shader` - The `Arc<Shader>` to be added to the buffer.
  pub fn push_shader(&mut self, shader: Arc<Shader>) {
    self.shaders.push(shader);
  }

  /// Returns the shader at the given index in the buffer, if it exists.
  ///
  /// # Arguments
  /// * `index` - The index of the shader to retrieve.
  ///
  /// # Returns
  /// An `Option<&Arc<Shader>>` containing the shader at the given index, or `None` if not found.
  pub fn shader(&self, index: usize) -> Option<&Arc<Shader>> {
    self.shaders.get(index)
  }

  /// Returns the number of shaders in the buffer.
  pub fn len(&self) -> usize {
    self.shaders.len()
  }

  /// Returns an iterator over the shaders in the buffer.
  pub fn iter(&self) -> impl Iterator<Item = &Arc<Shader>> {
    self.shaders.iter()
  }
}

impl fmt::Display for ShaderBuffer {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    writeln!(f, "ShaderBuffer:")?;
    writeln!(f, "{}", self.layout)?;
    writeln!(f, "Shaders ({}):", self.shaders.len())?;
    for (i, shader) in self.shaders.iter().enumerate() {
      writeln!(f, "Shader {}:\n{}", i, shader)?;
    }
    Ok(())
  }
}

/// Represents a collection of shaders along with their buffer and layout.
#[derive(Debug, Clone)]
pub struct Shaders {
  buffer: ShaderBuffer,
}

impl Shaders {
  /// Creates a new `Shaders` instance with the given layout.
  ///
  /// # Arguments
  /// * `layout` - The `ShaderLayout` for the shaders.
  ///
  /// # Returns
  /// A new `Shaders` instance.
  pub fn new(layout: ShaderLayout) -> Self {
    Shaders {
      buffer: ShaderBuffer::new(layout),
    }
  }

  /// Returns a reference to the shader buffer.
  pub fn data(&self) -> &ShaderBuffer {
    &self.buffer
  }

  /// Returns a reference to the layout of the shaders.
  pub fn layout(&self) -> &ShaderLayout {
    self.buffer.layout()
  }

  /// Adds a shader to the collection.
  pub fn push(&mut self, shader: Arc<Shader>) {
    self.buffer.push_shader(shader);
  }
}

impl fmt::Display for Shaders {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    writeln!(f, "Shaders:")?;
    writeln!(f, "{}", self.buffer)
  }
}

/// Implements display formatting for `Platform`.
impl fmt::Display for Platform {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let platform_name = match self {
      Platform::OpenGL => "OpenGL",
      Platform::Vulkan => "Vulkan",
      Platform::DirectX11 => "DirectX11",
      Platform::DirectX12 => "DirectX12",
      Platform::UnknownPlatform => "UnknownPlatform",
    };
    write!(f, "{}", platform_name)
  }
}

/// Implements display formatting for `ShaderType`.
impl fmt::Display for ShaderType {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let shader_type_name = match self {
      ShaderType::Pixel => "Pixel",
      ShaderType::Vertex => "Vertex",
      ShaderType::Compute => "Compute",
      ShaderType::Geometry => "Geometry",
      ShaderType::TessellationHull => "TessellationHull",
      ShaderType::TessellationDomain => "TessellationDomain",
      ShaderType::UnknownShader => "UnknownShader",
    };
    write!(f, "{}", shader_type_name)
  }
}

/// Iterator over Shaders, yielding references to Shader.
pub struct ShadersIterator<'a> {
  inner: std::slice::Iter<'a, Arc<Shader>>,
}

impl<'a> ShadersIterator<'a> {
  fn new(shaders: &'a ShaderBuffer) -> Self {
    ShadersIterator {
      inner: shaders.shaders.iter(),
    }
  }
}

impl<'a> Iterator for ShadersIterator<'a> {
  type Item = &'a Shader;

  fn next(&mut self) -> Option<Self::Item> {
    self.inner.next().map(|arc_shader| arc_shader.as_ref())
  }
}

impl<'a> IntoIterator for &'a Shaders {
  type Item = &'a Shader;
  type IntoIter = ShadersIterator<'a>;

  fn into_iter(self) -> Self::IntoIter {
    ShadersIterator::new(&self.buffer)
  }
}

/// Mutable iterator over Shaders, yielding mutable references to Shader.
pub struct ShadersIteratorMut<'a> {
  inner: std::slice::IterMut<'a, Arc<Shader>>,
}

impl<'a> ShadersIteratorMut<'a> {
  fn new(shaders: &'a mut ShaderBuffer) -> Self {
    ShadersIteratorMut {
      inner: shaders.shaders.iter_mut(),
    }
  }
}

impl<'a> Iterator for ShadersIteratorMut<'a> {
  type Item = &'a mut Shader;

  fn next(&mut self) -> Option<Self::Item> {
    self
      .inner
      .next()
      .map(|arc_shader| Arc::get_mut(arc_shader).expect("Failed to get mutable reference from Arc"))
  }
}

impl<'a> IntoIterator for &'a mut Shaders {
  type Item = &'a mut Shader;
  type IntoIter = ShadersIteratorMut<'a>;

  fn into_iter(self) -> Self::IntoIter {
    ShadersIteratorMut::new(&mut self.buffer)
  }
}
