use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::debugger::IcsError;
use crate::maths::matrix::Matrix;
use crate::memory::memory::Memory;
use crate::structures::entity::IcsAsset;
use crate::structures::hierarchy::NodeId;
use crate::ICS_ERROR;

/// Defines the various types of uniforms that can be used in a shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UniformType {
  Mat4,
  Mat3,
  Vec3,
  Vec4,
  Vec2,
  UVec2,
  UVec3,
  UVec4,
  Float,
  Int,
  Uint,
  Bool,
  Sampler2D,
  Unknown,
}

impl UniformType {
  /// Returns the number of components for this uniform type.
  ///
  /// # Returns
  /// The number of components for the uniform type. For matrices, this is the number of elements in the matrix,
  /// and for other types, it's the number of components (e.g., 1 for scalars).
  ///
  /// # Examples
  /// ```
  /// assert_eq!(UniformType::Mat4.components(), 16);
  /// assert_eq!(UniformType::Vec3.components(), 3);
  /// assert_eq!(UniformType::Float.components(), 1);
  /// ```
  pub fn components(&self) -> usize {
    match self {
      UniformType::Mat4 => 16,
      UniformType::Mat3 => 9,
      UniformType::Vec2 => 2,
      UniformType::Vec3 => 3,
      UniformType::Vec4 => 4,
      UniformType::UVec2 => 2,
      UniformType::UVec3 => 3,
      UniformType::UVec4 => 4,
      UniformType::Float => 1,
      UniformType::Int => 1,
      UniformType::Uint => 1,
      UniformType::Bool => 1,
      _ => 0,
    }
  }

  /// Converts the number of components into a corresponding `UniformType`.
  ///
  /// # Arguments
  /// * `components` - The number of components to map to a `UniformType`.
  ///
  /// # Returns
  /// A `UniformType` corresponding to the number of components.
  ///
  /// # Examples
  /// ```
  /// assert_eq!(UniformType::from_components(16), UniformType::Mat4);
  /// assert_eq!(UniformType::from_components(3), UniformType::Vec3);
  /// ```
  pub fn from_components(components: usize) -> UniformType {
    match components {
      1 => UniformType::Float,
      2 => UniformType::Vec2,
      3 => UniformType::Vec3,
      4 => UniformType::Vec4,
      9 => UniformType::Mat3,
      16 => UniformType::Mat4,
      _ => UniformType::Float, // Default case
    }
  }

  /// Checks if the uniform type is a matrix.
  ///
  /// # Returns
  /// `true` if the uniform type is a matrix (Mat3 or Mat4), otherwise `false`.
  ///
  /// # Examples
  /// ```
  /// assert!(UniformType::Mat4.is_matrix());
  /// assert!(!UniformType::Vec3.is_matrix());
  /// ```
  pub fn is_matrix(&self) -> bool {
    match self {
      UniformType::Mat4 | UniformType::Mat3 => true,
      _ => false,
    }
  }

  /// Checks if the uniform type is a vector.
  ///
  /// # Returns
  /// `true` if the uniform type is a vector (Vec2, Vec3, or Vec4), otherwise `false`.
  ///
  /// # Examples
  /// ```
  /// assert!(UniformType::Vec4.is_vector());
  /// assert!(!UniformType::Mat3.is_vector());
  /// ```
  pub fn is_vector(&self) -> bool {
    match self {
      UniformType::Vec4
      | UniformType::Vec3
      | UniformType::Vec2
      | UniformType::UVec2
      | UniformType::UVec3
      | UniformType::UVec4 => true,
      _ => false,
    }
  }

  /// Checks if the uniform type is an unsigned integer vector.
  pub fn is_unsigned_vector(&self) -> bool {
    matches!(
      self,
      UniformType::UVec2 | UniformType::UVec3 | UniformType::UVec4
    )
  }

  /// Converts the number of components into a corresponding unsigned vector type.
  pub fn unsigned_vector_from_components(components: usize) -> Option<UniformType> {
    match components {
      2 => Some(UniformType::UVec2),
      3 => Some(UniformType::UVec3),
      4 => Some(UniformType::UVec4),
      _ => None,
    }
  }

  /// Checks if the uniform type is a scalar (int or float).
  ///
  /// # Returns
  /// `true` if the uniform type is scalar (either `Int` or `Float`), otherwise `false`.
  ///
  /// # Examples
  /// ```
  /// assert!(UniformType::Int.is_scalar());
  /// assert!(!UniformType::Vec3.is_scalar());
  /// ```
  pub fn is_scalar(&self) -> bool {
    match self {
      UniformType::Int | UniformType::Uint | UniformType::Float | UniformType::Bool => true,
      _ => false,
    }
  }

  /// Checks if the uniform type is a dimensional type (matrix or vector).
  ///
  /// # Returns
  /// `true` if the uniform type is a matrix or a vector (Mat3, Mat4, Vec2, Vec3, or Vec4), otherwise `false`.
  ///
  /// # Examples
  /// ```
  /// assert!(UniformType::Mat3.is_dimensional());
  /// assert!(!UniformType::Bool.is_dimensional());
  /// ```
  pub fn is_dimensional(&self) -> bool {
    match self {
      UniformType::Mat3
      | UniformType::Mat4
      | UniformType::Vec4
      | UniformType::Vec3
      | UniformType::Vec2
      | UniformType::UVec2
      | UniformType::UVec3
      | UniformType::UVec4 => true,
      _ => false,
    }
  }

  /// Converts a string representation to a `UniformType`.
  ///
  /// # Arguments
  /// * `str` - The string representation of the uniform type.
  ///
  /// # Returns
  /// A `UniformType` corresponding to the string.
  ///
  /// # Panics
  /// Panics if the string does not match any known uniform type.
  ///
  /// # Examples
  /// ```
  /// assert_eq!(UniformType::from_string("float"), UniformType::Float);
  /// assert_eq!(UniformType::from_string("mat4"), UniformType::Mat4);
  /// ```
  pub fn from_string(str: &str) -> UniformType {
    match str {
      "bool" => UniformType::Bool,
      "int" => UniformType::Int,
      "uint" => UniformType::Uint,
      "float" => UniformType::Float,
      "vec2" => UniformType::Vec2,
      "vec3" => UniformType::Vec3,
      "vec4" => UniformType::Vec4,
      "uvec2" => UniformType::UVec2,
      "uvec3" => UniformType::UVec3,
      "uvec4" => UniformType::UVec4,
      "mat3" => UniformType::Mat3,
      "mat4" => UniformType::Mat4,
      _ => panic!("non-existant uniform type"),
    }
  }
}

/// Different types of values that can be pushed to a uniform buffer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Pushable {
  Mat4([[f32; 4]; 4]),
  Mat3([[f32; 3]; 3]),
  Vec2([f32; 2]),
  Vec3([f32; 3]),
  Vec4([f32; 4]),
  Float(f32),
  Int(i32),
  UInt(u32),
  Bool(bool),
  Unknown,
}

impl Pushable {
  /// Returns the corresponding `UniformType` for this value.
  ///
  /// # Returns
  /// A `UniformType` that matches the type of this value.
  ///
  /// # Examples
  /// ```
  /// assert_eq!(Pushable::Mat4([[1.0; 4]; 4]).to_type(), UniformType::Mat4);
  /// assert_eq!(Pushable::Float(1.0).to_type(), UniformType::Float);
  /// ```
  pub fn to_type(&self) -> UniformType {
    match self {
      Pushable::Mat4(_) => UniformType::Mat4,
      Pushable::Mat3(_) => UniformType::Mat3,
      Pushable::Vec4(_) => UniformType::Vec4,
      Pushable::Vec2(_) => UniformType::Vec2,
      Pushable::Vec3(_) => UniformType::Vec3,
      Pushable::Float(_) => UniformType::Float,
      Pushable::Int(_) => UniformType::Int,
      Pushable::UInt(_) => UniformType::Uint,
      Pushable::Bool(_) => UniformType::Bool,
      Pushable::Unknown => UniformType::Unknown,
    }
  }
}

impl Pushable {
  /// Attempts to retrieve the value as a 4x4 matrix.
  ///
  /// # Returns
  /// `Some([[f32; 4]; 4]])` if the value is a 4x4 matrix, otherwise `None`.
  ///
  /// # Examples
  /// ```
  /// let mat = Pushable::Mat4([[1.0; 4]; 4]);
  /// assert_eq!(mat.as_mat4(), Some([[1.0; 4]; 4]));
  /// assert_eq!(Pushable::Float(1.0).as_mat4(), None);
  /// ```
  pub fn as_mat4(&self) -> Option<[[f32; 4]; 4]> {
    match self {
      Pushable::Mat4(mat) => Some(*mat),
      _ => None,
    }
  }

  /// Attempts to retrieve the value as a 3x3 matrix.
  ///
  /// # Returns
  /// `Some([[f32; 3]; 3])` if the value is a 3x3 matrix, otherwise `None`.
  pub fn as_mat3(&self) -> Option<[[f32; 3]; 3]> {
    match self {
      Pushable::Mat3(mat) => Some(*mat),
      _ => None,
    }
  }

  /// Attempts to retrieve the value as a 2-component vector.
  ///
  /// # Returns
  /// `Some([f32; 2])` if the value is a 2-component vector, otherwise `None`.
  pub fn as_vec2(&self) -> Option<[f32; 2]> {
    match self {
      Pushable::Vec2(vec) => Some(*vec),
      _ => None,
    }
  }

  /// Attempts to retrieve the value as a 3-component vector.
  ///
  /// # Returns
  /// `Some([f32; 3])` if the value is a 3-component vector, otherwise `None`.
  ///
  /// # Examples
  /// ```
  /// let vec = Pushable::Vec3([1.0, 2.0, 3.0]);
  /// assert_eq!(vec.as_vec3(), Some([1.0, 2.0, 3.0]));
  /// assert_eq!(Pushable::Mat4([[1.0; 4]; 4]).as_vec3(), None);
  /// ```
  pub fn as_vec3(&self) -> Option<[f32; 3]> {
    match self {
      Pushable::Vec3(vec) => Some(*vec),
      _ => None,
    }
  }

  /// Attempts to retrieve the value as a 4-component vector.
  ///
  /// # Returns
  /// `Some([f32; 4])` if the value is a 4-component vector, otherwise `None`.
  ///
  /// # Examples
  /// ```
  /// let vec = Pushable::Vec4([1.0, 2.0, 3.0, 4.0]);
  /// assert_eq!(vec.as_vec4(), Some([1.0, 2.0, 3.0, 4.0]));
  /// assert_eq!(Pushable::Float(1.0).as_vec4(), None);
  /// ```
  pub fn as_vec4(&self) -> Option<[f32; 4]> {
    match self {
      Pushable::Vec4(vec) => Some(*vec),
      _ => None,
    }
  }

  /// Attempts to retrieve the value as a floating-point number.
  ///
  /// # Returns
  /// `Some(f32)` if the value is a floating-point number, otherwise `None`.
  ///
  /// # Examples
  /// ```
  /// let val = Pushable::Float(1.0);
  /// assert_eq!(val.as_float(), Some(1.0));
  /// assert_eq!(Pushable::Vec3([1.0, 2.0, 3.0]).as_float(), None);
  /// ```
  pub fn as_float(&self) -> Option<f32> {
    match self {
      Pushable::Float(val) => Some(*val),
      _ => None,
    }
  }

  /// Attempts to retrieve the value as an integer.
  ///
  /// # Returns
  /// `Some(i32)` if the value is an integer, otherwise `None`.
  ///
  /// # Examples
  /// ```
  /// let val = Pushable::Int(1);
  /// assert_eq!(val.as_int(), Some(1));
  /// assert_eq!(Pushable::Float(1.0).as_int(), None);
  /// ```
  pub fn as_int(&self) -> Option<i32> {
    match self {
      Pushable::Int(val) => Some(*val),
      _ => None,
    }
  }

  /// Attempts to retrieve the value as an unsigned integer.
  pub fn as_uint(&self) -> Option<u32> {
    match self {
      Pushable::UInt(val) => Some(*val),
      _ => None,
    }
  }

  /// Attempts to retrieve the value as a boolean.
  ///
  /// # Returns
  /// `Some(bool)` if the value is a boolean, otherwise `None`.
  pub fn as_bool(&self) -> Option<bool> {
    match self {
      Pushable::Bool(val) => Some(*val),
      _ => None,
    }
  }
}

/// Represents a single element within a uniform layout, including its type and memory offset.
#[derive(Debug, Clone, Copy)]
pub struct UniformElement {
  uniform_type: UniformType,
  offset: u32,
}

impl UniformElement {
  /// Creates a new `UniformElement` with the specified type and offset.
  pub fn new(uniform_type: UniformType, offset: u32) -> Self {
    UniformElement {
      uniform_type,
      offset,
    }
  }

  /// Returns the type of the uniform element.
  pub fn get_type(&self) -> &UniformType {
    &self.uniform_type
  }

  /// Determines the size of the element based on its type.
  pub fn size_of_element(uniform_type: &UniformType) -> usize {
    match uniform_type {
      UniformType::Mat4 => 64,
      UniformType::Mat3 => 48, // three vec4 columns (padded) in std140
      UniformType::Vec2 => 8,
      UniformType::Vec3 => 12,
      UniformType::Vec4 => 16,
      UniformType::UVec2 => 8,
      UniformType::UVec3 => 12,
      UniformType::UVec4 => 16,
      UniformType::Int | UniformType::Uint | UniformType::Float | UniformType::Bool => 4,
      _ => 0,
    }
  }

  /// Returns the std140 alignment for a uniform element.
  pub fn alignment_of_element(uniform_type: &UniformType) -> usize {
    match uniform_type {
      UniformType::Mat4 => 16,
      UniformType::Mat3 => 16,
      UniformType::Vec3 | UniformType::Vec4 | UniformType::UVec3 | UniformType::UVec4 => 16,
      UniformType::Vec2 => 8,
      UniformType::UVec2 => 8,
      UniformType::Int | UniformType::Uint | UniformType::Float | UniformType::Bool => 4,
      _ => 0,
    }
  }

  /// Returns the memory offset of the element.
  pub fn offset(&self) -> u32 {
    self.offset
  }

  /// Returns the size of the element.
  pub fn size(&self) -> usize {
    UniformElement::size_of_element(&self.uniform_type)
  }

  /// Calculates the end offset of the element.
  pub fn end_of_element(&self) -> u32 {
    self.offset + self.size() as u32
  }
}

/// Defines the layout of uniforms, specifying the structure and memory arrangement.
#[derive(Debug, Clone)]
pub struct UniformLayout {
  uniform_structure: Vec<UniformElement>,
  shader_stage: ShaderStage,
}

impl UniformLayout {
  /// Creates a new `UniformLayout` from a slice of `UniformType`.
  ///
  /// # Arguments
  ///
  /// * `types` - A slice containing the UniformTypes which makeup this uniform.
  pub fn new(types: &[UniformType], shader_stage: ShaderStage) -> Self {
    let mut offset = 0u32;
    let mut uniform_structure = Vec::new();

    for uniform_type in types {
      // Alignment based on std140 rules
      let aligned_offset = align_offset(offset, UniformElement::alignment_of_element(uniform_type));
      let element = UniformElement::new(*uniform_type, aligned_offset);
      offset = element.end_of_element();
      uniform_structure.push(element);
    }

    // total size should be a multiple of 16 bytes
    // let total_size = align_offset(offset, 16);
    // let padding = total_size - offset;
    // if padding > 0 {
    // }

    UniformLayout {
      uniform_structure,
      shader_stage,
    }
  }

  /// Calculates the total size of the layout in bytes.
  pub fn size(&self) -> u32 {
    self
      .uniform_structure
      .last()
      .map(|e| e.end_of_element())
      .unwrap_or(0)
  }

  /// Adds a new uniform type to the end of the layout.
  ///
  /// # Arguments
  ///
  /// * `uniform_type` - The type of the uniform to add.
  pub fn push_type_to_end(&mut self, uniform_type: UniformType) {
    let offset = self.size();
    let aligned_offset = align_offset(offset, UniformElement::alignment_of_element(&uniform_type));
    let element = UniformElement::new(uniform_type, aligned_offset);
    self.uniform_structure.push(element);
  }

  /// Finds the first `UniformElement` matching the specified type.
  ///
  /// # Arguments
  ///
  /// * `uniform_type` - The type to search for.
  ///
  /// # Returns
  ///
  /// An `Option` containing a reference to the matching `UniformElement` if found.
  pub fn find_in_structure(&self, uniform_type: &UniformType) -> Option<&UniformElement> {
    self
      .uniform_structure
      .iter()
      .find(|e| &e.uniform_type == uniform_type)
  }

  /// Retrieves a `UniformElement` by its index.
  ///
  /// # Arguments
  ///
  /// * `index` - The index of the element to retrieve.
  ///
  /// # Returns
  ///
  /// An `Option` containing a reference to the `UniformElement` if the index is valid.
  pub fn get(&self, index: usize) -> Option<&UniformElement> {
    self.uniform_structure.get(index)
  }

  /// Calculates the total aligned size of the layout.
  pub fn total_size(&self) -> usize {
    align_offset(self.size(), 16) as usize // std140 requires total size to be multiple of 16
  }

  /// Get the shader stage for this uniform layout
  pub fn stage(&self) -> ShaderStage {
    self.shader_stage
  }

  pub fn iter(&self) -> &Vec<UniformElement> {
    &self.uniform_structure
  }
}

/// Helper function to align offsets based on std140 rules.
///
/// # Arguments
///
/// * `offset` - The current offset.
/// * `alignment` - The required alignment.
///
/// # Returns
///
/// The next aligned offset.
fn align_offset(offset: u32, alignment: usize) -> u32 {
  let alignment = alignment as u32;
  ((offset + (alignment - 1)) / alignment) * alignment
}

/// Represents the data and layout of a uniform buffer.
#[derive(Debug, Clone)]
pub struct Uniform {
  data: Vec<u8>,
  layout: UniformLayout,
  pub is_in_gpu: bool,
  pub needs_update: bool,
}

impl IcsAsset for Uniform {}
impl IcsAsset for Mutex<Uniform> {}

impl Uniform {
  /// Initializes a new `Uniform` with allocated data based on the layout.
  ///
  /// # Arguments
  ///
  /// * `layout` - The layout of the uniforms.
  /// * `instance_count` - The number of instances.
  ///
  /// # Returns
  ///
  /// A new instance of `Uniform`.
  pub fn new(layout: UniformLayout) -> Self {
    let size = layout.total_size();
    Uniform {
      data: vec![0u8; size],
      layout,
      needs_update: false,
      is_in_gpu: false,
    }
  }

  /// Returns a reference to the uniform data.
  pub fn data(&self) -> &[u8] {
    &self.data
  }

  /// Returns a reference to the uniform layout.
  pub fn layout(&self) -> &UniformLayout {
    &self.layout
  }

  /// Pushes a value to a specific index in the uniform buffer.
  ///
  /// # Arguments
  ///
  /// * `pushables` - The slice of Pushables to push.
  ///
  /// # Errors
  ///
  /// Returns `IcsError` if push returns error.
  pub fn push_set(&mut self, pushables: &[Pushable]) -> Result<(), IcsError> {
    for (index, value) in pushables.iter().enumerate() {
      self.push(index, value)?;
    }
    Ok(())
  }

  /// Pushes a value to a specific index in the uniform buffer.
  ///
  /// # Arguments
  ///
  /// * `index` - The index within the layout to push the value to.
  /// * `value` - The value to push.
  ///
  /// # Errors
  ///
  /// Returns `IcsError` if the index is out of bounds or if the value type does not match.
  pub fn push(&mut self, index: usize, value: &Pushable) -> Result<(), IcsError> {
    let element = self.layout.get(index).ok_or_else(|| {
      ICS_ERROR!(
        why: "Uniform Buffer: Index out of bounds in layout",
        fix: "Check uniform pushes against the defined layout"
      )
    })?;

    let expected_size = element.size();
    let offset = element.offset() as usize;
    let end = offset + expected_size;

    if end > self.data.len() {
      return Err(ICS_ERROR!(
        why: "Uniform Buffer: Overflowing buffer on push",
        fix: "Ensure the buffer has sufficient size"
      ));
    }

    match element.get_type() {
      UniformType::Mat4 => {
        if let Some(mat) = value.as_mat4() {
          self.data[offset..end].copy_from_slice(Memory::as_bytes(&mat));
        } else {
          return Err(ICS_ERROR!(
            why: "Uniform Buffer: Expected a Mat4 value",
            fix: "Ensure the pushed uniform matches the layout type"
          ));
        }
      }

      UniformType::Mat3 => {
        if let Some(mat) = value.as_mat3() {
          let mut padded = [[0.0f32; 4]; 3];
          for i in 0..3 {
            for j in 0..3 {
              padded[i][j] = mat[i][j];
            }
          }
          let bytes = Memory::as_bytes(&padded);
          self.data[offset..(offset + bytes.len())].copy_from_slice(bytes);
        } else {
          return Err(ICS_ERROR!(
            why: "Uniform Buffer: Expected a Mat3 value",
            fix: "Ensure the pushed uniform matches the layout type"
          ));
        }
      }

      UniformType::Vec2 => {
        if let Some(vec) = value.as_vec2() {
          let bytes = Memory::as_bytes(&vec);
          self.data[offset..(offset + bytes.len())].copy_from_slice(bytes);
        } else {
          return Err(ICS_ERROR!(
            why: "Uniform Buffer: Expected a Vec2 value",
            fix: "Ensure the pushed uniform matches the layout type"
          ));
        }
      }

      UniformType::Vec3 => {
        if let Some(vec) = value.as_vec3() {
          let bytes = Memory::as_bytes(&vec);
          self.data[offset..(offset + bytes.len())].copy_from_slice(bytes);
        } else {
          return Err(ICS_ERROR!(
            why: "Uniform Buffer: Expected a Vec3 value",
            fix: "Ensure the pushed uniform matches the layout type"
          ));
        }
      }

      UniformType::Vec4 => {
        if let Some(vec) = value.as_vec4() {
          self.data[offset..end].copy_from_slice(Memory::as_bytes(&vec));
        } else {
          return Err(ICS_ERROR!(
            why: "Uniform Buffer: Expected a Vec4 value",
            fix: "Ensure the pushed uniform matches the layout type"
          ));
        }
      }

      UniformType::Float => {
        if let Some(val) = value.as_float() {
          self.data[offset..end].copy_from_slice(Memory::as_bytes(&val));
        } else {
          return Err(ICS_ERROR!(
            why: "Uniform Buffer: Expected a Float value",
            fix: "Ensure the pushed uniform matches the layout type"
          ));
        }
      }

      UniformType::Int => {
        if let Some(val) = value.as_int() {
          self.data[offset..end].copy_from_slice(Memory::as_bytes(&val));
        } else {
          return Err(ICS_ERROR!(
            why: "Uniform Buffer: Expected an Int value",
            fix: "Ensure the pushed uniform matches the layout type"
          ));
        }
      }

      UniformType::Uint => {
        if let Some(val) = value.as_uint() {
          self.data[offset..end].copy_from_slice(Memory::as_bytes(&val));
        } else {
          return Err(ICS_ERROR!(
            why: "Uniform Buffer: Expected a Uint value",
            fix: "Ensure the pushed uniform matches the layout type"
          ));
        }
      }

      UniformType::Bool => {
        if let Some(val) = value.as_bool() {
          let as_int: i32 = if val { 1 } else { 0 };
          self.data[offset..end].copy_from_slice(Memory::as_bytes(&as_int));
        } else {
          return Err(ICS_ERROR!(
            why: "Uniform Buffer: Expected a Bool value",
            fix: "Ensure the pushed uniform matches the layout type"
          ));
        }
      }

      _ => {
        return Err(ICS_ERROR!(
          why: "Uniform Buffer: Unknown uniform type or texture type selected",
          fix: "Check uniform type definitions"
        ));
      }
    }

    Ok(())
  }
}

/// Manages a collection of uniforms organized by hierarchy nodes and uniform keys.
/// Each node can have multiple uniforms with the same key.
#[derive(Debug, Clone)]
pub struct Uniforms {
  map: HashMap<NodeId, HashMap<ConcreteUniform, Vec<Arc<Mutex<Uniform>>>>>,
}

impl Uniforms {
  /// Creates a new, empty `Uniforms` manager.
  ///
  /// # Returns
  ///
  /// A new instance of `Uniforms`.
  pub fn new() -> Self {
    Uniforms {
      map: HashMap::new(),
    }
  }

  /// Adds a new uniform to a specific node and key.
  ///
  /// # Arguments
  ///
  /// * `node_name` - The name of the hierarchy node.
  /// * `uniform_key` - The key identifying the uniform within the node.
  /// * `uniform` - The `Arc<Mutex<Uniform>>` instance to add.
  ///
  /// # Errors
  ///
  /// Returns an `IcsError` if the operation fails.
  pub fn add_uniform(
    &mut self,
    node_name: &NodeId,
    uniform_key: ConcreteUniform,
    uniform: Arc<Mutex<Uniform>>,
  ) -> Result<(), IcsError> {
    let node = self
      .map
      .entry(node_name.clone())
      .or_insert_with(HashMap::new);
    let uniforms = node.entry(uniform_key).or_insert_with(Vec::new);
    uniforms.push(uniform);
    Ok(())
  }

  /// Retrieves all uniforms for a specific node and uniform key.
  ///
  /// # Arguments
  ///
  /// * `node_name` - The name of the hierarchy node.
  /// * `uniform_key` - The key identifying the uniform within the node.
  ///
  /// # Returns
  ///
  /// An `Option` containing a reference to the vector of `Arc<Mutex<Uniform>>` if found.
  pub fn uniforms(
    &self,
    node_name: &NodeId,
    uniform_key: &ConcreteUniform,
  ) -> Option<&Vec<Arc<Mutex<Uniform>>>> {
    self.map.get(node_name)?.get(uniform_key)
  }

  /// Retrieves all uniforms for a specific node.
  ///
  /// # Arguments
  ///
  /// * `node_name` - The name of the hierarchy node.
  ///
  /// # Returns
  ///
  /// An `Option` containing a reference to the inner `HashMap` of uniform keys and their vectors.
  pub fn uniforms_for_node(
    &self,
    node_name: &NodeId,
  ) -> Option<&HashMap<ConcreteUniform, Vec<Arc<Mutex<Uniform>>>>> {
    self.map.get(node_name)
  }

  /// Retrieves a specific uniform for a given node and key by index.
  ///
  /// # Arguments
  ///
  /// * `node_name` - The name of the hierarchy node.
  /// * `uniform_key` - The key identifying the uniform within the node.
  /// * `index` - The index of the uniform in the vector.
  ///
  /// # Returns
  ///
  /// An `Option` containing the `Arc<Mutex<Uniform>>` if found.
  pub fn uniform(
    &self,
    node_name: &NodeId,
    uniform_key: &ConcreteUniform,
    instance: usize,
  ) -> Option<Arc<Mutex<Uniform>>> {
    self
      .map
      .get(node_name)?
      .get(uniform_key)?
      .get(instance)
      .cloned()
  }

  /// Retrieves all uniforms across all nodes and keys.
  ///
  /// This function provides a reference to the entire map of uniforms, including their
  /// associated nodes and keys.
  ///
  /// # Returns
  ///
  /// A reference to the `HashMap` containing all uniforms.
  ///
  /// # Example
  ///
  /// ```
  /// let uniforms = Uniforms::new();
  /// let map = uniforms.map();
  /// ```
  pub fn map(&self) -> &HashMap<NodeId, HashMap<ConcreteUniform, Vec<Arc<Mutex<Uniform>>>>> {
    &self.map
  }

  /// Retrieves a mutable reference to all uniforms across all nodes and keys.
  pub fn map_mut(
    &mut self,
  ) -> &mut HashMap<NodeId, HashMap<ConcreteUniform, Vec<Arc<Mutex<Uniform>>>>> {
    &mut self.map
  }
}

impl std::fmt::Display for Uniforms {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    writeln!(f, "Uniforms:")?;
    for (node, uniforms) in &self.map {
      writeln!(f, "  Node '{}':", node)?;
      for (key, uniform_vec) in uniforms {
        writeln!(f, "    Uniform Key '{}':", key.to_string())?;
        for (i, uniform) in uniform_vec.iter().enumerate() {
          let uniform = uniform.lock().unwrap();
          writeln!(f, "      [{}]: {:?}", i, uniform)?;
        }
      }
    }
    Ok(())
  }
}

/// Represents the different shader stages where uniforms can be applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderStage {
  Vertex,
  Fragment,
  Geometry,
  Compute,
  TessellationControl,
  TessellationEvaluation,
  AllGraphics,
  All,
  Unknown,
}

impl ShaderStage {
  /// Converts a string to a `ShaderStage`.
  ///
  /// # Parameters
  ///
  /// - `str`: The string representing a shader stage, such as `"vertex"` or `"fragment"`.
  ///
  /// # Returns
  ///
  /// The corresponding `ShaderStage` variant.
  ///
  /// # Panics
  ///
  /// Panics if the string does not match a valid shader stage.
  ///
  /// # Example
  ///
  /// ```
  /// let stage = ShaderStage::from_string("vertex");
  /// ```
  pub fn from_string(str: &str) -> ShaderStage {
    match str {
      "vertex" => ShaderStage::Vertex,
      "fragment" | "pixel" => ShaderStage::Fragment,
      _ => panic!("Invalid stage string"),
    }
  }
}

impl std::fmt::Display for ShaderStage {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let stage_name = match self {
      ShaderStage::Vertex => "Vertex",
      ShaderStage::Fragment => "Fragment",
      ShaderStage::Geometry => "Geometry",
      ShaderStage::Compute => "Compute",
      ShaderStage::TessellationControl => "TessellationControl",
      ShaderStage::TessellationEvaluation => "TessellationEvaluation",
      ShaderStage::AllGraphics => "AllGraphics",
      ShaderStage::All => "All",
      ShaderStage::Unknown => "Unknown",
    };
    write!(f, "{}", stage_name)
  }
}

impl std::fmt::Display for UniformType {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let type_name = match self {
      UniformType::Mat4 => "Mat4",
      UniformType::Mat3 => "Mat3",
      UniformType::Vec2 => "Vec2",
      UniformType::Vec3 => "Vec3",
      UniformType::Vec4 => "Vec4",
      UniformType::UVec2 => "UVec2",
      UniformType::UVec3 => "UVec3",
      UniformType::UVec4 => "UVec4",
      UniformType::Float => "Float",
      UniformType::Int => "Int",
      UniformType::Uint => "Uint",
      UniformType::Unknown => "Unknown",
      UniformType::Sampler2D => "Sampler2D",
      UniformType::Bool => "Bool",
    };
    write!(f, "{}", type_name)
  }
}

impl std::fmt::Display for UniformLayout {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    writeln!(f, "UniformLayout:")?;
    writeln!(f, "ShaderStage: {}", self.shader_stage)?;
    for (i, element) in self.uniform_structure.iter().enumerate() {
      writeln!(f, "  [{}]: {}", i, element)?;
    }
    Ok(())
  }
}

impl std::fmt::Display for UniformElement {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "UniformElement {{ type: {}, offset: {}, size: {} }}",
      self.uniform_type,
      self.offset,
      self.size(),
    )
  }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum ConcreteUniform {
  State,
  Light,
  Camera,
  Material,
  ModelViewProject,
}

impl ConcreteUniform {
  /// Converts the `ConcreteUniform` to its string representation.
  fn to_string(&self) -> String {
    format!("{}", self)
  }

  /// Converts a string to a `ConcreteUniform`.
  ///
  /// # Parameters
  ///
  /// - `str`: The string representing a concrete uniform, such as `"state"`, `"light"`, etc.
  ///
  /// # Returns
  ///
  /// The corresponding `ConcreteUniform` variant.
  ///
  /// # Panics
  ///
  /// Panics if the string does not match a valid concrete uniform.
  pub fn from_string(str: &str) -> ConcreteUniform {
    match str {
      "state" => ConcreteUniform::State,
      "light" => ConcreteUniform::Light,
      "camera" => ConcreteUniform::Camera,
      "material" => ConcreteUniform::Material,
      "modelviewproject" | "mvp" => ConcreteUniform::ModelViewProject,
      _ => panic!("Invalid uniform type from string"),
    }
  }

  pub fn try_from_string(str: &str) -> Option<ConcreteUniform> {
    match str {
      "state" => Some(ConcreteUniform::State),
      "light" => Some(ConcreteUniform::Light),
      "camera" => Some(ConcreteUniform::Camera),
      "material" => Some(ConcreteUniform::Material),
      "modelviewproject" | "mvp" => Some(ConcreteUniform::ModelViewProject),
      _ => None,
    }
  }
}

impl std::fmt::Display for ConcreteUniform {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let str = match self {
      ConcreteUniform::ModelViewProject => "mvp",
      ConcreteUniform::Light => "light",
      ConcreteUniform::Camera => "camera",
      ConcreteUniform::Material => "material",
      ConcreteUniform::State => "state",
    };
    write!(f, "{}", str)
  }
}

#[derive(Clone, Debug)]
pub struct CustomUniform {
  pub key: ConcreteUniform,
  pub identifier: String,
  pub instances: usize,
  pub binding: usize,
  pub stage: ShaderStage,
  pub layout: Vec<(String, Vec<Pushable>, UniformType)>,
}

impl PartialEq for CustomUniform {
  fn eq(&self, other: &Self) -> bool {
    self.key == other.key
      && self.identifier == other.identifier
      && self.stage == other.stage
      && self
        .layout
        .iter()
        .zip(&other.layout)
        .all(|((name1, _, type1), (name2, _, type2))| name1 == name2 && type1 == type2)
  }
}

impl Eq for CustomUniform {}

impl CustomUniform {
  /// Creates a new custom uniform.
  ///
  /// # Parameters
  ///
  /// - `instances`: The number of instances for the uniform.
  /// - `key`: The `ConcreteUniform` key for the uniform.
  /// - `stage`: The shader stage the uniform is used in.
  /// - `layout`: The layout of the uniform, including its name, values, and type.
  ///
  /// # Returns
  ///
  /// A new `CustomUniform` instance.
  pub fn new(
    instances: usize,
    key: ConcreteUniform,
    stage: ShaderStage,
    layout: &[(String, Vec<Pushable>, UniformType)],
  ) -> CustomUniform {
    CustomUniform {
      key: key.clone(),
      identifier: key.to_string(),
      binding: 0,
      instances,
      stage,
      layout: layout.to_vec(),
    }
  }

  /// Retrieves the key associated with the custom uniform.
  pub fn key(&self) -> &ConcreteUniform {
    &self.key
  }

  /// Retrieves the binding index for the custom uniform.
  pub fn binding(&self) -> usize {
    self.binding
  }

  /// Retrieves the identifier used inside shader code for this uniform.
  pub fn identifier(&self) -> &str {
    &self.identifier
  }

  /// Sets the binding index for the custom uniform.
  ///
  /// # Arguments
  ///
  /// - `binding`: The new binding index.
  pub fn set_binding(&mut self, binding: usize) {
    self.binding = binding;
  }

  /// Retrieves the number of instances for the custom uniform.
  pub fn instances(&self) -> usize {
    self.instances
  }

  /// Sets the number of instances for the custom uniform.
  ///
  /// # Arguments
  ///
  /// - `instances`: The new number of instances.
  pub fn set_instance(&mut self, instances: usize) {
    self.instances = instances;
  }

  /// Retrieves the shader stage for the custom uniform.
  pub fn stage(&self) -> ShaderStage {
    self.stage
  }

  /// Retrieves the layout of the custom uniform.
  pub fn layout(&self) -> Vec<(String, Vec<Pushable>, UniformType)> {
    self.layout.clone()
  }

  // pub fn update(&mut self, instance: usize, entity: &Arc<Entity>, data: &[Pushable]) {
  //   let mut uniforms_guard = entity.uniforms().lock().unwrap();
  //   let uniforms_map = &mut uniforms_guard.map_mut();
  //   for (_, uniform_vec) in uniforms_map.get_mut(self.key()).unwrap().iter_mut() {
  //     let mut uniform = uniform_vec[instance].lock().unwrap();
  //     uniform.push_set(data).unwrap();
  //   }
  // }
  //
  // pub fn update_at(&mut self, instance: usize, entity: &Arc<Entity>, index: usize, data: &Pushable) {
  //   let mut uniforms_guard = entity.uniforms().lock().unwrap();
  //   let uniforms_map = &mut uniforms_guard.map_mut();
  //   for (_, uniform_map) in uniforms_map.iter_mut() {
  //     if let Some(uniform) = uniform_map.get_mut(&self.key()) {
  //       uniform[instance].lock().unwrap().push(index, data).unwrap();
  //     }
  //   }
  // }

  /// Creates a custom uniform from a concrete uniform type.
  ///
  /// # Parameters
  ///
  /// - `ty`: The `ConcreteUniform` type (e.g., `ModelViewProject`, `Camera`).
  ///
  /// # Returns
  ///
  /// The corresponding `CustomUniform`.
  ///
  /// # Panics
  ///
  /// Panics if the concrete type is not `ModelViewProject` or `Camera`.
  pub fn from_concrete(ty: ConcreteUniform) -> CustomUniform {
    match ty {
      ConcreteUniform::State => StateUniform::def(1),
      ConcreteUniform::ModelViewProject => ModelViewProjectUniform::def(1),
      ConcreteUniform::Camera => CameraUniform::def(1),
      // ConcreteUniform::Light => LightUniform::def(1),
      // ConcreteUniform::Material => MaterialUniform::def(1),
      _ => panic!("Can only build state, camera, or mvp uniforms from concrete type, use semantics for other concrete uniforms"),
    }
  }
}

#[derive(Clone, Debug)]
pub struct StateUniform {
  inner: CustomUniform,
}

impl StateUniform {
  pub fn def(instances: usize) -> CustomUniform {
    CustomUniform {
      key: ConcreteUniform::State,
      identifier: ConcreteUniform::State.to_string(),
      binding: 0,
      instances,
      stage: ShaderStage::All,
      layout: vec![
        (
          "camera_index".to_string(),
          vec![Pushable::Int(0); instances],
          UniformType::Int,
        ),
        (
          "camera_count".to_string(),
          vec![Pushable::Int(0); instances],
          UniformType::Int,
        ),
        (
          "object_index".to_string(),
          vec![Pushable::Int(0); instances],
          UniformType::Int,
        ),
        (
          "object_count".to_string(),
          vec![Pushable::Int(0); instances],
          UniformType::Int,
        ),
        (
          "material_index".to_string(),
          vec![Pushable::Int(0); instances],
          UniformType::Int,
        ),
        (
          "material_count".to_string(),
          vec![Pushable::Int(0); instances],
          UniformType::Int,
        ),
      ],
    }
  }

  pub fn binding(&mut self, binding: usize) {
    self.inner.binding = binding;
  }

  pub fn inner(&mut self) -> &mut CustomUniform {
    &mut self.inner
  }
}

#[derive(Clone, Debug)]
pub struct ModelViewProjectUniform {
  inner: CustomUniform,
}

impl ModelViewProjectUniform {
  /// Creates a new `CustomUniform` for the model-view-projection uniform.
  ///
  /// This function initializes a `CustomUniform` with the `ModelViewProject` key, the `Vertex` shader stage,
  /// and a layout containing three matrices (`model`, `view`, and `proj`), each initialized as identity matrices.
  ///
  /// # Arguments
  ///
  /// * `instances` - The number of instances for the uniform data.
  ///
  /// # Returns
  ///
  /// A `CustomUniform` configured for the model-view-projection uniform.
  ///
  /// # Example
  ///
  /// ```rust
  /// let mvp_uniform = ModelViewProjectUniform::def(1);
  /// ```
  pub fn def(instances: usize) -> CustomUniform {
    CustomUniform {
      key: ConcreteUniform::ModelViewProject,
      identifier: ConcreteUniform::ModelViewProject.to_string(),
      binding: 0,
      instances,
      stage: ShaderStage::Vertex,
      layout: vec![
        (
          "model".to_string(),
          vec![Pushable::Mat4(Matrix::<4, 4, f32>::identity().as_slice()); instances],
          UniformType::Mat4,
        ),
        (
          "view".to_string(),
          vec![Pushable::Mat4(Matrix::<4, 4, f32>::identity().as_slice()); instances],
          UniformType::Mat4,
        ),
        (
          "proj".to_string(),
          vec![Pushable::Mat4(Matrix::<4, 4, f32>::identity().as_slice()); instances],
          UniformType::Mat4,
        ),
      ],
    }
  }

  /// Sets the binding point for the uniform.
  ///
  /// # Arguments
  ///
  /// * `binding` - The binding point index for the uniform.
  pub fn binding(&mut self, binding: usize) {
    self.inner.binding = binding;
  }

  /// Provides mutable access to the underlying `CustomUniform`.
  ///
  /// # Example
  ///
  /// ```rust
  /// let mut mvp_uniform = ModelViewProjectUniform::def(1);
  /// let custom_uniform = mvp_uniform.inner();
  /// ```
  pub fn inner(&mut self) -> &mut CustomUniform {
    &mut self.inner
  }
}

#[derive(Clone)]
pub struct CameraUniform {
  inner: CustomUniform,
}

impl CameraUniform {
  /// Creates a new `CustomUniform` for the camera uniform.
  ///
  /// This function initializes a `CustomUniform` with the `Camera` key, the `Fragment` shader stage,
  /// and a layout containing one vector (`position`), initialized as `[0.0, 0.0, 0.0, 0.0]`.
  ///
  /// # Arguments
  ///
  /// * `instances` - The number of instances for the uniform data.
  ///
  /// # Returns
  ///
  /// A `CustomUniform` configured for the camera uniform.
  ///
  /// # Example
  ///
  /// ```rust
  /// let camera_uniform = CameraUniform::def(1);
  /// ```
  pub fn def(instances: usize) -> CustomUniform {
    CustomUniform {
      key: ConcreteUniform::Camera,
      identifier: ConcreteUniform::Camera.to_string(),
      binding: 0,
      instances,
      stage: ShaderStage::Fragment,
      layout: vec![(
        "position".to_string(),
        vec![Pushable::Vec4([0.0, 0.0, 0.0, 0.0]); instances],
        UniformType::Vec4,
      )],
    }
  }

  /// Sets the binding point for the uniform.
  ///
  /// # Arguments
  ///
  /// * `binding` - The binding point index for the uniform.
  pub fn binding(&mut self, binding: usize) {
    self.inner.binding = binding;
  }

  /// Provides mutable access to the underlying `CustomUniform`.
  ///
  /// # Example
  ///
  /// ```rust
  /// let mut cam_uniform = CameraUniform::def(1);
  /// let custom_uniform = cam_uniform.inner();
  /// ```
  pub fn inner(&mut self) -> &mut CustomUniform {
    &mut self.inner
  }
}

// #[derive(Clone)]
// pub struct LightUniform {
//   inner: CustomUniform,
// }
//
// impl LightUniform {
//   /// Creates a new `CustomUniform` describing a light (position + colour components).
//   pub fn def(instances: usize) -> CustomUniform {
//     CustomUniform {
//       key: ConcreteUniform::Light,
//       identifier: ConcreteUniform::Light.to_string(),
//       binding: 0,
//       instances,
//       stage: ShaderStage::Fragment,
//       layout: vec![
//         (
//           "position".to_string(),
//           vec![Pushable::Vec3([0.0, 0.0, 0.0]); instances],
//           UniformType::Vec3,
//         ),
//         (
//           "ambient".to_string(),
//           vec![Pushable::Vec3([0.1, 0.1, 0.1]); instances],
//           UniformType::Vec3,
//         ),
//         (
//           "diffuse".to_string(),
//           vec![Pushable::Vec3([1.0, 1.0, 1.0]); instances],
//           UniformType::Vec3,
//         ),
//         (
//           "specular".to_string(),
//           vec![Pushable::Vec3([1.0, 1.0, 1.0]); instances],
//           UniformType::Vec3,
//         ),
//       ],
//     }
//   }
//
//   /// Sets the binding point for the uniform.
//   ///
//   /// # Arguments
//   ///
//   /// * `binding` - The binding point index for the uniform.
//   pub fn binding(&mut self, binding: usize) {
//     self.inner.binding = binding;
//   }
//
//   /// Provides mutable access to the underlying `CustomUniform`.
//   ///
//   /// # Example
//   ///
//   /// ```rust
//   /// let mut cam_uniform = CameraUniform::def(1);
//   /// let custom_uniform = cam_uniform.inner();
//   /// ```
//   pub fn inner(&mut self) -> &mut CustomUniform {
//     &mut self.inner
//   }
// }
//
// #[derive(Clone)]
// pub struct MaterialUniform {
//   inner: CustomUniform,
// }
//
// impl MaterialUniform {
//   /// Creates a new `CustomUniform` describing material properties for Phong lighting.
//   pub fn def(instances: usize) -> CustomUniform {
//     CustomUniform {
//       key: ConcreteUniform::Material,
//       identifier: ConcreteUniform::Material.to_string(),
//       binding: 0,
//       instances,
//       stage: ShaderStage::Fragment,
//       layout: vec![
//         (
//           "ambient".to_string(),
//           vec![Pushable::Vec3([0.1, 0.1, 0.1]); instances],
//           UniformType::Vec3,
//         ),
//         (
//           "specular".to_string(),
//           vec![Pushable::Vec3([1.0, 1.0, 1.0]); instances],
//           UniformType::Vec3,
//         ),
//         (
//           "shininess".to_string(),
//           vec![Pushable::Float(32.0); instances],
//           UniformType::Float,
//         ),
//         (
//           "metallic".to_string(),
//           vec![Pushable::Float(0.0); instances],
//           UniformType::Float,
//         ),
//       ],
//     }
//   }
//
//   pub fn binding(&mut self, binding: usize) {
//     self.inner.binding = binding;
//   }
//
//   pub fn inner(&mut self) -> &mut CustomUniform {
//     &mut self.inner
//   }
// }
