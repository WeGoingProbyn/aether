use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};

use crate::debugger::IcsError;
use crate::maths::vector::Vector;
use crate::memory::memory::Memory;
use crate::{ICS_ERROR, ICS_WARN};

use super::hierarchy::{HierarchyNode, NodeId};
use super::indices::{Indices, IndicesPart};
use super::pipeline::PipelineAttribute;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AttributeType {
  Vector2D,
  Vector3D,
  Vector4D,
  Texture2D,
  Normal,
  ColourF3,
  ColourF4,
  UnknownF2,
  UnknownF3,
  UnknownF4,
}

impl fmt::Display for AttributeType {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let type_name = match self {
      AttributeType::Vector2D => "Vector2D",
      AttributeType::Vector3D => "Vector3D",
      AttributeType::Vector4D => "Vector4D",
      AttributeType::Texture2D => "Texture2D",
      AttributeType::Normal => "Normal",
      AttributeType::ColourF3 => "ColourF3",
      AttributeType::ColourF4 => "ColourF4",
      AttributeType::UnknownF2 => "UnknownF2",
      AttributeType::UnknownF3 => "UnknownF3",
      AttributeType::UnknownF4 => "UnknownF4",
    };
    write!(f, "{}", type_name)
  }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pushable {
  Vector2([f32; 2]),
  Vector3([f32; 3]),
  Vector4([f32; 4]),
}

impl Pushable {
  fn to_vector2(&self) -> Option<Vector<2, f32>> {
    match self {
      Pushable::Vector2(arr) => Some(Vector::<2, f32>::from(*arr)),
      _ => None,
    }
  }

  fn to_vector3(&self) -> Option<Vector<3, f32>> {
    match self {
      Pushable::Vector3(arr) => Some(Vector::<3, f32>::from(*arr)),
      _ => None,
    }
  }

  fn to_vector4(&self) -> Option<Vector<4, f32>> {
    match self {
      Pushable::Vector4(arr) => Some(Vector::<4, f32>::from(*arr)),
      _ => None,
    }
  }
}

impl Hash for Pushable {
  fn hash<H: Hasher>(&self, state: &mut H) {
    match self {
      Pushable::Vector2(arr) => {
        0u8.hash(state);
        for &value in arr.iter() {
          if value.is_nan() {
            ICS_WARN!("Vertices: Trying to hash a NaN");
          } else {
            value.to_bits().hash(state);
          }
        }
      }
      Pushable::Vector3(arr) => {
        1u8.hash(state);
        for &value in arr.iter() {
          if value.is_nan() {
            ICS_WARN!("Vertices: Trying to hash a NaN");
          } else {
            value.to_bits().hash(state);
          }
        }
      }
      Pushable::Vector4(arr) => {
        2u8.hash(state);
        for &value in arr.iter() {
          if value.is_nan() {
            ICS_WARN!("Vertices: Trying to hash a NaN");
          } else {
            value.to_bits().hash(state);
          }
        }
      }
    }
  }
}

/// Represents a single attribute with its type and data.
#[derive(Debug, Clone, Hash)]
pub struct Attribute {
  attribute_type: AttributeType,
  data: Vec<Pushable>, // Pushable encapsulated data for the attribute
}

impl Attribute {
  pub fn new(attribute_type: AttributeType) -> Self {
    Attribute {
      attribute_type,
      data: Vec::new(),
    }
  }

  /// Pushes new structured data to the attribute's data vector.
  pub fn push(&mut self, pushables: &[Pushable]) -> Result<(), IcsError> {
    for value in pushables {
      match self.attribute_type {
        AttributeType::Vector2D | AttributeType::Texture2D | AttributeType::UnknownF2 => {
          if let Pushable::Vector2(_) = value {
            self.data.push(value.clone());
          } else {
            return Err(ICS_ERROR!(
              why: "Attribute: Mismatched Pushable type for Vector2D",
              fix: "Ensure the Pushable matches the AttributeType"
            ));
          }
        }
        AttributeType::Vector3D
        | AttributeType::Normal
        | AttributeType::ColourF3
        | AttributeType::UnknownF3 => {
          if let Pushable::Vector3(_) = value {
            self.data.push(value.clone());
          } else {
            return Err(ICS_ERROR!(
              why: "Attribute: Mismatched Pushable type for Vector3D",
              fix: "Ensure the Pushable matches the AttributeType"
            ));
          }
        }
        AttributeType::Vector4D | AttributeType::ColourF4 | AttributeType::UnknownF4 => {
          if let Pushable::Vector4(_) = value {
            self.data.push(value.clone());
          } else {
            return Err(ICS_ERROR!(
              why: "Attribute: Mismatched Pushable type for Vector4D",
              fix: "Ensure the Pushable matches the AttributeType"
            ));
          }
        }
      }
    }
    Ok(())
  }

  pub fn pushables(&self) -> &Vec<Pushable> {
    &self.data
  }

  /// Retrieves the number of vertices for this attribute.
  pub fn vertex_count(&self) -> usize {
    self.data.len()
  }

  /// Returns the size in bytes of a single element of this attribute.
  pub fn size_of_element(&self) -> usize {
    match self.attribute_type {
      AttributeType::Vector2D | AttributeType::Texture2D | AttributeType::UnknownF2 => {
        std::mem::size_of::<Vector<2, f32>>()
      }
      AttributeType::Vector3D
      | AttributeType::Normal
      | AttributeType::ColourF3
      | AttributeType::UnknownF3 => std::mem::size_of::<Vector<3, f32>>(),
      AttributeType::Vector4D | AttributeType::ColourF4 | AttributeType::UnknownF4 => {
        std::mem::size_of::<Vector<4, f32>>()
      }
    }
  }

  /// Retrieves a serialized byte vector for a specific vertex.
  pub fn vertex_bytes(&self, index: usize) -> Option<Vec<u8>> {
    self.data.get(index).map(|pushable| match pushable {
      Pushable::Vector2(arr) => Memory::as_bytes(arr).to_vec(),
      Pushable::Vector3(arr) => Memory::as_bytes(arr).to_vec(),
      Pushable::Vector4(arr) => Memory::as_bytes(arr).to_vec(),
    })
  }
}

/// Manages all attributes, identified by their names.
#[derive(Debug, Clone)]
pub struct Vertices {
  // Maps attribute names to their corresponding Attribute data
  attributes: HashMap<String, Attribute>,
  available: HashSet<PipelineAttribute>,
}

impl Vertices {
  pub fn new() -> Self {
    Vertices {
      attributes: HashMap::new(),
      available: HashSet::new(),
    }
  }

  pub fn available_attributes(&self) -> &HashSet<PipelineAttribute> {
    &self.available
  }

  pub fn attribute(&self, name: &str) -> &Attribute {
    self.attributes.get(name).as_ref().unwrap()
  }

  fn pipeline_req_to_key(pipeline_req: &PipelineAttribute) -> &str {
    match pipeline_req {
      PipelineAttribute::Colour => "colour",
      PipelineAttribute::Normal => "normal",
      PipelineAttribute::Texture => "texture",
      PipelineAttribute::Tangent => "tangent",
      PipelineAttribute::Position => "position",
      PipelineAttribute::BitTangent => "bit_tangent",
    }
  }

  /// Adds a new attribute with the given name and type.
  pub fn add_attribute(
    &mut self,
    pipeline_req: PipelineAttribute,
    attribute_type: AttributeType,
  ) -> Result<(), IcsError> {
    let name = Vertices::pipeline_req_to_key(&pipeline_req);
    if self.attributes.contains_key(name) {
      return Err(ICS_ERROR!(
        why: "Attributes: Attribute name already exists",
        fix: "Use a unique name for each attribute"
      ));
    }
    self
      .attributes
      .insert(name.to_string(), Attribute::new(attribute_type.clone()));

    self.available.insert(pipeline_req);
    Ok(())
  }

  /// Pushes structured data to the specified attribute.
  pub fn push_attribute(
    &mut self,
    pipeline_req: PipelineAttribute,
    data: &[Pushable],
  ) -> Result<(), IcsError> {
    let name = Vertices::pipeline_req_to_key(&pipeline_req);
    if let Some(attr) = self.attributes.get_mut(name) {
      attr.push(data)
    } else {
      Err(ICS_ERROR!(
        why: format!("Attributes: Attribute '{}' does not exist", name),
        fix: "Add the attribute using `add_attribute` before pushing data"
      ))
    }
  }

  /// Retrieves serialized attribute data as Vec<u8> for a specific vertex.
  pub fn attribute_data_at(&self, name: &str, index: usize) -> Option<Vec<u8>> {
    self.attributes.get(name)?.vertex_bytes(index)
  }

  /// Returns the number of vertices (assuming all attributes have the same length)
  pub fn vertex_count(&self) -> usize {
    self
      .attributes
      .values()
      .map(|attr| attr.vertex_count())
      .max()
      .unwrap_or(0)
  }

  /// Retrieves the AttributeType for a given attribute name.
  pub fn get_attribute_type(&self, name: &str) -> Option<&AttributeType> {
    self.attributes.get(name).map(|attr| &attr.attribute_type)
  }

  /// Returns all attribute names.
  pub fn get_attribute_names(&self) -> Vec<String> {
    self.attributes.keys().cloned().collect()
  }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Element {
  pub name: String,
  pub attribute_type: AttributeType,
  pub offset: u32,
}

impl Element {
  pub fn new(name: String, attribute_type: AttributeType, offset: u32) -> Self {
    Element {
      name,
      attribute_type,
      offset,
    }
  }

  pub fn attribute_type(&self) -> &AttributeType {
    &self.attribute_type
  }

  pub fn size_of_element(attribute_type: &AttributeType) -> usize {
    match attribute_type {
      AttributeType::Vector2D | AttributeType::Texture2D | AttributeType::UnknownF2 => {
        std::mem::size_of::<Vector<2, f32>>()
      }
      AttributeType::Vector3D
      | AttributeType::Normal
      | AttributeType::ColourF3
      | AttributeType::UnknownF3 => std::mem::size_of::<Vector<3, f32>>(),
      AttributeType::Vector4D | AttributeType::ColourF4 | AttributeType::UnknownF4 => {
        std::mem::size_of::<Vector<4, f32>>()
      }
    }
  }

  pub fn offset(&self) -> u32 {
    self.offset
  }

  pub fn size(&self) -> usize {
    Element::size_of_element(&self.attribute_type)
  }

  pub fn end_of_element(&self) -> u32 {
    self.offset + self.size() as u32
  }
}

impl fmt::Display for Element {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(
      f,
      "Element {{ name: {}, type: {}, offset: {}, size: {} }}",
      self.name,
      self.attribute_type,
      self.offset,
      self.size()
    )
  }
}

/// Defines the layout of a vertex by specifying the order and types of attributes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Layout {
  pub vertex_structure: Vec<Element>,
  pub total_size: u32, // Total size of a single vertex in bytes
}

impl Layout {
  pub fn new(
    attribute_names: &[String],
    attribute_types: &[AttributeType],
  ) -> Result<Self, IcsError> {
    let mut vertex_structure = Vec::new();
    let mut offset = 0u32;

    for (i, attribute_type) in attribute_types.iter().enumerate() {
      // let attribute = vertices.attributes.get(name).ok_or_else(|| {
      //   ICS_ERROR!(
      //     why: format!("Layout: Attribute '{}' not found in Vertices", name),
      //     fix: "Ensure the attribute is added to Vertices before defining the layout"
      //   )
      // })?;

      let size = Element::size_of_element(&attribute_type);
      vertex_structure.push(Element::new(
        attribute_names[i].to_string(),
        attribute_type.clone(),
        offset,
      ));
      offset += size as u32;
    }

    Ok(Layout {
      vertex_structure,
      total_size: offset,
    })
  }

  pub fn size(&self) -> u32 {
    self.total_size
  }

  pub fn push(&mut self, element: Element) {
    let size = element.size() as u32;
    self.vertex_structure.push(element);
    self.total_size += size;
  }

  pub fn find_in_structure(&self, element_type: &AttributeType) -> Option<&Element> {
    self
      .vertex_structure
      .iter()
      .find(|e| &e.attribute_type == element_type)
  }

  pub fn get(&self, index: usize) -> Option<&Element> {
    self.vertex_structure.get(index)
  }

  pub fn structure(&self) -> &Vec<Element> {
    &self.vertex_structure
  }
}

impl fmt::Display for Layout {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    writeln!(f, "Layout:")?;
    for (i, element) in self.vertex_structure.iter().enumerate() {
      writeln!(f, "  [{}]: {}", i, element)?;
    }
    writeln!(f, "Total Size: {} bytes", self.total_size)
  }
}

#[derive(Debug, Clone)]
pub struct VertexBuffer {
  pub layout: Layout,
  pub byte_blob: Vec<u8>,
}

impl VertexBuffer {
  pub fn new(layout: Layout) -> Self {
    VertexBuffer {
      layout,
      byte_blob: Vec::new(),
    }
  }

  pub fn layout(&self) -> &Layout {
    &self.layout
  }

  pub fn as_slice(&self) -> &[u8] {
    &self.byte_blob
  }

  pub fn size(&self) -> u64 {
    self.byte_blob.len() as u64
  }

  pub fn number_vertices(&self) -> usize {
    if self.layout.size() == 0 {
      0
    } else {
      self.byte_blob.len() / self.layout.size() as usize
    }
  }

  /// Retrieves the raw byte data of a vertex at the specified index.
  pub fn vertex(&self, index: usize) -> Option<&[u8]> {
    let vertex_size = self.layout.size() as usize;
    let start = index * vertex_size;
    let end = start + vertex_size;

    if end > self.byte_blob.len() {
      None
    } else {
      Some(&self.byte_blob[start..end])
    }
  }

  /// Pushes a new vertex to the buffer by serializing the provided attributes.
  pub fn push_vertex(&mut self, attributes: &[Pushable]) -> Result<(), IcsError> {
    if attributes.len() != self.layout.vertex_structure.len() {
      return Err(ICS_ERROR!(
        why: "Vertex Buffer: Attribute count does not match layout",
        fix: "Check vertex buffer layout being used"
      ));
    }

    let vertex_size = self.layout.size() as usize;
    let start = self.byte_blob.len();
    self.byte_blob.resize(start + vertex_size, 0u8);
    let end = start + vertex_size;

    for (i, attribute) in attributes.iter().enumerate() {
      Self::push_attribute(&mut self.byte_blob[start..end], &self.layout, i, attribute)?;
    }

    Ok(())
  }

  /// Internal method to push an individual attribute into the byte buffer.
  fn push_attribute(
    buffer: &mut [u8],
    layout: &Layout,
    index: usize,
    value: &Pushable,
  ) -> Result<(), IcsError> {
    let element = layout.get(index).ok_or(ICS_ERROR!(
      why: "Vertex Buffer: Index out of bounds in layout",
      fix: "Check vertex buffer layout being used"
    ))?;

    let expected_size = element.size();
    let offset = element.offset() as usize;
    let end = offset + expected_size;

    if end > buffer.len() {
      return Err(ICS_ERROR!(
        why: "Vertex Buffer: Overflowing buffer on push",
        fix: "Check buffer size"
      ));
    }

    match element.attribute_type {
      // For 2D vectors
      AttributeType::Vector2D | AttributeType::Texture2D | AttributeType::UnknownF2 => {
        if let Some(vec) = value.to_vector2() {
          buffer[offset..end].copy_from_slice(&Memory::as_bytes(&vec));
        } else {
          return Err(ICS_ERROR!(
            why: "Vertex Buffer: Trying to push a 2-value Pushable but it does not match AttributeType",
            fix: "Ensure the Pushable matches the AttributeType"
          ));
        }
      }

      // For 3D vectors
      AttributeType::Vector3D
      | AttributeType::Normal
      | AttributeType::ColourF3
      | AttributeType::UnknownF3 => {
        if let Some(vec) = value.to_vector3() {
          buffer[offset..end].copy_from_slice(&Memory::as_bytes(&vec));
        } else {
          return Err(ICS_ERROR!(
            why: "Vertex Buffer: Trying to push a 3-value Pushable but it does not match AttributeType",
            fix: "Ensure the Pushable matches the AttributeType"
          ));
        }
      }

      // For 4D vectors
      AttributeType::Vector4D | AttributeType::ColourF4 | AttributeType::UnknownF4 => {
        if let Some(vec) = value.to_vector4() {
          buffer[offset..end].copy_from_slice(&Memory::as_bytes(&vec));
        } else {
          return Err(ICS_ERROR!(
            why: "Vertex Buffer: Trying to push a 4-value Pushable but it does not match AttributeType",
            fix: "Ensure the Pushable matches the AttributeType"
          ));
        }
      }
    }

    Ok(())
  }

  /// Displays a single vertex's data based on the layout.
  fn display_vertex(&self, f: &mut fmt::Formatter<'_>, data: &[u8]) -> fmt::Result {
    let mut offset = 0;
    for element in &self.layout.vertex_structure {
      let size = element.size();
      let element_data = &data[offset..offset + size];
      write!(f, "  {}: ", element.name)?;
      match element.attribute_type {
        AttributeType::Vector2D | AttributeType::Texture2D | AttributeType::UnknownF2 => {
          let array: Vector<2, f32> = Memory::from_bytes(element_data);
          writeln!(f, "{:?}", array)?;
        }
        AttributeType::Vector3D
        | AttributeType::Normal
        | AttributeType::ColourF3
        | AttributeType::UnknownF3 => {
          let array: Vector<3, f32> = Memory::from_bytes(element_data);
          writeln!(f, "{:?}", array)?;
        }
        AttributeType::Vector4D | AttributeType::ColourF4 | AttributeType::UnknownF4 => {
          let array: Vector<4, f32> = Memory::from_bytes(element_data);
          writeln!(f, "{:?}", array)?;
        }
      }
      offset += size;
    }
    Ok(())
  }

  /// Builds an interleaved vertex buffer from the provided `Vertices` and populates `byte_blob`.
  ///
  /// # Arguments
  ///
  /// * `vertices` - A reference to the `Vertices` containing attribute data.
  ///
  /// # Errors
  ///
  /// Returns an `IcsError` if:
  /// - Required attributes are missing.
  /// - Attribute types do not match the layout.
  /// - Vertex counts are inconsistent across attributes.
  /// - Any other serialization issues occur.
  pub fn build_from(&mut self, vertices: &Vertices) -> Result<(), IcsError> {
    // Ensure all required attributes are present and types match
    for element in &self.layout.vertex_structure {
      // Check if the attribute exists
      let attribute = vertices.attributes.get(&element.name).ok_or_else(|| {
        ICS_ERROR!(
          why: format!(
            "VertexBuffer: Missing required attribute '{}'",
            element.name
          ),
          fix: "Ensure the attribute is added to Vertices before building the VertexBuffer"
        )
      })?;

      // Check if the attribute type matches the layout's specification
      if &attribute.attribute_type != &element.attribute_type {
        return Err(ICS_ERROR!(
          why: format!(
            "VertexBuffer: Attribute '{}' type mismatch. Expected '{:?}', found '{:?}'",
            element.name, element.attribute_type, attribute.attribute_type
          ),
          fix: "Ensure attribute types in Vertices match the Layout's specifications"
        ));
      }
    }

    // Ensure all attributes have the same vertex count
    let vertex_count = if self.layout.size() > 0 {
      vertices.vertex_count()
    } else {
      0
    };

    for element in &self.layout.vertex_structure {
      let attribute = vertices.attributes.get(&element.name).unwrap();
      if attribute.vertex_count() != vertex_count {
        return Err(ICS_ERROR!(
          why: format!(
            "VertexBuffer: Attribute '{}' has vertex count {}, expected {}",
            element.name,
            attribute.vertex_count(),
            vertex_count
          ),
          fix: "Ensure all attributes have the same number of vertices"
        ));
      }
    }

    // Clear existing data and reserve space
    self.byte_blob.clear();
    self
      .byte_blob
      .reserve(vertex_count * self.layout.size() as usize);

    // Iterate over each vertex and interleave attribute data
    for i in 0..vertex_count {
      let mut pushables = Vec::with_capacity(self.layout.vertex_structure.len());

      for element in &self.layout.vertex_structure {
        let attribute = vertices.attributes.get(&element.name).unwrap();
        let data = attribute.vertex_bytes(i).ok_or_else(|| {
          ICS_ERROR!(
            why: format!(
              "VertexBuffer: Failed to retrieve data for attribute '{}' at vertex {}",
              element.name, i
            ),
            fix: "Ensure all vertex data is correctly populated"
          )
        })?;

        // Deserialize based on attribute type
        let pushable = match element.attribute_type {
          AttributeType::Vector2D | AttributeType::Texture2D | AttributeType::UnknownF2 => {
            let vec: Vector<2, f32> = Memory::from_bytes(&data);
            Pushable::Vector2(vec.data)
          }
          AttributeType::Vector3D
          | AttributeType::Normal
          | AttributeType::ColourF3
          | AttributeType::UnknownF3 => {
            let vec: Vector<3, f32> = Memory::from_bytes(&data);
            Pushable::Vector3(vec.data)
          }
          AttributeType::Vector4D | AttributeType::ColourF4 | AttributeType::UnknownF4 => {
            let vec: Vector<4, f32> = Memory::from_bytes(&data);
            Pushable::Vector4(vec.data)
          }
        };

        pushables.push(pushable);
      }

      // Push the interleaved attributes into the byte_blob
      self.push_vertex(&pushables)?;
    }

    Ok(())
  }

  /// Builds a new VertexBuffer from a subset of HierarchyNodes and their associated IndicesParts.
  ///
  /// Extracts vertices referenced by the provided `HierarchyNode`s through their `IndicesPart`s,
  /// remaps them to a new index space, and constructs a new interleaved `VertexBuffer` containing only the subset.
  ///
  /// Additionally, it returns a subset of `IndicesPart`s with remapped indices corresponding to the new `VertexBuffer`.
  ///
  /// # Arguments
  ///
  /// * `vertices` - A reference to the `Vertices` containing all attribute data.
  /// * `indices` - A reference to the `Indices` containing all `IndicesPart`s.
  /// * `hierarchy_nodes` - A slice of `HierarchyNode`s whose associated IndiceParts build the subset.
  ///
  /// # Errors
  ///
  /// Returns an `IcsError` if:
  /// - Any `HierarchyNode` references an invalid `IndicesPart` index.
  /// - Required attributes are missing or have type mismatches.
  /// - Vertex counts are inconsistent across attributes.
  /// - Any other serialization issues occur.
  pub fn build_subset_from(
    &self,
    vertices: &Vertices,
    indices: &Indices,
    hierarchy_nodes: &[HierarchyNode],
  ) -> Result<(VertexBuffer, HashMap<NodeId, IndicesPart>), IcsError> {
    // Collect all vertex indices from the provided HierarchyNodes via their IndicesParts
    let mut collected_indices = Vec::new();

    for node in hierarchy_nodes {
      if let Some(indicespart_index) = node.indicespart_index {
        if indicespart_index >= indices.parts().len() {
          return Err(ICS_ERROR!(
            why: format!(
              "VertexBuffer: HierarchyNode '{}' references invalid IndicesPart index {}",
              node.name, indicespart_index
            ),
            fix: "Ensure all HierarchyNodes reference valid IndicesPart indices"
          ));
        }
        let indices_part = &indices.parts()[indicespart_index];
        collected_indices.extend_from_slice(&indices_part.data());
      }
    }

    // Remove duplicate indices and preserve order
    let unique_indices: Vec<u32> = collected_indices
      .into_iter()
      .fold((Vec::new(), HashSet::new()), |(mut vec, mut set), x| {
        if set.insert(x) {
          vec.push(x);
        }
        (vec, set)
      })
      .0;

    // Create a mapping from old indices to new indices
    let mut index_mapping: HashMap<u32, u32> = HashMap::new();
    for (new_idx, &old_idx) in unique_indices.iter().enumerate() {
      index_mapping.insert(old_idx, new_idx as u32);
    }

    // Extract and interleave the attribute data based on the layout
    let mut new_vertex_buffer = VertexBuffer::new(self.layout.clone());
    new_vertex_buffer
      .byte_blob
      .reserve(unique_indices.len() as usize * self.layout.size() as usize);

    for &old_vertex_idx in &unique_indices {
      let mut pushables = Vec::with_capacity(self.layout.vertex_structure.len());

      for element in &self.layout.vertex_structure {
        let attribute = vertices.attributes.get(&element.name).ok_or_else(|| {
          ICS_ERROR!(
            why: format!(
              "VertexBuffer: Missing required attribute '{}'",
              element.name
            ),
            fix: "Ensure the attribute is added to Vertices before building the VertexBuffer"
          )
        })?;

        let data = attribute
          .vertex_bytes(old_vertex_idx as usize)
          .ok_or_else(|| {
            ICS_ERROR!(
              why: format!(
                "VertexBuffer: Failed to retrieve data for attribute '{}' at vertex index {}",
                element.name, old_vertex_idx
              ),
              fix: "Ensure all vertex data is correctly populated"
            )
          })?;

        // Deserialize based on attribute type
        let pushable = match element.attribute_type {
          AttributeType::Vector2D | AttributeType::Texture2D | AttributeType::UnknownF2 => {
            let vec: Vector<2, f32> = Memory::from_bytes(&data);
            Pushable::Vector2(vec.data)
          }
          AttributeType::Vector3D
          | AttributeType::Normal
          | AttributeType::ColourF3
          | AttributeType::UnknownF3 => {
            let vec: Vector<3, f32> = Memory::from_bytes(&data);
            Pushable::Vector3(vec.data)
          }
          AttributeType::Vector4D | AttributeType::ColourF4 | AttributeType::UnknownF4 => {
            let vec: Vector<4, f32> = Memory::from_bytes(&data);
            Pushable::Vector4(vec.data)
          }
        };

        pushables.push(pushable);
      }

      // Push the interleaved attributes into the new byte_blob
      new_vertex_buffer.push_vertex(&pushables)?;
    }

    // Build the subset of IndicesParts with remapped indices
    let mut new_indices_parts: HashMap<NodeId, IndicesPart> = HashMap::new();

    for node in hierarchy_nodes {
      if let Some(indicespart_index) = node.indicespart_index {
        let original_indices_part = &indices.parts()[indicespart_index];
        let primitive = original_indices_part.primitive();

        let mut remapped_indices_part = IndicesPart::new(primitive);
        for &old_idx in original_indices_part.data() {
          if let Some(&new_idx) = index_mapping.get(&old_idx) {
            remapped_indices_part.push(&[new_idx]);
          } else {
            return Err(ICS_ERROR!(
              why: format!(
                "VertexBuffer: Index {} in IndicesPart {} not found in unique_indices",
                old_idx, indicespart_index
              ),
              fix: "Ensure all indices in IndicesParts are present in the HierarchyNodes' IndicesParts"
            ));
          }
        }

        new_indices_parts.insert(node.name().clone(), remapped_indices_part);
      }
    }

    Ok((new_vertex_buffer, new_indices_parts))
  }
}

impl fmt::Display for VertexBuffer {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    writeln!(f, "VertexBuffer:")?;
    writeln!(f, "{}", self.layout)?;

    let vertex_size = self.layout.size() as usize;
    let num_vertices = self.number_vertices();

    writeln!(f, "Number of vertices: {}", num_vertices)?;
    for i in 0..num_vertices {
      let start = i * vertex_size;
      let end = start + vertex_size;
      let vertex_data = &self.byte_blob[start..end];
      writeln!(f, "Vertex {}:", i)?;
      self.display_vertex(f, vertex_data)?;
    }
    Ok(())
  }
}
