use std::fmt;
use std::hash::Hash;

/// Represents different types of geometric primitives used in rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Primitive {
  /// A list of points.
  PointList,
  /// A strip of connected lines.
  LineStrip,
  /// A list of independent lines.
  LineList,
  /// A list of independent triangles.
  TriangleList,
}

impl fmt::Display for Primitive {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let primitive_name = match self {
      Primitive::PointList => "PointList",
      Primitive::LineStrip => "LineStrip",
      Primitive::LineList => "LineList",
      Primitive::TriangleList => "TriangleList",
    };
    write!(f, "{}", primitive_name)
  }
}

impl Primitive {
  /// Returns the size in bytes of a single primitive element.
  ///
  /// # Examples
  ///
  /// ```
  /// let primitive = Primitive::TriangleList;
  /// assert_eq!(primitive.size_of_primitive(), 12); // 3 * 4 bytes for [u32; 3]
  /// ```
  pub fn size_of_primitive(&self) -> u32 {
    match self {
      Primitive::PointList => std::mem::size_of::<u32>() as u32,
      Primitive::LineStrip => std::mem::size_of::<u32>() as u32,
      Primitive::LineList => std::mem::size_of::<[u32; 2]>() as u32,
      Primitive::TriangleList => std::mem::size_of::<[u32; 3]>() as u32,
    }
  }

  /// Returns the number of indices per primitive element.
  ///
  /// # Examples
  ///
  /// ```
  /// let primitive = Primitive::LineList;
  /// assert_eq!(primitive.number_indices(), 2);
  /// ```
  pub fn number_indices(&self) -> u32 {
    match self {
      Primitive::PointList => 1,
      Primitive::LineStrip => 1,
      Primitive::LineList => 2,
      Primitive::TriangleList => 3,
    }
  }
}

/// Trait representing index types compatible with different primitives.
pub trait Index {
  /// The size of each index element in bytes.
  const ELEMENT_SIZE: u32;
  /// The primitives that the index type supports.
  const TYPES: &'static [Primitive];
}

impl Index for u32 {
  const ELEMENT_SIZE: u32 = std::mem::size_of::<u32>() as u32;
  const TYPES: &'static [Primitive] = &[Primitive::PointList, Primitive::LineStrip];
}

impl Index for [u32; 2] {
  const ELEMENT_SIZE: u32 = std::mem::size_of::<[u32; 2]>() as u32;
  const TYPES: &'static [Primitive] = &[Primitive::LineList];
}

impl Index for [u32; 3] {
  const ELEMENT_SIZE: u32 = std::mem::size_of::<[u32; 3]>() as u32;
  const TYPES: &'static [Primitive] = &[Primitive::TriangleList];
}

/// Represents a part of index data used in mesh rendering.
#[derive(Debug, Clone, Hash)]
pub struct IndicesPart {
  /// The primitive type of the indices.
  primitive: Primitive,
  /// The index data.
  index_data: Vec<u32>,
}

impl IndicesPart {
  /// Creates a new `IndicesPart` with the specified primitive type.
  ///
  /// # Arguments
  ///
  /// * `primitive` - The type of geometric primitive.
  ///
  /// # Examples
  ///
  /// ```
  /// let part = IndicesPart::new(Primitive::TriangleList);
  /// ```
  pub fn new(primitive: Primitive) -> Self {
    IndicesPart {
      primitive,
      index_data: Vec::new(),
    }
  }

  /// Adds index data to this `IndicesPart`.
  ///
  /// # Arguments
  ///
  /// * `indices` - A slice of index data to add.
  ///
  /// # Examples
  ///
  /// ```
  /// let mut part = IndicesPart::new(Primitive::TriangleList);
  /// part.push(&[0, 1, 2]);
  /// ```
  pub fn push(&mut self, indices: &[u32]) {
    self.index_data.extend_from_slice(indices);
  }

  /// Returns a reference to the index data.
  pub fn data(&self) -> &[u32] {
    &self.index_data
  }

  /// Returns the primitive type of this `IndicesPart`.
  pub fn primitive(&self) -> Primitive {
    self.primitive
  }
}

/// Represents a collection of `IndicesPart` used to build a mesh.
#[derive(Debug, Clone, Hash)]
pub struct Indices {
  /// A collection of index parts.
  parts: Vec<IndicesPart>,
}

impl Indices {
  /// Creates a new empty `Indices` collection.
  ///
  /// # Examples
  ///
  /// ```
  /// let indices = Indices::new();
  /// ```
  pub fn new() -> Self {
    Indices { parts: Vec::new() }
  }

  /// Adds an `IndicesPart` to the collection.
  ///
  /// # Arguments
  ///
  /// * `part` - The `IndicesPart` to be added.
  ///
  /// # Returns
  ///
  /// The index of the added part.
  pub fn add_part(&mut self, part: IndicesPart) -> usize {
    self.parts.push(part);
    self.parts.len() - 1
  }

  /// Returns a reference to the index parts.
  pub fn parts(&self) -> &Vec<IndicesPart> {
    &self.parts
  }

  /// Returns the combined data of all index parts.
  pub fn data(&self) -> Vec<u32> {
    self
      .parts
      .iter()
      .flat_map(|part| part.data().to_vec())
      .collect()
  }

  /// Returns the total number of indices across all parts.
  pub fn total_indices(&self) -> usize {
    self.parts.iter().map(|part| part.data().len()).sum()
  }
}

impl fmt::Display for Indices {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    writeln!(f, "Indices Collection:")?;
    for (i, part) in self.parts.iter().enumerate() {
      writeln!(f, "Part {}:", i)?;
      writeln!(f, "  Primitive: {}", part.primitive())?;
      writeln!(f, "  Data Length: {}", part.data().len())?;
    }
    Ok(())
  }
}
