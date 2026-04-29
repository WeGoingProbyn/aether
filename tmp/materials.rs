use std::{
  collections::{HashMap, HashSet},
  hash::Hash,
  sync::Arc,
};

use crate::{
  debugger::IcsError,
  structures::textures::{Texture, TextureType},
  ICS_DEBUG, ICS_ERROR, ICS_WARN,
};

use super::{
  entity::Entity,
  hierarchy::NodeId,
  lights::Light,
  mesh::Mesh,
  pipeline::PipelineAttribute,
  textures::Textures,
  uniforms::{ConcreteUniform, CustomUniform, Pushable, ShaderStage, UniformType},
};

/// Represents the type of material pipeline to be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaterialPipelineType {
  /// Phong shading model.
  Phong,
  /// Simple color-based shading.
  Colour,
  /// Skybox shading model.
  Skybox,
  /// Physically-based rendering (PBR) shading model.
  Pbr,
}

#[derive(Clone, Debug)]
pub struct MaterialFallbacks {
  pub colour: [f32; 4],
  pub diffuse: [f32; 3],
  pub specular: [f32; 3],
  pub ambient: [f32; 3],
  pub emission: [f32; 3],
  pub shininess: f32,
  pub metallic: f32,
  pub roughness: f32,
  pub ambient_occlusion: f32,
}

impl Default for MaterialFallbacks {
  fn default() -> Self {
    MaterialFallbacks {
      colour: [1.0, 1.0, 1.0, 0.0],
      diffuse: [1.0, 1.0, 1.0],
      specular: [0.75, 0.75, 0.75],
      ambient: [0.5, 0.5, 0.5],
      emission: [0.0, 0.0, 0.0],
      shininess: 32.0,
      metallic: 0.0,
      roughness: 0.5,
      ambient_occlusion: 1.0,
    }
  }
}

impl MaterialFallbacks {
  pub fn as_known_materials(&self) -> HashMap<String, (Pushable, UniformType)> {
    let mut out = HashMap::new();
    out.insert(
      "colour".to_string(),
      (Pushable::Vec4(self.colour), UniformType::Vec4),
    );
    out.insert(
      "diffuse".to_string(),
      (Pushable::Vec3(self.diffuse), UniformType::Vec3),
    );
    out.insert(
      "specular".to_string(),
      (Pushable::Vec3(self.specular), UniformType::Vec3),
    );
    out.insert(
      "ambient".to_string(),
      (Pushable::Vec3(self.ambient), UniformType::Vec3),
    );
    out.insert(
      "emission".to_string(),
      (Pushable::Vec3(self.emission), UniformType::Vec3),
    );
    out.insert(
      "shininess".to_string(),
      (Pushable::Float(self.shininess), UniformType::Float),
    );
    out.insert(
      "metallic".to_string(),
      (Pushable::Float(self.metallic), UniformType::Float),
    );
    out.insert(
      "roughness".to_string(),
      (Pushable::Float(self.roughness), UniformType::Float),
    );
    out.insert(
      "ambient_occlusion".to_string(),
      (Pushable::Float(self.ambient_occlusion), UniformType::Float),
    );
    out
  }

  pub fn merge_known_materials(
    &self,
    known: &HashMap<String, (Pushable, UniformType)>,
  ) -> HashMap<String, (Pushable, UniformType)> {
    let mut merged = self.as_known_materials();
    for (key, value) in known {
      merged.insert(key.clone(), *value);
    }
    merged
  }
}

/// Represents the source of a material variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaterialSource {
  /// The value is derived from a texture.
  Texture,
  /// The value is provided as a uniform.
  Uniform,
  /// The value is an attribute of the mesh.
  Attribute,
}

/// Defines different material types and their sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaterialVariable {
  Normal(MaterialSource),
  Colour(MaterialSource),
  Diffuse(MaterialSource),
  Ambient(MaterialSource),
  Specular(MaterialSource),
  Emission(MaterialSource),
  Shininess(MaterialSource),
  Metallic(MaterialSource),
  Roughness(MaterialSource),
  AmbientOcclusion(MaterialSource),
}

impl MaterialVariable {
  /// Converts the material variable into a string name for use in shaders and
  /// key matching in hashmap storing.
  ///
  /// # Errors
  ///
  /// Returns an `IcsError` if the material source is invalid for the given material variable.
  ///
  /// # Examples
  ///
  /// ```rust
  /// let var = MaterialVariable::Colour(MaterialSource::Texture);
  /// assert_eq!(var.to_name().unwrap(), "colour_map".to_string());
  /// ```
  pub fn to_name(&self) -> Result<String, IcsError> {
    match self {
      MaterialVariable::Colour(src) => match src {
        MaterialSource::Texture => Ok("colour_map".to_string()),
        MaterialSource::Uniform => Ok("colour".to_string()),
        MaterialSource::Attribute => Ok("in_colour".to_string()),
      },
      MaterialVariable::Diffuse(src) => match src {
        MaterialSource::Texture => Ok("diffuse_map".to_string()),
        MaterialSource::Uniform => Ok("diffuse".to_string()),
        MaterialSource::Attribute => Err(
          ICS_ERROR!(why: "Materials: Diffuse cannot be an Attribute.", fix: "Use a Uniform or Texture instead."),
        ),
      },
      MaterialVariable::Ambient(src) => match src {
        MaterialSource::Uniform => Ok("ambient".to_string()),
        MaterialSource::Texture => Ok("ambient_map".to_string()),
        MaterialSource::Attribute => Err(
          ICS_ERROR!(why: "Materials: Ambient cannot be an Attribute.", fix: "Use a Uniform or Texture instead."),
        ),
      },
      MaterialVariable::Specular(src) => match src {
        MaterialSource::Texture => Ok("specular_map".to_string()),
        MaterialSource::Uniform => Ok("specular".to_string()),
        MaterialSource::Attribute => Err(
          ICS_ERROR!(why: "Materials: Specular cannot be an Attribute.", fix: "Use a Uniform or Texture instead."),
        ),
      },
      MaterialVariable::Shininess(src) => match src {
        MaterialSource::Texture => Err(
          ICS_ERROR!(why: "Materials: Shininess cannot be a Texture.", fix: "Use a Uniform instead."),
        ),
        MaterialSource::Uniform => Ok("shininess".to_string()),
        MaterialSource::Attribute => Err(
          ICS_ERROR!(why: "Materials: Shininess cannot be an Attribute.", fix: "Use a Uniform instead."),
        ),
      },
      MaterialVariable::Normal(src) => match src {
        MaterialSource::Texture => Ok("normal_map".to_string()),
        MaterialSource::Uniform => Err(
          ICS_ERROR!(why: "Materials: Normal cannot be a Uniform.", fix: "Use a Texture or Attribute instead."),
        ),
        MaterialSource::Attribute => Ok("in_normal".to_string()),
      },
      MaterialVariable::Metallic(src) => match src {
        MaterialSource::Uniform => Ok("metallic".to_string()),
        MaterialSource::Texture => Ok("metallic_map".to_string()),
        MaterialSource::Attribute => Err(
          ICS_ERROR!(why: "Materials: Metallic cannot be an Attribute.", fix: "Use a Uniform or Texture instead."),
        ),
      },
      MaterialVariable::Roughness(src) => match src {
        MaterialSource::Uniform => Ok("roughness".to_string()),
        MaterialSource::Texture => Ok("roughness_map".to_string()),
        MaterialSource::Attribute => Err(
          ICS_ERROR!(why: "Materials: Roughness cannot be an Attribute.", fix: "Use a Uniform or Texture instead."),
        ),
      },
      MaterialVariable::AmbientOcclusion(src) => match src {
        MaterialSource::Uniform => Ok("ambient_occlusion".to_string()),
        MaterialSource::Texture => Ok("ao_map".to_string()),
        MaterialSource::Attribute => Err(
          ICS_ERROR!(why: "Materials: Ambient Occlusion cannot be an Attribute.", fix: "Use a Uniform or Texture instead."),
        ),
      },
      MaterialVariable::Emission(src) => match src {
        MaterialSource::Uniform => Ok("emission".to_string()),
        MaterialSource::Texture => Ok("emission_map".to_string()),
        MaterialSource::Attribute => Err(
          ICS_ERROR!(why: "Materials: Emission cannot be an Attribute.", fix: "Use a Uniform or Texture instead."),
        ),
      },
    }
  }
}

/// Represents the owner of a material.
pub enum MaterialOwner {
  Light(Light),
  Entity(Arc<Entity>),
}

/// Holds the available materials for a node within a mesh,
/// including: textures, uniforms and their layouts, and the required pipeline attributes
/// needed to satisfy the requested material pipeline type (phong, colour, pbr etc)
#[derive(Debug, Clone)]
pub struct NodeMaterials {
  ty: MaterialPipelineType,
  textures: HashMap<TextureType, Vec<(MaterialVariable, ShaderStage, Arc<Texture>)>>,
  uniforms: HashMap<ConcreteUniform, (ShaderStage, Vec<(MaterialVariable, Pushable, UniformType)>)>,
  all_uniform_layouts: HashMap<ConcreteUniform, CustomUniform>,
  required_attributes: Vec<PipelineAttribute>,
}

impl Hash for NodeMaterials {
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    self.ty.hash(state);
    self.required_attributes.hash(state);
    for texture_ty in self.texture_signature() {
      texture_ty.hash(state);
    }
  }
}

impl PartialEq for NodeMaterials {
  fn eq(&self, other: &Self) -> bool {
    self.ty == other.ty
      && self.required_attributes == other.required_attributes
      && self.texture_signature() == other.texture_signature()
  }

  fn ne(&self, other: &Self) -> bool {
    self.ty != other.ty || self.required_attributes != other.required_attributes
  }
}

impl Eq for NodeMaterials {}

impl NodeMaterials {
  fn texture_signature(&self) -> Vec<TextureType> {
    let mut types: Vec<TextureType> = self.textures.keys().copied().collect();
    types.sort_by_key(|ty| match ty {
      TextureType::Specular => 0,
      TextureType::Normal => 1,
      TextureType::Albedo => 2,
      TextureType::Height => 3,
      TextureType::Emission => 4,
      TextureType::AmbientOcclusion => 5,
      TextureType::Metallic => 6,
      TextureType::Roughness => 7,
      TextureType::UnknownTexture => 8,
    });
    types
  }
  /// Creates a new `NodeMaterials` instance based on the given material pipeline type.
  ///
  /// # Arguments
  /// * `pipeline_type` - The type of material pipeline (e.g., Phong, Colour, PBR).
  /// * `mesh_textures` - A map of texture types to associated textures.
  /// * `known_materials` - A map of known materials with their pushable values and uniform types.
  ///
  /// # Returns
  /// A `NodeMaterials` instance with resolved textures, uniforms, and required attributes.
  ///
  /// # Examples
  /// ```rust
  /// let materials = NodeMaterials::new(
  ///     MaterialPipelineType::PBR,
  ///     &mesh_textures,
  ///     &known_materials
  /// );
  /// ```
  ///
  /// # Possible Errors
  /// * If a required texture is missing, a debug warning is issued.
  pub fn new(
    pipeline_type: MaterialPipelineType,
    mesh_textures: &HashMap<TextureType, Vec<(String, Arc<Texture>)>>,
    known_materials: &HashMap<String, (Pushable, UniformType)>,
    available_attributes: &HashSet<PipelineAttribute>,
  ) -> Self {
    let mut requires = vec![];
    let mut textures = HashMap::new();
    let mut uniforms = HashMap::new();

    match pipeline_type {
      MaterialPipelineType::Phong => {
        NodeMaterials::resolve_phong(
          known_materials,
          mesh_textures,
          available_attributes,
          &mut requires,
          &mut textures,
          &mut uniforms,
        );
      }
      MaterialPipelineType::Colour => {
        NodeMaterials::resolve_colour(
          known_materials,
          mesh_textures,
          available_attributes,
          &mut requires,
          &mut textures,
          &mut uniforms,
        );
      }
      MaterialPipelineType::Skybox => {
        NodeMaterials::resolve_colour(
          known_materials,
          mesh_textures,
          available_attributes,
          &mut requires,
          &mut textures,
          &mut uniforms,
        );
      }
      MaterialPipelineType::Pbr => {
        NodeMaterials::resolve_pbr(
          known_materials,
          mesh_textures,
          available_attributes,
          &mut requires,
          &mut textures,
          &mut uniforms,
        );
      }
    }

    // Ensure required attributes are sorted for consistency.
    requires.sort();

    NodeMaterials {
      ty: pipeline_type,
      textures,
      uniforms,
      required_attributes: requires,
      all_uniform_layouts: HashMap::new(),
    }
  }

  /// Resolves the Colour material properties when MaerialPipelineType::Colour is requested.
  ///
  /// # Arguments
  /// * `mesh_textures` - Reference to the texture map.
  /// * `known_materials` - Reference to the known materials.
  /// * `textures` - Mutable reference to the textures map.
  /// * `uniforms` - Mutable reference to the uniforms map.
  /// * `requires` - Mutable reference to the list of required attributes.
  /// * `available_attributes` - Reference to the available attributes set associated with this mesh node.
  fn resolve_colour(
    known_materials: &HashMap<String, (Pushable, UniformType)>,
    mesh_textures: &HashMap<TextureType, Vec<(String, Arc<Texture>)>>,
    available_attributes: &HashSet<PipelineAttribute>,
    requires: &mut Vec<PipelineAttribute>,
    textures: &mut HashMap<TextureType, Vec<(MaterialVariable, ShaderStage, Arc<Texture>)>>,
    uniforms: &mut HashMap<
      ConcreteUniform,
      (ShaderStage, Vec<(MaterialVariable, Pushable, UniformType)>),
    >,
  ) {
    let mut needed_uniform = vec![];

    requires.push(PipelineAttribute::Position);

    if let Some(albedo_textures) = mesh_textures.get(&TextureType::Albedo) {
      if available_attributes.contains(&PipelineAttribute::Texture) {
        let col_texs: Vec<(MaterialVariable, ShaderStage, Arc<Texture>)> = albedo_textures
          .iter()
          .map(|(_, arc)| {
            (
              MaterialVariable::Colour(MaterialSource::Texture),
              ShaderStage::Fragment,
              arc.clone(),
            )
          })
          .collect();
        textures.insert(TextureType::Albedo, col_texs);
        requires.push(PipelineAttribute::Texture);
      } else {
        ICS_WARN!("Materials: Texture is present but texture coordinates aren't an available attribute, using default colour uniform");
        needed_uniform.push((
          MaterialVariable::Colour(MaterialSource::Uniform),
          Pushable::Vec4([1.0, 1.0, 1.0, 0.0]),
          UniformType::Vec4,
        ));
      }
    } else if available_attributes.contains(&PipelineAttribute::Colour) {
      requires.push(PipelineAttribute::Colour);
    } else {
      if let Some((data, ty)) = known_materials.get("colour") {
        needed_uniform.push((
          MaterialVariable::Colour(MaterialSource::Uniform),
          *data,
          *ty,
        ));
      } else {
        ICS_DEBUG!(
          "Materials: Requesting Colour for a mesh which doesnt contain an albedo texture or colour value, using a default"
        );
        needed_uniform.push((
          MaterialVariable::Colour(MaterialSource::Uniform),
          Pushable::Vec4([1.0, 1.0, 1.0, 0.0]),
          UniformType::Vec4,
        ));
      }
    }

    if needed_uniform.len() > 0 {
      uniforms.insert(
        ConcreteUniform::Material,
        (ShaderStage::Fragment, needed_uniform),
      );
    }
  }

  /// Resolves the Phong material properties when MaterialPipelineType::Phong is requested.
  ///
  /// # Arguments
  /// * `mesh_textures` - Reference to the texture map.
  /// * `known_materials` - Reference to the known materials.
  /// * `textures` - Mutable reference to the textures map.
  /// * `uniforms` - Mutable reference to the uniforms map.
  /// * `requires` - Mutable reference to the list of required attributes.
  /// * `available_attributes` - Reference to the available attributes set associated with this mesh node.
  fn resolve_phong(
    known_materials: &HashMap<String, (Pushable, UniformType)>,
    mesh_textures: &HashMap<TextureType, Vec<(String, Arc<Texture>)>>,
    available_attributes: &HashSet<PipelineAttribute>,
    requires: &mut Vec<PipelineAttribute>,
    textures: &mut HashMap<TextureType, Vec<(MaterialVariable, ShaderStage, Arc<Texture>)>>,
    uniforms: &mut HashMap<
      ConcreteUniform,
      (ShaderStage, Vec<(MaterialVariable, Pushable, UniformType)>),
    >,
  ) {
    // For Phong shading, we need Diffuse, Specular, and Normal maps.
    // Attempt to get these textures from the mesh textures.

    let mut needs_texcoord = false;
    let mut needed_uniform = vec![];

    requires.push(PipelineAttribute::Position);
    requires.push(PipelineAttribute::Normal);

    if let Some(diffuse_textures) = mesh_textures.get(&TextureType::Albedo) {
      // Provide both diffuse (for lighting coefficients) and colour (base albedo) semantics
      // so pipelines like WSPhong can resolve `colour` even when textures are present.
      let diff_texs: Vec<(MaterialVariable, ShaderStage, Arc<Texture>)> = diffuse_textures
        .iter()
        .flat_map(|(_, arc)| {
          vec![(
            MaterialVariable::Colour(MaterialSource::Texture),
            ShaderStage::Fragment,
            arc.clone(),
          )]
        })
        .collect();
      textures.insert(TextureType::Albedo, diff_texs);
      needs_texcoord = true;
    } else {
      if let Some((data, ty)) = known_materials.get("diffuse") {
        let (push, ty) = match (data, ty) {
          (Pushable::Vec3(v), _) => (Pushable::Vec3(*v), UniformType::Vec3),
          (Pushable::Vec4(v), _) => (Pushable::Vec3([v[0], v[1], v[2]]), UniformType::Vec3),
          _ => (Pushable::Vec3([0.5, 0.5, 0.5]), UniformType::Vec3),
        };
        needed_uniform.push((MaterialVariable::Diffuse(MaterialSource::Uniform), push, ty));
      } else {
        ICS_DEBUG!(
          "Materials: Requesting Phong for a mesh which doesnt contain an albedo texture or diffuse value, using a default"
        );
        needed_uniform.push((
          MaterialVariable::Diffuse(MaterialSource::Uniform),
          Pushable::Vec3([1.0, 1.0, 1.0]),
          UniformType::Vec3,
        ));
      }
    }

    if let Some(specular_textures) = mesh_textures.get(&TextureType::Specular) {
      let spec_texs: Vec<(MaterialVariable, ShaderStage, Arc<Texture>)> = specular_textures
        .iter()
        .map(|(_, arc)| {
          (
            MaterialVariable::Specular(MaterialSource::Texture),
            ShaderStage::Fragment,
            arc.clone(),
          )
        })
        .collect();
      textures.insert(TextureType::Specular, spec_texs);
      needs_texcoord = true;
    } else {
      if let Some((data, ty)) = known_materials.get("specular") {
        let (push, ty) = match (data, ty) {
          (Pushable::Vec3(v), _) => (Pushable::Vec3(*v), UniformType::Vec3),
          (Pushable::Vec4(v), _) => (Pushable::Vec3([v[0], v[1], v[2]]), UniformType::Vec3),
          _ => (Pushable::Vec3([0.5, 0.5, 0.5]), UniformType::Vec3),
        };
        needed_uniform.push((
          MaterialVariable::Specular(MaterialSource::Uniform),
          push,
          ty,
        ));
      } else {
        ICS_DEBUG!(
          "Materials: Requesting Phong for a mesh which doesnt contain a specular texture or value, using a default"
        );
        needed_uniform.push((
          MaterialVariable::Specular(MaterialSource::Uniform),
          Pushable::Vec3([0.75, 0.75, 0.75]),
          UniformType::Vec3,
        ));
      }
    }

    if let Some(normal_textures) = mesh_textures.get(&TextureType::Normal) {
      let norm_texs: Vec<(MaterialVariable, ShaderStage, Arc<Texture>)> = normal_textures
        .iter()
        .map(|(_, arc)| {
          (
            MaterialVariable::Normal(MaterialSource::Texture),
            ShaderStage::Fragment,
            arc.clone(),
          )
        })
        .collect();
      textures.insert(TextureType::Normal, norm_texs);
      needs_texcoord = true;

      if available_attributes.contains(&PipelineAttribute::Tangent) {
        requires.push(PipelineAttribute::Tangent);
      }
    }

    if let Some((data, ty)) = known_materials.get("ambient") {
      let (push, ty) = match (data, ty) {
        (Pushable::Vec3(v), _) => (Pushable::Vec3(*v), UniformType::Vec3),
        (Pushable::Vec4(v), _) => (Pushable::Vec3([v[0], v[1], v[2]]), UniformType::Vec3),
        _ => (Pushable::Vec3([0.5, 0.5, 0.5]), UniformType::Vec3),
      };
      needed_uniform.push((MaterialVariable::Ambient(MaterialSource::Uniform), push, ty));
    } else {
      ICS_DEBUG!(
        "Materials: Requesting Phong for a mesh which doesnt contain an ambient texture or value, using a default",
      );
      needed_uniform.push((
        MaterialVariable::Ambient(MaterialSource::Uniform),
        Pushable::Vec3([0.5, 0.5, 0.5]),
        UniformType::Vec3,
      ));
    }

    if let Some((data, ty)) = known_materials.get("shininess") {
      needed_uniform.push((
        MaterialVariable::Shininess(MaterialSource::Uniform),
        *data,
        *ty,
      ));
    } else {
      ICS_DEBUG!(
        "Materials: Requesting Phong for a mesh which doesnt contain a shininess value, using a default",
      );
      needed_uniform.push((
        MaterialVariable::Shininess(MaterialSource::Uniform),
        Pushable::Float(32.0),
        UniformType::Float,
      ));
    }

    if let Some((data, ty)) = known_materials.get("metallic") {
      needed_uniform.push((
        MaterialVariable::Metallic(MaterialSource::Uniform),
        *data,
        *ty,
      ));
    } else {
      ICS_DEBUG!(
        "Materials: Requesting Phong for a mesh which doesnt contain a metallic value, using a default",
      );
      needed_uniform.push((
        MaterialVariable::Shininess(MaterialSource::Uniform),
        Pushable::Float(0.0),
        UniformType::Float,
      ));
    }

    if let Some(emissive_textures) = mesh_textures.get(&TextureType::Emission) {
      let em_texs: Vec<(MaterialVariable, ShaderStage, Arc<Texture>)> = emissive_textures
        .iter()
        .map(|(_, arc)| {
          (
            MaterialVariable::Emission(MaterialSource::Texture),
            ShaderStage::Fragment,
            arc.clone(),
          )
        })
        .collect();
      textures.insert(TextureType::Emission, em_texs);
      needs_texcoord = true;
    }

    if needs_texcoord {
      requires.push(PipelineAttribute::Texture);
    } else if available_attributes.contains(&PipelineAttribute::Colour) {
      requires.push(PipelineAttribute::Colour);
    } else {
      if let Some((data, ty)) = known_materials.get("colour") {
        needed_uniform.push((
          MaterialVariable::Colour(MaterialSource::Uniform),
          *data,
          *ty,
        ));
      } else {
        ICS_DEBUG!(
          "Materials: Requesting Phong for a mesh which doesnt contain an albedo texture, a colour uniform or colour attribute, using a default",
        );
        needed_uniform.push((
          MaterialVariable::Colour(MaterialSource::Uniform),
          Pushable::Vec4([1.0, 1.0, 1.0, 0.0]),
          UniformType::Vec4,
        ));
      }
    }

    // Variables not provided by textures are defined in custom uniform.
    if needed_uniform.len() > 0 {
      uniforms.insert(
        ConcreteUniform::Material,
        (ShaderStage::Fragment, needed_uniform),
      );
    }
  }

  /// Resolves the PBR material properties when MaterialPipelineType::Pbr is requested.
  ///
  /// # Arguments
  /// * `mesh_textures` - Reference to the texture map.
  /// * `known_materials` - Reference to the known materials.
  /// * `textures` - Mutable reference to the textures map.
  /// * `uniforms` - Mutable reference to the uniforms map.
  /// * `requires` - Mutable reference to the list of required attributes.
  /// * `available_attributes` - Reference to the available attributes set associated with this mesh node.
  fn resolve_pbr(
    known_materials: &HashMap<String, (Pushable, UniformType)>,
    mesh_textures: &HashMap<TextureType, Vec<(String, Arc<Texture>)>>,
    available_attributes: &HashSet<PipelineAttribute>,
    requires: &mut Vec<PipelineAttribute>,
    textures: &mut HashMap<TextureType, Vec<(MaterialVariable, ShaderStage, Arc<Texture>)>>,
    uniforms: &mut HashMap<
      ConcreteUniform,
      (ShaderStage, Vec<(MaterialVariable, Pushable, UniformType)>),
    >,
  ) {
    let mut needs_texcoord = false;
    let mut needed_uniform = vec![];
    let to_vec4 = |data: Pushable| -> (Pushable, UniformType) {
      match data {
        Pushable::Vec4(_) => (data, UniformType::Vec4),
        Pushable::Vec3(v) => (Pushable::Vec4([v[0], v[1], v[2], 1.0]), UniformType::Vec4),
        Pushable::Float(v) => (Pushable::Vec4([v, v, v, v]), UniformType::Vec4),
        _ => (Pushable::Vec4([1.0, 1.0, 1.0, 1.0]), UniformType::Vec4),
      }
    };

    // Essential attributes for PBR
    requires.push(PipelineAttribute::Position);
    requires.push(PipelineAttribute::Normal);

    // Albedo (Base Color)
    if let Some(albedo_textures) = mesh_textures.get(&TextureType::Albedo) {
      let albedo_texs: Vec<(MaterialVariable, ShaderStage, Arc<Texture>)> = albedo_textures
        .iter()
        .map(|(_, arc)| {
          (
            MaterialVariable::Colour(MaterialSource::Texture),
            ShaderStage::Fragment,
            arc.clone(),
          )
        })
        .collect();
      textures.insert(TextureType::Albedo, albedo_texs);
      needs_texcoord = true;
    } else if available_attributes.contains(&PipelineAttribute::Colour) {
      requires.push(PipelineAttribute::Colour);
    } else {
      // Use uniform or default value for albedo
      if let Some((data, ty)) = known_materials.get("colour") {
        needed_uniform.push((
          MaterialVariable::Colour(MaterialSource::Uniform),
          *data,
          *ty,
        ));
      } else {
        ICS_DEBUG!(
        "Materials: Requesting PBR for a mesh which doesn't contain an albedo texture or colour value, using a default"
      );
        needed_uniform.push((
          MaterialVariable::Colour(MaterialSource::Uniform),
          Pushable::Vec4([1.0, 1.0, 1.0, 0.0]),
          UniformType::Vec4,
        ));
      }
    }

    // Metallic
    if let Some(metallic_textures) = mesh_textures.get(&TextureType::Metallic) {
      let metallic_texs: Vec<(MaterialVariable, ShaderStage, Arc<Texture>)> = metallic_textures
        .iter()
        .map(|(_, arc)| {
          (
            MaterialVariable::Metallic(MaterialSource::Texture),
            ShaderStage::Fragment,
            arc.clone(),
          )
        })
        .collect();
      textures.insert(TextureType::Metallic, metallic_texs);
      needs_texcoord = true;
    } else {
      if let Some((data, _ty)) = known_materials.get("metallic") {
        let (value, uniform_ty) = to_vec4(*data);
        needed_uniform.push((
          MaterialVariable::Metallic(MaterialSource::Uniform),
          value,
          uniform_ty,
        ));
      } else {
        ICS_DEBUG!(
        "Materials: Requesting PBR for a mesh which doesn't contain a metallic texture or value, using a default"
      );
        needed_uniform.push((
          MaterialVariable::Metallic(MaterialSource::Uniform),
          Pushable::Vec4([1.0, 1.0, 1.0, 1.0]),
          UniformType::Vec4,
        ));
      }
    }

    // Roughness
    if let Some(roughness_textures) = mesh_textures.get(&TextureType::Roughness) {
      let roughness_texs: Vec<(MaterialVariable, ShaderStage, Arc<Texture>)> = roughness_textures
        .iter()
        .map(|(_, arc)| {
          (
            MaterialVariable::Roughness(MaterialSource::Texture),
            ShaderStage::Fragment,
            arc.clone(),
          )
        })
        .collect();
      textures.insert(TextureType::Roughness, roughness_texs);
      needs_texcoord = true;
    } else {
      if let Some((data, _ty)) = known_materials.get("roughness") {
        let (value, uniform_ty) = to_vec4(*data);
        needed_uniform.push((
          MaterialVariable::Roughness(MaterialSource::Uniform),
          value,
          uniform_ty,
        ));
      } else {
        ICS_DEBUG!(
        "Materials: Requesting PBR for a mesh which doesn't contain a roughness texture or value, using a default"
      );
        needed_uniform.push((
          MaterialVariable::Roughness(MaterialSource::Uniform),
          Pushable::Vec4([0.5, 0.5, 0.5, 0.5]),
          UniformType::Vec4,
        ));
      }
    }

    // Normal Map (kept for future use; PBR shader doesn't sample it yet).
    if let Some(normal_textures) = mesh_textures.get(&TextureType::Normal) {
      let normal_texs: Vec<(MaterialVariable, ShaderStage, Arc<Texture>)> = normal_textures
        .iter()
        .map(|(_, arc)| {
          (
            MaterialVariable::Normal(MaterialSource::Texture),
            ShaderStage::Fragment,
            arc.clone(),
          )
        })
        .collect();
      textures.insert(TextureType::Normal, normal_texs);
      needs_texcoord = true;
    }

    // Ambient Occlusion
    if let Some(ao_textures) = mesh_textures.get(&TextureType::AmbientOcclusion) {
      let ao_texs: Vec<(MaterialVariable, ShaderStage, Arc<Texture>)> = ao_textures
        .iter()
        .map(|(_, arc)| {
          (
            MaterialVariable::AmbientOcclusion(MaterialSource::Texture),
            ShaderStage::Fragment,
            arc.clone(),
          )
        })
        .collect();
      textures.insert(TextureType::AmbientOcclusion, ao_texs);
      needs_texcoord = true;
    } else {
      if let Some((data, _ty)) = known_materials.get("ambient_occlusion") {
        let (value, uniform_ty) = to_vec4(*data);
        needed_uniform.push((
          MaterialVariable::AmbientOcclusion(MaterialSource::Uniform),
          value,
          uniform_ty,
        ));
      } else {
        ICS_DEBUG!(
        "Materials: Requesting PBR for a mesh which doesn't contain an ambient occlusion texture or value, using a default"
      );
        needed_uniform.push((
          MaterialVariable::AmbientOcclusion(MaterialSource::Uniform),
          Pushable::Vec4([1.0, 1.0, 1.0, 1.0]),
          UniformType::Vec4,
        ));
      }
    }

    // Emission
    if let Some(emissive_textures) = mesh_textures.get(&TextureType::Emission) {
      let emission_texs: Vec<(MaterialVariable, ShaderStage, Arc<Texture>)> = emissive_textures
        .iter()
        .map(|(_, arc)| {
          (
            MaterialVariable::Emission(MaterialSource::Texture),
            ShaderStage::Fragment,
            arc.clone(),
          )
        })
        .collect();
      textures.insert(TextureType::Emission, emission_texs);
      needs_texcoord = true;
    } else {
      if let Some((data, _ty)) = known_materials.get("emission") {
        let (value, uniform_ty) = to_vec4(*data);
        needed_uniform.push((
          MaterialVariable::Emission(MaterialSource::Uniform),
          value,
          uniform_ty,
        ));
      } else {
        needed_uniform.push((
          MaterialVariable::Emission(MaterialSource::Uniform),
          Pushable::Vec4([0.0, 0.0, 0.0, 1.0]),
          UniformType::Vec4,
        ));
      }
    }

    // Texture Coordinates
    if needs_texcoord {
      requires.push(PipelineAttribute::Texture);
    } else if available_attributes.contains(&PipelineAttribute::Colour) {
      requires.push(PipelineAttribute::Colour);
    } else {
      if let Some((data, ty)) = known_materials.get("colour") {
        needed_uniform.push((
          MaterialVariable::Colour(MaterialSource::Uniform),
          *data,
          *ty,
        ));
      } else {
        ICS_DEBUG!(
        "Materials: Requesting PBR for a mesh which doesn't contain texture coordinates or colour attribute, using a default"
      );
        needed_uniform.push((
          MaterialVariable::Colour(MaterialSource::Uniform),
          Pushable::Vec4([1.0, 1.0, 1.0, 0.0]),
          UniformType::Vec4,
        ));
      }
    }

    // Insert collected uniforms into the uniforms hashmap
    if needed_uniform.len() > 0 {
      uniforms.insert(
        ConcreteUniform::Material,
        (ShaderStage::Fragment, needed_uniform),
      );
    }
  }

  /// Get the attributed which are needed to fulfill the requested material pipeline.
  pub fn requires(&self) -> &Vec<PipelineAttribute> {
    &self.required_attributes
  }

  /// Get the uniforms which are needed to fulfill the requested material pipeline.
  pub fn uniforms(
    &self,
  ) -> &HashMap<ConcreteUniform, (ShaderStage, Vec<(MaterialVariable, Pushable, UniformType)>)> {
    &self.uniforms
  }

  /// Get the textures which are needed to fulfill the requested material pipeline.
  pub fn textures(
    &self,
  ) -> &HashMap<TextureType, Vec<(MaterialVariable, ShaderStage, Arc<Texture>)>> {
    &self.textures
  }

  /// Get the type of pipeline which has been used to build this node materials object.
  pub fn material_type(&self) -> MaterialPipelineType {
    self.ty
  }

  /// Add a uniform to this material pipeline layout.
  pub fn include_uniform_object(&mut self, ty: ConcreteUniform, custom: &CustomUniform) {
    self.all_uniform_layouts.insert(ty, custom.clone());
  }

  /// Get the uniform layout associated with this uniform for this material pipeline.
  pub fn uniform_layout_for(&self, ty: &ConcreteUniform) -> Option<&CustomUniform> {
    self.all_uniform_layouts.get(ty)
  }

  /// Get the mutable reference for the layout associated with this uniform and material pipeline.
  pub fn uniform_layout_for_mut(&mut self, ty: &ConcreteUniform) -> Option<&mut CustomUniform> {
    self.all_uniform_layouts.get_mut(ty)
  }

  /// Access all known uniform layouts for this material pipeline.
  pub fn uniform_layouts(&self) -> &HashMap<ConcreteUniform, CustomUniform> {
    &self.all_uniform_layouts
  }
}

impl std::fmt::Display for NodeMaterials {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    writeln!(f, "Textures:")?;
    for (texture_type, textures) in &self.textures {
      writeln!(f, " Texture type: {}", texture_type)?;
      for (texture_name, _, _) in textures {
        writeln!(f, "  name: {:?}", texture_name)?;
      }
    }

    for (uniform_name, tys) in &self.uniforms {
      writeln!(f, "Uniform name: {}", uniform_name)?;
      for (ty_name, _, ty) in &tys.1 {
        writeln!(f, " Name: {:?}, Type: {}", ty_name, ty)?;
      }
    }

    writeln!(f, "Requires attributes: {:?}", self.required_attributes)?;
    Ok(())
  }
}

impl Default for NodeMaterials {
  fn default() -> Self {
    NodeMaterials {
      ty: MaterialPipelineType::Colour,
      textures: HashMap::new(),
      uniforms: HashMap::new(),
      required_attributes: vec![],
      all_uniform_layouts: HashMap::new(),
    }
  }
}

/// Represents a collection of materials associated with a mesh.
///
/// This struct keeps track of material assignments to different nodes
/// in a mesh, organizes materials based on pipeline requirements, and
/// provides methods to retrieve and modify material data.
///
/// # Example
///
/// ```rust
/// let mesh = Arc::new(Mesh::new());
/// let mut materials = Materials::new(mesh.clone());
/// ```
#[derive(Clone, Debug)]
pub struct Materials {
  mesh: Arc<Mesh>,
  nodes: HashMap<NodeId, HashMap<MaterialPipelineType, NodeMaterials>>,
  by_requires: HashMap<Vec<PipelineAttribute>, Vec<(NodeId, MaterialPipelineType)>>,
}

impl Materials {
  /// Creates a new `Materials` instance for the given mesh.
  ///
  /// This method initializes the internal material storage and prepares
  /// an empty mapping for each node that contains an index part associated
  /// with a part of a mesh object.
  ///
  /// # Arguments
  /// * `mesh` - A reference-counted mesh object.
  ///
  /// # Returns
  /// A new instance of `Materials`.
  pub fn new(mesh: Arc<Mesh>) -> Materials {
    let mut map = HashMap::new();
    {
      match mesh.hierarchy().lock() {
        Ok(hierarchy) => {
          for node in hierarchy.iter() {
            if node.indicespart_index.is_some() {
              map.insert(node.name().clone(), HashMap::new());
            }
          }
        }
        Err(err) => {
          ICS_WARN!(
            "Materials: failed to lock mesh hierarchy when initializing materials: {}",
            err
          );
        }
      }
    }
    Materials {
      mesh,
      nodes: map,
      by_requires: HashMap::new(),
    }
  }

  /// Returns an immutable reference to the node material mapping.
  pub fn node_map(&self) -> &HashMap<NodeId, HashMap<MaterialPipelineType, NodeMaterials>> {
    &self.nodes
  }

  /// Assigns a material type to a node based on the given pipeline type and available attributes.
  ///
  /// # Arguments
  /// * `node_name` - The name of the node to assign materials to.
  /// * `pipeline_ty` - The type of material pipeline being assigned.
  /// * `known_materials` - A mapping of known materials.
  /// * `available_attributes` - A set of attributes available for the pipeline.
  ///
  /// # Errors
  /// This method will panic if the node does not exist in the material mapping.
  pub fn material_type_for(
    &mut self,
    node_name: &NodeId,
    pipeline_ty: MaterialPipelineType,
    known_materials: &HashMap<String, (Pushable, UniformType)>,
    available_attributes: &HashSet<PipelineAttribute>,
    fallbacks: &MaterialFallbacks,
  ) {
    let tmp = Textures::new();
    let tmp2 = HashMap::new();
    let textures = self.mesh.textures().as_ref().unwrap_or_else(|| &tmp);
    let textures_map = textures.map().get(node_name).unwrap_or_else(|| &tmp2);
    let merged_materials = fallbacks.merge_known_materials(known_materials);
    let node_materials = NodeMaterials::new(
      pipeline_ty,
      textures_map,
      &merged_materials,
      available_attributes,
    );

    if let Some(node_map) = self.nodes.get_mut(node_name) {
      self
        .by_requires
        .entry(node_materials.requires().clone())
        .or_insert(Vec::new())
        .push((node_name.clone(), pipeline_ty));

      node_map.insert(pipeline_ty, node_materials);
    }
  }

  /// Retrieves an immutable reference to the material map of a given node.
  ///
  /// # Arguments
  /// * `node_name` - The name of the node whose materials are retrieved.
  ///
  /// # Panics
  /// Panics if the node does not exist.
  pub fn node_materials(
    &self,
    node_name: &NodeId,
  ) -> Option<&HashMap<MaterialPipelineType, NodeMaterials>> {
    self.nodes.get(node_name)
  }

  /// Retrieves a mutable reference to the material map of a given node.
  ///
  /// # Arguments
  /// * `node_name` - The name of the node whose materials are retrieved.
  ///
  /// # Panics
  /// Panics if the node does not exist.
  pub fn node_materials_mut(
    &mut self,
    node_name: &NodeId,
  ) -> Option<&mut HashMap<MaterialPipelineType, NodeMaterials>> {
    self.nodes.get_mut(node_name)
  }

  /// Returns a mapping of material requirements to their corresponding node materials.
  ///
  /// This function iterates through the `by_requires` mapping and retrieves the
  /// corresponding `NodeMaterials` for each node.
  ///
  /// # Returns
  /// A `HashMap` where keys are lists of required pipeline attributes, and values are
  /// lists of node names paired with their `NodeMaterials` instances.
  pub fn by_requires(&self) -> HashMap<Vec<PipelineAttribute>, Vec<(NodeId, NodeMaterials)>> {
    let mut result = HashMap::new();

    for (requires, node_pipelines) in &self.by_requires {
      let mut materials_list = Vec::new();

      for (node_name, pipeline_ty) in node_pipelines {
        if let Some(node_map) = self.nodes.get(node_name) {
          if let Some(node_material) = node_map.get(pipeline_ty) {
            materials_list.push((node_name.clone(), node_material.clone()));
          }
        }
      }

      result.insert(requires.clone(), materials_list);
    }

    result
  }
}

impl std::fmt::Display for Materials {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    for (node_name, mat_map) in &self.nodes {
      writeln!(f, "Node name: {}", node_name)?;
      for (mat_type, node_mat) in mat_map {
        writeln!(f, "Type: {:?}", mat_type)?;
        writeln!(f, "{}", node_mat)?;
      }
    }
    Ok(())
  }
}
