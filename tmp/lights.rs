use std::{collections::HashMap, sync::Arc};

use super::{
  entity::Entity,
  materials::MaterialPipelineType,
  uniforms::{Pushable, ShaderStage, Uniform, UniformLayout, UniformType},
};
use crate::ICS_WARN;

#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash)]
pub enum LightType {
  Point,
  Spotlight,
  Directional,
}

#[derive(Clone)]
pub enum LightOwner {
  Scene,
  Entity((Arc<Entity>, String)),
}

#[derive(Clone)]
pub struct Light {
  ty: LightType,
  owner: LightOwner,
  position: [f32; 3],
  color: [f32; 3],
  intensity: f32,
  variables: Vec<(String, Vec<Pushable>, UniformType)>,
  pipeline_ty: MaterialPipelineType,
  arc: Option<Arc<Uniform>>,
}

impl Light {
  pub fn new(
    ty: LightType,
    owner: LightOwner,
    mat_ty: MaterialPipelineType,
    pos: [f32; 3],
    instances: usize,
  ) -> Light {
    let mut tmp = Light {
      ty,
      owner,
      position: pos,
      color: [1.0, 1.0, 1.0],
      intensity: 1.0,
      variables: vec![],
      pipeline_ty: mat_ty,
      arc: None,
    };
    tmp.for_pipeline(mat_ty, instances);

    // match tmp.owner {
    //   LightOwner::Scene => {},
    //   LightOwner::Entity((ref entity, ref node_name)) => {
    //     let mut hierarchy = entity.mesh().hierarchy().lock().unwrap();
    //     hierarchy.find_node_mut(&node_name, &mut |node|
    //       tmp.position = node.transform().unwrap().translation()
    //     );
    //   }
    // }
    tmp
  }

  pub fn pipeline_type(&self) -> MaterialPipelineType {
    self.pipeline_ty
  }

  pub fn with_arc(&mut self, arc: &Arc<Uniform>) {
    self.arc = Some(arc.clone());
  }

  pub fn uniform_arc(&self) -> &Option<Arc<Uniform>> {
    &self.arc
  }

  pub fn light_type(&self) -> &LightType {
    &self.ty
  }

  pub fn light_owner(&self) -> &LightOwner {
    &self.owner
  }

  pub fn variables(&self) -> &Vec<(String, Vec<Pushable>, UniformType)> {
    &self.variables
  }

  pub fn set_color_intensity(&mut self, color: [f32; 3], intensity: f32) {
    self.color = color;
    self.intensity = intensity;
    let instances = self
      .variables
      .first()
      .map(|(_, values, _)| values.len())
      .unwrap_or(1);
    self.for_pipeline(self.pipeline_ty, instances);
  }

  pub fn for_pipeline(&mut self, ty: MaterialPipelineType, instances: usize) {
    self.variables.clear();
    match ty {
      MaterialPipelineType::Phong => {
        let pos = [self.position[0], self.position[1], self.position[2]];
        self.variables.push((
          String::from("position"),
          vec![Pushable::Vec3(pos); instances],
          UniformType::Vec3,
        ));
        self.variables.push((
          String::from("diffuse"),
          vec![Pushable::Vec3([0.5, 0.5, 0.5]); instances],
          UniformType::Vec3,
        ));
        self.variables.push((
          String::from("ambient"),
          vec![Pushable::Vec3([0.5, 0.5, 0.5]); instances],
          UniformType::Vec3,
        ));
        self.variables.push((
          String::from("specular"),
          vec![Pushable::Vec3([0.5, 0.5, 0.5]); instances],
          UniformType::Vec3,
        ));
      }
      MaterialPipelineType::Pbr => {
        let pos = [self.position[0], self.position[1], self.position[2]];
        self.variables.push((
          String::from("position"),
          vec![Pushable::Vec3(pos); instances],
          UniformType::Vec3,
        ));
        let radiance = [
          self.color[0] * self.intensity,
          self.color[1] * self.intensity,
          self.color[2] * self.intensity,
        ];
        self.variables.push((
          String::from("diffuse"),
          vec![Pushable::Vec3(radiance); instances],
          UniformType::Vec3,
        ));
      }
      _ => {
        ICS_WARN!("Light: Trying to request light materials for a material pipeline which doesn't require lighting");
      }
    }
  }

  // pub fn elements(&self) -> &Vec<(MaterialVariable, Pushable, UniformType)> {
  //   &self.variables
  // }

  pub fn to_uniform_layout(&self) -> UniformLayout {
    let tys: Vec<UniformType> = self.variables.iter().map(|(_, _, ty)| *ty).collect();
    UniformLayout::new(&tys, ShaderStage::Fragment)
  }

  // pub fn from_entity_node(id: usize, ty: LightType, entity: Arc<Entity>, node_name: &str) -> Light {
  //   let true_ty = if ty == LightType::Directional {
  //     ICS_WARN!("Light: Trying to set a directional light to an entity node, using point light instead");
  //     LightType::Point
  //   } else {
  //     ty
  //   };
  //
  //   let mut position = [0.0, 0.0, 0.0];
  //   let owner = LightOwner::Entity((entity, node_name.to_string()));
  //
  //   match owner {
  //     LightOwner::Scene => {
  //       ICS_WARN!("Light: Trying to set a scene light using an entity, this light will likely be invalid");
  //     },
  //     LightOwner::Entity((ref entity, ref node_name)) => {
  //       {
  //         let mut hierarchy = entity.mesh().hierarchy().lock().unwrap();
  //         hierarchy.find_node_mut(node_name, &mut |node| {
  //           position = if let Some(transform) = node.relative_transform {
  //             transform.translation()
  //           } else if let Some(matrix) = node.relative_matrix {
  //             Transformation::translation_from_model_matrix(&matrix)
  //           } else {
  //             ICS_WARN!("Light: Trying to associate a light with an entity which does not have an associated mesh or world transform, placing at origin");
  //             [0.0, 0.0, 0.0]
  //           }
  //         });
  //       }
  //
  //       // {
  //       //   let materials = entity.materials().lock().unwrap();
  //       //   let node_materials = materials.node_materials(node_name);
  //       //
  //       //   let needed_uniforms = vec![]
  //       //   for (mat_ty, node_material) in node_materials {
  //       //     match mat_ty {
  //       //       MaterialPipelineType::Phong => {
  //       //         Light::light_material_from_phong(node_material);
  //       //       },
  //       //     }
  //       //   }
  //       // }
  //     }
  //   }
  //
  //   Light {
  //     id,
  //     ty: true_ty,
  //     owner,
  //     position,
  //     materials: NodeMaterials::default(),
  //   }
  // }

  // fn light_material_from_phong(node_material: &NodeMaterials) {
  //
  // }
}

#[derive(Clone)]
pub struct SceneLights {
  stack: HashMap<LightType, Vec<Light>>,
}

impl SceneLights {
  pub fn new() -> SceneLights {
    SceneLights {
      stack: HashMap::new(),
    }
  }

  pub fn add_light(&mut self, light: Light) -> usize {
    let ty = light.light_type().clone();
    self.stack.entry(ty).or_insert(vec![]).push(light);
    self.stack.get(&ty).unwrap_or(&vec![]).len()
  }

  pub fn lights_of(&self, light_ty: LightType) -> Option<&Vec<Light>> {
    self.stack.get(&light_ty)
  }
}
