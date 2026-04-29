// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use pleroma::core::storage::FieldStorage;
use utility::domain::FieldKey;

use crate::ir::{
  LayerId, LayerSource, RenderMeshId, ScalarLayer, ScalarSamples,
};

pub fn scalar_component_layer<const N: usize, S>(
  id: LayerId,
  label: impl Into<String>,
  target: RenderMeshId,
  field_key: FieldKey,
  storage: &S,
  component: usize,
) -> ScalarLayer
where
  S: FieldStorage<N>,
{
  assert!(
    component < N,
    "component {component} out of range for {N}-component field"
  );

  ScalarLayer::new(
    id,
    label,
    target,
    LayerSource::Field(field_key),
    ScalarSamples::PerCell(storage.component(component).as_ref().to_vec()),
  )
}
