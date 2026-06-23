// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Read-only extraction from simulation state into Eidolon IR.

pub mod coupler_debug;
pub mod diagnostics;
pub mod frame;
pub mod layer;
pub mod mesh;
pub mod producer;
pub mod query_extract;
pub mod snapshot_adapter;

pub use coupler_debug::*;
pub use frame::*;
pub use layer::*;
pub use mesh::*;
pub use producer::{
  CategoricalLayerConfig, ExtractConfig, FrameProducer, MeshConfig,
  ScalarLayerConfig, surface_class_set, surface_type_categorical_layer,
};
pub use query_extract::{
  QuantityChannel, default_atmosphere_quantities,
  default_surface_terrain_quantities, extract_quantity_frame,
};
pub use snapshot_adapter::{frame_to_initial_batch, frame_to_replace_batch};
