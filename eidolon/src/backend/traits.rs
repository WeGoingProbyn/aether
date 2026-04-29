// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use crate::ir::RenderFrame;

pub trait RenderBackend {
  type Error;

  fn apply_frame(&mut self, frame: &RenderFrame) -> Result<(), Self::Error>;
}
