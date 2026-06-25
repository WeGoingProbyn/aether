// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Consumer-supplied appearance for categorical (class) layers.
//!
//! The IR keeps categorical data art-free — `ClassSet` carries codes and
//! labels, no colours. A reference renderer still has to draw *something*, so
//! the consumer hands it a [`CategoricalStyle`]: a class-code → colour map plus
//! a fallback for unknown codes. The paint system uses it to colour a mesh whose
//! bound layer is categorical (e.g. land / ocean / ice surface type).

use std::collections::HashMap;

use bevy::prelude::Resource;

use crate::ir::Rgba;

/// Maps categorical class codes to display colours. Populate it from the
/// consumer side (the look is the consumer's decision); the reference renderer
/// only applies it.
#[derive(Resource, Debug, Clone)]
pub struct CategoricalStyle {
  colours: HashMap<u32, Rgba>,
  /// Colour for class codes with no explicit entry.
  fallback: Rgba,
}

impl Default for CategoricalStyle {
  fn default() -> Self {
    Self {
      colours: HashMap::new(),
      // A neutral grey so an unstyled categorical layer is visible but plainly
      // "unset" rather than invisible or misleadingly coloured.
      fallback: Rgba::new(0.5, 0.5, 0.5, 1.0),
    }
  }
}

impl CategoricalStyle {
  pub fn new() -> Self {
    Self::default()
  }

  /// Set the colour for `class`. Chainable.
  pub fn with_class(mut self, class: u32, colour: Rgba) -> Self {
    self.colours.insert(class, colour);
    self
  }

  /// Set the colour returned for unmapped class codes. Chainable.
  pub fn with_fallback(mut self, colour: Rgba) -> Self {
    self.fallback = colour;
    self
  }

  /// The colour for `class`, or the fallback when none is set.
  pub fn colour_for_class(&self, class: u32) -> Rgba {
    self.colours.get(&class).copied().unwrap_or(self.fallback)
  }

  pub fn is_empty(&self) -> bool {
    self.colours.is_empty()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn returns_mapped_colour_then_falls_back() {
    let style = CategoricalStyle::new()
      .with_class(1, Rgba::GREEN)
      .with_fallback(Rgba::BLACK);
    assert_eq!(style.colour_for_class(1), Rgba::GREEN);
    // Unmapped code → fallback, never a panic.
    assert_eq!(style.colour_for_class(99), Rgba::BLACK);
  }

  #[test]
  fn default_fallback_is_opaque() {
    let style = CategoricalStyle::new();
    assert!(style.is_empty());
    assert_eq!(style.colour_for_class(0).a, 1.0);
  }
}
