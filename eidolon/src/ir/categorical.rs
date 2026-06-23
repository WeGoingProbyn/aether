// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! The semantic vocabulary for a categorical render layer.
//!
//! A categorical layer (surface type, biome, land use, …) carries a class id
//! per element plus this [`ClassSet`] describing what those ids *mean* — their
//! stable code and a human-readable label. It is deliberately **art-free**: no
//! colours, no materials. A consumer maps each class to its own appearance;
//! eidolon only states which class each cell is.

/// One class in a [`ClassSet`]: a stable numeric code and a label.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClassInfo {
  pub code: u32,
  pub label: String,
}

impl ClassInfo {
  pub fn new(code: u32, label: impl Into<String>) -> Self {
    Self {
      code,
      label: label.into(),
    }
  }
}

/// The set of classes a categorical layer can take. The order is not
/// significant; lookups are by `code`.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ClassSet {
  classes: Vec<ClassInfo>,
}

impl ClassSet {
  pub fn new(classes: Vec<ClassInfo>) -> Self {
    Self { classes }
  }

  pub fn classes(&self) -> &[ClassInfo] {
    &self.classes
  }

  pub fn len(&self) -> usize {
    self.classes.len()
  }

  pub fn is_empty(&self) -> bool {
    self.classes.is_empty()
  }

  /// The label for a class code, if present.
  pub fn label_of(&self, code: u32) -> Option<&str> {
    self
      .classes
      .iter()
      .find(|c| c.code == code)
      .map(|c| c.label.as_str())
  }

  /// Whether every code is described by this set.
  pub fn contains(&self, code: u32) -> bool {
    self.classes.iter().any(|c| c.code == code)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::ir::{CategoricalSamples, LayerKind, LayerSamples};

  #[test]
  fn class_set_lookup() {
    let set = ClassSet::new(vec![
      ClassInfo::new(0, "Ocean"),
      ClassInfo::new(1, "Land"),
      ClassInfo::new(2, "Ice"),
    ]);
    assert_eq!(set.len(), 3);
    assert_eq!(set.label_of(1), Some("Land"));
    assert_eq!(set.label_of(7), None);
    assert!(set.contains(2));
    assert!(!set.contains(9));
  }

  #[test]
  fn categorical_samples_match_only_categorical_kind() {
    let samples =
      LayerSamples::Categorical(CategoricalSamples::PerCell(vec![0, 1, 2]));
    let kind = LayerKind::Categorical {
      classes: ClassSet::default(),
    };
    assert!(samples.matches_kind(&kind));
    assert!(!samples.matches_kind(&LayerKind::Mask));
  }
}
