// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

/// Two cube-spheres sharing the same 6 panels and angular dims, where the
/// upper mesh's bottom radial layer touches the lower mesh's top radial
/// layer. Pairings are pure index arithmetic — no interpolation.
pub struct RadialStackCoupler {
  panel_count: usize,
  angular_dims: [usize; 2],
  lower_top_layer_idx: usize,
  upper_bottom_layer_idx: usize,
}
