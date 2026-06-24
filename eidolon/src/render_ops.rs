// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Backend-agnostic geometry operations a reference renderer applies to IR
//! meshes. These are pure functions over plain `[f32; 3]` vertex arrays, so
//! they unit-test without a GPU or the `bevy` feature.

/// Displace sphere-surface vertices radially by a per-cell scalar field.
///
/// Terrain relief ships as ordinary scalar *data* — the IR mesh stays
/// undisplaced. A reference renderer that wants visible relief calls this to
/// turn elevation samples into vertex offsets: each vertex moves along its
/// outward radial (the planet is centred at the origin) by `sample * scale`,
/// where `scale` is the consumer's vertical exaggeration.
///
/// * `base` — undisplaced vertex positions.
/// * `vertex_to_cell[v]` — the cell whose sample drives vertex `v`
///   (`None` ⇒ the vertex is left at its base position).
/// * `samples[cell]` — the per-cell scalar (e.g. elevation in metres).
/// * `scale` — vertical exaggeration; `0.0` is a no-op.
///
/// Vertices at the origin, with no owning cell, with an out-of-range cell, or
/// with a non-finite sample keep their base position.
pub fn radial_displaced_positions(
  base: &[[f32; 3]],
  vertex_to_cell: &[Option<usize>],
  samples: &[f64],
  scale: f32,
) -> Vec<[f32; 3]> {
  base
    .iter()
    .enumerate()
    .map(|(v, &p)| {
      let radius = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
      if radius <= f32::EPSILON {
        return p;
      }
      let Some(Some(cell)) = vertex_to_cell.get(v) else {
        return p;
      };
      let Some(&sample) = samples.get(*cell) else {
        return p;
      };
      let offset = sample as f32 * scale;
      if !offset.is_finite() {
        return p;
      }
      let factor = 1.0 + offset / radius;
      [p[0] * factor, p[1] * factor, p[2] * factor]
    })
    .collect()
}

#[cfg(test)]
mod tests {
  use super::*;

  fn radius(p: [f32; 3]) -> f32 {
    (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
  }

  #[test]
  fn zero_scale_is_a_no_op() {
    let base = vec![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
    let out = radial_displaced_positions(
      &base,
      &[Some(0), Some(1)],
      &[100.0, -50.0],
      0.0,
    );
    assert_eq!(out, base);
  }

  #[test]
  fn constant_elevation_lifts_every_vertex_to_the_same_radius() {
    // Unit sphere, elevation 0.25 everywhere, scale 1 → radius 1.25.
    let base = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let out = radial_displaced_positions(
      &base,
      &[Some(0), Some(0), Some(0)],
      &[0.25],
      1.0,
    );
    for p in out {
      assert!(
        (radius(p) - 1.25).abs() < 1.0e-6,
        "radius was {}",
        radius(p)
      );
    }
  }

  #[test]
  fn negative_elevation_pulls_vertices_inward() {
    let base = vec![[2.0, 0.0, 0.0]];
    let out = radial_displaced_positions(&base, &[Some(0)], &[-0.5], 1.0);
    assert!((radius(out[0]) - 1.5).abs() < 1.0e-6);
  }

  #[test]
  fn exaggeration_scales_the_offset() {
    let base = vec![[1.0, 0.0, 0.0]];
    let out = radial_displaced_positions(&base, &[Some(0)], &[0.1], 3.0);
    assert!((radius(out[0]) - 1.3).abs() < 1.0e-6);
  }

  #[test]
  fn unmapped_or_nonfinite_vertices_keep_their_base() {
    let base = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]];
    let out = radial_displaced_positions(
      &base,
      // vertex 0: no cell; vertex 1: NaN sample; vertex 2: at origin.
      &[None, Some(0), Some(1)],
      &[f64::NAN, 5.0],
      1.0,
    );
    assert_eq!(out, base);
  }

  #[test]
  fn out_of_range_cell_keeps_base() {
    let base = vec![[1.0, 0.0, 0.0]];
    let out = radial_displaced_positions(&base, &[Some(7)], &[1.0], 1.0);
    assert_eq!(out, base);
  }
}
