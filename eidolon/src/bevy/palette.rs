// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Palette helpers shared by the paint system.
//!
//! For v0.1 paint runs CPU-side: each cell's scalar value is mapped to
//! an RGBA colour via the palette's interpolated stops, then written
//! into `Mesh::ATTRIBUTE_COLOR`. A future revision can bake the
//! palette into a `1×N` GPU LUT and sample it in a shader; this module
//! keeps the lookup helper isolated for that swap.

use crate::ir::{Palette, Rgba};

/// Sample a palette at a normalised position `t ∈ [0, 1]`. Returns the
/// nearest stop colour with linear interpolation in RGB space.
pub fn sample(palette: &Palette, t: f32) -> Rgba {
  if palette.stops.is_empty() {
    return Rgba::WHITE;
  }
  if palette.stops.len() == 1 {
    return palette.stops[0].colour;
  }
  let t = t.clamp(0.0, 1.0);
  let stops = &palette.stops;
  let last = stops.len() - 1;
  // Find the two stops bracketing t. Stops are sorted by position in
  // the IR; assert is a debug-only contract.
  for window_idx in 0..last {
    let lo = &stops[window_idx];
    let hi = &stops[window_idx + 1];
    if t >= lo.at && t <= hi.at {
      let span = (hi.at - lo.at).max(f32::EPSILON);
      let alpha = (t - lo.at) / span;
      return lerp_rgba(lo.colour, hi.colour, alpha);
    }
  }
  // Out of range above last stop.
  stops[last].colour
}

fn lerp_rgba(a: Rgba, b: Rgba, t: f32) -> Rgba {
  Rgba {
    r: a.r + (b.r - a.r) * t,
    g: a.g + (b.g - a.g) * t,
    b: a.b + (b.b - a.b) * t,
    a: a.a + (b.a - a.a) * t,
  }
}

/// Map a scalar to a colour via `palette` and an explicit `[min, max]`
/// range. Out-of-range values are clamped.
pub fn colour_for_scalar(
  palette: &Palette,
  value: f32,
  min: f32,
  max: f32,
) -> Rgba {
  if !value.is_finite() {
    return Rgba::WHITE;
  }
  let span = (max - min).max(f32::EPSILON);
  sample(palette, (value - min) / span)
}
