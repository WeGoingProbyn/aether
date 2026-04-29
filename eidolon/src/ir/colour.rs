// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rgba {
  pub r: f32,
  pub g: f32,
  pub b: f32,
  pub a: f32,
}

impl Rgba {
  pub const WHITE: Rgba = Rgba::new(1.0, 1.0, 1.0, 1.0);
  pub const BLACK: Rgba = Rgba::new(0.0, 0.0, 0.0, 1.0);
  pub const RED: Rgba = Rgba::new(1.0, 0.0, 0.0, 1.0);
  pub const GREEN: Rgba = Rgba::new(0.0, 1.0, 0.0, 1.0);
  pub const BLUE: Rgba = Rgba::new(0.0, 0.0, 1.0, 1.0);
  pub const CYAN: Rgba = Rgba::new(0.0, 1.0, 1.0, 1.0);
  pub const MAGENTA: Rgba = Rgba::new(1.0, 0.0, 1.0, 1.0);
  pub const YELLOW: Rgba = Rgba::new(1.0, 1.0, 0.0, 1.0);

  pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
    Self { r, g, b, a }
  }

  pub const fn as_array(self) -> [f32; 4] {
    [self.r, self.g, self.b, self.a]
  }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarRange {
  pub min: f64,
  pub max: f64,
}

impl ScalarRange {
  pub const fn new(min: f64, max: f64) -> Self {
    Self { min, max }
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Palette {
  pub name: &'static str,
  pub stops: Vec<PaletteStop>,
}

impl Palette {
  pub fn new(name: &'static str, stops: Vec<PaletteStop>) -> Self {
    Self { name, stops }
  }

  pub fn diagnostic() -> Self {
    Self::new(
      "diagnostic",
      vec![
        PaletteStop::new(0.0, Rgba::BLUE),
        PaletteStop::new(0.5, Rgba::CYAN),
        PaletteStop::new(1.0, Rgba::YELLOW),
      ],
    )
  }

  pub fn thermal() -> Self {
    Self::new(
      "thermal",
      vec![
        PaletteStop::new(0.0, Rgba::new(0.05, 0.08, 0.18, 1.0)),
        PaletteStop::new(0.5, Rgba::new(0.9, 0.25, 0.08, 1.0)),
        PaletteStop::new(1.0, Rgba::new(1.0, 0.95, 0.45, 1.0)),
      ],
    )
  }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PaletteStop {
  pub at: f32,
  pub colour: Rgba,
}

impl PaletteStop {
  pub const fn new(at: f32, colour: Rgba) -> Self {
    Self { at, colour }
  }
}
