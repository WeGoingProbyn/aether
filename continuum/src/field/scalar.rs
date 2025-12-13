pub struct ScalarField {
  pub field: Vec<f32>,
}

impl ScalarField {
  pub fn new(size: usize) -> Self {
    Self {
      field: vec![0.0; size],
    }
  }

  pub fn get(&self, idx: usize) -> f32 {
    self.field[idx]
  }

  pub fn set(&mut self, idx: usize, value: f32) {
    self.field[idx] = value;
  }
}
