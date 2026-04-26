// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use utility::profile;

use crate::geometry::CellId;

pub trait FieldStorage<const N: usize>: Send + Sync {
  type CellView<'a>: CellView<N>
  where
    Self: 'a;
  type ComponentView<'a>: AsRef<[f64]>
  where
    Self: 'a;

  fn state(&self, cell: CellId) -> Self::CellView<'_>;
  fn state_into(&self, cell: CellId, out: &mut [f64; N]) {
    *out = *self.state(cell).as_state();
  }
  fn component(&self, index: usize) -> Self::ComponentView<'_>;
  fn component_into(&self, index: usize, out: &mut [f64]) {
    debug_assert_eq!(out.len(), self.len());
    out.copy_from_slice(self.component(index).as_ref());
  }
  fn write(&mut self, cell: CellId, val: &[f64; N]);
  fn len(&self) -> usize;
  fn is_empty(&self) -> bool;
  fn axpy(&mut self, alpha: f64, other: &Self);
  fn weighted_sum(&mut self, a: f64, x: &Self, b: f64, y: &Self);
  fn clone_state(&self) -> Self;
}

pub trait CellView<const N: usize> {
  fn as_state(&self) -> &[f64; N];
}

impl<const N: usize> CellView<N> for [f64; N] {
  fn as_state(&self) -> &[f64; N] {
    self
  }
}

impl<const N: usize> CellView<N> for &[f64; N] {
  fn as_state(&self) -> &[f64; N] {
    self
  }
}

pub struct SoaField<const N: usize> {
  state: [Vec<f64>; N],
}

impl<const N: usize> FieldStorage<N> for SoaField<N> {
  type CellView<'a>
    = [f64; N]
  where
    Self: 'a;
  type ComponentView<'a>
    = &'a [f64]
  where
    Self: 'a;

  fn state(&self, cell: CellId) -> Self::CellView<'_> {
    let mut out = [0.0; N];
    self.state_into(cell, &mut out);
    out
  }

  fn state_into(&self, cell: CellId, out: &mut [f64; N]) {
    let index = cell.index();
    for (i, component) in self.state.iter().enumerate() {
      out[i] = component[index];
    }
  }

  fn component(&self, index: usize) -> Self::ComponentView<'_> {
    &self.state[index]
  }

  fn component_into(&self, index: usize, out: &mut [f64]) {
    debug_assert_eq!(out.len(), self.state[index].len());
    out.copy_from_slice(&self.state[index]);
  }

  fn write(&mut self, cell: CellId, val: &[f64; N]) {
    for (i, state) in self.state.iter_mut().enumerate() {
      state[cell.index()] = val[i];
    }
  }

  fn len(&self) -> usize {
    // all vecs are same length
    self.state[0].len()
  }

  fn is_empty(&self) -> bool {
    self.state[0].is_empty()
  }

  #[profile]
  fn axpy(&mut self, alpha: f64, other: &Self) {
    for i in 0..N {
      for (a, b) in self.state[i].iter_mut().zip(&other.state[i]) {
        *a += alpha * b;
      }
    }
  }

  #[profile]
  fn weighted_sum(&mut self, a: f64, x: &Self, b: f64, y: &Self) {
    for i in 0..N {
      for j in 0..self.state[i].len() {
        self.state[i][j] = a * x.state[i][j] + b * y.state[i][j];
      }
    }
  }

  #[profile]
  fn clone_state(&self) -> Self {
    SoaField {
      state: std::array::from_fn(|i| self.state[i].clone()),
    }
  }
}

impl<const N: usize> SoaField<N> {
  pub fn zeros(count: usize) -> SoaField<N> {
    SoaField {
      state: std::array::from_fn(|_| vec![0.0; count]),
    }
  }

  pub fn from_fn(count: usize, f: impl Fn(CellId) -> [f64; N]) -> Self {
    let mut state: [Vec<f64>; N] = std::array::from_fn(|_| vec![0.0; count]);
    for j in 0..count {
      let val = f(CellId::from(j));
      for i in 0..N {
        state[i][j] = val[i];
      }
    }
    SoaField { state }
  }
}

pub struct AosField<const N: usize> {
  state: Vec<[f64; N]>,
}

impl<const N: usize> FieldStorage<N> for AosField<N> {
  type CellView<'a>
    = &'a [f64; N]
  where
    Self: 'a;
  type ComponentView<'a>
    = Vec<f64>
  where
    Self: 'a;

  fn state(&self, cell: CellId) -> Self::CellView<'_> {
    &self.state[cell.index()]
  }

  fn state_into(&self, cell: CellId, out: &mut [f64; N]) {
    *out = self.state[cell.index()];
  }

  fn component(&self, index: usize) -> Self::ComponentView<'_> {
    self.state.iter().map(|s| s[index]).collect::<Vec<f64>>()
  }

  fn component_into(&self, index: usize, out: &mut [f64]) {
    debug_assert_eq!(out.len(), self.state.len());
    for (row, sample) in self.state.iter().enumerate() {
      out[row] = sample[index];
    }
  }

  fn write(&mut self, cell: CellId, val: &[f64; N]) {
    self.state[cell.index()] = *val;
  }

  fn len(&self) -> usize {
    self.state.len()
  }

  fn is_empty(&self) -> bool {
    self.state.is_empty()
  }

  #[profile]
  fn axpy(&mut self, alpha: f64, other: &Self) {
    for (a, b) in self.state.iter_mut().zip(&other.state) {
      for i in 0..N {
        a[i] += alpha * b[i];
      }
    }
  }

  #[profile]
  fn weighted_sum(&mut self, a: f64, x: &Self, b: f64, y: &Self) {
    for j in 0..self.state.len() {
      for i in 0..N {
        self.state[j][i] = a * x.state[j][i] + b * y.state[j][i];
      }
    }
  }

  #[profile]
  fn clone_state(&self) -> Self {
    AosField {
      state: self.state.clone(),
    }
  }
}

impl<const N: usize> AosField<N> {
  pub fn zeros(count: usize) -> Self {
    AosField {
      state: vec![[0.0; N]; count],
    }
  }

  pub fn from_fn(count: usize, f: impl Fn(CellId) -> [f64; N]) -> Self {
    AosField {
      state: (0..count).map(|j| f(CellId::from(j))).collect(),
    }
  }
}
