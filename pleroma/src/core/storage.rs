// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use tessera::{mesh::Mesh, partition::PartitionMesh};
use utility::serial::deserialize::{Deserialize, Deserializer};
use utility::serial::serialize::{Serialize, Serializer};
use utility::{domain::CellId, profile};

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

// Checkpoint round-trip: a structure-of-arrays field serializes as a sequence of
// `N` component vectors. Field state is the bulk of any checkpoint, and the JSON
// backend's `{}`-formatted f64 round-trips finite values bit-exactly.
impl<const N: usize> Serialize for SoaField<N> {
  fn serialize<S: Serializer>(&self, s: &mut S) -> Result<(), S::Error> {
    self.state.serialize(s)
  }
}

impl<const N: usize> Deserialize for SoaField<N> {
  fn deserialize<D: Deserializer>(d: &mut D) -> Result<Self, D::Error> {
    let components = <[Vec<f64>; N]>::deserialize(d)?;
    Ok(SoaField { state: components })
  }
}

#[derive(Clone)]
pub struct LocalPartitionField<const N: usize> {
  values: Vec<[f64; N]>,
  owned_count: usize,
}

impl<const N: usize> FieldStorage<N> for LocalPartitionField<N> {
  type CellView<'a>
    = &'a [f64; N]
  where
    Self: 'a;
  type ComponentView<'a>
    = Vec<f64>
  where
    Self: 'a;

  fn state(&self, cell: CellId) -> Self::CellView<'_> {
    &self.values[cell.index()]
  }

  fn state_into(&self, cell: CellId, out: &mut [f64; N]) {
    *out = self.values[cell.index()];
  }

  fn component(&self, index: usize) -> Self::ComponentView<'_> {
    self.values.iter().map(|s| s[index]).collect()
  }

  fn component_into(&self, index: usize, out: &mut [f64]) {
    debug_assert_eq!(out.len(), self.values.len());
    for (row, sample) in self.values.iter().enumerate() {
      out[row] = sample[index];
    }
  }

  fn write(&mut self, cell: CellId, val: &[f64; N]) {
    self.values[cell.index()] = *val;
  }

  fn len(&self) -> usize {
    self.values.len()
  }

  fn is_empty(&self) -> bool {
    self.values.is_empty()
  }

  #[profile]
  fn axpy(&mut self, alpha: f64, other: &Self) {
    debug_assert_eq!(self.values.len(), other.values.len());
    for (a, b) in self.values.iter_mut().zip(&other.values) {
      for i in 0..N {
        a[i] += alpha * b[i];
      }
    }
  }

  #[profile]
  fn weighted_sum(&mut self, a: f64, x: &Self, b: f64, y: &Self) {
    debug_assert_eq!(self.values.len(), x.values.len());
    debug_assert_eq!(self.values.len(), y.values.len());
    for j in 0..self.values.len() {
      for i in 0..N {
        self.values[j][i] = a * x.values[j][i] + b * y.values[j][i];
      }
    }
  }

  #[profile]
  fn clone_state(&self) -> Self {
    self.clone()
  }
}

impl<const N: usize> LocalPartitionField<N> {
  pub fn new(values: Vec<[f64; N]>, owned_count: usize) -> Self {
    assert!(
      owned_count <= values.len(),
      "owned_count must not exceed local value count",
    );
    LocalPartitionField {
      values,
      owned_count,
    }
  }

  pub fn zeros(owned_count: usize, ghost_count: usize) -> Self {
    LocalPartitionField {
      values: vec![[0.0; N]; owned_count + ghost_count],
      owned_count,
    }
  }

  pub fn owned_count(&self) -> usize {
    self.owned_count
  }

  pub fn ghost_count(&self) -> usize {
    self.values.len() - self.owned_count
  }

  pub fn values(&self) -> &[[f64; N]] {
    &self.values
  }

  pub fn values_mut(&mut self) -> &mut [[f64; N]] {
    &mut self.values
  }

  pub fn owned_values(&self) -> &[[f64; N]] {
    &self.values[..self.owned_count]
  }

  pub fn owned_values_mut(&mut self) -> &mut [[f64; N]] {
    &mut self.values[..self.owned_count]
  }
}

#[profile]
pub fn gather_partition_field<const D: usize, const N: usize, M>(
  field: &SoaField<N>,
  partition: &PartitionMesh<D, M>,
) -> LocalPartitionField<N>
where
  M: Mesh<D>,
{
  let mut values = Vec::with_capacity(partition.local_cell_count());
  for &global_cell in partition.local_to_global_cells() {
    let mut state = [0.0; N];
    field.state_into(global_cell, &mut state);
    values.push(state);
  }

  LocalPartitionField::new(values, partition.num_owned())
}

#[profile]
pub fn scatter_partition_owned<const D: usize, const N: usize, M>(
  local: &LocalPartitionField<N>,
  global: &mut SoaField<N>,
  partition: &PartitionMesh<D, M>,
) where
  M: Mesh<D>,
{
  assert_eq!(
    local.len(),
    partition.local_cell_count(),
    "local field length must match partition cell count",
  );
  assert_eq!(
    local.owned_count(),
    partition.num_owned(),
    "local field owned count must match partition owned count",
  );

  for local_index in 0..partition.num_owned() {
    let local_cell = CellId::from(local_index);
    let global_cell = partition.local_to_global(local_cell);
    global.write(global_cell, local.state(local_cell).as_state());
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

// Checkpoint round-trip: an array-of-structures field serializes as a sequence of
// per-cell `[f64; N]` states, reusing the fixed-array and `Vec` codecs.
impl<const N: usize> Serialize for AosField<N> {
  fn serialize<S: Serializer>(&self, s: &mut S) -> Result<(), S::Error> {
    self.state.serialize(s)
  }
}

impl<const N: usize> Deserialize for AosField<N> {
  fn deserialize<D: Deserializer>(d: &mut D) -> Result<Self, D::Error> {
    Ok(AosField {
      state: Vec::<[f64; N]>::deserialize(d)?,
    })
  }
}

#[cfg(test)]
mod tests {
  use std::sync::Arc;

  use tessera::{
    geometry::IdentityMap, mesh::StructuredBlock,
    partition::decompose_structured,
  };

  use super::*;

  fn test_decomposition()
  -> tessera::partition::Decomposition<3, StructuredBlock<3>> {
    let dims = [4, 2, 1];
    let mesh = Arc::new(StructuredBlock::uniform(
      [0.0; 3].into(),
      [1.0; 3],
      dims,
      Box::new(IdentityMap::<3>),
    ));
    decompose_structured(mesh, dims, 2, 1)
  }

  #[test]
  fn gather_partition_field_includes_owned_and_ghost_cells() {
    let decomposition = test_decomposition();
    let partition = &decomposition.partitions[0];
    let global = SoaField::<2>::from_fn(8, |cell| {
      let index = cell.index() as f64;
      [index, index + 100.0]
    });

    let local = gather_partition_field(&global, partition);

    assert_eq!(local.owned_count(), partition.num_owned());
    assert_eq!(local.ghost_count(), partition.ghost_cells().len());
    assert_eq!(local.len(), partition.local_cell_count());
    assert!(local.ghost_count() > 0);

    for (local_index, &global_cell) in
      partition.local_to_global_cells().iter().enumerate()
    {
      let expected = [
        global_cell.index() as f64,
        global_cell.index() as f64 + 100.0,
      ];
      assert_eq!(local.state(CellId::from(local_index)).as_state(), &expected);
    }
  }

  #[test]
  fn scatter_partition_owned_ignores_local_ghost_values() {
    let decomposition = test_decomposition();
    let partition = &decomposition.partitions[0];
    let mut global = SoaField::<1>::from_fn(8, |cell| [cell.index() as f64]);
    let mut local = gather_partition_field(&global, partition);

    let owned_globals =
      partition.local_to_global_cells()[..partition.num_owned()].to_vec();
    let ghost_globals =
      partition.local_to_global_cells()[partition.num_owned()..].to_vec();

    for local_index in 0..local.len() {
      let value = if local_index < local.owned_count() {
        [1000.0 + local_index as f64]
      } else {
        [9000.0 + local_index as f64]
      };
      local.write(CellId::from(local_index), &value);
    }

    scatter_partition_owned(&local, &mut global, partition);

    for (local_index, global_cell) in owned_globals.into_iter().enumerate() {
      assert_eq!(
        global.state(global_cell).as_state(),
        &[1000.0 + local_index as f64],
      );
    }

    for global_cell in ghost_globals {
      assert_eq!(
        global.state(global_cell).as_state(),
        &[global_cell.index() as f64],
      );
    }
  }

  fn json_roundtrip<T>(value: &T) -> T
  where
    T: Serialize + Deserialize,
  {
    use utility::serial::json::{JsonDeserializer, JsonSerializer};
    let mut buf = Vec::new();
    {
      let mut ser = JsonSerializer::new(&mut buf);
      value.serialize(&mut ser).expect("serialize");
    }
    let mut de = JsonDeserializer::new(buf.as_slice());
    T::deserialize(&mut de).expect("deserialize")
  }

  #[test]
  fn soa_field_survives_a_json_round_trip() {
    let field = SoaField::<3>::from_fn(5, |c| {
      let i = c.index() as f64;
      [i, -i * 0.5, 1.0e6 + i]
    });
    let restored = json_roundtrip(&field);
    assert_eq!(restored.len(), field.len());
    for i in 0..field.len() {
      assert_eq!(
        restored.state(CellId::from(i)).as_state(),
        field.state(CellId::from(i)).as_state(),
      );
    }
  }

  #[test]
  fn aos_field_survives_a_json_round_trip() {
    let field = AosField::<2>::from_fn(4, |c| {
      let i = c.index() as f64;
      [i * 2.0, i - 10.0]
    });
    let restored = json_roundtrip(&field);
    assert_eq!(restored.len(), field.len());
    for i in 0..field.len() {
      assert_eq!(
        restored.state(CellId::from(i)).as_state(),
        field.state(CellId::from(i)).as_state(),
      );
    }
  }

  #[test]
  fn local_partition_field_clone_and_algebra_cover_all_local_slots() {
    let mut lhs = LocalPartitionField::<1>::new(vec![[1.0], [2.0], [3.0]], 2);
    let rhs = LocalPartitionField::<1>::new(vec![[10.0], [20.0], [30.0]], 2);

    lhs.axpy(0.5, &rhs);
    assert_eq!(lhs.values(), &[[6.0], [12.0], [18.0]]);

    let cloned = lhs.clone_state();
    lhs.weighted_sum(0.25, &cloned, 0.75, &rhs);

    assert_eq!(lhs.values(), &[[9.0], [18.0], [27.0]]);
  }
}
