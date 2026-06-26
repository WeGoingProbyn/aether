// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

//! Unit coverage for `continuum::diagnostics::count_non_finite`: a clean field
//! reports zero, and injected NaN/Inf cells are counted exactly once each.

use continuum::diagnostics::count_non_finite;
use pleroma::core::storage::{FieldStorage, SoaField};
use utility::domain::CellId;

#[test]
fn clean_field_reports_zero() {
  let field = SoaField::<3>::from_fn(8, |_| [1.0, 2.0, 3.0]);
  assert_eq!(count_non_finite(&field), 0);
}

#[test]
fn counts_each_non_finite_cell_once() {
  let mut field = SoaField::<3>::from_fn(5, |_| [1.0, 2.0, 3.0]);
  // Cell 1: one NaN component. Cell 3: one +Inf component (and a second
  // non-finite component to confirm a cell is still counted only once).
  field.write(CellId::from(1), &[f64::NAN, 2.0, 3.0]);
  field.write(CellId::from(3), &[1.0, f64::INFINITY, f64::NEG_INFINITY]);
  assert_eq!(count_non_finite(&field), 2);
}
