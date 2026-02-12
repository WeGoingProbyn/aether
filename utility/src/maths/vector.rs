use std::ops::{
  Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, Sub, SubAssign,
};

use crate::maths::matrix::Matrix;

#[derive(PartialEq)]
pub struct Vector<T: Default, const C: usize> {
  inner: Matrix<T, C, 1>,
}

// ==================Lin alg functions=====================//

impl<T, const C: usize> Vector<T, C>
where
  T: Default + Clone + AddAssign + Mul<Output = T>,
{
  #[allow(unused)]
  pub fn dot_clone(&self, rhs: &Vector<T, C>) -> T {
    self.inner.dot_clone(&rhs.inner)
  }
}

impl<T, const C: usize> Vector<T, C>
where
  T: Default,
  for<'x> &'x T: Mul<&'x T, Output = T> + Add<&'x T, Output = T>,
{
  pub fn dot(&self, rhs: &Vector<T, C>) -> T {
    self.inner.dot(&rhs.inner)
  }
}

impl<T, const C: usize> Vector<T, C> 
where 
  T: Default + Clone + Add<Output = T>,
{
  pub fn into_sum(self) -> T {
    self.into_iter().reduce(|a, b| a + b).unwrap_or_default()
  }
}

impl<T, const C: usize> Vector<T, C> 
where 
  T: Default,
  for<'x> &'x T: Add<&'x T, Output = T>,
{
  pub fn sum(&self) -> T {
    self.iter().fold(T::default(), |acc, x| &acc + x)
  }
}

impl<const C: usize> Vector<f32, C>
where
  for<'x> &'x f32: Mul<&'x f32, Output = f32> + Add<&'x f32, Output = f32>,
{
  pub fn magnitude(&self) -> f32 {
    self.inner.dot(&self.inner).sqrt()
  }
}

impl<const C: usize> Vector<f32, C>
{
  pub fn normalise(&self) -> Vector<f32, C> {
    self / &self.magnitude()
  }
}

impl<T> Vector<T, 3>
where
  T: Default,
  for<'x> &'x T: Mul<&'x T, Output = T> + Sub<&'x T, Output = T>,
{
  #[allow(unused)]
  pub fn cross(&self, other: &Self) -> Vector<T, 3> {
    let mut out = Vector::<T, 3>::default();

    out[0] = &(&self[1] * &other[2]) - &(&self[2] * &other[1]);
    out[1] = &(&self[2] * &other[0]) - &(&self[0] * &other[2]);
    out[2] = &(&self[0] * &other[1]) - &(&self[1] * &other[0]);

    out
  }
}

impl<T> Vector<T, 3>
where
  T: Default + Sub<Output = T> + Mul<Output = T> + Clone,
{
  #[allow(unused)]
  pub fn cross_clone(&self, other: &Self) -> Vector<T, 3> {
    let mut out = Vector::<T, 3>::default();

    out[0] =
      self[1].clone() * other[2].clone() - self[2].clone() * other[1].clone();
    out[1] =
      self[2].clone() * other[0].clone() - self[0].clone() * other[2].clone();
    out[2] =
      self[0].clone() * other[1].clone() - self[1].clone() * other[0].clone();

    out
  }
}

// ============= Iterators =========================//

pub struct VectorIter<'a, T, const C: usize> 
where 
  T: Default,
{
  inner: &'a Matrix<T, C, 1>,
}

impl<'a, T, const C: usize> Iterator for VectorIter<'a, T, C> 
where
  T: Default,
{
  type Item = &'a T;
  fn next(&mut self) -> Option<Self::Item> {
    self.inner.iter().next()
  }
}

// impl<'a, T, const C: usize> Sum<&'a Vector<T, C>> for Vector<T, C>
// where 
//   T: Default,
//   for<'x> &'x T: Add<&'x T, Output = T>,
// {
//   fn sum<I>(iter: I) -> Self
// where
//     I: Iterator<Item = &'a Vector<T, C>>
//   {
//     iter.fold(Vector::default(), |acc, v| &acc + v)
//   }
// }
//
// impl<T, const C: usize> Sum for Vector<T, C>
// where
//   T: Default + Clone + Add<Output = T>,
// {
//   fn sum<I>(iter: I) -> Self
// where
//     I: Iterator<Item = Self>,
//   {
//     iter.fold(Vector::default(), |acc, v| acc + v)
//   }
// }

pub struct VectorIterMut<'a, T, const C: usize> 
where 
  T: Default,
{
  inner: &'a mut Matrix<T, C, 1>,
  col: usize,
}

impl<'a, T, const C: usize> Iterator for VectorIterMut<'a, T, C> 
where
  T: Default,
{
  type Item = &'a mut T;
  fn next(&mut self) -> Option<Self::Item> {
    if self.col >= C { return None }

    let next = &mut self.inner[0][self.col] as *mut T;
    self.col += 1;

    // This is safe because we never
    // index past the bound of C!
    Some(unsafe { &mut *next })
  }
}

pub struct VectorIterInto<T, const C: usize> 
where 
  T: Default + Clone,
{
  inner: Matrix<T, C, 1>,
  col: usize,
}

impl<T, const C: usize> Iterator for VectorIterInto<T, C> 
where
  T: Default + Clone,
{
  type Item = T;
  fn next(&mut self) -> Option<Self::Item> {
    if self.col >= C { return None }

    let next = self.inner[0][self.col].clone();
    self.col += 1;

    // This is safe because we never
    // index past the bound of C!
    Some(next)
  }
}

impl<T, const C: usize> Vector<T, C> 
where 
  T: Default,
{
  pub fn iter(&self) -> VectorIter<'_, T, C> {
    VectorIter {
      inner: &self.inner,
    }
  }

  pub fn iter_mut(&mut self) -> VectorIterMut<'_, T, C> {
    VectorIterMut {
      inner: &mut self.inner,
      col: 0,
    }
  }
}

impl<T, const C: usize> IntoIterator for Vector<T, C> 
where 
  T: Default + Clone,
{
  type Item = T;
  type IntoIter = VectorIterInto<T, C>;

  fn into_iter(self) -> Self::IntoIter {
    VectorIterInto {
      inner: self.inner,
      col: 0,
    }
  }
}

// ==================== Constructors =====================//

impl<T, const C: usize> From<[T; C]> for Vector<T, C>
where
  T: Default,
{
  fn from(item: [T; C]) -> Vector<T, C> {
    let new = [item; 1];
    Vector { inner: new.into() }
  }
}

impl<T, const C: usize> From<Matrix<T, C, 1>> for Vector<T, C>
where
  T: Default,
{
  fn from(item: Matrix<T, C, 1>) -> Vector<T, C> {
    Vector {
      inner: item.inner.into(),
    }
  }
}

impl<T, const C: usize> std::fmt::Debug for Vector<T, C>
where
  T: Default + std::fmt::Display,
{
  fn fmt(
    &self,
    f: &mut std::fmt::Formatter<'_>,
  ) -> Result<(), std::fmt::Error> {
    writeln!(f, "Vector<T = {}, C = {}>:", std::any::type_name::<T>(), C)?;
    write!(f, "{:?}", self.inner)?;
    Ok(())
  }
}

impl<T: Default, const C: usize> Default for Vector<T, C> {
  fn default() -> Self {
    Vector {
      inner: Matrix::<T, C, 1>::default(),
    }
  }
}

// ================ Display impl ========================//

impl<T, const C: usize> std::fmt::Display for Vector<T, C>
where
  T: Default + std::fmt::Display,
{
  fn fmt(
    &self,
    f: &mut std::fmt::Formatter<'_>,
  ) -> Result<(), std::fmt::Error> {
    write!(f, "{}", self.inner)?;
    Ok(())
  }
}

// ====================== Index impls ======================//

impl<T: Default, const C: usize> Index<usize> for Vector<T, C> {
  type Output = T;

  fn index(&self, index: usize) -> &T {
    &self.inner[0][index]
  }
}

impl<T: Default, const C: usize> IndexMut<usize> for Vector<T, C> {
  fn index_mut(&mut self, index: usize) -> &mut T {
    &mut self.inner[0][index]
  }
}

// ====================== Add impls ======================//

impl<'a, T: Default, const C: usize> Add<&'a Vector<T, C>> for &Vector<T, C>
where
  for<'x> &'x T: Add<&'x T, Output = T>,
{
  type Output = Vector<T, C>;

  fn add(self, rhs: &'a Vector<T, C>) -> Self::Output {
    Vector {
      inner: &self.inner + &rhs.inner,
    }
  }
}

impl<T, const C: usize> Add for Vector<T, C>
where
  T: Add<Output = T> + Default + Clone,
{
  type Output = Self;

  fn add(self, rhs: Self) -> Self {
    Vector {
      inner: self.inner + rhs.inner,
    }
  }
}

impl<T, const C: usize> AddAssign for Vector<T, C>
where
  T: Default + Clone + AddAssign,
{
  fn add_assign(&mut self, rhs: Self) {
    self.inner += rhs.inner;
  }
}

// ====================== Sub impls ======================//

impl<'a, T: Default, const C: usize> Sub<&'a Vector<T, C>> for &Vector<T, C>
where
  for<'x> &'x T: Sub<&'x T, Output = T>,
{
  type Output = Vector<T, C>;

  fn sub(self, rhs: &'a Vector<T, C>) -> Self::Output {
    Vector {
      inner: &self.inner - &rhs.inner,
    }
  }
}

impl<T, const C: usize> Sub for Vector<T, C>
where
  T: Sub<Output = T> + Default + Clone,
{
  type Output = Self;

  fn sub(self, rhs: Self) -> Self {
    Vector {
      inner: self.inner - rhs.inner,
    }
  }
}

impl<T, const C: usize> SubAssign for Vector<T, C>
where
  T: Default + Clone + SubAssign,
{
  fn sub_assign(&mut self, rhs: Self) {
    self.inner -= rhs.inner;
  }
}

// ====================== Div impls ======================//

impl<'a, T: Default, const C: usize> Div<&'a Vector<T, C>> for &Vector<T, C>
where
  for<'x> &'x T: Div<&'x T, Output = T>,
{
  type Output = Vector<T, C>;

  fn div(self, rhs: &'a Vector<T, C>) -> Self::Output {
    Vector {
      inner: &self.inner / &rhs.inner,
    }
  }
}

impl<T, const C: usize> Div for Vector<T, C>
where
  T: Div<Output = T> + Default + Clone,
{
  type Output = Self;

  fn div(self, rhs: Self) -> Self {
    Vector {
      inner: self.inner / rhs.inner,
    }
  }
}

impl<T, const C: usize> DivAssign for Vector<T, C>
where
  T: Default + Clone + DivAssign,
{
  fn div_assign(&mut self, rhs: Self) {
    self.inner /= rhs.inner;
  }
}

impl<'a, T: Default, const C: usize> Div<&'a T> for &Vector<T, C>
where
  for<'x> &'x T: Div<&'x T, Output = T>,
{
  type Output = Vector<T, C>;

  fn div(self, rhs: &'a T) -> Self::Output {
    Vector {
      inner: &self.inner / rhs,
    }
  }
}

impl<T, const C: usize> Div<T> for Vector<T, C>
where
  T: Div<Output = T> + Default + Clone,
{
  type Output = Self;

  fn div(self, rhs: T) -> Self {
    Vector {
      inner: self.inner / rhs,
    }
  }
}

impl<T, const C: usize> DivAssign<T> for Vector<T, C>
where
  T: Default + Clone + DivAssign,
{
  fn div_assign(&mut self, rhs: T) {
    self.inner /= rhs;
  }
}

// ====================== Mul impls ======================//

impl<T, const K: usize, const N: usize> Mul<&Matrix<T, N, K>> for &Vector<T, K>
where
  T: Default,
  for<'x> &'x T: Mul<&'x T, Output = T> + Add<&'x T, Output = T>,
{
  type Output = Vector<T, N>;

  fn mul(self, rhs: &Matrix<T, N, K>) -> Self::Output {
    (&self.inner * rhs).into()
  }
}

impl<T, const K: usize, const N: usize> Mul<Matrix<T, N, K>> for Vector<T, K>
where
  T: Default + Clone + Mul<T, Output = T> + Add<T, Output = T>,
{
  type Output = Vector<T, N>;

  fn mul(self, rhs: Matrix<T, N, K>) -> Self::Output {
    (self.inner * rhs).into()
  }
}

// ======================= Unit tests ===========================//

#[cfg(test)]
mod test {
  use crate::maths::matrix::Matrix;
  use crate::maths::vector::Vector;

  #[test]
  fn check_cross_product() {
    let one: Vector<f32, 3> = [1.0, 2.0, 3.0].into();
    let two: Vector<f32, 3> = [2.0, 3.0, 4.0].into();
    let res: Vector<f32, 3> = [-1.0, 2.0, -1.0].into();

    let out = one.cross_clone(&two);
    let out_ref = one.cross(&two);
    assert_eq!(out, res);
    assert_eq!(out_ref, res);
  }

  #[test]
  fn check_vec_mat_mul() {
    let a: Vector<f32, 3> = [3.0, 4.0, 2.0].into();
    let b: Matrix<f32, 4, 3> = [
      [13.0, 9.0, 7.0, 15.0],
      [8.0, 7.0, 4.0, 6.0],
      [6.0, 4.0, 0.0, 3.0],
    ]
    .into();

    let known: Vector<f32, 4> = [83.0, 63.0, 37.0, 75.0].into();

    let res = &a * &b;
    let res_clone = a * b;

    assert_eq!(res, known);
    assert_eq!(res_clone, known);
  }
}
