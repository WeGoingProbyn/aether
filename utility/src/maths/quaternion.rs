use std::ops::{
  Add, AddAssign, Div, DivAssign, Index, IndexMut, Sub, SubAssign,
};

use crate::maths::matrix::Matrix;

pub struct Quaternion<T>
where
  T: Default,
{
  inner: Matrix<T, 4, 1>,
}

impl<T> From<[T; 4]> for Quaternion<T>
where
  T: Default,
{
  fn from(item: [T; 4]) -> Quaternion<T> {
    let new = [item; 1];
    Quaternion { inner: new.into() }
  }
}

impl<T> std::fmt::Display for Quaternion<T>
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

impl<T> std::fmt::Debug for Quaternion<T>
where
  T: Default + std::fmt::Display,
{
  fn fmt(
    &self,
    f: &mut std::fmt::Formatter<'_>,
  ) -> Result<(), std::fmt::Error> {
    writeln!(f, "Quaternion<T = {}>:", std::any::type_name::<T>())?;
    write!(f, "{:?}", self.inner)?;
    Ok(())
  }
}

impl<T: Default> Default for Quaternion<T> {
  fn default() -> Self {
    Quaternion {
      inner: Matrix::<T, 4, 1>::default(),
    }
  }
}

impl<T: Default> Index<usize> for Quaternion<T> {
  type Output = T;

  fn index(&self, index: usize) -> &T {
    &self.inner[0][index]
  }
}

impl<T: Default> IndexMut<usize> for Quaternion<T> {
  fn index_mut(&mut self, index: usize) -> &mut T {
    &mut self.inner[0][index]
  }
}

impl<'a, T: Default> Add<&'a Quaternion<T>> for &Quaternion<T>
where
  for<'x> &'x T: Add<&'x T, Output = T>,
{
  type Output = Quaternion<T>;

  fn add(self, rhs: &'a Quaternion<T>) -> Self::Output {
    Quaternion {
      inner: &self.inner + &rhs.inner,
    }
  }
}

impl<T> Add for Quaternion<T>
where
  T: Add<Output = T> + Default + Clone,
{
  type Output = Self;

  fn add(self, rhs: Self) -> Self {
    Quaternion {
      inner: self.inner + rhs.inner,
    }
  }
}

impl<T> AddAssign for Quaternion<T>
where
  T: Default + Clone + AddAssign,
{
  fn add_assign(&mut self, rhs: Self) {
    self.inner += rhs.inner;
  }
}

impl<'a, T: Default> Sub<&'a Quaternion<T>> for &Quaternion<T>
where
  for<'x> &'x T: Sub<&'x T, Output = T>,
{
  type Output = Quaternion<T>;

  fn sub(self, rhs: &'a Quaternion<T>) -> Self::Output {
    Quaternion {
      inner: &self.inner - &rhs.inner,
    }
  }
}

impl<T> Sub for Quaternion<T>
where
  T: Sub<Output = T> + Default + Clone,
{
  type Output = Self;

  fn sub(self, rhs: Self) -> Self {
    Quaternion {
      inner: self.inner - rhs.inner,
    }
  }
}

impl<T> SubAssign for Quaternion<T>
where
  T: Default + Clone + SubAssign,
{
  fn sub_assign(&mut self, rhs: Self) {
    self.inner -= rhs.inner;
  }
}

impl<'a, T: Default> Div<&'a Quaternion<T>> for &Quaternion<T>
where
  for<'x> &'x T: Div<&'x T, Output = T>,
{
  type Output = Quaternion<T>;

  fn div(self, rhs: &'a Quaternion<T>) -> Self::Output {
    Quaternion {
      inner: &self.inner / &rhs.inner,
    }
  }
}

impl<T> Div for Quaternion<T>
where
  T: Div<Output = T> + Default + Clone,
{
  type Output = Self;

  fn div(self, rhs: Self) -> Self {
    Quaternion {
      inner: self.inner / rhs.inner,
    }
  }
}

impl<T> DivAssign for Quaternion<T>
where
  T: Default + Clone + DivAssign,
{
  fn div_assign(&mut self, rhs: Self) {
    self.inner /= rhs.inner;
  }
}
