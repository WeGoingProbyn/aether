use std::ops::{
  Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign
};

use crate::maths::matrix::Matrix;

/// [x, y, z, w]
pub struct Quaternion<T>
where
  T: Default,
{
  inner: Matrix<T, 4, 1>,
}

// =============== Constructors =================//

impl<T> From<[T; 4]> for Quaternion<T>
where
  T: Default,
{
  fn from(item: [T; 4]) -> Quaternion<T> {
    let new = [item; 1];
    Quaternion { inner: new.into() }
  }
}

impl<T: Default> Default for Quaternion<T> {
  fn default() -> Self {
    Quaternion {
      inner: Matrix::<T, 4, 1>::default(),
    }
  }
}

// ================ Lin alg impls ================//

impl<T: Default> Quaternion<T>
where          
  for<'x> &'x T: Neg<Output = T>,
{
  pub fn conjugate(&self) -> Quaternion<T> {
    let mut out = Quaternion {
      inner: -&self.inner,
    };
    out[3] = -&out[3];
    out
  }
}

impl<T> Quaternion<T> 
where 
  T: Default + Neg<Output = T> + Clone
{
  pub fn conjugate_clone(&self) -> Quaternion<T> {
    let mut out = Quaternion {
      inner: -self.inner.clone(),
    };
    out[3] = -out[3].clone();
    out
  }
}

// ================ Display impls =================//

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

// ================ Index impls =================//

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

// ================ Add impls =================//

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

impl<'a, T: Default> Add<&'a T> for &Quaternion<T>
where
  for<'x> &'x T: Add<&'x T, Output = T>,
{
  type Output = Quaternion<T>;

  fn add(self, rhs: &'a T) -> Self::Output {
    Quaternion {
      inner: &self.inner + rhs,
    }
  }
}

impl<T> Add<T> for Quaternion<T>
where
  T: Add<Output = T> + Default + Clone,
{
  type Output = Self;

  fn add(self, rhs: T) -> Self {
    Quaternion {
      inner: self.inner + rhs,
    }
  }
}

impl<T> AddAssign<T> for Quaternion<T>
where
  T: Default + Clone + AddAssign,
{
  fn add_assign(&mut self, rhs: T) {
    self.inner += rhs;
  }
}

// ================ Sub impls =================//

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

impl<'a, T: Default> Sub<&'a T> for &Quaternion<T>
where
  for<'x> &'x T: Sub<&'x T, Output = T>,
{
  type Output = Quaternion<T>;

  fn sub(self, rhs: &'a T) -> Self::Output {
    Quaternion {
      inner: &self.inner - rhs,
    }
  }
}

impl<T> Sub<T> for Quaternion<T>
where
  T: Sub<Output = T> + Default + Clone,
{
  type Output = Self;

  fn sub(self, rhs: T) -> Self {
    Quaternion {
      inner: self.inner - rhs,
    }
  }
}

impl<T> SubAssign<T> for Quaternion<T>
where
  T: Default + Clone + SubAssign,
{
  fn sub_assign(&mut self, rhs: T) {
    self.inner -= rhs;
  }
}

// ================ Div impls =================//

impl<'a, T: Default> Div<&'a T> for &Quaternion<T>
where
  for<'x> &'x T: Div<&'x T, Output = T>,
{
  type Output = Quaternion<T>;

  fn div(self, rhs: &'a T) -> Self::Output {
    Quaternion {
      inner: &self.inner / rhs,
    }
  }
}

impl<T> Div<T> for Quaternion<T>
where
  T: Div<Output = T> + Default + Clone,
{
  type Output = Self;

  fn div(self, rhs: T) -> Self {
    Quaternion {
      inner: self.inner / rhs,
    }
  }
}

impl<T> DivAssign<T> for Quaternion<T>
where
  T: Default + Clone + DivAssign,
{
  fn div_assign(&mut self, rhs: T) {
    self.inner /= rhs;
  }
}

// ================ Mul impls =================//

impl<'a, T: Default> Mul<&'a Quaternion<T>> for &Quaternion<T>
where
  for<'x> &'x T: Mul<&'x T, Output = T> + Sub<&'x T, Output = T> + Add<&'x T, Output = T>,
{
  type Output = Quaternion<T>;

  fn mul(self, rhs: &'a Quaternion<T>) -> Self::Output {
    let mut out = Quaternion::default();
    
    let a = &(&self[3] * &rhs[0]) + &(&self[0] * &rhs[3]);
    let b = &(&self[1] * &rhs[2]) - &(&self[2] * &rhs[1]);
    out[0] = &a + &b;

    let c = &(&self[3] * &rhs[1]) - &(&self[0] * &rhs[2]);
    let d = &(&self[1] * &rhs[3]) + &(&self[2] * &rhs[0]);
    out[1] = &c + &d;

    let d = &(&self[3] * &rhs[2]) + &(&self[0] * &rhs[1]);
    let e = &(&self[1] * &rhs[0]) - &(&self[2] * &rhs[3]);
    out[2] = &d - &e;

    let f = &(&self[3] * &rhs[3]) - &(&self[0] * &rhs[0]);
    let g = &(&self[1] * &rhs[1]) - &(&self[2] * &rhs[2]);
    out[3] = &f - &g;

    out
  }
}

impl<T> Mul for Quaternion<T>
where
  T: Mul<Output = T> + Sub<Output = T> + Add<Output =T> + Default + Clone,
{
  type Output = Self;

  fn mul(self, rhs: Self) -> Self {
    let mut out = Quaternion::default();
    
    let a = self[3].clone() * rhs[0].clone() + self[0].clone() * rhs[3].clone();
    let b = self[1].clone() * rhs[2].clone() - self[2].clone() * rhs[1].clone();
    out[0] = a + b;

    let c = self[3].clone() * rhs[1].clone() - self[0].clone() * rhs[2].clone();
    let d = self[1].clone() * rhs[3].clone() + self[2].clone() * rhs[0].clone();
    out[1] = c + d;

    let d = self[3].clone() * rhs[2].clone() + self[0].clone() * rhs[1].clone();
    let e = self[1].clone() * rhs[0].clone() - self[2].clone() * rhs[3].clone();
    out[2] = d - e;

    let f = self[3].clone() * rhs[3].clone() - self[0].clone() * rhs[0].clone();
    let g = self[1].clone() * rhs[1].clone() - self[2].clone() * rhs[2].clone();
    out[3] = f - g;

    out
  }
}

impl<T> MulAssign for Quaternion<T>
where
  T: Default + Clone + Mul<Output = T> + Add<Output = T> + Sub<Output = T>,
{
  fn mul_assign(&mut self, rhs: Self) {
    let mut out = Quaternion::default();
    
    let a = self[3].clone() * rhs[0].clone() + self[0].clone() * rhs[3].clone();
    let b = self[1].clone() * rhs[2].clone() - self[2].clone() * rhs[1].clone();
    out[0] = a + b;

    let c = self[3].clone() * rhs[1].clone() - self[0].clone() * rhs[2].clone();
    let d = self[1].clone() * rhs[3].clone() + self[2].clone() * rhs[0].clone();
    out[1] = c + d;

    let d = self[3].clone() * rhs[2].clone() + self[0].clone() * rhs[1].clone();
    let e = self[1].clone() * rhs[0].clone() - self[2].clone() * rhs[3].clone();
    out[2] = d - e;

    let f = self[3].clone() * rhs[3].clone() - self[0].clone() * rhs[0].clone();
    let g = self[1].clone() * rhs[1].clone() - self[2].clone() * rhs[2].clone();
    out[3] = f - g;

    *self = out
  }
}

impl<'a, T: Default> Mul<&'a T> for &Quaternion<T>
where
  for<'x> &'x T: Mul<&'x T, Output = T>,
{
  type Output = Quaternion<T>;

  fn mul(self, rhs: &'a T) -> Self::Output {
    Quaternion {
      inner: &self.inner * rhs,
    }
  }
}

impl<T> Mul<T> for Quaternion<T>
where
  T: Mul<Output = T> + Default + Clone,
{
  type Output = Self;

  fn mul(self, rhs: T) -> Self {
    Quaternion {
      inner: self.inner * rhs,
    }
  }
}

impl<T> MulAssign<T> for Quaternion<T>
where
  T: Default + Clone + MulAssign,
{
  fn mul_assign(&mut self, rhs: T) {
    self.inner *= rhs;
  }
}

// ================ Neg/Conjugate impls =================//

impl<T: Default> Neg for &Quaternion<T>
where
  for<'x> &'x T: Neg<Output = T>,
{
  type Output = Quaternion<T>;
  fn neg(self) -> Self::Output {
    Quaternion {
      inner: -&self.inner,
    }
  }
}

impl<T: Default> Neg for Quaternion<T>
where
  T: Neg<Output = T> + Clone,
{
  type Output = Quaternion<T>;
  fn neg(self) -> Self::Output {
    Quaternion {
      inner: -self.inner.clone(),
    }
  }
}

// ================ Iterator impls =================//

// ================ Unit tests =================//
