use std::ops::{
  Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub,
  SubAssign,
};

use crate::maths::matrix::Matrix;

#[derive(PartialEq, Clone)]
pub struct Vector<T: Default, const C: usize> {
  inner: Matrix<T, C, 1>,
}

// ==================Lin alg functions=====================//

macro_rules! impl_vector_float {
  ($t:ty) => {
    impl<const C: usize> Vector<$t, C>
    where
      for<'x> &'x $t: Mul<&'x $t, Output = $t> + Add<&'x $t, Output = $t>,
    {
      pub fn lerp(&self, v1: &Vector<$t, C>, t: $t) -> Vector<$t, C> {
        self * &(1.0 as $t - t) + v1 * &t
      }

      pub fn distance(&self, other: &Vector<$t, C>) -> $t {
        (self - other).magnitude()
      }

      pub fn magnitude(&self) -> $t {
        self.inner.dot(&self.inner).sqrt()
      }

      pub fn normalise(&self) -> Vector<$t, C> {
        self / &self.magnitude()
      }

      pub fn project(&self, other: &Vector<$t, C>) -> Vector<$t, C> {
        other * &(self.dot(other) / other.dot(other))
      }

      pub fn reflect(&self, normal: &Vector<$t, C>) -> Vector<$t, C> {
        let d = self.dot(normal);
        self - &(normal * &(2.0 as $t * d))
      }
    }
  };
}

impl_vector_float!(f32);
impl_vector_float!(f64);

impl<T, const C: usize> Vector<T, C>
where
  T: Default + Clone + AddAssign + Mul<Output = T>,
{
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
    if self.col >= C {
      return None;
    }

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
    if self.col >= C {
      return None;
    }

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
    VectorIter { inner: &self.inner }
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

impl<T, const C: usize> From<&Vector<T, C>> for Vector<T, C>
where
  T: Default + Clone,
{
  fn from(item: &Vector<T, C>) -> Vector<T, C> {
    Vector {
      inner: item.inner.clone(),
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

impl<'a, T: Default, const C: usize> Add<&'a T> for &Vector<T, C>
where
  for<'x> &'x T: Add<&'x T, Output = T>,
{
  type Output = Vector<T, C>;

  fn add(self, rhs: &'a T) -> Self::Output {
    Vector {
      inner: &self.inner + rhs,
    }
  }
}

impl<T, const C: usize> Add<T> for Vector<T, C>
where
  T: Add<Output = T> + Default + Clone,
{
  type Output = Self;

  fn add(self, rhs: T) -> Self {
    Vector {
      inner: self.inner + rhs,
    }
  }
}

impl<T, const C: usize> AddAssign<T> for Vector<T, C>
where
  T: Default + Clone + AddAssign,
{
  fn add_assign(&mut self, rhs: T) {
    self.inner += rhs;
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

impl<'a, T: Default, const C: usize> Sub<&'a T> for &Vector<T, C>
where
  for<'x> &'x T: Sub<&'x T, Output = T>,
{
  type Output = Vector<T, C>;

  fn sub(self, rhs: &'a T) -> Self::Output {
    Vector {
      inner: &self.inner - rhs,
    }
  }
}

impl<T, const C: usize> Sub<T> for Vector<T, C>
where
  T: Sub<Output = T> + Default + Clone,
{
  type Output = Self;

  fn sub(self, rhs: T) -> Self {
    Vector {
      inner: self.inner - rhs,
    }
  }
}

impl<T, const C: usize> SubAssign<T> for Vector<T, C>
where
  T: Default + Clone + SubAssign,
{
  fn sub_assign(&mut self, rhs: T) {
    self.inner -= rhs;
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

impl<T, const K: usize, const N: usize> MulAssign<Matrix<T, N, K>>
  for Vector<T, K>
where
  T: Default + Clone + Mul<T, Output = T> + Add<T, Output = T>,
{
  fn mul_assign(&mut self, rhs: Matrix<T, N, K>) {
    self.inner *= rhs
  }
}

impl<'a, T: Default, const C: usize> Mul<&'a T> for &Vector<T, C>
where
  for<'x> &'x T: Mul<&'x T, Output = T>,
{
  type Output = Vector<T, C>;

  fn mul(self, rhs: &'a T) -> Self::Output {
    Vector {
      inner: &self.inner * rhs,
    }
  }
}

impl<T, const C: usize> Mul<T> for Vector<T, C>
where
  T: Mul<Output = T> + Default + Clone,
{
  type Output = Self;

  fn mul(self, rhs: T) -> Self {
    Vector {
      inner: self.inner * rhs,
    }
  }
}

impl<T, const C: usize> MulAssign<T> for Vector<T, C>
where
  T: Default + Clone + MulAssign,
{
  fn mul_assign(&mut self, rhs: T) {
    self.inner *= rhs;
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

  fn approx_eq_vec<const C: usize>(
    a: &Vector<f32, C>,
    b: &Vector<f32, C>,
    eps: f32,
  ) -> bool {
    for i in 0..C {
      if (a[i] - b[i]).abs() > eps {
        return false;
      }
    }
    true
  }

  #[test]
  fn magnitude_unit_vector() {
    let v: Vector<f32, 3> = [1.0, 0.0, 0.0].into();
    assert!((v.magnitude() - 1.0).abs() < 1e-6);
  }

  #[test]
  fn magnitude_345() {
    let v: Vector<f32, 3> = [3.0, 4.0, 0.0].into();
    assert!((v.magnitude() - 5.0).abs() < 1e-6);
  }

  #[test]
  fn normalise_preserves_direction() {
    let v: Vector<f32, 3> = [3.0, 4.0, 0.0].into();
    let n = v.normalise();
    assert!((n.magnitude() - 1.0).abs() < 1e-6);
    assert!((n[0] - 0.6).abs() < 1e-6);
    assert!((n[1] - 0.8).abs() < 1e-6);
  }

  #[test]
  fn lerp_endpoints() {
    let a: Vector<f32, 3> = [0.0, 0.0, 0.0].into();
    let b: Vector<f32, 3> = [10.0, 20.0, 30.0].into();
    assert!(approx_eq_vec(&a.lerp(&b, 0.0), &a, 1e-6));
    assert!(approx_eq_vec(&a.lerp(&b, 1.0), &b, 1e-6));
  }

  #[test]
  fn lerp_midpoint() {
    let a: Vector<f32, 3> = [0.0, 0.0, 0.0].into();
    let b: Vector<f32, 3> = [10.0, 20.0, 30.0].into();
    let mid = a.lerp(&b, 0.5);
    let expected: Vector<f32, 3> = [5.0, 10.0, 15.0].into();
    assert!(approx_eq_vec(&mid, &expected, 1e-6));
  }

  #[test]
  fn distance_same_point() {
    let a: Vector<f32, 3> = [1.0, 2.0, 3.0].into();
    assert!(a.distance(&a) < 1e-6);
  }

  #[test]
  fn distance_known() {
    let a: Vector<f32, 3> = [0.0, 0.0, 0.0].into();
    let b: Vector<f32, 3> = [3.0, 4.0, 0.0].into();
    assert!((a.distance(&b) - 5.0).abs() < 1e-6);
  }

  #[test]
  fn project_onto_axis() {
    let v: Vector<f32, 3> = [3.0, 4.0, 0.0].into();
    let x_axis: Vector<f32, 3> = [1.0, 0.0, 0.0].into();
    let proj = v.project(&x_axis);
    let expected: Vector<f32, 3> = [3.0, 0.0, 0.0].into();
    assert!(approx_eq_vec(&proj, &expected, 1e-6));
  }

  #[test]
  fn project_parallel() {
    let v: Vector<f32, 3> = [5.0, 0.0, 0.0].into();
    let dir: Vector<f32, 3> = [2.0, 0.0, 0.0].into();
    let proj = v.project(&dir);
    assert!(approx_eq_vec(&proj, &v, 1e-6));
  }

  #[test]
  fn reflect_off_horizontal() {
    // ray going down-right, reflecting off horizontal surface (normal = up)
    let v: Vector<f32, 3> = [1.0, -1.0, 0.0].into();
    let normal: Vector<f32, 3> = [0.0, 1.0, 0.0].into();
    let r = v.reflect(&normal);
    let expected: Vector<f32, 3> = [1.0, 1.0, 0.0].into();
    assert!(approx_eq_vec(&r, &expected, 1e-6));
  }

  #[test]
  fn reflect_perpendicular() {
    // ray straight into surface reverses
    let v: Vector<f32, 3> = [0.0, -1.0, 0.0].into();
    let normal: Vector<f32, 3> = [0.0, 1.0, 0.0].into();
    let r = v.reflect(&normal);
    let expected: Vector<f32, 3> = [0.0, 1.0, 0.0].into();
    assert!(approx_eq_vec(&r, &expected, 1e-6));
  }

  #[test]
  fn dot_orthogonal_is_zero() {
    let a: Vector<f32, 3> = [1.0, 0.0, 0.0].into();
    let b: Vector<f32, 3> = [0.0, 1.0, 0.0].into();
    assert!(a.dot(&b).abs() < 1e-6);
  }

  #[test]
  fn dot_parallel() {
    let a: Vector<f32, 3> = [2.0, 3.0, 4.0].into();
    assert!((a.dot(&a) - a.magnitude() * a.magnitude()).abs() < 1e-4);
  }
}
