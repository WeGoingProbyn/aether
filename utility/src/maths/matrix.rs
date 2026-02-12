use std::ops::{
  Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign
};

#[derive(PartialEq, Clone)]
pub struct Matrix<T: Default, const C: usize, const R: usize> {
  pub(crate) inner: [[T; C]; R],
}

// ================= Lin Alg functions ======================//

impl<T, const C: usize, const R: usize> Matrix<T, C, R>
where
  T: Default + Clone + AddAssign + Mul<Output = T>,
{
  pub fn dot_clone(&self, rhs: &Matrix<T, C, R>) -> T {
    let mut res: T = T::default();
    for j in 0..R {
      for i in 0..C {
        res += self.inner[j][i].clone() * rhs.inner[j][i].clone();
      }
    }
    res
  }
}

impl<T, const C: usize, const R: usize> Matrix<T, C, R>
where
  T: Default,
  for<'x> &'x T: Mul<&'x T, Output = T> + Add<&'x T, Output = T>,
{
  pub fn dot(&self, rhs: &Matrix<T, C, R>) -> T {
    let mut res = T::default();
    for j in 0..R {
      for i in 0..C {
        let prod = &self.inner[j][i] * &rhs.inner[j][i];
        res = &res + &prod;
      }
    }
    res
  }
}

// ================= Constructors =======================//

impl<T, const N: usize> Matrix<T, N, N>
where
  T: Default + Copy,
{
  #[allow(unused)]
  pub fn identity(identity: T) -> Matrix<T, N, N> {
    Matrix {
      inner: std::array::from_fn(|j| {
        std::array::from_fn(|i| if i == j { identity } else { T::default() })
      }),
    }
  }
}

impl<T, const C: usize, const R: usize> From<[[T; C]; R]> for Matrix<T, C, R>
where
  T: Default,
{
  fn from(item: [[T; C]; R]) -> Matrix<T, C, R> {
    Matrix { inner: item }
  }
}

impl<T: Default, const C: usize, const R: usize> Default for Matrix<T, C, R> {
  fn default() -> Self {
    Matrix {
      inner: std::array::from_fn(|_| std::array::from_fn(|_| T::default())),
    }
  }
}

// ================= Display impls =======================//

impl<T, const C: usize, const R: usize> std::fmt::Display for Matrix<T, C, R>
where
  T: Default + std::fmt::Display,
{
  fn fmt(
    &self,
    f: &mut std::fmt::Formatter<'_>,
  ) -> Result<(), std::fmt::Error> {
    for (j, row) in self.inner.iter().enumerate() {
      write!(f, "[")?;
      for (i, _) in row.iter().enumerate() {
        write!(f, "{}", self.inner[j][i])?;
        if i < row.len() - 1 {
          write!(f, ",")?;
        }
      }
      writeln!(f, "]")?;
    }
    Ok(())
  }
}

impl<T, const C: usize, const R: usize> std::fmt::Debug for Matrix<T, C, R>
where
  T: Default + std::fmt::Display,
{
  fn fmt(
    &self,
    f: &mut std::fmt::Formatter<'_>,
  ) -> Result<(), std::fmt::Error> {
    writeln!(
      f,
      "Matrix<T = {}, C = {}, R = {}>:",
      std::any::type_name::<T>(),
      C,
      R
    )?;
    for (j, row) in self.inner.iter().enumerate() {
      write!(f, "[")?;
      for (i, _) in row.iter().enumerate() {
        write!(f, "{}", self.inner[j][i])?;
        if i < row.len() - 1 {
          write!(f, ",")?;
        }
      }
      writeln!(f, "]")?;
    }
    Ok(())
  }
}

// ================= Index impls =======================//

impl<T: Default, const C: usize, const R: usize> Index<usize>
  for Matrix<T, C, R>
{
  type Output = [T; C];

  fn index(&self, index: usize) -> &[T; C] {
    &self.inner[index]
  }
}

impl<T: Default, const C: usize, const R: usize> IndexMut<usize>
  for Matrix<T, C, R>
{
  fn index_mut(&mut self, index: usize) -> &mut [T; C] {
    &mut self.inner[index]
  }
}

// ================= Add impls =======================//

impl<'a, T: Default, const C: usize, const R: usize> Add<&'a Matrix<T, C, R>>
  for &Matrix<T, C, R>
where
  for<'x> &'x T: Add<&'x T, Output = T>,
{
  type Output = Matrix<T, C, R>;

  fn add(self, rhs: &'a Matrix<T, C, R>) -> Self::Output {
    Matrix {
      inner: std::array::from_fn(|j| {
        std::array::from_fn(|i| &self.inner[j][i] + &rhs.inner[j][i])
      }),
    }
  }
}

impl<T, const C: usize, const R: usize> Add for Matrix<T, C, R>
where
  T: Add<Output = T> + Default + Clone,
{
  type Output = Self;

  fn add(self, rhs: Self) -> Self {
    Matrix {
      inner: std::array::from_fn(|j| {
        std::array::from_fn(|i| {
          self.inner[j][i].clone() + rhs.inner[j][i].clone()
        })
      }),
    }
  }
}

impl<T, const C: usize, const R: usize> AddAssign for Matrix<T, C, R>
where
  T: Default + Clone + AddAssign,
{
  fn add_assign(&mut self, rhs: Self) {
    for j in 0..R {
      for i in 0..C {
        self.inner[j][i] += rhs.inner[j][i].clone();
      }
    }
  }
}

impl<'a, T: Default, const C: usize, const R: usize> Add<&'a T>
  for &Matrix<T, C, R>
where
  for<'x> &'x T: Add<&'x T, Output = T>,
{
  type Output = Matrix<T, C, R>;

  fn add(self, rhs: &'a T) -> Self::Output {
    Matrix {
      inner: std::array::from_fn(|j| {
        std::array::from_fn(|i| &self.inner[j][i] + rhs)
      }),
    }
  }
}

impl<T, const C: usize, const R: usize> Add<T> for Matrix<T, C, R>
where
  T: Add<Output = T> + Default + Clone,
{
  type Output = Self;

  fn add(self, rhs: T) -> Self {
    Matrix {
      inner: std::array::from_fn(|j| {
        std::array::from_fn(|i| {
          self.inner[j][i].clone() + rhs.clone()
        })
      }),
    }
  }
}

impl<T, const C: usize, const R: usize> AddAssign<T> for Matrix<T, C, R>
where
  T: Default + Clone + AddAssign,
{
  fn add_assign(&mut self, rhs: T) {
    for j in 0..R {
      for i in 0..C {
        self.inner[j][i] += rhs.clone();
      }
    }
  }
}

// ================= Sub impls =======================//

impl<'a, T: Default, const C: usize, const R: usize> Sub<&'a Matrix<T, C, R>>
  for &Matrix<T, C, R>
where
  for<'x> &'x T: Sub<&'x T, Output = T>,
{
  type Output = Matrix<T, C, R>;

  fn sub(self, rhs: &'a Matrix<T, C, R>) -> Self::Output {
    Matrix {
      inner: std::array::from_fn(|j| {
        std::array::from_fn(|i| &self.inner[j][i] - &rhs.inner[j][i])
      }),
    }
  }
}

impl<T, const C: usize, const R: usize> Sub for Matrix<T, C, R>
where
  T: Sub<Output = T> + Default + Clone,
{
  type Output = Self;

  fn sub(self, rhs: Self) -> Self {
    Matrix {
      inner: std::array::from_fn(|j| {
        std::array::from_fn(|i| {
          self.inner[j][i].clone() - rhs.inner[j][i].clone()
        })
      }),
    }
  }
}

impl<T, const C: usize, const R: usize> SubAssign for Matrix<T, C, R>
where
  T: Default + Clone + SubAssign,
{
  fn sub_assign(&mut self, rhs: Self) {
    for j in 0..R {
      for i in 0..C {
        self.inner[j][i] -= rhs.inner[j][i].clone();
      }
    }
  }
}

impl<'a, T: Default, const C: usize, const R: usize> Sub<&'a T>
  for &Matrix<T, C, R>
where
  for<'x> &'x T: Sub<&'x T, Output = T>,
{
  type Output = Matrix<T, C, R>;

  fn sub(self, rhs: &'a T) -> Self::Output {
    Matrix {
      inner: std::array::from_fn(|j| {
        std::array::from_fn(|i| &self.inner[j][i] - rhs)
      }),
    }
  }
}

impl<T, const C: usize, const R: usize> Sub<T> for Matrix<T, C, R>
where
  T: Sub<Output = T> + Default + Clone,
{
  type Output = Self;

  fn sub(self, rhs: T) -> Self {
    Matrix {
      inner: std::array::from_fn(|j| {
        std::array::from_fn(|i| {
          self.inner[j][i].clone() - rhs.clone()
        })
      }),
    }
  }
}

impl<T, const C: usize, const R: usize> SubAssign<T> for Matrix<T, C, R>
where
  T: Default + Clone + SubAssign,
{
  fn sub_assign(&mut self, rhs: T) {
    for j in 0..R {
      for i in 0..C {
        self.inner[j][i] -= rhs.clone();
      }
    }
  }
}

// ================= Div impls =======================//

impl<'a, T: Default, const C: usize, const R: usize> Div<&'a Matrix<T, C, R>>
  for &Matrix<T, C, R>
where
  for<'x> &'x T: Div<&'x T, Output = T>,
{
  type Output = Matrix<T, C, R>;

  fn div(self, rhs: &'a Matrix<T, C, R>) -> Self::Output {
    Matrix {
      inner: std::array::from_fn(|j| {
        std::array::from_fn(|i| &self.inner[j][i] / &rhs.inner[j][i])
      }),
    }
  }
}

impl<T, const C: usize, const R: usize> Div for Matrix<T, C, R>
where
  T: Div<Output = T> + Default + Clone,
{
  type Output = Self;

  fn div(self, rhs: Self) -> Self {
    Matrix {
      inner: std::array::from_fn(|j| {
        std::array::from_fn(|i| {
          self.inner[j][i].clone() / rhs.inner[j][i].clone()
        })
      }),
    }
  }
}

impl<T, const C: usize, const R: usize> DivAssign for Matrix<T, C, R>
where
  T: Default + Clone + DivAssign,
{
  fn div_assign(&mut self, rhs: Self) {
    for j in 0..R {
      for i in 0..C {
        self.inner[j][i] /= rhs.inner[j][i].clone();
      }
    }
  }
}

impl<'a, T: Default, const C: usize, const R: usize> Div<&'a T>
  for &Matrix<T, C, R>
where
  for<'x> &'x T: Div<&'x T, Output = T>,
{
  type Output = Matrix<T, C, R>;

  fn div(self, rhs: &'a T) -> Self::Output {
    Matrix {
      inner: std::array::from_fn(|j| {
        std::array::from_fn(|i| &self.inner[j][i] / rhs)
      }),
    }
  }
}

impl<T, const C: usize, const R: usize> Div<T> for Matrix<T, C, R>
where
  T: Div<Output = T> + Default + Clone,
{
  type Output = Self;

  fn div(self, rhs: T) -> Self {
    Matrix {
      inner: std::array::from_fn(|j| {
        std::array::from_fn(|i| {
          self.inner[j][i].clone() / rhs.clone()
        })
      }),
    }
  }
}

impl<T, const C: usize, const R: usize> DivAssign<T> for Matrix<T, C, R>
where
  T: Default + Clone + DivAssign,
{
  fn div_assign(&mut self, rhs: T) {
    for j in 0..R {
      for i in 0..C {
        self.inner[j][i] /= rhs.clone();
      }
    }
  }
}

// ================= Mul impls =======================//

impl<T, const K: usize, const N: usize, const R: usize> Mul<&Matrix<T, R, K>>
  for &Matrix<T, K, N>
where
  T: Default,
  for<'x> &'x T: Mul<&'x T, Output = T> + Add<&'x T, Output = T>,
{
  type Output = Matrix<T, R, N>;

  fn mul(self, rhs: &Matrix<T, R, K>) -> Self::Output {
    let mut out = Matrix::<T, R, N>::default();

    for j in 0..N {
      for i in 0..R {
        let mut acc = T::default();
        for k in 0..K {
          let prod = &self.inner[j][k] * &rhs.inner[k][i];
          acc = &acc + &prod;
        }
        out.inner[j][i] = acc;
      }
    }
    out
  }
}

impl<T, const K: usize, const N: usize, const R: usize> Mul<Matrix<T, R, K>>
  for Matrix<T, K, N>
where
  T: Mul<Output = T> + Add<Output = T> + Default + Clone,
{
  type Output = Matrix<T, R, N>;

  fn mul(self, rhs: Matrix<T, R, K>) -> Self::Output {
    let mut out = Matrix::<T, R, N>::default();

    for j in 0..N {
      for i in 0..R {
        let mut acc = T::default();
        for k in 0..K {
          let prod = self.inner[j][k].clone() * rhs.inner[k][i].clone();
          acc = acc + prod;
        }
        out.inner[j][i] = acc;
      }
    }
    out
  }
}

impl<T, const K: usize, const N: usize, const R: usize> MulAssign<Matrix<T, R, K>>
  for Matrix<T, K, N>
where
  T: Mul<Output = T> + Add<Output = T> + Default + Clone,
{
  fn mul_assign(&mut self, rhs: Matrix<T, R, K>) {
    for j in 0..N {
      for i in 0..R {
        let mut acc = T::default();
        for k in 0..K {
          let prod = self.inner[j][k].clone() * rhs.inner[k][i].clone();
          acc = acc + prod;
        }
        self.inner[j][i] = acc;
      }
    }
  }
}

impl<'a, T: Default, const C: usize, const R: usize> Mul<&'a T>
  for &Matrix<T, C, R>
where
  for<'x> &'x T: Mul<&'x T, Output = T>,
{
  type Output = Matrix<T, C, R>;

  fn mul(self, rhs: &'a T) -> Self::Output {
    Matrix {
      inner: std::array::from_fn(|j| {
        std::array::from_fn(|i| &self.inner[j][i] * rhs)
      }),
    }
  }
}

impl<T, const C: usize, const R: usize> Mul<T> for Matrix<T, C, R>
where
  T: Mul<Output = T> + Default + Clone,
{
  type Output = Self;

  fn mul(self, rhs: T) -> Self {
    Matrix {
      inner: std::array::from_fn(|j| {
        std::array::from_fn(|i| {
          self.inner[j][i].clone() * rhs.clone()
        })
      }),
    }
  }
}

impl<T, const C: usize, const R: usize> MulAssign<T> for Matrix<T, C, R>
where
  T: Default + Clone + MulAssign,
{
  fn mul_assign(&mut self, rhs: T) {
    for j in 0..R {
      for i in 0..C {
        self.inner[j][i] *= rhs.clone();
      }
    }
  }
}

// ==================== Neg impl =====================//

impl<T, const C: usize, const R: usize> Neg for &Matrix<T, C, R> 
where 
  T: Default,
  for<'x> &'x T: Neg<Output = T>,
{
  type Output = Matrix<T, C, R>;

  fn neg(self) -> Self::Output {
    let mut out = Matrix::default();
    for j in 0..R {
      for i in 0..C {
        out[j][i] = -&self[j][i];
      }
    }
    out
  }
}

impl<T, const C: usize, const R: usize> Neg for Matrix<T, C, R> 
where 
  T: Default + Clone + Neg<Output = T>,
{
  type Output = Self;
  fn neg(self) -> Self::Output {
    let mut out = Matrix::default();
    for j in 0..R {
      for i in 0..C {
        out[j][i] = -self[j][i].clone();
      }
    }
    out
  }
}

// ================= Iterators =======================//

pub struct MatrixIter<'a, T, const C: usize, const R: usize> 
where 
  T: Default,
{
  inner: &'a Matrix<T, C, R>,
  row: usize,
  col: usize,
}

impl<'a, T, const C: usize, const R: usize> Iterator for MatrixIter<'a, T, C, R> 
where 
  T: Default,
{
  type Item = &'a T;

  fn next(&mut self) -> Option<Self::Item> {
    if self.row >= R { return None }

    let next = &self.inner[self.row][self.col];
    self.col += 1;

    if self.col >= C {
      self.col = 0;
      self.row += 1;
    }

    Some(next)
  }
}

pub struct MatrixIterMut<'a, T, const C: usize, const R: usize> 
where 
  T: Default,
{
  inner: &'a mut Matrix<T, C, R>,
  row: usize,
  col: usize,
}

impl<'a, T, const C: usize, const R: usize> Iterator for MatrixIterMut<'a, T, C, R> 
where 
  T: Default,
{
  type Item = &'a mut T;

  fn next(&mut self) -> Option<Self::Item> {
    if self.row >= R { return None }

    let next = &mut self.inner[self.row][self.col] as *mut T;
    self.col += 1;

    if self.col >= C {
      self.col = 0;
      self.row += 1;
    }

    // This is safe as long as we never
    // hand out the same mut reference twice!
    Some(unsafe { &mut *next })
  }
}

pub struct MatrixIterInto<T, const C: usize, const R: usize> 
where 
  T: Default,
{
  inner: Matrix<T, C, R>,
  row: usize,
  col: usize,
}

impl<T, const C: usize, const R: usize> Iterator for MatrixIterInto<T, C, R> 
where 
  T: Default + Clone,
{
  type Item = T;

  fn next(&mut self) -> Option<Self::Item> {
    if self.row >= R { return None }

    let next = self.inner[self.row][self.col].clone();
    self.col += 1;

    if self.col >= C {
      self.col = 0;
      self.row += 1;
    }

    Some(next)
  }
}

impl<T, const C: usize, const R: usize> Matrix<T, C, R> 
where 
  T: Default,
{
  pub fn iter(&self) -> MatrixIter<'_, T, C, R> {
    MatrixIter { 
      inner: self, 
      row: 0, 
      col: 0, 
    }
  }

  pub fn iter_mut(&mut self) -> MatrixIterMut<'_, T, C, R> {
    MatrixIterMut { 
      inner: self, 
      row: 0, 
      col: 0, 
    }
  }
}

impl<T, const C: usize, const R: usize> IntoIterator for Matrix<T, C, R> 
where 
  T: Default + Clone,
{
  type Item = T;
  type IntoIter = MatrixIterInto<T, C, R>;

  fn into_iter(self) -> Self::IntoIter {
    MatrixIterInto {
      inner: self,
      row: 0,
      col: 0,
    }
  }
}

// impl<'a, T, const C: usize, const R: usize> Sum<&'a Matrix<T, C, R>> for Matrix<T, C, R>
// where 
//   T: Default,
//   for<'x> &'x T: Add<&'x T, Output = T>,
// {
//   fn sum<I>(iter: I) -> Self
// where
//     I: Iterator<Item = &'a Matrix<T, C, R>>
//   {
//     iter.fold(Matrix::default(), |acc, v| &acc + v)
//   }
// }
//
// impl<T, const C: usize, const R: usize> Sum for Matrix<T, C, R>
// where
//   T: Default + Clone + Add<Output = T>,
// {
//   fn sum<I>(iter: I) -> Self
// where
//     I: Iterator<Item = Self>,
//   {
//     iter.fold(Matrix::default(), |acc, v| acc + v)
//   }
// }

// ======================= Unit tests ===========================//

#[cfg(test)]
mod test {
  use crate::maths::matrix::Matrix;

  #[test]
  fn check_dot_product() {
    let mut a = Matrix::<f32, 3, 1>::default();
    let mut b = Matrix::<f32, 3, 1>::default();

    a[0][0] = 1.0;
    a[0][1] = 2.0;
    a[0][2] = 3.0;

    b[0][0] = 4.0;
    b[0][1] = 5.0;
    b[0][2] = 6.0;

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    let out = a.dot(&b);
    let out_clone = a.dot_clone(&b);
    assert_eq!(out, 32.0);
    assert_eq!(out_clone, 32.0);
  }

  #[test]
  fn check_matrix_mul() {
    let a: Matrix<f32, 3, 2> = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into();

    let b: Matrix<f32, 2, 3> = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]].into();

    let res = &a * &b;
    let res_clone = a * b;

    let known: Matrix<f32, 2, 2> = [[58.0, 64.0], [139.0, 154.0]].into();

    assert_eq!(res, known);
    assert_eq!(res_clone, known);
  }
}
