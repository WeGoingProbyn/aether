// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

use std::ops::{
  Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub,
  SubAssign,
};

#[derive(PartialEq, Copy, Clone)]
pub struct Matrix<T: Default + Copy, const C: usize, const R: usize> {
  pub(crate) inner: [[T; C]; R],
}

// ================= Lin Alg functions ======================//

macro_rules! impl_matrix_float {
  ($t:ty) => {
    impl<const C: usize, const R: usize> Matrix<$t, C, R> {
      pub fn transpose(&self) -> Matrix<$t, R, C> {
        let mut result = [[0.0 as $t; R]; C];
        for r in 0..R {
          for c in 0..C {
            result[c][r] = self[r][c];
          }
        }
        result.into()
      }
    }

    // 2x2
    impl Matrix<$t, 2, 2> {
      pub fn determinant(&self) -> $t {
        self[0][0] * self[1][1] - self[0][1] * self[1][0]
      }

      pub fn inverse(&self) -> Option<Matrix<$t, 2, 2>> {
        let det = self.determinant();
        if det.abs() < 1e-10 as $t {
          return None;
        }
        let inv_det = 1.0 as $t / det;
        Some(
          [
            [self[1][1] * inv_det, -self[0][1] * inv_det],
            [-self[1][0] * inv_det, self[0][0] * inv_det],
          ]
          .into(),
        )
      }
    }

    // 3x3
    impl Matrix<$t, 3, 3> {
      pub fn determinant(&self) -> $t {
        let m = |r: usize, c: usize| self[r][c];
        m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1))
          - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0))
          + m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0))
      }

      pub fn inverse(&self) -> Option<Matrix<$t, 3, 3>> {
        let det = self.determinant();
        if det.abs() < 1e-10 as $t {
          return None;
        }
        let inv = 1.0 as $t / det;
        let m = |r: usize, c: usize| self[r][c];
        Some(
          [
            [
              (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) * inv,
              (m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2)) * inv,
              (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) * inv,
            ],
            [
              (m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2)) * inv,
              (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) * inv,
              (m(0, 2) * m(1, 0) - m(0, 0) * m(1, 2)) * inv,
            ],
            [
              (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0)) * inv,
              (m(0, 1) * m(2, 0) - m(0, 0) * m(2, 1)) * inv,
              (m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0)) * inv,
            ],
          ]
          .into(),
        )
      }
    }

    // 4x4
    impl Matrix<$t, 4, 4> {
      pub fn determinant(&self) -> $t {
        let m = |r: usize, c: usize| self[r][c];

        let s0 = m(0, 0) * m(1, 1) - m(1, 0) * m(0, 1);
        let s1 = m(0, 0) * m(1, 2) - m(1, 0) * m(0, 2);
        let s2 = m(0, 0) * m(1, 3) - m(1, 0) * m(0, 3);
        let s3 = m(0, 1) * m(1, 2) - m(1, 1) * m(0, 2);
        let s4 = m(0, 1) * m(1, 3) - m(1, 1) * m(0, 3);
        let s5 = m(0, 2) * m(1, 3) - m(1, 2) * m(0, 3);

        let c5 = m(2, 2) * m(3, 3) - m(3, 2) * m(2, 3);
        let c4 = m(2, 1) * m(3, 3) - m(3, 1) * m(2, 3);
        let c3 = m(2, 1) * m(3, 2) - m(3, 1) * m(2, 2);
        let c2 = m(2, 0) * m(3, 3) - m(3, 0) * m(2, 3);
        let c1 = m(2, 0) * m(3, 2) - m(3, 0) * m(2, 2);
        let c0 = m(2, 0) * m(3, 1) - m(3, 0) * m(2, 1);

        s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0
      }

      pub fn inverse(&self) -> Option<Matrix<$t, 4, 4>> {
        let m = |r: usize, c: usize| self[r][c];

        let s0 = m(0, 0) * m(1, 1) - m(1, 0) * m(0, 1);
        let s1 = m(0, 0) * m(1, 2) - m(1, 0) * m(0, 2);
        let s2 = m(0, 0) * m(1, 3) - m(1, 0) * m(0, 3);
        let s3 = m(0, 1) * m(1, 2) - m(1, 1) * m(0, 2);
        let s4 = m(0, 1) * m(1, 3) - m(1, 1) * m(0, 3);
        let s5 = m(0, 2) * m(1, 3) - m(1, 2) * m(0, 3);

        let c5 = m(2, 2) * m(3, 3) - m(3, 2) * m(2, 3);
        let c4 = m(2, 1) * m(3, 3) - m(3, 1) * m(2, 3);
        let c3 = m(2, 1) * m(3, 2) - m(3, 1) * m(2, 2);
        let c2 = m(2, 0) * m(3, 3) - m(3, 0) * m(2, 3);
        let c1 = m(2, 0) * m(3, 2) - m(3, 0) * m(2, 2);
        let c0 = m(2, 0) * m(3, 1) - m(3, 0) * m(2, 1);

        let det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
        if det.abs() < 1e-10 as $t {
          return None;
        }
        let inv = 1.0 as $t / det;

        Some(
          [
            [
              (m(1, 1) * c5 - m(1, 2) * c4 + m(1, 3) * c3) * inv,
              (-m(0, 1) * c5 + m(0, 2) * c4 - m(0, 3) * c3) * inv,
              (m(3, 1) * s5 - m(3, 2) * s4 + m(3, 3) * s3) * inv,
              (-m(2, 1) * s5 + m(2, 2) * s4 - m(2, 3) * s3) * inv,
            ],
            [
              (-m(1, 0) * c5 + m(1, 2) * c2 - m(1, 3) * c1) * inv,
              (m(0, 0) * c5 - m(0, 2) * c2 + m(0, 3) * c1) * inv,
              (-m(3, 0) * s5 + m(3, 2) * s2 - m(3, 3) * s1) * inv,
              (m(2, 0) * s5 - m(2, 2) * s2 + m(2, 3) * s1) * inv,
            ],
            [
              (m(1, 0) * c4 - m(1, 1) * c2 + m(1, 3) * c0) * inv,
              (-m(0, 0) * c4 + m(0, 1) * c2 - m(0, 3) * c0) * inv,
              (m(3, 0) * s4 - m(3, 1) * s2 + m(3, 3) * s0) * inv,
              (-m(2, 0) * s4 + m(2, 1) * s2 - m(2, 3) * s0) * inv,
            ],
            [
              (-m(1, 0) * c3 + m(1, 1) * c1 - m(1, 2) * c0) * inv,
              (m(0, 0) * c3 - m(0, 1) * c1 + m(0, 2) * c0) * inv,
              (-m(3, 0) * s3 + m(3, 1) * s1 - m(3, 2) * s0) * inv,
              (m(2, 0) * s3 - m(2, 1) * s1 + m(2, 2) * s0) * inv,
            ],
          ]
          .into(),
        )
      }
    }

    impl Matrix<$t, 4, 4> {
      pub fn scale(
        v: &utility::maths::vector::Vector<$t, 3>,
      ) -> Matrix<$t, 4, 4> {
        [
          [v[0], 0.0 as $t, 0.0 as $t, 0.0 as $t],
          [0.0 as $t, v[1], 0.0 as $t, 0.0 as $t],
          [0.0 as $t, 0.0 as $t, v[2], 0.0 as $t],
          [0.0 as $t, 0.0 as $t, 0.0 as $t, 1.0 as $t],
        ]
        .into()
      }

      pub fn translation(v: &[$t; 3]) -> Matrix<$t, 4, 4> {
        [
          [1.0 as $t, 0.0 as $t, 0.0 as $t, v[0]],
          [0.0 as $t, 1.0 as $t, 0.0 as $t, v[1]],
          [0.0 as $t, 0.0 as $t, 1.0 as $t, v[2]],
          [0.0 as $t, 0.0 as $t, 0.0 as $t, 1.0 as $t],
        ]
        .into()
      }

      /// Rotation around an arbitrary axis by `angle` radians (Rodrigues' formula)
      pub fn rotation(
        axis: &utility::maths::vector::Vector<$t, 3>,
        angle: $t,
      ) -> Matrix<$t, 4, 4> {
        let n = axis.normalise();
        let c = angle.cos();
        let s = angle.sin();
        let t = 1.0 as $t - c;
        let x = n[0];
        let y = n[1];
        let z = n[2];
        [
          [
            t * x * x + c,
            t * x * y - s * z,
            t * x * z + s * y,
            0.0 as $t,
          ],
          [
            t * x * y + s * z,
            t * y * y + c,
            t * y * z - s * x,
            0.0 as $t,
          ],
          [
            t * x * z - s * y,
            t * y * z + s * x,
            t * z * z + c,
            0.0 as $t,
          ],
          [0.0 as $t, 0.0 as $t, 0.0 as $t, 1.0 as $t],
        ]
        .into()
      }

      /// Right-handed look-at view matrix
      pub fn look_at(
        eye: &utility::maths::vector::Vector<$t, 3>,
        target: &utility::maths::vector::Vector<$t, 3>,
        up: &utility::maths::vector::Vector<$t, 3>,
      ) -> Matrix<$t, 4, 4> {
        let f = (target - eye).normalise();
        let r = f.cross(up).normalise();
        let u = r.cross(&f);

        [
          [r[0], r[1], r[2], -r.dot(eye)],
          [u[0], u[1], u[2], -u.dot(eye)],
          [-f[0], -f[1], -f[2], f.dot(eye)],
          [0.0 as $t, 0.0 as $t, 0.0 as $t, 1.0 as $t],
        ]
        .into()
      }

      /// Right-handed perspective projection matrix
      /// `fov` is vertical field of view in radians
      pub fn perspective(
        fov: $t,
        aspect: $t,
        near: $t,
        far: $t,
      ) -> Matrix<$t, 4, 4> {
        let f = 1.0 as $t / (fov / 2.0 as $t).tan();
        let nf = near - far;

        [
          [f / aspect, 0.0 as $t, 0.0 as $t, 0.0 as $t],
          [0.0 as $t, f, 0.0 as $t, 0.0 as $t],
          [
            0.0 as $t,
            0.0 as $t,
            (far + near) / nf,
            (2.0 as $t * far * near) / nf,
          ],
          [0.0 as $t, 0.0 as $t, -1.0 as $t, 0.0 as $t],
        ]
        .into()
      }
    }
  };
}

impl_matrix_float!(f32);
impl_matrix_float!(f64);

impl<T, const C: usize, const R: usize> Matrix<T, C, R>
where
  T: Default + Copy,
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
  T: Default + Copy,
{
  fn from(item: [[T; C]; R]) -> Matrix<T, C, R> {
    Matrix { inner: item }
  }
}

impl<T, const C: usize, const R: usize> Default for Matrix<T, C, R>
where
  T: Default + Copy,
{
  fn default() -> Self {
    Matrix {
      inner: std::array::from_fn(|_| std::array::from_fn(|_| T::default())),
    }
  }
}

// ================= Display impls =======================//

impl<T, const C: usize, const R: usize> std::fmt::Display for Matrix<T, C, R>
where
  T: Default + Copy + std::fmt::Display,
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
  T: Default + Copy + std::fmt::Display,
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

impl<T, const C: usize, const R: usize> Index<usize> for Matrix<T, C, R>
where
  T: Default + Copy,
{
  type Output = [T; C];

  fn index(&self, index: usize) -> &[T; C] {
    &self.inner[index]
  }
}

impl<T, const C: usize, const R: usize> IndexMut<usize> for Matrix<T, C, R>
where
  T: Default + Copy,
{
  fn index_mut(&mut self, index: usize) -> &mut [T; C] {
    &mut self.inner[index]
  }
}

// ================= Operator impls =======================//

macro_rules! matrix_op_impl {
  ($trait:ident, $method:ident) => {
    impl<'a, 'b, T, const C: usize, const R: usize> $trait<&'a Matrix<T, C, R>>
      for &'b Matrix<T, C, R>
    where
      for<'x, 'y> &'x T: $trait<&'y T, Output = T>,
      T: Default + Copy,
    {
      type Output = Matrix<T, C, R>;

      fn $method(self, rhs: &'a Matrix<T, C, R>) -> Self::Output {
        Matrix {
          inner: std::array::from_fn(|j| {
            std::array::from_fn(|i| {
              $trait::$method(&self.inner[j][i], &rhs.inner[j][i])
            })
          }),
        }
      }
    }

    impl<'a, T, const C: usize, const R: usize> $trait<&'a Matrix<T, C, R>>
      for Matrix<T, C, R>
    where
      for<'x, 'y> &'x T: $trait<&'y T, Output = T>,
      T: Default + Copy,
    {
      type Output = Matrix<T, C, R>;

      fn $method(self, rhs: &'a Matrix<T, C, R>) -> Self::Output {
        $trait::$method(&self, rhs)
      }
    }

    impl<'a, T, const C: usize, const R: usize> $trait<Matrix<T, C, R>>
      for &'a Matrix<T, C, R>
    where
      for<'x, 'y> &'x T: $trait<&'y T, Output = T>,
      T: Default + Copy,
    {
      type Output = Matrix<T, C, R>;

      fn $method(self, rhs: Matrix<T, C, R>) -> Self::Output {
        $trait::$method(self, &rhs)
      }
    }

    impl<T, const C: usize, const R: usize> $trait<Matrix<T, C, R>>
      for Matrix<T, C, R>
    where
      for<'x, 'y> &'x T: $trait<&'y T, Output = T>,
      T: Default + Copy,
    {
      type Output = Matrix<T, C, R>;

      fn $method(self, rhs: Matrix<T, C, R>) -> Self::Output {
        $trait::$method(&self, &rhs)
      }
    }
  };
}

macro_rules! matrix_op_scalar_impl {
  ($trait:ident, $method:ident) => {
    impl<'a, 'b, T, const C: usize, const R: usize> $trait<&'a T>
      for &'b Matrix<T, C, R>
    where
      for<'x, 'y> &'x T: $trait<&'y T, Output = T>,
      T: Default + Copy,
    {
      type Output = Matrix<T, C, R>;

      fn $method(self, rhs: &'a T) -> Self::Output {
        Matrix {
          inner: std::array::from_fn(|j| {
            std::array::from_fn(|i| $trait::$method(&self.inner[j][i], rhs))
          }),
        }
      }
    }

    impl<'a, T, const C: usize, const R: usize> $trait<&'a T>
      for Matrix<T, C, R>
    where
      for<'x, 'y> &'x T: $trait<&'y T, Output = T>,
      T: Default + Copy,
    {
      type Output = Matrix<T, C, R>;

      fn $method(self, rhs: &'a T) -> Self::Output {
        $trait::$method(&self, rhs)
      }
    }

    impl<'a, T, const C: usize, const R: usize> $trait<T>
      for &'a Matrix<T, C, R>
    where
      for<'x, 'y> &'x T: $trait<&'y T, Output = T>,
      T: Default + Copy,
    {
      type Output = Matrix<T, C, R>;

      fn $method(self, rhs: T) -> Self::Output {
        $trait::$method(self, &rhs)
      }
    }

    impl<T, const C: usize, const R: usize> $trait<T> for Matrix<T, C, R>
    where
      for<'x, 'y> &'x T: $trait<&'y T, Output = T>,
      T: Default + Copy,
    {
      type Output = Matrix<T, C, R>;

      fn $method(self, rhs: T) -> Self::Output {
        $trait::$method(&self, &rhs)
      }
    }
  };
}

macro_rules! matrix_op_assign_impl {
  ($trait:ident, $method:ident) => {
    impl<'a, T, const C: usize, const R: usize> $trait<&'a Matrix<T, C, R>>
      for Matrix<T, C, R>
    where
      T: Default + Copy + $trait,
    {
      fn $method(&mut self, rhs: &'a Matrix<T, C, R>) {
        for j in 0..R {
          for i in 0..C {
            $trait::$method(&mut self.inner[j][i], rhs.inner[j][i])
          }
        }
      }
    }

    impl<T, const C: usize, const R: usize> $trait<Matrix<T, C, R>>
      for Matrix<T, C, R>
    where
      T: Default + Copy + $trait,
    {
      fn $method(&mut self, rhs: Matrix<T, C, R>) {
        $trait::$method(self, &rhs);
      }
    }
  };
}

macro_rules! matrix_op_assign_scalar_impl {
  ($trait:ident, $method:ident) => {
    impl<T, const C: usize, const R: usize> $trait<T> for &mut Matrix<T, C, R>
    where
      T: Default + Copy + $trait,
    {
      fn $method(&mut self, rhs: T) {
        for j in 0..R {
          for i in 0..C {
            $trait::$method(&mut self.inner[j][i], rhs)
          }
        }
      }
    }

    impl<T, const C: usize, const R: usize> $trait<T> for Matrix<T, C, R>
    where
      T: Default + Copy + $trait,
    {
      fn $method(&mut self, rhs: T) {
        for j in 0..R {
          for i in 0..C {
            $trait::$method(&mut self.inner[j][i], rhs)
          }
        }
      }
    }
  };
}

matrix_op_impl!(Add, add);
matrix_op_scalar_impl!(Add, add);
matrix_op_assign_impl!(AddAssign, add_assign);
matrix_op_assign_scalar_impl!(AddAssign, add_assign);

matrix_op_impl!(Sub, sub);
matrix_op_scalar_impl!(Sub, sub);
matrix_op_assign_impl!(SubAssign, sub_assign);
matrix_op_assign_scalar_impl!(SubAssign, sub_assign);

matrix_op_impl!(Div, div);
matrix_op_scalar_impl!(Div, div);
matrix_op_assign_impl!(DivAssign, div_assign);
matrix_op_assign_scalar_impl!(DivAssign, div_assign);

matrix_op_scalar_impl!(Mul, mul);
matrix_op_assign_scalar_impl!(MulAssign, mul_assign);

// ================= Mul impls =======================//

impl<T, const K: usize, const N: usize, const R: usize> Mul<&Matrix<T, R, K>>
  for &Matrix<T, K, N>
where
  for<'x, 'y> &'x T: Mul<&'y T, Output = T> + Add<&'y T, Output = T>,
  T: Default + Copy,
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
  for<'x, 'y> &'x T: Mul<&'y T, Output = T> + Add<&'y T, Output = T>,
  T: Default + Copy,
{
  type Output = Matrix<T, R, N>;

  fn mul(self, rhs: Matrix<T, R, K>) -> Self::Output {
    Mul::mul(&self, &rhs)
  }
}

impl<'a, T, const K: usize, const N: usize, const R: usize>
  Mul<&'a Matrix<T, R, K>> for Matrix<T, K, N>
where
  for<'x, 'y> &'x T: Mul<&'y T, Output = T> + Add<&'y T, Output = T>,
  T: Default + Copy,
{
  type Output = Matrix<T, R, N>;

  fn mul(self, rhs: &'a Matrix<T, R, K>) -> Self::Output {
    Mul::mul(&self, rhs)
  }
}

impl<T, const K: usize, const N: usize, const R: usize> Mul<Matrix<T, R, K>>
  for &Matrix<T, K, N>
where
  for<'x, 'y> &'x T: Mul<&'y T, Output = T> + Add<&'y T, Output = T>,
  T: Default + Copy,
{
  type Output = Matrix<T, R, N>;

  fn mul(self, rhs: Matrix<T, R, K>) -> Self::Output {
    Mul::mul(self, &rhs)
  }
}

// ==================== Neg impl =====================//

impl<T, const C: usize, const R: usize> Neg for &Matrix<T, C, R>
where
  for<'x> &'x T: Neg<Output = T>,
  T: Default + Copy,
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
  for<'x> &'x T: Neg<Output = T>,
  T: Default + Copy,
{
  type Output = Self;
  fn neg(self) -> Self::Output {
    -(&self)
  }
}

// ================= Iterators =======================//

pub struct MatrixIter<'a, T, const C: usize, const R: usize>
where
  T: Default + Copy,
{
  inner: &'a Matrix<T, C, R>,
  row: usize,
  col: usize,
}

impl<'a, T, const C: usize, const R: usize> Iterator for MatrixIter<'a, T, C, R>
where
  T: Default + Copy,
{
  type Item = &'a T;

  fn next(&mut self) -> Option<Self::Item> {
    if self.row >= R {
      return None;
    }

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
  T: Default + Copy,
{
  inner: &'a mut Matrix<T, C, R>,
  row: usize,
  col: usize,
}

impl<'a, T, const C: usize, const R: usize> Iterator
  for MatrixIterMut<'a, T, C, R>
where
  T: Default + Copy,
{
  type Item = &'a mut T;

  fn next(&mut self) -> Option<Self::Item> {
    if self.row >= R {
      return None;
    }

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
  T: Default + Copy,
{
  inner: Matrix<T, C, R>,
  row: usize,
  col: usize,
}

impl<T, const C: usize, const R: usize> Iterator for MatrixIterInto<T, C, R>
where
  T: Default + Copy,
{
  type Item = T;

  fn next(&mut self) -> Option<Self::Item> {
    if self.row >= R {
      return None;
    }

    let next = self.inner[self.row][self.col];
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
  T: Default + Copy,
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
  T: Default + Copy,
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
    assert_eq!(out, 32.0);
  }

  #[test]
  fn check_matrix_mul() {
    let a: Matrix<f32, 3, 2> = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into();

    let b: Matrix<f32, 2, 3> = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]].into();

    let res = a * b;

    let known: Matrix<f32, 2, 2> = [[58.0, 64.0], [139.0, 154.0]].into();

    assert_eq!(res, known);
  }

  fn approx_eq_mat<const C: usize, const R: usize>(
    a: &Matrix<f32, C, R>,
    b: &Matrix<f32, C, R>,
    eps: f32,
  ) -> bool {
    for r in 0..R {
      for c in 0..C {
        if (a[r][c] - b[r][c]).abs() > eps {
          return false;
        }
      }
    }
    true
  }

  #[test]
  fn transpose_2x3() {
    let m: Matrix<f32, 3, 2> = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into();
    let t = m.transpose();
    let expected: Matrix<f32, 2, 3> =
      [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]].into();
    assert_eq!(t, expected);
  }

  #[test]
  fn transpose_roundtrip() {
    let m: Matrix<f32, 3, 3> =
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]].into();
    assert_eq!(m.transpose().transpose(), m);
  }

  #[test]
  fn determinant_2x2() {
    let m: Matrix<f32, 2, 2> = [[3.0, 8.0], [4.0, 6.0]].into();
    let det = m.determinant();
    assert!((det - (-14.0)).abs() < 1e-6);
  }

  #[test]
  fn determinant_3x3() {
    let m: Matrix<f32, 3, 3> =
      [[6.0, 1.0, 1.0], [4.0, -2.0, 5.0], [2.0, 8.0, 7.0]].into();
    let det = m.determinant();
    assert!((det - (-306.0)).abs() < 1e-4);
  }

  #[test]
  fn determinant_4x4() {
    let m: Matrix<f32, 4, 4> = [
      [1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
      [2.0, 6.0, 4.0, 8.0],
      [3.0, 1.0, 1.0, 2.0],
    ]
    .into();
    let det = m.determinant();
    assert!((det - 72.0).abs() < 1e-3);
  }

  #[test]
  fn determinant_identity_is_one() {
    let i3 = Matrix::<f32, 3, 3>::identity(1.0);
    let i4 = Matrix::<f32, 4, 4>::identity(1.0);
    assert!((i3.determinant() - 1.0).abs() < 1e-6);
    assert!((i4.determinant() - 1.0).abs() < 1e-6);
  }

  #[test]
  fn determinant_singular_is_zero() {
    let m: Matrix<f32, 3, 3> =
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]].into();
    assert!(m.determinant().abs() < 1e-4);
  }

  #[test]
  fn inverse_2x2() {
    let m: Matrix<f32, 2, 2> = [[4.0, 7.0], [2.0, 6.0]].into();
    let inv = m.inverse().unwrap();
    let identity = Matrix::<f32, 2, 2>::identity(1.0);
    let product = m * inv;
    assert!(approx_eq_mat(&product, &identity, 1e-5));
  }

  #[test]
  fn inverse_3x3() {
    let m: Matrix<f32, 3, 3> =
      [[3.0, 0.0, 2.0], [2.0, 0.0, -2.0], [0.0, 1.0, 1.0]].into();
    let inv = m.inverse().unwrap();
    let identity = Matrix::<f32, 3, 3>::identity(1.0);
    let product = m * inv;
    assert!(approx_eq_mat(&product, &identity, 1e-5));
  }

  #[test]
  fn inverse_4x4() {
    let m: Matrix<f32, 4, 4> = [
      [1.0, 1.0, 1.0, -1.0],
      [1.0, 1.0, -1.0, 1.0],
      [1.0, -1.0, 1.0, 1.0],
      [-1.0, 1.0, 1.0, 1.0],
    ]
    .into();
    let inv = m.inverse().unwrap();
    let identity = Matrix::<f32, 4, 4>::identity(1.0);
    let product = m * inv;
    assert!(approx_eq_mat(&product, &identity, 1e-5));
  }

  #[test]
  fn inverse_singular_returns_none() {
    let m: Matrix<f32, 3, 3> =
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]].into();
    assert!(m.inverse().is_none());
  }

  #[test]
  fn inverse_identity_is_identity() {
    let i = Matrix::<f32, 4, 4>::identity(1.0);
    let inv = i.inverse().unwrap();
    assert!(approx_eq_mat(&i, &inv, 1e-6));
  }

  #[test]
  fn scale_matrix() {
    use crate::maths::vector::Vector;
    let s: Vector<f32, 3> = [2.0, 3.0, 4.0].into();
    let m = Matrix::<f32, 4, 4>::scale(&s);
    assert!((m[0][0] - 2.0).abs() < 1e-6);
    assert!((m[1][1] - 3.0).abs() < 1e-6);
    assert!((m[2][2] - 4.0).abs() < 1e-6);
    assert!((m[3][3] - 1.0).abs() < 1e-6);
  }

  #[test]
  fn translation_matrix() {
    let m = Matrix::<f32, 4, 4>::translation(&[5.0, 6.0, 7.0]);
    assert!((m[0][3] - 5.0).abs() < 1e-6);
    assert!((m[1][3] - 6.0).abs() < 1e-6);
    assert!((m[2][3] - 7.0).abs() < 1e-6);
    // diagonal is 1
    assert!((m[0][0] - 1.0).abs() < 1e-6);
    assert!((m[1][1] - 1.0).abs() < 1e-6);
    assert!((m[2][2] - 1.0).abs() < 1e-6);
    assert!((m[3][3] - 1.0).abs() < 1e-6);
  }

  #[test]
  fn rotation_90_degrees_around_z() {
    use crate::maths::vector::Vector;
    let axis: Vector<f32, 3> = [0.0, 0.0, 1.0].into();
    let angle = std::f32::consts::FRAC_PI_2;
    let m = Matrix::<f32, 4, 4>::rotation(&axis, angle);
    // Rotating (1,0,0) around Z by 90° gives (0,1,0)
    // m * [1,0,0] = column 0 of rotation part
    assert!((m[0][0] - 0.0).abs() < 1e-5); // cos(90) ≈ 0
    assert!((m[1][0] - 1.0).abs() < 1e-5); // sin(90) ≈ 1
    assert!((m[2][0] - 0.0).abs() < 1e-5);
  }

  #[test]
  fn perspective_basic() {
    let fov = std::f32::consts::FRAC_PI_2; // 90°
    let m = Matrix::<f32, 4, 4>::perspective(fov, 1.0, 0.1, 100.0);
    // f = 1/tan(45°) = 1.0, aspect = 1.0, so m[0][0] = 1.0
    assert!((m[0][0] - 1.0).abs() < 1e-5);
    assert!((m[1][1] - 1.0).abs() < 1e-5);
    assert!((m[3][2] - (-1.0)).abs() < 1e-5);
  }
}
