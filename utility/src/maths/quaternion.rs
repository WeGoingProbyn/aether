use std::ops::{
  Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub,
  SubAssign,
};

use crate::maths::{matrix::Matrix, vector::Vector};

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

macro_rules! impl_quaternion_float {
  ($t:ty) => {
    impl Quaternion<$t> {
      /// Create quaternion from axis (will be normalised) and angle in radians
      pub fn from_axis_angle(
        axis: &Vector<$t, 3>,
        angle: $t,
      ) -> Quaternion<$t> {
        let half = angle / 2.0 as $t;
        let s = half.sin();
        let n = axis.normalise();
        [n[0] * s, n[1] * s, n[2] * s, half.cos()].into()
      }

      /// Create quaternion from euler angles (radians)
      /// pitch = X, yaw = Y, roll = Z, applied as Z * Y * X
      pub fn from_euler(pitch: $t, yaw: $t, roll: $t) -> Quaternion<$t> {
        let hp = pitch / 2.0 as $t;
        let hy = yaw / 2.0 as $t;
        let hr = roll / 2.0 as $t;

        let (sp, cp) = (hp.sin(), hp.cos());
        let (sy, cy) = (hy.sin(), hy.cos());
        let (sr, cr) = (hr.sin(), hr.cos());

        [
          cy * sp * cr + sy * cp * sr,
          sy * cp * cr - cy * sp * sr,
          cy * cp * sr - sy * sp * cr,
          cy * cp * cr + sy * sp * sr,
        ]
        .into()
      }

      /// Extract rotation from a 4x4 matrix (Shepperd's method)
      pub fn from_matrix(m: &Matrix<$t, 4, 4>) -> Quaternion<$t> {
        let trace = m[0][0] + m[1][1] + m[2][2];

        if trace > 0.0 as $t {
          let s = (trace + 1.0 as $t).sqrt() * 2.0 as $t;
          [
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
            s / 4.0 as $t,
          ]
          .into()
        } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
          let s = (1.0 as $t + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0 as $t;
          [
            s / 4.0 as $t,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[2][1] - m[1][2]) / s,
          ]
          .into()
        } else if m[1][1] > m[2][2] {
          let s = (1.0 as $t - m[0][0] + m[1][1] - m[2][2]).sqrt() * 2.0 as $t;
          [
            (m[0][1] + m[1][0]) / s,
            s / 4.0 as $t,
            (m[1][2] + m[2][1]) / s,
            (m[0][2] - m[2][0]) / s,
          ]
          .into()
        } else {
          let s = (1.0 as $t - m[0][0] - m[1][1] + m[2][2]).sqrt() * 2.0 as $t;
          [
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            s / 4.0 as $t,
            (m[1][0] - m[0][1]) / s,
          ]
          .into()
        }
      }

      /// Convert to 4x4 rotation matrix
      pub fn to_matrix(&self) -> Matrix<$t, 4, 4> {
        let x = self[0];
        let y = self[1];
        let z = self[2];
        let w = self[3];

        let x2 = x + x;
        let y2 = y + y;
        let z2 = z + z;

        let xx = x * x2;
        let xy = x * y2;
        let xz = x * z2;
        let yy = y * y2;
        let yz = y * z2;
        let zz = z * z2;
        let wx = w * x2;
        let wy = w * y2;
        let wz = w * z2;

        [
          [1.0 as $t - (yy + zz), xy - wz, xz + wy, 0.0 as $t],
          [xy + wz, 1.0 as $t - (xx + zz), yz - wx, 0.0 as $t],
          [xz - wy, yz + wx, 1.0 as $t - (xx + yy), 0.0 as $t],
          [0.0 as $t, 0.0 as $t, 0.0 as $t, 1.0 as $t],
        ]
        .into()
      }

      /// Magnitude squared
      pub fn magnitude_sq(&self) -> $t {
        self[0] * self[0]
          + self[1] * self[1]
          + self[2] * self[2]
          + self[3] * self[3]
      }

      /// Magnitude
      pub fn magnitude(&self) -> $t {
        self.magnitude_sq().sqrt()
      }

      /// Inverse. Returns None for zero quaternions.
      pub fn inverse(&self) -> Option<Quaternion<$t>> {
        let mag_sq = self.magnitude_sq();
        if mag_sq < 1e-10 as $t {
          return None;
        }
        let inv = 1.0 as $t / mag_sq;
        Some(
          [
            -self[0] * inv,
            -self[1] * inv,
            -self[2] * inv,
            self[3] * inv,
          ]
          .into(),
        )
      }

      /// Rotate a vector by this quaternion
      pub fn rotate_vector(&self, v: &Vector<$t, 3>) -> Vector<$t, 3> {
        let u: Vector<$t, 3> = [self[0], self[1], self[2]].into();
        let w = self[3];

        let uv = u.cross(v);
        let uuv = u.cross(&uv);

        v + &(&uv * &(2.0 as $t * w) + &uuv * &(2.0 as $t))
      }

      /// Spherical linear interpolation between two quaternions
      pub fn slerp(&self, other: &Quaternion<$t>, t: $t) -> Quaternion<$t> {
        let mut dot = self[0] * other[0]
          + self[1] * other[1]
          + self[2] * other[2]
          + self[3] * other[3];

        // If dot is negative, negate one to take the shorter path
        let mut other_sign = [other[0], other[1], other[2], other[3]];
        if dot < 0.0 as $t {
          dot = -dot;
          other_sign[0] = -other_sign[0];
          other_sign[1] = -other_sign[1];
          other_sign[2] = -other_sign[2];
          other_sign[3] = -other_sign[3];
        }

        // If quaternions are very close, fall back to lerp to avoid division by zero
        if dot > 0.9995 as $t {
          return [
            self[0] + t * (other_sign[0] - self[0]),
            self[1] + t * (other_sign[1] - self[1]),
            self[2] + t * (other_sign[2] - self[2]),
            self[3] + t * (other_sign[3] - self[3]),
          ]
          .into();
        }

        let theta = dot.acos();
        let sin_theta = theta.sin();
        let a = ((1.0 as $t - t) * theta).sin() / sin_theta;
        let b = (t * theta).sin() / sin_theta;

        [
          a * self[0] + b * other_sign[0],
          a * self[1] + b * other_sign[1],
          a * self[2] + b * other_sign[2],
          a * self[3] + b * other_sign[3],
        ]
        .into()
      }
    }
  };
}

impl_quaternion_float!(f32);
impl_quaternion_float!(f64);

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
  T: Default + Neg<Output = T> + Clone,
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
  for<'x> &'x T:
    Mul<&'x T, Output = T> + Sub<&'x T, Output = T> + Add<&'x T, Output = T>,
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
    let g = &(&self[1] * &rhs[1]) + &(&self[2] * &rhs[2]);
    out[3] = &f - &g;

    out
  }
}

impl<T> Mul for Quaternion<T>
where
  T: Mul<Output = T> + Sub<Output = T> + Add<Output = T> + Default + Clone,
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
    let g = self[1].clone() * rhs[1].clone() + self[2].clone() * rhs[2].clone();
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
    let g = self[1].clone() * rhs[1].clone() + self[2].clone() * rhs[2].clone();
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

pub struct QuatIter<'a, T>
where
  T: Default,
{
  inner: &'a Matrix<T, 4, 1>,
}

impl<'a, T> Iterator for QuatIter<'a, T>
where
  T: Default,
{
  type Item = &'a T;
  fn next(&mut self) -> Option<Self::Item> {
    self.inner.iter().next()
  }
}

pub struct QuatIterMut<'a, T>
where
  T: Default,
{
  inner: &'a mut Matrix<T, 4, 1>,
  col: usize,
}

impl<'a, T> Iterator for QuatIterMut<'a, T>
where
  T: Default,
{
  type Item = &'a mut T;
  fn next(&mut self) -> Option<Self::Item> {
    if self.col >= 4 {
      return None;
    }

    let next = &mut self.inner[0][self.col] as *mut T;
    self.col += 1;

    // This is safe because we never
    // index past the bound of C!
    Some(unsafe { &mut *next })
  }
}

pub struct QuatIterInto<T>
where
  T: Default + Clone,
{
  inner: Matrix<T, 4, 1>,
  col: usize,
}

impl<T> Iterator for QuatIterInto<T>
where
  T: Default + Clone,
{
  type Item = T;
  fn next(&mut self) -> Option<Self::Item> {
    if self.col >= 4 {
      return None;
    }

    let next = self.inner[0][self.col].clone();
    self.col += 1;

    // This is safe because we never
    // index past the bound of C!
    Some(next)
  }
}

impl<T> Quaternion<T>
where
  T: Default,
{
  pub fn iter(&self) -> QuatIter<'_, T> {
    QuatIter { inner: &self.inner }
  }

  pub fn iter_mut(&mut self) -> QuatIterMut<'_, T> {
    QuatIterMut {
      inner: &mut self.inner,
      col: 0,
    }
  }
}

impl<T> IntoIterator for Quaternion<T>
where
  T: Default + Clone,
{
  type Item = T;
  type IntoIter = QuatIterInto<T>;

  fn into_iter(self) -> Self::IntoIter {
    QuatIterInto {
      inner: self.inner,
      col: 0,
    }
  }
}

// ================ Unit tests =================//

#[cfg(test)]
mod test {
  use crate::maths::quaternion::Quaternion;
  use crate::maths::vector::Vector;

  fn approx_eq_quat(
    a: &Quaternion<f32>,
    b: &Quaternion<f32>,
    eps: f32,
  ) -> bool {
    for i in 0..4 {
      if (a[i] - b[i]).abs() > eps {
        return false;
      }
    }
    true
  }

  fn approx_eq_vec3(a: &Vector<f32, 3>, b: &Vector<f32, 3>, eps: f32) -> bool {
    for i in 0..3 {
      if (a[i] - b[i]).abs() > eps {
        return false;
      }
    }
    true
  }

  #[test]
  fn from_axis_angle_identity() {
    let axis: Vector<f32, 3> = [0.0, 1.0, 0.0].into();
    let q = Quaternion::<f32>::from_axis_angle(&axis, 0.0);
    // angle=0 gives identity quaternion [0,0,0,1]
    assert!((q[0]).abs() < 1e-6);
    assert!((q[1]).abs() < 1e-6);
    assert!((q[2]).abs() < 1e-6);
    assert!((q[3] - 1.0).abs() < 1e-6);
  }

  #[test]
  fn from_axis_angle_180_degrees() {
    let axis: Vector<f32, 3> = [0.0, 0.0, 1.0].into();
    let q = Quaternion::<f32>::from_axis_angle(&axis, std::f32::consts::PI);
    // w = cos(pi/2) = 0, z = sin(pi/2) = 1
    assert!((q[3]).abs() < 1e-5);
    assert!((q[2] - 1.0).abs() < 1e-5);
  }

  #[test]
  fn magnitude_unit_quaternion() {
    let axis: Vector<f32, 3> = [1.0, 0.0, 0.0].into();
    let q = Quaternion::<f32>::from_axis_angle(&axis, 1.5);
    assert!((q.magnitude() - 1.0).abs() < 1e-6);
  }

  #[test]
  fn inverse_of_unit_is_conjugate() {
    let axis: Vector<f32, 3> = [1.0, 1.0, 0.0].into();
    let q = Quaternion::<f32>::from_axis_angle(&axis, 0.8);
    let inv = q.inverse().unwrap();
    let conj = q.conjugate();
    // for unit quaternions, inverse == conjugate
    assert!(approx_eq_quat(&inv, &conj, 1e-6));
  }

  #[test]
  fn inverse_q_times_q_is_identity() {
    let axis: Vector<f32, 3> = [1.0, 2.0, 3.0].into();
    let q = Quaternion::<f32>::from_axis_angle(&axis, 1.2);
    let inv = q.inverse().unwrap();
    let result = &q * &inv;
    // should be identity: [0,0,0,1]
    assert!((result[0]).abs() < 1e-5);
    assert!((result[1]).abs() < 1e-5);
    assert!((result[2]).abs() < 1e-5);
    assert!((result[3] - 1.0).abs() < 1e-5);
  }

  #[test]
  fn inverse_zero_returns_none() {
    let q: Quaternion<f32> = [0.0, 0.0, 0.0, 0.0].into();
    assert!(q.inverse().is_none());
  }

  #[test]
  fn rotate_vector_90_around_z() {
    let axis: Vector<f32, 3> = [0.0, 0.0, 1.0].into();
    let q =
      Quaternion::<f32>::from_axis_angle(&axis, std::f32::consts::FRAC_PI_2);
    let v: Vector<f32, 3> = [1.0, 0.0, 0.0].into();
    let rotated = q.rotate_vector(&v);
    let expected: Vector<f32, 3> = [0.0, 1.0, 0.0].into();
    assert!(approx_eq_vec3(&rotated, &expected, 1e-5));
  }

  #[test]
  fn rotate_vector_180_around_y() {
    let axis: Vector<f32, 3> = [0.0, 1.0, 0.0].into();
    let q = Quaternion::<f32>::from_axis_angle(&axis, std::f32::consts::PI);
    let v: Vector<f32, 3> = [1.0, 0.0, 0.0].into();
    let rotated = q.rotate_vector(&v);
    let expected: Vector<f32, 3> = [-1.0, 0.0, 0.0].into();
    assert!(approx_eq_vec3(&rotated, &expected, 1e-5));
  }

  #[test]
  fn rotate_vector_preserves_magnitude() {
    let axis: Vector<f32, 3> = [1.0, 1.0, 1.0].into();
    let q = Quaternion::<f32>::from_axis_angle(&axis, 1.3);
    let v: Vector<f32, 3> = [3.0, 4.0, 5.0].into();
    let rotated = q.rotate_vector(&v);
    assert!((rotated.magnitude() - v.magnitude()).abs() < 1e-4);
  }

  #[test]
  fn to_matrix_matches_rotate_vector() {
    let axis: Vector<f32, 3> = [1.0, 2.0, 3.0].into();
    let q = Quaternion::<f32>::from_axis_angle(&axis, 0.7);
    let v: Vector<f32, 3> = [4.0, 5.0, 6.0].into();

    // rotate via quaternion
    let r1 = q.rotate_vector(&v);

    // rotate via matrix: extract 3x3 rotation from 4x4
    let m = q.to_matrix();
    let r2: Vector<f32, 3> = [
      m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
      m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
      m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
    .into();

    assert!(approx_eq_vec3(&r1, &r2, 1e-4));
  }

  #[test]
  fn to_matrix_from_matrix_roundtrip() {
    let axis: Vector<f32, 3> = [0.0, 1.0, 0.0].into();
    let q = Quaternion::<f32>::from_axis_angle(&axis, 1.0);
    let m = q.to_matrix();
    let q2 = Quaternion::<f32>::from_matrix(&m);
    // quaternions can differ by sign and represent same rotation
    let same = approx_eq_quat(&q, &q2, 1e-5);
    let neg: Quaternion<f32> = [-q[0], -q[1], -q[2], -q[3]].into();
    let same_neg = approx_eq_quat(&neg, &q2, 1e-5);
    assert!(same || same_neg);
  }

  #[test]
  fn from_euler_pitch_only() {
    // pitch 90° around X should be same as axis_angle(X, 90°)
    let q =
      Quaternion::<f32>::from_euler(std::f32::consts::FRAC_PI_2, 0.0, 0.0);
    let axis: Vector<f32, 3> = [1.0, 0.0, 0.0].into();
    let expected =
      Quaternion::<f32>::from_axis_angle(&axis, std::f32::consts::FRAC_PI_2);
    let same = approx_eq_quat(&q, &expected, 1e-5);
    let neg: Quaternion<f32> =
      [-expected[0], -expected[1], -expected[2], -expected[3]].into();
    let same_neg = approx_eq_quat(&q, &neg, 1e-5);
    assert!(same || same_neg);
  }

  #[test]
  fn slerp_endpoints() {
    let axis1: Vector<f32, 3> = [0.0, 1.0, 0.0].into();
    let axis2: Vector<f32, 3> = [0.0, 0.0, 1.0].into();
    let a = Quaternion::<f32>::from_axis_angle(&axis1, 0.5);
    let b = Quaternion::<f32>::from_axis_angle(&axis2, 1.0);

    let s0 = a.slerp(&b, 0.0);
    let s1 = a.slerp(&b, 1.0);

    assert!(approx_eq_quat(&s0, &a, 1e-5));
    // slerp(a,b,1) may differ by sign from b
    let same = approx_eq_quat(&s1, &b, 1e-5);
    let neg: Quaternion<f32> = [-b[0], -b[1], -b[2], -b[3]].into();
    let same_neg = approx_eq_quat(&s1, &neg, 1e-5);
    assert!(same || same_neg);
  }

  #[test]
  fn slerp_midpoint_unit_length() {
    let axis1: Vector<f32, 3> = [1.0, 0.0, 0.0].into();
    let axis2: Vector<f32, 3> = [0.0, 1.0, 0.0].into();
    let a = Quaternion::<f32>::from_axis_angle(&axis1, 0.0);
    let b = Quaternion::<f32>::from_axis_angle(&axis2, std::f32::consts::PI);
    let mid = a.slerp(&b, 0.5);
    // slerp preserves unit length
    assert!((mid.magnitude() - 1.0).abs() < 1e-5);
  }

  #[test]
  fn slerp_same_quaternion() {
    let axis: Vector<f32, 3> = [1.0, 0.0, 0.0].into();
    let q = Quaternion::<f32>::from_axis_angle(&axis, 0.5);
    let mid = q.slerp(&q, 0.5);
    assert!(approx_eq_quat(&mid, &q, 1e-5));
  }

  #[test]
  fn conjugate_negates_xyz() {
    let q: Quaternion<f32> = [1.0, 2.0, 3.0, 4.0].into();
    let c = q.conjugate();
    assert!((c[0] - (-1.0)).abs() < 1e-6);
    assert!((c[1] - (-2.0)).abs() < 1e-6);
    assert!((c[2] - (-3.0)).abs() < 1e-6);
    assert!((c[3] - 4.0).abs() < 1e-6);
  }

  #[test]
  fn quaternion_multiply_non_commutative() {
    let a: Quaternion<f32> = [1.0, 0.0, 0.0, 1.0].into();
    let b: Quaternion<f32> = [0.0, 1.0, 0.0, 1.0].into();
    let ab = &a * &b;
    let ba = &b * &a;
    // quaternion multiplication is NOT commutative
    assert!(!approx_eq_quat(&ab, &ba, 1e-6));
  }
}
