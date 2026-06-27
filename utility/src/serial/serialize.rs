// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

pub trait Serialize {
  fn serialize<S: Serializer>(
    &self,
    serializer: &mut S,
  ) -> Result<(), S::Error>;
}

pub trait Serializer {
  type Error: std::fmt::Display;

  fn serialize_bool(&mut self, v: bool) -> Result<(), Self::Error>;
  fn serialize_u8(&mut self, v: u8) -> Result<(), Self::Error>;
  fn serialize_u16(&mut self, v: u16) -> Result<(), Self::Error>;
  fn serialize_u32(&mut self, v: u32) -> Result<(), Self::Error>;
  fn serialize_u64(&mut self, v: u64) -> Result<(), Self::Error>;
  fn serialize_i8(&mut self, v: i8) -> Result<(), Self::Error>;
  fn serialize_i16(&mut self, v: i16) -> Result<(), Self::Error>;
  fn serialize_i32(&mut self, v: i32) -> Result<(), Self::Error>;
  fn serialize_i64(&mut self, v: i64) -> Result<(), Self::Error>;
  fn serialize_f32(&mut self, v: f32) -> Result<(), Self::Error>;
  fn serialize_f64(&mut self, v: f64) -> Result<(), Self::Error>;
  fn serialize_str(&mut self, v: &str) -> Result<(), Self::Error>;
  fn serialize_bytes(&mut self, v: &[u8]) -> Result<(), Self::Error>;
  fn serialize_none(&mut self) -> Result<(), Self::Error>;
  fn serialize_some<T: Serialize>(&mut self, v: &T) -> Result<(), Self::Error>;
  fn serialize_unit(&mut self) -> Result<(), Self::Error>;
  fn serialize_seq_begin(&mut self, length: usize) -> Result<(), Self::Error>;
  fn serialize_seq_element<T: Serialize>(
    &mut self,
    v: &T,
  ) -> Result<(), Self::Error>;
  fn serialize_seq_end(&mut self) -> Result<(), Self::Error>;
  fn serialize_struct_begin(
    &mut self,
    name: &str,
    len: usize,
  ) -> Result<(), Self::Error>;
  fn serialize_struct_field<T: Serialize>(
    &mut self,
    key: &str,
    v: &T,
  ) -> Result<(), Self::Error>;
  fn serialize_struct_end(&mut self) -> Result<(), Self::Error>;
  fn serialize_enum_begin(&mut self, variant: &str) -> Result<(), Self::Error>;
  fn serialize_enum_end(&mut self) -> Result<(), Self::Error>;
}

macro_rules! impl_serialize_primitive {
  ($($ty:ty => $method:ident),* $(,)?) => {
    $(
      impl Serialize for $ty {
        fn serialize<S: Serializer>(&self, s: &mut S) -> Result<(), S::Error> {
          s.$method(*self)
        }
      }
    )*
  };
}

impl_serialize_primitive! {
  bool => serialize_bool,
  u8   => serialize_u8,
  u16  => serialize_u16,
  u32  => serialize_u32,
  u64  => serialize_u64,
  i8   => serialize_i8,
  i16  => serialize_i16,
  i32  => serialize_i32,
  i64  => serialize_i64,
  f32  => serialize_f32,
  f64  => serialize_f64,
}

impl Serialize for String {
  fn serialize<S: Serializer>(&self, s: &mut S) -> Result<(), S::Error> {
    s.serialize_str(self)
  }
}

impl Serialize for &str {
  fn serialize<S: Serializer>(&self, s: &mut S) -> Result<(), S::Error> {
    s.serialize_str(self)
  }
}

impl<T: Serialize> Serialize for &[T] {
  fn serialize<S: Serializer>(&self, s: &mut S) -> Result<(), S::Error> {
    s.serialize_seq_begin(self.len())?;
    for item in *self {
      s.serialize_seq_element(item)?;
    }
    s.serialize_seq_end()
  }
}

impl<const N: usize, T: Serialize> Serialize for [T; N] {
  fn serialize<S: Serializer>(&self, s: &mut S) -> Result<(), S::Error> {
    s.serialize_seq_begin(N)?;
    for item in self {
      s.serialize_seq_element(item)?;
    }
    s.serialize_seq_end()
  }
}

impl<T: Serialize> Serialize for Vec<T> {
  fn serialize<S: Serializer>(&self, s: &mut S) -> Result<(), S::Error> {
    s.serialize_seq_begin(self.len())?;
    for item in self {
      s.serialize_seq_element(item)?;
    }
    s.serialize_seq_end()
  }
}

impl<T: Serialize> Serialize for Option<T> {
  fn serialize<S: Serializer>(&self, s: &mut S) -> Result<(), S::Error> {
    match self {
      Some(v) => s.serialize_some(v),
      None => s.serialize_none(),
    }
  }
}
