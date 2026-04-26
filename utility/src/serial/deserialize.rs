// Copyright 2026 William Probyn
// SPDX-License-Identifier: Apache-2.0

pub trait Deserialize: Sized {
  fn deserialize<D: Deserializer>(
    deserializer: &mut D,
  ) -> Result<Self, D::Error>;
}

pub trait Deserializer: Sized {
  type Error: std::fmt::Display;

  fn deserialize_bool(&mut self) -> Result<bool, Self::Error>;
  fn deserialize_u8(&mut self) -> Result<u8, Self::Error>;
  fn deserialize_u16(&mut self) -> Result<u16, Self::Error>;
  fn deserialize_u32(&mut self) -> Result<u32, Self::Error>;
  fn deserialize_u64(&mut self) -> Result<u64, Self::Error>;
  fn deserialize_i8(&mut self) -> Result<i8, Self::Error>;
  fn deserialize_i16(&mut self) -> Result<i16, Self::Error>;
  fn deserialize_i32(&mut self) -> Result<i32, Self::Error>;
  fn deserialize_i64(&mut self) -> Result<i64, Self::Error>;
  fn deserialize_f32(&mut self) -> Result<f32, Self::Error>;
  fn deserialize_f64(&mut self) -> Result<f64, Self::Error>;
  fn deserialize_str(&mut self) -> Result<String, Self::Error>;
  fn deserialize_bytes(&mut self) -> Result<Vec<u8>, Self::Error>;
  fn deserialize_option<T: Deserialize>(
    &mut self,
  ) -> Result<Option<T>, Self::Error>;
  fn deserialize_unit(&mut self) -> Result<(), Self::Error>;
  fn deserialize_seq_begin(&mut self) -> Result<usize, Self::Error>; // returns len
  fn deserialize_seq_element<T: Deserialize>(
    &mut self,
  ) -> Result<T, Self::Error>;
  fn deserialize_seq_has_next(&mut self) -> Result<bool, Self::Error>;
  fn deserialize_seq_end(&mut self) -> Result<(), Self::Error>;
  fn deserialize_struct_begin(
    &mut self,
    name: &str,
  ) -> Result<usize, Self::Error>;
  fn deserialize_struct_field<T: Deserialize>(
    &mut self,
    key: &str,
  ) -> Result<T, Self::Error>;
  fn deserialize_struct_end(&mut self) -> Result<(), Self::Error>;
  fn deserialize_enum_begin(
    &mut self,
    variants: &[&str],
  ) -> Result<usize, Self::Error>;
  fn deserialize_enum_end(&mut self) -> Result<(), Self::Error>;
}

macro_rules! impl_deserialize_primitive {
  ($($ty:ty => $method:ident),* $(,)?) => {
    $(
      impl Deserialize for $ty {
        fn deserialize<D: Deserializer>(d: &mut D) -> Result<$ty, D::Error> {
          d.$method()
        }
      }
    )*
  };
}

impl_deserialize_primitive! {
  bool => deserialize_bool,
  u8   => deserialize_u8,
  u16  => deserialize_u16,
  u32  => deserialize_u32,
  u64  => deserialize_u64,
  i8   => deserialize_i8,
  i16  => deserialize_i16,
  i32  => deserialize_i32,
  i64  => deserialize_i64,
  f32  => deserialize_f32,
  f64  => deserialize_f64,
}

impl Deserialize for String {
  fn deserialize<D: Deserializer>(d: &mut D) -> Result<Self, D::Error> {
    d.deserialize_str()
  }
}

impl<T: Deserialize> Deserialize for Vec<T> {
  fn deserialize<D: Deserializer>(d: &mut D) -> Result<Self, D::Error> {
    d.deserialize_seq_begin()?;
    let mut items = Vec::new();
    while d.deserialize_seq_has_next()? {
      items.push(d.deserialize_seq_element::<T>()?);
    }
    d.deserialize_seq_end()?;
    Ok(items)
  }
}

impl<T: Deserialize> Deserialize for Option<T> {
  fn deserialize<D: Deserializer>(d: &mut D) -> Result<Self, D::Error> {
    d.deserialize_option::<T>()
  }
}
