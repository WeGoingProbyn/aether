use crate::{error::{AetherError, ErrorDomain, UtilityErrorKind}, serial::{TextReader, deserialize::Deserializer, serialize::Serializer}};

pub struct JsonSerializer<W: std::io::Write> {
  writer: W,
  needs_comma: bool
}

impl<W: std::io::Write> JsonSerializer<W> {
  pub fn new(writer: W) -> JsonSerializer<W> {
    JsonSerializer {
      writer,
      needs_comma: false,
    }
  }
}

impl<W: std::io::Write> Serializer for JsonSerializer<W> {
  type Error = AetherError;

  fn serialize_u8(&mut self, v: u8) -> Result<(), Self::Error> {
    write!(self.writer, "{}", v)?;
    Ok(())
  }

  fn serialize_i8(&mut self, v: i8) -> Result<(), Self::Error> {
    write!(self.writer, "{}", v)?;
    Ok(())
  }

  fn serialize_u16(&mut self, v: u16) -> Result<(), Self::Error> {
    write!(self.writer, "{}", v)?;
    Ok(())
  }

  fn serialize_i16(&mut self, v: i16) -> Result<(), Self::Error> {
    write!(self.writer, "{}", v)?;
    Ok(())
  }

  fn serialize_u32(&mut self, v: u32) -> Result<(), Self::Error> {
    write!(self.writer, "{}", v)?;
    Ok(())
  }

  fn serialize_i32(&mut self, v: i32) -> Result<(), Self::Error> {
    write!(self.writer, "{}", v)?;
    Ok(())
  }

  fn serialize_u64(&mut self, v: u64) -> Result<(), Self::Error> {
    write!(self.writer, "{}", v)?;
    Ok(())
  }

  fn serialize_i64(&mut self, v: i64) -> Result<(), Self::Error> {
    write!(self.writer, "{}", v)?;
    Ok(())
  }

  fn serialize_f32(&mut self, v: f32) -> Result<(), Self::Error> {
    write!(self.writer, "{}", v)?;
    Ok(())
  }

  fn serialize_f64(&mut self, v: f64) -> Result<(), Self::Error> {
    write!(self.writer, "{}", v)?;
    Ok(())
  }

  fn serialize_str(&mut self, v: &str) -> Result<(), Self::Error> {
    write!(self.writer, "\"{}\"", v)?;
    Ok(())
  }

  fn serialize_unit(&mut self) -> Result<(), Self::Error> {
    write!(self.writer, "null")?;
    Ok(())
  }

  fn serialize_bool(&mut self, v: bool) -> Result<(), Self::Error> {
    write!(self.writer, "{}", v)?;
    Ok(())
  }

  fn serialize_none(&mut self) -> Result<(), Self::Error> {
    write!(self.writer, "null")?;
    Ok(())
  }

  fn serialize_some<T: super::serialize::Serialize>(&mut self, v: &T) -> Result<(), Self::Error> {
    v.serialize(self)?;
    Ok(())
  }

  fn serialize_bytes(&mut self, v: &[u8]) -> Result<(), Self::Error> {
    self.serialize_seq_begin(v.len())?;
    for byte in v {
      self.serialize_seq_element(byte)?;            
    }
    self.serialize_seq_end()
  }

  fn serialize_seq_begin(&mut self, _length: usize) -> Result<(), Self::Error> {
    write!(self.writer, "[")?;
    self.needs_comma = false;
    Ok(())
  }

  fn serialize_seq_end(&mut self) -> Result<(), Self::Error> {
    write!(self.writer, "]")?;
    self.needs_comma = true;
    Ok(())
  }

  fn serialize_seq_element<T: super::serialize::Serialize>(&mut self, v: &T) -> Result<(), Self::Error> {
    if self.needs_comma {
      write!(self.writer, ",")?;
    }
    v.serialize(self)?;
    self.needs_comma = true;
    Ok(())
  }

  fn serialize_struct_end(&mut self) -> Result<(), Self::Error> {
    write!(self.writer, "}}")?;
    self.needs_comma = true;
    Ok(())
  }

  fn serialize_struct_begin(&mut self, _: &str, _: usize) -> Result<(), Self::Error> {
    write!(self.writer, "{{")?;    
    self.needs_comma = false;
    Ok(())
  }

  fn serialize_struct_field<T: super::serialize::Serialize>(&mut self, key: &str, v: &T) -> Result<(), Self::Error> {
    if self.needs_comma {
      write!(self.writer, ",")?;
    }
    write!(self.writer, "\"{}\":", key)?;
    v.serialize(self)?;
    self.needs_comma = true;
    Ok(())
  }

  fn serialize_enum_begin(&mut self, variant: &str) -> Result<(), Self::Error> {
    write!(self.writer, "{{\"{}\":", variant)?;
    Ok(())
  }

  fn serialize_enum_end(&mut self) -> Result<(), Self::Error> {
    write!(self.writer, "}}")?;
    Ok(())
  }
}

pub struct JsonDeserializer<R: std::io::Read> {
  reader: TextReader<R>,
}

impl<R: std::io::Read> JsonDeserializer<R> {
  pub fn new(reader: R) -> JsonDeserializer<R> {
    JsonDeserializer {
      reader: TextReader::new(reader),
    }
  }
}

impl<R: std::io::Read> Deserializer for JsonDeserializer<R> {
  type Error = AetherError;

  fn deserialize_u8(&mut self) -> Result<u8, Self::Error> {
    self.reader.skip_whitespace()?;
    let s = self.reader.read_while(|b| b.is_ascii_digit() )?;
    s.parse::<u8>().map_err(|_|
      AetherError::new(ErrorDomain::Utility(UtilityErrorKind::JsonDeserializer))
        .context(format!("failed to parse '{}' as u8", s))
    )
  }

  fn deserialize_i8(&mut self) -> Result<i8, Self::Error> {
    self.reader.skip_whitespace()?;
    let s = self.reader.read_while(|b| b.is_ascii_digit() || b == b'-')?;
    s.parse::<i8>().map_err(|_|
      AetherError::new(ErrorDomain::Utility(UtilityErrorKind::JsonDeserializer))
        .context(format!("failed to parse '{}' as i8", s))
    )
  }

  fn deserialize_u16(&mut self) -> Result<u16, Self::Error> {
    self.reader.skip_whitespace()?;
    let s = self.reader.read_while(|b| b.is_ascii_digit() )?;
    s.parse::<u16>().map_err(|_|
      AetherError::new(ErrorDomain::Utility(UtilityErrorKind::JsonDeserializer))
        .context(format!("failed to parse '{}' as u16", s))
    )
  }

  fn deserialize_i16(&mut self) -> Result<i16, Self::Error> {
    self.reader.skip_whitespace()?;
    let s = self.reader.read_while(|b| b.is_ascii_digit() || b == b'-')?;
    s.parse::<i16>().map_err(|_|
      AetherError::new(ErrorDomain::Utility(UtilityErrorKind::JsonDeserializer))
        .context(format!("failed to parse '{}' as i16", s))
    )
  }

  fn deserialize_u32(&mut self) -> Result<u32, Self::Error> {
    self.reader.skip_whitespace()?;
    let s = self.reader.read_while(|b| b.is_ascii_digit() )?;
    s.parse::<u32>().map_err(|_|
      AetherError::new(ErrorDomain::Utility(UtilityErrorKind::JsonDeserializer))
        .context(format!("failed to parse '{}' as u32", s))
    )
  }

  fn deserialize_i32(&mut self) -> Result<i32, Self::Error> {
    self.reader.skip_whitespace()?;
    let s = self.reader.read_while(|b| b.is_ascii_digit() || b == b'-' )?;
    s.parse::<i32>().map_err(|_|
      AetherError::new(ErrorDomain::Utility(UtilityErrorKind::JsonDeserializer))
        .context(format!("failed to parse '{}' as i32", s))
    )
  }

  fn deserialize_u64(&mut self) -> Result<u64, Self::Error> {
    self.reader.skip_whitespace()?;
    let s = self.reader.read_while(|b| b.is_ascii_digit() )?;
    s.parse::<u64>().map_err(|_|
      AetherError::new(ErrorDomain::Utility(UtilityErrorKind::JsonDeserializer))
        .context(format!("failed to parse '{}' as u64", s))
    )
  }

  fn deserialize_i64(&mut self) -> Result<i64, Self::Error> {
    self.reader.skip_whitespace()?;
    let s = self.reader.read_while(|b| b.is_ascii_digit() || b == b'-' )?;
    s.parse::<i64>().map_err(|_|
      AetherError::new(ErrorDomain::Utility(UtilityErrorKind::JsonDeserializer))
        .context(format!("failed to parse '{}' as i64", s))
    )
  }

  fn deserialize_f32(&mut self) -> Result<f32, Self::Error> {
    self.reader.skip_whitespace()?;
    let s = self.reader.read_while(|b| b.is_ascii_digit() || b == b'.' || b == b'-' || b == b'e' || b == b'E')?;
    s.parse::<f32>().map_err(|_|
      AetherError::new(ErrorDomain::Utility(UtilityErrorKind::JsonDeserializer))
        .context(format!("failed to parse '{}' as f32", s))
    )
  }

  fn deserialize_f64(&mut self) -> Result<f64, Self::Error> {
    self.reader.skip_whitespace()?;
    let s = self.reader.read_while(|b| b.is_ascii_digit() || b == b'.' || b == b'-' || b == b'e' || b == b'E')?;
    s.parse::<f64>().map_err(|_|
      AetherError::new(ErrorDomain::Utility(UtilityErrorKind::JsonDeserializer))
        .context(format!("failed to parse '{}' as f64", s))
    )
  }

  fn deserialize_str(&mut self) -> Result<String, Self::Error> {
    self.reader.skip_whitespace()?;
    self.reader.read_string_lit()
  }

  fn deserialize_bool(&mut self) -> Result<bool, Self::Error> {
    self.reader.skip_whitespace()?;
    let s = self.reader.read_while(|b| b.is_ascii_alphabetic() )?;
    match s.as_str() {                                
      "true" => Ok(true),
      "false" => Ok(false),
      _ => Err(
        AetherError::new(ErrorDomain::Utility(UtilityErrorKind::JsonDeserializer))
        .context(format!("expected 'true' or 'false', got '{}'", s))
      )
    }  
  }

  fn deserialize_unit(&mut self) -> Result<(), Self::Error> {
    self.reader.skip_whitespace()?;
    let s = self.reader.read_while(|b| b.is_ascii_alphabetic() )?;
    match s.as_str() {
      "null" => Ok(()),
      _ => Err(
        AetherError::new(ErrorDomain::Utility(UtilityErrorKind::JsonDeserializer))
        .context(format!("expected null, got '{}'", s))
      )
    }
  }

  fn deserialize_option<T: super::deserialize::Deserialize>(&mut self) -> Result<Option<T>, Self::Error> {
    self.reader.skip_whitespace()?;                   
    match self.reader.peek_byte()? {
      Some(b'n') => {
        self.reader.read_while(|b| b.is_ascii_alphabetic())?;
        Ok(None)
      }
      _ => Ok(Some(T::deserialize(self)?)),
    }
  }

  fn deserialize_bytes(&mut self) -> Result<Vec<u8>, Self::Error> {
    self.reader.skip_whitespace()?;
    self.reader.expect_byte(b'[')?;
    let mut bytes = Vec::new();
    loop {
      self.reader.skip_whitespace()?;
      match self.reader.peek_byte()? {
        Some(b']') => { self.reader.read_byte()?; break; }
        Some(b',') => { self.reader.read_byte()?; }
        Some(_) => {
          let s = self.reader.read_while(|b| b.is_ascii_digit())?;
          bytes.push(s.parse::<u8>().map_err(|_|
            AetherError::new(ErrorDomain::Utility(UtilityErrorKind::JsonDeserializer))
              .context(format!("failed to parse '{}' as u8 in bytes array", s))
          )?);
        }
        None => return Err(
          AetherError::new(ErrorDomain::Utility(UtilityErrorKind::JsonDeserializer))
            .context("unexpected EOF in bytes array")
        ),
      }
    }
    Ok(bytes)
  }

  fn deserialize_seq_begin(&mut self) -> Result<usize, Self::Error> {
    self.reader.skip_whitespace()?;
    self.reader.expect_byte(b'[')?;
    // JSON doesn't encode length upfront, return 0
    Ok(0)
  }

  fn deserialize_seq_element<T: super::deserialize::Deserialize>(&mut self) -> Result<T, Self::Error> {
    self.reader.skip_whitespace()?;
    // consume comma if present
    if let Some(b',') = self.reader.peek_byte()? {
      self.reader.read_byte()?;
    }
    T::deserialize(self)
  }

  fn deserialize_seq_has_next(&mut self) -> Result<bool, Self::Error> {
    self.reader.skip_whitespace()?;
    match self.reader.peek_byte()? {
      Some(b']') => Ok(false),
      Some(_) => Ok(true),
      None => Err(
        AetherError::new(ErrorDomain::Utility(UtilityErrorKind::JsonDeserializer))
          .context("unexpected EOF while reading sequence")
      ),
    }
  }

  fn deserialize_seq_end(&mut self) -> Result<(), Self::Error> {
    self.reader.skip_whitespace()?;
    self.reader.expect_byte(b']')?;
    Ok(())
  }

  fn deserialize_struct_begin(&mut self, _name: &str) -> Result<usize, Self::Error> {
    self.reader.skip_whitespace()?;
    self.reader.expect_byte(b'{')?;
    Ok(0)
  }

  fn deserialize_struct_field<T: super::deserialize::Deserialize>(&mut self, _key: &str) -> Result<T, Self::Error> {
    self.reader.skip_whitespace()?;
    // consume comma if present
    if let Some(b',') = self.reader.peek_byte()? {
      self.reader.read_byte()?;
    }
    // read and discard the key
    self.reader.skip_whitespace()?;
    self.reader.read_string_lit()?;
    // consume the colon
    self.reader.skip_whitespace()?;
    self.reader.expect_byte(b':')?;
    // deserialize the value
    T::deserialize(self)
  }

  fn deserialize_struct_end(&mut self) -> Result<(), Self::Error> {
    self.reader.skip_whitespace()?;
    self.reader.expect_byte(b'}')?;
    Ok(())
  }

  fn deserialize_enum_begin(&mut self, variants: &[&str]) -> Result<usize, Self::Error> {
    self.reader.skip_whitespace()?;
    self.reader.expect_byte(b'{')?;
    self.reader.skip_whitespace()?;
    let variant_name = self.reader.read_string_lit()?;
    self.reader.skip_whitespace()?;
    self.reader.expect_byte(b':')?;
    // find the variant index
    variants.iter().position(|&v| v == variant_name).ok_or_else(||
      AetherError::new(ErrorDomain::Utility(UtilityErrorKind::JsonDeserializer))
        .context(format!("unknown variant '{}', expected one of {:?}", variant_name, variants))
    )
  }

  fn deserialize_enum_end(&mut self) -> Result<(), Self::Error> {
    self.reader.skip_whitespace()?;
    self.reader.expect_byte(b'}')?;
    Ok(())
  }
}
