use crate::error::{AetherError, AetherResult, ErrorDomain};

pub(crate) struct TextReader<R: std::io::Read> {
  reader: R,
  peeked: Option<u8>,
}

impl<R: std::io::Read> TextReader<R> {
  pub(crate) fn new(reader: R) -> TextReader<R> {
    TextReader {
      reader,
      peeked: None,
    }
  }

  pub(crate) fn peek_byte(&mut self) -> AetherResult<Option<u8>> {
    if self.peeked.is_none() {
      let mut buf = [0u8; 1];
      match self.reader.read(&mut buf)? {
        0 => return Ok(None),
        _ => self.peeked = Some(buf[0]),
      }
    }
    Ok(self.peeked)
  }

  pub(crate) fn read_byte(&mut self) -> AetherResult<Option<u8>> {
    if let Some(b) = self.peeked.take() {
      return Ok(Some(b));
    }
    let mut buf = [0u8; 1];
    match self.reader.read(&mut buf)? {
      0 => Ok(None),
      _ => Ok(Some(buf[0])),
    }
  }

  pub(crate) fn skip_whitespace(&mut self) -> AetherResult<()> {
    while let Some(b) = self.peek_byte()? {
      match b {
        b' ' | b'\n' | b'\r' | b'\t' => {
          self.read_byte()?;
        }
        _ => break,
      }
    }
    Ok(())
  }

  pub(crate) fn expect_byte(&mut self, expected: u8) -> AetherResult<()> {
    match self.read_byte()? {
      Some(b) if b == expected => Ok(()),
      Some(b) => Err(AetherError::new(ErrorKind::UnexpectedByte).context(
        format!("a text reader expected byte: {} but found: {}", expected, b),
      )),
      None => Err(AetherError::new(ErrorKind::UnexpectedEof).context(format!(
        "a text reader expected byte: {} but found an end of file",
        expected
      ))),
    }
  }

  pub(crate) fn read_while(
    &mut self,
    pred: fn(u8) -> bool,
  ) -> AetherResult<String> {
    let mut buf: Vec<u8> = vec![];
    while let Some(b) = self.peek_byte()? {
      if pred(b) {
        self.read_byte()?;
        buf.push(b);
      } else {
        break;
      }
    }
    Ok(String::from_utf8(buf)?)
  }

  pub(crate) fn read_string_lit(&mut self) -> AetherResult<String> {
    self.expect_byte(b'"')?;
    let mut buf: Vec<u8> = vec![];

    loop {
      let b = self.read_byte()?.ok_or_else(|| {
        AetherError::new(ErrorKind::UnexpectedEof)
          .context("unexpected EOF while reading string literal")
      })?;

      match b {
        b'"' => break,
        b'\\' => {
          let esc = self.read_byte()?.ok_or_else(|| {
            AetherError::new(ErrorKind::UnexpectedEof).context(
              "unexpected EOF after escape character in string literal",
            )
          })?;

          match esc {
            b'"' => buf.push(b'"'),
            b'\\' => buf.push(b'\\'),
            b'/' => buf.push(b'/'),
            b'b' => buf.push(0x08),
            b'f' => buf.push(0x0C),
            b'n' => buf.push(b'\n'),
            b'r' => buf.push(b'\r'),
            b't' => buf.push(b'\t'),
            b'u' => {
              let code = self.read_hex_u16()?;

              // Handle UTF-16 surrogate pairs
              let codepoint = if (0xD800..=0xDBFF).contains(&code) {
                self.expect_byte(b'\\')?;
                self.expect_byte(b'u')?;
                let low = self.read_hex_u16()?;

                if !(0xDC00..=0xDFFF).contains(&low) {
                  return Err(
                    AetherError::new(ErrorKind::UnexpectedByte).context(
                      format!(
                        "invalid low surrogate in unicode escape: 0x{low:04X}"
                      ),
                    ),
                  );
                }

                let high_ten = (code as u32) - 0xD800;
                let low_ten = (low as u32) - 0xDC00;
                0x10000 + ((high_ten << 10) | low_ten)
              } else if (0xDC00..=0xDFFF).contains(&code) {
                return Err(
                  AetherError::new(ErrorKind::UnexpectedByte).context(format!(
                    "unexpected low surrogate in unicode escape: 0x{code:04X}"
                  )),
                );
              } else {
                code as u32
              };

              let ch = char::from_u32(codepoint).ok_or_else(|| {
                AetherError::new(ErrorKind::UnexpectedByte).context(format!(
                  "invalid unicode codepoint: 0x{codepoint:04X}"
                ))
              })?;

              let mut utf8 = [0u8; 4];
              let encoded = ch.encode_utf8(&mut utf8);
              buf.extend_from_slice(encoded.as_bytes());
            }
            _ => {
              return Err(AetherError::new(ErrorKind::UnexpectedByte).context(
                format!(
                  "invalid escape sequence '\\{}' in string literal",
                  esc as char
                ),
              ));
            }
          }
        }
        _ => buf.push(b),
      }
    }

    Ok(String::from_utf8(buf)?)
  }

  fn read_hex_u16(&mut self) -> AetherResult<u16> {
    let mut value = 0u16;

    for _ in 0..4 {
      let b = self.read_byte()?.ok_or_else(|| {
        AetherError::new(ErrorKind::UnexpectedEof)
          .context("unexpected EOF in unicode escape")
      })?;

      let nibble = match b {
        b'0'..=b'9' => b - b'0',
        b'a'..=b'f' => 10 + (b - b'a'),
        b'A'..=b'F' => 10 + (b - b'A'),
        _ => {
          return Err(AetherError::new(ErrorKind::UnexpectedByte).context(
            format!("invalid hex digit '{}' in unicode escape", b as char),
          ));
        }
      };

      value = (value << 4) | nibble as u16;
    }

    Ok(value)
  }
}

pub enum ErrorKind {
  UnexpectedEof,
  UnexpectedByte,
}

impl ErrorDomain for ErrorKind {
  fn domain(&self) -> &str {
    "serial"
  }
}

impl std::fmt::Display for ErrorKind {
  fn fmt(
    &self,
    f: &mut std::fmt::Formatter<'_>,
  ) -> Result<(), std::fmt::Error> {
    let string = match self {
      ErrorKind::UnexpectedEof => "a reader has encountered an unexpected EOF",
      ErrorKind::UnexpectedByte => {
        "a reader has encountered an unexpected byte"
      }
    };

    write!(f, "{}", string)?;
    Ok(())
  }
}
