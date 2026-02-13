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
        b' ' | b'\n' | b'\r' | b'\t' => { self.read_byte()?; },
        _ => break,
      }
    }
    Ok(())
  }

  pub(crate) fn expect_byte(&mut self, expected: u8) -> AetherResult<()> {
    match self.read_byte()? {
      Some(b) if b == expected => Ok(()),
      Some(b) => Err(
        AetherError::new(ErrorKind::UnexpectedByte)
        .context(format!("a text reader expected byte: {} but found: {}", expected, b))
      ),
      None => Err(
        AetherError::new(ErrorKind::UnexpectedEof)
        .context(format!("a text reader expected byte: {} but found an end of file", expected))
      ),
    }
  }

  pub(crate) fn read_while(&mut self, pred: fn(u8) -> bool) -> AetherResult<String> {
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
    let s = self.read_while(|b| b != b'"')?;
    self.expect_byte(b'"')?;                          
    Ok(s)
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
      ErrorKind::UnexpectedByte => "a reader has encountered an unexpected byte",
    };

    write!(f, "{}", string)?;
    Ok(())
  }
}
