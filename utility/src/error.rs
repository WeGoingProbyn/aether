use std::boxed::Box;
use std::error::Error;
use std::marker::{Send, Sync};
use std::panic::Location;

pub type AetherResult<T> = Result<T, AetherError>;

pub trait Unpoison<T> {
  fn unpoison(self) -> T;
}

impl<T> Unpoison<T> for Result<T, std::sync::PoisonError<T>> {
  fn unpoison(self) -> T {
    self.unwrap_or_else(|e| e.into_inner())
  }
}

pub struct AetherError {
  context: Vec<String>,
  locations: Vec<Location<'static>>,
  domain: ErrorDomain,
  parent: Option<Box<dyn Error + Send + Sync + 'static>>,
}

impl AetherError {
  #[track_caller]
  pub fn new(domain: ErrorDomain) -> AetherError {
    AetherError {
      context: vec![],
      locations: vec![*Location::caller()],
      domain,
      parent: None,
    }
  }

  #[track_caller]
  pub fn context(mut self, ctx: impl Into<String>) -> AetherError {
    self.context.push(ctx.into());
    self.locations.push(*Location::caller());
    self
  }

  pub fn domain(self, domain: ErrorDomain) -> AetherError {
    AetherError {
      context: self.context,
      locations: self.locations,
      domain,
      parent: self.parent,
    }
  }

  pub fn parent(
    self,
    parent: impl Error + Send + Sync + 'static,
  ) -> AetherError {
    let boxed: Box<dyn Error + Send + Sync + 'static> = Box::new(parent);

    match boxed.downcast::<AetherError>() {
      Ok(ae_box) => {
        let ae = *ae_box; // AetherError
        let mut context = self.context;
        let mut locations = self.locations;

        context.extend(ae.context);
        locations.extend(ae.locations);

        AetherError {
          context,
          locations,
          domain: self.domain,
          parent: ae.parent,
        }
      }
      Err(boxed) => AetherError {
        context: self.context,
        locations: self.locations,
        domain: self.domain,
        parent: Some(boxed),
      },
    }
  }
}

impl std::error::Error for AetherError {
  fn source(&self) -> Option<&(dyn Error + 'static)> {
    self
      .parent
      .as_deref()
      .map(|e| e as &(dyn std::error::Error + 'static))
  }
}

impl std::fmt::Display for AetherError {
  fn fmt(
    &self,
    f: &mut std::fmt::Formatter<'_>,
  ) -> Result<(), std::fmt::Error> {
    write!(f, "\n{}\n", self.domain)?;
    Ok(())
  }
}

impl std::fmt::Debug for AetherError {
  fn fmt(
    &self,
    f: &mut std::fmt::Formatter<'_>,
  ) -> Result<(), std::fmt::Error> {
    write!(f, "\ncontext:")?;

    let longest = self.context.iter().map(|s| s.len()).max().unwrap_or(0);

    for (ctx, loc) in self.context.iter().zip(self.locations.iter()) {
      let pad = " ".repeat(longest - ctx.len());
      write!(f, "\n\t{}{} @ {}", ctx, pad, loc)?;
    }

    write!(f, "\n\t└─ stack (most recent first)")?;

    match self.source() {
      Some(e) => write!(f, "\nsource: {}", e)?,
      None => write!(f, "{}", self)?,
    }
    Ok(())
  }
}

impl From<std::io::Error> for AetherError {
  fn from(value: std::io::Error) -> Self {
    AetherError::new(ErrorDomain::Utility(UtilityErrorKind::IoError))
      .context("io error")
      .parent(value)
  }
}

impl From<std::string::FromUtf8Error> for AetherError {
  fn from(value: std::string::FromUtf8Error) -> Self {
    AetherError::new(ErrorDomain::Utility(UtilityErrorKind::Utf8Decode))
      .context("utf8 decode error")
      .parent(value)
  }
}

pub enum ErrorDomain {
  Utility(UtilityErrorKind),
}

impl std::fmt::Display for ErrorDomain {
  fn fmt(
    &self,
    f: &mut std::fmt::Formatter<'_>,
  ) -> Result<(), std::fmt::Error> {
    write!(f, "domain:\t")?;
    match self {
      ErrorDomain::Utility(m) => write!(f, "utility\n  kind:\t{}", m)?,
    }
    Ok(())
  }
}

pub enum UtilityErrorKind {
  Unknown,
  IoError,
  Utf8Decode,
  UnexpectedEof,
  UnexpectedByte,
  JsonDeserializer,
}

impl std::fmt::Display for UtilityErrorKind {
  fn fmt(
    &self,
    f: &mut std::fmt::Formatter<'_>,
  ) -> Result<(), std::fmt::Error> {
    let string = match self {
      UtilityErrorKind::Unknown => "an unknown error has occured",
      UtilityErrorKind::UnexpectedByte => "a reader has encountered an unexpected byte",
      UtilityErrorKind::UnexpectedEof => "a reader has encountered an unexpected end of file",
      UtilityErrorKind::IoError => "encountered a std::io::Error",
      UtilityErrorKind::Utf8Decode => "encountered a std::string::FromUtf8Error",
      UtilityErrorKind::JsonDeserializer => "an error occured while deserializing json format",
    };

    write!(f, "{}", string)?;
    Ok(())
  }
}
