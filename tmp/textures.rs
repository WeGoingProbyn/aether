use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use image::{ImageFormat, Rgb32FImage, RgbImage, Rgba32FImage, RgbaImage};

use crate::debugger::IcsError;
use crate::structures::{entity::IcsAsset, hierarchy::NodeId};
use crate::utility::file_io::file_to_cursor;
use crate::ICS_ERROR;

/// Enum representing different types of textures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureType {
  Specular,
  Normal,
  Albedo,
  Height,
  Emission,
  AmbientOcclusion,
  Metallic,
  Roughness,
  UnknownTexture,
}

impl fmt::Display for TextureType {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let texture_type_name = match self {
      TextureType::Specular => "Specular",
      TextureType::Normal => "Normal",
      TextureType::Albedo => "Albedo",
      TextureType::Height => "Height",
      TextureType::Emission => "Emission",
      TextureType::AmbientOcclusion => "AmbientOcclusion",
      TextureType::Metallic => "Metallic",
      TextureType::Roughness => "Roughness",
      TextureType::UnknownTexture => "UnknownTexture",
    };
    write!(f, "{}", texture_type_name)
  }
}

/// Enum representing different pixel formats for textures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
  Rgba8,
  Rgb8,
  Rgb32f,
  Rgba32f,
}

pub enum ImageData {
  Rgb8(RgbImage),
  Rgba8(RgbaImage),
  Rgb32f(Rgb32FImage),
  Rgba32f(Rgba32FImage),
}

/// Texture filtering mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FilterMode {
  /// Nearest-neighbor filtering (pixelated).
  Nearest,
  /// Linear interpolation (smooth).
  #[default]
  Linear,
}

/// Texture address/wrap mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum WrapMode {
  /// Repeat the texture (tile).
  #[default]
  Repeat,
  /// Mirror the texture at each boundary.
  MirroredRepeat,
  /// Clamp to edge pixels.
  ClampToEdge,
  /// Clamp to border color.
  ClampToBorder,
}

/// Border color for ClampToBorder wrap mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BorderColor {
  /// Transparent black (0, 0, 0, 0).
  TransparentBlack,
  /// Opaque black (0, 0, 0, 1).
  #[default]
  OpaqueBlack,
  /// Opaque white (1, 1, 1, 1).
  OpaqueWhite,
}

/// Mipmap filtering mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MipmapMode {
  /// Nearest mip level selection.
  Nearest,
  /// Linear interpolation between mip levels.
  #[default]
  Linear,
}

/// Mipmap generation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MipmapStrategy {
  /// Generate full mip chain automatically.
  #[default]
  GenerateFull,
  /// Use only the base level (no mipmaps).
  None,
  /// Use a fixed number of mip levels.
  FixedLevels(u32),
}

// ============================================================================
// Streaming/Partial Residency Hooks (Future Extension Points)
// ============================================================================

/// Hint for texture streaming priority.
///
/// These hints help the streaming system determine which textures
/// to prioritize for loading/unloading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum StreamingPriority {
  /// Texture should remain resident at all times.
  AlwaysResident,
  /// High priority - load quickly when needed.
  High,
  /// Normal priority - default streaming behavior.
  #[default]
  Normal,
  /// Low priority - can be unloaded when memory is scarce.
  Low,
}

/// Current residency state for a streaming texture.
///
/// Tracks which mip levels are currently loaded in GPU memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ResidencyState {
  /// All requested mip levels are resident.
  #[default]
  FullyResident,
  /// Only some mip levels are resident (partial residency).
  PartiallyResident {
    /// The lowest (most detailed) mip level currently loaded.
    min_resident_mip: u32,
    /// The highest (least detailed) mip level currently loaded.
    max_resident_mip: u32,
  },
  /// No mip levels are currently resident.
  NotResident,
  /// Texture is currently being streamed in.
  Streaming,
}

/// Streaming configuration for a texture.
///
/// This struct contains all streaming-related settings that can be
/// configured per-texture for future streaming implementations.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamingConfig {
  /// Priority hint for the streaming system.
  pub priority: StreamingPriority,
  /// Minimum mip level to keep resident (0 = highest detail).
  pub min_resident_mip: u32,
  /// Maximum mip level to stream (higher = lower detail fallback).
  pub max_streaming_mip: u32,
  /// Whether this texture participates in streaming at all.
  pub streaming_enabled: bool,
}

impl Default for StreamingConfig {
  fn default() -> Self {
    Self {
      priority: StreamingPriority::Normal,
      min_resident_mip: 0,
      max_streaming_mip: u32::MAX,
      streaming_enabled: false, // Disabled by default until streaming is implemented
    }
  }
}

impl StreamingConfig {
  /// Creates a configuration for always-resident textures.
  pub fn always_resident() -> Self {
    Self {
      priority: StreamingPriority::AlwaysResident,
      streaming_enabled: false,
      ..Self::default()
    }
  }

  /// Creates a configuration for streamable textures.
  pub fn streamable(priority: StreamingPriority) -> Self {
    Self {
      priority,
      streaming_enabled: true,
      ..Self::default()
    }
  }

  /// Builder: set minimum resident mip level.
  pub fn with_min_resident_mip(mut self, mip: u32) -> Self {
    self.min_resident_mip = mip;
    self
  }
}

/// Per-texture sampler settings.
///
/// These settings control how the GPU samples the texture, including
/// filtering, addressing, and mipmap behavior.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplerSettings {
  /// Magnification filter (when texture is enlarged).
  pub mag_filter: FilterMode,
  /// Minification filter (when texture is shrunk).
  pub min_filter: FilterMode,
  /// Mipmap filtering mode.
  pub mipmap_mode: MipmapMode,
  /// Address mode for U (horizontal) coordinate.
  pub wrap_u: WrapMode,
  /// Address mode for V (vertical) coordinate.
  pub wrap_v: WrapMode,
  /// Address mode for W (depth) coordinate.
  pub wrap_w: WrapMode,
  /// Border color for ClampToBorder mode.
  pub border_color: BorderColor,
  /// Enable anisotropic filtering.
  pub anisotropy_enable: bool,
  /// Maximum anisotropy level (1.0 to 16.0, will be clamped to device limits).
  pub max_anisotropy: f32,
  /// Mipmap LOD bias.
  pub mip_lod_bias: f32,
  /// Minimum LOD level.
  pub min_lod: f32,
  /// Maximum LOD level (None = use texture's mip levels).
  pub max_lod: Option<f32>,
  /// Mipmap generation strategy.
  pub mipmap_strategy: MipmapStrategy,
  /// Enable depth comparison (for shadow maps).
  pub compare_enable: bool,
}

impl Default for SamplerSettings {
  fn default() -> Self {
    Self {
      mag_filter: FilterMode::Linear,
      min_filter: FilterMode::Linear,
      mipmap_mode: MipmapMode::Linear,
      wrap_u: WrapMode::Repeat,
      wrap_v: WrapMode::Repeat,
      wrap_w: WrapMode::Repeat,
      border_color: BorderColor::OpaqueBlack,
      anisotropy_enable: true,
      max_anisotropy: 16.0,
      mip_lod_bias: 0.0,
      min_lod: 0.0,
      max_lod: None,
      mipmap_strategy: MipmapStrategy::GenerateFull,
      compare_enable: false,
    }
  }
}

impl SamplerSettings {
  /// Creates settings for a standard diffuse/albedo texture.
  pub fn diffuse() -> Self {
    Self::default()
  }

  /// Creates settings for a normal map (linear filtering, no sRGB).
  pub fn normal_map() -> Self {
    Self::default()
  }

  /// Creates settings for a pixel-art texture (nearest filtering, no mipmaps).
  pub fn pixel_art() -> Self {
    Self {
      mag_filter: FilterMode::Nearest,
      min_filter: FilterMode::Nearest,
      mipmap_mode: MipmapMode::Nearest,
      anisotropy_enable: false,
      max_anisotropy: 1.0,
      mipmap_strategy: MipmapStrategy::None,
      ..Self::default()
    }
  }

  /// Creates settings for a UI texture (linear, clamp to edge).
  pub fn ui() -> Self {
    Self {
      wrap_u: WrapMode::ClampToEdge,
      wrap_v: WrapMode::ClampToEdge,
      wrap_w: WrapMode::ClampToEdge,
      anisotropy_enable: false,
      max_anisotropy: 1.0,
      mipmap_strategy: MipmapStrategy::None,
      ..Self::default()
    }
  }

  /// Creates settings for a skybox/cubemap texture.
  pub fn skybox() -> Self {
    Self {
      wrap_u: WrapMode::ClampToEdge,
      wrap_v: WrapMode::ClampToEdge,
      wrap_w: WrapMode::ClampToEdge,
      ..Self::default()
    }
  }

  /// Creates settings for a shadow map.
  pub fn shadow_map() -> Self {
    Self {
      mag_filter: FilterMode::Linear,
      min_filter: FilterMode::Linear,
      wrap_u: WrapMode::ClampToBorder,
      wrap_v: WrapMode::ClampToBorder,
      wrap_w: WrapMode::ClampToBorder,
      border_color: BorderColor::OpaqueWhite,
      anisotropy_enable: false,
      max_anisotropy: 1.0,
      compare_enable: true,
      mipmap_strategy: MipmapStrategy::None,
      ..Self::default()
    }
  }

  /// Builder: set magnification filter.
  pub fn with_mag_filter(mut self, filter: FilterMode) -> Self {
    self.mag_filter = filter;
    self
  }

  /// Builder: set minification filter.
  pub fn with_min_filter(mut self, filter: FilterMode) -> Self {
    self.min_filter = filter;
    self
  }

  /// Builder: set wrap mode for all axes.
  pub fn with_wrap(mut self, mode: WrapMode) -> Self {
    self.wrap_u = mode;
    self.wrap_v = mode;
    self.wrap_w = mode;
    self
  }

  /// Builder: set anisotropy level.
  pub fn with_anisotropy(mut self, level: f32) -> Self {
    self.anisotropy_enable = level > 1.0;
    self.max_anisotropy = level.clamp(1.0, 16.0);
    self
  }

  /// Builder: disable anisotropy.
  pub fn without_anisotropy(mut self) -> Self {
    self.anisotropy_enable = false;
    self.max_anisotropy = 1.0;
    self
  }

  /// Builder: set mipmap strategy.
  pub fn with_mipmap_strategy(mut self, strategy: MipmapStrategy) -> Self {
    self.mipmap_strategy = strategy;
    self
  }

  /// Builder: set max LOD.
  pub fn with_max_lod(mut self, max_lod: f32) -> Self {
    self.max_lod = Some(max_lod);
    self
  }

  /// Generates a hash for sampler caching.
  pub fn cache_hash(&self) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    self.mag_filter.hash(&mut hasher);
    self.min_filter.hash(&mut hasher);
    self.mipmap_mode.hash(&mut hasher);
    self.wrap_u.hash(&mut hasher);
    self.wrap_v.hash(&mut hasher);
    self.wrap_w.hash(&mut hasher);
    self.border_color.hash(&mut hasher);
    self.anisotropy_enable.hash(&mut hasher);
    // Hash anisotropy as integer to avoid float hashing issues
    ((self.max_anisotropy * 10.0) as u32).hash(&mut hasher);
    ((self.mip_lod_bias * 100.0) as i32).hash(&mut hasher);
    ((self.min_lod * 100.0) as u32).hash(&mut hasher);
    if let Some(max) = self.max_lod {
      1u8.hash(&mut hasher);
      ((max * 100.0) as u32).hash(&mut hasher);
    } else {
      0u8.hash(&mut hasher);
    }
    self.compare_enable.hash(&mut hasher);
    hasher.finish()
  }
}

impl fmt::Display for PixelFormat {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let format_name = match self {
      PixelFormat::Rgba8 => "RGBA8",
      PixelFormat::Rgb8 => "RGB8",
      PixelFormat::Rgb32f => "RGB32F",
      PixelFormat::Rgba32f => "RGBA32F",
    };
    write!(f, "{}", format_name)
  }
}

/// Represents a texture, containing its data and type.
#[derive(Debug, Clone)]
pub struct Texture {
  data: Vec<u8>,
  width: u32,
  height: u32,
  pixel_format: PixelFormat,
  /// Sampler settings for this texture.
  sampler: SamplerSettings,
  /// Streaming configuration (for future streaming support).
  streaming: StreamingConfig,
}

impl Texture {
  pub fn from_file(pixel_format: PixelFormat, path: &str) -> Result<Self, IcsError> {
    let buffer = file_to_cursor(path)?;

    let extension = path.rsplit_once('.').map(|(_, ext)| ext).ok_or_else(|| {
      ICS_ERROR!(
        why: "Texture: Missing file extension",
        fix: "Provide a file path with an image extension"
      )
    })?;
    let image_format = match extension {
      "png" => ImageFormat::Png,
      "jpg" | "jpeg" => ImageFormat::Jpeg,
      _ => {
        return Err(ICS_ERROR!(
          why: format!("Texture: Unsupported image extension '{}'", extension),
          fix: "Provide a .png or .jpg texture"
        ))
      }
    };
    Self::from_bytes_with_format(pixel_format, buffer.into_inner(), image_format)
  }

  /// Creates a Texture from raw bytes with automatic format detection.
  ///
  /// # Arguments
  /// * `pixel_format` - The desired output pixel format
  /// * `bytes` - Raw image file bytes (PNG, JPEG, etc.)
  ///
  /// # Returns
  /// A new Texture instance, or an error if decoding fails.
  pub fn from_bytes(pixel_format: PixelFormat, bytes: &[u8]) -> Result<Self, IcsError> {
    let format = image::guess_format(bytes).map_err(|e| {
      ICS_ERROR!(
        why: "Texture: Failed to detect image format from bytes",
        fix: "Ensure the bytes contain a valid PNG or JPEG image",
        src: e
      )
    })?;
    Self::from_bytes_with_format(pixel_format, bytes.to_vec(), format)
  }

  /// Creates a Texture from raw bytes with a specified format.
  fn from_bytes_with_format(
    pixel_format: PixelFormat,
    bytes: Vec<u8>,
    image_format: ImageFormat,
  ) -> Result<Self, IcsError> {
    let buffer = std::io::Cursor::new(bytes);
    let image = image::load(buffer, image_format).map_err(|e| {
      ICS_ERROR!(
        why: "Texture: Failed to decode image",
        fix: "Ensure the image file is valid and readable",
        src: e
      )
    })?;
    let image_as = match pixel_format {
      PixelFormat::Rgb8 => ImageData::Rgb8(image.to_rgb8()),
      PixelFormat::Rgba8 => ImageData::Rgba8(image.to_rgba8()),
      PixelFormat::Rgb32f => ImageData::Rgb32f(image.to_rgb32f()),
      PixelFormat::Rgba32f => ImageData::Rgba32f(image.to_rgba32f()),
    };
    let (width, height, data) = match image_as {
      ImageData::Rgb8(image) => {
        let width = image.width();
        let height = image.height();
        let data = image.into_raw();
        (width, height, data)
      }
      ImageData::Rgba8(image) => {
        let width = image.width();
        let height = image.height();
        let data = image.into_raw();
        (width, height, data)
      }
      ImageData::Rgb32f(image) => {
        let width = image.width();
        let height = image.height();
        let data = Texture::convert_f32_to_u8(&image.into_raw());
        (width, height, data)
      }
      ImageData::Rgba32f(image) => {
        let width = image.width();
        let height = image.height();
        let data = Texture::convert_f32_to_u8(&image.into_raw());
        (width, height, data)
      }
    };
    Ok(Texture {
      data,
      width,
      height,
      pixel_format,
      sampler: SamplerSettings::default(),
      streaming: StreamingConfig::default(),
    })
  }

  /// Creates a new Texture with the specified type and image data.
  ///
  /// # Arguments
  ///
  /// * texture_type - The type of the texture.
  /// * data - The raw image data of the texture.
  /// * width - The width of the texture in pixels.
  /// * height - The height of the texture in pixels.
  /// * pixel_format - The pixel format of the texture.
  ///
  /// # Returns
  ///
  /// A new instance of Texture.
  pub fn new(pixel_format: PixelFormat, path: &str) -> Self {
    Self::from_file(pixel_format, path).unwrap_or_else(|err| {
      panic!("Texture::new failed: {}", err);
    })
  }

  pub fn from_rgba8(width: u32, height: u32, data: Vec<u8>) -> Result<Self, IcsError> {
    let expected = width as usize * height as usize * 4;
    if data.len() != expected {
      return Err(ICS_ERROR!(
        why: format!(
          "Texture: RGBA8 data size {} does not match expected {}",
          data.len(),
          expected
        ),
        fix: "Ensure RGBA8 byte length is width * height * 4"
      ));
    }
    Ok(Texture {
      data,
      width,
      height,
      pixel_format: PixelFormat::Rgba8,
      sampler: SamplerSettings::default(),
      streaming: StreamingConfig::default(),
    })
  }

  fn convert_f32_to_u8(data_f32: &[f32]) -> Vec<u8> {
    data_f32
      .iter()
      .map(|&value| {
        // Clamp the value to [0.0, 1.0]
        let clamped = value.clamp(0.0, 1.0);
        // Scale to [0, 255], round to nearest integer
        ((clamped * 255.0).round()) as u8
      })
      .collect()
  }

  /// Returns a reference to the raw image data of the texture.
  pub fn data(&self) -> &Vec<u8> {
    &self.data
  }

  /// Returns the dimensions of the texture as a tuple (width, height).
  pub fn dimensions(&self) -> (u32, u32) {
    (self.width, self.height)
  }

  /// Returns the pixel format of the texture.
  pub fn pixel_format(&self) -> PixelFormat {
    self.pixel_format
  }

  /// Returns the size of the texture data in bytes.
  pub fn size(&self) -> usize {
    self.data.len()
  }

  /// Returns the sampler settings for this texture.
  pub fn sampler(&self) -> &SamplerSettings {
    &self.sampler
  }

  /// Sets the sampler settings for this texture.
  pub fn set_sampler(&mut self, settings: SamplerSettings) {
    self.sampler = settings;
  }

  /// Builder: set sampler settings.
  pub fn with_sampler(mut self, settings: SamplerSettings) -> Self {
    self.sampler = settings;
    self
  }

  /// Returns the number of mip levels based on the mipmap strategy.
  pub fn mip_levels(&self) -> u32 {
    match self.sampler.mipmap_strategy {
      MipmapStrategy::None => 1,
      MipmapStrategy::FixedLevels(levels) => levels.min(self.max_mip_levels()),
      MipmapStrategy::GenerateFull => self.max_mip_levels(),
    }
  }

  /// Returns the maximum possible mip levels for this texture's dimensions.
  pub fn max_mip_levels(&self) -> u32 {
    let max_dim = self.width.max(self.height);
    (max_dim as f32).log2().floor() as u32 + 1
  }

  // =========================================================================
  // Streaming Configuration (Future Extension Points)
  // =========================================================================

  /// Returns the streaming configuration for this texture.
  pub fn streaming(&self) -> &StreamingConfig {
    &self.streaming
  }

  /// Sets the streaming configuration for this texture.
  pub fn set_streaming(&mut self, config: StreamingConfig) {
    self.streaming = config;
  }

  /// Builder: set streaming configuration.
  pub fn with_streaming(mut self, config: StreamingConfig) -> Self {
    self.streaming = config;
    self
  }

  /// Returns whether this texture is configured for streaming.
  pub fn is_streamable(&self) -> bool {
    self.streaming.streaming_enabled
  }

  /// Returns the streaming priority for this texture.
  pub fn streaming_priority(&self) -> StreamingPriority {
    self.streaming.priority
  }
}

impl fmt::Display for Texture {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    writeln!(
      f,
      "Texture (Format: {}, Dimensions: {}x{})",
      self.pixel_format, self.width, self.height
    )
  }
}

/// Represents a collection of textures along with their buffer and layout.
#[derive(Debug, Clone)]
pub struct Textures {
  // HashMap<node_name, HashMap<type, Vec<(texture_name, texture)>>>
  map: HashMap<NodeId, HashMap<TextureType, Vec<(String, Arc<Texture>)>>>,
}

impl Textures {
  pub fn new() -> Textures {
    Textures {
      map: HashMap::new(),
    }
  }

  /// Adds a new texture to a specific hierarchy node.
  ///
  /// # Arguments
  ///
  /// * `node_name` - The name of the hierarchy node.
  /// * `texture_type` - The type of the texture.
  /// * `texture` - The Arc-wrapped Texture to add.
  ///
  /// # Returns
  ///
  /// A Result indicating success or containing an IcsError.
  pub fn add_texture(
    &mut self,
    node_name: &NodeId,
    texture_name: &str,
    texture_type: TextureType,
    texture: Arc<Texture>,
  ) -> Result<(), IcsError> {
    let set = self
      .map
      .entry(node_name.clone())
      .or_insert_with(HashMap::new);

    set
      .entry(texture_type)
      .or_insert_with(|| Vec::new())
      .push((texture_name.to_string(), texture));
    Ok(())
  }

  /// Retrieves textures from a specific hierarchy node and texture type.
  ///
  /// # Arguments
  ///
  /// * `node_name` - The name of the hierarchy node.
  /// * `texture_type` - The type of the texture.
  ///
  /// # Returns
  ///
  /// An Option containing a reference to the Vec<(String, Arc<Texture>)> if found.
  pub fn textures(
    &self,
    node_name: &NodeId,
    texture_type: TextureType,
  ) -> Option<&Vec<(String, Arc<Texture>)>> {
    self.map.get(node_name)?.get(&texture_type)
  }

  /// Retrieves all textures for a specific hierarchy node.
  ///
  /// # Arguments
  ///
  /// * `node_name` - The name of the hierarchy node.
  ///
  /// # Returns
  ///
  /// An Option containing a reference to the TextureSet if found.
  pub fn textures_for(
    &self,
    node_name: &NodeId,
  ) -> Option<&HashMap<TextureType, Vec<(String, Arc<Texture>)>>> {
    self.map.get(node_name)
  }

  /// Retrieves all texture sets.
  pub fn map(&self) -> &HashMap<NodeId, HashMap<TextureType, Vec<(String, Arc<Texture>)>>> {
    &self.map
  }
}

impl fmt::Display for Textures {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    writeln!(f, "Textures:")?;
    for (node, textures) in &self.map {
      writeln!(f, "  Node '{}':", node)?;
      for (tex_type, texs) in textures {
        writeln!(f, "   Type '{}':", tex_type)?;
        for (tex_name, tex) in texs {
          writeln!(f, "     {}: {}", tex_name, tex)?;
        }
      }
    }
    Ok(())
  }
}

impl IcsAsset for Texture {}
