//! Backend implementations for the cache's pluggable storage and inference traits.
//!
//! `vector_memory` and `object_disk` use only `std` and always compile; the rest
//! are gated behind their backend's Cargo feature.

pub mod object_disk;
pub mod vector_memory;

#[cfg(feature = "voyage")]
pub mod embed_voyage;

#[cfg(feature = "local-embed")]
pub mod embed_local;

#[cfg(feature = "keyword")]
pub mod entity_keyword;

#[cfg(feature = "gliner")]
pub mod entity_gliner;

#[cfg(feature = "turbopuffer")]
pub mod vector_turbopuffer;

#[cfg(feature = "s3")]
pub mod object_s3;

// Serializes every test that mutates a shared process environment variable (e.g. a
// backend's API-key env), so the registry's offline build_cache test and the voyage
// backend's missing-key test never race over the same variable.
#[cfg(test)]
pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
