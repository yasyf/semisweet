#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use pyo3::prelude::*;

mod backends;
mod cache;
mod client;
mod daemon;
mod embedding;
mod entity;
mod error;
mod newtype;
mod object;
mod paths;
mod protocol;
mod registry;
mod scoring;
mod vector;

pub use cache::Cache;
pub use client::{ClientStub, Launcher, connect_or_spawn};
pub use daemon::run_daemon;
pub use embedding::EmbeddingBackend;
pub use entity::EntityBackend;
pub use error::{BackendError, Error, Result};
pub use newtype::{Context, Dim, Embedding, Entity, EntryId, Key, Namespace, QueryText};
pub use object::ObjectStorageBackend;
pub use paths::{lock_path, log_path, pid_path, socket_path, spawn_lock_path};
pub use protocol::{
    ClientId, PROTOCOL_VERSION, ProtocolError, ProtocolVersion, Request, Response, read_frame,
    write_frame,
};
pub use registry::{
    DynCache, EmbeddingChoice, EntityChoice, NamespaceConfig, ObjectChoice, ScoringDto,
    VectorChoice, build_cache,
};
pub use scoring::{ContextMode, ScoringConfig};
pub use vector::{Filter, ScoredHit, VectorEntry, VectorStorageBackend};

/// The `semisweet` Python extension module.
#[pymodule]
fn semisweet(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
