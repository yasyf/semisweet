#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple, PyType};

mod backends;
mod cache;
mod client;
mod compression;
mod daemon;
mod embedding;
mod entity;
mod error;
mod newtype;
mod object;
mod paths;
mod protocol;
mod python;
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
    ClientId, PROTOCOL_VERSION, ProtocolError, Request, Response, read_frame, write_frame,
};
pub use registry::{
    DynCache, EmbeddingChoice, EntityChoice, NamespaceConfig, ObjectChoice, ScoringDto,
    VectorChoice, build_cache,
};
pub use scoring::{ContextMode, ScoringConfig};
pub use vector::{Filter, ScoredHit, VectorEntry, VectorStorageBackend};

// The curated public surface, mirroring `semisweet.pyi`'s `__all__`. `PyModule::add*`
// appends every registered name to the module's `__all__` index, so the internal daemon
// entry point `_run_daemon` (registered so the launcher can import it straight from the
// extension submodule — see `client::spawn`) would otherwise leak into the public
// surface. The module sets `__all__` to this list last, overwriting the auto-built one.
const PUBLIC_API: [&str; 17] = [
    "CacheQuery",
    "SemanticCache",
    "LocalEmbedding",
    "VoyageEmbedding",
    "KeywordEntities",
    "GlinerEntities",
    "MemoryVectors",
    "TurbopufferVectors",
    "DiskStorage",
    "S3Storage",
    "Scoring",
    "shutdown_daemon",
    "SemisweetError",
    "ConfigError",
    "NamespaceError",
    "BackendError",
    "DaemonError",
];

fn new_exception<'py>(
    py: Python<'py>,
    type_factory: &Bound<'py, PyAny>,
    name: &str,
    bases: Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyType>> {
    let namespace = PyDict::new(py);
    namespace.set_item("__module__", "semisweet")?;
    let created = type_factory.call1((name, bases, namespace))?;
    Ok(created.cast_into::<PyType>()?)
}

fn add_exceptions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let builtins = py.import("builtins")?;
    let type_factory = builtins.getattr("type")?;
    let exception = builtins.getattr("Exception")?;
    let value_error = builtins.getattr("ValueError")?;
    let key_error = builtins.getattr("KeyError")?;
    let runtime_error = builtins.getattr("RuntimeError")?;

    let semisweet_error = new_exception(
        py,
        &type_factory,
        "SemisweetError",
        PyTuple::new(py, [exception])?,
    )?;
    let base = semisweet_error.as_any();
    let config_error = new_exception(
        py,
        &type_factory,
        "ConfigError",
        PyTuple::new(py, [base.clone(), value_error])?,
    )?;
    let namespace_error = new_exception(
        py,
        &type_factory,
        "NamespaceError",
        PyTuple::new(py, [base.clone(), key_error])?,
    )?;
    let backend_error = new_exception(
        py,
        &type_factory,
        "BackendError",
        PyTuple::new(py, [base.clone(), runtime_error.clone()])?,
    )?;
    let daemon_error = new_exception(
        py,
        &type_factory,
        "DaemonError",
        PyTuple::new(py, [base.clone(), runtime_error])?,
    )?;

    m.add("SemisweetError", &semisweet_error)?;
    m.add("ConfigError", &config_error)?;
    m.add("NamespaceError", &namespace_error)?;
    m.add("BackendError", &backend_error)?;
    m.add("DaemonError", &daemon_error)?;

    error::register_exceptions(
        py,
        &config_error,
        &namespace_error,
        &backend_error,
        &daemon_error,
    );
    Ok(())
}

// pyo3 exposes async methods as native callables, which `asyncio.iscoroutinefunction`
// cannot recognize: a native function carries no `CO_COROUTINE` code flag and has no
// settable `__dict__` for the `_is_coroutine` marker. Wrap the Rust awaitables in real
// Python `async def` coroutine functions so the public surface is both awaitable and
// introspectable. The wrappers resolve `_SemanticCache`/`_shutdown_daemon` from the
// module namespace at call time.
const ASYNC_FACADE: &std::ffi::CStr = cr#"
class SemanticCache(_SemanticCache):
    """A semantic cache scoped to one namespace, backed by a shared daemon process."""

    async def get(self, query):
        """Return the cached value for ``query`` as ``bytes``, or ``None`` on a miss."""
        return await _SemanticCache.get(self, query)

    async def set(self, query, value):
        """Store ``value`` under ``query``; return ``True`` if the daemon accepted it."""
        return await _SemanticCache.set(self, query, value)

    async def delete(self, query):
        """Evict the entry matching ``query``; return ``True`` if one was removed."""
        return await _SemanticCache.delete(self, query)


async def shutdown_daemon():
    """Signal the shared daemon to shut down; return ``True`` if one was running."""
    return await _shutdown_daemon()
"#;

// The Rust class and shutdown function live only in the facade's globals, never as
// module attributes, so the raw pyo3 internals stay unreachable as `semisweet._*`. The
// `async def` wrappers resolve them from their `__globals__` at call time.
fn install_async_facade(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let globals = PyDict::new(py);
    globals.set_item("_SemanticCache", py.get_type::<python::PySemanticCache>())?;
    globals.set_item(
        "_shutdown_daemon",
        pyo3::wrap_pyfunction!(python::shutdown_daemon, m)?,
    )?;
    py.run(ASYNC_FACADE, Some(&globals), None)?;
    m.add(
        "SemanticCache",
        globals
            .get_item("SemanticCache")?
            .ok_or_else(|| Error::Daemon("async facade did not define SemanticCache".to_owned()))?,
    )?;
    m.add(
        "shutdown_daemon",
        globals.get_item("shutdown_daemon")?.ok_or_else(|| {
            Error::Daemon("async facade did not define shutdown_daemon".to_owned())
        })?,
    )?;
    Ok(())
}

/// The compiled `semisweet.core` extension module. The pure-Python `semisweet` package
/// re-exports these symbols and layers the async object cache + `@cache` decorator on top.
#[pymodule]
fn core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python::PyCacheQuery>()?;
    m.add_class::<python::PyLocalEmbedding>()?;
    m.add_class::<python::PyVoyageEmbedding>()?;
    m.add_class::<python::PyKeywordEntities>()?;
    m.add_class::<python::PyGlinerEntities>()?;
    m.add_class::<python::PyMemoryVectors>()?;
    m.add_class::<python::PyTurbopufferVectors>()?;
    m.add_class::<python::PyDiskStorage>()?;
    m.add_class::<python::PyS3Storage>()?;
    m.add_class::<python::PyScoring>()?;
    m.add_function(pyo3::wrap_pyfunction!(python::_run_daemon, m)?)?;
    add_exceptions(m)?;
    install_async_facade(m)?;
    m.add("__all__", PyList::new(m.py(), PUBLIC_API)?)?;
    Ok(())
}
