#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple, PyType};

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

    let semisweet_error =
        new_exception(py, &type_factory, "SemisweetError", PyTuple::new(py, [exception])?)?;
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

    error::register_exceptions(py, &config_error, &namespace_error, &backend_error, &daemon_error);
    Ok(())
}

/// The `semisweet` Python extension module.
#[pymodule]
fn semisweet(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python::PyCacheQuery>()?;
    m.add_class::<python::PySemanticCache>()?;
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
    Ok(())
}
