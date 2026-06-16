use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::sync::PyOnceLock;
use pyo3::types::PyType;
use pyo3::{Bound, Py, PyErr, Python};

use crate::protocol::ProtocolError;

pub type Result<T> = std::result::Result<T, Error>;

pub type BackendError = Box<dyn std::error::Error + Send + Sync>;

static CONFIG_ERROR: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static NAMESPACE_ERROR: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static BACKEND_ERROR: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static DAEMON_ERROR: PyOnceLock<Py<PyType>> = PyOnceLock::new();

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("query text is empty")]
    EmptyQuery,
    #[error("context is empty")]
    EmptyContext,
    #[error("key is empty")]
    EmptyKey,
    #[error("entity is empty")]
    EmptyEntity,
    #[error("namespace is empty")]
    EmptyNamespace,
    #[error("invalid namespace: {0}")]
    InvalidNamespace(String),
    #[error("embedding is empty")]
    EmptyEmbedding,
    #[error("embedding has zero norm")]
    ZeroEmbedding,
    #[error("embedding has non-finite component or norm")]
    NonFiniteEmbedding,
    #[error("embedding dimension {got} != index dimension {want}")]
    DimMismatch { got: usize, want: usize },
    #[error("required environment variable `{0}` is not set")]
    MissingEnv(&'static str),
    #[error("unknown backend `{0}`")]
    UnknownBackend(String),
    #[error("invalid namespace config: {0}")]
    InvalidConfig(String),
    #[error("no namespace `{0}`")]
    NamespaceMissing(String),
    #[error("entity extraction failed")]
    EntityExtraction(#[source] BackendError),
    #[error("embedding failed")]
    Embedding(#[source] BackendError),
    #[error("vector storage failed")]
    VectorStorage(#[source] BackendError),
    #[error("object storage failed")]
    ObjectStorage(#[source] BackendError),
    #[error("daemon is shutting down")]
    DaemonShutdown,
    #[error("daemon lifecycle error: {0}")]
    Daemon(String),
    #[error("protocol version mismatch: client {client}, daemon {daemon}")]
    ProtocolVersionMismatch { client: u32, daemon: u32 },
    #[error("daemon i/o failed")]
    Io(#[from] std::io::Error),
    #[error("protocol codec failed")]
    Codec(#[from] postcard::Error),
}

fn build(cell: &PyOnceLock<Py<PyType>>, fallback: fn(String) -> PyErr, message: String) -> PyErr {
    Python::attach(|py| match cell.get(py) {
        Some(ty) => PyErr::from_type(ty.bind(py).clone(), message),
        None => fallback(message),
    })
}

pub(crate) fn register_exceptions(
    py: Python<'_>,
    config: &Bound<'_, PyType>,
    namespace: &Bound<'_, PyType>,
    backend: &Bound<'_, PyType>,
    daemon: &Bound<'_, PyType>,
) {
    let _ = CONFIG_ERROR.set(py, config.clone().unbind());
    let _ = NAMESPACE_ERROR.set(py, namespace.clone().unbind());
    let _ = BACKEND_ERROR.set(py, backend.clone().unbind());
    let _ = DAEMON_ERROR.set(py, daemon.clone().unbind());
}

impl From<Error> for PyErr {
    fn from(err: Error) -> PyErr {
        let message = err.to_string();
        match err {
            Error::EmptyQuery
            | Error::EmptyContext
            | Error::EmptyKey
            | Error::EmptyEntity
            | Error::EmptyNamespace
            | Error::InvalidNamespace(_)
            | Error::EmptyEmbedding
            | Error::ZeroEmbedding
            | Error::NonFiniteEmbedding
            | Error::DimMismatch { .. }
            | Error::MissingEnv(_)
            | Error::UnknownBackend(_)
            | Error::InvalidConfig(_)
            | Error::ProtocolVersionMismatch { .. } => {
                build(&CONFIG_ERROR, PyValueError::new_err, message)
            }
            Error::NamespaceMissing(_) => build(&NAMESPACE_ERROR, PyKeyError::new_err, message),
            Error::EntityExtraction(_)
            | Error::Embedding(_)
            | Error::VectorStorage(_)
            | Error::ObjectStorage(_) => build(&BACKEND_ERROR, PyRuntimeError::new_err, message),
            Error::DaemonShutdown | Error::Daemon(_) | Error::Io(_) | Error::Codec(_) => {
                build(&DAEMON_ERROR, PyRuntimeError::new_err, message)
            }
        }
    }
}

impl From<ProtocolError> for PyErr {
    fn from(err: ProtocolError) -> PyErr {
        match err {
            ProtocolError::InvalidRequest(message) => {
                build(&CONFIG_ERROR, PyValueError::new_err, message)
            }
            ProtocolError::UnknownNamespace(namespace) => build(
                &NAMESPACE_ERROR,
                PyKeyError::new_err,
                format!("unknown namespace `{namespace}`"),
            ),
            ProtocolError::VersionMismatch { client, daemon } => build(
                &CONFIG_ERROR,
                PyValueError::new_err,
                format!("protocol version mismatch: client {client}, daemon {daemon}"),
            ),
            ProtocolError::BackendInit(message) => {
                build(&BACKEND_ERROR, PyRuntimeError::new_err, message)
            }
        }
    }
}
