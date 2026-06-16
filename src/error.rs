use pyo3::PyErr;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};

use crate::protocol::ProtocolError;

pub type Result<T> = std::result::Result<T, Error>;

pub type BackendError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("query text is empty")]
    EmptyQuery,
    #[error("key is empty")]
    EmptyKey,
    #[error("entity is empty")]
    EmptyEntity,
    #[error("namespace is empty")]
    EmptyNamespace,
    #[error("embedding is empty")]
    EmptyEmbedding,
    #[error("embedding has zero norm")]
    ZeroEmbedding,
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

impl From<Error> for PyErr {
    fn from(err: Error) -> PyErr {
        match err {
            Error::EmptyQuery
            | Error::EmptyKey
            | Error::EmptyEntity
            | Error::EmptyNamespace
            | Error::EmptyEmbedding
            | Error::ZeroEmbedding
            | Error::DimMismatch { .. }
            | Error::MissingEnv(_)
            | Error::UnknownBackend(_)
            | Error::InvalidConfig(_) => PyValueError::new_err(err.to_string()),
            Error::ProtocolVersionMismatch { .. } => PyValueError::new_err(err.to_string()),
            Error::NamespaceMissing(_) => PyKeyError::new_err(err.to_string()),
            Error::EntityExtraction(_)
            | Error::Embedding(_)
            | Error::VectorStorage(_)
            | Error::ObjectStorage(_)
            | Error::DaemonShutdown
            | Error::Daemon(_)
            | Error::Io(_)
            | Error::Codec(_) => PyRuntimeError::new_err(err.to_string()),
        }
    }
}

impl From<ProtocolError> for PyErr {
    fn from(err: ProtocolError) -> PyErr {
        match err {
            ProtocolError::InvalidRequest(message) => PyValueError::new_err(message),
            ProtocolError::UnknownNamespace(namespace) => {
                PyKeyError::new_err(format!("unknown namespace `{namespace}`"))
            }
            ProtocolError::VersionMismatch { client, daemon } => PyRuntimeError::new_err(format!(
                "protocol version mismatch: client {client}, daemon {daemon}"
            )),
            ProtocolError::BackendInit(message) => PyRuntimeError::new_err(message),
        }
    }
}
