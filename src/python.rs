//! The pyo3 binding layer. Every `#[pyfunction]`/`#[pymethods]` here is a thin
//! wrapper: it converts Python values in, calls plain Rust, and converts results
//! out. No business logic lives at this edge — validation and RPC live in the
//! newtype, registry, and client modules. Errors cross the boundary through the
//! single `From<Error>`/`From<ProtocolError>` conversions in `crate::error`.
//!
//! The public surface is declarative and keyword-only: a cache is built by passing
//! backend objects (`LocalEmbedding`, `KeywordEntities`, `MemoryVectors`, …) to
//! `SemanticCache`, each of which is itself a thin keyword-only constructor over the
//! serde `*Choice` config the daemon understands. Every argument is optional; an
//! omitted axis falls back to the fully-local default.

use std::borrow::Cow;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Mutex;

use pyo3::prelude::*;

use crate::client::{ClientStub, Launcher, connect_or_spawn};
use crate::error::{Error, Result};
use crate::newtype::{Context, Key, Namespace, QueryText};
use crate::protocol::{Request, Response};
use crate::registry::{
    EmbeddingChoice, EntityChoice, NamespaceConfig, ObjectChoice, ScoringDto, VectorChoice,
};

const DEFAULT_GLINER_LABELS: [&str; 6] = [
    "person",
    "organization",
    "location",
    "date",
    "product",
    "event",
];

fn register(
    executable: PathBuf,
    namespace: String,
    config_json: String,
) -> Result<(ClientStub, Response)> {
    let mut stub = connect_or_spawn(&Launcher::Python { executable })?;
    let response = stub.request(&Request::RegisterNamespace {
        namespace,
        config_json,
    })?;
    Ok((stub, response))
}

#[pyclass(name = "CacheQuery", frozen)]
pub(crate) struct PyCacheQuery {
    query: String,
    keys: Vec<String>,
    context: Option<String>,
}

#[pymethods]
impl PyCacheQuery {
    #[new]
    #[pyo3(signature = (*, query, keys = None, context = None))]
    fn new(query: String, keys: Option<HashSet<String>>, context: Option<String>) -> Result<Self> {
        let query = QueryText::new(query)?.as_str().to_owned();
        let keys = keys.unwrap_or_default();
        let mut validated = Vec::with_capacity(keys.len());
        for key in keys {
            validated.push(Key::new(key)?.as_str().to_owned());
        }
        let context = match context {
            Some(context) => Some(Context::new(context)?.as_str().to_owned()),
            None => None,
        };
        Ok(Self {
            query,
            keys: validated,
            context,
        })
    }
}

// --- Embedding backends ---

#[pyclass(name = "LocalEmbedding", frozen)]
pub(crate) struct PyLocalEmbedding {
    choice: EmbeddingChoice,
}

#[pymethods]
impl PyLocalEmbedding {
    #[new]
    #[pyo3(signature = (*, model = None))]
    fn new(model: Option<String>) -> Self {
        Self {
            choice: EmbeddingChoice::Local { model },
        }
    }
}

#[pyclass(name = "VoyageEmbedding", frozen)]
pub(crate) struct PyVoyageEmbedding {
    choice: EmbeddingChoice,
}

#[pymethods]
impl PyVoyageEmbedding {
    #[new]
    #[pyo3(signature = (*, model = None, dim = None))]
    fn new(model: Option<String>, dim: Option<usize>) -> Self {
        Self {
            choice: EmbeddingChoice::Voyage { model, dim },
        }
    }
}

#[derive(FromPyObject)]
enum EmbeddingArg<'py> {
    Local(PyRef<'py, PyLocalEmbedding>),
    Voyage(PyRef<'py, PyVoyageEmbedding>),
}

impl EmbeddingArg<'_> {
    fn choice(&self) -> EmbeddingChoice {
        match self {
            Self::Local(backend) => backend.choice.clone(),
            Self::Voyage(backend) => backend.choice.clone(),
        }
    }
}

// --- Entity backends ---

#[pyclass(name = "KeywordEntities", frozen)]
pub(crate) struct PyKeywordEntities {
    choice: EntityChoice,
}

#[pymethods]
impl PyKeywordEntities {
    #[new]
    #[pyo3(signature = (*, lang = None))]
    fn new(lang: Option<String>) -> Self {
        Self {
            choice: EntityChoice::Keyword { language: lang },
        }
    }
}

#[pyclass(name = "GlinerEntities", frozen)]
pub(crate) struct PyGlinerEntities {
    choice: EntityChoice,
}

#[pymethods]
impl PyGlinerEntities {
    #[new]
    #[pyo3(signature = (*, labels = None, repo = None, model = None, tokenizer = None))]
    fn new(
        labels: Option<Vec<String>>,
        repo: Option<String>,
        model: Option<String>,
        tokenizer: Option<String>,
    ) -> Self {
        let labels = labels.unwrap_or_else(|| {
            DEFAULT_GLINER_LABELS
                .iter()
                .map(|label| (*label).to_owned())
                .collect()
        });
        Self {
            choice: EntityChoice::Gliner {
                labels,
                repo,
                model,
                tokenizer,
            },
        }
    }
}

#[derive(FromPyObject)]
enum EntityArg<'py> {
    Keyword(PyRef<'py, PyKeywordEntities>),
    Gliner(PyRef<'py, PyGlinerEntities>),
}

impl EntityArg<'_> {
    fn choice(&self) -> EntityChoice {
        match self {
            Self::Keyword(backend) => backend.choice.clone(),
            Self::Gliner(backend) => backend.choice.clone(),
        }
    }
}

// --- Vector index backends ---

#[pyclass(name = "MemoryVectors", frozen)]
pub(crate) struct PyMemoryVectors {
    choice: VectorChoice,
}

#[pymethods]
impl PyMemoryVectors {
    #[new]
    fn new() -> Self {
        Self {
            choice: VectorChoice::Memory,
        }
    }
}

#[pyclass(name = "TurbopufferVectors", frozen)]
pub(crate) struct PyTurbopufferVectors {
    choice: VectorChoice,
}

#[pymethods]
impl PyTurbopufferVectors {
    #[new]
    fn new() -> Self {
        Self {
            choice: VectorChoice::Turbopuffer,
        }
    }
}

#[derive(FromPyObject)]
enum VectorArg<'py> {
    Memory(PyRef<'py, PyMemoryVectors>),
    Turbopuffer(PyRef<'py, PyTurbopufferVectors>),
}

impl VectorArg<'_> {
    fn choice(&self) -> VectorChoice {
        match self {
            Self::Memory(backend) => backend.choice.clone(),
            Self::Turbopuffer(backend) => backend.choice.clone(),
        }
    }
}

// --- Object storage backends ---

#[pyclass(name = "DiskStorage", frozen)]
pub(crate) struct PyDiskStorage {
    choice: ObjectChoice,
}

#[pymethods]
impl PyDiskStorage {
    #[new]
    #[pyo3(signature = (*, root = None))]
    fn new(root: Option<String>) -> Self {
        Self {
            choice: ObjectChoice::Disk { root },
        }
    }
}

#[pyclass(name = "S3Storage", frozen)]
pub(crate) struct PyS3Storage {
    choice: ObjectChoice,
}

#[pymethods]
impl PyS3Storage {
    #[new]
    #[pyo3(signature = (*, bucket = None, region = None, endpoint = None, prefix = None))]
    fn new(
        bucket: Option<String>,
        region: Option<String>,
        endpoint: Option<String>,
        prefix: Option<String>,
    ) -> Self {
        // Thin pass-through: the daemon's S3ObjectStore resolves bucket/region/endpoint
        // (from these kwargs or the environment) alongside the AWS credentials, so the
        // whole S3 config comes from one process.
        Self {
            choice: ObjectChoice::S3 {
                bucket,
                region,
                endpoint,
                prefix: prefix.unwrap_or_default(),
            },
        }
    }
}

#[derive(FromPyObject)]
enum StorageArg<'py> {
    Disk(PyRef<'py, PyDiskStorage>),
    S3(PyRef<'py, PyS3Storage>),
}

impl StorageArg<'_> {
    fn choice(&self) -> ObjectChoice {
        match self {
            Self::Disk(backend) => backend.choice.clone(),
            Self::S3(backend) => backend.choice.clone(),
        }
    }
}

// --- Scoring ---

#[pyclass(name = "Scoring", frozen)]
pub(crate) struct PyScoring {
    dto: ScoringDto,
}

#[pymethods]
impl PyScoring {
    #[new]
    #[pyo3(signature = (*, base = None, floor = None, entity_bonus_weight = None, top_k = None, entity_filter = None, context = None))]
    fn new(
        base: Option<f32>,
        floor: Option<f32>,
        entity_bonus_weight: Option<f32>,
        top_k: Option<usize>,
        entity_filter: Option<bool>,
        context: Option<String>,
    ) -> Result<Self> {
        let defaults = ScoringDto::default();
        let dto = ScoringDto {
            base_threshold: base.unwrap_or(defaults.base_threshold),
            floor_threshold: floor.unwrap_or(defaults.floor_threshold),
            entity_bonus_weight: entity_bonus_weight.unwrap_or(defaults.entity_bonus_weight),
            top_k: top_k.unwrap_or(defaults.top_k),
            entity_filter: entity_filter.unwrap_or(defaults.entity_filter),
            context: context.unwrap_or(defaults.context),
        };
        // Validate eagerly so a bad threshold fails at construction, not at first use.
        dto.to_config()?;
        Ok(Self { dto })
    }
}

// --- The cache itself ---

#[pyclass(name = "SemanticCache")]
pub(crate) struct PySemanticCache {
    namespace: String,
    stub: Mutex<ClientStub>,
}

impl PySemanticCache {
    fn round_trip(&self, request: &Request) -> Result<Response> {
        let mut stub = self
            .stub
            .lock()
            .map_err(|_| Error::Daemon("client connection mutex poisoned".to_owned()))?;
        stub.request(request)
    }
}

#[pymethods]
impl PySemanticCache {
    #[new]
    #[pyo3(signature = (*, namespace, embedding = None, entities = None, vectors = None, storage = None, scoring = None))]
    fn new(
        py: Python<'_>,
        namespace: String,
        embedding: Option<EmbeddingArg<'_>>,
        entities: Option<EntityArg<'_>>,
        vectors: Option<VectorArg<'_>>,
        storage: Option<StorageArg<'_>>,
        scoring: Option<PyRef<'_, PyScoring>>,
    ) -> PyResult<Self> {
        let namespace = Namespace::new(namespace)?.as_str().to_owned();
        let config = NamespaceConfig {
            embedding: embedding
                .map(|arg| arg.choice())
                .unwrap_or(EmbeddingChoice::Local { model: None }),
            entity: entities
                .map(|arg| arg.choice())
                .unwrap_or(EntityChoice::Keyword { language: None }),
            vector: vectors
                .map(|arg| arg.choice())
                .unwrap_or(VectorChoice::Memory),
            object: storage
                .map(|arg| arg.choice())
                .unwrap_or(ObjectChoice::Disk { root: None }),
            scoring: scoring
                .map(|scoring| scoring.dto.clone())
                .unwrap_or_default(),
        };
        let config_json =
            serde_json::to_string(&config).map_err(|e| Error::InvalidConfig(e.to_string()))?;
        let executable: PathBuf = py
            .import("sys")?
            .getattr("executable")?
            .extract::<String>()?
            .into();
        let register_namespace = namespace.clone();
        let (stub, response) =
            py.detach(move || register(executable, register_namespace, config_json))?;
        match response {
            Response::Registered => Ok(Self {
                namespace,
                stub: Mutex::new(stub),
            }),
            Response::Error(error) => Err(error.into()),
            other => Err(Error::Daemon(format!(
                "unexpected response to RegisterNamespace: {other:?}"
            ))
            .into()),
        }
    }

    fn get(&self, py: Python<'_>, query: &PyCacheQuery) -> PyResult<Option<Cow<'static, [u8]>>> {
        let request = Request::Get {
            namespace: self.namespace.clone(),
            query: query.query.clone(),
            keys: query.keys.clone(),
            context: query.context.clone(),
        };
        let response = py.detach(|| self.round_trip(&request))?;
        match response {
            Response::Value(value) => Ok(value.map(|buf| Cow::Owned(buf.into_vec()))),
            Response::Error(error) => Err(error.into()),
            other => Err(Error::Daemon(format!("unexpected response to Get: {other:?}")).into()),
        }
    }

    fn set(&self, py: Python<'_>, query: &PyCacheQuery, value: Vec<u8>) -> PyResult<bool> {
        let request = Request::Set {
            namespace: self.namespace.clone(),
            query: query.query.clone(),
            keys: query.keys.clone(),
            context: query.context.clone(),
            value: serde_bytes::ByteBuf::from(value),
        };
        let response = py.detach(|| self.round_trip(&request))?;
        match response {
            Response::Accepted(accepted) => Ok(accepted),
            Response::Error(error) => Err(error.into()),
            other => Err(Error::Daemon(format!("unexpected response to Set: {other:?}")).into()),
        }
    }

    fn delete(&self, py: Python<'_>, query: &PyCacheQuery) -> PyResult<bool> {
        let request = Request::Del {
            namespace: self.namespace.clone(),
            query: query.query.clone(),
            keys: query.keys.clone(),
            context: query.context.clone(),
        };
        let response = py.detach(|| self.round_trip(&request))?;
        match response {
            Response::Deleted(deleted) => Ok(deleted),
            Response::Error(error) => Err(error.into()),
            other => Err(Error::Daemon(format!("unexpected response to Del: {other:?}")).into()),
        }
    }
}

#[pyfunction]
pub(crate) fn _run_daemon() -> PyResult<()> {
    crate::run_daemon()?;
    Ok(())
}
