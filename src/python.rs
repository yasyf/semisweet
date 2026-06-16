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

use std::collections::HashSet;
use std::fmt;
use std::hash::{DefaultHasher, Hash, Hasher};
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

fn py_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}

fn render_opt_str(value: &Option<String>) -> String {
    match value {
        Some(value) => format!("'{value}'"),
        None => "None".to_owned(),
    }
}

fn render_opt<T: fmt::Display>(value: &Option<T>) -> String {
    match value {
        Some(value) => value.to_string(),
        None => "None".to_owned(),
    }
}

fn render_str_list(values: &[String]) -> String {
    let items = values
        .iter()
        .map(|value| format!("'{value}'"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{items}]")
}

// Keys are an unordered set: render them sorted so the repr (and the hash derived
// from it) is canonical regardless of the set's iteration order.
fn render_str_set(values: &[String]) -> String {
    if values.is_empty() {
        return "set()".to_owned();
    }
    let mut sorted: Vec<&String> = values.iter().collect();
    sorted.sort();
    let items = sorted
        .iter()
        .map(|value| format!("'{value}'"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{{items}}}")
}

fn hash_str(value: &str) -> isize {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish() as isize
}

fn embedding_repr(choice: &EmbeddingChoice) -> String {
    match choice {
        EmbeddingChoice::Local { model } => {
            format!("LocalEmbedding(model={})", render_opt_str(model))
        }
        EmbeddingChoice::Voyage { model, dim } => format!(
            "VoyageEmbedding(model={}, dim={})",
            render_opt_str(model),
            render_opt(dim)
        ),
    }
}

fn entity_repr(choice: &EntityChoice) -> String {
    match choice {
        EntityChoice::Keyword { language } => {
            format!("KeywordEntities(lang={})", render_opt_str(language))
        }
        EntityChoice::Gliner {
            labels,
            repo,
            model,
            tokenizer,
        } => format!(
            "GlinerEntities(labels={}, repo={}, model={}, tokenizer={})",
            render_str_list(labels),
            render_opt_str(repo),
            render_opt_str(model),
            render_opt_str(tokenizer)
        ),
    }
}

fn vector_repr(choice: &VectorChoice) -> String {
    match choice {
        VectorChoice::Memory => "MemoryVectors()".to_owned(),
        VectorChoice::Turbopuffer => "TurbopufferVectors()".to_owned(),
    }
}

fn object_repr(choice: &ObjectChoice) -> String {
    match choice {
        ObjectChoice::Disk { root } => format!("DiskStorage(root={})", render_opt_str(root)),
        ObjectChoice::S3 {
            bucket,
            region,
            endpoint,
            prefix,
        } => format!(
            "S3Storage(bucket={}, region={}, endpoint={}, prefix='{prefix}')",
            render_opt_str(bucket),
            render_opt_str(region),
            render_opt_str(endpoint)
        ),
    }
}

/// A frozen, hashable cache lookup: a natural-language `query` plus an optional set
/// of entity `keys` and optional `context`. Keys are treated as an unordered set, so
/// two queries with the same query text, key set, and context compare and hash equal.
///
/// Keyword-only args: `query` (str, required), `keys` (set[str]), `context` (str).
/// Raises `ConfigError` if the query, any key, or the context is empty or invalid.
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

    /// Render the configured query, key set, and context.
    fn __repr__(&self) -> String {
        format!(
            "CacheQuery(query='{}', keys={}, context={})",
            self.query,
            render_str_set(&self.keys),
            render_opt_str(&self.context)
        )
    }

    /// Value equality: equal query text, equal context, and the same set of keys.
    fn __eq__(&self, other: &Self) -> bool {
        self.query == other.query
            && self.context == other.context
            && self.keys.iter().collect::<HashSet<_>>()
                == other.keys.iter().collect::<HashSet<_>>()
    }

    /// Hash consistent with `__eq__` (keys hashed as a set).
    fn __hash__(&self) -> isize {
        hash_str(&self.__repr__())
    }
}

// --- Embedding backends ---

/// Embedding backend that runs a sentence-transformer model in-process.
///
/// Keyword-only arg: `model` (str Hugging Face repo id; defaults to the daemon's
/// built-in model). On a cold cache the first use triggers a BLOCKING model download
/// from Hugging Face; set the `SEMISWEET_MODEL_CACHE` env var to control where the
/// weights are cached.
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

    /// Render the configured model.
    fn __repr__(&self) -> String {
        embedding_repr(&self.choice)
    }

    /// Value equality: same configured model.
    fn __eq__(&self, other: &Self) -> bool {
        self.choice == other.choice
    }

    /// Hash consistent with `__eq__`.
    fn __hash__(&self) -> isize {
        hash_str(&self.__repr__())
    }
}

/// Embedding backend backed by the Voyage AI API.
///
/// Keyword-only args: `model` (str) and `dim` (int output dimension); both default to
/// the daemon's defaults. Requires `VOYAGE_API_KEY` in the daemon's environment — a
/// missing key surfaces as `ConfigError` when the namespace is registered.
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

    /// Render the configured model and dimension.
    fn __repr__(&self) -> String {
        embedding_repr(&self.choice)
    }

    /// Value equality: same configured model and dimension.
    fn __eq__(&self, other: &Self) -> bool {
        self.choice == other.choice
    }

    /// Hash consistent with `__eq__`.
    fn __hash__(&self) -> isize {
        hash_str(&self.__repr__())
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

/// Entity backend that extracts keyword entities with no model download.
///
/// Keyword-only arg: `lang` (str language code; defaults to the daemon's default).
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

    /// Render the configured language.
    fn __repr__(&self) -> String {
        entity_repr(&self.choice)
    }

    /// Value equality: same configured language.
    fn __eq__(&self, other: &Self) -> bool {
        self.choice == other.choice
    }

    /// Hash consistent with `__eq__`.
    fn __hash__(&self) -> isize {
        hash_str(&self.__repr__())
    }
}

/// Entity backend backed by a GLiNER ONNX model.
///
/// Keyword-only args: `labels` (list[str] of entity types to extract; defaults to
/// person/organization/location/date/product/event), `repo` (str Hugging Face repo
/// id), `model` (str explicit ONNX path) and `tokenizer` (str explicit tokenizer.json
/// path). Unless explicit `model`/`tokenizer` paths are given, the first use on a cold
/// cache triggers a BLOCKING model download from Hugging Face; set the
/// `SEMISWEET_MODEL_CACHE` env var to control where the weights are cached.
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

    /// Render the configured labels, repo, and model/tokenizer paths.
    fn __repr__(&self) -> String {
        entity_repr(&self.choice)
    }

    /// Value equality: same configured labels, repo, and model/tokenizer paths.
    fn __eq__(&self, other: &Self) -> bool {
        self.choice == other.choice
    }

    /// Hash consistent with `__eq__`.
    fn __hash__(&self) -> isize {
        hash_str(&self.__repr__())
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

/// In-process vector index. Vectors live only for the lifetime of the daemon and are
/// lost when it shuts down. Takes no arguments.
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

    /// Render the backend.
    fn __repr__(&self) -> String {
        vector_repr(&self.choice)
    }

    /// Value equality: all instances are equal.
    fn __eq__(&self, other: &Self) -> bool {
        self.choice == other.choice
    }

    /// Hash consistent with `__eq__`.
    fn __hash__(&self) -> isize {
        hash_str(&self.__repr__())
    }
}

/// Vector index backed by turbopuffer. Requires turbopuffer credentials in the
/// daemon's environment. Takes no arguments.
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

    /// Render the backend.
    fn __repr__(&self) -> String {
        vector_repr(&self.choice)
    }

    /// Value equality: all instances are equal.
    fn __eq__(&self, other: &Self) -> bool {
        self.choice == other.choice
    }

    /// Hash consistent with `__eq__`.
    fn __hash__(&self) -> isize {
        hash_str(&self.__repr__())
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

/// Object storage on the local filesystem.
///
/// Keyword-only arg: `root` (str directory; defaults to the daemon's data directory).
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

    /// Render the configured root directory.
    fn __repr__(&self) -> String {
        object_repr(&self.choice)
    }

    /// Value equality: same configured root directory.
    fn __eq__(&self, other: &Self) -> bool {
        self.choice == other.choice
    }

    /// Hash consistent with `__eq__`.
    fn __hash__(&self) -> isize {
        hash_str(&self.__repr__())
    }
}

/// Object storage on S3 or an S3-compatible endpoint.
///
/// Keyword-only args: `bucket`, `region`, `endpoint`, and `prefix` (all str). Unset
/// values, along with the AWS credentials, are resolved from the daemon's environment.
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

    /// Render the configured bucket, region, endpoint, and prefix.
    fn __repr__(&self) -> String {
        object_repr(&self.choice)
    }

    /// Value equality: same configured bucket, region, endpoint, and prefix.
    fn __eq__(&self, other: &Self) -> bool {
        self.choice == other.choice
    }

    /// Hash consistent with `__eq__`.
    fn __hash__(&self) -> isize {
        hash_str(&self.__repr__())
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

/// Frozen, hashable scoring configuration for a namespace.
///
/// Keyword-only args: `base` (float base similarity threshold), `floor` (float minimum
/// threshold, must be <= base), `entity_bonus_weight` (float >= 0), `top_k` (int > 0),
/// `entity_filter` (bool), and `context` ('ignore' or 'tiebreak'). Each defaults to the
/// daemon's default. Raises `ConfigError` if a threshold falls outside [0, 1], `floor`
/// exceeds `base`, the weight is negative or non-finite, `top_k` is zero, or `context`
/// is unknown.
///
/// Equality compares the float thresholds with `==`; `__hash__` deliberately excludes
/// them and hashes only `top_k`, `entity_filter`, and `context`, so equal objects
/// always hash equal despite float comparison subtleties.
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

    /// Render every configured scoring field.
    fn __repr__(&self) -> String {
        format!(
            "Scoring(base={}, floor={}, entity_bonus_weight={}, top_k={}, entity_filter={}, context='{}')",
            self.dto.base_threshold,
            self.dto.floor_threshold,
            self.dto.entity_bonus_weight,
            self.dto.top_k,
            py_bool(self.dto.entity_filter),
            self.dto.context
        )
    }

    /// Value equality across every field, including the float thresholds.
    fn __eq__(&self, other: &Self) -> bool {
        self.dto == other.dto
    }

    /// Hash over `top_k`, `entity_filter`, and `context` only; the float thresholds are
    /// excluded so the hash stays consistent with `__eq__`.
    fn __hash__(&self) -> isize {
        let mut hasher = DefaultHasher::new();
        self.dto.top_k.hash(&mut hasher);
        self.dto.entity_filter.hash(&mut hasher);
        self.dto.context.hash(&mut hasher);
        hasher.finish() as isize
    }
}

// --- The cache itself ---

/// A semantic cache scoped to one namespace, backed by a shared daemon process.
///
/// Construct it with keyword-only backend objects: `namespace` (str, required),
/// `embedding`, `entities`, `vectors`, `storage`, and `scoring`. Every backend is
/// optional and falls back to a fully-local default. Construction spawns or connects
/// the daemon and registers the namespace, which on a cold cache triggers a BLOCKING
/// model download from Hugging Face for the local embedding and GLiNER backends; set
/// the `SEMISWEET_MODEL_CACHE` env var to control where the weights are cached.
///
/// Usable as a context manager: `with SemanticCache(namespace='ns') as cache: ...`
/// closes the daemon connection on exit. Raises `ConfigError` for an invalid namespace
/// or backend config and `DaemonError` for daemon or IO failures.
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

    fn disconnect(&self) -> Result<()> {
        let mut stub = self
            .stub
            .lock()
            .map_err(|_| Error::Daemon("client connection mutex poisoned".to_owned()))?;
        stub.bye()
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

    /// Look up the cached value for `query`. Returns the stored `bytes` on a semantic
    /// hit, or `None` on a miss. Raises `BackendError`/`DaemonError` on failure.
    fn get(&self, py: Python<'_>, query: &PyCacheQuery) -> PyResult<Option<Vec<u8>>> {
        let request = Request::Get {
            namespace: self.namespace.clone(),
            query: query.query.clone(),
            keys: query.keys.clone(),
            context: query.context.clone(),
        };
        let response = py.detach(|| self.round_trip(&request))?;
        match response {
            Response::Value(value) => Ok(value.map(|buf| buf.into_vec())),
            Response::Error(error) => Err(error.into()),
            other => Err(Error::Daemon(format!("unexpected response to Get: {other:?}")).into()),
        }
    }

    /// Store `value` (`bytes`) under `query`. Returns `True` if the daemon accepted the
    /// write. Raises `BackendError`/`DaemonError` on failure.
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

    /// Evict the entry matching `query`. Returns `True` if an entry was removed. Raises
    /// `BackendError`/`DaemonError` on failure.
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

    /// Render the cache's namespace.
    fn __repr__(&self) -> String {
        format!("SemanticCache(namespace='{}')", self.namespace)
    }

    /// Send a graceful goodbye so the daemon sheds this connection. A later operation
    /// transparently reconnects. Raises `DaemonError` if the goodbye fails.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.disconnect())?;
        Ok(())
    }

    /// Return self for use as a context manager.
    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Close the daemon connection on exit. Returns `False` so a pending exception is
    /// not suppressed.
    fn __exit__(
        &self,
        py: Python<'_>,
        _exc_type: &Bound<'_, PyAny>,
        _exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        self.close(py)?;
        Ok(false)
    }
}

#[pyfunction]
pub(crate) fn _run_daemon() -> PyResult<()> {
    crate::run_daemon()?;
    Ok(())
}
