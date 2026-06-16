//! The pyo3 binding layer. Every `#[pyfunction]`/`#[pymethods]` here is a thin
//! wrapper: it converts Python values in, calls plain Rust, and converts results
//! out. No business logic lives at this edge — validation and RPC live in the
//! newtype, registry, and client modules. Errors cross the boundary through the
//! single `From<Error>`/`From<ProtocolError>` conversions in `crate::error`.

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
    #[pyo3(signature = (query, keys, context = None))]
    fn new(query: String, keys: HashSet<String>, context: Option<String>) -> Result<Self> {
        let query = QueryText::new(query)?.as_str().to_owned();
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

#[pyclass]
pub(crate) struct SemanticCacheBuilder {
    namespace: String,
    embedding: Option<EmbeddingChoice>,
    entity: Option<EntityChoice>,
    vector: Option<VectorChoice>,
    object: Option<ObjectChoice>,
    scoring: ScoringDto,
}

impl SemanticCacheBuilder {
    fn new(namespace: String) -> Self {
        Self {
            namespace,
            embedding: None,
            entity: None,
            vector: None,
            object: None,
            scoring: ScoringDto::default(),
        }
    }

    fn namespace_config(&self) -> Result<NamespaceConfig> {
        Ok(NamespaceConfig {
            embedding: self
                .embedding
                .clone()
                .ok_or(Error::IncompleteBuilder("embedding"))?,
            entity: self
                .entity
                .clone()
                .ok_or(Error::IncompleteBuilder("entity"))?,
            vector: self
                .vector
                .clone()
                .ok_or(Error::IncompleteBuilder("vector"))?,
            object: self
                .object
                .clone()
                .ok_or(Error::IncompleteBuilder("object"))?,
            scoring: self.scoring.clone(),
        })
    }
}

#[pymethods]
impl SemanticCacheBuilder {
    fn embedding_voyage<'py>(
        mut slf: PyRefMut<'py, Self>,
        model: String,
        dim: usize,
    ) -> PyRefMut<'py, Self> {
        slf.embedding = Some(EmbeddingChoice::Voyage { model, dim });
        slf
    }

    fn embedding_local(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.embedding = Some(EmbeddingChoice::Local);
        slf
    }

    fn entities_keyword(
        mut slf: PyRefMut<'_, Self>,
        language: Option<String>,
    ) -> PyRefMut<'_, Self> {
        slf.entity = Some(EntityChoice::Keyword { language });
        slf
    }

    fn entities_gliner(mut slf: PyRefMut<'_, Self>, labels: Vec<String>) -> PyRefMut<'_, Self> {
        slf.entity = Some(EntityChoice::Gliner { labels });
        slf
    }

    fn vector_memory(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.vector = Some(VectorChoice::Memory);
        slf
    }

    fn vector_turbopuffer(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.vector = Some(VectorChoice::Turbopuffer);
        slf
    }

    fn object_disk(mut slf: PyRefMut<'_, Self>, root: Option<String>) -> PyRefMut<'_, Self> {
        slf.object = Some(ObjectChoice::Disk { root });
        slf
    }

    #[pyo3(signature = (bucket, region, endpoint, prefix))]
    fn object_s3<'py>(
        mut slf: PyRefMut<'py, Self>,
        bucket: String,
        region: String,
        endpoint: Option<String>,
        prefix: String,
    ) -> PyRefMut<'py, Self> {
        slf.object = Some(ObjectChoice::S3 {
            bucket,
            region,
            endpoint,
            prefix,
        });
        slf
    }

    fn threshold<'py>(
        mut slf: PyRefMut<'py, Self>,
        base: f32,
        floor: f32,
        entity_bonus_weight: f32,
        top_k: usize,
        entity_filter: bool,
        context: String,
    ) -> Result<PyRefMut<'py, Self>> {
        let candidate = ScoringDto {
            base_threshold: base,
            floor_threshold: floor,
            entity_bonus_weight,
            top_k,
            entity_filter,
            context,
        };
        candidate.to_config()?;
        slf.scoring = candidate;
        Ok(slf)
    }

    fn build(&self, py: Python<'_>) -> PyResult<PySemanticCache> {
        let config = self.namespace_config()?;
        let config_json =
            serde_json::to_string(&config).map_err(|e| Error::InvalidConfig(e.to_string()))?;
        let executable: PathBuf = py
            .import("sys")?
            .getattr("executable")?
            .extract::<String>()?
            .into();
        let namespace = self.namespace.clone();
        let (stub, response) = py.detach(move || register(executable, namespace, config_json))?;
        match response {
            Response::Registered { ready: true } => Ok(PySemanticCache {
                namespace: self.namespace.clone(),
                stub: Mutex::new(stub),
            }),
            Response::Registered { ready: false } => {
                Err(Error::Daemon("namespace registered but not ready".to_owned()).into())
            }
            Response::Error(error) => Err(error.into()),
            other => Err(Error::Daemon(format!(
                "unexpected response to RegisterNamespace: {other:?}"
            ))
            .into()),
        }
    }
}

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
    #[staticmethod]
    fn builder(namespace: String) -> Result<SemanticCacheBuilder> {
        let namespace = Namespace::new(namespace)?;
        Ok(SemanticCacheBuilder::new(namespace.as_str().to_owned()))
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
