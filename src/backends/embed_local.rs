//! Local ONNX `EmbeddingBackend` running BGE-small-en-v1.5 (384-dim) on CPU via
//! [`fastembed`]. The model downloads once on first construction, then runs offline.

use std::path::PathBuf;
use std::sync::Mutex;

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

use crate::embedding::EmbeddingBackend;
use crate::error::{Error, Result};
use crate::newtype::{Dim, Embedding};

const MODEL_DIM: usize = 384;
const QUERY_INSTRUCTION: &str = "Represent this sentence for searching relevant passages: ";
// fastembed's default model cache is a path relative to the process working
// directory. The daemon runs with `cwd = /`, so an absolute override is required for
// it to find a model downloaded elsewhere; this env var supplies one.
const MODEL_CACHE_ENV: &str = "SEMISWEET_MODEL_CACHE";

pub struct LocalEmbedding {
    model: Mutex<TextEmbedding>,
    dim: Dim,
}

impl LocalEmbedding {
    pub fn new() -> Result<Self> {
        let mut options = TextInitOptions::new(EmbeddingModel::BGESmallENV15);
        if let Some(cache_dir) = std::env::var_os(MODEL_CACHE_ENV) {
            options = options.with_cache_dir(PathBuf::from(cache_dir));
        }
        let model = TextEmbedding::try_new(options).map_err(|e| Error::Embedding(e.into()))?;
        Ok(Self {
            model: Mutex::new(model),
            dim: Dim::new(MODEL_DIM)?,
        })
    }

    fn embed_one(&self, text: &str) -> Result<Embedding> {
        let mut model = self
            .model
            .lock()
            .map_err(|_| Error::Embedding("embedding model mutex poisoned".into()))?;
        let values = model
            .embed([text], None)
            .map_err(|e| Error::Embedding(e.into()))?
            .pop()
            .ok_or_else(|| Error::Embedding("fastembed returned no embedding".into()))?;
        Embedding::new(values)
    }
}

impl EmbeddingBackend for LocalEmbedding {
    fn dim(&self) -> Dim {
        self.dim
    }

    fn embed_query(&self, text: &str) -> Result<Embedding> {
        self.embed_one(&format!("{QUERY_INSTRUCTION}{text}"))
    }

    fn embed_document(&self, text: &str) -> Result<Embedding> {
        self.embed_one(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "downloads BGE model"]
    fn embeds_unit_norm_and_deterministic() {
        let backend = LocalEmbedding::new().expect("model init");
        assert_eq!(backend.dim().get(), MODEL_DIM);

        let first = backend
            .embed_document("hello world")
            .expect("embed document");
        assert_eq!(first.dim().get(), MODEL_DIM);

        let norm = first.values().iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm was {norm}");

        let again = backend
            .embed_document("hello world")
            .expect("re-embed document");
        assert_eq!(first.values(), again.values());

        let query = backend.embed_query("hello world").expect("embed query");
        assert_eq!(query.dim().get(), MODEL_DIM);
        let query_norm = query.values().iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (query_norm - 1.0).abs() < 1e-5,
            "query norm was {query_norm}"
        );
    }
}
