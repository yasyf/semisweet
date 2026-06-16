//! Local ONNX `EmbeddingBackend` running BGE-small-en-v1.5 (384-dim) on CPU via
//! [`fastembed`]. The model downloads once on first construction, then runs offline.

use std::path::PathBuf;
use std::sync::Mutex;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

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
    // The asymmetric query prefix is a BGE/E5-family convention; an explicitly chosen
    // model embeds query and document symmetrically rather than getting a foreign prefix.
    query_instruction: &'static str,
}

impl LocalEmbedding {
    pub fn new(model: Option<&str>) -> Result<Self> {
        let (model_name, dim, query_instruction) = match model {
            None => (EmbeddingModel::BGESmallENV15, MODEL_DIM, QUERY_INSTRUCTION),
            Some(code) => {
                let info = TextEmbedding::list_supported_models()
                    .into_iter()
                    .find(|info| info.model_code.eq_ignore_ascii_case(code))
                    .ok_or_else(|| {
                        Error::InvalidConfig(format!("unknown local embedding model `{code}`"))
                    })?;
                (info.model, info.dim, "")
            }
        };
        let mut options = InitOptions::new(model_name);
        if let Some(cache_dir) = std::env::var_os(MODEL_CACHE_ENV) {
            options = options.with_cache_dir(PathBuf::from(cache_dir));
        }
        let model = TextEmbedding::try_new(options).map_err(|e| Error::Embedding(e.into()))?;
        Ok(Self {
            model: Mutex::new(model),
            dim: Dim::new(dim)?,
            query_instruction,
        })
    }

    fn embed_one(&self, text: &str) -> Result<Embedding> {
        let model = self
            .model
            .lock()
            .map_err(|_| Error::Embedding("embedding model mutex poisoned".into()))?;
        let values = model
            .embed(vec![text], None)
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
        self.embed_one(&format!("{}{text}", self.query_instruction))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "downloads BGE model"]
    fn embeds_unit_norm_and_deterministic() {
        let backend = LocalEmbedding::new(None).expect("model init");
        assert_eq!(backend.dim().get(), MODEL_DIM);

        let first = backend.embed_query("hello world").expect("embed query");
        assert_eq!(first.dim().get(), MODEL_DIM);

        let norm = first.values().iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm was {norm}");

        let again = backend.embed_query("hello world").expect("re-embed query");
        assert_eq!(first.values(), again.values());
    }
}
