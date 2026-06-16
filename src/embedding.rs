use std::sync::Arc;

use crate::error::Result;
use crate::newtype::{Dim, Embedding};

pub trait EmbeddingBackend: Send + Sync {
    fn dim(&self) -> Dim;
    fn embed_query(&self, text: &str) -> Result<Embedding>;
    fn embed_document(&self, text: &str) -> Result<Embedding>;
}

impl EmbeddingBackend for Arc<dyn EmbeddingBackend> {
    fn dim(&self) -> Dim {
        (**self).dim()
    }

    fn embed_query(&self, text: &str) -> Result<Embedding> {
        (**self).embed_query(text)
    }

    fn embed_document(&self, text: &str) -> Result<Embedding> {
        (**self).embed_document(text)
    }
}
