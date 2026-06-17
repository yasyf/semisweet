use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::sync::Arc;

use crate::error::Result;
use crate::newtype::{Context, Embedding, Entity, EntryId, Key, Namespace, QueryText};

#[derive(Debug, Clone, Default)]
pub struct Filter {
    pub keys_all: BTreeSet<Key>,
    pub entities_any: BTreeSet<Entity>,
}

impl Filter {
    pub fn new(keys_all: BTreeSet<Key>, entities_any: BTreeSet<Entity>) -> Self {
        Self {
            keys_all,
            entities_any,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VectorEntry {
    pub id: EntryId,
    pub vector: Embedding,
    pub query_text: QueryText,
    pub keys: BTreeSet<Key>,
    pub entities: BTreeSet<Entity>,
    pub context: Option<Context>,
    pub context_vector: Option<Embedding>,
}

#[derive(Debug, Clone)]
pub struct ScoredHit {
    pub id: EntryId,
    pub dense_score: f32,
    pub sparse_score: f32,
    pub entities: BTreeSet<Entity>,
    pub context: Option<Context>,
    pub context_vector: Option<Embedding>,
}

pub trait VectorStorageBackend: Send + Sync {
    fn upsert(&self, ns: &Namespace, entry: VectorEntry) -> Result<()>;
    fn query(
        &self,
        ns: &Namespace,
        vector: &Embedding,
        query_text: &QueryText,
        filter: &Filter,
        top_k: NonZeroUsize,
    ) -> Result<Vec<ScoredHit>>;
    fn delete(&self, ns: &Namespace, id: &EntryId) -> Result<()>;
}

impl VectorStorageBackend for Arc<dyn VectorStorageBackend> {
    fn upsert(&self, ns: &Namespace, entry: VectorEntry) -> Result<()> {
        (**self).upsert(ns, entry)
    }

    fn query(
        &self,
        ns: &Namespace,
        vector: &Embedding,
        query_text: &QueryText,
        filter: &Filter,
        top_k: NonZeroUsize,
    ) -> Result<Vec<ScoredHit>> {
        (**self).query(ns, vector, query_text, filter, top_k)
    }

    fn delete(&self, ns: &Namespace, id: &EntryId) -> Result<()> {
        (**self).delete(ns, id)
    }
}
