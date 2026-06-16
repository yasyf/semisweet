use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::sync::Arc;

use crate::error::Result;
use crate::newtype::{Context, Embedding, Entity, EntryId, Key, Namespace};

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
    pub keys: BTreeSet<Key>,
    pub entities: BTreeSet<Entity>,
    pub context: Option<Context>,
}

#[derive(Debug, Clone)]
pub struct ScoredHit {
    pub id: EntryId,
    pub score: f32,
    pub entities: BTreeSet<Entity>,
    pub context: Option<Context>,
}

pub trait VectorStorageBackend: Send + Sync {
    fn upsert(&self, ns: &Namespace, entry: VectorEntry) -> Result<()>;
    fn query(
        &self,
        ns: &Namespace,
        vector: &Embedding,
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
        filter: &Filter,
        top_k: NonZeroUsize,
    ) -> Result<Vec<ScoredHit>> {
        (**self).query(ns, vector, filter, top_k)
    }

    fn delete(&self, ns: &Namespace, id: &EntryId) -> Result<()> {
        (**self).delete(ns, id)
    }
}
