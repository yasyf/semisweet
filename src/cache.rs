use std::collections::BTreeSet;

use crate::embedding::EmbeddingBackend;
use crate::entity::EntityBackend;
use crate::error::Result;
use crate::newtype::{Context, Entity, EntryId, Key, Namespace, QueryText};
use crate::object::ObjectStorageBackend;
use crate::scoring::ScoringConfig;
use crate::vector::{Filter, ScoredHit, VectorEntry, VectorStorageBackend};

pub struct Cache<E, Emb, V, O> {
    namespace: Namespace,
    entity: E,
    embedding: Emb,
    vector: V,
    object: O,
    scoring: ScoringConfig,
}

impl<E, Emb, V, O> Cache<E, Emb, V, O>
where
    E: EntityBackend,
    Emb: EmbeddingBackend,
    V: VectorStorageBackend,
    O: ObjectStorageBackend,
{
    pub fn new(
        namespace: Namespace,
        entity: E,
        embedding: Emb,
        vector: V,
        object: O,
        scoring: ScoringConfig,
    ) -> Self {
        Self {
            namespace,
            entity,
            embedding,
            vector,
            object,
            scoring,
        }
    }

    fn extract_entities(
        &self,
        query: &QueryText,
        context: &Option<Context>,
        fast: bool,
    ) -> Result<BTreeSet<Entity>> {
        let entities = self.entity.extract(query.as_str(), fast)?;
        if entities.is_empty()
            && let Some(context) = context
        {
            return self.entity.extract(context.as_str(), fast);
        }
        Ok(entities)
    }

    fn find_match(
        &self,
        query: &QueryText,
        keys: &BTreeSet<Key>,
        context: &Option<Context>,
    ) -> Result<Option<ScoredHit>> {
        let entities = self.extract_entities(query, context, true)?;
        let embedding = self.embedding.embed_query(query.as_str())?;
        let context_vector = match context {
            Some(context) if self.scoring.uses_context_dense() => {
                Some(self.embedding.embed_query(context.as_str())?)
            }
            _ => None,
        };
        let filter = Filter {
            keys_all: keys.clone(),
            entities_any: if self.scoring.entity_filter {
                entities.clone()
            } else {
                BTreeSet::new()
            },
        };
        let hits = self.vector.query(
            &self.namespace,
            &embedding,
            query,
            &filter,
            self.scoring.top_k,
        )?;
        self.scoring
            .select(&entities, context, &context_vector, hits)
    }

    pub fn get(
        &self,
        query: &QueryText,
        keys: &BTreeSet<Key>,
        context: &Option<Context>,
    ) -> Result<Option<Vec<u8>>> {
        match self.find_match(query, keys, context)? {
            Some(hit) => self.object.get(&self.namespace, &hit.id),
            None => Ok(None),
        }
    }

    pub fn set(
        &self,
        query: &QueryText,
        keys: &BTreeSet<Key>,
        context: &Option<Context>,
        value: &[u8],
    ) -> Result<()> {
        let entry = self.prepare(query, keys, context)?;
        self.commit(&entry, value)
    }

    /// The slow half of a write — entity extraction + embedding — yielding the entry to
    /// commit. Split from [`commit`](Self::commit) so the write-behind worker runs it
    /// off the pending lock and only afterwards decides whether the entry is still wanted.
    pub fn prepare(
        &self,
        query: &QueryText,
        keys: &BTreeSet<Key>,
        context: &Option<Context>,
    ) -> Result<VectorEntry> {
        let entities = self.extract_entities(query, context, false)?;
        let embedding = self.embedding.embed_query(query.as_str())?;
        let context_vector = match context {
            Some(context) => Some(self.embedding.embed_query(context.as_str())?),
            None => None,
        };
        Ok(VectorEntry {
            id: EntryId::derive(query, keys),
            vector: embedding,
            query_text: query.clone(),
            keys: keys.clone(),
            entities,
            context: context.clone(),
            context_vector,
        })
    }

    /// The fast half of a write — object put + vector upsert.
    pub fn commit(&self, entry: &VectorEntry, value: &[u8]) -> Result<()> {
        self.object.put(&self.namespace, &entry.id, value)?;
        self.vector.upsert(&self.namespace, entry.clone())
    }

    pub fn delete(
        &self,
        query: &QueryText,
        keys: &BTreeSet<Key>,
        context: &Option<Context>,
    ) -> Result<bool> {
        match self.find_match(query, keys, context)? {
            Some(hit) => {
                self.vector.delete(&self.namespace, &hit.id)?;
                self.object.delete(&self.namespace, &hit.id)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use crate::newtype::{Dim, Embedding};

    use super::*;

    type VectorStore = HashMap<Namespace, HashMap<EntryId, VectorEntry>>;
    type FakeCache = Cache<
        Arc<dyn EntityBackend>,
        Arc<dyn EmbeddingBackend>,
        Arc<dyn VectorStorageBackend>,
        Arc<dyn ObjectStorageBackend>,
    >;

    fn ents(names: &[&str]) -> BTreeSet<Entity> {
        names
            .iter()
            .map(|n| Entity::new((*n).to_owned()).unwrap())
            .collect()
    }

    fn keyset(names: &[&str]) -> BTreeSet<Key> {
        names
            .iter()
            .map(|n| Key::new((*n).to_owned()).unwrap())
            .collect()
    }

    fn query(text: &str) -> QueryText {
        QueryText::new(text.to_owned()).unwrap()
    }

    /// A stand-in for the backends' BM25: token Jaccard of the query against the stored query
    /// text, in `[0, 1]`. Identical text scores 1.0 so an exact-match GET fuses high enough to
    /// clear the recalibrated threshold.
    fn jaccard(query_tokens: &BTreeSet<&str>, doc: &str) -> f32 {
        let doc_tokens: BTreeSet<&str> = doc.split_whitespace().collect();
        if query_tokens.is_empty() || doc_tokens.is_empty() {
            return 0.0;
        }
        let intersection = query_tokens.intersection(&doc_tokens).count();
        let union = query_tokens.union(&doc_tokens).count();
        intersection as f32 / union as f32
    }

    struct FakeEmbedding {
        dim: Dim,
        vectors: HashMap<String, Vec<f32>>,
    }

    impl FakeEmbedding {
        fn new(dim: usize, vectors: &[(&str, Vec<f32>)]) -> Self {
            Self {
                dim: Dim::new(dim).unwrap(),
                vectors: vectors
                    .iter()
                    .map(|(k, v)| ((*k).to_owned(), v.clone()))
                    .collect(),
            }
        }
    }

    impl EmbeddingBackend for FakeEmbedding {
        fn dim(&self) -> Dim {
            self.dim
        }

        fn embed_query(&self, text: &str) -> Result<Embedding> {
            Embedding::new(self.vectors.get(text).expect("query registered").clone())
        }
    }

    struct FakeEntity {
        fast: BTreeSet<Entity>,
        full: BTreeSet<Entity>,
    }

    impl FakeEntity {
        fn uniform(names: &[&str]) -> Self {
            let set = ents(names);
            Self {
                fast: set.clone(),
                full: set,
            }
        }

        fn split(fast: &[&str], full: &[&str]) -> Self {
            Self {
                fast: ents(fast),
                full: ents(full),
            }
        }
    }

    impl EntityBackend for FakeEntity {
        fn extract(&self, _text: &str, fast: bool) -> Result<BTreeSet<Entity>> {
            Ok(if fast {
                self.fast.clone()
            } else {
                self.full.clone()
            })
        }
    }

    struct FakeVectorStore {
        inner: Mutex<VectorStore>,
    }

    impl FakeVectorStore {
        fn new() -> Self {
            Self {
                inner: Mutex::new(HashMap::new()),
            }
        }

        fn stored_entry(&self, ns: &Namespace, id: &EntryId) -> Option<VectorEntry> {
            self.inner
                .lock()
                .unwrap()
                .get(ns)
                .and_then(|entries| entries.get(id).cloned())
        }
    }

    impl VectorStorageBackend for FakeVectorStore {
        fn upsert(&self, ns: &Namespace, entry: VectorEntry) -> Result<()> {
            self.inner
                .lock()
                .unwrap()
                .entry(ns.clone())
                .or_default()
                .insert(entry.id, entry);
            Ok(())
        }

        fn query(
            &self,
            ns: &Namespace,
            vector: &Embedding,
            query_text: &QueryText,
            filter: &Filter,
            top_k: std::num::NonZeroUsize,
        ) -> Result<Vec<ScoredHit>> {
            let store = self.inner.lock().unwrap();
            let Some(entries) = store.get(ns) else {
                return Ok(Vec::new());
            };
            let query_tokens: BTreeSet<&str> = query_text.as_str().split_whitespace().collect();
            let mut hits: Vec<ScoredHit> = entries
                .values()
                .filter(|entry| filter.keys_all.is_subset(&entry.keys))
                .filter(|entry| {
                    filter.entities_any.is_empty()
                        || !filter.entities_any.is_disjoint(&entry.entities)
                })
                .map(|entry| ScoredHit {
                    id: entry.id,
                    dense_score: vector.dot(&entry.vector).unwrap(),
                    sparse_score: jaccard(&query_tokens, entry.query_text.as_str()),
                    entities: entry.entities.clone(),
                    context: entry.context.clone(),
                    context_vector: entry.context_vector.clone(),
                })
                .collect();
            hits.sort_by(|a, b| {
                (b.dense_score + b.sparse_score).total_cmp(&(a.dense_score + a.sparse_score))
            });
            hits.truncate(top_k.get());
            Ok(hits)
        }

        fn delete(&self, ns: &Namespace, id: &EntryId) -> Result<()> {
            if let Some(entries) = self.inner.lock().unwrap().get_mut(ns) {
                entries.remove(id);
            }
            Ok(())
        }
    }

    struct FakeObjectStore {
        inner: Mutex<HashMap<(Namespace, EntryId), Vec<u8>>>,
    }

    impl FakeObjectStore {
        fn new() -> Self {
            Self {
                inner: Mutex::new(HashMap::new()),
            }
        }

        fn remove_object(&self, ns: &Namespace, id: &EntryId) {
            self.inner.lock().unwrap().remove(&(ns.clone(), *id));
        }
    }

    impl ObjectStorageBackend for FakeObjectStore {
        fn put(&self, ns: &Namespace, id: &EntryId, value: &[u8]) -> Result<()> {
            self.inner
                .lock()
                .unwrap()
                .insert((ns.clone(), *id), value.to_vec());
            Ok(())
        }

        fn get(&self, ns: &Namespace, id: &EntryId) -> Result<Option<Vec<u8>>> {
            Ok(self.inner.lock().unwrap().get(&(ns.clone(), *id)).cloned())
        }

        fn delete(&self, ns: &Namespace, id: &EntryId) -> Result<()> {
            self.inner.lock().unwrap().remove(&(ns.clone(), *id));
            Ok(())
        }
    }

    fn namespace() -> Namespace {
        Namespace::new("test".to_owned()).unwrap()
    }

    fn build_cache(
        entity: FakeEntity,
        embedding: FakeEmbedding,
        scoring: ScoringConfig,
    ) -> (FakeCache, Arc<FakeVectorStore>, Arc<FakeObjectStore>) {
        let vector = Arc::new(FakeVectorStore::new());
        let object = Arc::new(FakeObjectStore::new());
        let entity_dyn: Arc<dyn EntityBackend> = Arc::new(entity);
        let embedding_dyn: Arc<dyn EmbeddingBackend> = Arc::new(embedding);
        let vector_dyn: Arc<dyn VectorStorageBackend> = vector.clone();
        let object_dyn: Arc<dyn ObjectStorageBackend> = object.clone();
        let cache = Cache::new(
            namespace(),
            entity_dyn,
            embedding_dyn,
            vector_dyn,
            object_dyn,
            scoring,
        );
        (cache, vector, object)
    }

    #[test]
    fn set_then_get_round_trips_exact_bytes() {
        let (cache, _vector, _object) = build_cache(
            FakeEntity::uniform(&["drug"]),
            FakeEmbedding::new(4, &[("what is aspirin", vec![1.0, 0.0, 0.0, 0.0])]),
            ScoringConfig::default(),
        );
        let q = query("what is aspirin");
        let keys = keyset(&["patient1"]);

        cache.set(&q, &keys, &None, b"the answer").unwrap();
        let got = cache.get(&q, &keys, &None).unwrap();

        assert_eq!(got, Some(b"the answer".to_vec()));
    }

    #[test]
    fn get_misses_for_unrelated_query() {
        let (cache, _vector, _object) = build_cache(
            FakeEntity::uniform(&["drug"]),
            FakeEmbedding::new(
                4,
                &[
                    ("what is aspirin", vec![1.0, 0.0, 0.0, 0.0]),
                    ("weather forecast", vec![0.0, 1.0, 0.0, 0.0]),
                ],
            ),
            ScoringConfig::default(),
        );
        let keys = keyset(&["patient1"]);

        cache
            .set(&query("what is aspirin"), &keys, &None, b"answer")
            .unwrap();
        let got = cache.get(&query("weather forecast"), &keys, &None).unwrap();

        assert_eq!(got, None);
    }

    #[test]
    fn set_writes_object_then_delete_clears_both_and_is_idempotent() {
        let (cache, vector, object) = build_cache(
            FakeEntity::uniform(&["drug"]),
            FakeEmbedding::new(4, &[("what is aspirin", vec![1.0, 0.0, 0.0, 0.0])]),
            ScoringConfig::default(),
        );
        let q = query("what is aspirin");
        let keys = keyset(&["patient1"]);
        let id = EntryId::derive(&q, &keys);

        cache.set(&q, &keys, &None, b"answer").unwrap();
        assert_eq!(
            object.get(&namespace(), &id).unwrap(),
            Some(b"answer".to_vec())
        );
        assert!(vector.stored_entry(&namespace(), &id).is_some());

        assert!(cache.delete(&q, &keys, &None).unwrap());
        assert!(vector.stored_entry(&namespace(), &id).is_none());
        assert_eq!(object.get(&namespace(), &id).unwrap(), None);

        assert!(!cache.delete(&q, &keys, &None).unwrap());
    }

    #[test]
    fn get_misses_when_object_is_absent_but_vector_remains() {
        let (cache, _vector, object) = build_cache(
            FakeEntity::uniform(&["drug"]),
            FakeEmbedding::new(4, &[("what is aspirin", vec![1.0, 0.0, 0.0, 0.0])]),
            ScoringConfig::default(),
        );
        let q = query("what is aspirin");
        let keys = keyset(&["patient1"]);
        let id = EntryId::derive(&q, &keys);

        cache.set(&q, &keys, &None, b"answer").unwrap();
        object.remove_object(&namespace(), &id);

        assert_eq!(cache.get(&q, &keys, &None).unwrap(), None);
    }

    #[test]
    fn get_uses_fast_entities_set_uses_full_and_reget_still_hits() {
        let (cache, vector, _object) = build_cache(
            FakeEntity::split(&["drug"], &["drug", "dose", "route"]),
            FakeEmbedding::new(4, &[("interaction query", vec![1.0, 0.0, 0.0, 0.0])]),
            ScoringConfig::default(),
        );
        let q = query("interaction query");
        let keys = keyset(&["patient1"]);
        let id = EntryId::derive(&q, &keys);

        cache.set(&q, &keys, &None, b"payload").unwrap();

        let stored = vector.stored_entry(&namespace(), &id).unwrap();
        assert_eq!(stored.entities, ents(&["drug", "dose", "route"]));

        let got = cache.get(&q, &keys, &None).unwrap();
        assert_eq!(got, Some(b"payload".to_vec()));
    }
}
