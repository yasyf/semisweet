use std::collections::BTreeSet;
use std::time::SystemTime;

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
        let filter = Filter {
            keys_all: keys.clone(),
            entities_any: if self.scoring.entity_filter {
                entities.clone()
            } else {
                BTreeSet::new()
            },
        };
        let hits = self
            .vector
            .query(&self.namespace, &embedding, &filter, self.scoring.top_k)?;
        Ok(self.scoring.select(&entities, context, hits))
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
        let entities = self.extract_entities(query, context, false)?;
        let embedding = self.embedding.embed_query(query.as_str())?;
        let id = EntryId::derive(query, keys);
        self.object.put(&self.namespace, &id, value)?;
        self.vector.upsert(
            &self.namespace,
            VectorEntry {
                id,
                vector: embedding,
                keys: keys.clone(),
                entities,
                context: context.clone(),
                date: SystemTime::now(),
            },
        )
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

        fn embed_document(&self, text: &str) -> Result<Embedding> {
            Embedding::new(self.vectors.get(text).expect("document registered").clone())
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
            filter: &Filter,
            top_k: std::num::NonZeroUsize,
        ) -> Result<Vec<ScoredHit>> {
            let store = self.inner.lock().unwrap();
            let Some(entries) = store.get(ns) else {
                return Ok(Vec::new());
            };
            let mut hits: Vec<ScoredHit> = entries
                .values()
                .filter(|entry| filter.keys_all.is_subset(&entry.keys))
                .filter(|entry| {
                    filter.entities_any.is_empty()
                        || !filter.entities_any.is_disjoint(&entry.entities)
                })
                .map(|entry| ScoredHit {
                    id: entry.id,
                    score: vector.dot(&entry.vector).unwrap(),
                    entities: entry.entities.clone(),
                    context: entry.context.clone(),
                })
                .collect();
            hits.sort_by(|a, b| b.score.total_cmp(&a.score));
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
