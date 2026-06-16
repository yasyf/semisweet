//! In-memory `VectorStorageBackend` (std `RwLock`) — brute-force cosine search.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::{PoisonError, RwLock};

use crate::error::{Error, Result};
use crate::newtype::{Dim, Embedding, EntryId, Namespace};
use crate::vector::{Filter, ScoredHit, VectorEntry, VectorStorageBackend};

struct Shard {
    dim: Dim,
    entries: HashMap<EntryId, VectorEntry>,
}

#[derive(Default)]
pub struct MemoryVectorStore {
    shards: RwLock<HashMap<Namespace, Shard>>,
}

impl MemoryVectorStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl VectorStorageBackend for MemoryVectorStore {
    fn upsert(&self, ns: &Namespace, entry: VectorEntry) -> Result<()> {
        let dim = entry.vector.dim();
        let mut shards = self.shards.write().unwrap_or_else(PoisonError::into_inner);
        match shards.get_mut(ns) {
            Some(shard) => {
                if dim != shard.dim {
                    return Err(Error::DimMismatch {
                        got: dim.get(),
                        want: shard.dim.get(),
                    });
                }
                shard.entries.insert(entry.id, entry);
            }
            None => {
                let mut entries = HashMap::new();
                entries.insert(entry.id, entry);
                shards.insert(ns.clone(), Shard { dim, entries });
            }
        }
        Ok(())
    }

    fn query(
        &self,
        ns: &Namespace,
        vector: &Embedding,
        filter: &Filter,
        top_k: NonZeroUsize,
    ) -> Result<Vec<ScoredHit>> {
        let shards = self.shards.read().unwrap_or_else(PoisonError::into_inner);
        let Some(shard) = shards.get(ns) else {
            return Ok(Vec::new());
        };
        if vector.dim() != shard.dim {
            return Err(Error::DimMismatch {
                got: vector.dim().get(),
                want: shard.dim.get(),
            });
        }
        let mut hits: Vec<ScoredHit> = Vec::new();
        for entry in shard.entries.values() {
            if !filter.keys_all.is_subset(&entry.keys) {
                continue;
            }
            if !filter.entities_any.is_empty() && filter.entities_any.is_disjoint(&entry.entities) {
                continue;
            }
            let score = vector.dot(&entry.vector)?;
            hits.push(ScoredHit {
                id: entry.id,
                score,
                entities: entry.entities.clone(),
                context: entry.context.clone(),
            });
        }
        hits.sort_by(|a, b| b.score.total_cmp(&a.score));
        hits.truncate(top_k.get());
        Ok(hits)
    }

    fn delete(&self, ns: &Namespace, id: &EntryId) -> Result<()> {
        let mut shards = self.shards.write().unwrap_or_else(PoisonError::into_inner);
        if let Some(shard) = shards.get_mut(ns) {
            shard.entries.remove(id);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashSet};

    use crate::newtype::{Context, Entity, Key, QueryText};

    use super::*;

    fn keyset(keys: &[&str]) -> BTreeSet<Key> {
        keys.iter()
            .map(|k| Key::new((*k).to_owned()).unwrap())
            .collect()
    }

    fn entityset(entities: &[&str]) -> BTreeSet<Entity> {
        entities
            .iter()
            .map(|e| Entity::new((*e).to_owned()).unwrap())
            .collect()
    }

    fn entry(
        label: &str,
        vector: Vec<f32>,
        keys: &[&str],
        entities: &[&str],
        context: Option<&str>,
    ) -> VectorEntry {
        let query = QueryText::new(label.to_owned()).unwrap();
        let keys = keyset(keys);
        let id = EntryId::derive(&query, &keys);
        VectorEntry {
            id,
            vector: Embedding::new(vector).unwrap(),
            keys,
            entities: entityset(entities),
            context: context.map(|c| Context::new(c.to_owned()).unwrap()),
        }
    }

    fn ns() -> Namespace {
        Namespace::new("prod".to_owned()).unwrap()
    }

    fn top_k(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).unwrap()
    }

    #[test]
    fn upsert_then_query_round_trip_orders_by_cosine() {
        let store = MemoryVectorStore::new();
        let near = entry("near", vec![1.0, 0.0], &[], &[], None);
        let mid = entry("mid", vec![0.8, 0.6], &[], &[], None);
        let far = entry("far", vec![0.0, 1.0], &[], &[], None);
        store.upsert(&ns(), near.clone()).unwrap();
        store.upsert(&ns(), mid.clone()).unwrap();
        store.upsert(&ns(), far.clone()).unwrap();

        let query = Embedding::new(vec![1.0, 0.0]).unwrap();
        let hits = store
            .query(&ns(), &query, &Filter::default(), top_k(10))
            .unwrap();

        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].id, near.id);
        assert_eq!(hits[1].id, mid.id);
        assert_eq!(hits[2].id, far.id);
        assert!((hits[0].score - 1.0).abs() < 1e-6);
        assert!((hits[1].score - 0.8).abs() < 1e-6);
        assert!(hits[2].score.abs() < 1e-6);
    }

    #[test]
    fn query_returns_payload_metadata() {
        let store = MemoryVectorStore::new();
        let e = entry("e", vec![1.0, 0.0], &[], &["aspirin"], Some("dosage info"));
        store.upsert(&ns(), e.clone()).unwrap();

        let query = Embedding::new(vec![1.0, 0.0]).unwrap();
        let hits = store
            .query(&ns(), &query, &Filter::default(), top_k(10))
            .unwrap();

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].entities, entityset(&["aspirin"]));
        assert_eq!(
            hits[0].context,
            Some(Context::new("dosage info".to_owned()).unwrap())
        );
    }

    #[test]
    fn keys_all_filter_requires_all_keys_present() {
        let store = MemoryVectorStore::new();
        let both = entry("both", vec![1.0, 0.0], &["a", "b"], &[], None);
        let only_a = entry("only_a", vec![1.0, 0.0], &["a"], &[], None);
        store.upsert(&ns(), both.clone()).unwrap();
        store.upsert(&ns(), only_a.clone()).unwrap();

        let query = Embedding::new(vec![1.0, 0.0]).unwrap();

        let filter_a = Filter::new(keyset(&["a"]), BTreeSet::new());
        let hits_a = store.query(&ns(), &query, &filter_a, top_k(10)).unwrap();
        let ids_a: HashSet<EntryId> = hits_a.iter().map(|h| h.id).collect();
        assert_eq!(ids_a, [both.id, only_a.id].into_iter().collect());

        let filter_ab = Filter::new(keyset(&["a", "b"]), BTreeSet::new());
        let hits_ab = store.query(&ns(), &query, &filter_ab, top_k(10)).unwrap();
        assert_eq!(hits_ab.len(), 1);
        assert_eq!(hits_ab[0].id, both.id);

        let filter_ac = Filter::new(keyset(&["a", "c"]), BTreeSet::new());
        let hits_ac = store.query(&ns(), &query, &filter_ac, top_k(10)).unwrap();
        assert!(hits_ac.is_empty());
    }

    #[test]
    fn entities_any_filter_matches_any_overlap() {
        let store = MemoryVectorStore::new();
        let xy = entry("xy", vec![1.0, 0.0], &[], &["x", "y"], None);
        let z = entry("z", vec![1.0, 0.0], &[], &["z"], None);
        store.upsert(&ns(), xy.clone()).unwrap();
        store.upsert(&ns(), z.clone()).unwrap();

        let query = Embedding::new(vec![1.0, 0.0]).unwrap();

        let filter_x = Filter::new(BTreeSet::new(), entityset(&["x"]));
        let hits_x = store.query(&ns(), &query, &filter_x, top_k(10)).unwrap();
        assert_eq!(hits_x.len(), 1);
        assert_eq!(hits_x[0].id, xy.id);

        let filter_w = Filter::new(BTreeSet::new(), entityset(&["w"]));
        let hits_w = store.query(&ns(), &query, &filter_w, top_k(10)).unwrap();
        assert!(hits_w.is_empty());
    }

    #[test]
    fn empty_entities_any_imposes_no_entity_constraint() {
        let store = MemoryVectorStore::new();
        let with = entry("with", vec![1.0, 0.0], &[], &["x"], None);
        let without = entry("without", vec![1.0, 0.0], &[], &[], None);
        store.upsert(&ns(), with.clone()).unwrap();
        store.upsert(&ns(), without.clone()).unwrap();

        let query = Embedding::new(vec![1.0, 0.0]).unwrap();
        let hits = store
            .query(&ns(), &query, &Filter::default(), top_k(10))
            .unwrap();
        let ids: HashSet<EntryId> = hits.iter().map(|h| h.id).collect();
        assert_eq!(ids, [with.id, without.id].into_iter().collect());
    }

    #[test]
    fn top_k_truncates_to_highest_scoring() {
        let store = MemoryVectorStore::new();
        let near = entry("near", vec![1.0, 0.0], &[], &[], None);
        let mid = entry("mid", vec![0.8, 0.6], &[], &[], None);
        let far = entry("far", vec![0.0, 1.0], &[], &[], None);
        store.upsert(&ns(), near.clone()).unwrap();
        store.upsert(&ns(), mid.clone()).unwrap();
        store.upsert(&ns(), far.clone()).unwrap();

        let query = Embedding::new(vec![1.0, 0.0]).unwrap();
        let hits = store
            .query(&ns(), &query, &Filter::default(), top_k(2))
            .unwrap();

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, near.id);
        assert_eq!(hits[1].id, mid.id);
    }

    #[test]
    fn upsert_rejects_dim_mismatch() {
        let store = MemoryVectorStore::new();
        store
            .upsert(&ns(), entry("a", vec![1.0, 0.0], &[], &[], None))
            .unwrap();
        let err = store
            .upsert(&ns(), entry("b", vec![1.0, 0.0, 0.0], &[], &[], None))
            .unwrap_err();
        assert!(matches!(err, Error::DimMismatch { got: 3, want: 2 }));
    }

    #[test]
    fn query_rejects_dim_mismatch() {
        let store = MemoryVectorStore::new();
        store
            .upsert(&ns(), entry("a", vec![1.0, 0.0], &[], &[], None))
            .unwrap();
        let query = Embedding::new(vec![1.0, 0.0, 0.0]).unwrap();
        let err = store
            .query(&ns(), &query, &Filter::default(), top_k(10))
            .unwrap_err();
        assert!(matches!(err, Error::DimMismatch { got: 3, want: 2 }));
    }

    #[test]
    fn delete_removes_entry_so_query_misses() {
        let store = MemoryVectorStore::new();
        let e = entry("e", vec![1.0, 0.0], &[], &[], None);
        store.upsert(&ns(), e.clone()).unwrap();

        let query = Embedding::new(vec![1.0, 0.0]).unwrap();
        assert_eq!(
            store
                .query(&ns(), &query, &Filter::default(), top_k(10))
                .unwrap()
                .len(),
            1
        );

        store.delete(&ns(), &e.id).unwrap();
        assert!(
            store
                .query(&ns(), &query, &Filter::default(), top_k(10))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn delete_is_ok_for_absent_id_and_unknown_namespace() {
        let store = MemoryVectorStore::new();
        let e = entry("e", vec![1.0, 0.0], &[], &[], None);
        store.delete(&ns(), &e.id).unwrap();
        store.upsert(&ns(), e.clone()).unwrap();
        let other = entry("other", vec![1.0, 0.0], &[], &[], None);
        store.delete(&ns(), &other.id).unwrap();
        assert_eq!(
            store
                .query(
                    &ns(),
                    &Embedding::new(vec![1.0, 0.0]).unwrap(),
                    &Filter::default(),
                    top_k(10)
                )
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn re_upsert_same_id_overwrites_vector_and_metadata() {
        let store = MemoryVectorStore::new();
        let first = entry("e", vec![0.0, 1.0], &["k"], &["old"], Some("old context"));
        let second = entry("e", vec![1.0, 0.0], &["k"], &["new"], Some("new context"));
        assert_eq!(first.id, second.id);
        store.upsert(&ns(), first).unwrap();
        store.upsert(&ns(), second.clone()).unwrap();

        let query = Embedding::new(vec![1.0, 0.0]).unwrap();
        let hits = store
            .query(&ns(), &query, &Filter::default(), top_k(10))
            .unwrap();

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, second.id);
        assert!((hits[0].score - 1.0).abs() < 1e-6);
        assert_eq!(hits[0].entities, entityset(&["new"]));
        assert_eq!(
            hits[0].context,
            Some(Context::new("new context".to_owned()).unwrap())
        );
    }

    #[test]
    fn query_of_never_written_namespace_is_empty() {
        let store = MemoryVectorStore::new();
        let query = Embedding::new(vec![1.0, 0.0]).unwrap();
        let hits = store
            .query(&ns(), &query, &Filter::default(), top_k(10))
            .unwrap();
        assert!(hits.is_empty());
    }
}
