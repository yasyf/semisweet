//! In-memory `VectorStorageBackend` (std `RwLock`) — brute-force cosine retrieval paired with a
//! per-namespace BM25 lexical index over stored contexts. Retrieval is dense-only; the context
//! BM25 scores the query's context against each candidate's stored context for the context gate.

use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::sync::{PoisonError, RwLock};

use crate::error::{Error, Result};
use crate::newtype::{Context, Dim, Embedding, EntryId, Namespace};
use crate::vector::{Filter, ScoredHit, VectorEntry, VectorStorageBackend};

const DEFAULT_BM25_K1: f32 = 1.2;
const DEFAULT_BM25_B: f32 = 0.75;

fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|token| token.to_lowercase())
        .collect()
}

/// Index `entry`'s context into `bm25`, keyed by its id; an entry with no context is left out
/// of the index (it scores 0). `Bm25Index::insert` removes any prior posting for the id first,
/// so re-upserting the same id stays idempotent.
fn index_context(bm25: &mut Bm25Index, entry: &VectorEntry) {
    if let Some(context) = &entry.context {
        bm25.insert(entry.id, context.as_str());
    }
}

/// A per-namespace BM25 index over stored contexts. The backend already scans every entry for
/// cosine, so no postings lists are needed — only the document-frequency, per-document
/// term-frequency, and length statistics BM25 needs to score a candidate it is already visiting.
struct Bm25Index {
    document_frequency: HashMap<String, usize>,
    term_frequency: HashMap<EntryId, HashMap<String, u32>>,
    document_length: HashMap<EntryId, u32>,
    total_length: u64,
    document_count: usize,
}

impl Bm25Index {
    fn new() -> Self {
        Self {
            document_frequency: HashMap::new(),
            term_frequency: HashMap::new(),
            document_length: HashMap::new(),
            total_length: 0,
            document_count: 0,
        }
    }

    fn insert(&mut self, id: EntryId, text: &str) {
        self.remove(&id);
        let mut term_frequency: HashMap<String, u32> = HashMap::new();
        let mut length: u32 = 0;
        for token in tokenize(text) {
            *term_frequency.entry(token).or_insert(0) += 1;
            length += 1;
        }
        for term in term_frequency.keys() {
            *self.document_frequency.entry(term.clone()).or_insert(0) += 1;
        }
        self.total_length += u64::from(length);
        self.document_length.insert(id, length);
        self.term_frequency.insert(id, term_frequency);
        self.document_count += 1;
    }

    fn remove(&mut self, id: &EntryId) {
        let Some(term_frequency) = self.term_frequency.remove(id) else {
            return;
        };
        for term in term_frequency.keys() {
            if let Some(count) = self.document_frequency.get_mut(term) {
                *count -= 1;
                if *count == 0 {
                    self.document_frequency.remove(term);
                }
            }
        }
        if let Some(length) = self.document_length.remove(id) {
            self.total_length -= u64::from(length);
        }
        self.document_count -= 1;
    }

    /// BM25 of `query_terms` against document `id`, normalized to `[0, 1]` by the query's own
    /// upper bound — the score a document saturated in every query term would earn. The bound puts
    /// the sparse score on the same `[0, 1]` scale as the dense cosine without any per-result-batch
    /// normalization (unlike min-max), so a hit's score never shifts with which other hits are
    /// returned. It still moves with the corpus itself (avgdl and IDF), as BM25 inherently does.
    fn score(&self, id: &EntryId, query_terms: &HashSet<String>) -> f32 {
        if self.document_count == 0 || query_terms.is_empty() {
            return 0.0;
        }
        let Some(term_frequency) = self.term_frequency.get(id) else {
            return 0.0;
        };
        let length = self.document_length.get(id).copied().unwrap_or(0) as f32;
        let average_length = self.total_length as f32 / self.document_count as f32;
        let mut numerator = 0.0f32;
        let mut upper_bound = 0.0f32;
        for term in query_terms {
            let idf = self.idf(term);
            upper_bound += idf * (DEFAULT_BM25_K1 + 1.0);
            let tf = term_frequency.get(term).copied().unwrap_or(0) as f32;
            if tf > 0.0 {
                let denominator = tf
                    + DEFAULT_BM25_K1
                        * (1.0 - DEFAULT_BM25_B + DEFAULT_BM25_B * length / average_length);
                numerator += idf * (tf * (DEFAULT_BM25_K1 + 1.0)) / denominator;
            }
        }
        if upper_bound <= 0.0 {
            return 0.0;
        }
        (numerator / upper_bound).clamp(0.0, 1.0)
    }

    /// Lucene-style IDF, always `>= 0`, sidestepping BM25's negative-IDF wart.
    fn idf(&self, term: &str) -> f32 {
        let n = self.document_count as f32;
        let df = self.document_frequency.get(term).copied().unwrap_or(0) as f32;
        (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
    }
}

struct Shard {
    dim: Dim,
    entries: HashMap<EntryId, VectorEntry>,
    bm25: Bm25Index,
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
                index_context(&mut shard.bm25, &entry);
                shard.entries.insert(entry.id, entry);
            }
            None => {
                let mut shard = Shard {
                    dim,
                    entries: HashMap::new(),
                    bm25: Bm25Index::new(),
                };
                index_context(&mut shard.bm25, &entry);
                shard.entries.insert(entry.id, entry);
                shards.insert(ns.clone(), shard);
            }
        }
        Ok(())
    }

    fn query(
        &self,
        ns: &Namespace,
        vector: &Embedding,
        query_context: Option<&Context>,
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
        // Retrieval is dense-only; the context BM25 only scores the query's context against each
        // candidate's stored context, becoming the `sparse_score` gate signal. With no query
        // context every candidate scores 0.
        let context_terms: Option<HashSet<String>> =
            query_context.map(|ctx| tokenize(ctx.as_str()).into_iter().collect());
        let mut hits: Vec<ScoredHit> = Vec::new();
        for entry in shard.entries.values() {
            if !filter.keys_all.is_subset(&entry.keys) {
                continue;
            }
            if !filter.entities_any.is_empty() && filter.entities_any.is_disjoint(&entry.entities) {
                continue;
            }
            let dense_score = vector.dot(&entry.vector)?;
            let sparse_score = match &context_terms {
                Some(terms) => shard.bm25.score(&entry.id, terms),
                None => 0.0,
            };
            hits.push(ScoredHit {
                id: entry.id,
                dense_score,
                sparse_score,
                entities: entry.entities.clone(),
                context: entry.context.clone(),
            });
        }
        hits.sort_by(|a, b| b.dense_score.total_cmp(&a.dense_score));
        hits.truncate(top_k.get());
        Ok(hits)
    }

    fn delete(&self, ns: &Namespace, id: &EntryId) -> Result<()> {
        let mut shards = self.shards.write().unwrap_or_else(PoisonError::into_inner);
        if let Some(shard) = shards.get_mut(ns) {
            shard.entries.remove(id);
            shard.bm25.remove(id);
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
        let context = context.map(|c| Context::new(c.to_owned()).unwrap());
        let id = EntryId::derive(&query, &keys, &context);
        VectorEntry {
            id,
            vector: Embedding::new(vector).unwrap(),
            keys,
            entities: entityset(entities),
            context,
        }
    }

    fn ns() -> Namespace {
        Namespace::new("prod".to_owned()).unwrap()
    }

    fn ctx(text: &str) -> Context {
        Context::new(text.to_owned()).unwrap()
    }

    fn top_k(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).unwrap()
    }

    fn store() -> MemoryVectorStore {
        MemoryVectorStore::new()
    }

    #[test]
    fn upsert_then_query_round_trip_orders_by_cosine() {
        let store = store();
        let near = entry("near", vec![1.0, 0.0], &[], &[], None);
        let mid = entry("mid", vec![0.8, 0.6], &[], &[], None);
        let far = entry("far", vec![0.0, 1.0], &[], &[], None);
        store.upsert(&ns(), near.clone()).unwrap();
        store.upsert(&ns(), mid.clone()).unwrap();
        store.upsert(&ns(), far.clone()).unwrap();

        let query = Embedding::new(vec![1.0, 0.0]).unwrap();
        let hits = store
            .query(&ns(), &query, None, &Filter::default(), top_k(10))
            .unwrap();

        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].id, near.id);
        assert_eq!(hits[1].id, mid.id);
        assert_eq!(hits[2].id, far.id);
        assert!((hits[0].dense_score - 1.0).abs() < 1e-6);
        assert!((hits[1].dense_score - 0.8).abs() < 1e-6);
        assert!(hits[2].dense_score.abs() < 1e-6);
    }

    #[test]
    fn query_returns_payload_metadata() {
        let store = store();
        let e = entry("e", vec![1.0, 0.0], &[], &["aspirin"], Some("dosage info"));
        store.upsert(&ns(), e.clone()).unwrap();

        let query = Embedding::new(vec![1.0, 0.0]).unwrap();
        let hits = store
            .query(&ns(), &query, None, &Filter::default(), top_k(10))
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
        let store = store();
        let both = entry("both", vec![1.0, 0.0], &["a", "b"], &[], None);
        let only_a = entry("only_a", vec![1.0, 0.0], &["a"], &[], None);
        store.upsert(&ns(), both.clone()).unwrap();
        store.upsert(&ns(), only_a.clone()).unwrap();

        let query = Embedding::new(vec![1.0, 0.0]).unwrap();

        let filter_a = Filter::new(keyset(&["a"]), BTreeSet::new());
        let hits_a = store
            .query(&ns(), &query, None, &filter_a, top_k(10))
            .unwrap();
        let ids_a: HashSet<EntryId> = hits_a.iter().map(|h| h.id).collect();
        assert_eq!(ids_a, [both.id, only_a.id].into_iter().collect());

        let filter_ab = Filter::new(keyset(&["a", "b"]), BTreeSet::new());
        let hits_ab = store
            .query(&ns(), &query, None, &filter_ab, top_k(10))
            .unwrap();
        assert_eq!(hits_ab.len(), 1);
        assert_eq!(hits_ab[0].id, both.id);

        let filter_ac = Filter::new(keyset(&["a", "c"]), BTreeSet::new());
        let hits_ac = store
            .query(&ns(), &query, None, &filter_ac, top_k(10))
            .unwrap();
        assert!(hits_ac.is_empty());
    }

    #[test]
    fn entities_any_filter_matches_any_overlap() {
        let store = store();
        let xy = entry("xy", vec![1.0, 0.0], &[], &["x", "y"], None);
        let z = entry("z", vec![1.0, 0.0], &[], &["z"], None);
        store.upsert(&ns(), xy.clone()).unwrap();
        store.upsert(&ns(), z.clone()).unwrap();

        let query = Embedding::new(vec![1.0, 0.0]).unwrap();

        let filter_x = Filter::new(BTreeSet::new(), entityset(&["x"]));
        let hits_x = store
            .query(&ns(), &query, None, &filter_x, top_k(10))
            .unwrap();
        assert_eq!(hits_x.len(), 1);
        assert_eq!(hits_x[0].id, xy.id);

        let filter_w = Filter::new(BTreeSet::new(), entityset(&["w"]));
        let hits_w = store
            .query(&ns(), &query, None, &filter_w, top_k(10))
            .unwrap();
        assert!(hits_w.is_empty());
    }

    #[test]
    fn empty_entities_any_imposes_no_entity_constraint() {
        let store = store();
        let with = entry("with", vec![1.0, 0.0], &[], &["x"], None);
        let without = entry("without", vec![1.0, 0.0], &[], &[], None);
        store.upsert(&ns(), with.clone()).unwrap();
        store.upsert(&ns(), without.clone()).unwrap();

        let query = Embedding::new(vec![1.0, 0.0]).unwrap();
        let hits = store
            .query(&ns(), &query, None, &Filter::default(), top_k(10))
            .unwrap();
        let ids: HashSet<EntryId> = hits.iter().map(|h| h.id).collect();
        assert_eq!(ids, [with.id, without.id].into_iter().collect());
    }

    #[test]
    fn top_k_truncates_to_highest_scoring() {
        let store = store();
        let near = entry("near", vec![1.0, 0.0], &[], &[], None);
        let mid = entry("mid", vec![0.8, 0.6], &[], &[], None);
        let far = entry("far", vec![0.0, 1.0], &[], &[], None);
        store.upsert(&ns(), near.clone()).unwrap();
        store.upsert(&ns(), mid.clone()).unwrap();
        store.upsert(&ns(), far.clone()).unwrap();

        let query = Embedding::new(vec![1.0, 0.0]).unwrap();
        let hits = store
            .query(&ns(), &query, None, &Filter::default(), top_k(2))
            .unwrap();

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, near.id);
        assert_eq!(hits[1].id, mid.id);
    }

    #[test]
    fn upsert_rejects_dim_mismatch() {
        let store = store();
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
        let store = store();
        store
            .upsert(&ns(), entry("a", vec![1.0, 0.0], &[], &[], None))
            .unwrap();
        let query = Embedding::new(vec![1.0, 0.0, 0.0]).unwrap();
        let err = store
            .query(&ns(), &query, None, &Filter::default(), top_k(10))
            .unwrap_err();
        assert!(matches!(err, Error::DimMismatch { got: 3, want: 2 }));
    }

    #[test]
    fn delete_removes_entry_so_query_misses() {
        let store = store();
        let e = entry("e", vec![1.0, 0.0], &[], &[], None);
        store.upsert(&ns(), e.clone()).unwrap();

        let query = Embedding::new(vec![1.0, 0.0]).unwrap();
        assert_eq!(
            store
                .query(&ns(), &query, None, &Filter::default(), top_k(10))
                .unwrap()
                .len(),
            1
        );

        store.delete(&ns(), &e.id).unwrap();
        assert!(
            store
                .query(&ns(), &query, None, &Filter::default(), top_k(10))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn delete_is_ok_for_absent_id_and_unknown_namespace() {
        let store = store();
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
                    None,
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
        let store = store();
        // Context is part of the identity, so an overwrite shares it; only the vector and
        // the non-identity metadata (entities) change between the two upserts.
        let first = entry("e", vec![0.0, 1.0], &["k"], &["old"], Some("context"));
        let second = entry("e", vec![1.0, 0.0], &["k"], &["new"], Some("context"));
        assert_eq!(first.id, second.id);
        store.upsert(&ns(), first).unwrap();
        store.upsert(&ns(), second.clone()).unwrap();

        let query = Embedding::new(vec![1.0, 0.0]).unwrap();
        let hits = store
            .query(&ns(), &query, None, &Filter::default(), top_k(10))
            .unwrap();

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, second.id);
        assert!((hits[0].dense_score - 1.0).abs() < 1e-6);
        assert_eq!(hits[0].entities, entityset(&["new"]));
        assert_eq!(
            hits[0].context,
            Some(Context::new("context".to_owned()).unwrap())
        );
    }

    #[test]
    fn query_of_never_written_namespace_is_empty() {
        let store = store();
        let query = Embedding::new(vec![1.0, 0.0]).unwrap();
        let hits = store
            .query(&ns(), &query, None, &Filter::default(), top_k(10))
            .unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn context_bm25_rarer_term_outranks_common_term() {
        // Same vector for every doc, so dense ties and only the context BM25 (IDF) signal
        // separates them: the doc whose context shares the rare term `apple` must out-score the
        // one whose context shares the common `fruit`.
        let store = store();
        let apple = entry("apple", vec![1.0, 0.0], &[], &[], Some("apple"));
        let fruit = entry("fruit", vec![1.0, 0.0], &[], &[], Some("fruit"));
        store.upsert(&ns(), apple.clone()).unwrap();
        store.upsert(&ns(), fruit.clone()).unwrap();
        for filler in ["fruit red", "fruit green", "fruit yellow"] {
            store
                .upsert(&ns(), entry(filler, vec![1.0, 0.0], &[], &[], Some(filler)))
                .unwrap();
        }

        let query = Embedding::new(vec![1.0, 0.0]).unwrap();
        let hits = store
            .query(
                &ns(),
                &query,
                Some(&ctx("apple fruit")),
                &Filter::default(),
                top_k(10),
            )
            .unwrap();
        let sparse_of = |id: EntryId| hits.iter().find(|h| h.id == id).unwrap().sparse_score;
        assert!(sparse_of(apple.id) > sparse_of(fruit.id));
        assert!(hits.iter().all(|h| (0.0..=1.0).contains(&h.sparse_score)));
    }

    #[test]
    fn context_bm25_zero_without_term_overlap() {
        // Both entries store a context, but only the one whose context shares a term with the
        // query context scores above zero.
        let store = store();
        let aspirin = entry("a", vec![1.0, 0.0], &[], &[], Some("aspirin dose"));
        let weather = entry("b", vec![1.0, 0.0], &[], &[], Some("weather forecast"));
        store.upsert(&ns(), aspirin.clone()).unwrap();
        store.upsert(&ns(), weather.clone()).unwrap();

        let query = Embedding::new(vec![1.0, 0.0]).unwrap();
        let hits = store
            .query(
                &ns(),
                &query,
                Some(&ctx("aspirin")),
                &Filter::default(),
                top_k(10),
            )
            .unwrap();
        let sparse_of = |id: EntryId| hits.iter().find(|h| h.id == id).unwrap().sparse_score;
        assert!(sparse_of(aspirin.id) > 0.0);
        assert_eq!(sparse_of(weather.id), 0.0);
    }

    #[test]
    fn context_bm25_zero_when_query_or_entry_has_no_context() {
        // No query context -> every candidate scores 0; a stored-context-less candidate scores 0
        // even when the query carries a context.
        let store = store();
        let with_ctx = entry("a", vec![1.0, 0.0], &[], &[], Some("aspirin dose"));
        let without_ctx = entry("b", vec![1.0, 0.0], &[], &[], None);
        store.upsert(&ns(), with_ctx.clone()).unwrap();
        store.upsert(&ns(), without_ctx.clone()).unwrap();
        let query = Embedding::new(vec![1.0, 0.0]).unwrap();

        let no_query_ctx = store
            .query(&ns(), &query, None, &Filter::default(), top_k(10))
            .unwrap();
        assert!(no_query_ctx.iter().all(|h| h.sparse_score == 0.0));

        let with_query_ctx = store
            .query(
                &ns(),
                &query,
                Some(&ctx("aspirin")),
                &Filter::default(),
                top_k(10),
            )
            .unwrap();
        let sparse_of = |id: EntryId| {
            with_query_ctx
                .iter()
                .find(|h| h.id == id)
                .unwrap()
                .sparse_score
        };
        assert!(sparse_of(with_ctx.id) > 0.0);
        assert_eq!(sparse_of(without_ctx.id), 0.0);
    }
}
