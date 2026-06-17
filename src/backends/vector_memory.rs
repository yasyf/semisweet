//! In-memory `VectorStorageBackend` (std `RwLock`) — brute-force cosine search paired with a
//! per-namespace BM25 lexical index for the sparse half of hybrid retrieval.

use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::sync::{PoisonError, RwLock};

use crate::error::{Error, Result};
use crate::newtype::{Dim, Embedding, EntryId, Namespace, QueryText};
use crate::vector::{Filter, ScoredHit, VectorEntry, VectorStorageBackend};

const BM25_K1: f32 = 1.2;
const BM25_B: f32 = 0.75;

fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|token| token.to_lowercase())
        .collect()
}

/// A per-namespace BM25 index over stored query texts. The backend already scans every entry
/// for cosine, so no postings lists are needed — only the document-frequency, per-document
/// term-frequency, and length statistics BM25 needs to score a candidate it is already visiting.
#[derive(Default)]
struct Bm25Index {
    document_frequency: HashMap<String, usize>,
    term_frequency: HashMap<EntryId, HashMap<String, u32>>,
    document_length: HashMap<EntryId, u32>,
    total_length: u64,
    document_count: usize,
}

impl Bm25Index {
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
            upper_bound += idf * (BM25_K1 + 1.0);
            let tf = term_frequency.get(term).copied().unwrap_or(0) as f32;
            if tf > 0.0 {
                let denominator = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * length / average_length);
                numerator += idf * (tf * (BM25_K1 + 1.0)) / denominator;
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
                shard.bm25.insert(entry.id, entry.query_text.as_str());
                shard.entries.insert(entry.id, entry);
            }
            None => {
                let mut entries = HashMap::new();
                let mut bm25 = Bm25Index::default();
                bm25.insert(entry.id, entry.query_text.as_str());
                entries.insert(entry.id, entry);
                shards.insert(ns.clone(), Shard { dim, entries, bm25 });
            }
        }
        Ok(())
    }

    fn query(
        &self,
        ns: &Namespace,
        vector: &Embedding,
        query_text: &QueryText,
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
        let query_terms: HashSet<String> = tokenize(query_text.as_str()).into_iter().collect();
        let mut hits: Vec<ScoredHit> = Vec::new();
        for entry in shard.entries.values() {
            if !filter.keys_all.is_subset(&entry.keys) {
                continue;
            }
            if !filter.entities_any.is_empty() && filter.entities_any.is_disjoint(&entry.entities) {
                continue;
            }
            let dense_score = vector.dot(&entry.vector)?;
            let sparse_score = shard.bm25.score(&entry.id, &query_terms);
            hits.push(ScoredHit {
                id: entry.id,
                dense_score,
                sparse_score,
                entities: entry.entities.clone(),
                context: entry.context.clone(),
                context_vector: entry.context_vector.clone(),
            });
        }
        // Return the union of each leg's top-k — the top-k by dense and the top-k by sparse among
        // entries with any lexical overlap — mirroring turbopuffer's ANN ∪ BM25 multi-query. The
        // weighted fusion that actually ranks these lives in `scoring`; truncating by either single
        // axis (or by an unweighted sum) would bias against the other, so we keep both legs whole.
        let k = top_k.get();
        let keep: HashSet<EntryId> = {
            let mut by_dense: Vec<&ScoredHit> = hits.iter().collect();
            by_dense.sort_by(|a, b| b.dense_score.total_cmp(&a.dense_score));
            let mut keep: HashSet<EntryId> = by_dense.iter().take(k).map(|hit| hit.id).collect();
            let mut by_sparse: Vec<&ScoredHit> =
                hits.iter().filter(|hit| hit.sparse_score > 0.0).collect();
            by_sparse.sort_by(|a, b| b.sparse_score.total_cmp(&a.sparse_score));
            keep.extend(by_sparse.iter().take(k).map(|hit| hit.id));
            keep
        };
        hits.retain(|hit| keep.contains(&hit.id));
        hits.sort_by(|a, b| b.dense_score.total_cmp(&a.dense_score));
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
        let id = EntryId::derive(&query, &keys);
        VectorEntry {
            id,
            vector: Embedding::new(vector).unwrap(),
            query_text: query,
            keys,
            entities: entityset(entities),
            context: context.map(|c| Context::new(c.to_owned()).unwrap()),
            context_vector: None,
        }
    }

    fn ns() -> Namespace {
        Namespace::new("prod".to_owned()).unwrap()
    }

    fn qt(text: &str) -> QueryText {
        QueryText::new(text.to_owned()).unwrap()
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
            .query(&ns(), &query, &qt("query"), &Filter::default(), top_k(10))
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
        let store = MemoryVectorStore::new();
        let e = entry("e", vec![1.0, 0.0], &[], &["aspirin"], Some("dosage info"));
        store.upsert(&ns(), e.clone()).unwrap();

        let query = Embedding::new(vec![1.0, 0.0]).unwrap();
        let hits = store
            .query(&ns(), &query, &qt("query"), &Filter::default(), top_k(10))
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
        let hits_a = store
            .query(&ns(), &query, &qt("query"), &filter_a, top_k(10))
            .unwrap();
        let ids_a: HashSet<EntryId> = hits_a.iter().map(|h| h.id).collect();
        assert_eq!(ids_a, [both.id, only_a.id].into_iter().collect());

        let filter_ab = Filter::new(keyset(&["a", "b"]), BTreeSet::new());
        let hits_ab = store
            .query(&ns(), &query, &qt("query"), &filter_ab, top_k(10))
            .unwrap();
        assert_eq!(hits_ab.len(), 1);
        assert_eq!(hits_ab[0].id, both.id);

        let filter_ac = Filter::new(keyset(&["a", "c"]), BTreeSet::new());
        let hits_ac = store
            .query(&ns(), &query, &qt("query"), &filter_ac, top_k(10))
            .unwrap();
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
        let hits_x = store
            .query(&ns(), &query, &qt("query"), &filter_x, top_k(10))
            .unwrap();
        assert_eq!(hits_x.len(), 1);
        assert_eq!(hits_x[0].id, xy.id);

        let filter_w = Filter::new(BTreeSet::new(), entityset(&["w"]));
        let hits_w = store
            .query(&ns(), &query, &qt("query"), &filter_w, top_k(10))
            .unwrap();
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
            .query(&ns(), &query, &qt("query"), &Filter::default(), top_k(10))
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
            .query(&ns(), &query, &qt("query"), &Filter::default(), top_k(2))
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
            .query(&ns(), &query, &qt("query"), &Filter::default(), top_k(10))
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
                .query(&ns(), &query, &qt("query"), &Filter::default(), top_k(10))
                .unwrap()
                .len(),
            1
        );

        store.delete(&ns(), &e.id).unwrap();
        assert!(
            store
                .query(&ns(), &query, &qt("query"), &Filter::default(), top_k(10))
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
                    &qt("query"),
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
            .query(&ns(), &query, &qt("query"), &Filter::default(), top_k(10))
            .unwrap();

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, second.id);
        assert!((hits[0].dense_score - 1.0).abs() < 1e-6);
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
            .query(&ns(), &query, &qt("query"), &Filter::default(), top_k(10))
            .unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn bm25_rarer_term_outranks_common_term() {
        // Same vector for every doc, so dense ties and only the BM25 (IDF) signal separates
        // them: the doc sharing the rare term `apple` must out-score the one sharing `fruit`.
        let store = MemoryVectorStore::new();
        let apple = entry("apple", vec![1.0, 0.0], &[], &[], None);
        let fruit = entry("fruit", vec![1.0, 0.0], &[], &[], None);
        store.upsert(&ns(), apple.clone()).unwrap();
        store.upsert(&ns(), fruit.clone()).unwrap();
        for filler in ["fruit red", "fruit green", "fruit yellow"] {
            store
                .upsert(&ns(), entry(filler, vec![1.0, 0.0], &[], &[], None))
                .unwrap();
        }

        let query = Embedding::new(vec![1.0, 0.0]).unwrap();
        let hits = store
            .query(
                &ns(),
                &query,
                &qt("apple fruit"),
                &Filter::default(),
                top_k(10),
            )
            .unwrap();
        let sparse_of = |id: EntryId| hits.iter().find(|h| h.id == id).unwrap().sparse_score;
        assert!(sparse_of(apple.id) > sparse_of(fruit.id));
        assert!(hits.iter().all(|h| (0.0..=1.0).contains(&h.sparse_score)));
    }

    #[test]
    fn bm25_sparse_is_zero_without_lexical_overlap() {
        let store = MemoryVectorStore::new();
        store
            .upsert(&ns(), entry("aspirin dose", vec![1.0, 0.0], &[], &[], None))
            .unwrap();
        let query = Embedding::new(vec![1.0, 0.0]).unwrap();
        let overlap = store
            .query(&ns(), &query, &qt("aspirin"), &Filter::default(), top_k(10))
            .unwrap();
        let disjoint = store
            .query(&ns(), &query, &qt("weather"), &Filter::default(), top_k(10))
            .unwrap();
        assert!(overlap[0].sparse_score > 0.0);
        assert_eq!(disjoint[0].sparse_score, 0.0);
    }

    #[test]
    fn bm25_reflects_reupsert_text_change() {
        // Re-upserting the same id with new text must drop the old terms and index the new ones.
        let store = MemoryVectorStore::new();
        let id = EntryId::derive(&qt("alpha"), &BTreeSet::new());
        let make = |text: &str| VectorEntry {
            id,
            vector: Embedding::new(vec![1.0, 0.0]).unwrap(),
            query_text: qt(text),
            keys: BTreeSet::new(),
            entities: BTreeSet::new(),
            context: None,
            context_vector: None,
        };
        store.upsert(&ns(), make("alpha")).unwrap();
        store.upsert(&ns(), make("beta gamma")).unwrap();

        let query = Embedding::new(vec![1.0, 0.0]).unwrap();
        let hits_alpha = store
            .query(&ns(), &query, &qt("alpha"), &Filter::default(), top_k(10))
            .unwrap();
        let hits_beta = store
            .query(&ns(), &query, &qt("beta"), &Filter::default(), top_k(10))
            .unwrap();
        assert_eq!(hits_alpha.len(), 1);
        assert_eq!(hits_alpha[0].sparse_score, 0.0);
        assert!(hits_beta[0].sparse_score > 0.0);
    }
}
