//! turbopuffer HTTP `VectorStorageBackend` over the `/v2` API.

use std::collections::{BTreeSet, HashMap};
use std::num::NonZeroUsize;
use std::sync::RwLock;

use reqwest::StatusCode;
use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::error::{Error, Result};
use crate::newtype::{Context, Dim, Embedding, Entity, EntryId, Namespace, QueryText};
use crate::vector::{Filter, ScoredHit, VectorEntry, VectorStorageBackend};

const API_KEY_ENV: &str = "TURBOPUFFER_API_KEY";
const API_BASE_ENV: &str = "TURBOPUFFER_API_BASE";
const DEFAULT_BASE_URL: &str = "https://api.turbopuffer.com";
const DEFAULT_NAMESPACE_PREFIX: &str = "semisweet-";
const DISTANCE_METRIC: &str = "cosine_distance";
// turbopuffer hides its FTS corpus statistics, so the raw BM25 `$dist` is squashed into `[0, 1]`
// by `s / (s + BM25_SATURATION)`. The constant is the BM25 score at which sparse relevance is 0.5.
const BM25_SATURATION: f32 = 5.0;
const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
const CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

#[derive(Deserialize)]
struct MultiQueryResponse {
    #[serde(default)]
    results: Vec<QueryResult>,
}

#[derive(Deserialize)]
struct QueryResult {
    #[serde(default)]
    rows: Vec<QueryRow>,
}

#[derive(Deserialize)]
struct QueryRow {
    id: String,
    #[serde(rename = "$dist")]
    dist: f32,
    #[serde(default)]
    entities: Vec<String>,
    #[serde(default)]
    context: Option<String>,
    #[serde(default)]
    context_vector: Option<Vec<f32>>,
}

pub struct TurbopufferVectorStore {
    client: Client,
    api_key: String,
    base_url: String,
    namespace_prefix: String,
    dims: RwLock<HashMap<Namespace, Dim>>,
}

impl TurbopufferVectorStore {
    pub fn new() -> Result<Self> {
        let api_key = std::env::var(API_KEY_ENV).map_err(|_| Error::MissingEnv(API_KEY_ENV))?;
        let base_url = std::env::var(API_BASE_ENV).unwrap_or_else(|_| DEFAULT_BASE_URL.to_owned());
        Self::from_parts(api_key, base_url, DEFAULT_NAMESPACE_PREFIX.to_owned())
    }

    fn from_parts(api_key: String, base_url: String, namespace_prefix: String) -> Result<Self> {
        let client = Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .connect_timeout(CONNECT_TIMEOUT)
            .build()
            .map_err(|e| Error::VectorStorage(Box::new(e)))?;
        Ok(Self {
            client,
            api_key,
            base_url,
            namespace_prefix,
            dims: RwLock::new(HashMap::new()),
        })
    }

    fn physical_namespace(&self, ns: &Namespace) -> String {
        format!("{}{}", self.namespace_prefix, ns.as_str())
    }

    fn cached_dim(&self, ns: &Namespace) -> Result<Option<Dim>> {
        let guard = self
            .dims
            .read()
            .map_err(|_| Error::VectorStorage("dim cache lock poisoned".into()))?;
        Ok(guard.get(ns).copied())
    }

    fn cache_dim(&self, ns: &Namespace, dim: Dim) -> Result<()> {
        let mut guard = self
            .dims
            .write()
            .map_err(|_| Error::VectorStorage("dim cache lock poisoned".into()))?;
        guard.insert(ns.clone(), dim);
        Ok(())
    }

    fn execute(&self, url: &str, body: &Value) -> Result<(StatusCode, String)> {
        let response = self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
            .json(body)
            .send()
            .map_err(|e| Error::VectorStorage(Box::new(e)))?;
        let status = response.status();
        let text = response
            .text()
            .map_err(|e| Error::VectorStorage(Box::new(e)))?;
        Ok((status, text))
    }
}

impl VectorStorageBackend for TurbopufferVectorStore {
    fn upsert(&self, ns: &Namespace, entry: VectorEntry) -> Result<()> {
        let dim = entry.vector.dim();
        let cached = self.cached_dim(ns)?;
        if let Some(existing) = cached
            && existing != dim
        {
            return Err(Error::DimMismatch {
                got: dim.get(),
                want: existing.get(),
            });
        }

        let mut row = Map::new();
        row.insert("id".to_owned(), json!(entry.id.to_string()));
        row.insert("vector".to_owned(), json!(entry.vector.values()));
        row.insert("query_text".to_owned(), json!(entry.query_text.as_str()));
        row.insert(
            "entities".to_owned(),
            json!(
                entry
                    .entities
                    .iter()
                    .map(|e| e.as_str())
                    .collect::<Vec<_>>()
            ),
        );
        row.insert(
            "keys".to_owned(),
            json!(entry.keys.iter().map(|k| k.as_str()).collect::<Vec<_>>()),
        );
        if let Some(context) = &entry.context {
            row.insert("context".to_owned(), json!(context.as_str()));
        }
        if let Some(context_vector) = &entry.context_vector {
            row.insert("context_vector".to_owned(), json!(context_vector.values()));
        }

        let body = json!({
            "upsert_rows": [Value::Object(row)],
            "distance_metric": DISTANCE_METRIC,
            "schema": {
                "vector": { "type": format!("[{}]f32", dim.get()), "ann": true },
                "query_text": { "type": "string", "full_text_search": true },
                "entities": { "type": "[]string", "filterable": true },
                "keys": { "type": "[]string", "filterable": true },
                "context": { "type": "string" },
                "context_vector": { "type": "[]float", "filterable": false },
            },
        });

        let url = format!(
            "{}/v2/namespaces/{}",
            self.base_url,
            self.physical_namespace(ns)
        );
        let (status, text) = self.execute(&url, &body)?;
        if status.is_success() {
            return self.cache_dim(ns, dim);
        }
        if status.is_client_error() && is_dimension_error(&text) {
            return Err(match cached {
                Some(want) => Error::DimMismatch {
                    got: dim.get(),
                    want: want.get(),
                },
                None => Error::VectorStorage(
                    format!(
                        "turbopuffer rejected upsert dimension {} for namespace `{}`; expected remote dimension unknown: {text}",
                        dim.get(),
                        self.physical_namespace(ns)
                    )
                    .into(),
                ),
            });
        }
        Err(Error::VectorStorage(
            format!("turbopuffer upsert failed ({status}): {text}").into(),
        ))
    }

    fn query(
        &self,
        ns: &Namespace,
        vector: &Embedding,
        query_text: &QueryText,
        filter: &Filter,
        top_k: NonZeroUsize,
    ) -> Result<Vec<ScoredHit>> {
        // Hybrid retrieval is one multi-query round trip: a dense ANN leg and a lexical BM25 leg.
        // turbopuffer's server-side fusion is rank-based (RRF) only, so we omit `rerank_by`, read
        // the raw `$dist` from each leg, and fuse with weighted scores client-side in `scoring`.
        let attributes = json!(["entities", "keys", "context", "context_vector"]);
        let filters = filter_to_tp(filter);

        let mut ann = Map::new();
        ann.insert(
            "rank_by".to_owned(),
            json!(["vector", "ANN", vector.values()]),
        );
        ann.insert("top_k".to_owned(), json!(top_k.get()));
        ann.insert("distance_metric".to_owned(), json!(DISTANCE_METRIC));
        ann.insert("include_attributes".to_owned(), attributes.clone());
        if let Some(filters) = &filters {
            ann.insert("filters".to_owned(), filters.clone());
        }

        let mut bm25 = Map::new();
        bm25.insert(
            "rank_by".to_owned(),
            json!(["query_text", "BM25", query_text.as_str()]),
        );
        bm25.insert("top_k".to_owned(), json!(top_k.get()));
        bm25.insert("include_attributes".to_owned(), attributes);
        if let Some(filters) = &filters {
            bm25.insert("filters".to_owned(), filters.clone());
        }

        let body = json!({ "queries": [Value::Object(ann), Value::Object(bm25)] });
        let url = format!(
            "{}/v2/namespaces/{}/query",
            self.base_url,
            self.physical_namespace(ns)
        );
        let (status, text) = self.execute(&url, &body)?;
        if status == StatusCode::NOT_FOUND {
            return Ok(Vec::new());
        }
        if !status.is_success() {
            if status.is_client_error() && is_dimension_error(&text) {
                return Err(match self.cached_dim(ns)? {
                    Some(want) => Error::DimMismatch {
                        got: vector.dim().get(),
                        want: want.get(),
                    },
                    None => Error::VectorStorage(
                        format!(
                            "turbopuffer rejected query dimension {}; expected remote dimension unknown: {text}",
                            vector.dim().get()
                        )
                        .into(),
                    ),
                });
            }
            return Err(Error::VectorStorage(
                format!("turbopuffer query failed ({status}): {text}").into(),
            ));
        }

        let parsed: MultiQueryResponse =
            serde_json::from_str(&text).map_err(|e| Error::VectorStorage(Box::new(e)))?;
        fuse_multi_query(parsed)
    }

    fn delete(&self, ns: &Namespace, id: &EntryId) -> Result<()> {
        let body = json!({ "deletes": [id.to_string()] });
        let url = format!(
            "{}/v2/namespaces/{}",
            self.base_url,
            self.physical_namespace(ns)
        );
        let (status, text) = self.execute(&url, &body)?;
        if status.is_success() {
            return Ok(());
        }
        Err(Error::VectorStorage(
            format!("turbopuffer delete failed ({status}): {text}").into(),
        ))
    }
}

fn filter_to_tp(filter: &Filter) -> Option<Value> {
    let mut clauses: Vec<Value> = Vec::new();
    if !filter.keys_all.is_empty() {
        let conds: Vec<Value> = filter
            .keys_all
            .iter()
            .map(|k| json!(["keys", "Contains", k.as_str()]))
            .collect();
        clauses.push(json!(["And", conds]));
    }
    if !filter.entities_any.is_empty() {
        let conds: Vec<Value> = filter
            .entities_any
            .iter()
            .map(|e| json!(["entities", "Contains", e.as_str()]))
            .collect();
        clauses.push(json!(["Or", conds]));
    }
    if clauses.is_empty() {
        None
    } else if clauses.len() == 1 {
        clauses.pop()
    } else {
        Some(json!(["And", clauses]))
    }
}

fn squash_bm25(score: f32) -> f32 {
    score / (score + BM25_SATURATION)
}

struct ParsedRow {
    id: EntryId,
    entities: BTreeSet<Entity>,
    context: Option<Context>,
    context_vector: Option<Embedding>,
}

fn parse_row(row: &QueryRow) -> Result<ParsedRow> {
    let id = EntryId::from_hex(&row.id).ok_or_else(|| {
        Error::VectorStorage(format!("turbopuffer returned an unparseable id `{}`", row.id).into())
    })?;
    let entities = row
        .entities
        .iter()
        .filter_map(|e| Entity::normalize(e))
        .collect::<BTreeSet<_>>();
    let context = match &row.context {
        Some(value) => Some(Context::new(value.clone())?),
        None => None,
    };
    let context_vector = match &row.context_vector {
        Some(values) => Some(Embedding::new(values.clone())?),
        None => None,
    };
    Ok(ParsedRow {
        id,
        entities,
        context,
        context_vector,
    })
}

/// Fuse the two legs of a hybrid multi-query into one candidate set. The first leg is the dense
/// ANN result (`$dist` is cosine distance → `dense = 1 - dist`); the second is BM25 (`$dist` is the
/// BM25 score → squashed into `[0, 1]`). Candidates are unioned by id, preserving the dense leg's
/// order then the BM25-only tail; a candidate absent from one leg keeps that component at 0.
fn fuse_multi_query(parsed: MultiQueryResponse) -> Result<Vec<ScoredHit>> {
    let mut legs = parsed.results.into_iter();
    let ann_rows = legs.next().map(|leg| leg.rows).unwrap_or_default();
    let bm25_rows = legs.next().map(|leg| leg.rows).unwrap_or_default();

    let mut hits: HashMap<EntryId, ScoredHit> = HashMap::new();
    let mut order: Vec<EntryId> = Vec::new();

    for row in &ann_rows {
        let parsed = parse_row(row)?;
        let id = parsed.id;
        if !hits.contains_key(&id) {
            order.push(id);
        }
        hits.insert(
            id,
            ScoredHit {
                id,
                dense_score: 1.0 - row.dist,
                sparse_score: 0.0,
                entities: parsed.entities,
                context: parsed.context,
                context_vector: parsed.context_vector,
            },
        );
    }
    for row in &bm25_rows {
        let parsed = parse_row(row)?;
        let id = parsed.id;
        match hits.get_mut(&id) {
            Some(hit) => hit.sparse_score = squash_bm25(row.dist),
            None => {
                order.push(id);
                hits.insert(
                    id,
                    ScoredHit {
                        id,
                        dense_score: 0.0,
                        sparse_score: squash_bm25(row.dist),
                        entities: parsed.entities,
                        context: parsed.context,
                        context_vector: parsed.context_vector,
                    },
                );
            }
        }
    }
    Ok(order
        .into_iter()
        .filter_map(|id| hits.remove(&id))
        .collect())
}

// turbopuffer exposes no stable structured error code or field for a vector
// dimension conflict — its `/v2` 4xx bodies carry only a human-readable message
// — so this is a best-effort classification of that text. Callers must gate on a
// 4xx status so a transient 5xx is never read as a dimension error. We match the
// explicit "dimension" wording, and only treat a "schema" rejection as
// dimension-related when it also names the "vector" field, so an unrelated schema
// error on another attribute is not misclassified.
fn is_dimension_error(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower.contains("dimension") || (lower.contains("schema") && lower.contains("vector"))
}

#[cfg(test)]
mod tests {
    use httpmock::prelude::*;

    use super::*;
    use crate::newtype::Key;

    const NS_NAME: &str = "ns";
    const PHYSICAL_PATH: &str = "/v2/namespaces/semisweet-ns";

    fn namespace() -> Namespace {
        Namespace::new(NS_NAME.to_owned()).unwrap()
    }

    fn store_for(server: &MockServer) -> TurbopufferVectorStore {
        TurbopufferVectorStore::from_parts(
            "test-key".to_owned(),
            server.base_url(),
            DEFAULT_NAMESPACE_PREFIX.to_owned(),
        )
        .unwrap()
    }

    fn entry_with_dim(values: Vec<f32>) -> VectorEntry {
        let query = QueryText::new("dose".to_owned()).unwrap();
        let keys: BTreeSet<Key> = [Key::new("k1".to_owned()).unwrap()].into_iter().collect();
        let entities: BTreeSet<Entity> = [Entity::normalize("aspirin").unwrap()]
            .into_iter()
            .collect();
        VectorEntry {
            id: EntryId::derive(&query, &keys),
            vector: Embedding::new(values).unwrap(),
            query_text: query,
            keys,
            entities,
            context: Some(Context::new("ctx".to_owned()).unwrap()),
            context_vector: None,
        }
    }

    #[test]
    fn new_requires_api_key() {
        let saved = std::env::var(API_KEY_ENV).ok();
        unsafe { std::env::remove_var(API_KEY_ENV) };
        let result = TurbopufferVectorStore::new();
        if let Some(value) = saved {
            unsafe { std::env::set_var(API_KEY_ENV, value) };
        }
        assert!(matches!(
            result,
            Err(Error::MissingEnv("TURBOPUFFER_API_KEY"))
        ));
    }

    #[test]
    fn filter_to_tp_builds_and_or_tree() {
        assert!(filter_to_tp(&Filter::default()).is_none());

        let keys: BTreeSet<Key> = [
            Key::new("k1".to_owned()).unwrap(),
            Key::new("k2".to_owned()).unwrap(),
        ]
        .into_iter()
        .collect();
        let entities: BTreeSet<Entity> = [Entity::normalize("e1").unwrap()].into_iter().collect();

        let only_keys = filter_to_tp(&Filter::new(keys.clone(), BTreeSet::new())).unwrap();
        assert_eq!(
            only_keys,
            json!([
                "And",
                [["keys", "Contains", "k1"], ["keys", "Contains", "k2"]]
            ])
        );

        let both = filter_to_tp(&Filter::new(keys, entities)).unwrap();
        assert_eq!(
            both,
            json!([
                "And",
                [
                    [
                        "And",
                        [["keys", "Contains", "k1"], ["keys", "Contains", "k2"]]
                    ],
                    ["Or", [["entities", "Contains", "e1"]]]
                ]
            ])
        );
    }

    #[test]
    fn upsert_sends_rows_schema_and_metric() {
        let server = MockServer::start();
        let store = store_for(&server);
        let entry = entry_with_dim(vec![0.0, 1.0]);
        let expected = json!({
            "upsert_rows": [{
                "id": entry.id.to_string(),
                "vector": [0.0, 1.0],
                "query_text": "dose",
                "entities": ["aspirin"],
                "keys": ["k1"],
                "context": "ctx",
            }],
            "distance_metric": "cosine_distance",
            "schema": {
                "vector": { "type": "[2]f32", "ann": true },
                "query_text": { "type": "string", "full_text_search": true },
                "entities": { "type": "[]string", "filterable": true },
                "keys": { "type": "[]string", "filterable": true },
                "context": { "type": "string" },
                "context_vector": { "type": "[]float", "filterable": false },
            },
        });
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path(PHYSICAL_PATH)
                .header("Authorization", "Bearer test-key")
                .json_body(expected);
            then.status(200).json_body(json!({ "rows_affected": 1 }));
        });

        store.upsert(&namespace(), entry).unwrap();
        mock.assert();
    }

    #[test]
    fn query_sends_multi_query_and_fuses_scores() {
        let server = MockServer::start();
        let store = store_for(&server);

        let row_id = EntryId::derive(
            &QueryText::new("dose".to_owned()).unwrap(),
            &[Key::new("k1".to_owned()).unwrap()].into_iter().collect(),
        );
        let keys: BTreeSet<Key> = [
            Key::new("k1".to_owned()).unwrap(),
            Key::new("k2".to_owned()).unwrap(),
        ]
        .into_iter()
        .collect();
        let entities: BTreeSet<Entity> = [Entity::normalize("e1").unwrap()].into_iter().collect();

        let filters = json!([
            "And",
            [
                [
                    "And",
                    [["keys", "Contains", "k1"], ["keys", "Contains", "k2"]]
                ],
                ["Or", [["entities", "Contains", "e1"]]]
            ]
        ]);
        let attributes = json!(["entities", "keys", "context", "context_vector"]);
        let expected_body = json!({
            "queries": [
                {
                    "rank_by": ["vector", "ANN", [0.0, 1.0]],
                    "top_k": 3,
                    "distance_metric": "cosine_distance",
                    "include_attributes": attributes,
                    "filters": filters,
                },
                {
                    "rank_by": ["query_text", "BM25", "dose"],
                    "top_k": 3,
                    "include_attributes": attributes,
                    "filters": filters,
                }
            ],
        });
        let response = json!({
            "results": [
                { "rows": [{
                    "id": row_id.to_string(),
                    "$dist": 0.25,
                    "entities": ["aspirin"],
                    "keys": ["k1"],
                    "context": "ctx",
                }] },
                { "rows": [{ "id": row_id.to_string(), "$dist": 5.0 }] }
            ],
        });
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v2/namespaces/semisweet-ns/query")
                .json_body(expected_body);
            then.status(200).json_body(response);
        });

        let vector = Embedding::new(vec![0.0, 1.0]).unwrap();
        let query_text = QueryText::new("dose".to_owned()).unwrap();
        let filter = Filter::new(keys, entities);
        let hits = store
            .query(
                &namespace(),
                &vector,
                &query_text,
                &filter,
                NonZeroUsize::new(3).unwrap(),
            )
            .unwrap();
        mock.assert();

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, row_id);
        assert!((hits[0].dense_score - 0.75).abs() < 1e-6);
        // squash_bm25(5.0) = 5 / (5 + 5) = 0.5
        assert!((hits[0].sparse_score - 0.5).abs() < 1e-6);
        assert_eq!(
            hits[0].entities,
            [Entity::normalize("aspirin").unwrap()]
                .into_iter()
                .collect::<BTreeSet<_>>()
        );
        assert_eq!(hits[0].context.as_ref().map(|c| c.as_str()), Some("ctx"));
    }

    #[test]
    fn query_row_missing_dist_is_error() {
        let server = MockServer::start();
        let store = store_for(&server);
        let row_id = EntryId::derive(
            &QueryText::new("dose".to_owned()).unwrap(),
            &[Key::new("k1".to_owned()).unwrap()].into_iter().collect(),
        );
        let response = json!({
            "results": [
                { "rows": [{
                    "id": row_id.to_string(),
                    "entities": ["aspirin"],
                    "keys": ["k1"],
                    "context": "ctx",
                }] },
                { "rows": [] }
            ],
        });
        let mock = server.mock(|when, then| {
            when.method(POST).path("/v2/namespaces/semisweet-ns/query");
            then.status(200).json_body(response);
        });

        let vector = Embedding::new(vec![0.0, 1.0]).unwrap();
        let err = store
            .query(
                &namespace(),
                &vector,
                &QueryText::new("dose".to_owned()).unwrap(),
                &Filter::default(),
                NonZeroUsize::new(1).unwrap(),
            )
            .unwrap_err();
        mock.assert();
        assert!(matches!(err, Error::VectorStorage(_)));
    }

    #[test]
    fn query_dimension_error_without_cached_dim_is_storage_error() {
        let server = MockServer::start();
        let store = store_for(&server);
        let mock = server.mock(|when, then| {
            when.method(POST).path("/v2/namespaces/semisweet-ns/query");
            then.status(400)
                .body("vector dimension 2 does not match index dimension 768");
        });

        let vector = Embedding::new(vec![0.0, 1.0]).unwrap();
        let err = store
            .query(
                &namespace(),
                &vector,
                &QueryText::new("dose".to_owned()).unwrap(),
                &Filter::default(),
                NonZeroUsize::new(1).unwrap(),
            )
            .unwrap_err();
        mock.assert();
        assert!(matches!(err, Error::VectorStorage(_)));
    }

    #[test]
    fn upsert_dimension_error_without_cached_dim_is_storage_error() {
        let server = MockServer::start();
        let store = store_for(&server);
        let mock = server.mock(|when, then| {
            when.method(POST).path(PHYSICAL_PATH);
            then.status(400)
                .body("vector dimension 2 does not match index dimension 768");
        });

        let err = store
            .upsert(&namespace(), entry_with_dim(vec![0.0, 1.0]))
            .unwrap_err();
        mock.assert();
        assert!(matches!(err, Error::VectorStorage(_)));
    }

    #[test]
    fn query_unknown_namespace_is_a_miss() {
        let server = MockServer::start();
        let store = store_for(&server);
        let mock = server.mock(|when, then| {
            when.method(POST).path("/v2/namespaces/semisweet-ns/query");
            then.status(404).body("namespace not found");
        });

        let vector = Embedding::new(vec![0.0, 1.0]).unwrap();
        let hits = store
            .query(
                &namespace(),
                &vector,
                &QueryText::new("dose".to_owned()).unwrap(),
                &Filter::default(),
                NonZeroUsize::new(1).unwrap(),
            )
            .unwrap();
        mock.assert();
        assert!(hits.is_empty());
    }

    #[test]
    fn query_server_error_surfaces_storage_error() {
        let server = MockServer::start();
        let store = store_for(&server);
        let mock = server.mock(|when, then| {
            when.method(POST).path("/v2/namespaces/semisweet-ns/query");
            then.status(500).body("kaboom");
        });

        let vector = Embedding::new(vec![0.0, 1.0]).unwrap();
        let err = store
            .query(
                &namespace(),
                &vector,
                &QueryText::new("dose".to_owned()).unwrap(),
                &Filter::default(),
                NonZeroUsize::new(1).unwrap(),
            )
            .unwrap_err();
        mock.assert();
        assert!(matches!(err, Error::VectorStorage(_)));
    }

    #[test]
    fn delete_posts_deletes_body() {
        let server = MockServer::start();
        let store = store_for(&server);
        let query = QueryText::new("dose".to_owned()).unwrap();
        let keys: BTreeSet<Key> = [Key::new("k1".to_owned()).unwrap()].into_iter().collect();
        let id = EntryId::derive(&query, &keys);
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path(PHYSICAL_PATH)
                .json_body(json!({ "deletes": [id.to_string()] }));
            then.status(200).json_body(json!({ "rows_affected": 1 }));
        });

        store.delete(&namespace(), &id).unwrap();
        mock.assert();
    }

    #[test]
    fn upsert_rejects_dimension_change() {
        let server = MockServer::start();
        let store = store_for(&server);
        let mock = server.mock(|when, then| {
            when.method(POST).path(PHYSICAL_PATH);
            then.status(200).json_body(json!({ "rows_affected": 1 }));
        });

        store
            .upsert(&namespace(), entry_with_dim(vec![0.0, 1.0]))
            .unwrap();
        let err = store
            .upsert(&namespace(), entry_with_dim(vec![0.0, 0.0, 1.0]))
            .unwrap_err();
        mock.assert();
        assert!(matches!(err, Error::DimMismatch { got: 3, want: 2 }));
    }

    #[test]
    #[ignore = "requires a live TURBOPUFFER_API_KEY and network access"]
    fn live_upsert_then_query_roundtrip() {
        let store = match TurbopufferVectorStore::new() {
            Ok(store) => store,
            Err(_) => return,
        };
        let ns = Namespace::new("semisweet-live-test".to_owned()).unwrap();
        let entry = entry_with_dim(vec![0.1, 0.2, 0.3]);
        let id = entry.id;
        store.upsert(&ns, entry).unwrap();

        let vector = Embedding::new(vec![0.1, 0.2, 0.3]).unwrap();
        let hits = store
            .query(
                &ns,
                &vector,
                &QueryText::new("dose".to_owned()).unwrap(),
                &Filter::default(),
                NonZeroUsize::new(5).unwrap(),
            )
            .unwrap();
        assert!(hits.iter().any(|hit| hit.id == id));
        store.delete(&ns, &id).unwrap();
    }
}
