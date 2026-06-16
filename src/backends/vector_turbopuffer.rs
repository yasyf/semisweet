//! turbopuffer HTTP `VectorStorageBackend` over the `/v2` API.

use std::collections::{BTreeSet, HashMap};
use std::num::NonZeroUsize;
use std::sync::RwLock;
use std::time::UNIX_EPOCH;

use reqwest::StatusCode;
use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::error::{Error, Result};
use crate::newtype::{Context, Dim, Embedding, Entity, EntryId, Namespace};
use crate::vector::{Filter, ScoredHit, VectorEntry, VectorStorageBackend};

const API_KEY_ENV: &str = "TURBOPUFFER_API_KEY";
const API_BASE_ENV: &str = "TURBOPUFFER_API_BASE";
const DEFAULT_BASE_URL: &str = "https://api.turbopuffer.com";
const DEFAULT_NAMESPACE_PREFIX: &str = "semisweet-";
const DISTANCE_METRIC: &str = "cosine_distance";

#[derive(Deserialize)]
struct QueryResponse {
    #[serde(default)]
    rows: Vec<QueryRow>,
}

#[derive(Deserialize)]
struct QueryRow {
    id: String,
    #[serde(rename = "$dist", default)]
    dist: f32,
    #[serde(default)]
    entities: Vec<String>,
    #[serde(default)]
    context: Option<String>,
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
        Ok(Self::from_parts(
            api_key,
            base_url,
            DEFAULT_NAMESPACE_PREFIX.to_owned(),
        ))
    }

    fn from_parts(api_key: String, base_url: String, namespace_prefix: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url,
            namespace_prefix,
            dims: RwLock::new(HashMap::new()),
        }
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
        if let Some(existing) = cached {
            if existing != dim {
                return Err(Error::DimMismatch {
                    got: dim.get(),
                    want: existing.get(),
                });
            }
        }

        let date = entry
            .date
            .duration_since(UNIX_EPOCH)
            .map_err(|e| Error::VectorStorage(Box::new(e)))?
            .as_secs();

        let mut row = Map::new();
        row.insert("id".to_owned(), json!(entry.id.to_string()));
        row.insert("vector".to_owned(), json!(entry.vector.values()));
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
        row.insert("date".to_owned(), json!(date));

        let body = json!({
            "upsert_rows": [Value::Object(row)],
            "distance_metric": DISTANCE_METRIC,
            "schema": {
                "vector": { "type": format!("[{}]f32", dim.get()), "ann": true },
                "entities": { "type": "[]string", "filterable": true },
                "keys": { "type": "[]string", "filterable": true },
                "context": { "type": "string" },
                "date": { "type": "uint" },
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
            return Err(Error::DimMismatch {
                got: dim.get(),
                want: cached.unwrap_or(dim).get(),
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
        filter: &Filter,
        top_k: NonZeroUsize,
    ) -> Result<Vec<ScoredHit>> {
        let mut body = Map::new();
        body.insert(
            "rank_by".to_owned(),
            json!(["vector", "ANN", vector.values()]),
        );
        body.insert("top_k".to_owned(), json!(top_k.get()));
        body.insert("distance_metric".to_owned(), json!(DISTANCE_METRIC));
        body.insert(
            "include_attributes".to_owned(),
            json!(["entities", "keys", "context"]),
        );
        if let Some(filters) = filter_to_tp(filter) {
            body.insert("filters".to_owned(), filters);
        }

        let url = format!(
            "{}/v2/namespaces/{}/query",
            self.base_url,
            self.physical_namespace(ns)
        );
        let (status, text) = self.execute(&url, &Value::Object(body))?;
        if status == StatusCode::NOT_FOUND {
            return Ok(Vec::new());
        }
        if !status.is_success() {
            if status.is_client_error() && is_dimension_error(&text) {
                return Err(Error::DimMismatch {
                    got: vector.dim().get(),
                    want: self.cached_dim(ns)?.unwrap_or(vector.dim()).get(),
                });
            }
            return Err(Error::VectorStorage(
                format!("turbopuffer query failed ({status}): {text}").into(),
            ));
        }

        let parsed: QueryResponse =
            serde_json::from_str(&text).map_err(|e| Error::VectorStorage(Box::new(e)))?;
        parsed.rows.into_iter().map(row_to_hit).collect()
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

fn row_to_hit(row: QueryRow) -> Result<ScoredHit> {
    let id = EntryId::from_hex(&row.id).ok_or_else(|| {
        Error::VectorStorage(format!("turbopuffer returned an unparseable id `{}`", row.id).into())
    })?;
    let entities = row
        .entities
        .iter()
        .filter_map(|e| Entity::normalize(e))
        .collect::<BTreeSet<_>>();
    let context = match row.context {
        Some(value) => Some(Context::new(value)?),
        None => None,
    };
    Ok(ScoredHit {
        id,
        score: 1.0 - row.dist,
        entities,
        context,
    })
}

fn is_dimension_error(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower.contains("dimension") || lower.contains("schema")
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use httpmock::prelude::*;

    use super::*;
    use crate::newtype::{Key, QueryText};

    const NS_NAME: &str = "ns";
    const PHYSICAL_PATH: &str = "/v2/namespaces/semisweet-ns";
    const FIXED_DATE_SECS: u64 = 1_700_000_000;

    fn namespace() -> Namespace {
        Namespace::new(NS_NAME.to_owned()).unwrap()
    }

    fn store_for(server: &MockServer) -> TurbopufferVectorStore {
        TurbopufferVectorStore::from_parts(
            "test-key".to_owned(),
            server.base_url(),
            DEFAULT_NAMESPACE_PREFIX.to_owned(),
        )
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
            keys,
            entities,
            context: Some(Context::new("ctx".to_owned()).unwrap()),
            date: UNIX_EPOCH + Duration::from_secs(FIXED_DATE_SECS),
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
                "entities": ["aspirin"],
                "keys": ["k1"],
                "context": "ctx",
                "date": FIXED_DATE_SECS,
            }],
            "distance_metric": "cosine_distance",
            "schema": {
                "vector": { "type": "[2]f32", "ann": true },
                "entities": { "type": "[]string", "filterable": true },
                "keys": { "type": "[]string", "filterable": true },
                "context": { "type": "string" },
                "date": { "type": "uint" },
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
    fn query_sends_ann_filters_and_parses_scores() {
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

        let expected_body = json!({
            "rank_by": ["vector", "ANN", [0.0, 1.0]],
            "top_k": 3,
            "distance_metric": "cosine_distance",
            "include_attributes": ["entities", "keys", "context"],
            "filters": [
                "And",
                [
                    ["And", [["keys", "Contains", "k1"], ["keys", "Contains", "k2"]]],
                    ["Or", [["entities", "Contains", "e1"]]]
                ]
            ],
        });
        let response = json!({
            "rows": [{
                "id": row_id.to_string(),
                "$dist": 0.25,
                "entities": ["aspirin"],
                "keys": ["k1"],
                "context": "ctx",
            }],
        });
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v2/namespaces/semisweet-ns/query")
                .json_body(expected_body);
            then.status(200).json_body(response);
        });

        let vector = Embedding::new(vec![0.0, 1.0]).unwrap();
        let filter = Filter::new(keys, entities);
        let hits = store
            .query(
                &namespace(),
                &vector,
                &filter,
                NonZeroUsize::new(3).unwrap(),
            )
            .unwrap();
        mock.assert();

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, row_id);
        assert!((hits[0].score - 0.75).abs() < 1e-6);
        assert_eq!(
            hits[0].entities,
            [Entity::normalize("aspirin").unwrap()]
                .into_iter()
                .collect::<BTreeSet<_>>()
        );
        assert_eq!(hits[0].context.as_ref().map(|c| c.as_str()), Some("ctx"));
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
                &Filter::default(),
                NonZeroUsize::new(5).unwrap(),
            )
            .unwrap();
        assert!(hits.iter().any(|hit| hit.id == id));
        store.delete(&ns, &id).unwrap();
    }
}
