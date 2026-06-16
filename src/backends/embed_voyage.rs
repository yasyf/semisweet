//! Voyage HTTP `EmbeddingBackend`.
//!
//! Thin wrapper over Voyage AI's `/v1/embeddings` endpoint using a blocking
//! `reqwest` client. Mirrors the request contract bioqa uses: a single-element
//! `input` array, the `query` `input_type`, the configured `output_dimension`,
//! and `output_dtype: "float"`.

use std::fmt;
use std::num::NonZeroUsize;

use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

use crate::embedding::EmbeddingBackend;
use crate::error::{Error, Result};
use crate::newtype::{Dim, Embedding};

const DEFAULT_BASE_URL: &str = "https://api.voyageai.com/v1/embeddings";
const API_KEY_ENV: &str = "VOYAGE_API_KEY";
const BASE_URL_ENV: &str = "VOYAGE_API_BASE";
const OUTPUT_DTYPE: &str = "float";
const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
const CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

#[derive(Debug)]
enum VoyageError {
    Http { status: u16, body: String },
    EmptyData,
}

impl fmt::Display for VoyageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VoyageError::Http { status, body } => {
                write!(f, "voyage embeddings api returned status {status}: {body}")
            }
            VoyageError::EmptyData => f.write_str("voyage embeddings response contained no data"),
        }
    }
}

impl std::error::Error for VoyageError {}

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    input: [&'a str; 1],
    model: &'a str,
    input_type: &'a str,
    output_dimension: usize,
    output_dtype: &'a str,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

pub struct VoyageEmbedding {
    client: Client,
    api_key: String,
    model: String,
    dim: Dim,
    base_url: String,
}

impl VoyageEmbedding {
    pub fn new(model: String, output_dimension: NonZeroUsize) -> Result<Self> {
        let api_key = std::env::var(API_KEY_ENV).map_err(|_| Error::MissingEnv(API_KEY_ENV))?;
        let base_url = std::env::var(BASE_URL_ENV).unwrap_or_else(|_| DEFAULT_BASE_URL.to_owned());
        let client = Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .connect_timeout(CONNECT_TIMEOUT)
            .build()
            .map_err(|e| Error::Embedding(Box::new(e)))?;
        Ok(Self {
            client,
            api_key,
            model,
            dim: Dim::new(output_dimension.get())?,
            base_url,
        })
    }

    fn embed(&self, text: &str) -> Result<Embedding> {
        let request = EmbeddingRequest {
            input: [text],
            model: &self.model,
            input_type: "query",
            output_dimension: self.dim.get(),
            output_dtype: OUTPUT_DTYPE,
        };
        let response = self
            .client
            .post(&self.base_url)
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .map_err(|e| Error::Embedding(Box::new(e)))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().map_err(|e| Error::Embedding(Box::new(e)))?;
            return Err(Error::Embedding(Box::new(VoyageError::Http {
                status: status.as_u16(),
                body,
            })));
        }

        let mut parsed: EmbeddingResponse =
            response.json().map_err(|e| Error::Embedding(Box::new(e)))?;
        parsed.data.sort_by_key(|d| d.index);
        let first = parsed
            .data
            .into_iter()
            .next()
            .ok_or_else(|| Error::Embedding(Box::new(VoyageError::EmptyData)))?;
        Embedding::new(first.embedding)
    }
}

impl EmbeddingBackend for VoyageEmbedding {
    fn dim(&self) -> Dim {
        self.dim
    }

    fn embed_query(&self, text: &str) -> Result<Embedding> {
        self.embed(text)
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as _;

    use httpmock::prelude::*;
    use serde_json::json;

    use super::*;
    use crate::backends::ENV_LOCK;

    fn test_backend(server: &MockServer, dim: usize) -> VoyageEmbedding {
        VoyageEmbedding {
            client: Client::new(),
            api_key: "test-key".to_owned(),
            model: "voyage-3.5-lite".to_owned(),
            dim: Dim::new(dim).unwrap(),
            base_url: server.url("/v1/embeddings"),
        }
    }

    #[test]
    fn embed_query_posts_contract_and_parses_unit_vector() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1/embeddings")
                .header("authorization", "Bearer test-key")
                .json_body(json!({
                    "input": ["aspirin dose"],
                    "model": "voyage-3.5-lite",
                    "input_type": "query",
                    "output_dimension": 2,
                    "output_dtype": "float",
                }));
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({
                    "data": [{ "embedding": [3.0, 4.0], "index": 0 }],
                }));
        });

        let backend = test_backend(&server, 2);
        let embedding = backend.embed_query("aspirin dose").unwrap();

        mock.assert();
        assert_eq!(embedding.dim().get(), 2);
        let norm = embedding.values().iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((embedding.values()[0] - 0.6).abs() < 1e-6);
        assert!((embedding.values()[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn data_is_sorted_by_index_before_taking_first() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST).path("/v1/embeddings");
            then.status(200).json_body(json!({
                "data": [
                    { "embedding": [0.0, 1.0], "index": 1 },
                    { "embedding": [1.0, 0.0], "index": 0 },
                ],
            }));
        });

        let backend = test_backend(&server, 2);
        let embedding = backend.embed_query("x").unwrap();

        mock.assert();
        assert!((embedding.values()[0] - 1.0).abs() < 1e-6);
        assert!(embedding.values()[1].abs() < 1e-6);
    }

    #[test]
    fn non_success_status_maps_to_embedding_error_with_status_and_body() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST).path("/v1/embeddings");
            then.status(429).body("rate limit exceeded");
        });

        let backend = test_backend(&server, 2);
        let err = backend.embed_query("x").unwrap_err();

        mock.assert();
        assert!(matches!(err, Error::Embedding(_)));
        let source = err.source().unwrap().to_string();
        assert!(source.contains("429"), "source was: {source}");
        assert!(
            source.contains("rate limit exceeded"),
            "source was: {source}"
        );
    }

    #[test]
    fn empty_data_maps_to_embedding_error() {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST).path("/v1/embeddings");
            then.status(200).json_body(json!({ "data": [] }));
        });

        let backend = test_backend(&server, 2);
        let err = backend.embed_query("x").unwrap_err();

        mock.assert();
        assert!(matches!(err, Error::Embedding(_)));
        assert!(err.source().unwrap().to_string().contains("no data"));
    }

    #[test]
    fn missing_api_key_env_errors() {
        let _guard = ENV_LOCK.lock().unwrap();
        // SAFETY: serialized by ENV_LOCK against the other env-touching tests.
        unsafe {
            std::env::remove_var(API_KEY_ENV);
        }
        let dim = NonZeroUsize::new(512).unwrap();
        let result = VoyageEmbedding::new("voyage-3.5-lite".to_owned(), dim);
        assert!(matches!(result, Err(Error::MissingEnv("VOYAGE_API_KEY"))));
    }

    #[test]
    #[ignore = "hits the live Voyage API; requires VOYAGE_API_KEY"]
    fn live_embed_query_returns_configured_dim() {
        let dim = NonZeroUsize::new(512).unwrap();
        let backend = VoyageEmbedding::new("voyage-3.5-lite".to_owned(), dim).unwrap();
        let embedding = backend.embed_query("what is aspirin used for").unwrap();
        assert_eq!(embedding.dim().get(), 512);
        let norm = embedding.values().iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }
}
