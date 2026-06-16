//! GLiNER zero-shot NER `EntityBackend`, on upstream `gline-rs` span mode.
//!
//! A single inference runs at [`FULL_THRESHOLD`]; both the `fast` and full
//! results are then derived by filtering that one span list at a higher or
//! lower cutoff, so `extract(t, fast = true)` is a subset of `extract(t, false)`
//! by construction.
//!
//! The ONNX model + tokenizer are fetched from the Hugging Face Hub on first
//! construction — mirroring fastembed's BGE download — and cached on disk. Explicit
//! `model`/`tokenizer` paths bypass the download.

use std::collections::BTreeSet;
use std::path::PathBuf;

use gliner::model::GLiNER;
use gliner::model::input::text::TextInput;
use gliner::model::params::Parameters;
use gliner::model::pipeline::span::SpanMode;
use hf_hub::api::sync::ApiBuilder;

use crate::entity::EntityBackend;
use crate::error::{Error, Result};
use crate::newtype::Entity;

const DEFAULT_REPO: &str = "onnx-community/gliner_small-v2.1";
const MODEL_FILE: &str = "onnx/model.onnx";
const TOKENIZER_FILE: &str = "tokenizer.json";
// Shared with fastembed's BGE cache so a single env var pins all local model files.
const MODEL_CACHE_ENV: &str = "SEMISWEET_MODEL_CACHE";

const FULL_THRESHOLD: f32 = 0.50;
const FAST_CUTOFF: f32 = 0.85;

fn entities_from_spans(spans: &[(String, f32)], cutoff: f32) -> BTreeSet<Entity> {
    spans
        .iter()
        .filter(|(_, score)| *score >= cutoff)
        .filter_map(|(text, _)| Entity::normalize(text))
        .collect()
}

fn download(repo: &str, file: &str) -> Result<PathBuf> {
    let mut builder = ApiBuilder::new();
    if let Some(cache_dir) = std::env::var_os(MODEL_CACHE_ENV) {
        builder = builder.with_cache_dir(PathBuf::from(cache_dir));
    }
    let api = builder
        .build()
        .map_err(|e| Error::EntityExtraction(Box::new(e)))?;
    api.model(repo.to_owned())
        .get(file)
        .map_err(|e| Error::EntityExtraction(Box::new(e)))
}

pub struct GlinerEntities {
    model: GLiNER<SpanMode>,
    labels: Vec<String>,
}

impl GlinerEntities {
    pub fn new(
        labels: Vec<String>,
        repo: Option<String>,
        model: Option<String>,
        tokenizer: Option<String>,
    ) -> Result<Self> {
        let repo = repo.unwrap_or_else(|| DEFAULT_REPO.to_owned());
        let tokenizer_path = match tokenizer {
            Some(path) => PathBuf::from(path),
            None => download(&repo, TOKENIZER_FILE)?,
        };
        let model_path = match model {
            Some(path) => PathBuf::from(path),
            None => download(&repo, MODEL_FILE)?,
        };
        let params = Parameters::default().with_threshold(FULL_THRESHOLD);
        let model = GLiNER::<SpanMode>::new(params, Default::default(), tokenizer_path, model_path)
            .map_err(Error::EntityExtraction)?;
        Ok(Self { model, labels })
    }

    fn scored_spans(&self, text: &str) -> Result<Vec<(String, f32)>> {
        let input = TextInput::new(vec![text.to_owned()], self.labels.clone())
            .map_err(Error::EntityExtraction)?;
        let output = self
            .model
            .inference(input)
            .map_err(Error::EntityExtraction)?;
        Ok(output
            .spans
            .iter()
            .flatten()
            .map(|span| (span.text().to_owned(), span.probability()))
            .collect())
    }
}

impl EntityBackend for GlinerEntities {
    fn extract(&self, text: &str, fast: bool) -> Result<BTreeSet<Entity>> {
        let cutoff = if fast { FAST_CUTOFF } else { FULL_THRESHOLD };
        let spans = self.scored_spans(text)?;
        Ok(entities_from_spans(&spans, cutoff))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entity(raw: &str) -> Entity {
        Entity::normalize(raw).unwrap()
    }

    #[test]
    fn fast_cutoff_keeps_a_subset_of_full() {
        let spans = vec![
            ("Aspirin".to_owned(), 0.95),
            ("James Bond".to_owned(), 0.60),
            ("noise".to_owned(), 0.40),
        ];

        let full = entities_from_spans(&spans, FULL_THRESHOLD);
        let fast = entities_from_spans(&spans, FAST_CUTOFF);

        assert!(fast.is_subset(&full));
        assert_eq!(
            full,
            BTreeSet::from([entity("aspirin"), entity("james bond")])
        );
        assert_eq!(fast, BTreeSet::from([entity("aspirin")]));
    }

    #[test]
    fn normalization_collapses_whitespace_lowercases_and_dedupes() {
        let spans = vec![
            ("  Aspirin   Tablet ".to_owned(), 0.90),
            ("ASPIRIN TABLET".to_owned(), 0.88),
            ("   ".to_owned(), 0.99),
        ];

        let entities = entities_from_spans(&spans, FAST_CUTOFF);

        assert_eq!(entities, BTreeSet::from([entity("aspirin tablet")]));
    }

    #[test]
    fn boundary_score_is_inclusive_and_empty_list_is_empty() {
        let spans = vec![("Edge".to_owned(), FAST_CUTOFF)];
        assert_eq!(
            entities_from_spans(&spans, FAST_CUTOFF),
            BTreeSet::from([entity("edge")])
        );
        assert!(entities_from_spans(&[], FULL_THRESHOLD).is_empty());
    }

    #[test]
    #[ignore = "downloads GLiNER model"]
    fn extract_fast_is_subset_of_full_end_to_end() {
        let labels = ["person", "organization", "location"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        let backend = GlinerEntities::new(labels, None, None, None).unwrap();
        let text = "Barack Obama was born in Hawaii and later worked with Microsoft.";

        let full = backend.extract(text, false).unwrap();
        let fast = backend.extract(text, true).unwrap();

        assert!(fast.is_subset(&full));
        assert!(!full.is_empty());
        assert!(full.iter().any(|e| e.as_str().contains("obama")));
    }
}
