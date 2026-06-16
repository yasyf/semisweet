//! Pure-Rust YAKE keyword `EntityBackend`.

use std::collections::BTreeSet;

use yake_rust::{Config, StopWords, get_n_best};

use crate::entity::EntityBackend;
use crate::error::{Error, Result};
use crate::newtype::Entity;

const DEFAULT_LANGUAGE: &str = "en";
const FULL_N: usize = 20;
const FAST_M: usize = 5;

pub struct KeywordEntities {
    config: Config,
    stop_words: StopWords,
}

impl KeywordEntities {
    pub fn new() -> Result<Self> {
        Self::with_language(DEFAULT_LANGUAGE)
    }

    pub fn with_language(language: &str) -> Result<Self> {
        let stop_words = StopWords::predefined(language).ok_or_else(|| {
            Error::EntityExtraction(format!("no stopwords for language `{language}`").into())
        })?;
        Ok(Self {
            config: Config::default(),
            stop_words,
        })
    }
}

impl EntityBackend for KeywordEntities {
    fn extract(&self, text: &str, fast: bool) -> Result<BTreeSet<Entity>> {
        if text.trim().is_empty() {
            return Ok(BTreeSet::new());
        }
        let ranked = get_n_best(FULL_N, text, &self.stop_words, &self.config);
        let take = if fast { FAST_M } else { FULL_N };
        Ok(ranked
            .into_iter()
            .take(take)
            .filter_map(|item| Entity::normalize(&item.keyword))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOGLE_TEXT: &str = "Google is acquiring the data science startup Kaggle. \
        Kaggle hosts machine learning competitions for data scientists. \
        Google paid a large sum for Kaggle.";

    const ML_TEXT: &str = "Machine learning enables computers to learn patterns from data. \
        Deep learning models improve as they process more training data.";

    const BANK_TEXT: &str = "The central bank raised interest rates to combat rising inflation. \
        Higher interest rates slow economic growth across the housing market.";

    fn corpus() -> [&'static str; 3] {
        [GOOGLE_TEXT, ML_TEXT, BANK_TEXT]
    }

    fn english() -> KeywordEntities {
        KeywordEntities::new().expect("english stopwords are bundled")
    }

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn backend_is_send_sync() {
        assert_send_sync::<KeywordEntities>();
    }

    #[test]
    fn fast_is_strict_subset_of_full() {
        let backend = english();
        for text in corpus() {
            let fast = backend.extract(text, true).expect("fast extract");
            let full = backend.extract(text, false).expect("full extract");
            assert!(!fast.is_empty(), "fast yielded nothing for: {text}");
            assert!(fast.is_subset(&full), "fast not subset of full for: {text}");
            assert!(fast.len() <= FAST_M, "fast over cap for: {text}");
            assert!(full.len() <= FULL_N, "full over cap for: {text}");
            assert!(fast.len() <= full.len());
        }
    }

    #[test]
    fn extraction_is_deterministic() {
        let backend = english();
        let other = english();
        for text in corpus() {
            for fast in [true, false] {
                let first = backend.extract(text, fast).expect("first extract");
                let again = backend.extract(text, fast).expect("repeat extract");
                let fresh = other.extract(text, fast).expect("fresh-instance extract");
                assert_eq!(first, again, "repeat differs for: {text}");
                assert_eq!(first, fresh, "instance differs for: {text}");
            }
        }
    }

    #[test]
    fn entities_are_normalized() {
        let backend = english();
        let full = backend.extract(GOOGLE_TEXT, false).expect("full extract");
        for entity in &full {
            assert_eq!(
                Some(entity.clone()),
                Entity::normalize(entity.as_str()),
                "non-canonical entity {entity:?}"
            );
        }
        let padded = Entity::normalize("  Machine   LEARNING  Competitions ").expect("normalize");
        assert!(
            full.contains(&padded),
            "missing normalized {padded:?} in {full:?}"
        );
    }

    #[test]
    fn blank_text_yields_no_entities() {
        let backend = english();
        assert!(
            backend
                .extract("", false)
                .expect("empty extract")
                .is_empty()
        );
        assert!(
            backend
                .extract("   \t\n ", true)
                .expect("whitespace extract")
                .is_empty()
        );
    }

    #[test]
    fn known_text_yields_expected_entities() {
        let backend = english();
        let fast: Vec<String> = backend
            .extract(GOOGLE_TEXT, true)
            .expect("fast extract")
            .iter()
            .map(|entity| entity.as_str().to_owned())
            .collect();
        assert_eq!(
            fast,
            [
                "data science",
                "data science startup",
                "science startup",
                "science startup kaggle",
                "startup kaggle",
            ]
        );
        let full = backend.extract(GOOGLE_TEXT, false).expect("full extract");
        for term in [
            "google",
            "kaggle",
            "data science",
            "machine learning competitions",
        ] {
            let entity = Entity::normalize(term).expect("normalize");
            assert!(full.contains(&entity), "missing {term:?} in {full:?}");
        }
    }

    #[test]
    fn unknown_language_is_rejected() {
        assert!(matches!(
            KeywordEntities::with_language("zz"),
            Err(Error::EntityExtraction(_))
        ));
    }
}
