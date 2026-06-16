//! The seam between the Phase 2 backends and the daemon: a serde config describing
//! one namespace's backend choices, and `build_cache`, which turns that config into a
//! ready `DynCache`. Phase 4's Python builder emits the JSON carried in
//! `RegisterNamespace.config_json`; a backend whose Cargo feature is not compiled is
//! reported as `UnknownBackend` rather than silently substituted.

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::cache::Cache;
use crate::embedding::EmbeddingBackend;
use crate::entity::EntityBackend;
use crate::error::{Error, Result};
use crate::newtype::Namespace;
use crate::object::ObjectStorageBackend;
use crate::scoring::{ContextMode, ScoringConfig};
use crate::vector::VectorStorageBackend;

pub type DynCache = Cache<
    Arc<dyn EntityBackend>,
    Arc<dyn EmbeddingBackend>,
    Arc<dyn VectorStorageBackend>,
    Arc<dyn ObjectStorageBackend>,
>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NamespaceConfig {
    pub embedding: EmbeddingChoice,
    pub entity: EntityChoice,
    pub vector: VectorChoice,
    pub object: ObjectChoice,
    #[serde(default)]
    pub scoring: ScoringDto,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum EmbeddingChoice {
    #[serde(rename = "voyage")]
    Voyage { model: String, dim: usize },
    #[serde(rename = "local")]
    Local,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum EntityChoice {
    #[serde(rename = "keyword")]
    Keyword { language: Option<String> },
    #[serde(rename = "gliner")]
    Gliner { labels: Vec<String> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum VectorChoice {
    #[serde(rename = "memory")]
    Memory,
    #[serde(rename = "turbopuffer")]
    Turbopuffer,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum ObjectChoice {
    #[serde(rename = "disk")]
    Disk { root: Option<String> },
    #[serde(rename = "s3")]
    S3 {
        bucket: String,
        region: String,
        endpoint: Option<String>,
        prefix: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoringDto {
    pub base_threshold: f32,
    pub floor_threshold: f32,
    pub entity_bonus_weight: f32,
    pub top_k: usize,
    pub entity_filter: bool,
    pub context: String,
}

impl Default for ScoringDto {
    fn default() -> Self {
        let defaults = ScoringConfig::default();
        Self {
            base_threshold: defaults.base_threshold,
            floor_threshold: defaults.floor_threshold,
            entity_bonus_weight: defaults.entity_bonus_weight,
            top_k: defaults.top_k.get(),
            entity_filter: defaults.entity_filter,
            context: context_name(defaults.context).to_owned(),
        }
    }
}

fn context_name(mode: ContextMode) -> &'static str {
    match mode {
        ContextMode::Ignore => "ignore",
        ContextMode::Tiebreak => "tiebreak",
    }
}

impl ScoringDto {
    fn to_config(&self) -> Result<ScoringConfig> {
        if self.floor_threshold > self.base_threshold {
            return Err(Error::InvalidConfig(format!(
                "floor_threshold {} exceeds base_threshold {}",
                self.floor_threshold, self.base_threshold
            )));
        }
        let top_k = NonZeroUsize::new(self.top_k)
            .ok_or_else(|| Error::InvalidConfig("top_k must be greater than zero".to_owned()))?;
        let context = match self.context.as_str() {
            "ignore" => ContextMode::Ignore,
            "tiebreak" => ContextMode::Tiebreak,
            other => {
                return Err(Error::InvalidConfig(format!(
                    "unknown context mode `{other}`"
                )));
            }
        };
        Ok(ScoringConfig {
            base_threshold: self.base_threshold,
            floor_threshold: self.floor_threshold,
            entity_bonus_weight: self.entity_bonus_weight,
            top_k,
            entity_filter: self.entity_filter,
            context,
        })
    }
}

pub fn build_cache(namespace: &str, config: &NamespaceConfig) -> Result<DynCache> {
    let namespace = Namespace::new(namespace.to_owned())?;
    let scoring = config.scoring.to_config()?;
    let entity = build_entity(&config.entity)?;
    let embedding = build_embedding(&config.embedding)?;
    let vector = build_vector(&config.vector)?;
    let object = build_object(&config.object)?;
    Ok(Cache::new(
        namespace, entity, embedding, vector, object, scoring,
    ))
}

fn build_entity(choice: &EntityChoice) -> Result<Arc<dyn EntityBackend>> {
    match choice {
        EntityChoice::Keyword { language } => build_keyword(language.as_deref()),
        EntityChoice::Gliner { labels } => build_gliner(labels),
    }
}

#[cfg(feature = "keyword")]
fn build_keyword(language: Option<&str>) -> Result<Arc<dyn EntityBackend>> {
    use crate::backends::entity_keyword::KeywordEntities;
    let backend = match language {
        Some(language) => KeywordEntities::with_language(language)?,
        None => KeywordEntities::new()?,
    };
    Ok(Arc::new(backend))
}

#[cfg(not(feature = "keyword"))]
fn build_keyword(_language: Option<&str>) -> Result<Arc<dyn EntityBackend>> {
    Err(Error::UnknownBackend("keyword".to_owned()))
}

#[cfg(feature = "gliner")]
fn build_gliner(labels: &[String]) -> Result<Arc<dyn EntityBackend>> {
    use crate::backends::entity_gliner::GlinerEntities;
    let backend = GlinerEntities::new(labels.to_vec())?;
    Ok(Arc::new(backend))
}

#[cfg(not(feature = "gliner"))]
fn build_gliner(_labels: &[String]) -> Result<Arc<dyn EntityBackend>> {
    Err(Error::UnknownBackend("gliner".to_owned()))
}

fn build_embedding(choice: &EmbeddingChoice) -> Result<Arc<dyn EmbeddingBackend>> {
    match choice {
        EmbeddingChoice::Voyage { model, dim } => build_voyage(model, *dim),
        EmbeddingChoice::Local => build_local(),
    }
}

#[cfg(feature = "voyage")]
fn build_voyage(model: &str, dim: usize) -> Result<Arc<dyn EmbeddingBackend>> {
    use crate::backends::embed_voyage::VoyageEmbedding;
    let dim = NonZeroUsize::new(dim).ok_or_else(|| {
        Error::InvalidConfig("embedding dim must be greater than zero".to_owned())
    })?;
    let backend = VoyageEmbedding::new(model.to_owned(), dim)?;
    Ok(Arc::new(backend))
}

#[cfg(not(feature = "voyage"))]
fn build_voyage(_model: &str, _dim: usize) -> Result<Arc<dyn EmbeddingBackend>> {
    Err(Error::UnknownBackend("voyage".to_owned()))
}

#[cfg(feature = "local-embed")]
fn build_local() -> Result<Arc<dyn EmbeddingBackend>> {
    use crate::backends::embed_local::LocalEmbedding;
    Ok(Arc::new(LocalEmbedding::new()?))
}

#[cfg(not(feature = "local-embed"))]
fn build_local() -> Result<Arc<dyn EmbeddingBackend>> {
    Err(Error::UnknownBackend("local".to_owned()))
}

fn build_vector(choice: &VectorChoice) -> Result<Arc<dyn VectorStorageBackend>> {
    match choice {
        VectorChoice::Memory => {
            use crate::backends::vector_memory::MemoryVectorStore;
            Ok(Arc::new(MemoryVectorStore::new()))
        }
        VectorChoice::Turbopuffer => build_turbopuffer(),
    }
}

#[cfg(feature = "turbopuffer")]
fn build_turbopuffer() -> Result<Arc<dyn VectorStorageBackend>> {
    use crate::backends::vector_turbopuffer::TurbopufferVectorStore;
    Ok(Arc::new(TurbopufferVectorStore::new()?))
}

#[cfg(not(feature = "turbopuffer"))]
fn build_turbopuffer() -> Result<Arc<dyn VectorStorageBackend>> {
    Err(Error::UnknownBackend("turbopuffer".to_owned()))
}

fn build_object(choice: &ObjectChoice) -> Result<Arc<dyn ObjectStorageBackend>> {
    match choice {
        ObjectChoice::Disk { root } => {
            use crate::backends::object_disk::DiskObjectStore;
            let store = match root {
                Some(root) => DiskObjectStore::new(PathBuf::from(root)),
                None => DiskObjectStore::with_default_root()?,
            };
            Ok(Arc::new(store))
        }
        ObjectChoice::S3 {
            bucket,
            region,
            endpoint,
            prefix,
        } => build_s3(bucket, region, endpoint.clone(), prefix),
    }
}

#[cfg(feature = "s3")]
fn build_s3(
    bucket: &str,
    region: &str,
    endpoint: Option<String>,
    prefix: &str,
) -> Result<Arc<dyn ObjectStorageBackend>> {
    use crate::backends::object_s3::S3ObjectStore;
    let store = S3ObjectStore::new(
        bucket.to_owned(),
        region.to_owned(),
        endpoint,
        prefix.to_owned(),
    )?;
    Ok(Arc::new(store))
}

#[cfg(not(feature = "s3"))]
fn build_s3(
    _bucket: &str,
    _region: &str,
    _endpoint: Option<String>,
    _prefix: &str,
) -> Result<Arc<dyn ObjectStorageBackend>> {
    Err(Error::UnknownBackend("s3".to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn offline_config(root: &str) -> NamespaceConfig {
        NamespaceConfig {
            embedding: EmbeddingChoice::Voyage {
                model: "voyage-3.5-lite".to_owned(),
                dim: 512,
            },
            entity: EntityChoice::Keyword { language: None },
            vector: VectorChoice::Memory,
            object: ObjectChoice::Disk {
                root: Some(root.to_owned()),
            },
            scoring: ScoringDto::default(),
        }
    }

    #[test]
    fn namespace_config_round_trips_through_json() {
        let config = NamespaceConfig {
            embedding: EmbeddingChoice::Voyage {
                model: "voyage-3".to_owned(),
                dim: 1024,
            },
            entity: EntityChoice::Gliner {
                labels: vec!["drug".to_owned(), "dose".to_owned()],
            },
            vector: VectorChoice::Turbopuffer,
            object: ObjectChoice::S3 {
                bucket: "cache".to_owned(),
                region: "us-east-1".to_owned(),
                endpoint: None,
                prefix: "ns/".to_owned(),
            },
            scoring: ScoringDto::default(),
        };
        let json = serde_json::to_string(&config).unwrap();
        let back: NamespaceConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, back);
    }

    #[test]
    fn scoring_omitted_in_json_falls_back_to_default() {
        let json = r#"{
            "embedding": {"kind": "local"},
            "entity": {"kind": "keyword", "language": null},
            "vector": {"kind": "memory"},
            "object": {"kind": "disk", "root": null}
        }"#;
        let config: NamespaceConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.scoring, ScoringDto::default());
    }

    #[test]
    fn choice_tags_are_stable() {
        assert_eq!(
            serde_json::to_string(&EmbeddingChoice::Local).unwrap(),
            r#"{"kind":"local"}"#
        );
        assert_eq!(
            serde_json::to_string(&VectorChoice::Turbopuffer).unwrap(),
            r#"{"kind":"turbopuffer"}"#
        );
        let s3 = ObjectChoice::S3 {
            bucket: "b".to_owned(),
            region: "r".to_owned(),
            endpoint: None,
            prefix: "p".to_owned(),
        };
        assert_eq!(
            serde_json::to_string(&s3).unwrap(),
            r#"{"kind":"s3","bucket":"b","region":"r","endpoint":null,"prefix":"p"}"#
        );
    }

    #[test]
    fn build_cache_assembles_offline_backends() {
        let _guard = crate::backends::ENV_LOCK.lock().unwrap();
        let root = tempfile::tempdir().unwrap();
        // SAFETY: serialized by ENV_LOCK against the other env-touching tests.
        unsafe {
            std::env::set_var("VOYAGE_API_KEY", "test-key");
        }
        let config = offline_config(root.path().to_str().unwrap());
        let result = build_cache("clinical", &config);
        // SAFETY: serialized by ENV_LOCK against the other env-touching tests.
        unsafe {
            std::env::remove_var("VOYAGE_API_KEY");
        }
        assert!(
            result.is_ok(),
            "offline build_cache should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn build_cache_rejects_floor_above_base() {
        let mut config = offline_config("/tmp/unused");
        config.scoring.base_threshold = 0.90;
        config.scoring.floor_threshold = 0.95;
        assert!(matches!(
            build_cache("clinical", &config),
            Err(Error::InvalidConfig(_))
        ));
    }

    #[test]
    fn build_cache_rejects_feature_off_backend() {
        let mut config = offline_config("/tmp/unused");
        config.entity = EntityChoice::Gliner {
            labels: vec!["drug".to_owned()],
        };
        assert!(matches!(
            build_cache("clinical", &config),
            Err(Error::UnknownBackend(name)) if name == "gliner"
        ));
    }
}
