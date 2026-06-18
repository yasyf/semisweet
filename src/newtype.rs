use std::collections::BTreeSet;
use std::fmt;
use std::num::NonZeroUsize;

use uuid::Uuid;

use crate::error::{Error, Result};

const NAMESPACE_UUID: Uuid = Uuid::from_bytes([
    0x9e, 0x6f, 0x4d, 0x2a, 0x1b, 0x73, 0x5c, 0x84, 0xa0, 0x3f, 0xd1, 0x2e, 0x7b, 0x88, 0x46, 0x90,
]);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Namespace(String);

impl Namespace {
    pub fn new(value: String) -> Result<Self> {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(Error::EmptyNamespace);
        }
        if trimmed.contains(['/', '\\', '\0']) {
            return Err(Error::InvalidNamespace(format!(
                "namespace `{trimmed}` must be a single safe path segment"
            )));
        }
        if trimmed == "." || trimmed == ".." {
            return Err(Error::InvalidNamespace(format!(
                "namespace `{trimmed}` must not be a path traversal component"
            )));
        }
        Ok(Self(trimmed.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Namespace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key(String);

impl Key {
    pub fn new(value: String) -> Result<Self> {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(Error::EmptyKey);
        }
        Ok(Self(trimmed.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Entity(String);

impl Entity {
    pub fn new(value: String) -> Result<Self> {
        if value.trim().is_empty() {
            return Err(Error::EmptyEntity);
        }
        Ok(Self(value))
    }

    pub fn normalize(raw: &str) -> Option<Self> {
        let collapsed = raw.split_whitespace().collect::<Vec<_>>().join(" ");
        if collapsed.is_empty() {
            return None;
        }
        Some(Self(collapsed.to_lowercase()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QueryText(String);

impl QueryText {
    pub fn new(value: String) -> Result<Self> {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(Error::EmptyQuery);
        }
        Ok(Self(trimmed.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Context(String);

impl Context {
    pub fn new(value: String) -> Result<Self> {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(Error::EmptyContext);
        }
        Ok(Self(trimmed.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dim(NonZeroUsize);

impl Dim {
    pub fn new(value: usize) -> Result<Self> {
        NonZeroUsize::new(value)
            .map(Self)
            .ok_or(Error::EmptyEmbedding)
    }

    pub fn get(&self) -> usize {
        self.0.get()
    }
}

#[derive(Debug, Clone)]
pub struct Embedding {
    dim: Dim,
    values: Vec<f32>,
}

impl Embedding {
    pub fn new(mut values: Vec<f32>) -> Result<Self> {
        let dim = Dim::new(values.len())?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(Error::NonFiniteEmbedding);
        }
        let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if !norm.is_finite() {
            return Err(Error::NonFiniteEmbedding);
        }
        if norm == 0.0 {
            return Err(Error::ZeroEmbedding);
        }
        for v in &mut values {
            *v /= norm;
        }
        Ok(Self { dim, values })
    }

    pub fn dim(&self) -> Dim {
        self.dim
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn dot(&self, other: &Embedding) -> Result<f32> {
        if self.dim != other.dim {
            return Err(Error::DimMismatch {
                got: other.dim.get(),
                want: self.dim.get(),
            });
        }
        Ok(self
            .values
            .iter()
            .zip(&other.values)
            .map(|(a, b)| a * b)
            .sum())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntryId([u8; 16]);

impl EntryId {
    /// Derives the content-addressed id from the query, optional context, and key set
    /// together. Context is part of the identity: the same query and keys with a
    /// different context derive a distinct id, so context variants are stored as
    /// separate entries instead of overwriting one another. The context is injected
    /// between the query and the keys with a one-byte presence tag, and every field is
    /// length-prefixed, so no `(query, context, keys)` triple can collide with another.
    pub fn derive(query: &QueryText, keys: &BTreeSet<Key>, context: &Option<Context>) -> Self {
        let mut buf: Vec<u8> = Vec::new();
        Self::push_field(&mut buf, query.as_str().as_bytes());
        match context {
            Some(context) => {
                buf.push(1);
                Self::push_field(&mut buf, context.as_str().as_bytes());
            }
            None => buf.push(0),
        }
        for key in keys {
            Self::push_field(&mut buf, key.as_str().as_bytes());
        }
        let uuid = Uuid::new_v5(&NAMESPACE_UUID, &buf);
        Self(*uuid.as_bytes())
    }

    pub fn from_hex(hex: &str) -> Option<Self> {
        if hex.len() != 32 {
            return None;
        }
        let mut bytes = [0u8; 16];
        for (slot, pair) in bytes.iter_mut().zip(hex.as_bytes().chunks_exact(2)) {
            let pair = std::str::from_utf8(pair).ok()?;
            *slot = u8::from_str_radix(pair, 16).ok()?;
        }
        Some(Self(bytes))
    }

    fn push_field(buf: &mut Vec<u8>, field: &[u8]) {
        buf.extend_from_slice(&(field.len() as u64).to_le_bytes());
        buf.extend_from_slice(field);
    }
}

impl fmt::Display for EntryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespace_rejects_blank_and_unsafe_segments() {
        assert!(matches!(
            Namespace::new("   ".to_owned()),
            Err(Error::EmptyNamespace)
        ));
        assert!(matches!(
            Namespace::new(String::new()),
            Err(Error::EmptyNamespace)
        ));
        assert!(matches!(
            Namespace::new("a/b".to_owned()),
            Err(Error::InvalidNamespace(_))
        ));
        assert!(matches!(
            Namespace::new("../etc".to_owned()),
            Err(Error::InvalidNamespace(_))
        ));
        assert!(matches!(
            Namespace::new("a\u{0}b".to_owned()),
            Err(Error::InvalidNamespace(_))
        ));
        assert!(matches!(
            Namespace::new("..".to_owned()),
            Err(Error::InvalidNamespace(_))
        ));
        assert_eq!(Namespace::new("prod".to_owned()).unwrap().as_str(), "prod");
    }

    #[test]
    fn query_rejects_blank() {
        assert!(QueryText::new("  ".to_owned()).is_err());
        assert_eq!(
            QueryText::new("  hello  ".to_owned()).unwrap().as_str(),
            "hello"
        );
    }

    #[test]
    fn embedding_normalizes_to_unit_length() {
        let emb = Embedding::new(vec![3.0, 4.0]).unwrap();
        let norm = emb.values().iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert_eq!(emb.dim().get(), 2);
    }

    #[test]
    fn embedding_rejects_empty_zero_and_non_finite() {
        assert!(matches!(Embedding::new(vec![]), Err(Error::EmptyEmbedding)));
        assert!(matches!(
            Embedding::new(vec![0.0, 0.0]),
            Err(Error::ZeroEmbedding)
        ));
        assert!(matches!(
            Embedding::new(vec![f32::NAN, 1.0]),
            Err(Error::NonFiniteEmbedding)
        ));
        assert!(matches!(
            Embedding::new(vec![f32::INFINITY, 0.0]),
            Err(Error::NonFiniteEmbedding)
        ));
    }

    #[test]
    fn entity_normalize_collapses_and_lowercases() {
        let normalized = Entity::normalize("  Aspirin  Tablet ").unwrap();
        let expected = Entity::new("aspirin tablet".to_owned()).unwrap();
        assert_eq!(normalized, expected);
        assert!(Entity::normalize("   ").is_none());
    }

    #[test]
    fn entry_id_is_deterministic_and_order_independent() {
        let query = QueryText::new("what is the dose".to_owned()).unwrap();
        let a = Key::new("alpha".to_owned()).unwrap();
        let b = Key::new("beta".to_owned()).unwrap();

        let forward: BTreeSet<Key> = [a.clone(), b.clone()].into_iter().collect();
        let reverse: BTreeSet<Key> = [b, a].into_iter().collect();

        let id1 = EntryId::derive(&query, &forward, &None);
        let id2 = EntryId::derive(&query, &reverse, &None);
        assert_eq!(id1, id2);
        assert_eq!(id1.to_string(), id2.to_string());
    }

    #[test]
    fn entry_id_order_independent_with_context_present() {
        let query = QueryText::new("what is the dose".to_owned()).unwrap();
        let context = Some(Context::new("oncology".to_owned()).unwrap());
        let a = Key::new("alpha".to_owned()).unwrap();
        let b = Key::new("beta".to_owned()).unwrap();

        let forward: BTreeSet<Key> = [a.clone(), b.clone()].into_iter().collect();
        let reverse: BTreeSet<Key> = [b, a].into_iter().collect();

        assert_eq!(
            EntryId::derive(&query, &forward, &context),
            EntryId::derive(&query, &reverse, &context)
        );
    }

    #[test]
    fn context_distinguishes_entries() {
        let query = QueryText::new("what is the dose".to_owned()).unwrap();
        let keys: BTreeSet<Key> = [Key::new("patient-7".to_owned()).unwrap()]
            .into_iter()
            .collect();
        let context_a = Some(Context::new("a".to_owned()).unwrap());
        let context_b = Some(Context::new("b".to_owned()).unwrap());

        let none = EntryId::derive(&query, &keys, &None);
        let a = EntryId::derive(&query, &keys, &context_a);
        let b = EntryId::derive(&query, &keys, &context_b);

        assert_ne!(none, a);
        assert_ne!(none, b);
        assert_ne!(a, b);
    }

    #[test]
    fn context_with_same_value_derives_equal_ids() {
        let query = QueryText::new("what is the dose".to_owned()).unwrap();
        let keys: BTreeSet<Key> = [Key::new("patient-7".to_owned()).unwrap()]
            .into_iter()
            .collect();
        let context = Some(Context::new("oncology".to_owned()).unwrap());

        assert_eq!(
            EntryId::derive(&query, &keys, &context),
            EntryId::derive(&query, &keys, &context)
        );
    }

    #[test]
    fn context_not_confused_with_key() {
        let query = QueryText::new("dose".to_owned()).unwrap();
        let k1 = Key::new("k1".to_owned()).unwrap();
        let k2 = Key::new("k2".to_owned()).unwrap();

        let two_keys: BTreeSet<Key> = [k1.clone(), k2.clone()].into_iter().collect();
        let one_key: BTreeSet<Key> = [k1].into_iter().collect();
        let context = Some(Context::new("k2".to_owned()).unwrap());

        assert_ne!(
            EntryId::derive(&query, &two_keys, &None),
            EntryId::derive(&query, &one_key, &context)
        );
    }

    #[test]
    fn entry_id_resists_delimiter_injection() {
        let query = QueryText::new("dose".to_owned()).unwrap();

        let split_first: BTreeSet<Key> = [
            Key::new("a".to_owned()).unwrap(),
            Key::new("b\u{0}c".to_owned()).unwrap(),
        ]
        .into_iter()
        .collect();
        let split_second: BTreeSet<Key> = [
            Key::new("a\u{0}b".to_owned()).unwrap(),
            Key::new("c".to_owned()).unwrap(),
        ]
        .into_iter()
        .collect();

        assert_ne!(
            EntryId::derive(&query, &split_first, &None),
            EntryId::derive(&query, &split_second, &None)
        );
    }

    #[test]
    fn dot_of_orthogonal_is_zero_and_identical_is_one() {
        let x = Embedding::new(vec![1.0, 0.0]).unwrap();
        let y = Embedding::new(vec![0.0, 1.0]).unwrap();
        assert!(x.dot(&y).unwrap().abs() < 1e-6);
        assert!((x.dot(&x).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dot_rejects_dim_mismatch() {
        let x = Embedding::new(vec![1.0, 0.0]).unwrap();
        let y = Embedding::new(vec![1.0, 0.0, 0.0]).unwrap();
        assert!(matches!(
            x.dot(&y),
            Err(Error::DimMismatch { got: 3, want: 2 })
        ));
    }
}
