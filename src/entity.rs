use std::collections::BTreeSet;
use std::sync::Arc;

use crate::error::Result;
use crate::newtype::Entity;

pub trait EntityBackend: Send + Sync {
    /// `extract(t, fast = true)` MUST be a subset of `extract(t, fast = false)`.
    fn extract(&self, text: &str, fast: bool) -> Result<BTreeSet<Entity>>;
}

impl EntityBackend for Arc<dyn EntityBackend> {
    fn extract(&self, text: &str, fast: bool) -> Result<BTreeSet<Entity>> {
        (**self).extract(text, fast)
    }
}
