use std::sync::Arc;

use crate::error::Result;
use crate::newtype::{EntryId, Namespace};

pub trait ObjectStorageBackend: Send + Sync {
    fn put(&self, ns: &Namespace, id: &EntryId, value: &[u8]) -> Result<()>;
    fn get(&self, ns: &Namespace, id: &EntryId) -> Result<Option<Vec<u8>>>;
    fn delete(&self, ns: &Namespace, id: &EntryId) -> Result<()>;
}

impl ObjectStorageBackend for Arc<dyn ObjectStorageBackend> {
    fn put(&self, ns: &Namespace, id: &EntryId, value: &[u8]) -> Result<()> {
        (**self).put(ns, id, value)
    }

    fn get(&self, ns: &Namespace, id: &EntryId) -> Result<Option<Vec<u8>>> {
        (**self).get(ns, id)
    }

    fn delete(&self, ns: &Namespace, id: &EntryId) -> Result<()> {
        (**self).delete(ns, id)
    }
}
