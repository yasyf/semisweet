//! Shared daemon state threaded into every connection task: the namespace registry,
//! the blocking inference pool (gets/dels + namespace builds), and the bounded
//! write-behind queue (sets).

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::inference::InferencePool;
use super::writer::WriteQueue;
use crate::error::{Error, Result};
use crate::registry::DynCache;

const WRITE_QUEUE_CAPACITY: usize = 1024;
const WRITE_WORKERS: usize = 2;

pub(super) struct NamespaceEntry {
    pub(super) cache: Arc<DynCache>,
}

pub(super) struct DaemonState {
    registry: RwLock<HashMap<String, Arc<NamespaceEntry>>>,
    pub(super) inference: InferencePool,
    pub(super) writer: WriteQueue,
}

impl DaemonState {
    pub(super) fn new() -> Self {
        Self {
            registry: RwLock::new(HashMap::new()),
            inference: InferencePool::new(),
            writer: WriteQueue::new(WRITE_QUEUE_CAPACITY, WRITE_WORKERS),
        }
    }

    pub(super) fn resolve(&self, namespace: &str) -> Result<Option<Arc<NamespaceEntry>>> {
        let registry = self
            .registry
            .read()
            .map_err(|_| Error::Daemon("registry lock poisoned".to_owned()))?;
        Ok(registry.get(namespace).cloned())
    }

    pub(super) fn register(&self, namespace: String, entry: Arc<NamespaceEntry>) -> Result<()> {
        let mut registry = self
            .registry
            .write()
            .map_err(|_| Error::Daemon("registry lock poisoned".to_owned()))?;
        registry.insert(namespace, entry);
        Ok(())
    }
}
