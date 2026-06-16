//! Shared daemon state threaded into every connection task: the namespace registry,
//! the blocking inference pool (gets/dels + namespace builds), and the bounded
//! write-behind queue (sets).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, RwLock};

use super::inference::InferencePool;
use super::writer::WriteQueue;
use crate::error::{Error, Result};
use crate::newtype::EntryId;
use crate::registry::DynCache;

const WRITE_QUEUE_CAPACITY: usize = 1024;
const WRITE_WORKERS: usize = 2;

/// An in-memory exact-match shadow of the writes still draining through the
/// write-behind queue. A `set` records its value here keyed by entry id; a `get` for
/// the same `(query, keys)` reads it back immediately, giving read-after-write
/// consistency with no polling and no clock. The writer drops each shadow once
/// `Cache::set` has made the entry durable, so the buffer only ever holds in-flight
/// writes.
#[derive(Default)]
pub(super) struct PendingWrites {
    entries: Mutex<HashMap<EntryId, Arc<[u8]>>>,
}

impl PendingWrites {
    pub(super) fn insert(&self, id: EntryId, value: Arc<[u8]>) -> Result<()> {
        self.lock()?.insert(id, value);
        Ok(())
    }

    pub(super) fn get(&self, id: &EntryId) -> Result<Option<Arc<[u8]>>> {
        Ok(self.lock()?.get(id).cloned())
    }

    /// Whether the shadow still holds exactly `value` (same allocation). A newer `set`
    /// replaces the slot with a different `Arc` and a `delete` clears it, so a stale
    /// write-behind job uses this to recognise that it has been superseded or deleted.
    pub(super) fn holds(&self, id: &EntryId, value: &Arc<[u8]>) -> Result<bool> {
        Ok(self
            .lock()?
            .get(id)
            .is_some_and(|current| Arc::ptr_eq(current, value)))
    }

    /// Unconditionally drops the shadow, returning whether it was present — i.e.
    /// whether a write was still in flight for this entry. Used by `delete`.
    pub(super) fn remove(&self, id: &EntryId) -> Result<bool> {
        Ok(self.lock()?.remove(id).is_some())
    }

    /// Drops the shadow only if it still holds exactly `value`, so a committing job
    /// never clears a newer set's shadow. Returns whether it removed anything.
    pub(super) fn remove_if(&self, id: &EntryId, value: &Arc<[u8]>) -> Result<bool> {
        let mut entries = self.lock()?;
        if entries
            .get(id)
            .is_some_and(|current| Arc::ptr_eq(current, value))
        {
            entries.remove(id);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn lock(&self) -> Result<MutexGuard<'_, HashMap<EntryId, Arc<[u8]>>>> {
        self.entries
            .lock()
            .map_err(|_| Error::Daemon("pending-writes lock poisoned".to_owned()))
    }
}

pub(super) struct NamespaceEntry {
    pub(super) cache: Arc<DynCache>,
    pub(super) pending: Arc<PendingWrites>,
}

pub(super) struct DaemonState {
    registry: RwLock<HashMap<String, Arc<NamespaceEntry>>>,
    // Serializes registration per namespace name so a namespace's (expensive) cache build
    // runs at most once: concurrent registrations of the same name queue on its lock, and
    // all but the first observe the already-registered entry and skip the build.
    build_locks: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    pub(super) inference: InferencePool,
    pub(super) writer: WriteQueue,
}

impl DaemonState {
    pub(super) fn new() -> Self {
        Self {
            registry: RwLock::new(HashMap::new()),
            build_locks: Mutex::new(HashMap::new()),
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

    /// The per-namespace build lock, created on first request and stable thereafter. A
    /// registration holds this across the resolve-build-register sequence so a second
    /// registration of the same namespace waits, then resolves the first one's entry
    /// instead of rebuilding.
    pub(super) fn build_lock(&self, namespace: &str) -> Result<Arc<tokio::sync::Mutex<()>>> {
        let mut build_locks = self
            .build_locks
            .lock()
            .map_err(|_| Error::Daemon("build-locks lock poisoned".to_owned()))?;
        Ok(build_locks
            .entry(namespace.to_owned())
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .clone())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::newtype::QueryText;

    fn id(query: &str) -> EntryId {
        EntryId::derive(&QueryText::new(query.to_owned()).unwrap(), &BTreeSet::new())
    }

    #[test]
    fn pending_shadow_round_trips_until_removed() {
        let pending = PendingWrites::default();
        let key = id("what is the capital of france");
        let value: Arc<[u8]> = Arc::from(vec![1u8, 2, 3]);

        assert!(pending.get(&key).unwrap().is_none());
        assert!(!pending.holds(&key, &value).unwrap());

        pending.insert(key, value.clone()).unwrap();
        assert!(pending.holds(&key, &value).unwrap());
        assert_eq!(
            pending.get(&key).unwrap().as_deref(),
            Some(&[1u8, 2, 3][..])
        );

        assert!(
            pending.remove(&key).unwrap(),
            "remove reports the shadow was present"
        );
        assert!(pending.get(&key).unwrap().is_none());
        assert!(
            !pending.remove(&key).unwrap(),
            "removing an absent shadow reports false"
        );
    }

    #[test]
    fn remove_if_only_clears_the_matching_value() {
        let pending = PendingWrites::default();
        let key = id("q");
        let v1: Arc<[u8]> = Arc::from(vec![1u8]);
        let v2: Arc<[u8]> = Arc::from(vec![2u8]);

        pending.insert(key, v1.clone()).unwrap();
        // A newer set replaces the slot; the superseded value no longer holds.
        pending.insert(key, v2.clone()).unwrap();
        assert!(!pending.holds(&key, &v1).unwrap());
        assert!(pending.holds(&key, &v2).unwrap());

        // A stale job for v1 must not clear v2's shadow.
        assert!(!pending.remove_if(&key, &v1).unwrap());
        assert_eq!(pending.get(&key).unwrap().as_deref(), Some(&[2u8][..]));

        // The current value clears itself.
        assert!(pending.remove_if(&key, &v2).unwrap());
        assert!(pending.get(&key).unwrap().is_none());
    }

    #[test]
    fn build_lock_is_stable_per_namespace() {
        let state = DaemonState::new();
        let alpha = state.build_lock("alpha").unwrap();
        let alpha_again = state.build_lock("alpha").unwrap();
        let beta = state.build_lock("beta").unwrap();

        assert!(
            Arc::ptr_eq(&alpha, &alpha_again),
            "the same namespace reuses one build lock so its cache builds at most once"
        );
        assert!(
            !Arc::ptr_eq(&alpha, &beta),
            "distinct namespaces get independent build locks"
        );
    }

    #[test]
    fn pending_shadow_isolates_distinct_entries() {
        let pending = PendingWrites::default();
        let a = id("alpha");
        let b = id("beta");
        pending.insert(a, Arc::from(vec![0xAA])).unwrap();
        pending.insert(b, Arc::from(vec![0xBB])).unwrap();

        assert_eq!(pending.get(&a).unwrap().as_deref(), Some(&[0xAA][..]));
        pending.remove(&a).unwrap();
        assert!(pending.get(&a).unwrap().is_none());
        assert_eq!(pending.get(&b).unwrap().as_deref(), Some(&[0xBB][..]));
    }
}
