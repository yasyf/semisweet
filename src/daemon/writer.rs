//! A bounded write-behind queue for `Set`. The handler enqueues and replies
//! immediately; background threads drain the queue and run `Cache::set`, which owns
//! the (entity-extract -> embed -> object.put -> vector.upsert) ordering. The bound
//! is the backpressure signal: a full queue makes the handler reply `Accepted(false)`
//! rather than grow without limit or block the accept loop.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::thread;

use super::state::PendingWrites;
use crate::newtype::{Context, EntryId, Key, QueryText};
use crate::registry::DynCache;

pub(super) struct WriteJob {
    pub(super) cache: Arc<DynCache>,
    pub(super) pending: Arc<PendingWrites>,
    pub(super) id: EntryId,
    pub(super) query: QueryText,
    pub(super) keys: BTreeSet<Key>,
    pub(super) context: Option<Context>,
    pub(super) value: Arc<[u8]>,
}

impl WriteJob {
    fn run(self) {
        // Run the slow half (entity extract + embed) first, off the pending lock.
        let entry = match self.cache.prepare(&self.query, &self.keys, &self.context) {
            Ok(entry) => entry,
            Err(error) => {
                eprintln!("semisweet-daemon: write failed: {error}");
                return;
            }
        };
        // Embedding done; only now decide whether to commit. Commit only if the shadow
        // still holds exactly this value: a `delete` that landed during the (long) embed
        // cleared it, or a newer `set` replaced it — either way this stale job skips, with
        // no lost update and no resurrected delete. Doing the embed before this check is
        // what shrinks the resurrect-on-delete window from the whole embed to just the
        // upsert below.
        match self.pending.holds(&self.id, &self.value) {
            Ok(true) => {}
            Ok(false) => return,
            Err(error) => {
                eprintln!("semisweet-daemon: pending check failed: {error}");
                return;
            }
        }
        let result = self.cache.commit(&entry, &self.value);
        // Drop the shadow once the entry is durable (or the write failed), but only if it
        // still holds OUR value — a set that raced in during the commit keeps its own
        // shadow for its own job. Done after `commit` so a concurrent `get` never sees a
        // gap where the entry is in neither the buffer nor the stores.
        if let Err(error) = self.pending.remove_if(&self.id, &self.value) {
            eprintln!("semisweet-daemon: pending cleanup failed: {error}");
        }
        if let Err(error) = result {
            eprintln!("semisweet-daemon: write failed: {error}");
        }
    }
}

pub(super) struct WriteQueue {
    sender: crossbeam_channel::Sender<WriteJob>,
    // A non-draining receiver that keeps the channel connected for the queue's whole
    // life, so `try_push` reports real backpressure (a full queue) rather than a
    // disconnect even if every worker thread has exited.
    _keep_alive: crossbeam_channel::Receiver<WriteJob>,
}

impl WriteQueue {
    pub(super) fn new(capacity: usize, workers: usize) -> Self {
        let (sender, receiver) = crossbeam_channel::bounded::<WriteJob>(capacity);
        for _ in 0..workers {
            let receiver = receiver.clone();
            thread::spawn(move || {
                while let Ok(job) = receiver.recv() {
                    job.run();
                }
            });
        }
        Self {
            sender,
            _keep_alive: receiver,
        }
    }

    /// Enqueues a write, returning `false` if the bounded queue is full.
    pub(super) fn try_push(&self, job: WriteJob) -> bool {
        self.sender.try_send(job).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::backends::object_disk::DiskObjectStore;
    use crate::backends::vector_memory::MemoryVectorStore;
    use crate::cache::Cache;
    use crate::embedding::EmbeddingBackend;
    use crate::entity::EntityBackend;
    use crate::error::Result;
    use crate::newtype::{Dim, Embedding, Entity, Namespace};
    use crate::object::ObjectStorageBackend;
    use crate::scoring::ScoringConfig;
    use crate::vector::VectorStorageBackend;

    struct StubEmbedding;

    impl EmbeddingBackend for StubEmbedding {
        fn dim(&self) -> Dim {
            Dim::new(2).unwrap()
        }

        fn embed_query(&self, _text: &str) -> Result<Embedding> {
            Embedding::new(vec![1.0, 0.0])
        }

        fn embed_document(&self, _text: &str) -> Result<Embedding> {
            Embedding::new(vec![1.0, 0.0])
        }
    }

    struct StubEntity;

    impl EntityBackend for StubEntity {
        fn extract(&self, _text: &str, _fast: bool) -> Result<BTreeSet<Entity>> {
            Ok(BTreeSet::new())
        }
    }

    fn stub_cache() -> Arc<DynCache> {
        let root = tempfile::tempdir().unwrap();
        let entity: Arc<dyn EntityBackend> = Arc::new(StubEntity);
        let embedding: Arc<dyn EmbeddingBackend> = Arc::new(StubEmbedding);
        let vector: Arc<dyn VectorStorageBackend> = Arc::new(MemoryVectorStore::new());
        let object: Arc<dyn ObjectStorageBackend> =
            Arc::new(DiskObjectStore::new(root.path().to_path_buf()));
        Arc::new(Cache::new(
            Namespace::new("test".to_owned()).unwrap(),
            entity,
            embedding,
            vector,
            object,
            ScoringConfig::default(),
        ))
    }

    fn job(cache: &Arc<DynCache>) -> WriteJob {
        let query = QueryText::new("q".to_owned()).unwrap();
        let keys = BTreeSet::new();
        let id = EntryId::derive(&query, &keys);
        WriteJob {
            cache: cache.clone(),
            pending: Arc::new(PendingWrites::default()),
            id,
            query,
            keys,
            context: None,
            value: Arc::from(vec![1u8, 2, 3]),
        }
    }

    #[test]
    fn full_queue_reports_backpressure() {
        let cache = stub_cache();
        // No drain workers: the bounded queue fills and stays full.
        let queue = WriteQueue::new(2, 0);
        assert!(queue.try_push(job(&cache)), "first push fits");
        assert!(queue.try_push(job(&cache)), "second push fills the queue");
        assert!(!queue.try_push(job(&cache)), "third push hits the bound");
    }

    #[test]
    fn drained_job_clears_its_pending_shadow() {
        let cache = stub_cache();
        let pending = Arc::new(PendingWrites::default());
        let query = QueryText::new("q".to_owned()).unwrap();
        let keys = BTreeSet::new();
        let id = EntryId::derive(&query, &keys);
        let value: Arc<[u8]> = Arc::from(vec![9u8, 8, 7]);
        pending.insert(id, value.clone()).unwrap();

        let job = WriteJob {
            cache,
            pending: pending.clone(),
            id,
            query,
            keys,
            context: None,
            value,
        };
        job.run();

        assert!(
            pending.get(&id).unwrap().is_none(),
            "a successful write must drop its read-after-write shadow"
        );
    }

    #[test]
    fn deleted_entry_is_not_resurrected() {
        let cache = stub_cache();
        // Empty pending: a delete removed the shadow before this queued job ran.
        let pending = Arc::new(PendingWrites::default());
        let query = QueryText::new("q".to_owned()).unwrap();
        let keys = BTreeSet::new();
        let id = EntryId::derive(&query, &keys);

        let job = WriteJob {
            cache: cache.clone(),
            pending,
            id,
            query: query.clone(),
            keys: keys.clone(),
            context: None,
            value: Arc::from(vec![1u8]),
        };
        job.run();

        let context: Option<Context> = None;
        assert!(
            cache.get(&query, &keys, &context).unwrap().is_none(),
            "a job whose shadow was deleted must not commit to the store"
        );
    }

    #[test]
    fn superseded_write_is_skipped_and_keeps_the_newer_shadow() {
        let cache = stub_cache();
        let pending = Arc::new(PendingWrites::default());
        let query = QueryText::new("q".to_owned()).unwrap();
        let keys = BTreeSet::new();
        let id = EntryId::derive(&query, &keys);
        let v1: Arc<[u8]> = Arc::from(vec![1u8]);
        let v2: Arc<[u8]> = Arc::from(vec![2u8]);

        // A newer set (v2) replaced v1 in the shadow before v1's job ran.
        pending.insert(id, v2.clone()).unwrap();
        let stale = WriteJob {
            cache: cache.clone(),
            pending: pending.clone(),
            id,
            query: query.clone(),
            keys: keys.clone(),
            context: None,
            value: v1,
        };
        stale.run();

        // v1's job committed nothing and left v2's shadow intact for v2's own job.
        let context: Option<Context> = None;
        assert!(cache.get(&query, &keys, &context).unwrap().is_none());
        assert_eq!(pending.get(&id).unwrap().as_deref(), Some(&[2u8][..]));
    }
}
