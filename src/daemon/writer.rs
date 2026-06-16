//! A bounded write-behind queue for `Set`. The handler enqueues and replies
//! immediately; background threads drain the queue and run `Cache::set`, which owns
//! the (entity-extract -> embed -> object.put -> vector.upsert) ordering. The bound
//! is the backpressure signal: a full queue makes the handler reply `Accepted(false)`
//! rather than grow without limit or block the accept loop.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::thread;

use crate::newtype::{Context, Key, QueryText};
use crate::registry::DynCache;

pub(super) struct WriteJob {
    pub(super) cache: Arc<DynCache>,
    pub(super) query: QueryText,
    pub(super) keys: BTreeSet<Key>,
    pub(super) context: Option<Context>,
    pub(super) value: Vec<u8>,
}

impl WriteJob {
    fn run(self) {
        if let Err(error) = self
            .cache
            .set(&self.query, &self.keys, &self.context, &self.value)
        {
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
        WriteJob {
            cache: cache.clone(),
            query: QueryText::new("q".to_owned()).unwrap(),
            keys: BTreeSet::new(),
            context: None,
            value: vec![1, 2, 3],
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
}
