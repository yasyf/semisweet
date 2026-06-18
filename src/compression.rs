//! Transparent value-payload compression at the object-store boundary.
//!
//! [`CompressedStore`] wraps any [`ObjectStorageBackend`], zstd-compressing on `put` and
//! decompressing on `get`. zstd frames are self-describing (they carry their own size and
//! parameters), so `get` always decompresses unambiguously without sidecar metadata.

use std::io;

use crate::error::{Error, Result};
use crate::newtype::{EntryId, Namespace};
use crate::object::ObjectStorageBackend;

// zstd's maximum standard level. Levels 20-22 are "ultra" and need a larger window plus
// more decompression memory; 19 is the max-ratio sweet spot. Writes are async write-behind
// so the extra compression cost is hidden, and zstd decompression speed is level-independent.
const COMPRESSION_LEVEL: i32 = 19;

fn compression_err(err: io::Error) -> Error {
    Error::Compression(Box::new(err))
}

fn compress(value: &[u8]) -> Result<Vec<u8>> {
    zstd::encode_all(value, COMPRESSION_LEVEL).map_err(compression_err)
}

fn decompress(value: &[u8]) -> Result<Vec<u8>> {
    zstd::decode_all(value).map_err(compression_err)
}

pub(crate) struct CompressedStore<O> {
    inner: O,
}

impl<O> CompressedStore<O> {
    pub(crate) fn new(inner: O) -> Self {
        Self { inner }
    }
}

impl<O: ObjectStorageBackend> ObjectStorageBackend for CompressedStore<O> {
    fn put(&self, ns: &Namespace, id: &EntryId, value: &[u8]) -> Result<()> {
        let compressed = compress(value)?;
        self.inner.put(ns, id, &compressed)
    }

    fn get(&self, ns: &Namespace, id: &EntryId) -> Result<Option<Vec<u8>>> {
        match self.inner.get(ns, id)? {
            Some(bytes) => Ok(Some(decompress(&bytes)?)),
            None => Ok(None),
        }
    }

    fn delete(&self, ns: &Namespace, id: &EntryId) -> Result<()> {
        self.inner.delete(ns, id)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap};
    use std::sync::Mutex;

    use super::*;
    use crate::backends::object_disk::DiskObjectStore;
    use crate::newtype::{Key, QueryText};

    // The first four bytes of every zstd frame: magic number 0xFD2FB528, little-endian.
    const ZSTD_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];

    struct RecordingStore {
        objects: Mutex<HashMap<(Namespace, EntryId), Vec<u8>>>,
    }

    impl RecordingStore {
        fn new() -> Self {
            Self {
                objects: Mutex::new(HashMap::new()),
            }
        }

        fn stored(&self, ns: &Namespace, id: &EntryId) -> Option<Vec<u8>> {
            self.objects
                .lock()
                .unwrap()
                .get(&(ns.clone(), *id))
                .cloned()
        }
    }

    impl ObjectStorageBackend for RecordingStore {
        fn put(&self, ns: &Namespace, id: &EntryId, value: &[u8]) -> Result<()> {
            self.objects
                .lock()
                .unwrap()
                .insert((ns.clone(), *id), value.to_vec());
            Ok(())
        }

        fn get(&self, ns: &Namespace, id: &EntryId) -> Result<Option<Vec<u8>>> {
            Ok(self
                .objects
                .lock()
                .unwrap()
                .get(&(ns.clone(), *id))
                .cloned())
        }

        fn delete(&self, ns: &Namespace, id: &EntryId) -> Result<()> {
            self.objects.lock().unwrap().remove(&(ns.clone(), *id));
            Ok(())
        }
    }

    fn namespace() -> Namespace {
        Namespace::new("prod".to_owned()).unwrap()
    }

    fn entry_id(query: &str) -> EntryId {
        let q = QueryText::new(query.to_owned()).unwrap();
        EntryId::derive(&q, &BTreeSet::<Key>::new(), &None)
    }

    fn compressible(len: usize) -> Vec<u8> {
        vec![b'a'; len]
    }

    #[test]
    fn round_trips_empty() {
        assert_eq!(decompress(&compress(b"").unwrap()).unwrap(), b"");
    }

    #[test]
    fn round_trips_small_binary() {
        let value = &[0u8, 255, 1, 128, 64, 0, 0, 7];
        assert_eq!(decompress(&compress(value).unwrap()).unwrap(), value);
    }

    #[test]
    fn round_trips_and_shrinks_compressible() {
        let value = compressible(100_000);
        let compressed = compress(&value).unwrap();
        assert!(
            compressed.len() < 1_000,
            "100k repeated bytes should compress to well under 1k, got {}",
            compressed.len()
        );
        assert_eq!(decompress(&compressed).unwrap(), value);
    }

    #[test]
    fn store_persists_compressed_bytes_and_returns_original() {
        let inner = RecordingStore::new();
        let store = CompressedStore::new(inner);
        let ns = namespace();
        let id = entry_id("what is the dose");
        let value = compressible(50_000);

        store.put(&ns, &id, &value).unwrap();

        // The bytes that reached the inner store are a zstd frame, smaller than the input,
        // and decompress back to it — proving compression happens below the boundary.
        let persisted = store.inner.stored(&ns, &id).unwrap();
        assert_eq!(persisted[..4], ZSTD_MAGIC);
        assert!(persisted.len() < value.len());
        assert_eq!(decompress(&persisted).unwrap(), value);

        // The boundary itself is byte-exact.
        assert_eq!(store.get(&ns, &id).unwrap(), Some(value));
    }

    #[test]
    fn get_absent_returns_none() {
        let store = CompressedStore::new(RecordingStore::new());
        assert_eq!(
            store.get(&namespace(), &entry_id("never written")).unwrap(),
            None
        );
    }

    #[test]
    fn delete_clears_the_entry() {
        let store = CompressedStore::new(RecordingStore::new());
        let ns = namespace();
        let id = entry_id("dose");

        store.put(&ns, &id, b"payload").unwrap();
        store.delete(&ns, &id).unwrap();
        assert_eq!(store.get(&ns, &id).unwrap(), None);
    }

    #[test]
    fn disk_backend_writes_a_zstd_frame_to_disk() {
        let tmp = tempfile::tempdir().unwrap();
        let store = CompressedStore::new(DiskObjectStore::new(tmp.path().to_path_buf()));
        let ns = namespace();
        let id = entry_id("big answer");
        let value = compressible(100_000);

        store.put(&ns, &id, &value).unwrap();

        let on_disk = std::fs::read(tmp.path().join(ns.as_str()).join(id.to_string())).unwrap();
        assert_eq!(on_disk[..4], ZSTD_MAGIC);
        assert!(on_disk.len() < value.len());
        assert_eq!(store.get(&ns, &id).unwrap(), Some(value));
    }
}
