//! On-disk `ObjectStorageBackend`: std `fs` with atomic temp-then-rename writes.

use std::fs;
use std::io;
use std::path::PathBuf;
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{Error, Result};
use crate::newtype::{EntryId, Namespace};
use crate::object::ObjectStorageBackend;

const PACKAGE_DIR: &str = "semisweet";

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn obj_err(err: io::Error) -> Error {
    Error::ObjectStorage(Box::new(err))
}

pub struct DiskObjectStore {
    root: PathBuf,
}

impl DiskObjectStore {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn with_default_root() -> Result<Self> {
        let home = std::env::var_os("HOME").ok_or(Error::MissingEnv("HOME"))?;
        let root = PathBuf::from(home).join(".cache").join(PACKAGE_DIR);
        Ok(Self::new(root))
    }

    fn namespace_dir(&self, ns: &Namespace) -> PathBuf {
        self.root.join(ns.as_str())
    }

    fn object_path(&self, ns: &Namespace, id: &EntryId) -> PathBuf {
        self.namespace_dir(ns).join(id.to_string())
    }
}

impl ObjectStorageBackend for DiskObjectStore {
    fn put(&self, ns: &Namespace, id: &EntryId, value: &[u8]) -> Result<()> {
        let dir = self.namespace_dir(ns);
        fs::create_dir_all(&dir).map_err(obj_err)?;

        let pid = process::id();
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let temp = dir.join(format!("{id}.tmp.{pid}.{counter}"));

        fs::write(&temp, value).map_err(obj_err)?;
        fs::rename(&temp, dir.join(id.to_string())).map_err(obj_err)?;
        Ok(())
    }

    fn get(&self, ns: &Namespace, id: &EntryId) -> Result<Option<Vec<u8>>> {
        match fs::read(self.object_path(ns, id)) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(obj_err(err)),
        }
    }

    fn delete(&self, ns: &Namespace, id: &EntryId) -> Result<()> {
        match fs::remove_file(self.object_path(ns, id)) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(obj_err(err)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use tempfile::TempDir;

    use super::*;
    use crate::newtype::{Key, QueryText};

    fn store() -> (TempDir, DiskObjectStore) {
        let tmp = tempfile::tempdir().unwrap();
        let store = DiskObjectStore::new(tmp.path().to_path_buf());
        (tmp, store)
    }

    fn namespace(name: &str) -> Namespace {
        Namespace::new(name.to_owned()).unwrap()
    }

    fn entry_id(query: &str) -> EntryId {
        let q = QueryText::new(query.to_owned()).unwrap();
        EntryId::derive(&q, &BTreeSet::<Key>::new(), &None)
    }

    #[test]
    fn put_then_get_returns_exact_bytes() {
        let (_tmp, store) = store();
        let ns = namespace("prod");
        let id = entry_id("what is the dose");
        let value = b"the answer is 42".to_vec();

        store.put(&ns, &id, &value).unwrap();
        assert_eq!(store.get(&ns, &id).unwrap(), Some(value));
    }

    #[test]
    fn get_absent_returns_none() {
        let (_tmp, store) = store();
        let ns = namespace("prod");
        let id = entry_id("never written");

        assert_eq!(store.get(&ns, &id).unwrap(), None);
    }

    #[test]
    fn overwrite_same_id_replaces_value() {
        let (_tmp, store) = store();
        let ns = namespace("prod");
        let id = entry_id("dose");

        store.put(&ns, &id, b"first").unwrap();
        store.put(&ns, &id, b"second").unwrap();
        assert_eq!(store.get(&ns, &id).unwrap(), Some(b"second".to_vec()));
    }

    #[test]
    fn delete_then_get_returns_none() {
        let (_tmp, store) = store();
        let ns = namespace("prod");
        let id = entry_id("dose");

        store.put(&ns, &id, b"payload").unwrap();
        store.delete(&ns, &id).unwrap();
        assert_eq!(store.get(&ns, &id).unwrap(), None);
    }

    #[test]
    fn delete_absent_is_ok() {
        let (_tmp, store) = store();
        let ns = namespace("prod");
        let id = entry_id("never written");

        assert!(store.delete(&ns, &id).is_ok());
    }

    #[test]
    fn two_namespaces_isolate_the_same_id() {
        let (_tmp, store) = store();
        let alpha = namespace("alpha");
        let beta = namespace("beta");
        let id = entry_id("shared question");

        store.put(&alpha, &id, b"alpha value").unwrap();
        store.put(&beta, &id, b"beta value").unwrap();

        assert_eq!(
            store.get(&alpha, &id).unwrap(),
            Some(b"alpha value".to_vec())
        );
        assert_eq!(store.get(&beta, &id).unwrap(), Some(b"beta value".to_vec()));
    }

    #[test]
    fn large_value_round_trips() {
        let (_tmp, store) = store();
        let ns = namespace("prod");
        let id = entry_id("big");
        let value: Vec<u8> = (0..3 * 1024 * 1024).map(|i| (i % 251) as u8).collect();

        store.put(&ns, &id, &value).unwrap();
        assert_eq!(store.get(&ns, &id).unwrap(), Some(value));
    }

    #[test]
    fn put_leaves_no_temp_sidecar() {
        let (_tmp, store) = store();
        let ns = namespace("prod");
        let id = entry_id("dose");

        store.put(&ns, &id, b"payload").unwrap();

        let entries: Vec<String> = fs::read_dir(store.namespace_dir(&ns))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();

        assert_eq!(entries, vec![id.to_string()]);
        assert!(!entries.iter().any(|name| name.contains(".tmp.")));
    }
}
