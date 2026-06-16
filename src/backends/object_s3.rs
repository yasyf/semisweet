//! S3-compatible `ObjectStorageBackend` over rust-s3's truly-sync (attohttpc) path.

use std::fmt;

use s3::Bucket;
use s3::creds::Credentials;
use s3::region::Region;

use crate::error::{Error, Result};
use crate::newtype::{EntryId, Namespace};
use crate::object::ObjectStorageBackend;

fn backend_err<E: std::error::Error + Send + Sync + 'static>(err: E) -> Error {
    Error::ObjectStorage(Box::new(err))
}

fn object_key(prefix: &str, ns: &Namespace, id: &EntryId) -> String {
    format!("{prefix}{}/{id}", ns.as_str())
}

#[derive(Debug)]
struct UnexpectedStatus {
    operation: &'static str,
    key: String,
    status: u16,
}

impl fmt::Display for UnexpectedStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "s3 {} of `{}` returned status {}",
            self.operation, self.key, self.status
        )
    }
}

impl std::error::Error for UnexpectedStatus {}

pub struct S3ObjectStore {
    bucket: Bucket,
    prefix: String,
}

impl S3ObjectStore {
    pub fn new(
        bucket: Option<String>,
        region: Option<String>,
        endpoint: Option<String>,
        prefix: String,
    ) -> Result<Self> {
        // bucket/region/endpoint/creds all resolve here, in the one process that owns the
        // S3 client, so a config is never assembled from two environments.
        let bucket_name = bucket
            .or_else(|| std::env::var("SEMISWEET_S3_BUCKET").ok())
            .ok_or(Error::MissingEnv("SEMISWEET_S3_BUCKET"))?;
        let region = region
            .or_else(|| std::env::var("AWS_REGION").ok())
            .ok_or(Error::MissingEnv("AWS_REGION"))?;
        let access_key = std::env::var("AWS_ACCESS_KEY_ID")
            .map_err(|_| Error::MissingEnv("AWS_ACCESS_KEY_ID"))?;
        let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY")
            .map_err(|_| Error::MissingEnv("AWS_SECRET_ACCESS_KEY"))?;
        let credentials = Credentials::new(Some(&access_key), Some(&secret_key), None, None, None)
            .map_err(backend_err)?;

        let endpoint = endpoint.or_else(|| std::env::var("S3_ENDPOINT").ok());
        let (region, path_style) = match endpoint {
            Some(endpoint) => (Region::Custom { region, endpoint }, true),
            None => (region.parse::<Region>().map_err(backend_err)?, false),
        };

        let mut bucket = *Bucket::new(&bucket_name, region, credentials).map_err(backend_err)?;
        if path_style {
            bucket = *bucket.with_path_style();
        }

        Ok(Self { bucket, prefix })
    }
}

impl ObjectStorageBackend for S3ObjectStore {
    fn put(&self, ns: &Namespace, id: &EntryId, value: &[u8]) -> Result<()> {
        let key = object_key(&self.prefix, ns, id);
        let response = self.bucket.put_object(&key, value).map_err(backend_err)?;
        match response.status_code() {
            200..=299 => Ok(()),
            status => Err(Error::ObjectStorage(Box::new(UnexpectedStatus {
                operation: "put",
                key,
                status,
            }))),
        }
    }

    fn get(&self, ns: &Namespace, id: &EntryId) -> Result<Option<Vec<u8>>> {
        let key = object_key(&self.prefix, ns, id);
        let response = self.bucket.get_object(&key).map_err(backend_err)?;
        match response.status_code() {
            404 => Ok(None),
            200..=299 => Ok(Some(response.to_vec())),
            status => Err(Error::ObjectStorage(Box::new(UnexpectedStatus {
                operation: "get",
                key,
                status,
            }))),
        }
    }

    fn delete(&self, ns: &Namespace, id: &EntryId) -> Result<()> {
        let key = object_key(&self.prefix, ns, id);
        let response = self.bucket.delete_object(&key).map_err(backend_err)?;
        match response.status_code() {
            200..=299 | 404 => Ok(()),
            status => Err(Error::ObjectStorage(Box::new(UnexpectedStatus {
                operation: "delete",
                key,
                status,
            }))),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::newtype::{Key, QueryText};

    fn sample_id() -> EntryId {
        let query = QueryText::new("dose".to_owned()).unwrap();
        let keys: BTreeSet<Key> = [Key::new("patient-7".to_owned()).unwrap()]
            .into_iter()
            .collect();
        EntryId::derive(&query, &keys)
    }

    #[test]
    fn object_key_joins_prefix_namespace_and_id() {
        let ns = Namespace::new("prod".to_owned()).unwrap();
        let id = sample_id();
        let id_hex = id.to_string();

        assert_eq!(id_hex.len(), 32);
        assert!(id_hex.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(
            object_key("objects/", &ns, &id),
            format!("objects/prod/{id_hex}")
        );
        assert_eq!(object_key("", &ns, &id), format!("prod/{id_hex}"));
    }

    #[test]
    fn new_without_credentials_reports_missing_env() {
        unsafe {
            std::env::remove_var("AWS_ACCESS_KEY_ID");
            std::env::remove_var("AWS_SECRET_ACCESS_KEY");
        }

        let result = S3ObjectStore::new(
            Some("bucket".to_owned()),
            Some("us-east-1".to_owned()),
            None,
            "prefix/".to_owned(),
        );
        assert!(matches!(
            result,
            Err(Error::MissingEnv("AWS_ACCESS_KEY_ID"))
        ));
    }

    #[test]
    #[ignore = "requires docker/minio"]
    fn round_trips_objects_against_minio() {
        use s3::BucketConfiguration;
        use testcontainers_modules::minio::MinIO;
        use testcontainers_modules::testcontainers::runners::SyncRunner;

        let node = MinIO::default().start().unwrap();
        let host = node.get_host().unwrap();
        let port = node.get_host_port_ipv4(9000).unwrap();
        let endpoint = format!("http://{host}:{port}");

        unsafe {
            std::env::set_var("AWS_ACCESS_KEY_ID", "minioadmin");
            std::env::set_var("AWS_SECRET_ACCESS_KEY", "minioadmin");
            std::env::set_var("RUST_S3_SKIP_LOCATION_CONSTRAINT", "true");
        }

        let bucket_name = "semisweet-test";
        let region = "us-east-1".to_owned();
        let credentials =
            Credentials::new(Some("minioadmin"), Some("minioadmin"), None, None, None).unwrap();
        Bucket::create_with_path_style(
            bucket_name,
            Region::Custom {
                region: region.clone(),
                endpoint: endpoint.clone(),
            },
            credentials,
            BucketConfiguration::default(),
        )
        .unwrap();

        let store = S3ObjectStore::new(
            Some(bucket_name.to_owned()),
            Some(region),
            Some(endpoint),
            "objects/".to_owned(),
        )
        .unwrap();

        let ns = Namespace::new("prod".to_owned()).unwrap();
        let id = sample_id();
        let payload = b"semisweet payload".to_vec();

        assert_eq!(store.get(&ns, &id).unwrap(), None);
        store.put(&ns, &id, &payload).unwrap();
        assert_eq!(store.get(&ns, &id).unwrap().as_deref(), Some(&payload[..]));
        store.delete(&ns, &id).unwrap();
        assert_eq!(store.get(&ns, &id).unwrap(), None);
        store.delete(&ns, &id).unwrap();
    }
}
