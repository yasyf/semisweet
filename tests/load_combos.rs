//! Concurrency/load exercise across every backend combo, driven through the real
//! daemon so the read-after-write pending buffer is on the path (turbopuffer is
//! eventually-consistent only *after* the buffer drains, so the property asserted
//! under load is the immediate set->get hit the buffer guarantees).
//!
//! `#[ignore]`: needs the BGE/GLiNER model downloads, and for some combos live keys or
//! a Docker daemon. Run the full matrix with:
//!
//!   VOYAGE_API_KEY=... TURBOPUFFER_API_KEY=... \
//!   cargo test --all-features --test load_combos -- --ignored --nocapture
//!
//! Combos whose credentials are absent print a skip line instead of failing.

use std::path::PathBuf;
use std::sync::Mutex;
use std::thread;

use semisweet::{
    EmbeddingChoice, EntityChoice, Launcher, NamespaceConfig, ObjectChoice, Request, Response,
    ScoringDto, VectorChoice, connect_or_spawn,
};

const BIN: &str = env!("CARGO_BIN_EXE_semisweet-daemon");
const THREADS: usize = 6;
const PER_THREAD: usize = 6;

static ENV_LOCK: Mutex<()> = Mutex::new(());

fn launcher() -> Launcher {
    Launcher::Exe(PathBuf::from(BIN))
}

fn item(namespace: &str, thread: usize, index: usize) -> (String, String, Vec<u8>) {
    let query =
        format!("{namespace} worker {thread} item {index} on distinct subject {thread}-{index}");
    let key = format!("k{thread}-{index}");
    let value = format!("value::{namespace}::{thread}::{index}").into_bytes();
    (query, key, value)
}

fn register(namespace: &str, config: &NamespaceConfig) {
    let mut stub = connect_or_spawn(&launcher()).expect("connect to daemon");
    let config_json = serde_json::to_string(config).unwrap();
    let response = stub
        .request(&Request::RegisterNamespace {
            namespace: namespace.to_owned(),
            config_json,
        })
        .expect("register namespace");
    assert_eq!(
        response,
        Response::Registered,
        "namespace {namespace} should register",
    );
}

/// Hammer one namespace from `THREADS` independent client connections, asserting
/// read-after-write on every set, then delete everything from a fresh connection.
fn hammer(namespace: &str, label: &str) {
    let handles: Vec<_> = (0..THREADS)
        .map(|worker| {
            let namespace = namespace.to_owned();
            let label = label.to_owned();
            thread::spawn(move || {
                let mut stub = connect_or_spawn(&launcher()).expect("worker connect");
                for index in 0..PER_THREAD {
                    let (query, key, value) = item(&namespace, worker, index);
                    let accepted = stub
                        .request(&Request::Set {
                            namespace: namespace.clone(),
                            query: query.clone(),
                            keys: vec![key.clone()],
                            context: None,
                            value: serde_bytes::ByteBuf::from(value.clone()),
                        })
                        .expect("set");
                    assert_eq!(accepted, Response::Accepted(true), "{label}: set accepted");

                    let got = stub
                        .request(&Request::Get {
                            namespace: namespace.clone(),
                            query,
                            keys: vec![key],
                            context: None,
                        })
                        .expect("get");
                    assert_eq!(
                        got,
                        Response::Value(Some(serde_bytes::ByteBuf::from(value))),
                        "{label}: read-after-write must hold for worker {worker} item {index}",
                    );
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().expect("worker thread");
    }

    // Cleanup from one connection: delete removes the pending shadow and makes the
    // matching write-behind job skip, so the entries do not accumulate in any backend.
    let mut stub = connect_or_spawn(&launcher()).expect("cleanup connect");
    for worker in 0..THREADS {
        for index in 0..PER_THREAD {
            let (query, key, _) = item(namespace, worker, index);
            stub.request(&Request::Del {
                namespace: namespace.to_owned(),
                query,
                keys: vec![key],
                context: None,
            })
            .expect("delete");
        }
    }

    println!(
        "[load] {label}: {} read-after-write roundtrips across {THREADS} concurrent connections — OK",
        THREADS * PER_THREAD,
    );
}

fn config(
    embedding: EmbeddingChoice,
    entity: EntityChoice,
    vector: VectorChoice,
    root: &str,
) -> NamespaceConfig {
    NamespaceConfig {
        embedding,
        entity,
        vector,
        object: ObjectChoice::Disk {
            root: Some(root.to_owned()),
        },
        scoring: ScoringDto::default(),
    }
}

fn gliner_labels() -> Vec<String> {
    ["person", "organization", "location"]
        .iter()
        .map(|label| (*label).to_owned())
        .collect()
}

#[test]
#[ignore = "load test: needs model downloads, and live keys / docker for some combos"]
fn all_backend_combos_under_load() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let dir = std::env::temp_dir().join(format!("ssload-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let objects = dir.join("objects");
    let objects = objects.to_str().unwrap().to_owned();

    // Bring up MinIO (if Docker is present) before the daemon spawns, so the daemon
    // inherits the S3 endpoint/credentials it needs to build the S3 object store.
    let minio = start_minio();

    // SAFETY: serialized by ENV_LOCK; no other thread touches the environment here.
    unsafe {
        std::env::set_var("SEMISWEET_SOCKET", dir.join("d.sock"));
        std::env::set_var("SEMISWEET_LOCK", dir.join("d.lock"));
        std::env::set_var("SEMISWEET_LOG", dir.join("d.log"));
        std::env::set_var("SEMISWEET_IDLE_SECS", "120");
        std::env::set_var(
            "SEMISWEET_MODEL_CACHE",
            concat!(env!("CARGO_MANIFEST_DIR"), "/.fastembed_cache"),
        );
    }

    let voyage = std::env::var_os("VOYAGE_API_KEY").is_some();
    let turbopuffer = std::env::var_os("TURBOPUFFER_API_KEY").is_some();

    // Always-available local combos (BGE + keyword/GLiNER auto-download).
    register(
        "load-local-keyword-memory",
        &config(
            EmbeddingChoice::Local { model: None },
            EntityChoice::Keyword { language: None },
            VectorChoice::Memory,
            &objects,
        ),
    );
    hammer(
        "load-local-keyword-memory",
        "local + keyword + memory + disk",
    );

    register(
        "load-local-gliner-memory",
        &config(
            EmbeddingChoice::Local { model: None },
            EntityChoice::Gliner {
                labels: gliner_labels(),
                repo: None,
                model: None,
                tokenizer: None,
            },
            VectorChoice::Memory,
            &objects,
        ),
    );
    hammer("load-local-gliner-memory", "local + gliner + memory + disk");

    if voyage {
        register(
            "load-voyage-keyword-memory",
            &config(
                EmbeddingChoice::Voyage {
                    model: Some("voyage-3.5-lite".to_owned()),
                    dim: Some(512),
                },
                EntityChoice::Keyword { language: None },
                VectorChoice::Memory,
                &objects,
            ),
        );
        hammer(
            "load-voyage-keyword-memory",
            "voyage + keyword + memory + disk",
        );
    } else {
        println!("[load] skip voyage combos: VOYAGE_API_KEY unset");
    }

    if turbopuffer {
        register(
            "load-local-keyword-turbopuffer",
            &config(
                EmbeddingChoice::Local { model: None },
                EntityChoice::Keyword { language: None },
                VectorChoice::Turbopuffer,
                &objects,
            ),
        );
        hammer(
            "load-local-keyword-turbopuffer",
            "local + keyword + turbopuffer + disk",
        );
    } else {
        println!("[load] skip turbopuffer combos: TURBOPUFFER_API_KEY unset");
    }

    if let Some((_node, endpoint, bucket)) = minio.as_ref() {
        register(
            "load-local-keyword-s3",
            &NamespaceConfig {
                embedding: EmbeddingChoice::Local { model: None },
                entity: EntityChoice::Keyword { language: None },
                vector: VectorChoice::Memory,
                object: ObjectChoice::S3 {
                    bucket: Some(bucket.clone()),
                    region: Some("us-east-1".to_owned()),
                    endpoint: Some(endpoint.clone()),
                    prefix: "objects/".to_owned(),
                },
                scoring: ScoringDto::default(),
            },
        );
        hammer(
            "load-local-keyword-s3",
            "local + keyword + memory + s3 (minio)",
        );
    } else {
        println!("[load] skip s3 combo: Docker/MinIO unavailable");
    }

    // `minio` (and its container) is dropped here, at the end of the test.
    let _ = std::fs::remove_dir_all(&dir);
}

type MinioContainer =
    testcontainers_modules::testcontainers::Container<testcontainers_modules::minio::MinIO>;

/// Start a MinIO container and create the load bucket, returning the live container
/// (kept alive by the caller), its endpoint, and the bucket name. `None` if Docker is
/// unavailable, in which case the S3 combo is skipped rather than failed.
fn start_minio() -> Option<(MinioContainer, String, String)> {
    use s3::{Bucket, BucketConfiguration, Region, creds::Credentials};
    use testcontainers_modules::minio::MinIO;
    use testcontainers_modules::testcontainers::runners::SyncRunner;

    let node = MinIO::default().start().ok()?;
    let host = node.get_host().ok()?;
    let port = node.get_host_port_ipv4(9000).ok()?;
    let endpoint = format!("http://{host}:{port}");
    let bucket = "semisweet-load".to_owned();

    // SAFETY: the caller holds ENV_LOCK; these are read by the daemon at spawn time.
    unsafe {
        std::env::set_var("AWS_ACCESS_KEY_ID", "minioadmin");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "minioadmin");
        std::env::set_var("RUST_S3_SKIP_LOCATION_CONSTRAINT", "true");
    }

    let credentials =
        Credentials::new(Some("minioadmin"), Some("minioadmin"), None, None, None).ok()?;
    Bucket::create_with_path_style(
        &bucket,
        Region::Custom {
            region: "us-east-1".to_owned(),
            endpoint: endpoint.clone(),
        },
        credentials,
        BucketConfiguration::default(),
    )
    .ok()?;

    Some((node, endpoint, bucket))
}
