//! Real cache-serving integration test: the orphan daemon is spawned via the
//! `semisweet-daemon` binary (built with `--features local-embed` so it carries a real
//! embedding) and driven through the public `ClientStub`. Each test gets a unique
//! runtime dir via the env overrides and a generous idle so the model load never trips
//! the idle shutdown. Marked `#[ignore]` because it needs the BGE model on disk.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, MutexGuard};

use semisweet::{
    ClientStub, EmbeddingChoice, EntityChoice, Launcher, NamespaceConfig, ObjectChoice, Request,
    Response, ScoringDto, VectorChoice, connect_or_spawn,
};

const BIN: &str = env!("CARGO_BIN_EXE_semisweet-daemon");
const NAMESPACE: &str = "serving";

static ENV_LOCK: Mutex<()> = Mutex::new(());
static COUNTER: AtomicU32 = AtomicU32::new(0);

struct TestEnv {
    dir: PathBuf,
    pid: PathBuf,
    _guard: MutexGuard<'static, ()>,
}

impl TestEnv {
    fn new(idle_secs: u64) -> Self {
        let guard = ENV_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("ssserve{}-{}", std::process::id(), n));
        std::fs::create_dir_all(&dir).unwrap();
        // SAFETY: the tests are serialized by ENV_LOCK, so no other thread reads or
        // writes the environment while these overrides are installed.
        unsafe {
            std::env::set_var("SEMISWEET_SOCKET", dir.join("d.sock"));
            std::env::set_var("SEMISWEET_LOCK", dir.join("d.lock"));
            std::env::set_var("SEMISWEET_LOG", dir.join("d.log"));
            std::env::set_var("SEMISWEET_IDLE_SECS", idle_secs.to_string());
            // The daemon runs with cwd=/, so point its model cache at the absolute
            // path where this crate's tests already downloaded BGE.
            std::env::set_var(
                "SEMISWEET_MODEL_CACHE",
                concat!(env!("CARGO_MANIFEST_DIR"), "/.fastembed_cache"),
            );
        }
        Self {
            pid: dir.join("d.pid"),
            dir,
            _guard: guard,
        }
    }

    fn launcher(&self) -> Launcher {
        Launcher::Exe(PathBuf::from(BIN))
    }
}

impl Drop for TestEnv {
    fn drop(&mut self) {
        if let Ok(raw) = std::fs::read_to_string(&self.pid)
            && let Ok(pid) = raw.trim().parse::<u32>()
        {
            let _ = std::process::Command::new("kill")
                .args(["-9", &pid.to_string()])
                .status();
        }
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn disk_config(root: &str) -> NamespaceConfig {
    NamespaceConfig {
        embedding: EmbeddingChoice::Local { model: None },
        entity: EntityChoice::Keyword { language: None },
        vector: VectorChoice::Memory,
        object: ObjectChoice::Disk {
            root: Some(root.to_owned()),
        },
        scoring: ScoringDto::default(),
    }
}

fn get_request(query: &str) -> Request {
    Request::Get {
        namespace: NAMESPACE.to_owned(),
        query: query.to_owned(),
        keys: vec!["u1".to_owned()],
        context: None,
    }
}

fn set_request(query: &str, value: Vec<u8>) -> Request {
    Request::Set {
        namespace: NAMESPACE.to_owned(),
        query: query.to_owned(),
        keys: vec!["u1".to_owned()],
        context: None,
        value: serde_bytes::ByteBuf::from(value),
    }
}

fn del_request(query: &str) -> Request {
    Request::Del {
        namespace: NAMESPACE.to_owned(),
        query: query.to_owned(),
        keys: vec!["u1".to_owned()],
        context: None,
    }
}

fn assert_hit(stub: &mut ClientStub, query: &str, expected: &[u8]) {
    assert_eq!(
        stub.request(&get_request(query)).unwrap(),
        Response::Value(Some(serde_bytes::ByteBuf::from(expected.to_vec()))),
        "read-after-write: get should return the just-set value with no polling"
    );
}

#[test]
#[ignore = "needs BGE model"]
fn serving_roundtrips_over_the_real_daemon() {
    let env = TestEnv::new(120);
    let objects = env.dir.join("objects");
    let config_json = serde_json::to_string(&disk_config(objects.to_str().unwrap())).unwrap();

    let mut stub = connect_or_spawn(&env.launcher()).unwrap();

    let registered = stub
        .request(&Request::RegisterNamespace {
            namespace: NAMESPACE.to_owned(),
            config_json,
        })
        .unwrap();
    assert_eq!(registered, Response::Registered);

    // (i) async set; read-after-write means the very next get returns the value
    // immediately, served from the pending shadow with no polling.
    let paris = b"paris".to_vec();
    assert_eq!(
        stub.request(&set_request("what is the capital of france", paris.clone()))
            .unwrap(),
        Response::Accepted(true)
    );
    assert_hit(&mut stub, "what is the capital of france", &paris);

    // (ii) an unrelated query misses.
    assert_eq!(
        stub.request(&get_request("how do tides work")).unwrap(),
        Response::Value(None)
    );

    // (iii) delete removes the entry; the next get misses.
    assert_eq!(
        stub.request(&del_request("what is the capital of france"))
            .unwrap(),
        Response::Deleted(true)
    );
    assert_eq!(
        stub.request(&get_request("what is the capital of france"))
            .unwrap(),
        Response::Value(None)
    );

    // (iv) a ~10 MiB value round-trips intact, proving the 64 MiB framing fix.
    let large: Vec<u8> = (0..10 * 1024 * 1024).map(|i| (i % 251) as u8).collect();
    assert_eq!(
        stub.request(&set_request("a very large blob", large.clone()))
            .unwrap(),
        Response::Accepted(true)
    );
    assert_hit(&mut stub, "a very large blob", &large);

    stub.bye().unwrap();
}
