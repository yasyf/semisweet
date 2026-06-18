//! Concurrency benchmark for the embedding hot path — the measure-first baseline for
//! the `LocalEmbedding` mutex fix.
//!
//! Layer A (always runs, model-free) contrasts two fake `EmbeddingBackend`s that only
//! differ in their locking discipline: `SleepEmbedding` models the lock-free backend
//! (concurrent inference behind `&self`), `MutexSleepEmbedding` models today's design
//! (one mutex serializing every embed). Sweeping the thread count shows the serialized
//! variant staying flat while the lock-free one scales — the pathology the fix removes.
//!
//! Layer B (gated on `SEMISWEET_MODEL_CACHE`, needs the BGE download) drives concurrent
//! `set`/`get` against a real `build_cache`, so the same scaling can be measured on the
//! actual `LocalEmbedding` path before and after the fix. It skips with a printed line
//! when the model cache is unset, so CI runs Layer A alone.

use std::collections::BTreeSet;
use std::hint::black_box;
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use semisweet::{
    Context, Dim, DynCache, Embedding, EmbeddingBackend, EmbeddingChoice, EntityChoice, Key,
    NamespaceConfig, ObjectChoice, QueryText, Result, ScoringDto, VectorChoice, build_cache,
};

const EMBED_DIM: usize = 384;
const EMBED_DELAY: Duration = Duration::from_micros(500);
const PER_THREAD: usize = 8;
const THREAD_COUNTS: [usize; 4] = [1, 2, 4, 8];

fn unit_embedding(dim: Dim) -> Result<Embedding> {
    Embedding::new(vec![1.0; dim.get()])
}

/// Post-fix backend: inference runs concurrently behind `&self`, no shared lock.
struct SleepEmbedding {
    delay: Duration,
    dim: Dim,
}

impl EmbeddingBackend for SleepEmbedding {
    fn dim(&self) -> Dim {
        self.dim
    }

    fn embed_query(&self, _text: &str) -> Result<Embedding> {
        thread::sleep(self.delay);
        unit_embedding(self.dim)
    }
}

/// Current backend: a single mutex serializes every embed, so worker threads queue.
struct MutexSleepEmbedding {
    delay: Duration,
    dim: Dim,
    lock: Mutex<()>,
}

impl EmbeddingBackend for MutexSleepEmbedding {
    fn dim(&self) -> Dim {
        self.dim
    }

    fn embed_query(&self, _text: &str) -> Result<Embedding> {
        let _guard = self.lock.lock().expect("bench embed mutex poisoned");
        thread::sleep(self.delay);
        unit_embedding(self.dim)
    }
}

fn drive_embed(backend: &dyn EmbeddingBackend, threads: usize, per_thread: usize) {
    thread::scope(|scope| {
        for worker in 0..threads {
            scope.spawn(move || {
                for index in 0..per_thread {
                    let embedding = backend
                        .embed_query(&format!("bench query {worker}-{index}"))
                        .expect("bench embed");
                    black_box(embedding);
                }
            });
        }
    });
}

fn bench_embed_scaling(c: &mut Criterion) {
    let dim = Dim::new(EMBED_DIM).expect("nonzero embedding dim");
    let serialized = MutexSleepEmbedding {
        delay: EMBED_DELAY,
        dim,
        lock: Mutex::new(()),
    };
    let lock_free = SleepEmbedding {
        delay: EMBED_DELAY,
        dim,
    };

    let mut group = c.benchmark_group("embed_concurrency");
    for &threads in &THREAD_COUNTS {
        group.throughput(Throughput::Elements((threads * PER_THREAD) as u64));
        group.bench_with_input(
            BenchmarkId::new("serialized_mutex", threads),
            &threads,
            |b, &threads| b.iter(|| drive_embed(&serialized, threads, PER_THREAD)),
        );
        group.bench_with_input(
            BenchmarkId::new("lock_free", threads),
            &threads,
            |b, &threads| b.iter(|| drive_embed(&lock_free, threads, PER_THREAD)),
        );
    }
    group.finish();
}

fn drive_cache(cache: &DynCache, threads: usize, per_thread: usize) {
    thread::scope(|scope| {
        for worker in 0..threads {
            scope.spawn(move || {
                for index in 0..per_thread {
                    let query = QueryText::new(format!("bench query {worker} item {index}"))
                        .expect("bench query");
                    let keys: BTreeSet<Key> = [Key::new(format!("k{worker}")).expect("bench key")]
                        .into_iter()
                        .collect();
                    let context: Option<Context> = None;
                    cache
                        .set(&query, &keys, &context, b"benchmark-value")
                        .expect("bench set");
                    black_box(cache.get(&query, &keys, &context).expect("bench get"));
                }
            });
        }
    });
}

fn bench_real_cache(c: &mut Criterion) {
    if std::env::var_os("SEMISWEET_MODEL_CACHE").is_none() {
        eprintln!(
            "[bench] skip real-cache layer: SEMISWEET_MODEL_CACHE unset (needs the BGE model)"
        );
        return;
    }

    let root = tempfile::tempdir().expect("bench temp dir");
    let config = NamespaceConfig {
        embedding: EmbeddingChoice::Local { model: None },
        entity: EntityChoice::Keyword { language: None },
        vector: VectorChoice::Memory,
        object: ObjectChoice::Disk {
            root: Some(root.path().to_string_lossy().into_owned()),
        },
        scoring: ScoringDto::default(),
    };
    let cache = build_cache("bench", &config).expect("bench build_cache");

    let mut group = c.benchmark_group("real_cache_concurrency");
    group.sample_size(10);
    for &threads in &THREAD_COUNTS {
        group.throughput(Throughput::Elements((threads * PER_THREAD) as u64));
        group.bench_with_input(
            BenchmarkId::new("set_get", threads),
            &threads,
            |b, &threads| b.iter(|| drive_cache(&cache, threads, PER_THREAD)),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_embed_scaling, bench_real_cache);
criterion_main!(benches);
