# semisweet

![semisweet banner](docs/assets/readme-banner.webp)

[![CI](https://img.shields.io/github/actions/workflow/status/yasyf/semisweet/ci.yml?branch=main&label=CI)](https://github.com/yasyf/semisweet/actions/workflows/ci.yml)
[![License: PolyForm-Noncommercial-1.0.0](https://img.shields.io/badge/License-PolyForm--Noncommercial--1.0.0-blue.svg)](https://github.com/yasyf/semisweet/blob/main/LICENSE)

An in-memory semantic cache backed by turbopuffer.

semisweet caches by meaning. It stores a payload against a query embedding and serves it again for any semantically close query, so repeated RAG answers, LLM completions, and tool results resolve from process memory in microseconds — no recompute, no vector-store round trip. The hot index lives in process; swap in [turbopuffer](https://turbopuffer.com) when recall has to outlive a process or outgrow RAM. Payloads offload to disk or S3 instead of bloating the index, so the index carries only vectors and filter metadata. The core is Rust, exposed to Python through pyo3.

A lookup is a hybrid match. semisweet embeds the query, pulls the nearest entries, and accepts the best when cosine similarity clears a threshold. Shared entities between the query and a candidate relax that threshold toward a floor and add a similarity bonus, so "france's capital?" lands on the entry you stored for "what is the capital of france".

## Install

semisweet builds from source with [uv](https://docs.astral.sh/uv/) and maturin:

```bash
git clone https://github.com/yasyf/semisweet
cd semisweet
uv venv && uvx maturin develop --uv
```

The default build is batteries-included: a bare `SemanticCache(namespace="...")` runs fully offline on local BGE-small embeddings, YAKE keyword entities, an in-process index, and on-disk payloads — no API keys, no config. The Voyage, turbopuffer, and S3 backends compile in alongside it, and the whole thing installs as a single abi3 wheel for Python 3.9+. Opt into GLiNER span-label entities with one extra feature:

```bash
uvx maturin develop --uv --features gliner
```

## Quickstart

Build a fully in-process cache, store a value, and read it back through a differently worded query. The defaults — local BGE embeddings, keyword entities, an in-process index, on-disk payloads — need no API keys, so a bare namespace is the whole setup.

```python
import semisweet

# Every backend defaults, so a namespace is all you need. Pass LocalEmbedding,
# KeywordEntities, MemoryVectors, or DiskStorage to override an axis.
cache = semisweet.SemanticCache(namespace="research-cache")

# `set` is read-after-write: a `get` for the same query returns the value at once,
# while the durable write drains in the background.
cache.set(semisweet.CacheQuery(query="what is the capital of france"), b"paris")

# A reworded query still hits the stored entry.
hit = cache.get(semisweet.CacheQuery(query="france's capital?"))
print(hit)
```

```
b'paris'
```

`CacheQuery(query=..., keys=..., context=...)` is keyword-only: `query` is the text to match on, `keys` an optional set of exact-match filter tags, `context` optional fallback text. `keys` is a contains-all filter — an entry matches only when it carries every key in the query — and `context` feeds entity extraction and breaks lexical ties when the query alone is thin. `get` returns the stored `bytes` on a hit or `None` on a miss; `delete` drops the entry and returns whether it existed.

## Backends

semisweet has four pluggable backend axes, each set once by passing a backend object to `SemanticCache`. Eight builtins ship across them, every constructor keyword-only with all-optional arguments:

| Axis | Builtins |
|------|----------|
| Embedding | `LocalEmbedding(model=...)` — BGE-small on CPU; `VoyageEmbedding(model=..., dim=...)` — Voyage HTTP API |
| Entities | `KeywordEntities(lang=...)` — YAKE keywords; `GlinerEntities(labels=..., repo=...)` — GLiNER span labels (`--features gliner`) |
| Vector index | `MemoryVectors()` — in-process; `TurbopufferVectors()` — turbopuffer |
| Object store | `DiskStorage(root=...)` — local filesystem; `S3Storage(bucket=..., region=..., endpoint=..., prefix=...)` — S3-compatible |

Payloads always live in the object store, so offload is first-class. `LocalEmbedding` ships in the default build; `GlinerEntities` compiles behind the `gliner` cargo feature. Both download their model from the Hugging Face Hub on first use and run offline after — the same auto-download as fastembed's BGE, cached under `SEMISWEET_MODEL_CACHE`. Point `GlinerEntities` at a different model through its `repo` / `model` / `tokenizer` keywords. Each remote backend reads its credentials from the environment as you select it:

| Variable | Read by |
|----------|---------|
| `VOYAGE_API_KEY` | `VoyageEmbedding` |
| `TURBOPUFFER_API_KEY` | `TurbopufferVectors` |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `S3_ENDPOINT`, `SEMISWEET_S3_BUCKET` | `S3Storage` |

## Architecture

Models and the in-memory index cost real time to load, so semisweet keeps them out of your Python process. The first cache you build lazily spawns a single per-user daemon: an orphan process that holds the loaded models and the in-memory index and serves every cache across every Python process you run. It outlives the process that spawned it and idle-shuts-down once no client has talked to it for a timeout, so a burst of scripts shares one warm copy of the models instead of paying the load cost each time.

## Development

Rust unit tests live beside the code in `src/`; Python binding tests live in `tests/`:

```bash
cargo test
uv venv && uvx maturin develop --uv && uv pip install pytest && pytest
```

See [AGENTS.md](AGENTS.md) for the full conventions.

## License

PolyForm-Noncommercial-1.0.0. See [LICENSE](https://github.com/yasyf/semisweet/blob/main/LICENSE).
