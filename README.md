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

The default wheel is pure Rust plus HTTP (Voyage embeddings, turbopuffer, on-disk and S3 object storage, YAKE keyword entities) and installs as a single abi3 wheel for Python 3.9+. To embed locally on CPU with BGE-small instead of calling Voyage, compile the ONNX backend in:

```bash
uvx maturin develop --uv --features local-embed
```

## Quickstart

Build a fully in-process cache — local embeddings, keyword entities, an in-memory index, on-disk payloads — then store a value and read it back through a differently worded query. This path needs the `local-embed` build above and no API keys.

```python
import semisweet

cache = (
    semisweet.SemanticCache.builder("research-cache")
    .embedding_local()       # BGE-small on CPU
    .entities_keyword(None)  # YAKE keyword extraction
    .vector_memory()         # in-process vector index
    .object_disk(None)       # payloads under the user data dir
    .build()
)

# `set` is write-behind: it returns as soon as the daemon accepts the write.
cache.set(semisweet.CacheQuery("what is the capital of france", set()), b"paris")

# A reworded query still hits the stored entry.
hit = cache.get(semisweet.CacheQuery("france's capital?", set()))
print(hit)
```

```
b'paris'
```

`CacheQuery(query, keys, context=None)` carries the text to match on, a set of exact-match filter tags, and optional fallback text. `keys` is a contains-all filter — an entry matches only when it carries every key in the query — and `context` feeds entity extraction and breaks lexical ties when the query alone is thin. `get` returns the stored `bytes` on a hit or `None` on a miss; `delete` drops the entry and returns whether it existed.

## Backends

semisweet has four pluggable backend axes, each chosen once on the builder. Eight builtins ship across them:

| Axis | Builtins |
|------|----------|
| Embedding | `embedding_local()` — BGE-small on CPU (`--features local-embed`); `embedding_voyage(model, dim)` — Voyage HTTP API |
| Entities | `entities_keyword(language)` — YAKE keywords; `entities_gliner(labels)` — GLiNER span labels (`--features gliner`) |
| Vector index | `vector_memory()` — in-process; `vector_turbopuffer()` — turbopuffer |
| Object store | `object_disk(root)` — local filesystem; `object_s3(bucket, region, endpoint, prefix)` — S3-compatible |

Payloads always live in the object store, so offload is first-class. The model-backed backends (`local-embed`, `gliner`) compile only behind their cargo features; the default wheel stays pure Rust plus HTTP. Each backend reads its credentials from the environment as you select it:

| Variable | Read by |
|----------|---------|
| `VOYAGE_API_KEY` | `embedding_voyage` |
| `TURBOPUFFER_API_KEY` | `vector_turbopuffer` |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_ENDPOINT` | `object_s3` |
| `SEMISWEET_GLINER_TOKENIZER`, `SEMISWEET_GLINER_MODEL` | `entities_gliner` |

## Architecture

Models and the in-memory index cost real time to load, so semisweet keeps them out of your Python process. The first cache you build lazily spawns a single per-user daemon: an orphan process that holds the loaded models and the in-memory index and serves every cache across every Python process you run. It outlives the process that spawned it and idle-shuts-down once no client has talked to it for a timeout, so a burst of scripts shares one warm copy of the models instead of paying the load cost each time.

## Development

Rust unit tests live beside the code in `src/`; Python binding tests live in `tests/`:

```bash
cargo test
uv venv && uvx maturin develop --uv && pytest
```

See [AGENTS.md](AGENTS.md) for the full conventions.

## License

PolyForm-Noncommercial-1.0.0. See [LICENSE](https://github.com/yasyf/semisweet/blob/main/LICENSE).
