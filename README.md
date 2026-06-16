# semisweet

![semisweet banner](docs/assets/readme-banner.webp)

[![CI](https://img.shields.io/github/actions/workflow/status/yasyf/semisweet/ci.yml?branch=main&label=CI)](https://github.com/yasyf/semisweet/actions/workflows/ci.yml)
[![License: PolyForm-Noncommercial-1.0.0](https://img.shields.io/badge/License-PolyForm--Noncommercial--1.0.0-blue.svg)](https://github.com/yasyf/semisweet/blob/main/LICENSE)

An in-memory semantic cache backed by turbopuffer.

semisweet caches by meaning. Store a payload against a query, and any semantically close query gets it back from process memory in microseconds — no recompute, no vector-store round trip. The hot index lives in process; point it at [turbopuffer](https://turbopuffer.com) when recall has to outlive a process or outgrow RAM, and payloads offload to disk or S3 so the index stays lean. The core is Rust, exposed to Python through pyo3.

## Install

```bash
git clone https://github.com/yasyf/semisweet
cd semisweet
uv venv && uvx maturin develop --uv
```

The default build runs fully offline — local BGE embeddings, keyword entities, an in-process index, on-disk payloads — and installs as a single abi3 wheel for Python 3.9+. Voyage, turbopuffer, and S3 are built in; add GLiNER entities with `--features gliner`.

## Quickstart

A namespace is the whole setup: every backend defaults to the local stack, so this needs no API keys.

```python
import semisweet

cache = semisweet.SemanticCache(namespace="research-cache")

# `set` is read-after-write: a `get` for the same query returns the value at once.
cache.set(semisweet.CacheQuery(query="what is the capital of france"), b"paris")

# A reworded query still hits the stored entry.
hit = cache.get(semisweet.CacheQuery(query="france's capital?"))
print(hit)
```

```
b'paris'
```

`CacheQuery(query=..., keys=..., context=...)` is keyword-only. `keys` is an optional contains-all filter; `context` is optional fallback text for entity extraction and tie-breaking. `get` returns `bytes` or `None`, and `delete` returns whether the entry existed.

## Backends

Swap any axis by passing a backend object — all keyword-only, every argument optional:

| Axis | Builtins |
|------|----------|
| Embedding | `LocalEmbedding(model=...)` — BGE-small on CPU; `VoyageEmbedding(model=..., dim=...)` — Voyage HTTP API |
| Entities | `KeywordEntities(lang=...)` — YAKE keywords; `GlinerEntities(labels=..., repo=...)` — GLiNER spans (`--features gliner`) |
| Vector index | `MemoryVectors()` — in-process; `TurbopufferVectors()` — turbopuffer |
| Object store | `DiskStorage(root=...)` — local filesystem; `S3Storage(bucket=..., region=..., endpoint=..., prefix=...)` — S3-compatible |

Local models (BGE, and GLiNER under `--features gliner`) auto-download from the Hugging Face Hub on first use, cached under `SEMISWEET_MODEL_CACHE`. Remote backends read credentials from the environment:

| Variable | Read by |
|----------|---------|
| `VOYAGE_API_KEY` | `VoyageEmbedding` |
| `TURBOPUFFER_API_KEY` | `TurbopufferVectors` |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `S3_ENDPOINT`, `SEMISWEET_S3_BUCKET` | `S3Storage` |

The first cache spawns a shared per-user daemon that holds the models and index, so only the first call pays the load cost.

## Development

```bash
cargo test
uv venv && uvx maturin develop --uv && uv pip install pytest && pytest
```

See [AGENTS.md](AGENTS.md) for conventions.

## License

PolyForm-Noncommercial-1.0.0. See [LICENSE](https://github.com/yasyf/semisweet/blob/main/LICENSE).
