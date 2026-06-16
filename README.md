# semisweet

![semisweet banner](docs/assets/readme-banner.webp)

[![CI](https://img.shields.io/github/actions/workflow/status/yasyf/semisweet/ci.yml?branch=main&label=CI)](https://github.com/yasyf/semisweet/actions/workflows/ci.yml)
[![License: PolyForm-Noncommercial-1.0.0](https://img.shields.io/badge/License-PolyForm--Noncommercial--1.0.0-blue.svg)](https://github.com/yasyf/semisweet/blob/main/LICENSE)

An in-memory semantic cache backed by turbopuffer.

semisweet keeps your hot embeddings and their payloads in process, so semantic lookups resolve in microseconds instead of a round trip to a vector store. When the working set outgrows memory, it falls back to [turbopuffer](https://turbopuffer.com) for durable, larger-than-RAM recall — and the cache core is Rust, exposed to Python through pyo3.

> **Status:** early development. The extension builds and imports today; the public cache API is still taking shape.

## Install

semisweet builds from source with [uv](https://docs.astral.sh/uv/) and maturin:

```bash
git clone https://github.com/yasyf/semisweet
cd semisweet
uv venv && uvx maturin develop --uv
```

## Quickstart

With the extension installed, import it:

```bash
python -c "import semisweet; print(semisweet.__name__)"
# => semisweet
```

## What problems does this solve?

- **Network round trips on every lookup.** RAG and agent loops re-query the same vectors constantly; semisweet serves the hot set from process memory and reaches for turbopuffer only on a miss.
- **Working sets larger than RAM.** Hold the hot tier in memory and let turbopuffer keep the durable, oversized remainder — one interface over both.
- **Python overhead on the hot path.** The cache core is Rust; pyo3 exposes it without copying through slow glue code.
- **One artifact to ship.** A single abi3 wheel covers Python 3.9+, so consumers install without a native toolchain.

## License

PolyForm-Noncommercial-1.0.0. See [LICENSE](https://github.com/yasyf/semisweet/blob/main/LICENSE).
