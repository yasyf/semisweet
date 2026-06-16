# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Core semantic cache with hybrid scoring: a `get` embeds the query, vector-searches the nearest entries, and accepts a hit when cosine similarity clears a threshold; shared entities relax the threshold toward a floor and add a similarity bonus, gated by contains-all filter keys.
- Four pluggable backend axes with eight builtins: embeddings (local BGE-small, Voyage), entities (YAKE keywords, GLiNER), vector storage (in-memory, turbopuffer), and object storage (on-disk, S3). Payloads offload to the object store; the model-backed backends compile behind the `local-embed` and `gliner` cargo features, leaving a pure-Rust-plus-HTTP default wheel.
- A lazily-started per-user orphan daemon that holds the loaded models and in-memory index, serves every Python process over a length-framed IPC protocol, and idle-shuts-down after a no-client timeout.
- Python API via pyo3: `SemanticCache.builder(namespace)`, the backend selector methods, `CacheQuery(query, keys, context=None)`, and `set` (write-behind), `get`, and `delete`. Ships as a single abi3 wheel for Python 3.9+.

[Unreleased]: https://github.com/yasyf/semisweet/commits/main
