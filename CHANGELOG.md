# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Core semantic cache with dense-vector scoring: a `get` runs a dense-vector search and accepts a hit when the cosine clears the dense floor, gated by a contains-all key filter, an entity hard-gate (reject when the query/candidate entity overlap is below `max(1, n/3)` of the query's `n` entities), and — when the query carries `context` — a context hard-gate delegated to a backend BM25 match over the stored `context` field (the in-memory backend's own BM25 index, or turbopuffer's native `rank_by` full-text search), surfaced as the hit's sparse score and rejected when it falls below `context_gate`. The dense floor is context-present: a context-bearing query whose context match clears the gate is backstopped on precision by that hard-gate, so it need only reach the lower `context_threshold`, while a context-less query (or `context` set to 'ignore') must reach the full `threshold`. Tune `threshold`, `context_gate`, `context_threshold`, `top_k`, `entity_filter`, and `context` ('ignore' or 'gate') on `Scoring`; the defaults are calibrated precision-first (`threshold` 0.92, `context_gate` 0.10, `context_threshold` 0.88).
- Declarative, keyword-only Python API: a cache is built by passing optional backend objects to `SemanticCache(namespace=..., embedding=..., entities=..., vectors=..., storage=..., scoring=...)`, each axis chosen from eight builtins (`LocalEmbedding`, `VoyageEmbedding`, `KeywordEntities`, `GlinerEntities`, `MemoryVectors`, `TurbopufferVectors`, `DiskStorage`, `S3Storage`). Every argument is optional; a bare namespace runs fully offline on local BGE-small + YAKE keywords + in-memory index + on-disk objects. `CacheQuery(query=..., keys=..., context=...)` is keyword-only too.
- Batteries-included default build (local BGE-small embeddings, Voyage, turbopuffer, S3, keyword entities), with GLiNER span-label entities behind the `gliner` cargo feature. Local models auto-download from the Hugging Face Hub on first use and run offline after: BGE via fastembed, GLiNER via upstream `gline-rs` + `hf-hub` (no env-var model paths; override with `GlinerEntities` keywords).
- Read-after-write with no clock or polling: a per-namespace in-memory shadow holds each in-flight write so a `get` for the same query returns it immediately, and the write-behind queue drains it to the durable backends in the background.
- A lazily-started per-user orphan daemon that holds the loaded models and in-memory index, serves every Python process over a length-framed IPC protocol, and idle-shuts-down after a no-client timeout. Ships as a single abi3 wheel for Python 3.9+ with `set`, `get`, and `delete`.
- Module-level `shutdown_daemon()` coroutine that signals the shared daemon to exit and reports whether one was running. Most programs never call it, since the daemon idle-shuts-down on its own.
- Async, object-aware caching at the top level: `semisweet.SemanticCache` stores and returns whole Python objects, and `@semisweet.cache` memoizes an `async def` through a semantic cache keyed on the function's `module:qualname`. A Pydantic model round-trips to its exact class through a type-tagged JSON envelope; any other value falls back to `pickle`. A miss is the `MISS` sentinel, so `None` is a cacheable value. The decorator embeds the sole string argument (or the one named by `query=`), scopes entries with `keys=`, and boosts matching results with `context=`.
- Optional `pydantic` extra (`pip install "semisweet[pydantic]"`) for the JSON-envelope serialization path; without it, every value serializes through `pickle`.
- Value payloads are transparently compressed with zstd (max-ratio level) before reaching the object store and decompressed on read, cutting at-rest disk and S3 transfer for typical text/JSON/answer payloads. Always on, and the cache stays byte-exact across the round trip. Cache directories written before this change should be cleared, since objects are now read back through the decompressor.

### Changed
- An entry's identity now includes its `context`: the same query and keys with a different `context` are stored as distinct cache entries (enabling context disambiguation) rather than the later write overwriting the earlier one.
- Python API is async-native. `SemanticCache(...)` is a synchronous config object that does no I/O; `get`, `set`, and `delete` are coroutines that connect lazily on first await, transparently spawning and connecting the shared per-user daemon.
- Rust client transport moved to tokio — an async `UnixStream` with length-delimited framing — matching the daemon's runtime.
- Reframed the project as "an async, in-memory semantic cache with pluggable backends": turbopuffer is one optional vector backend, not the storage layer.
- Split into a mixed Rust/Python layout: the compiled extension is now the `semisweet.core` submodule, and the top-level `semisweet` package re-exports `CacheQuery`, the backend config classes, the exception hierarchy, and `shutdown_daemon`. Top-level `SemanticCache` is the object-aware async cache; the raw, bytes-valued cache is `semisweet.core.SemanticCache`.

### Removed
- The sync context manager (`with SemanticCache(...) as cache`, `close()`/`disconnect()`). The daemon is an invisible implementation detail, so there is nothing to open or close by hand.

### Fixed
- `get` returns `bytes` (or `None`) instead of `list[int]`.

[Unreleased]: https://github.com/yasyf/semisweet/commits/main
