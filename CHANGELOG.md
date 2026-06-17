# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Core semantic cache with hybrid retrieval and scoring: a `get` runs a hybrid dense-vector + BM25 lexical search, fuses the two into one weighted score, and accepts a hit when the fused score clears a threshold; shared entities relax the threshold toward a floor, gated by contains-all filter keys. Both the query and the optional `context` are matched with the same hybrid dense+sparse signal across every vector backend (the in-memory index builds its own BM25; turbopuffer uses native full-text BM25 via a multi-query round trip). A present query `context` that overlaps a candidate's stored context relaxes the threshold further, bioqa-style — a boost that never penalizes its absence. Tune `dense_weight`, `sparse_weight`, and `context_bonus_weight` on `Scoring`. The defaults are calibrated so a hit with no lexical overlap clears exactly the original pure-dense cosine threshold, and any lexical or context overlap only lowers the bar, never raises it.
- Declarative, keyword-only Python API: a cache is built by passing optional backend objects to `SemanticCache(namespace=..., embedding=..., entities=..., vectors=..., storage=..., scoring=...)`, each axis chosen from eight builtins (`LocalEmbedding`, `VoyageEmbedding`, `KeywordEntities`, `GlinerEntities`, `MemoryVectors`, `TurbopufferVectors`, `DiskStorage`, `S3Storage`). Every argument is optional; a bare namespace runs fully offline on local BGE-small + YAKE keywords + in-memory index + on-disk objects. `CacheQuery(query=..., keys=..., context=...)` is keyword-only too.
- Batteries-included default build (local BGE-small embeddings, Voyage, turbopuffer, S3, keyword entities), with GLiNER span-label entities behind the `gliner` cargo feature. Local models auto-download from the Hugging Face Hub on first use and run offline after: BGE via fastembed, GLiNER via upstream `gline-rs` + `hf-hub` (no env-var model paths; override with `GlinerEntities` keywords).
- Read-after-write with no clock or polling: a per-namespace in-memory shadow holds each in-flight write so a `get` for the same query returns it immediately, and the write-behind queue drains it to the durable backends in the background.
- A lazily-started per-user orphan daemon that holds the loaded models and in-memory index, serves every Python process over a length-framed IPC protocol, and idle-shuts-down after a no-client timeout. Ships as a single abi3 wheel for Python 3.9+ with `set`, `get`, and `delete`.
- Module-level `shutdown_daemon()` coroutine that signals the shared daemon to exit and reports whether one was running. Most programs never call it, since the daemon idle-shuts-down on its own.
- Async, object-aware caching at the top level: `semisweet.SemanticCache` stores and returns whole Python objects, and `@semisweet.cache` memoizes an `async def` through a semantic cache keyed on the function's `module:qualname`. A Pydantic model round-trips to its exact class through a type-tagged JSON envelope; any other value falls back to `pickle`. A miss is the `MISS` sentinel, so `None` is a cacheable value. The decorator embeds the sole string argument (or the one named by `query=`), scopes entries with `keys=`, and boosts matching results with `context=`.
- Optional `pydantic` extra (`pip install "semisweet[pydantic]"`) for the JSON-envelope serialization path; without it, every value serializes through `pickle`.
- Value payloads are transparently compressed with zstd (max-ratio level) before reaching the object store and decompressed on read, cutting at-rest disk and S3 transfer for typical text/JSON/answer payloads. Always on, and the cache stays byte-exact across the round trip. Cache directories written before this change should be cleared, since objects are now read back through the decompressor.

### Changed
- Python API is async-native. `SemanticCache(...)` is a synchronous config object that does no I/O; `get`, `set`, and `delete` are coroutines that connect lazily on first await, transparently spawning and connecting the shared per-user daemon.
- Rust client transport moved to tokio — an async `UnixStream` with length-delimited framing — matching the daemon's runtime.
- Reframed the project as "an async, in-memory semantic cache with pluggable backends": turbopuffer is one optional vector backend, not the storage layer.
- Split into a mixed Rust/Python layout: the compiled extension is now the `semisweet.core` submodule, and the top-level `semisweet` package re-exports `CacheQuery`, the backend config classes, the exception hierarchy, and `shutdown_daemon`. Top-level `SemanticCache` is the object-aware async cache; the raw, bytes-valued cache is `semisweet.core.SemanticCache`.

### Removed
- The sync context manager (`with SemanticCache(...) as cache`, `close()`/`disconnect()`). The daemon is an invisible implementation detail, so there is nothing to open or close by hand.

### Fixed
- `get` returns `bytes` (or `None`) instead of `list[int]`.

[Unreleased]: https://github.com/yasyf/semisweet/commits/main
