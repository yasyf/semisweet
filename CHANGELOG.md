# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Core semantic cache with hybrid scoring: a `get` embeds the query, vector-searches the nearest entries, and accepts a hit when cosine similarity clears a threshold; shared entities relax the threshold toward a floor and add a similarity bonus, gated by contains-all filter keys.
- Declarative, keyword-only Python API: a cache is built by passing optional backend objects to `SemanticCache(namespace=..., embedding=..., entities=..., vectors=..., storage=..., scoring=...)`, each axis chosen from eight builtins (`LocalEmbedding`, `VoyageEmbedding`, `KeywordEntities`, `GlinerEntities`, `MemoryVectors`, `TurbopufferVectors`, `DiskStorage`, `S3Storage`). Every argument is optional; a bare namespace runs fully offline on local BGE-small + YAKE keywords + in-memory index + on-disk objects. `CacheQuery(query=..., keys=..., context=...)` is keyword-only too.
- Batteries-included default build (local BGE-small embeddings, Voyage, turbopuffer, S3, keyword entities), with GLiNER span-label entities behind the `gliner` cargo feature. Local models auto-download from the Hugging Face Hub on first use and run offline after: BGE via fastembed, GLiNER via upstream `gline-rs` + `hf-hub` (no env-var model paths; override with `GlinerEntities` keywords).
- Read-after-write with no clock or polling: a per-namespace in-memory shadow holds each in-flight write so a `get` for the same query returns it immediately, and the write-behind queue drains it to the durable backends in the background.
- A lazily-started per-user orphan daemon that holds the loaded models and in-memory index, serves every Python process over a length-framed IPC protocol, and idle-shuts-down after a no-client timeout. Ships as a single abi3 wheel for Python 3.9+ with `set`, `get`, and `delete`.
- Module-level `shutdown_daemon()` coroutine that signals the shared daemon to exit and reports whether one was running. Most programs never call it, since the daemon idle-shuts-down on its own.

### Changed
- Python API is async-native. `SemanticCache(...)` is a synchronous config object that does no I/O; `get`, `set`, and `delete` are coroutines that connect lazily on first await, transparently spawning and connecting the shared per-user daemon.
- Rust client transport moved to tokio — an async `UnixStream` with length-delimited framing — matching the daemon's runtime.
- Reframed the project as "an async, in-memory semantic cache with pluggable backends": turbopuffer is one optional vector backend, not the storage layer.

### Removed
- The sync context manager (`with SemanticCache(...) as cache`, `close()`/`disconnect()`). The daemon is an invisible implementation detail, so there is nothing to open or close by hand.

### Fixed
- `get` returns `bytes` (or `None`) instead of `list[int]`.

[Unreleased]: https://github.com/yasyf/semisweet/commits/main
