"""Type stubs for the compiled ``semisweet.core`` extension module.

``semisweet.core`` is the raw, bytes-valued layer of semisweet: an async, in-memory
semantic cache with pluggable backends. The public surface is declarative and
keyword-only: build a cache out of backend config objects and look it up with frozen,
hashable :class:`CacheQuery` values. The pure-Python ``semisweet`` package re-exports
these names and layers an async object cache + ``@cache`` decorator on top.
"""

from collections.abc import Sequence

__all__ = [
    "CacheQuery",
    "SemanticCache",
    "LocalEmbedding",
    "VoyageEmbedding",
    "KeywordEntities",
    "GlinerEntities",
    "MemoryVectors",
    "TurbopufferVectors",
    "DiskStorage",
    "S3Storage",
    "Scoring",
    "shutdown_daemon",
    "SemisweetError",
    "ConfigError",
    "NamespaceError",
    "BackendError",
    "DaemonError",
]

class CacheQuery:
    """A frozen, hashable cache lookup: a query plus optional entity keys and context."""

    def __init__(
        self, *, query: str, keys: set[str] | None = None, context: str | None = None
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class LocalEmbedding:
    """Embedding backend that runs a sentence-transformer model in-process."""

    def __init__(self, *, model: str | None = None) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class VoyageEmbedding:
    """Embedding backend backed by the Voyage AI API."""

    def __init__(self, *, model: str | None = None, dim: int | None = None) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class KeywordEntities:
    """Entity backend that extracts keyword entities with no model download."""

    def __init__(self, *, lang: str | None = None) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class GlinerEntities:
    """Entity backend backed by a GLiNER ONNX model."""

    def __init__(
        self,
        *,
        labels: Sequence[str] | None = None,
        repo: str | None = None,
        model: str | None = None,
        tokenizer: str | None = None,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class MemoryVectors:
    """In-process vector index; vectors live only for the lifetime of the daemon."""

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class TurbopufferVectors:
    """Vector index backed by turbopuffer; needs turbopuffer credentials in the daemon."""

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class DiskStorage:
    """Object storage on the local filesystem."""

    def __init__(self, *, root: str | None = None) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class S3Storage:
    """Object storage on S3 or an S3-compatible endpoint."""

    def __init__(
        self,
        *,
        bucket: str | None = None,
        region: str | None = None,
        endpoint: str | None = None,
        prefix: str | None = None,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class Scoring:
    """Frozen, hashable scoring configuration for a namespace."""

    def __init__(
        self,
        *,
        base: float | None = None,
        floor: float | None = None,
        entity_bonus_weight: float | None = None,
        top_k: int | None = None,
        entity_filter: bool | None = None,
        context: str | None = None,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class SemanticCache:
    """A raw, bytes-valued semantic cache scoped to one namespace, backed by a shared daemon."""

    def __init__(
        self,
        *,
        namespace: str,
        embedding: LocalEmbedding | VoyageEmbedding | None = None,
        entities: KeywordEntities | GlinerEntities | None = None,
        vectors: MemoryVectors | TurbopufferVectors | None = None,
        storage: DiskStorage | S3Storage | None = None,
        scoring: Scoring | None = None,
    ) -> None: ...
    async def get(self, query: CacheQuery) -> bytes | None:
        """Return the cached value for ``query``, or ``None`` on a miss."""
        ...
    async def set(self, query: CacheQuery, value: bytes) -> bool:
        """Store ``value`` under ``query``; return ``True`` if the daemon accepted it."""
        ...
    async def delete(self, query: CacheQuery) -> bool:
        """Evict the entry matching ``query``; return ``True`` if one was removed."""
        ...
    def __repr__(self) -> str: ...

class SemisweetError(Exception):
    """Base class for every error raised by semisweet."""

class ConfigError(SemisweetError, ValueError):
    """Invalid configuration or value; also a builtin ``ValueError``."""

class NamespaceError(SemisweetError, KeyError):
    """Unknown namespace; also a builtin ``KeyError``."""

class BackendError(SemisweetError, RuntimeError):
    """An embedding, entity, vector, or object-store backend failed; also a ``RuntimeError``."""

class DaemonError(SemisweetError, RuntimeError):
    """The daemon connection, lifecycle, or IO failed; also a ``RuntimeError``."""

async def shutdown_daemon() -> bool:
    """Signal the shared daemon to shut down; return ``True`` if one was running."""
    ...
