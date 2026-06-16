"""The async, object-aware semantic cache.

Wraps the raw, bytes-valued :class:`semisweet.core.SemanticCache` with serialization
so callers store and retrieve arbitrary Python objects. Construction is cheap pure
config; the daemon is connected lazily, inside the awaited ``get``/``set``/``delete``.

Named ``objectcache`` rather than ``cache`` so the module never collides with the
package-level :func:`semisweet.cache` decorator attribute.
"""

from __future__ import annotations

from .core import (
    CacheQuery,
    DiskStorage,
    GlinerEntities,
    KeywordEntities,
    LocalEmbedding,
    MemoryVectors,
    S3Storage,
    Scoring,
    TurbopufferVectors,
    VoyageEmbedding,
)
from .core import SemanticCache as _RawSemanticCache
from .serde import dumps, loads

__all__ = ["MISS", "SemanticCache"]


class _Missing:
    """The type of :data:`MISS`."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "MISS"


MISS = _Missing()
"""Sentinel returned by :meth:`SemanticCache.get` on a miss; distinct from a cached ``None``."""


class SemanticCache:
    """An async semantic cache that stores and returns arbitrary Python objects.

    Mirrors :class:`semisweet.core.SemanticCache` but serializes values: pydantic
    models round-trip to their exact class, everything else via ``pickle``. A miss is
    the :data:`MISS` sentinel, so ``None`` is a cacheable value.
    """

    def __init__(
        self,
        *,
        namespace: str,
        embedding: LocalEmbedding | VoyageEmbedding | None = None,
        entities: KeywordEntities | GlinerEntities | None = None,
        vectors: MemoryVectors | TurbopufferVectors | None = None,
        storage: DiskStorage | S3Storage | None = None,
        scoring: Scoring | None = None,
    ) -> None:
        self._namespace = namespace
        self._raw = _RawSemanticCache(
            namespace=namespace,
            embedding=embedding,
            entities=entities,
            vectors=vectors,
            storage=storage,
            scoring=scoring,
        )

    async def get(self, query: CacheQuery) -> object:
        """Return the cached object for ``query``, or :data:`MISS` on a miss."""
        data = await self._raw.get(query)
        return MISS if data is None else loads(data)

    async def set(self, query: CacheQuery, value: object) -> bool:
        """Store ``value`` under ``query``; return ``True`` if the daemon accepted it."""
        return await self._raw.set(query, dumps(value))

    async def delete(self, query: CacheQuery) -> bool:
        """Evict the entry matching ``query``; return ``True`` if one was removed."""
        return await self._raw.delete(query)

    def __repr__(self) -> str:
        return f"SemanticCache(namespace={self._namespace!r})"
