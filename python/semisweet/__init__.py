"""semisweet: an async, in-memory semantic cache with pluggable backends.

The compiled extension lives at :mod:`semisweet.core`; its configuration classes,
exception hierarchy, :class:`CacheQuery`, and :func:`shutdown_daemon` are re-exported
here. The raw, bytes-valued cache stays at ``semisweet.core.SemanticCache``; this
package's :class:`SemanticCache` is the async, object-aware cache, and :func:`cache`
is the decorator that memoizes an ``async def`` through it.
"""

from .objectcache import MISS, SemanticCache
from .core import (
    BackendError,
    CacheQuery,
    ConfigError,
    DaemonError,
    DiskStorage,
    GlinerEntities,
    KeywordEntities,
    LocalEmbedding,
    MemoryVectors,
    NamespaceError,
    S3Storage,
    Scoring,
    SemisweetError,
    TurbopufferVectors,
    VoyageEmbedding,
    shutdown_daemon,
)
from .decorator import cache

__all__ = [
    "SemanticCache",
    "CacheQuery",
    "cache",
    "MISS",
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
