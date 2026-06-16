"""The ``@cache`` decorator: memoize an async function through a semantic cache.

The namespace is the function's ``module:qualname``. One argument is embedded as the
semantic query (the sole ``str``-annotated parameter by default, or the one named by
``query=``); other arguments are ignored unless named in ``keys`` (exact-match scoping)
or ``context`` (scoring hint). Decorates ``async def`` only.
"""

from __future__ import annotations

import functools
import inspect
import threading
from collections.abc import Awaitable, Callable

from .objectcache import MISS, SemanticCache
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

__all__ = ["cache"]

_caches: dict[str, SemanticCache] = {}
_caches_lock = threading.Lock()


def _query_param(qualname: str, sig: inspect.Signature) -> str:
    candidates = [
        name
        for name, param in sig.parameters.items()
        if param.kind
        in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        and param.annotation in (str, "str")
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"@cache on {qualname}: cannot infer the query parameter "
            f"(found str-typed parameters {candidates}); pass query='<name>'."
        )
    return candidates[0]


def _cache_for(namespace: str, build: Callable[[], SemanticCache]) -> SemanticCache:
    with _caches_lock:
        existing = _caches.get(namespace)
        if existing is None:
            existing = build()
            _caches[namespace] = existing
        return existing


def cache(
    fn: Callable[..., Awaitable[object]] | None = None,
    *,
    query: str | None = None,
    keys: tuple[str, ...] = (),
    context: str | None = None,
    embedding: LocalEmbedding | VoyageEmbedding | None = None,
    entities: KeywordEntities | GlinerEntities | None = None,
    vectors: MemoryVectors | TurbopufferVectors | None = None,
    storage: DiskStorage | S3Storage | None = None,
    scoring: Scoring | None = None,
) -> Callable[..., object]:
    """Memoize an ``async def`` through a per-namespace semantic cache.

    Use bare (``@cache``) or configured (``@cache(query="q", keys=("model",))``). The
    backend arguments configure the underlying cache. A miss runs and stores the wrapped
    coroutine's result; an unpicklable, non-pydantic result fails at store time.
    """
    if fn is None:
        return functools.partial(
            cache,
            query=query,
            keys=keys,
            context=context,
            embedding=embedding,
            entities=entities,
            vectors=vectors,
            storage=storage,
            scoring=scoring,
        )
    if not inspect.iscoroutinefunction(fn):
        raise TypeError(
            f"@cache supports only `async def`; {fn} is not a coroutine function"
        )

    qualname = fn.__qualname__
    namespace = f"{fn.__module__}:{qualname}"
    sig = inspect.signature(fn)
    query_param = query if query is not None else _query_param(qualname, sig)
    referenced = (query_param, *keys, *((context,) if context is not None else ()))
    for name in referenced:
        if name not in sig.parameters:
            raise ValueError(f"@cache on {qualname}: no parameter named {name!r}")

    def build() -> SemanticCache:
        return SemanticCache(
            namespace=namespace,
            embedding=embedding,
            entities=entities,
            vectors=vectors,
            storage=storage,
            scoring=scoring,
        )

    @functools.wraps(fn)
    async def wrapper(*args: object, **kwargs: object) -> object:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        cache_query = CacheQuery(
            query=str(bound.arguments[query_param]),
            keys={str(bound.arguments[name]) for name in keys} or None,
            context=str(bound.arguments[context]) if context is not None else None,
        )
        cache_obj = _cache_for(namespace, build)
        hit = await cache_obj.get(cache_query)
        if hit is not MISS:
            return hit
        result = await fn(*args, **kwargs)
        await cache_obj.set(cache_query, result)
        return result

    return wrapper
