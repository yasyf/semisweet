"""End-to-end object cache + decorator against a real daemon (needs the BGE model).

These drive the lazily-spawned daemon on the fully-local stack and run whenever the BGE
model is cached, exercising the ``semisweet.core._run_daemon`` spawn path, object serde,
and pydantic rehydration through the real ``CacheQuery``. ``_Answer`` is module-level so
its type descriptor resolves on load.
"""

import pytest
from pydantic import BaseModel

import semisweet
from semisweet import CacheQuery, DiskStorage, KeywordEntities, LocalEmbedding, MemoryVectors


class _Answer(BaseModel):
    text: str
    confidence: float


def _backends(runtime) -> dict[str, object]:
    return {
        "embedding": LocalEmbedding(),
        "entities": KeywordEntities(),
        "vectors": MemoryVectors(),
        "storage": DiskStorage(root=str(runtime / "objects")),
    }


@pytest.mark.needs_model
async def test_object_cache_roundtrips_a_pydantic_model(runtime):
    cache = semisweet.SemanticCache(namespace="obj-e2e", **_backends(runtime))
    query = CacheQuery(query="what is the capital of france")
    model = _Answer(text="paris", confidence=0.99)

    assert await cache.set(query, model) is True
    got = await cache.get(query)
    assert type(got) is _Answer
    assert got == model

    assert await cache.delete(query) is True
    assert await cache.get(query) is semisweet.MISS


@pytest.mark.needs_model
async def test_decorator_serves_repeat_calls_from_cache(runtime):
    calls = 0

    @semisweet.cache(query="question", **_backends(runtime))
    async def answer(question: str) -> _Answer:
        nonlocal calls
        calls += 1
        return _Answer(text="paris", confidence=0.9)

    first = await answer("what is the capital of france")
    assert calls == 1
    assert type(first) is _Answer

    second = await answer("what is the capital of france")
    assert calls == 1  # read-after-write: served from the cache, not recomputed
    assert second == first
