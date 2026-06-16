"""The async, object-aware ``semisweet.SemanticCache`` over a mocked raw boundary.

These run with no daemon and no model (the ``mock_raw_cache`` fixture fakes the raw
``core.SemanticCache``). They exercise real serde and the real, hashable ``CacheQuery``.
"""

from pydantic import BaseModel

import semisweet


class _Doc(BaseModel):
    title: str
    score: int


async def test_set_then_get_returns_an_equal_object(mock_raw_cache):
    cache = semisweet.SemanticCache(namespace="ns")
    query = semisweet.CacheQuery(query="capital of france")
    assert await cache.set(query, {"answer": "paris"}) is True
    assert await cache.get(query) == {"answer": "paris"}


async def test_pydantic_value_rehydrates_to_the_same_class(mock_raw_cache):
    cache = semisweet.SemanticCache(namespace="ns")
    query = semisweet.CacheQuery(query="doc")
    doc = _Doc(title="t", score=9)
    await cache.set(query, doc)
    got = await cache.get(query)
    assert type(got) is _Doc
    assert got == doc


async def test_miss_returns_the_MISS_sentinel(mock_raw_cache):
    cache = semisweet.SemanticCache(namespace="ns")
    assert await cache.get(semisweet.CacheQuery(query="absent")) is semisweet.MISS


async def test_a_stored_none_is_distinct_from_a_miss(mock_raw_cache):
    cache = semisweet.SemanticCache(namespace="ns")
    query = semisweet.CacheQuery(query="holds none")
    await cache.set(query, None)
    assert await cache.get(query) is None


async def test_delete_removes_the_entry(mock_raw_cache):
    cache = semisweet.SemanticCache(namespace="ns")
    query = semisweet.CacheQuery(query="x")
    await cache.set(query, 123)
    assert await cache.delete(query) is True
    assert await cache.get(query) is semisweet.MISS
