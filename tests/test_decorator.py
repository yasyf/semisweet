"""The ``@cache`` decorator: decoration-time validation and async hit/miss flow.

Decoration-time checks raise immediately and need no fixture. The call-flow tests use
``mock_raw_cache`` (no daemon, no model); the real, hashable ``CacheQuery`` makes equal
queries collide and differing ``keys`` stay apart.
"""

import pytest

import semisweet
from semisweet import cache


def test_requires_a_coroutine_function():
    with pytest.raises(TypeError):

        @cache
        def sync_fn(q: str):
            return q


def test_ambiguous_query_parameter_raises():
    with pytest.raises(ValueError):

        @cache
        async def two_strings(a: str, b: str): ...


def test_no_string_parameter_raises():
    with pytest.raises(ValueError):

        @cache
        async def no_strings(n: int): ...


def test_unknown_query_name_raises():
    with pytest.raises(ValueError):

        @cache(query="missing")
        async def fn(q: str): ...


def test_unknown_key_name_raises():
    with pytest.raises(ValueError):

        @cache(query="q", keys=("absent",))
        async def fn(q: str): ...


async def test_namespace_is_module_qualname(mock_raw_cache):
    @cache(query="q")
    async def labeled(q: str) -> str:
        return "v"

    await labeled("hello")
    import semisweet.decorator as decorator

    assert f"{labeled.__module__}:{labeled.__qualname__}" in decorator._caches


async def test_hit_skips_recompute_and_returns_the_cached_value(mock_raw_cache):
    calls = 0

    @cache(query="q")
    async def fn(q: str) -> str:
        nonlocal calls
        calls += 1
        return f"answer-{calls}"

    first = await fn("same question")
    second = await fn("same question")
    assert calls == 1
    assert first == second == "answer-1"


async def test_keys_scope_distinct_entries(mock_raw_cache):
    calls = 0

    @cache(query="q", keys=("model",))
    async def fn(q: str, model: str) -> int:
        nonlocal calls
        calls += 1
        return calls

    await fn("x", model="a")
    await fn("x", model="b")  # same query, different key -> recompute
    assert calls == 2
    await fn("x", model="a")  # back to the first key -> hit
    assert calls == 2


async def test_context_is_threaded_into_the_query(mock_raw_cache):
    @cache(query="q", context="ctx")
    async def fn(q: str, ctx: str) -> int:
        return 1

    await fn("question text", ctx="some context")
    recorded = mock_raw_cache.instances[-1].last_query
    assert recorded == semisweet.CacheQuery(query="question text", context="some context")


async def test_a_cached_none_result_is_served_as_a_hit(mock_raw_cache):
    calls = 0

    @cache(query="q")
    async def fn(q: str):
        nonlocal calls
        calls += 1
        return None

    assert await fn("x") is None
    assert await fn("x") is None
    assert calls == 1
