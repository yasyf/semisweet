"""End-to-end tests for the declarative pyo3 API against a real spawned daemon.

The construction/validation tests need no daemon and always run. The roundtrip tests
(``needs_model``) drive a real, lazily-spawned daemon on the fully-local stack — local
BGE + keyword entities + in-memory vectors + on-disk objects — and run whenever the BGE
model is cached. ``set`` is read-after-write: a ``get`` straight after a ``set`` returns
the value with no polling.
"""

import os

import pytest

from semisweet import (
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

# --- construction & validation (no daemon) ---

OPTIONAL_ONLY_BACKENDS = [
    LocalEmbedding,
    VoyageEmbedding,
    KeywordEntities,
    GlinerEntities,
    MemoryVectors,
    TurbopufferVectors,
    DiskStorage,
    S3Storage,
    Scoring,
]


def test_empty_query_raises():
    with pytest.raises(ValueError):
        CacheQuery(query="")


def test_cache_query_rejects_positional_args():
    # The whole surface is keyword-only; a positional call must not bind silently.
    with pytest.raises(TypeError):
        CacheQuery("what is the capital of france")


def test_scoring_floor_above_base_raises():
    # floor > base is rejected eagerly in the Scoring constructor, no daemon involved.
    with pytest.raises(ValueError):
        Scoring(base=0.90, floor=0.95)


@pytest.mark.parametrize(
    "backend", OPTIONAL_ONLY_BACKENDS, ids=[b.__name__ for b in OPTIONAL_ONLY_BACKENDS]
)
def test_backend_constructs_with_every_kwarg_optional(backend):
    # Every backend class is a thin, all-optional config object; a missing-but-required
    # value (e.g. an S3 bucket) is resolved/enforced daemon-side when the cache is built,
    # not at construction.
    backend()


# --- roundtrip (needs the BGE model) ---


@pytest.mark.needs_model
def test_set_then_get_is_read_after_write(make_cache):
    cache = make_cache("roundtrip")
    query = CacheQuery(query="what is the capital of france")

    assert cache.set(query, b"paris") is True
    # No polling: the value is served from the in-memory pending shadow immediately.
    assert cache.get(query) == b"paris"


@pytest.mark.needs_model
def test_unrelated_query_misses(make_cache):
    cache = make_cache("miss")
    stored = CacheQuery(query="what is the capital of france")

    assert cache.set(stored, b"paris") is True
    assert cache.get(stored) == b"paris"
    assert cache.get(CacheQuery(query="how do tides work")) is None


@pytest.mark.needs_model
def test_delete_removes_entry(make_cache):
    cache = make_cache("delete")
    query = CacheQuery(query="what is the capital of france")

    assert cache.set(query, b"paris") is True
    assert cache.get(query) == b"paris"

    assert cache.delete(query) is True
    assert cache.get(query) is None


@pytest.mark.needs_model
def test_large_payload_roundtrips_exact_bytes(make_cache):
    cache = make_cache("largepayload")
    query = CacheQuery(query="summarize the quarterly earnings report")
    # 5 MiB exercises the on-disk object store and the 64 MiB IPC framing end to end.
    payload = os.urandom(5 * 1024 * 1024)

    assert cache.set(query, payload) is True
    got = cache.get(query)
    assert got is not None
    assert len(got) == 5 * 1024 * 1024
    assert got == payload


@pytest.mark.needs_model
def test_context_assisted_hit_roundtrips(make_cache):
    cache = make_cache("context")
    query = CacheQuery(
        query="what dose should the patient take",
        context="patient is currently on warfarin therapy",
    )

    assert cache.set(query, b"5mg daily") is True
    assert cache.get(query) == b"5mg daily"


@pytest.mark.needs_model
def test_keys_filter_isolates_entries_for_same_query(make_cache):
    cache = make_cache("keys")
    text = "what is the patient's current medication"
    v1 = CacheQuery(query=text, keys={"v1"})
    v2 = CacheQuery(query=text, keys={"v2"})

    # Same query text, disjoint `keys`: the deterministic id keys on query+keys, so
    # these are two distinct entries the keys filter keeps apart.
    assert cache.set(v1, b"aspirin") is True
    assert cache.set(v2, b"ibuprofen") is True

    assert cache.get(v1) == b"aspirin"
    assert cache.get(v2) == b"ibuprofen"


@pytest.mark.needs_model
def test_distinct_namespaces_isolate_entries(make_cache):
    # Two caches share one daemon and object root but use different namespaces; the
    # same query+keys yields the same id, so only namespacing keeps them apart.
    cache_alpha = make_cache("alpha")
    cache_beta = make_cache("beta")
    query = CacheQuery(query="what is the capital of france")

    assert cache_alpha.set(query, b"paris") is True
    assert cache_beta.set(query, b"berlin") is True

    assert cache_alpha.get(query) == b"paris"
    assert cache_beta.get(query) == b"berlin"
