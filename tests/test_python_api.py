"""End-to-end tests for the pyo3 Python API against a real spawned daemon.

The offline validation tests (`ValueError` cases) need no daemon and always run.
The roundtrip test drives a real, lazily-spawned daemon configured with the local
BGE embedding, keyword entities, an in-memory vector index, and on-disk objects; it
runs whenever the BGE model is cached on disk.
"""

import os
import shutil
import signal
import tempfile
import time
from pathlib import Path

import pytest

from semisweet import CacheQuery, SemanticCache

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_CACHE = REPO_ROOT / ".fastembed_cache"
BGE_MODEL_DIR = MODEL_CACHE / "models--Xenova--bge-small-en-v1.5"

POLL_TIMEOUT = 20.0
POLL_INTERVAL = 0.1

needs_model = pytest.mark.skipif(
    not BGE_MODEL_DIR.exists(),
    reason=f"BGE model not cached at {BGE_MODEL_DIR}",
)


def _kill_daemon(pid_path: Path) -> None:
    try:
        pid = int(pid_path.read_text().strip())
    except (FileNotFoundError, ValueError):
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


@pytest.fixture
def runtime(monkeypatch):
    # A unique, short runtime dir per test: tests never share a daemon, and the
    # unix socket path stays well under the macOS ~104-byte sun_path cap.
    directory = Path(tempfile.mkdtemp(prefix="ssp"))
    monkeypatch.setenv("SEMISWEET_SOCKET", str(directory / "d.sock"))
    monkeypatch.setenv("SEMISWEET_LOCK", str(directory / "d.lock"))
    monkeypatch.setenv("SEMISWEET_LOG", str(directory / "d.log"))
    monkeypatch.setenv("SEMISWEET_IDLE_SECS", "30")
    monkeypatch.setenv("SEMISWEET_MODEL_CACHE", str(MODEL_CACHE))
    try:
        yield directory
    finally:
        _kill_daemon(directory / "d.pid")
        shutil.rmtree(directory, ignore_errors=True)


def _local_cache(namespace: str, runtime: Path) -> SemanticCache:
    return (
        SemanticCache.builder(namespace)
        .embedding_local()
        .entities_keyword(None)
        .vector_memory()
        .object_disk(str(runtime / "objects"))
        .build()
    )


def _poll_get(cache: SemanticCache, query: CacheQuery) -> bytes | None:
    deadline = time.monotonic() + POLL_TIMEOUT
    while time.monotonic() < deadline:
        value = cache.get(query)
        if value is not None:
            return value
        time.sleep(POLL_INTERVAL)
    return cache.get(query)


@needs_model
def test_set_then_get_returns_exact_bytes(runtime):
    cache = _local_cache("roundtrip", runtime)
    query = CacheQuery("what is the capital of france", set())

    assert cache.set(query, b"paris") is True
    # `set` is async: the daemon's write-behind queue drains in the background, so
    # poll `get` until the entry is retrievable, then assert the exact payload.
    assert _poll_get(cache, query) == b"paris"


@needs_model
def test_unrelated_query_misses(runtime):
    cache = _local_cache("miss", runtime)
    stored = CacheQuery("what is the capital of france", set())

    assert cache.set(stored, b"paris") is True
    assert _poll_get(cache, stored) == b"paris"

    assert cache.get(CacheQuery("how do tides work", set())) is None


@needs_model
def test_delete_removes_entry(runtime):
    cache = _local_cache("delete", runtime)
    query = CacheQuery("what is the capital of france", set())

    assert cache.set(query, b"paris") is True
    assert _poll_get(cache, query) == b"paris"

    assert cache.delete(query) is True
    assert cache.get(query) is None


@needs_model
def test_large_payload_offloads_and_roundtrips_exact_bytes(runtime):
    cache = _local_cache("largepayload", runtime)
    query = CacheQuery("summarize the quarterly earnings report", set())
    # 5 MiB exercises the on-disk object store and the 64 MiB IPC framing end to
    # end; the payload travels Python -> daemon -> object store -> back, intact.
    payload = os.urandom(5 * 1024 * 1024)

    assert cache.set(query, payload) is True
    got = _poll_get(cache, query)
    assert got is not None
    assert len(got) == 5 * 1024 * 1024
    assert got == payload


@needs_model
def test_context_assisted_hit_roundtrips(runtime):
    cache = _local_cache("context", runtime)
    query = CacheQuery(
        "what dose should the patient take",
        set(),
        "patient is currently on warfarin therapy",
    )

    assert cache.set(query, b"5mg daily") is True
    assert _poll_get(cache, query) == b"5mg daily"


@needs_model
def test_keys_filter_isolates_entries_for_same_query(runtime):
    cache = _local_cache("keys", runtime)
    text = "what is the patient's current medication"
    v1 = CacheQuery(text, {"v1"})
    v2 = CacheQuery(text, {"v2"})

    # Same query text, disjoint `keys` sets: the deterministic id keys on
    # query+keys, so these are two distinct entries the keys filter keeps apart.
    assert cache.set(v1, b"aspirin") is True
    assert cache.set(v2, b"ibuprofen") is True

    assert _poll_get(cache, v1) == b"aspirin"
    assert _poll_get(cache, v2) == b"ibuprofen"


@needs_model
def test_distinct_namespaces_isolate_entries(runtime):
    # Two caches share one daemon and object root but use different namespaces;
    # the same query+keys yields the same id, so only namespacing keeps them apart.
    cache_alpha = _local_cache("alpha", runtime)
    cache_beta = _local_cache("beta", runtime)
    query = CacheQuery("what is the capital of france", set())

    assert cache_alpha.set(query, b"paris") is True
    assert cache_beta.set(query, b"berlin") is True

    assert _poll_get(cache_alpha, query) == b"paris"
    assert _poll_get(cache_beta, query) == b"berlin"


def test_builder_floor_above_base_raises():
    builder = (
        SemanticCache.builder("ns")
        .embedding_local()
        .entities_keyword(None)
        .vector_memory()
        .object_disk(None)
    )
    # floor > base is rejected eagerly, before any daemon is spawned.
    with pytest.raises(ValueError):
        builder.threshold(0.90, 0.95, 0.04, 10, True, "ignore")


def test_empty_query_raises():
    with pytest.raises(ValueError):
        CacheQuery("", set())
