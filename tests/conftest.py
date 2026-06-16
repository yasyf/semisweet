"""Shared fixtures and the ``needs_model`` gate for the semisweet test suite.

Construction/validation tests need no daemon and always run. The roundtrip tests
drive a real, lazily-spawned daemon; they are marked ``needs_model`` and self-skip
unless the local BGE model is cached on disk (the CI model job pre-downloads it).
"""

import os
import shutil
import signal
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from semisweet.core import (
    DiskStorage,
    KeywordEntities,
    LocalEmbedding,
    MemoryVectors,
    SemanticCache,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_CACHE = REPO_ROOT / ".fastembed_cache"


def _model_is_cached() -> bool:
    # fastembed caches BGE-small under `models--<org>--bge-small-*`; match any org so
    # the gate tracks the CI prefetch step without pinning the upstream repo name.
    return MODEL_CACHE.is_dir() and any(MODEL_CACHE.glob("models--*bge-small*"))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "needs_model: requires the local BGE model cached on disk"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if _model_is_cached():
        return
    skip = pytest.mark.skip(reason=f"BGE model not cached at {MODEL_CACHE}")
    for item in items:
        if "needs_model" in item.keywords:
            item.add_marker(skip)


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
def runtime(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    # A unique, short runtime dir per test: tests never share a daemon, and the unix
    # socket path stays well under the macOS ~104-byte sun_path cap.
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


CacheFactory = Callable[..., SemanticCache]


@pytest.fixture
def make_cache(runtime: Path) -> CacheFactory:
    """Build a fully-local cache in a fresh namespace, overridable per axis.

    Defaults to local BGE + keyword entities + in-memory vectors + on-disk objects,
    all under the test's runtime dir. Pass ``embedding=``/``entities=``/``vectors=``/
    ``storage=``/``scoring=`` to swap an axis for a combo test.
    """

    def _make(namespace: str, **overrides: object) -> SemanticCache:
        params: dict[str, object] = {
            "namespace": namespace,
            "embedding": LocalEmbedding(),
            "entities": KeywordEntities(),
            "vectors": MemoryVectors(),
            "storage": DiskStorage(root=str(runtime / "objects")),
        }
        params.update(overrides)
        return SemanticCache(**params)

    return _make


@pytest.fixture(autouse=True)
def _clear_decorator_caches() -> Iterator[None]:
    # The @cache registry is module-global; keep it empty around every test so a cache
    # bound to a previous test's daemon/namespace is never reused.
    import semisweet.decorator as decorator

    decorator._caches.clear()
    yield
    decorator._caches.clear()


@pytest.fixture
def mock_raw_cache(monkeypatch: pytest.MonkeyPatch) -> type:
    """Replace the raw ``core.SemanticCache`` with an in-memory fake: no daemon, no model.

    The object layer's serde and the real, hashable ``CacheQuery`` stay exercised; only the
    daemon round-trip is faked, keyed by the query value.
    """

    class FakeRaw:
        instances: list["FakeRaw"] = []

        def __init__(self, *, namespace: str, **_: object) -> None:
            self.namespace = namespace
            self.store: dict[object, bytes] = {}
            self.last_query: object = None
            FakeRaw.instances.append(self)

        async def get(self, query: object) -> bytes | None:
            return self.store.get(query)

        async def set(self, query: object, value: bytes) -> bool:
            self.last_query = query
            self.store[query] = value
            return True

        async def delete(self, query: object) -> bool:
            return self.store.pop(query, None) is not None

    monkeypatch.setattr("semisweet.objectcache._RawSemanticCache", FakeRaw)
    return FakeRaw
