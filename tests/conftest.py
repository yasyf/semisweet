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

from semisweet import (
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
