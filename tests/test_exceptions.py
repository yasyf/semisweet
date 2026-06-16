"""The multi-inheritance exception hierarchy and its dual catchability.

These run with no daemon and no model: the hierarchy is wired at module import,
and an empty-query ``CacheQuery`` fails eagerly in the constructor through the
crate's ``Error -> PyErr`` (``ConfigError``) path. Every semisweet error stays
catchable both as ``SemisweetError`` and as the matching builtin.
"""

import pytest

import semisweet
from semisweet import (
    BackendError,
    ConfigError,
    DaemonError,
    NamespaceError,
    SemisweetError,
)


def test_semisweet_error_is_exception_but_not_value_error():
    assert issubclass(SemisweetError, Exception)
    assert not issubclass(SemisweetError, ValueError)


def test_config_error_is_semisweet_error_and_value_error():
    assert issubclass(ConfigError, SemisweetError)
    assert issubclass(ConfigError, ValueError)


def test_namespace_error_is_semisweet_error_and_key_error():
    assert issubclass(NamespaceError, SemisweetError)
    assert issubclass(NamespaceError, KeyError)


def test_backend_error_is_semisweet_error_and_runtime_error():
    assert issubclass(BackendError, SemisweetError)
    assert issubclass(BackendError, RuntimeError)


def test_daemon_error_is_semisweet_error_and_runtime_error():
    assert issubclass(DaemonError, SemisweetError)
    assert issubclass(DaemonError, RuntimeError)


def test_empty_query_raises_config_error():
    # QueryText::new("") -> Error::EmptyQuery -> ConfigError at construction time.
    with pytest.raises(ConfigError):
        semisweet.CacheQuery(query="")


def test_empty_query_is_catchable_as_builtin_value_error():
    # ConfigError multiply-inherits ValueError, so the same failure is catchable
    # by callers that only know the builtin.
    with pytest.raises(ValueError):
        semisweet.CacheQuery(query="")


def test_empty_query_is_catchable_as_semisweet_error():
    with pytest.raises(SemisweetError):
        semisweet.CacheQuery(query="")
