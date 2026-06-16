"""Serialization for the object cache: the pydantic JSON envelope and the pickle fallback.

No daemon, no model. ``_Sample`` is module-level so its ``module``/``qualname`` descriptor
resolves back to this exact class on load.
"""

import json
import pickle

import pytest
from pydantic import BaseModel

from semisweet.serde import dumps, loads


class _Sample(BaseModel):
    x: int
    y: str


def test_pickle_path_tags_zero_and_roundtrips_plain_objects():
    value = {"a": 1, "b": [2, 3], "c": "three"}
    blob = dumps(value)
    assert blob[0] == 0
    assert loads(blob) == value


def test_pydantic_path_tags_one_with_a_type_descriptor_envelope():
    blob = dumps(_Sample(x=5, y="hi"))
    assert blob[0] == 1
    envelope = json.loads(blob[1:])
    assert envelope["module"] == _Sample.__module__
    assert envelope["qualname"] == "_Sample"


def test_pydantic_value_rehydrates_to_the_same_class():
    model = _Sample(x=5, y="hi")
    restored = loads(dumps(model))
    assert type(restored) is _Sample
    assert restored == model


def test_unknown_tag_byte_raises_value_error():
    with pytest.raises(ValueError):
        loads(bytes([7]) + b"junk")


def test_pydantic_absent_falls_back_to_pickle(monkeypatch):
    # With pydantic unavailable the model branch is unreachable; the model still
    # round-trips, but through pickle (tag 0), not the JSON envelope.
    monkeypatch.setattr("semisweet.serde._base_model", lambda: None)
    model = _Sample(x=1, y="z")
    blob = dumps(model)
    assert blob[0] == 0
    assert loads(blob) == model


def test_unpicklable_value_fails_loud():
    with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
        dumps(lambda: 1)


def test_unresolvable_descriptor_raises_on_load():
    blob = bytes([1]) + json.dumps(
        {"module": "no_such_module_xyz", "qualname": "Nope", "json": "{}"}
    ).encode("utf-8")
    with pytest.raises((ModuleNotFoundError, ImportError)):
        loads(blob)
