"""Value serialization for the object-aware cache.

A stored value is one tag byte followed by a payload. A pydantic ``BaseModel`` is
encoded as a portable, type-tagged JSON envelope so it rehydrates to the exact class;
every other object falls back to ``pickle``. Pydantic is optional: when it is not
installed, the model branch is unreachable and everything pickles.
"""

from __future__ import annotations

import importlib
import json
import pickle
from functools import cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import TypeGuard

    from pydantic import BaseModel

__all__ = ["dumps", "loads"]

_PICKLE = 0
_PYDANTIC = 1
_PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL


@cache
def _base_model() -> type[BaseModel] | None:
    try:
        from pydantic import BaseModel
    except ImportError:
        return None
    return BaseModel


def _is_base_model(value: object) -> TypeGuard[BaseModel]:
    base = _base_model()
    return base is not None and isinstance(value, base)


def _resolve(module: str, qualname: str) -> type[BaseModel]:
    obj: Any = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def dumps(value: object) -> bytes:
    """Serialize ``value`` to the cache wire format: a tag byte plus its payload."""
    if _is_base_model(value):
        envelope = json.dumps(
            {
                "module": type(value).__module__,
                "qualname": type(value).__qualname__,
                "json": value.model_dump_json(),
            }
        )
        return bytes([_PYDANTIC]) + envelope.encode("utf-8")
    return bytes([_PICKLE]) + pickle.dumps(value, protocol=_PICKLE_PROTOCOL)


def loads(data: bytes) -> object:
    """Deserialize ``data`` produced by :func:`dumps` back into a Python object."""
    tag, payload = data[0], data[1:]
    if tag == _PICKLE:
        return pickle.loads(payload)
    if tag == _PYDANTIC:
        envelope = json.loads(payload)
        model = _resolve(envelope["module"], envelope["qualname"])
        return model.model_validate_json(envelope["json"])
    raise ValueError(f"unknown serde tag byte: {tag}")
