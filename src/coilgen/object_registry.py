from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from uuid import uuid4
from weakref import WeakValueDictionary

from .models import Mesh, TargetField


_MEMORY_PREFIX = "memory"
_MESHES: dict[str, Mesh] = {}
_TARGET_FIELDS: dict[str, TargetField] = {}
_OWNED_INPUTS: WeakValueDictionary[str, _InputOwner] = WeakValueDictionary()


@dataclass
class _InputOwner:
    value: Mesh | TargetField


class _OwnedInputKey(str):
    """A config keeps its input alive; copies share the private snapshot.

    Keys are same-process references, just like explicit registry entries.
    Projects capture the referenced data before crossing process boundaries.
    """

    def __new__(cls, key: str, owner: _InputOwner):
        instance = super().__new__(cls, key)
        instance._owner = _OWNED_INPUTS.get(str(instance), owner)
        _OWNED_INPUTS[str(instance)] = instance._owner
        return instance

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return _OwnedInputKey, (str(self), self._owner)


def own_in_memory_mesh(mesh: Mesh) -> str:
    """Create an isolated mesh reference whose lifetime follows its config."""
    return _OwnedInputKey(_make_key("mesh"), _InputOwner(deepcopy(mesh)))


def own_in_memory_target_field(target: TargetField) -> str:
    """Create an isolated target reference whose lifetime follows its config."""
    return _OwnedInputKey(_make_key("target"), _InputOwner(deepcopy(target)))


def _registered_input(key: str, registry: dict, expected_type: type):
    clean = str(key).strip()
    owner = _OWNED_INPUTS.get(clean)
    value = owner.value if owner is not None else registry[clean]
    if not isinstance(value, expected_type):
        raise KeyError(clean)
    return deepcopy(value)


def _register(value, kind: str, key: str | None, registry: dict) -> str:
    memory_key = _make_key(kind, key)
    if memory_key in _OWNED_INPUTS:
        raise ValueError("Cannot replace an input owned by another configuration.")
    registry[memory_key] = deepcopy(value)
    return memory_key


def _make_key(kind: str, key: str | None = None) -> str:
    clean = str(key).strip() if key is not None else ""
    if clean.startswith(f"{_MEMORY_PREFIX}:{kind}:"):
        return clean
    suffix = clean or uuid4().hex
    return f"{_MEMORY_PREFIX}:{kind}:{suffix}"


def is_in_memory_mesh_key(value: object) -> bool:
    return str(value).strip().startswith(f"{_MEMORY_PREFIX}:mesh:")


def is_in_memory_target_key(value: object) -> bool:
    return str(value).strip().startswith(f"{_MEMORY_PREFIX}:target:")


def register_in_memory_mesh(mesh: Mesh, *, key: str | None = None) -> str:
    """Register a mesh for same-process use by ``CoilGenConfig``."""

    return _register(mesh, "mesh", key, _MESHES)


def get_in_memory_mesh(key: str) -> Mesh:
    try:
        return _registered_input(key, _MESHES, Mesh)
    except KeyError as exc:
        raise KeyError(f"No in-memory mesh is registered under {key!r}.") from exc


def register_in_memory_target_field(target_field: TargetField, *, key: str | None = None) -> str:
    """Register a target field for same-process use by ``CoilGenConfig``."""

    return _register(target_field, "target", key, _TARGET_FIELDS)


def get_in_memory_target_field(key: str) -> TargetField:
    try:
        return _registered_input(key, _TARGET_FIELDS, TargetField)
    except KeyError as exc:
        raise KeyError(f"No in-memory target field is registered under {key!r}.") from exc


def clear_in_memory_objects() -> None:
    """Clear meshes and target fields registered for same-process configs."""

    _MESHES.clear()
    _TARGET_FIELDS.clear()
    _OWNED_INPUTS.clear()
