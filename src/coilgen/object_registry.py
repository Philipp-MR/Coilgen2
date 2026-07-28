from __future__ import annotations

from uuid import uuid4

from .models import Mesh, TargetField


_MEMORY_PREFIX = "memory"
_MESHES: dict[str, Mesh] = {}
_TARGET_FIELDS: dict[str, TargetField] = {}


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

    memory_key = _make_key("mesh", key)
    _MESHES[memory_key] = mesh.copy()
    return memory_key


def get_in_memory_mesh(key: str) -> Mesh:
    try:
        return _MESHES[str(key).strip()].copy()
    except KeyError as exc:
        raise KeyError(f"No in-memory mesh is registered under {key!r}.") from exc


def register_in_memory_target_field(target_field: TargetField, *, key: str | None = None) -> str:
    """Register a target field for same-process use by ``CoilGenConfig``."""

    memory_key = _make_key("target", key)
    _TARGET_FIELDS[memory_key] = TargetField(
        coords=target_field.coords.copy(),
        b=target_field.b.copy(),
        weights=target_field.weights.copy(),
        suppressed=target_field.suppressed.copy(),
        target_field_group_inds=target_field.target_field_group_inds.copy(),
        target_gradient_dbdxyz=None
        if target_field.target_gradient_dbdxyz is None
        else target_field.target_gradient_dbdxyz.copy(),
        suppression_normals=None
        if target_field.suppression_normals is None
        else target_field.suppression_normals.copy(),
        suppression_area_weights=None
        if target_field.suppression_area_weights is None
        else target_field.suppression_area_weights.copy(),
        target_diagnostics=dict(target_field.target_diagnostics),
    )
    return memory_key


def get_in_memory_target_field(key: str) -> TargetField:
    try:
        target = _TARGET_FIELDS[str(key).strip()]
    except KeyError as exc:
        raise KeyError(f"No in-memory target field is registered under {key!r}.") from exc
    return TargetField(
        coords=target.coords.copy(),
        b=target.b.copy(),
        weights=target.weights.copy(),
        suppressed=target.suppressed.copy(),
        target_field_group_inds=target.target_field_group_inds.copy(),
        target_gradient_dbdxyz=None if target.target_gradient_dbdxyz is None else target.target_gradient_dbdxyz.copy(),
        suppression_normals=None if target.suppression_normals is None else target.suppression_normals.copy(),
        suppression_area_weights=None
        if target.suppression_area_weights is None
        else target.suppression_area_weights.copy(),
        target_diagnostics=dict(target.target_diagnostics),
    )


def clear_in_memory_objects() -> None:
    """Clear meshes and target fields registered for same-process configs."""

    _MESHES.clear()
    _TARGET_FIELDS.clear()
