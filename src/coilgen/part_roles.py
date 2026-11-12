"""One vocabulary for the role each winding body plays.

Meshes, designs and support descriptions historically spelled the shield role
differently (``"shield"`` in ``designs`` and ``problem.SupportMesh``,
``"active_shield"`` in the pipeline). Everything that reads a winding role goes
through :func:`normalize_part_role`, so both spellings select the same body.
"""

from __future__ import annotations

PRIMARY = "primary"
ACTIVE_SHIELD = "active_shield"
OTHER = "other"

PART_ROLES = (PRIMARY, ACTIVE_SHIELD, OTHER)
PART_ROLE_LABELS = {PRIMARY: "Primary", ACTIVE_SHIELD: "Active shield", OTHER: "Other"}

_ALIASES = {
    "primary": PRIMARY,
    "main": PRIMARY,
    "active_shield": ACTIVE_SHIELD,
    "active shield": ACTIVE_SHIELD,
    "activeshield": ACTIVE_SHIELD,
    "shield": ACTIVE_SHIELD,
    "other": OTHER,
    "auxiliary": OTHER,
}


def normalize_part_role(value) -> str | None:
    """Return ``primary``, ``active_shield`` or ``other``; ``None`` when no winding role is stated."""
    if not isinstance(value, str):
        return None
    return _ALIASES.get(value.strip().lower().replace("-", "_"))


def normalize_part_roles(values) -> list[str | None] | None:
    if not isinstance(values, (list, tuple)):
        return None
    return [normalize_part_role(value) for value in values]


def part_role_label(value) -> str:
    """Human name for a role, or an empty string when none is recorded."""
    return PART_ROLE_LABELS.get(normalize_part_role(value), "")


def body_display_name(body_id: str, role=None) -> str:
    """``"Primary · part001_body0001"``; the identifier itself is never renamed."""
    label = part_role_label(role)
    return f"{label} · {body_id}" if label and body_id else body_id or label
