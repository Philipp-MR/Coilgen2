"""Ask before a new support type throws away bodies the user set up.

Both clients call :func:`support_switch_prompt` before they apply a new support
source. It decides whether to ask, what to say and which shielding settings a
user may clear when they leave a shielded support. Generated supports keep their
parameters in the configuration, so switching away from them needs no question.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..part_roles import ACTIVE_SHIELD, normalize_part_roles, part_role_label
from ..persistence import json_value
from .cylinder_pair import SHIELDED_PAIR_SOURCE, displayed_support_source, is_shielded_pair
from .support_parameters import SUPPORT_EDITORS

IMPORTED_SUPPORT = "imported STL"
_NEW_SUPPORT_TEXT = {
    "create cylinder mesh": "1 cylinder",
    "create bi-planary mesh": "2 planes",
    "create planary mesh": "1 plane",
    "create conical frustum mesh": "1 conical frustum",
    SHIELDED_PAIR_SOURCE: "a shielded cylinder pair",
    IMPORTED_SUPPORT: "the imported STL",
}
CLEAR_SHIELDING_LABEL = "Also clear the shielding target (suppression surface, field limits and per-support routing)."


@dataclass(frozen=True)
class SupportSwitch:
    """What to ask before replacing the current support."""

    title: str
    message: str
    clear_changes: dict = field(default_factory=dict)
    clear_label: str = ""
    confirm_label: str = "Replace support"

    def to_state(self) -> dict:
        return {
            "title": self.title,
            "message": self.message,
            "can_clear": bool(self.clear_changes),
            "clear_label": self.clear_label,
            "confirm_label": self.confirm_label,
        }


def shielding_clear_changes(project) -> dict:
    """Config edits that remove the shielding target: suppression surface and limits, per-support routing."""
    from ..config import CoilGenConfig

    defaults = CoilGenConfig()
    keys = [
        "secondary_target_mesh_file",
        *(name for name in CoilGenConfig.__dataclass_fields__ if name.startswith("suppression_")),
        "routing_circuit_scope",
    ]
    values = project.values
    return {
        key: getattr(defaults, key)
        for key in keys
        if key in values and json_value(values[key]) != json_value(getattr(defaults, key))
    }


def _new_support_text(source) -> str:
    if source in _NEW_SUPPORT_TEXT:
        return _NEW_SUPPORT_TEXT[source]
    editor = next((editor for editor in SUPPORT_EDITORS if editor.source == source), None)
    return f"the {editor.label.lower()} support" if editor else "the new support"


def _generated(source) -> bool:
    return str(source).strip().lower().startswith("create ")


def _plural(count, word):
    return f"{count} {word}" if count == 1 else f"{count} {word.replace('body', 'bodies')}"


def support_switch_prompt(project, new_source, *, new_text=None) -> SupportSwitch | None:
    """Return the question to ask before ``new_source`` replaces the support, or None.

    Ask when the current support is a project-owned mesh with two or more bodies
    or with recorded body roles. When it is shielded and the new support is not,
    also offer to clear the suppression settings. Never raises: a support that
    cannot be read is not described.
    """
    if project is None or not new_source:
        return None
    config = project.config
    current = str(config.coil_mesh_file)
    if new_source in (current, displayed_support_source(project)) or _generated(current):
        return None
    if current.strip().lower() in {"", "none"}:
        return None
    try:
        from ..geometry import load_mesh_from_config
        from .support_import import support_bodies

        mesh = load_mesh_from_config(config, project.project_root)
        count = len(support_bodies(mesh))
    except (ValueError, TypeError, KeyError, OSError):
        return None
    roles = normalize_part_roles(mesh.metadata.get("part_roles")) or []
    if count < 2 and not any(roles):
        return None
    labels = [part_role_label(role) for role in roles if part_role_label(role)]
    described = _plural(count, "body") + (f" ({', '.join(labels)})" if labels else "")
    message = f"This replaces {described} with {new_text or _new_support_text(new_source)}. Undo restores it."
    shielded = (
        ACTIVE_SHIELD in roles
        or mesh.metadata.get("geometry_type") == "active_shielded_cylinder"
        or is_shielded_pair(project)
    )
    clear = {}
    if shielded and new_source != SHIELDED_PAIR_SOURCE and new_source != IMPORTED_SUPPORT:
        clear = shielding_clear_changes(project)
    return SupportSwitch(
        title="Replace the support?",
        message=message,
        clear_changes=clear,
        clear_label=CLEAR_SHIELDING_LABEL if clear else "",
    )


__all__ = [
    "CLEAR_SHIELDING_LABEL",
    "IMPORTED_SUPPORT",
    "SupportSwitch",
    "shielding_clear_changes",
    "support_switch_prompt",
]
