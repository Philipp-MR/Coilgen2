"""The shielded cylinder pair as a support type of the shared settings form.

A primary cylinder, an active-shield cylinder and the closed suppression
cylinder share one axis (:class:`CylindricalGeometry`). The form shows them in
millimetres and degrees; Apply rebuilds the owned meshes with
:meth:`CylindricalGeometry.apply`. The Qt dialog and the browser form both use
the opening text format defined here.
"""

from __future__ import annotations

from dataclasses import replace
import math
import re

from .gradient_geometry import CylinderOpening, CylinderSurface, CylindricalGeometry

SHIELDED_PAIR_SOURCE = "shielded cylinder pair"
SHIELDED_PAIR_LABEL = "Shielded cylinder pair"
OPENING_SURFACES = ("primary", "shield", "both")
OPENINGS_HELP = (
    "One opening per entry, separated by ';': angle °, width °, axial centre mm, height mm and "
    "surface (primary, shield or both). Example: 90, 30, 0, 60, both. The angle is measured from "
    "+x toward +y; the axial centre from the middle of the cylinder."
)

# key, node, label, kind, unit, help
PAIR_FIELDS = (
    *(
        (f"pair_{role}_{name}", "geometry", f"{title} {label}", kind, unit, "")
        for role, title in (("primary", "Primary"), ("shield", "Shield"))
        for name, label, kind, unit in (
            ("radius_mm", "radius", "number", "mm"),
            ("length_mm", "length", "number", "mm"),
            ("angular", "azimuthal divisions", "integer", ""),
            ("axial", "axial divisions", "integer", ""),
        )
    ),
    ("pair_openings", "geometry", "Openings", "text", "", OPENINGS_HELP),
)
SUPPRESSION_SHAPE_FIELDS = (
    (
        "suppression_radius_mm",
        "target",
        "Suppression surface radius",
        "number",
        "mm",
        "A closed cylinder around both supports. The external field is limited on its surface.",
    ),
    ("suppression_length_mm", "target", "Suppression surface length", "number", "mm", ""),
)
PAIR_KEYS = frozenset(field[0] for field in (*PAIR_FIELDS, *SUPPRESSION_SHAPE_FIELDS))
SUPPRESSION_SHAPE_KEYS = frozenset(field[0] for field in SUPPRESSION_SHAPE_FIELDS)
# Config fields of the shielding target, shown with the suppression surface.
SHIELDING_CONFIG_KEYS = frozenset(
    {"suppression_rms_limit_uT", "suppression_field_component", "suppression_surface_weighting"}
)
SUPPRESSION_COMPONENTS = ("vector", "target", "bz", "b_normal", "magnitude")
SUPPRESSION_WEIGHTINGS = ("area", "uniform")
# Subheadings of the settings form: the first field of each group carries it.
FIELD_SECTIONS = {
    **{key: "Primary" for key, *_ in PAIR_FIELDS if key.startswith("pair_primary_")},
    **{key: "Active shield" for key, *_ in PAIR_FIELDS if key.startswith("pair_shield_")},
    "pair_openings": "Openings",
    **{key: "Shielding target" for key in (*SUPPRESSION_SHAPE_KEYS, *SHIELDING_CONFIG_KEYS)},
}

_CACHE: dict = {}


def project_geometry(project):
    """The project's verified cylindrical geometry, or None for any other support."""
    try:
        key = (str(project.project_root), project.fingerprint)
    except (AttributeError, TypeError, ValueError):
        return None
    if key not in _CACHE:
        try:
            geometry = CylindricalGeometry.from_project(project)
        except (ValueError, TypeError, KeyError, OSError):
            geometry = None
        if len(_CACHE) >= 16:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = geometry
    return _CACHE[key]


def is_shielded_pair(project) -> bool:
    geometry = project_geometry(project)
    return geometry is not None and geometry.shield is not None


def displayed_support_source(project) -> str:
    """The support type the form shows: the pair, or the configured source."""
    return SHIELDED_PAIR_SOURCE if is_shielded_pair(project) else str(project.config.coil_mesh_file)


def shielding_target_active(config) -> bool:
    """A suppression surface (owned mesh or generated) is part of the model."""
    if str(config.secondary_target_mesh_file).strip().lower() not in {"", "none"}:
        return True
    return any(
        str(getattr(config, name)).strip().lower() not in {"", "none"}
        for name in type(config).__dataclass_fields__
        if name.startswith("create_") and name.endswith("_suppression_surface")
    )


def _tidy(value):
    return round(value, 6)


def draft_geometry(project) -> CylindricalGeometry:
    """Current pair, or a starting pair grown from the current cylinder (else the template's sizes)."""
    current = project_geometry(project)
    if current is None:
        primary = CylinderSurface(0.08, 0.3)
    else:
        primary = current.primary
    shield = current.shield if current is not None else None
    suppression = current.suppression if current is not None else None
    if shield is None:
        radius = primary.radius_m * 1.375
        if suppression is not None:
            # Fit a new shield inside an existing suppression surface.
            inside = suppression.radius_m * math.cos(math.pi / suppression.angular_divisions)
            radius = min(radius, (primary.radius_m + inside) / 2)
        shield = CylinderSurface(
            _tidy(radius),
            primary.length_m,
            primary.angular_divisions,
            primary.axial_divisions,
            primary.center_m,
            primary.axis,
            primary.azimuth_rad,
        )
    if suppression is None:
        suppression = CylinderSurface(
            _tidy(max(primary.radius_m, shield.radius_m) * 15 / 11),
            _tidy(max(primary.length_m, shield.length_m) * 4 / 3),
            24,
            12,
            primary.center_m,
            primary.axis,
        )
    return CylindricalGeometry(primary, shield, suppression)


def format_openings(geometry) -> str:
    """``"90, 30, 0, 60, both; …"``; an opening on both surfaces is listed once."""
    if geometry is None:
        return ""
    shield = list(geometry.shield.openings) if geometry.shield is not None else []
    rows = []
    for opening in geometry.primary.openings:
        if opening in shield:
            shield.remove(opening)
            rows.append((opening, "both"))
        else:
            rows.append((opening, "primary"))
    rows.extend((opening, "shield") for opening in shield)
    return "; ".join(
        f"{math.degrees(opening.phi_center_rad):.6g}, {math.degrees(opening.phi_width_rad):.6g}, "
        f"{opening.z_center_m * 1e3:.6g}, {opening.z_height_m * 1e3:.6g}, {surface}"
        for opening, surface in rows
    )


def parse_openings(text, *, has_shield=True) -> dict:
    """Parse the opening text into ``{"primary": (...), "shield": (...)}``."""
    result = {"primary": [], "shield": []}
    entries = [entry.strip() for entry in re.split(r"[;\n]", str(text or "")) if entry.strip()]
    for index, entry in enumerate(entries, 1):
        parts = [part.strip() for part in entry.split(",")]
        if len(parts) not in (4, 5):
            raise ValueError(
                f"Opening {index}: enter angle °, width °, axial centre mm, height mm and surface, separated by commas."
            )
        try:
            phi, width, z, height = (float(part) for part in parts[:4])
        except ValueError as exc:
            raise ValueError(f"Opening {index}: the first four values must be numbers.") from exc
        surface = parts[4].lower() if len(parts) == 5 else ("both" if has_shield else "primary")
        if surface not in OPENING_SURFACES:
            raise ValueError(f"Opening {index}: the surface must be primary, shield or both.")
        if surface != "primary" and not has_shield:
            raise ValueError(f"Opening {index} is on the shield, but there is no shield.")
        try:
            opening = CylinderOpening(math.radians(phi), math.radians(width), z * 1e-3, height * 1e-3)
        except ValueError as exc:
            raise ValueError(f"Opening {index}: {exc}") from exc
        for role in ("primary", "shield") if surface == "both" else (surface,):
            result[role].append(opening)
    return {role: tuple(openings) for role, openings in result.items()}


def _surface_values(prefix, surface):
    return {
        f"{prefix}_radius_mm": surface.radius_m * 1e3,
        f"{prefix}_length_mm": surface.length_m * 1e3,
        f"{prefix}_angular": surface.angular_divisions,
        f"{prefix}_axial": surface.axial_divisions,
    }


def pair_values(project) -> dict:
    """Form values of every pair key, in millimetres."""
    geometry = draft_geometry(project)
    values = {
        **_surface_values("pair_primary", geometry.primary),
        **_surface_values("pair_shield", geometry.shield),
        "pair_openings": format_openings(geometry),
        "suppression_radius_mm": geometry.suppression.radius_m * 1e3,
        "suppression_length_mm": geometry.suppression.length_m * 1e3,
    }
    return values


def hidden_keys(project, source) -> set:
    """Pair and shielding fields that do not apply to this project and support type."""
    hidden = set()
    if source != SHIELDED_PAIR_SOURCE:
        hidden.update(key for key, *_ in PAIR_FIELDS)
    geometry = project_geometry(project)
    if geometry is None or geometry.suppression is None:
        hidden.update(SUPPRESSION_SHAPE_KEYS)
    if not shielding_target_active(project.config):
        hidden.update(SHIELDING_CONFIG_KEYS)
    return hidden


def form_values(project, values: dict) -> dict:
    """Fill the pair fields of a settings form and show the pair as the support type."""
    keys = PAIR_KEYS.intersection(values)
    if keys:
        try:
            pair = pair_values(project)
        except ValueError:
            pair = {}
        values.update({key: pair.get(key) for key in keys})
    if "coil_mesh_file" in values:
        values["coil_mesh_file"] = displayed_support_source(project)
    return values


def mark_sections(entries) -> list:
    """Give the first form entry of each pair or shielding group its subheading."""
    previous = None
    for entry in entries:
        section = FIELD_SECTIONS.get(entry["key"])
        entry["section"] = section if section and section != previous else ""
        previous = section
    return entries


def _number(values, shown, key, label, integer=False):
    """Edited number, or None when the field is unchanged (keeps exact stored values)."""
    if key not in values:
        return None
    try:
        number = float(values[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    if number == shown[key]:
        return None
    if integer:
        if not number.is_integer():
            raise ValueError(f"{label} must be a whole number.")
        return int(number)
    return number * 1e-3


def _edited_surface(surface, values, shown, prefix, title, openings=None):
    changes = {}
    for name, field, label, integer in (
        ("radius_mm", "radius_m", "radius", False),
        ("length_mm", "length_m", "length", False),
        ("angular", "angular_divisions", "azimuthal divisions", True),
        ("axial", "axial_divisions", "axial divisions", True),
    ):
        value = _number(values, shown, f"{prefix}_{name}", f"{title} {label}", integer)
        if value is not None:
            changes[field] = value
    if openings is not None:
        changes["openings"] = openings
    if not changes:
        return surface
    try:
        return CylinderSurface(**{**_fields(surface), **changes})
    except ValueError as exc:
        raise ValueError(f"{title}: {exc}") from exc


def _fields(surface):
    return {name: getattr(surface, name) for name in surface.__dataclass_fields__}


def pair_geometry(project, values) -> CylindricalGeometry:
    """The pair the form describes; unchanged fields keep their stored values exactly."""
    base = draft_geometry(project)
    shown = pair_values(project)
    openings = None
    if "pair_openings" in values and str(values["pair_openings"] or "").strip() != shown["pair_openings"]:
        openings = parse_openings(values["pair_openings"])
    primary = _edited_surface(base.primary, values, shown, "pair_primary", "Primary", openings and openings["primary"])
    shield = _edited_surface(base.shield, values, shown, "pair_shield", "Shield", openings and openings["shield"])
    suppression = _edited_surface(base.suppression, values, shown, "suppression", "Suppression surface")
    return CylindricalGeometry(primary, shield, suppression)


def apply_geometry(project, geometry, config_changes=None, *, expected_fingerprint=None):
    """Rebuild the owned meshes and commit related config edits (such as the RMS limit) as one revision."""
    updated = geometry.apply(project, expected_fingerprint=expected_fingerprint)
    if config_changes:
        updated = updated.update_inputs(config_changes)
    if updated is not project:
        updated = replace(updated, revision=project.revision + 1)
    return updated


def apply_pair_settings(project, values):
    """Split the pair fields off a settings form; rebuild the owned meshes when they changed.

    Returns ``(other_values, project)``. The pseudo source ``SHIELDED_PAIR_SOURCE``
    never reaches the configuration: choosing it builds the meshes instead.
    """
    remaining = {key: value for key, value in values.items() if key not in PAIR_KEYS}
    pair = {key: value for key, value in values.items() if key in PAIR_KEYS}
    choose_pair = remaining.get("coil_mesh_file") == SHIELDED_PAIR_SOURCE
    if choose_pair:
        remaining.pop("coil_mesh_file")
    elif "coil_mesh_file" in remaining and remaining["coil_mesh_file"] != displayed_support_source(project):
        # Another support type replaces the pair; its pair fields no longer apply.
        return remaining, project
    if not pair and not choose_pair:
        return remaining, project
    current = project_geometry(project)
    if not choose_pair and current is None:
        raise ValueError("These dimensions apply only to a cylindrical support built in CoilGen.")
    geometry = pair_geometry(project, pair)
    if not choose_pair:
        # Editing an existing geometry keeps its optional surfaces as they are.
        geometry = CylindricalGeometry(
            geometry.primary,
            geometry.shield if current.shield is not None else None,
            geometry.suppression if current.suppression is not None else None,
        )
        if geometry == current:
            return remaining, project
    return remaining, geometry.apply(project)


__all__ = [
    "FIELD_SECTIONS",
    "OPENINGS_HELP",
    "PAIR_FIELDS",
    "PAIR_KEYS",
    "SHIELDED_PAIR_LABEL",
    "SHIELDED_PAIR_SOURCE",
    "SHIELDING_CONFIG_KEYS",
    "SUPPRESSION_COMPONENTS",
    "SUPPRESSION_SHAPE_FIELDS",
    "SUPPRESSION_SHAPE_KEYS",
    "SUPPRESSION_WEIGHTINGS",
    "apply_geometry",
    "apply_pair_settings",
    "hidden_keys",
    "form_values",
    "mark_sections",
    "displayed_support_source",
    "draft_geometry",
    "format_openings",
    "is_shielded_pair",
    "pair_geometry",
    "pair_values",
    "parse_openings",
    "project_geometry",
    "shielding_target_active",
]
