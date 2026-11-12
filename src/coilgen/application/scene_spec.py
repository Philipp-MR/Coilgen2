"""Renderer-neutral scene pieces shared by the desktop and browser workbenches.

Both clients list the same datasets, build the same model preview and draw
the same B0 marker from these functions; the browser additionally renders
:func:`scene_layers`. Availability comes from :mod:`coilgen.application.evidence`,
so neither client can show data the other would reject.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..field_components import field_component_values, normalized_b0_direction
from .evidence import (
    FIELD_KEYS,
    PATH_PREFIXES,
    WIRE_FIELDS,
    field_points_name,
    paired_field_name,
    path_array_names,
    print_former_array_names,
    resolve_dataset,
)
from .field_presentation import paired_color_limits

B0_INDICATOR_COLOR = "#c2410c"
SUPPORT_COLOR = "#79949d"
WINDING_COLOR = "#205a75"
GROOVED_SUPPORT_COLOR = "#c8b48a"  # printable former with conductor grooves
FIELD_CMAP = "coolwarm"
DENSITY_CMAP = "viridis"

# Plot menu of both workbenches: (group, ((label, dataset key), ...)).
DATASET_GROUPS = (
    ("CURRENT MODEL", (("Model: support", "support"), ("Model: support + target field", "preview_target"))),
    (
        "SAVED RUN · GEOMETRY",
        (
            ("Stream function [A]", "stream_function"),
            ("Surface current density |K| [A/m]", "surface_current_density"),
            ("Contours before routing", "contours"),
            ("Connected routes", "routes"),
            ("Evaluated winding field paths", "evaluated_winding"),
            ("Pipeline conductor centerlines", "fabrication"),
            ("Pipeline swept conductor", "swept"),
            ("Grooved print former", "print_former"),
        ),
    ),
    (
        "SAVED RUN · FIELD MAPS",
        (
            ("Target field", "target"),
            ("Stream function field", "sheet"),
            ("Independent stream function field validation", "validation"),
            ("Contour wire field", "contour_field"),
            ("Contour residual (field − target)", "contour_residual"),
            ("Evaluated winding field", "routed_field"),
            ("Winding residual (field − target)", "routed_residual"),
        ),
    ),
    (
        "FABRICATION REVIEW",
        (
            ("Review conductor centerlines", "review_conductor"),
            ("Review electrical field paths", "review_electrical"),
            ("Review field", "fabricated"),
            ("Independent winding validation", "winding_validation"),
            ("Independent winding residual", "winding_validation_residual"),
        ),
    ),
)
DATASET_LABELS = {key: label for _group, entries in DATASET_GROUPS for label, key in entries}

# For legacy or partial records, prefer the most developed saved geometry,
# then field maps. Both workbenches use this order when opening a result.
PREFERRED_RUN_PLOTS = (
    "review_conductor",
    "swept",
    "fabrication",
    "evaluated_winding",
    "routes",
    "contours",
    "stream_function",
    "target",
    "sheet",
    "validation",
    "routed_field",
    "contour_field",
    "fabricated",
    "review_electrical",
    "surface_current_density",
    "winding_validation",
    "winding_validation_residual",
    "routed_residual",
    "contour_residual",
    "print_former",
)


# Field maps are shown in µT: low-field gradients and residuals are µT-scale per ampere.
FIELD_DISPLAY_UNIT = "µT"
FIELD_DISPLAY_SCALE = 1e6
B0_LABEL_OFFSET = 0.82  # label centre along B0, in arrow units


def b0_indicator_mesh(direction, pv, *, label=True):
    """Unit-size arrow along B0, optionally with a 3D "B0" label at its tip.

    The desktop draws the arrow as a camera-synchronised corner marker with a
    screen-aligned caption (``label=False``). The browser identifies it with
    the orange direction chip. An optional mesh label faces the default
    isometric camera for other renderers.
    """
    axis = normalized_b0_direction(direction)
    arrow = pv.Arrow(start=-0.55 * axis, direction=axis, scale=1.1, tip_length=0.3, tip_radius=0.12, shaft_radius=0.045)
    if not label:
        return arrow
    text = pv.Text3D("B0", depth=0.04)
    points = np.asarray(text.points, dtype=float)
    points = (points - points.mean(axis=0)) * (0.32 / max(float(np.ptp(points[:, 0])), 1e-12))
    # Text x/y/z -> camera right/up/towards-viewer for the (1, 1, 1) view.
    toward = np.ones(3) / np.sqrt(3.0)
    up = np.array([0.0, 0.0, 1.0]) - toward / np.sqrt(3.0)
    up /= np.linalg.norm(up)
    right = np.cross(up, toward)
    text.points = points @ np.vstack((right, up, toward)) + B0_LABEL_OFFSET * axis
    return arrow.merge(text)


def model_preview(config, root, requirements=None):
    """Support mesh and target samples of the current model, without a solve.

    Returns ``(mesh, arrays, message, diagnostics)``; rendering is left to the
    client. This is the desktop viewport's preview, shared with the browser.
    """
    from ..geometry import load_mesh_from_config, load_optional_target_mesh, load_secondary_target_mesh
    from ..target import define_target_field
    from .diagnostics import preview_diagnostics

    if str(config.coil_mesh_file).strip().lower() in {"", "none"}:
        message = "Choose a support surface to preview the model."
        return None, {}, message, preview_diagnostics(config, None, message=message)
    mesh = load_mesh_from_config(config, root)
    arrays, message, target = {}, "", None
    imported_target = str(config.target_field_definition_file).strip().lower() not in {"", "none"}
    if requirements is not None:
        from .research_requirements import requirements_target

        target = requirements_target(requirements)
        arrays = {"target_points_m": target.coords, "target_field_T": target.b, "target_suppressed": target.suppressed}
        message = f"{len(requirements['regions'])} named regions · registered desired vectors in SI · explicit regional bounds"
    elif not imported_target and str(config.field_shape_function).strip().lower() in {"", "none"}:
        message = "Choose a target field to preview it on this support."
    elif config.target_region_resolution > 40 and config.target_field_definition_file == "none":
        message = "Target preview omitted above resolution 40; computations retain the full grid."
    else:
        try:
            target_mesh = load_optional_target_mesh(config.target_mesh_file, config, root)
            secondary = load_secondary_target_mesh(config, root, coil_meshes=[mesh])
            target = define_target_field([mesh], target_mesh, secondary, config)
            arrays = {
                "target_points_m": target.coords,
                "target_field_T": target.b,
                "target_suppressed": target.suppressed,
            }
        except Exception as exc:
            # A failed target should not erase a valid current-carrying support
            # or substitute samples from the previous model or saved result.
            message = f"Target preview unavailable: {type(exc).__name__}: {exc}"
    arrays.update(former_preview_arrays(config, mesh))
    return mesh, arrays, message, preview_diagnostics(config, mesh, target, message=message)


def former_preview_arrays(config, mesh):
    """Thickened support (without grooves) shown before computing a 3D-print former.

    The grooves need the routed conductor; the solid side of the preview
    follows the support normal and is confirmed against the conductor frame
    when the former is built.
    """
    mode = str(getattr(config, "support_body_mode", "off"))
    if mode == "off" or mesh is None:
        return {}
    try:
        from ..support_body import thickened_support_mesh

        slab, _ = thickened_support_mesh(mesh, float(config.support_body_thickness), -1.0 if mode == "inside" else 1.0)
    except Exception:  # noqa: BLE001 - a preview must never block the model view
        return {}
    return {"former_preview_vertices_m": slab.vertices, "former_preview_faces": slab.faces}


@dataclass(frozen=True)
class SceneLayer:
    """One drawable item in SI coordinates."""

    name: str
    kind: str  # "surface", "lines" or "points"
    points: np.ndarray
    faces: np.ndarray | None = None
    lines: tuple[np.ndarray, ...] = ()
    scalars: np.ndarray | None = None
    scalar_label: str = ""
    color: str = SUPPORT_COLOR
    opacity: float = 1.0
    symmetric: bool = False
    extra: dict = field(default_factory=dict)


def _support_layer(arrays, opacity=0.15, scalars=None, label=""):
    return SceneLayer(
        "support",
        "surface",
        np.asarray(arrays["vertices_m"], dtype=float),
        faces=np.asarray(arrays["faces"], dtype=np.int64),
        scalars=scalars,
        scalar_label=label,
        opacity=opacity if scalars is None else 1.0,
    )


def scene_layers(arrays, key, *, component="Bz", b0_direction=(0.0, 0.0, 1.0), original=False, show_target=False):
    """Layers for one dataset key, or ``[]`` with the reason when unavailable.

    Returns ``(layers, reason)``. Field maps use the selected component;
    residual maps show field minus target, symmetric around zero.
    """
    capability = resolve_dataset(arrays, key, original=original)
    if not capability.available:
        return [], capability.reason
    has_support = resolve_dataset(arrays, "support").available
    layers = []
    if key == "support":
        former = {"former_preview_vertices_m", "former_preview_faces"} <= arrays.keys()
        # With a former preview, the thin support recedes so the printed wall reads clearly.
        layers.append(_support_layer(arrays, opacity=0.3 if former else 0.72))
        if former:
            layers.append(
                SceneLayer(
                    "former_preview",
                    "surface",
                    np.asarray(arrays["former_preview_vertices_m"], dtype=float),
                    faces=np.asarray(arrays["former_preview_faces"], dtype=np.int64),
                    color=GROOVED_SUPPORT_COLOR,
                    opacity=0.85,
                    extra={"role": "former_preview"},
                )
            )
    elif key == "stream_function":
        values = np.asarray(arrays["stream_function_A"], dtype=float).reshape(-1)
        layers.append(_support_layer(arrays, scalars=values, label="Stream function [A]"))
    elif key == "surface_current_density":
        density = np.linalg.norm(np.asarray(arrays["sheet_current_density_A_per_m"], dtype=float), axis=1)
        if has_support:
            layers.append(_support_layer(arrays))
        layers.append(
            SceneLayer(
                "current_density",
                "points",
                np.asarray(arrays["sheet_current_density_points_m"], dtype=float),
                scalars=density,
                scalar_label="|K| [A/m]",
            )
        )
    elif key in PATH_PREFIXES:
        names = path_array_names(arrays, key)
        if has_support:
            layers.append(_support_layer(arrays, opacity=0.38))
        paths = tuple(np.asarray(arrays[name], dtype=float) for name in names)
        layers.append(
            SceneLayer(key, "lines", np.vstack(paths), lines=paths, color=WINDING_COLOR, extra={"count": len(paths)})
        )
    elif key == "print_former":
        for name in print_former_array_names(arrays):
            stem = name.removesuffix("_vertices_m")
            layers.append(
                SceneLayer(
                    stem,
                    "surface",
                    np.asarray(arrays[name], dtype=float),
                    faces=np.asarray(arrays[f"{stem}_faces"], dtype=np.int64),
                    color=GROOVED_SUPPORT_COLOR,
                    extra={"role": "grooved_support", "reviewed": stem.startswith("review_")},
                )
            )
    elif key == "swept":
        grooved = [
            name.removesuffix("_vertices_m")
            for name in sorted(arrays)
            if name.startswith("grooved_support_") and name.endswith("_vertices_m")
        ]
        grooved = [stem for stem in grooved if f"{stem}_faces" in arrays]
        for stem in grooved:
            layers.append(
                SceneLayer(
                    stem,
                    "surface",
                    np.asarray(arrays[f"{stem}_vertices_m"], dtype=float),
                    faces=np.asarray(arrays[f"{stem}_faces"], dtype=np.int64),
                    color=GROOVED_SUPPORT_COLOR,
                    opacity=0.55,
                    extra={"role": "grooved_support"},
                )
            )
        if has_support and not grooved:
            layers.append(_support_layer(arrays))
        for name in sorted(name for name in arrays if name.startswith("swept_") and name.endswith("_vertices_m")):
            layers.append(
                SceneLayer(
                    name.removesuffix("_vertices_m"),
                    "surface",
                    np.asarray(arrays[name], dtype=float),
                    faces=np.asarray(arrays[name.removesuffix("_vertices_m") + "_faces"], dtype=np.int64),
                    color=WINDING_COLOR,
                )
            )
    elif key in FIELD_KEYS:
        points_name = field_points_name(arrays, key)
        reference_name = paired_field_name(arrays, key)
        field_name = FIELD_KEYS[key]
        if key in WIRE_FIELDS and original:
            field_name = field_name.replace("_field_T", "_field_at_stream_current_T")
        values = np.asarray(arrays[field_name], dtype=float)
        residual = key.endswith("_residual")
        if residual:
            values = values - np.asarray(arrays[reference_name], dtype=float)
        scalars = np.asarray(field_component_values(values, component, b0_direction=b0_direction), dtype=float)
        if scalars.ndim == 2:
            scalars = np.linalg.norm(scalars, axis=1)
        scalars = scalars * FIELD_DISPLAY_SCALE
        limits = None
        comparable = bool(reference_name and not residual and key not in {"target", "preview_target"})
        if comparable:
            desired = field_component_values(arrays[reference_name], component, b0_direction=b0_direction)
            desired = desired * FIELD_DISPLAY_SCALE
            limits = paired_color_limits(scalars, desired, signed=component != "Bmag")
            if show_target:
                scalars = desired
        if has_support:
            layers.append(_support_layer(arrays, opacity=0.24 if key == "preview_target" else 0.15))
        prefix = "Δ" if residual else "Target " if comparable and show_target else ""
        label = "|B − target|" if residual and component == "Bmag" else f"{prefix}{component}"
        layers.append(
            SceneLayer(
                key,
                "points",
                np.asarray(arrays[points_name], dtype=float),
                scalars=scalars,
                scalar_label=f"{label} [{FIELD_DISPLAY_UNIT}]",
                symmetric=residual and component != "Bmag",
                extra={"paired_target": comparable, "color_limits": limits},
            )
        )
    support = next((index for index, layer in enumerate(layers) if layer.name == "support"), None)
    if support is not None:
        layers[support + 1 : support + 1] = support_outline_layers(arrays)
    return layers, ""


def support_outline_layers(arrays):
    """Open support edges (cylinder ends and windows) as one bold line layer."""
    from .scene_views import OUTLINE_COLOR, outline_paths

    paths = outline_paths(arrays["vertices_m"], arrays["faces"])
    if not paths:
        return []
    return [
        SceneLayer(
            "support_edges",
            "lines",
            np.vstack(paths),
            lines=paths,
            color=OUTLINE_COLOR,
            extra={"role": "open_edges", "line_width": 4, "count": len(paths)},
        )
    ]


def highlight_layers(points, segments=()):
    """Located check evidence: the witness points and the recorded intervals between them."""
    from .scene_views import HIGHLIGHT_COLOR, valid_witness

    witness = valid_witness(points, segments)
    if witness is None:
        return []
    points, segments = witness
    layers = [SceneLayer("highlight_points", "points", points, color=HIGHLIGHT_COLOR, extra={"point_size": 14})]
    if segments:
        lines = tuple(points[list(pair)] for pair in segments)
        layers.append(
            SceneLayer(
                "highlight_segments",
                "lines",
                np.vstack(lines),
                lines=lines,
                color=HIGHLIGHT_COLOR,
                extra={"line_width": 7},
            )
        )
    return layers
