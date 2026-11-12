"""Human-readable selections from an original run request, independent of the editor."""

from collections.abc import Mapping
import json
import math
from pathlib import PureWindowsPath
from types import SimpleNamespace

from ..target_regions import target_region_summary
from .session import PIPELINE_STAGES
from .support_parameters import SUPPORT_EDITORS
from .targets import TARGET_NORMALIZATION_MODES


CONTOUR_METHODS = {
    "off": "Original contours",
    "parameterized_bspline": "Surface B-spline smoothing",
    "resample": "Resample without smoothing",
}

_CHOICES = {
    "design_context": {
        "mr_gradient": "MRI gradient",
        "mr_shim": "MRI shim",
        "mr_rf": "MRI RF",
        "free": "General field design",
    },
    "target_gradient_normalization": {value: label for names, label in TARGET_NORMALIZATION_MODES for value in names},
    "sf_opt_method": {
        "constrained_quadratic": "Constrained quadratic solve",
        "tikhonov": "Tikhonov fit",
        "target_error_constrained_efficiency": "Field-budget search",
    },
    "stream_function_basis": {"vertex": "Mesh vertices"},
    "objective": {
        "power": "Joule power",
        "minimax": "Peak surface current density",
        "magnetic_energy": "Magnetic energy",
        "mixture": "Weighted physical objectives",
    },
    "contour_discretization_method": {
        "level_count": "Specified contour count",
        "engineering": "Wire current and spacing",
        "auto": "Automatic",
    },
    "conductor_mounting_mode": {
        "surface_outer": "Outer side (along normal)",
        "surface_inner": "Inner side (against normal)",
        "centered": "Centered on support",
    },
    "conductor_path_reference": {"centroid": "Profile centroid", "profile_origin": "Profile origin (0, 0)"},
    "layout_terminal_strategy": {
        "undefined": "Not decided",
        "pads": "Terminal pads",
        "wire_stubs": "Wire stubs",
        "waived": "Handled outside CoilGen",
    },
    "support_body_mode": {
        "off": "Off",
        "inside": "Grooved solid inside the winding",
        "outside": "Grooved solid outside the winding",
    },
    "routing_circuit_scope": {"per_part": "Separate circuit per support", "system": "System circuit"},
    "suppression_field_component": {
        "vector": "All three components (Bx, By, Bz)",
        "target": "Same component as the target",
        "bz": "Bz only",
        "b_normal": "Normal to the surface",
        "magnitude": "Field magnitude |B|",
    },
    "suppression_surface_weighting": {"area": "By surface area", "uniform": "Equal per sample"},
    "interconnection_axis": {"support_axis": "Along the support bore", "b0": "Along the main field"},
    "inter_group_connection_path_method": {
        "mesh_geodesic": "Geodesic on the mesh",
        "direct_3d": "Direct 3D connection",
    },
    "contour_smoothing_scale_mode": {"mesh_coupled": "Local mesh wavelength", "legacy": "Legacy fit strength"},
    "kernel_backend": {
        "auto": "Automatic",
        "numpy": "NumPy",
        "numba": "Numba",
        "reference": "Reference implementation",
    },
}


def choice_title(key, value):
    """Use the same physical labels in the form and submitted summary."""
    if isinstance(value, str):
        lookup = value.strip().lower().replace("-", "_") if key == "target_gradient_normalization" else value
        return _CHOICES.get(key, {}).get(lookup, value)
    return value


def _display(value, scale=1, unit=""):
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        return f"{value * scale:.8g}" + (f" {unit}" if unit else "")
    if isinstance(value, (list, tuple)) and all(isinstance(v, (int, float)) for v in value):
        return "(" + ", ".join(f"{v * scale:.8g}" for v in value) + ")" + (f" {unit}" if unit else "")
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def displacement_label(metres):
    """Display measured smoothing displacement in practical metric units."""
    if isinstance(metres, bool) or not isinstance(metres, (float, int)) or not math.isfinite(metres) or metres < 0:
        return "Unrecorded"
    if metres == 0 or metres >= 1e-3:
        return f"{metres * 1e3:.4g} mm"
    if metres >= 1e-6:
        return f"{metres * 1e6:.4g} µm"
    return f"{metres * 1e9:.4g} nm"


def contour_fit_caption(metadata):
    """Describe measured pre-routing fits without inferring outcomes for unrecorded runs."""
    parts = metadata.get("contour_smoothing")
    if not parts or not isinstance(parts, (tuple, list)):
        return ""
    keys = ("smoothed_count", "fit_rejected_count", "spacing_rejected_count")
    counts = [0, 0, 0]
    for part in parts:
        if not isinstance(part, Mapping):
            return ""
        for index, key in enumerate(keys):
            value = part.get(key)
            if type(value) is not int or value < 0:
                return ""
            counts[index] += value
    if not sum(counts):
        return ""
    applied, fit, spacing = counts
    note = []
    local = [part.get("local_fit_count") for part in parts]
    if all(type(n) is int and 0 <= n <= part["smoothed_count"] for n, part in zip(local, parts)) and sum(local):
        note.append(f"{sum(local)} using local corner blends")
    if fit:
        note.append(f"{fit} kept after fit checks")
    if spacing:
        note.append(f"{spacing} kept for spacing")
    return (
        f"Before routing: {applied} of {sum(counts)} contours smoothed" + ("; " + ", ".join(note) if note else "") + "."
    )


def design_choice_groups(request):
    """Summarize recorded choices only. The complete request retains its exact precision."""
    config = request.get("config") or {}
    requirements = request.get("requirements")
    groups = []

    def group(title, source, specs):
        rows = [
            {
                "name": name,
                "value": "No specified minimum (0 mm)"
                if key == "min_turn_distance" and source[key] == 0
                else _display(choice_title(key, source[key]), *formatting),
            }
            for key, name, *formatting in specs
            if key in source and source[key] is not None
        ]
        groups.append({"title": title, "rows": rows})
        return rows

    geometry = group("Support and mesh", config, [("coil_mesh_file", "Support surface")])
    editor = next((e for e in SUPPORT_EDITORS if e.source == config.get("coil_mesh_file")), None)
    if editor:
        geometry[0]["value"] = editor.label
    elif geometry and isinstance(config.get("coil_mesh_file"), str):
        source = PureWindowsPath(config["coil_mesh_file"])
        if source.suffix.lower() in {".stl", ".npz"}:
            geometry[0]["value"] = "Imported mesh · " + source.name
    if editor and (parameters := config.get(editor.parameter_field)):
        try:
            geometry.extend(
                {
                    "name": field.label,
                    "value": _display(editor.display_value(field, parameters), 1, field.suffix.strip()),
                }
                for field in editor.fields
            )
        except (IndexError, TypeError, ValueError, ArithmeticError):
            geometry.append({"name": "Recorded mesh parameters", "value": _display(parameters)})

    field = group(
        "Target and magnetic field",
        config,
        [
            ("design_context", "Design context"),
            ("target_field_component", "Target component"),
            ("target_gradient_normalization", "Target normalization"),
            ("b_0_direction", "World B0 direction (x, y, z)"),
            ("b_0_strength_T", "MRI B0 strength", 1, "T"),
        ],
    )
    imported = config.get("target_field_definition_file")
    if requirements:
        field[:] = [row for row in field if row["name"] not in {"Target component", "Target normalization"}]
        field.append({"name": "Target source", "value": "Registered desired fields in the named study regions"})
        for region in requirements.get("regions", []):
            limits = [
                f"{label} {_display(region[key], 1e6, 'µT')}"
                for key, label in (("rms_limit_T", "RMS limit"), ("peak_limit_T", "peak limit"))
                if region.get(key) is not None
            ]
            field.append({"name": region["name"], "value": " · ".join([region["component"], *limits])})
    elif imported and str(imported).lower() != "none":
        field.append({"name": "Imported target", "value": str(imported)})
    elif "field_shape_function" in config:
        field.append(
            {"name": "Target expression (world x, y, z in metres)", "value": str(config["field_shape_function"])}
        )
    if not requirements and config.get("design_context") == "mr_gradient" and "target_gradient_strength" in config:
        field.append(
            {
                "name": "Configured gradient amplitude",
                "value": _display(config["target_gradient_strength"], 1000, "mT/m"),
            }
        )
    try:
        region = target_region_summary(SimpleNamespace(**config))
    except (AttributeError, ValueError, KeyError, TypeError):
        region = None
    if region and not requirements:
        field.append({"name": "Target region", "value": region})
    physics = config.get("research_physics") or {}
    if physics.get("background_B_per_face_T") is not None:
        field.append(
            {
                "name": "Mechanical background field",
                "value": f"Per-face vectors ({len(physics['background_B_per_face_T'])} faces, T)",
            }
        )
    elif physics.get("b0_T") is not None:
        field.append({"name": "Mechanical background field (x, y, z)", "value": _display(physics["b0_T"], 1, "T")})

    solve = group(
        "Stream function optimization",
        config,
        [
            ("sf_opt_method", "Stream-function method"),
            ("stream_function_basis", "Basis"),
            ("target_error_tolerance_percent", "Target error tolerance", 1, "%"),
            ("conductor_thickness", "Homogenized sheet thickness", 1000, "mm"),
        ],
    )
    if physics.get("objective"):
        solve.append({"name": "Physics objective", "value": choice_title("objective", physics["objective"])})

    wire = group(
        "Winding and conductor",
        config,
        [
            ("levels", "Contour levels"),
            ("contour_discretization_method", "Contour selection"),
            ("min_turn_distance", "Minimum centerline spacing", 1000, "mm"),
            ("conductor_cross_section_width", "Conductor width", 1000, "mm"),
            ("conductor_cross_section_height", "Conductor height", 1000, "mm"),
            ("sweep_min_bend_radius", "Minimum bend radius", 1000, "mm"),
            ("conductor_mounting_mode", "Conductor side of support"),
        ],
    )
    if config.get("support_body_mode", "off") not in {"off", None}:
        wire.append(
            {"name": "Grooved support body", "value": choice_title("support_body_mode", config["support_body_mode"])}
        )
        wire.extend(
            {"name": name, "value": f"{float(config[key]) * 1000:g} mm"}
            for key, name in (
                ("support_body_thickness", "Support body thickness"),
                ("support_body_groove_clearance", "Groove clearance"),
                ("support_body_groove_extra_depth", "Extra groove depth"),
            )
            if key in config
        )
    if config.get("contour_only_sweep_mode"):
        wire.append({"name": "Winding topology", "value": "Independent closed turns"})
    elif config.get("skip_postprocessing") is False:
        wire.extend(
            {"name": name, "value": _display(choice_title(key, config[key]))}
            for key, name in (
                ("routing_circuit_scope", "Circuit scope"),
                ("interconnection_axis", "Interconnection direction"),
                ("inter_group_connection_path_method", "Connection method"),
            )
            if key in config
        )

    mode = config.get("contour_path_regularization_mode")
    smoothing = group("Contour smoothing", config, [])
    if mode is not None:
        from ..contours import _normalise_contour_path_regularization_mode

        mode = _normalise_contour_path_regularization_mode(mode)
        smoothing.append({"name": "Contour path treatment", "value": CONTOUR_METHODS[mode]})
    if mode == "parameterized_bspline":
        for key, name, scale, unit in [
            ("contour_smoothing_scale_mode", "Smoothing scale", 1, ""),
            ("contour_smoothing_max_displacement", "Maximum allowed displacement", 1000, "mm"),
            ("contour_smoothing_preserve_min_turn_distance", "Guard inter-turn spacing", 1, ""),
        ]:
            if key in config:
                value = _display(choice_title(key, config[key]), scale, unit)
                if key == "contour_smoothing_max_displacement" and config[key] == 0:
                    value = "No displacement cap (0 mm)"
                if (
                    key == "contour_smoothing_preserve_min_turn_distance"
                    and config[key]
                    and config.get("min_turn_distance") == 0
                ):
                    value = "Yes · preserve the extracted minimum spacing within numerical tolerance"
                smoothing.append({"name": name, "value": value})
        if "contour_smoothing_scale_mode" in config:
            from ..contour_smoothing import _normalise_scale_mode

            mesh = _normalise_scale_mode(config["contour_smoothing_scale_mode"]) == "mesh_coupled"
            key, name = (
                ("contour_smoothing_mesh_gain", "Gain at mesh wavelength")
                if mesh
                else ("contour_smoothing_factor", "Legacy smoothing strength")
            )
            if key in config:
                smoothing.append({"name": name, "value": _display(config[key])})
            if mesh and "contour_smoothing_local_fallback" in config:
                local = config["contour_smoothing_local_fallback"]
                value = _display(local)
                if local and (
                    not config.get("contour_smoothing_preserve_min_turn_distance")
                    or not config.get("contour_smoothing_max_displacement")
                ):
                    value = "Requested; inactive without a positive displacement cap and spacing guard"
                smoothing.append(
                    {
                        "name": "Try local corner blends if a fit is rejected",
                        "value": value,
                    }
                )

    study = group(
        "Study and validation",
        request,
        [
            ("validation_resolution", "Independent grid resolution"),
            ("design_margin", "Design budget ratio"),
            ("fabricate", "Fabrication review requested"),
            ("kernel_backend", "Kernel backend"),
        ],
    )
    if "pipeline_stage" in request:
        stage = request["pipeline_stage"]
        study.insert(0, {"name": "Requested boundary", "value": dict(PIPELINE_STAGES).get(stage, stage)})
    return [g for g in groups if g["rows"]]
