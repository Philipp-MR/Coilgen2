"""Scientific preflight shared by desktop tools, workers and ordinary scripts.

Checks input/configuration/environment readiness without assembling field
operators or invoking a solver. Readiness is not engineering acceptance.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from ..config_validation import validate_config


@dataclass(frozen=True, slots=True)
class PreflightIssue:
    message: str
    node: str
    key: str
    action: str
    severity: str = "error"
    title: str = "Invalid setting"

    @property
    def blocking(self):
        return self.severity == "error"


@dataclass(frozen=True, slots=True)
class PreflightReport:
    issues: tuple[PreflightIssue, ...] = ()

    @property
    def blocking(self):
        return tuple(issue for issue in self.issues if issue.blocking)

    @property
    def warnings(self):
        return tuple(issue for issue in self.issues if not issue.blocking)

    @property
    def ready(self):
        return not self.blocking

    def raise_for_errors(self) -> None:
        """Reject execution while retaining each actionable native field."""
        if self.blocking:
            details = "\n".join(f"[ERROR] {issue.key}: {issue.message}" for issue in self.blocking)
            raise ValueError("CoilGen preflight failed:\n" + details)


def settings_node(key):
    key = key.split("[")[0]
    # Field-error limits belong to the study formulation, not the target
    # definition, even though their native keys share the "target" prefix.
    if key.startswith("target_error_"):
        return "optimization"
    if key.startswith(("secondary_target", "target", "field_shape", "mri_shim", "create_")) or "suppression" in key:
        return "target"
    if "mesh" in key or key.startswith(("generated_mesh", "surface_is_cylinder", "geometry_source")):
        return "geometry"
    if key in {
        "conductor_thickness",
        "specific_conductivity_conductor",
        "conductor_cross_section_width",
        "conductor_cross_section_height",
        "conductor dimensions",
        "conductor_mounting_mode",
        "cross_sectional_points",
        "sweep_insulation_thickness",
    }:
        return "material"
    if key.startswith(
        ("sweep", "inter_", "intra_", "contour", "routing", "pcb", "normal_shift", "layout_terminal")
    ) or key in {
        "levels",
        "skip_sweep",
        "skip_postprocessing",
        "skip_normal_shift",
        "min_turn_distance",
        "min_wire_distance_m",
        "min_points_per_loop",
        "pot_offset_factor",
        "smooth_factor",
        "cut_width",
        "force_cut_selection",
    }:
        return "winding"
    if key.startswith("validation"):
        return "validation"
    return "optimization"


def _input_issue(key, message, *, missing=False):
    node = (
        "geometry"
        if key == "coil_mesh_file"
        else "optimization"
        if key in {"sf_source_file", "preoptimized_stream_function_file"}
        else "target"
    )
    action = (
        "Import a support STL/NPZ or select a generated support."
        if node == "geometry"
        else "Import a valid target map, or explicitly choose an analytic target."
        if key == "target_field_definition_file"
        else "Import the configured region mesh, or clear this optional region input if it is not used."
        if key.endswith("mesh_file")
        else "Restore the stream-function input, or explicitly choose optimization instead of an imported solution."
    )
    return PreflightIssue(message, node, key, action, title="Missing input" if missing else "Invalid input")


def _input_paths(config, root, *, prepared_only=False):
    from ..geometry import geometry_factory
    from ..object_registry import is_in_memory_mesh_key, is_in_memory_target_key
    from ..preoptimization import preoptimized_stream_function_source

    fields = ["coil_mesh_file", "target_mesh_file", "secondary_target_mesh_file", "target_field_definition_file"]
    if not prepared_only:
        source = preoptimized_stream_function_source(config)
        if source != "none":
            fields.append(
                "preoptimized_stream_function_file"
                if str(config.preoptimized_stream_function_file).strip().lower() not in {"", "none"}
                else "sf_source_file"
            )
    for key in fields:
        source = str(getattr(config, key)).strip()
        if source.lower() in {"", "none"} or is_in_memory_mesh_key(source) or is_in_memory_target_key(source):
            continue
        if key == "coil_mesh_file" and geometry_factory(source):
            continue
        path = Path(source).expanduser()
        if not path.is_absolute():
            path = (config.resolved_geometry_source_path(root) if key.endswith("mesh_file") else root) / path
        yield key, path


def _validate_input(key, path, config):
    """Use actual backend file readers; never assemble field operators."""
    if key.endswith("mesh_file"):
        from ..geometry import read_mesh_npz, read_stl

        mesh = read_mesh_npz(path) if path.suffix.lower() == ".npz" else read_stl(path)
        if not mesh.n_vertices or (key == "coil_mesh_file" and not mesh.n_faces):
            raise ValueError("The imported mesh has no usable vertices or support triangles.")
        if not np.all(np.isfinite(mesh.vertices)) or np.any(mesh.faces < 0) or np.any(mesh.faces >= mesh.n_vertices):
            raise ValueError("Mesh coordinates must be finite and triangle indices must reference actual vertices.")
        if np.any(np.diff(np.sort(mesh.faces, axis=1), axis=1) == 0):
            raise ValueError("Every mesh triangle must reference three distinct vertices.")
        if key == "coil_mesh_file":
            triangles = mesh.vertices[mesh.faces]
            area_vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            zero_area = np.flatnonzero(np.all(area_vectors == 0, axis=1))
            if zero_area.size:
                raise ValueError(
                    f"Support triangle {int(zero_area[0]) + 1} has zero area. "
                    "Remove coincident or collinear triangle vertices before importing the support."
                )
    elif key == "target_field_definition_file":
        from ..target import load_target_field_definition

        load_target_field_definition(path, None, config)


def _solver_issues(config, *, prepared_only):
    from ..optimization import _normalise_constrained_quadratic_fallback_policy, _normalise_stream_function_method
    from ..optimization_solvers import conic_solver_order
    from ..preoptimization import has_preoptimized_stream_function

    if prepared_only or has_preoptimized_stream_function(config):
        return ()
    physical = config.research_physics is not None
    if not physical and _normalise_stream_function_method(config.sf_opt_method) != "constrained_quadratic":
        return ()
    strict = physical or _normalise_constrained_quadratic_fallback_policy(config) == "strict"
    try:
        if find_spec("cvxpy") is None:
            raise ModuleNotFoundError("CVXPY is not installed for the selected conic optimization.")
        if not conic_solver_order(config.constrained_optimization_solver):
            raise RuntimeError("No supported conic solver is installed (CLARABEL, ECOS or SCS).")
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        message = str(exc)
        if not strict:
            message += " The configured legacy fallback may run, without a direct conic-solver acceptance certificate."
        return (
            PreflightIssue(
                message,
                "optimization",
                "constrained_optimization_solver",
                "Install the research dependencies in this Python environment, or explicitly choose an available optimization method/backend.",
                severity="error" if strict else "warning",
                title="Unavailable solver",
            ),
        )
    return ()


def _expression_issue(config):
    from ..target import _ALLOWED_FUNCTIONS, _normalize_field_expression

    if config.target_field_definition_file != "none":
        return None
    try:
        tree = ast.parse(_normalize_field_expression(config.field_shape_function), mode="eval")
        # All operands are evaluated in simple numerical arithmetic/calls.
        # Bindings, conditionals, comparisons and legacy attribute expressions
        # retain their backend semantics; no invented coordinates are sampled.
        simple_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Call,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.operator,
            ast.unaryop,
        )
        if all(isinstance(node, simple_nodes) for node in ast.walk(tree)):
            unknown = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} - {
                *_ALLOWED_FUNCTIONS,
                "x",
                "y",
                "z",
            }
            if unknown:
                raise ValueError(f"Unknown target symbol(s): {', '.join(sorted(unknown))}.")
    except (SyntaxError, ValueError) as exc:
        message = f"Target expression syntax is invalid: {exc.msg}." if isinstance(exc, SyntaxError) else str(exc)
        return PreflightIssue(
            message,
            "target",
            "field_shape_function",
            "Correct the target expression using world coordinates x, y, z and the supported numerical functions.",
            title="Invalid target expression",
        )
    return None


def _conductor_profile_issue(config, stage):
    """Check profiles only at stages that consume finite cross-section geometry.

    The sweep uses the polygon in either connected or independent-turn mode.
    Connected return-path shifting also reads its physical envelope, even when
    sweeping is disabled. Preparation, stream function optimization, contour extraction,
    and bare centreline routing do not require the sweep's convexity contract.
    """
    connected_routing = not config.skip_postprocessing and not config.contour_only_sweep_mode
    uses_geometry = stage in {"routed", "evaluated"} and (
        not config.skip_sweep or (connected_routing and not config.skip_normal_shift)
    )
    if not uses_geometry:
        return None
    from .conductor_profiles import validate_conductor_profile

    try:
        validate_conductor_profile(
            config.cross_sectional_points,
            width_m=config.conductor_cross_section_width,
            height_m=config.conductor_cross_section_height,
        )
    except (ValueError, TypeError, OverflowError) as exc:
        return PreflightIssue(
            str(exc),
            "material",
            "cross_sectional_points",
            "Open Conductor → Cross-section profile. Correct the point coordinates and segments, or choose a rectangle with positive width and height.",
            title="Invalid conductor profile",
        )
    return None


def _buildability_issues(config, stage):
    """Warn when routing settings cannot satisfy the conductor's own checks.

    These are geometric necessities, derived from the same required
    centerline clearance the fabrication review enforces. They warn rather
    than block, so DRAFT geometry can still be computed for inspection.
    """
    connected = not config.skip_postprocessing and not config.contour_only_sweep_mode
    if stage not in {"routed", "evaluated"} or not connected or config.skip_sweep:
        return []
    from ..layout import _within_group_cut_width
    from ..manufacturing import _required_centerline_clearance, conductor_cross_section_points

    try:
        clearance = _required_centerline_clearance(conductor_cross_section_points(config), config)
        cut_width = float(_within_group_cut_width(config))
    except (ValueError, TypeError, AttributeError):
        return []
    issues = []
    mm = 1e3
    if not config.skip_normal_shift and float(config.normal_shift_length) < clearance:
        issues.append(
            PreflightIssue(
                f"Return paths are lifted {config.normal_shift_length * mm:.3g} mm, but this conductor needs "
                f"{clearance * mm:.3g} mm centerline clearance, so every crossing will fail the clearance check.",
                settings_node("normal_shift_length"),
                "normal_shift_length",
                f"Set Return-path lift to at least {np.ceil(clearance * mm * 10) / 10:.3g} mm.",
                severity="warning",
                title="Return-path lift below conductor clearance",
            )
        )
    pitch = max(
        float(config.min_turn_distance),
        float(config.conductor_cross_section_width) + 2.0 * float(config.sweep_insulation_thickness),
    )
    radius = float(config.sweep_min_bend_radius)
    if radius > 0.0:
        # Beyond two quarter-circle bends, extra pitch adds a straight bridge.
        bend_rise = min(pitch, 2.0 * radius)
        needed = float(np.sqrt(bend_rise * (4.0 * radius - bend_rise)))
        if cut_width < needed:
            issues.append(
                PreflightIssue(
                    f"The {cut_width * mm:.3g} mm opening cut is too narrow for an S-bend of {radius * mm:.3g} mm "
                    f"radius between turns: at least {needed * mm:.3g} mm at {pitch * mm:.3g} mm pitch.",
                    settings_node("sweep_min_bend_radius"),
                    "sweep_min_bend_radius",
                    "Reduce Minimum bend radius, or widen the within-group cut in Advanced settings.",
                    severity="warning",
                    title="Opening cut narrower than the bend radius allows",
                )
            )
    spacing = float(config.min_turn_distance)
    if 0.0 < spacing < clearance:
        # Zero leaves spacing to the stream function; only an explicit
        # spacing below the conductor's own clearance is self-contradictory.
        issues.append(
            PreflightIssue(
                f"Minimum centerline spacing is set to {spacing * mm:.3g} mm, below the {clearance * mm:.3g} mm "
                "this conductor needs; turns that close fail the clearance check.",
                settings_node("min_turn_distance"),
                "min_turn_distance",
                f"Set Minimum centerline spacing to at least {np.ceil(clearance * mm * 10) / 10:.3g} mm, or to 0 "
                "to leave it to the optimizer.",
                severity="warning",
                title="Turn spacing below conductor clearance",
            )
        )
    return issues


def check_preflight(project, *, validator=None, pipeline_stage=None) -> PreflightReport:
    """Map actual validator/file/environment failures to settings and actions.

    Ordinary defaults and unused optional inputs are not invented requirements.
    This check is not a feasibility, convergence or manufacturing certificate.
    """
    from ..pipeline_stages import PipelineStage

    requested_stage = (
        getattr(project, "study", {}).get("pipeline_stage", "evaluated") if pipeline_stage is None else pipeline_stage
    )
    try:
        stage = PipelineStage(requested_stage).value
    except (ValueError, TypeError):
        return PreflightReport(
            (
                PreflightIssue(
                    "Pipeline stage must be prepared, sheet, contours, routed, or evaluated.",
                    "validation",
                    "pipeline_stage",
                    "Choose a supported computation boundary.",
                    title="Invalid study controls",
                ),
            )
        )
    issues = []
    try:
        config = project.config
    except (OSError, ValueError, KeyError) as exc:
        values = getattr(project, "values", {})
        key = next(
            (
                key
                for key, value in values.items()
                if isinstance(value, str) and value.startswith("asset:") and value[6:] in str(exc)
            ),
            "coil_mesh_file",
        )
        return PreflightReport((_input_issue(key, f"Project-owned input cannot be resolved: {exc}"),))
    root = Path(project.project_root)
    if getattr(project, "study", {}).get("requirements") is not None:
        from .research_requirements import requirements_validation_config

        config = requirements_validation_config(config)
    prepared_only = stage == "prepared"
    profile_issue = _conductor_profile_issue(config, stage)
    try:
        report = validator(project) if validator is not None else validate_config(config, project_root=root)
    except (ValueError, TypeError, OverflowError) as exc:
        return PreflightReport(
            tuple(issue for issue in (profile_issue,) if issue is not None)
            + (
                PreflightIssue(
                    str(exc),
                    "optimization",
                    "configuration",
                    "Correct the malformed configuration in its settings or expert editor.",
                    title="Invalid configuration",
                ),
            )
        )
    for diagnostic in report.messages:
        issues.append(
            PreflightIssue(
                diagnostic.message,
                settings_node(diagnostic.field),
                diagnostic.field,
                diagnostic.suggestion or f"Open settings and correct {diagnostic.field}.",
                severity=diagnostic.level,
                title="Missing support"
                if diagnostic.field == "coil_mesh_file" and str(config.coil_mesh_file).strip().lower() in {"", "none"}
                else "Invalid setting"
                if diagnostic.level == "error"
                else "Modeling note",
            )
        )
    physics = config.research_physics
    if physics is not None and not any(issue.blocking for issue in issues):
        from ..field_components import normalized_b0_direction
        from ..research.convex import ResearchPhysicsSpec

        spec = ResearchPhysicsSpec.from_dict(physics)
        reference = normalized_b0_direction(config.b_0_direction) * config.b_0_strength_T
        if (
            spec.background_B_per_face_T is None
            and spec.b0_T is not None
            and not np.allclose(spec.b0_T, reference, rtol=1e-12, atol=1e-15)
        ):
            mechanical = ", ".join(f"{v:.5g}" for v in spec.b0_T)
            mri = ", ".join(f"{v:.5g}" for v in reference)
            issues.append(
                PreflightIssue(
                    f"Mechanical loads use ({mechanical}) T, while the MRI reference is ({mri}) T. Check research_physics.b0_T in Advanced settings.",
                    "target",
                    "research_physics",
                    "Confirm the independent mechanical load case or align the two vectors.",
                    severity="warning",
                    title="Separate mechanical background",
                )
            )
    if profile_issue is not None:
        from .conductor_profiles import is_rectangular_profile

        # Preserve existing width/height diagnostics without reporting a second
        # error for the same invalid rectangle dimensions.
        dimension_keys = {"conductor_cross_section_width", "conductor_cross_section_height"}
        if not any(
            issue.blocking
            and (
                issue.key == "cross_sectional_points"
                or (is_rectangular_profile(config.cross_sectional_points) and issue.key in dimension_keys)
            )
            for issue in issues
        ):
            issues.append(profile_issue)
    from .support_parameters import SUPPORT_EDITORS

    editor = next((entry for entry in SUPPORT_EDITORS if entry.source == config.coil_mesh_file), None)
    if editor is not None and not any(
        issue.blocking and issue.key.split("[")[0] == editor.parameter_field for issue in issues
    ):
        try:
            editor.validate_parameters(getattr(config, editor.parameter_field))
        except ValueError as exc:
            rotation_error = "rotation axis" in str(exc).lower()
            issues.append(
                PreflightIssue(
                    str(exc),
                    "geometry",
                    "rotation_axis_x" if rotation_error else editor.parameter_field,
                    "Define a nonzero rotation axis or set the angle to zero."
                    if rotation_error
                    else "Correct the support dimensions, orientation or mesh sampling in Geometry.",
                    title="Undefined support rotation" if rotation_error else "Invalid support geometry",
                )
            )
    if config.target_field_definition_file == "none" and str(config.field_shape_function).strip().lower() == "none":
        issues.append(
            PreflightIssue(
                "No analytic target expression or imported target map is configured.",
                "target",
                "field_shape_function",
                "Enter an analytic target expression, select a spherical harmonic, or import a target map.",
                title="Missing target",
            )
        )
    if not any(issue.blocking and issue.key == "field_shape_function" for issue in issues):
        expression_issue = _expression_issue(config)
        if expression_issue is not None:
            issues.append(expression_issue)
    for key, path in _input_paths(config, root, prepared_only=prepared_only):
        if any(issue.blocking and issue.key == key for issue in issues):
            continue
        if not path.is_file():
            issues = [
                issue
                for issue in issues
                if not (issue.key == key and issue.severity == "warning" and "not found" in issue.message.lower())
            ]
            issues.append(_input_issue(key, f"Configured input was not found: {path}", missing=True))
            continue
        try:
            _validate_input(key, path, config)
        except (OSError, ValueError, KeyError, TypeError, NotImplementedError, EOFError, BadZipFile) as exc:
            issues.append(_input_issue(key, f"Configured input cannot be used: {exc}"))
    if not any(issue.blocking and issue.node == "optimization" for issue in issues):
        issues.extend(_solver_issues(config, prepared_only=prepared_only))
    if not any(issue.blocking for issue in issues):
        issues.extend(_buildability_issues(config, stage))
    return PreflightReport(tuple(issues))
