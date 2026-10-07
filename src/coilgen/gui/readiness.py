"""Actionable, lightweight readiness checks; backend execution remains authoritative."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from PySide6.QtCore import QEvent, QSize, QTimer, Qt, Signal
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from ..application.service import DesignService
from .colors import apply_tone, tone_color
from .icons import icon


@dataclass(frozen=True, slots=True)
class ReadinessIssue:
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
class ReadinessReport:
    issues: tuple[ReadinessIssue, ...] = ()

    @property
    def blocking(self):
        return tuple(issue for issue in self.issues if issue.blocking)

    @property
    def warnings(self):
        return tuple(issue for issue in self.issues if not issue.blocking)

    @property
    def ready(self):
        return not self.blocking


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
    return ReadinessIssue(message, node, key, action, title="Missing input" if missing else "Invalid input")


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
            ReadinessIssue(
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
        return ReadinessIssue(
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
    sweeping is disabled. Preparation, sheet optimization, contour extraction,
    and bare centreline routing do not require the sweep's convexity contract.
    """
    connected_routing = not config.skip_postprocessing and not config.contour_only_sweep_mode
    uses_geometry = stage in {"routed", "evaluated"} and (
        not config.skip_sweep or (connected_routing and not config.skip_normal_shift)
    )
    if not uses_geometry:
        return None
    from ..application.conductor_profiles import validate_conductor_profile

    try:
        validate_conductor_profile(
            config.cross_sectional_points,
            width_m=config.conductor_cross_section_width,
            height_m=config.conductor_cross_section_height,
        )
    except (ValueError, TypeError, OverflowError) as exc:
        return ReadinessIssue(
            str(exc),
            "material",
            "cross_sectional_points",
            "Open Conductor → Cross-section profile. Correct the point coordinates and segments, or choose a rectangle with positive width and height.",
            title="Invalid conductor profile",
        )
    return None


def check_readiness(project, *, service=DesignService, pipeline_stage=None) -> ReadinessReport:
    """Map actual validator/file/environment failures to settings and actions.

    Ordinary defaults and unused optional inputs are not invented requirements.
    This check is not a feasibility, convergence or manufacturing certificate.
    """
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
        return ReadinessReport((_input_issue(key, f"Project-owned input cannot be resolved: {exc}"),))
    root = Path(project.project_root)
    stage = pipeline_stage or getattr(project, "study", {}).get("pipeline_stage", "evaluated")
    prepared_only = stage == "prepared"
    profile_issue = _conductor_profile_issue(config, stage)
    try:
        report = getattr(service, "validate", DesignService.validate)(project)
    except (ValueError, TypeError, OverflowError) as exc:
        return ReadinessReport(
            tuple(issue for issue in (profile_issue,) if issue is not None)
            + (
                ReadinessIssue(
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
            ReadinessIssue(
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
    if profile_issue is not None:
        from ..application.conductor_profiles import is_rectangular_profile

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
    from .editors import SUPPORT_EDITORS

    editor = next((entry for entry in SUPPORT_EDITORS if entry.source == config.coil_mesh_file), None)
    if editor is not None and not any(
        issue.blocking and issue.key.split("[")[0] == editor.parameter_field for issue in issues
    ):
        try:
            editor.validate_parameters(getattr(config, editor.parameter_field))
        except ValueError as exc:
            rotation_error = "rotation axis" in str(exc).lower()
            issues.append(
                ReadinessIssue(
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
            ReadinessIssue(
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
    return ReadinessReport(tuple(issues))


def _readiness_icon(issue):
    """Use the actual setting key; ambiguous conditions retain a generic bound."""
    key = issue.key.lower()
    if key == "cross_sectional_points":
        return "conductor-profile"
    if "bend_radius" in key:
        return "bend-radius"
    if "clearance" in key or key in {"min_wire_distance_m", "min_turn_distance"}:
        return "wire-spacing"
    if "penetration" in key or "intersection" in key:
        return "collision"
    if key.startswith(("target_error_", "layout_target_error_")):
        return "field-error"
    if "current_density" in key:
        return "current-density"
    if key.startswith("layout_terminal"):
        return "terminals"
    return "constraint"


class ReadinessPanel(QWidget):
    """Nonmodal action list; warnings remain visible without blocking Compute."""

    issue_requested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("compute_readiness")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.issues = ()
        self._issue_labels = []
        self._issue_icons = []
        self._updating_colors = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        self.heading = QLabel()
        self.heading.setObjectName("compute_readiness_heading")
        self.heading.setWordWrap(True)
        font = self.heading.font()
        font.setBold(True)
        self.heading.setFont(font)
        layout.addWidget(self.heading)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(200)
        content = QWidget()
        self.rows = QVBoxLayout(content)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        self.hide()

    def set_issues(self, issues):
        self.issues = tuple(issues)
        self._issue_labels.clear()
        self._issue_icons.clear()
        while self.rows.count():
            item = self.rows.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        blocking = sum(issue.blocking for issue in self.issues)
        self.heading.setText(
            "Cannot compute yet — resolve the item below."
            if blocking == 1
            else f"Cannot compute yet — resolve these {blocking} items."
            if blocking
            else "Modeling notes — Compute is available."
        )
        for issue in self.issues:
            row = QWidget()
            layout = QHBoxLayout(row)
            symbol = QLabel()
            symbol.setFixedSize(24, 24)
            symbol.setAccessibleName(issue.title)
            symbol.setAccessibleDescription("Blocking input error" if issue.blocking else "Unverified modeling note")
            symbol.setToolTip(f"{issue.severity.capitalize()}: {issue.key}")
            self._issue_icons.append(symbol)
            layout.addWidget(symbol, 0, Qt.AlignmentFlag.AlignTop)
            text = QLabel(f"{issue.title}: {issue.message}\n{issue.action}")
            text.setTextFormat(Qt.TextFormat.PlainText)
            text.setWordWrap(True)
            self._issue_labels.append(text)
            layout.addWidget(text, 1)
            button = QPushButton("Open settings")
            button.setIcon(icon("settings"))
            button.setIconSize(QSize(18, 18))
            button.setAccessibleName(f"Open settings: {issue.title}")
            button.setAccessibleDescription(issue.action)
            button.clicked.connect(lambda _checked=False, item=issue: self.issue_requested.emit(item))
            layout.addWidget(button)
            self.rows.addWidget(row)
        self._refresh_colors()
        self.setVisible(bool(self.issues))

    def _refresh_colors(self):
        if self._updating_colors:
            return
        self._updating_colors = True
        try:
            tone = "error" if any(issue.blocking for issue in self.issues) else "warning" if self.issues else "neutral"
            palette = self.palette()
            background = palette.color(QPalette.ColorRole.Window).name()
            border = tone_color(tone, palette).name()
            self.setStyleSheet(
                f"QWidget#compute_readiness {{ background: {background}; border: 1px solid {border}; border-radius: 5px; }}"
            )
            apply_tone(self.heading, tone)
            for issue, label in zip(self.issues, self._issue_labels):
                apply_tone(label, issue.severity)
            for issue, symbol in zip(self.issues, self._issue_icons):
                badge = "failed" if issue.blocking else "unverified"
                symbol.setProperty("evidenceBadge", badge)
                symbol.setProperty("scientificIcon", _readiness_icon(issue))
                symbol.setPixmap(
                    icon(_readiness_icon(issue), badge=badge).pixmap(QSize(24, 24), symbol.devicePixelRatioF())
                )
        finally:
            self._updating_colors = False

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange, QEvent.Type.StyleChange):
            if hasattr(self, "heading") and not self._updating_colors:
                QTimer.singleShot(0, self, self._refresh_colors)
