"""One design session, shared by the desktop and browser workbenches.

The Qt workbench and the browser workbench are two views of the same
operations: both submit computations through :func:`submit_design` with the
controls from :func:`request_controls`, edit the same immutable
``ProjectDocument`` revisions, read the same saved ``RunRecord`` evidence and
export through :mod:`coilgen.application.cad`. :class:`WorkbenchSession` holds
the state a client needs (assembly, active channel, undo history, jobs) without
importing Qt, so the browser server and tests can drive it directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path, PureWindowsPath
from typing import Any, Callable

from ..field_components import TARGET_COMPONENTS, normalized_b0_direction
from .assembly import ProjectAssembly
from .execution import plan_execution
from .preflight import check_preflight
from .project import ProjectDocument
from .service import DesignService
from .support_parameters import SUPPORT_EDITORS

PIPELINE_STAGES = (
    ("prepared", "Prepare model"),
    ("sheet", "Optimize stream function"),
    ("contours", "Generate contours"),
    ("routed", "Connect / sweep conductors"),
    ("evaluated", "Full design evaluation"),
)
ACTIVE_STATUSES = frozenset({"queued", "running", "cancelling"})


def stage_study_changes(stage: str, full_fabrication: bool) -> dict:
    """Study edit for a selected pipeline boundary; review needs evaluation."""
    return {"pipeline_stage": stage, "fabricate": bool(full_fabrication) if stage == "evaluated" else False}


def request_controls(project, *, pipeline_stage, validation_resolution, design_margin, fabricate) -> dict:
    """Study controls of one submission, identical for every client."""
    return {
        **({"engineering": project.study["engineering"]} if "engineering" in project.study else {}),
        "pipeline_stage": pipeline_stage,
        "validation_resolution": validation_resolution,
        "design_margin": design_margin,
        "fabricate": fabricate,
        "kernel_backend": project.study["kernel_backend"],
    }


def submit_design(project, jobs, controls, *, service=None, reuse=None, resume_checkpoint=None) -> dict:
    """Plan, verify and queue one immutable project revision.

    Returns the queued (or already active, identical) job state, annotated
    with the project identity and controls exactly as the desktop records it.
    """
    service = DesignService() if service is None else service
    plan = plan_execution(
        project,
        reuse=reuse,
        **controls,
        **({"resume_checkpoint": resume_checkpoint} if resume_checkpoint else {}),
    )
    plan.require_current(project)
    if isinstance(service, DesignService):
        state = service.submit_plan(project, jobs, plan)
    else:
        state = service.submit(project, jobs, **plan.request_controls())
    state.setdefault("pipeline_stage", controls["pipeline_stage"])
    state.setdefault("project_id", project.id)
    state.setdefault("project_fingerprint", project.fingerprint)
    state.setdefault("project_revision", project.revision)
    for key, value in controls.items():
        state.setdefault(key, value)
    return state


@dataclass(frozen=True)
class SessionAction:
    id: str
    label: str
    group: str


# Every user operation of the shared session. Both workbenches bind each id
# (tests/test_session_parity.py checks the browser; the desktop menus carry the
# same operations).
ACTIONS = (
    SessionAction("new_project", "New project", "File"),
    SessionAction("load_template", "Load template", "File"),
    SessionAction("open_project", "Open project", "File"),
    SessionAction("save_project", "Save project", "File"),
    SessionAction("import_support_mesh", "Import support mesh (STL)", "File"),
    SessionAction("undo", "Undo", "Edit"),
    SessionAction("redo", "Redo", "Edit"),
    SessionAction("apply_settings", "Apply settings", "Edit"),
    SessionAction("advanced_settings", "Advanced settings", "Edit"),
    SessionAction("select_channel", "Select channel", "Channels"),
    SessionAction("duplicate_channel", "Duplicate channel", "Channels"),
    SessionAction("remove_channel", "Remove channel", "Channels"),
    SessionAction("compute", "Compute", "Run"),
    SessionAction("compute_channels", "Compute enabled channels", "Run"),
    SessionAction("cancel_job", "Cancel", "Run"),
    SessionAction("open_run", "Open completed run", "Results"),
    SessionAction("show_dataset", "Plot dataset", "Results"),
    SessionAction("export_geometry", "Export conductor geometry", "Results"),
)
ACTION_IDS = frozenset(action.id for action in ACTIONS)


@dataclass(frozen=True)
class SettingField:
    """A form field over one or more configuration (or study) entries."""

    node: str
    key: str
    label: str
    kind: str  # number, integer, choice, text, bool, vector
    get: Callable[[Any, dict], Any]
    set: Callable[[Any, Any, dict], dict]
    choices: tuple = ()
    unit: str = ""
    help: str = ""
    study: bool = False


def _scaled(name, scale):
    return (
        lambda config, study: float(getattr(config, name)) * scale,
        lambda value, config, study: {name: float(value) / scale},
    )


def _plain(name, cast=None):
    return (
        lambda config, study: getattr(config, name),
        lambda value, config, study: {name: cast(value) if cast else value},
    )


def _inverted(name):
    return (lambda config, study: not bool(getattr(config, name)), lambda value, config, study: {name: not bool(value)})


def _vector(name):
    def parse(value, config, study):
        values = value if isinstance(value, (list, tuple)) else str(value).replace(";", ",").split(",")
        numbers = tuple(float(item) for item in values)
        if len(numbers) != 3:
            raise ValueError(f"{name} needs three comma-separated numbers.")
        return {name: numbers}

    return (lambda config, study: ", ".join(str(float(v)).removesuffix(".0") for v in getattr(config, name)), parse)


def _contour_treatment(config, study):
    from ..contours import _normalise_contour_path_regularization_mode
    from ..contour_smoothing import _normalise_scale_mode

    mode = _normalise_contour_path_regularization_mode(config.contour_path_regularization_mode)
    if mode == "parameterized_bspline":
        return _normalise_scale_mode(config.contour_smoothing_scale_mode)
    return mode


def _set_contour_treatment(value, config, study):
    if value == _contour_treatment(config, study):
        return {}
    if value in {"off", "resample"}:
        return {"contour_path_regularization_mode": value}
    if value in {"mesh_coupled", "legacy"}:
        return {"contour_path_regularization_mode": "parameterized_bspline", "contour_smoothing_scale_mode": value}
    raise ValueError("Choose one of the available contour path treatments.")


_CYLINDER_KEYS = {
    "length_mm": "cylinder_length_mm",
    "radius_mm": "cylinder_radius_mm",
    "angular_divisions": "cylinder_angular",
    "length_divisions": "cylinder_axial",
    "rotation_angle_deg": "cylinder_rotation_deg",
    **{f"rotation_axis_{axis}": f"cylinder_rotation_axis_{axis}" for axis in "xyz"},
}
_CYLINDER_LABELS = {
    "cylinder_length_mm": "Cylinder length",
    "cylinder_radius_mm": "Cylinder radius",
    "cylinder_angular": "Azimuthal divisions",
    "cylinder_rotation_deg": "Cylinder rotation angle",
}
_SUPPORT_PREFIXES = {
    "create cylinder mesh": "cylinder",
    "create bi-planary mesh": "biplanar",
    "create planary mesh": "planar",
    "create conical frustum mesh": "conical",
}
_SUPPORT_FIELDS = {
    (_CYLINDER_KEYS[field.key] if prefix == "cylinder" else f"{prefix}_{field.key}"): (editor, field)
    for editor in SUPPORT_EDITORS
    if (prefix := _SUPPORT_PREFIXES.get(editor.source))
    for field in editor.fields
    if prefix != "cylinder" or field.key in _CYLINDER_KEYS
}


def _support_edits(config, values, source):
    edits = {}
    for key, value in values.items():
        editor, field = _SUPPORT_FIELDS[key]
        parameters = getattr(config, editor.parameter_field)
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{field.label} must be finite.")
        if field.integer and not number.is_integer():
            raise ValueError(f"{field.label} must be a whole number.")
        if number != editor.display_value(field, parameters):
            if editor.source != source:
                raise ValueError("Apply or discard the dimension edits before changing support surfaces.")
            edits[field.key] = int(number) if field.integer else number
    if not edits:
        return {}
    editor = next(editor for editor in SUPPORT_EDITORS if editor.source == source)
    return {editor.parameter_field: editor.apply_edits(getattr(config, editor.parameter_field), edits)}


def _support_parameter(key):
    editor, field = _SUPPORT_FIELDS[key]
    return (
        lambda config, study: editor.display_value(field, getattr(config, editor.parameter_field)),
        lambda value, config, study: _support_edits(config, {key: value}, config.coil_mesh_file),
    )


def _linked_background_change(config, changes):
    """Track an MRI-field edit only when the uniform mechanical background was linked."""
    physics = config.research_physics
    if physics is None or physics.get("background_B_per_face_T") is not None:
        return {}
    direction = changes.get("b_0_direction", config.b_0_direction)
    strength = changes.get("b_0_strength_T", config.b_0_strength_T)
    if strength == config.b_0_strength_T and tuple(direction) == tuple(config.b_0_direction):
        return {}

    from ..research.convex import ResearchPhysicsSpec

    background = ResearchPhysicsSpec.from_dict(physics).b0_T
    previous = normalized_b0_direction(config.b_0_direction) * config.b_0_strength_T
    if background is None or not all(
        math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15) for a, b in zip(background, previous, strict=True)
    ):
        return {}
    return {"research_physics": {**physics, "b0_T": (normalized_b0_direction(direction) * strength).tolist()}}


def _study(name, cast):
    return (lambda config, study: study.get(name), lambda value, config, study: {name: cast(value)})


def _field(node, key, label, kind, accessors, **kwargs) -> SettingField:
    get, set_ = accessors
    return SettingField(node, key, label, kind, get, set_, **kwargs)


SETTINGS_NODES = (
    ("geometry", "Geometry and mesh"),
    ("target", "Target field and B0"),
    ("conductor", "Conductor and material"),
    ("optimization", "Stream function optimization"),
    ("winding", "Winding and routing"),
    ("validation", "Validation settings"),
)
MESH_SOURCES = tuple(_SUPPORT_PREFIXES)

SETTINGS = (
    _field("geometry", "coil_mesh_file", "Support surface", "choice", _plain("coil_mesh_file"), choices=MESH_SOURCES),
    *(
        _field(
            "geometry",
            key,
            _CYLINDER_LABELS.get(key, field.label),
            "integer" if field.integer else "number",
            _support_parameter(key),
            unit=field.suffix.strip(),
        )
        for key, (_editor, field) in _SUPPORT_FIELDS.items()
    ),
    _field(
        "geometry",
        "imported_mesh_length_scale_to_m",
        "Imported mesh unit scale",
        "number",
        _plain("imported_mesh_length_scale_to_m", float),
        help="0.001 for STL files in millimetres.",
    ),
    _field(
        "target",
        "field_shape_function",
        "Target expression (x, y, z in metres)",
        "text",
        _plain("field_shape_function", str),
    ),
    _field(
        "target",
        "target_field_component",
        "Target field component",
        "choice",
        _plain("target_field_component"),
        choices=TARGET_COMPONENTS,
    ),
    _field(
        "target",
        "target_gradient_strength",
        "Gradient amplitude",
        "number",
        _scaled("target_gradient_strength", 1000.0),
        unit="mT/m",
    ),
    _field(
        "target",
        "target_gradient_normalization",
        "Normalization",
        "choice",
        _plain("target_gradient_normalization"),
        choices=("auto", "off", "none", "linear_fit", "legacy"),
    ),
    _field(
        "target",
        "target_gradient_axis",
        "Gradient axis",
        "choice",
        _plain("target_gradient_axis"),
        choices=("x", "y", "z", "magnitude", "auto"),
        help="Use magnitude (|G|) for combined targets such as x+y. Current and error always use the full expression.",
    ),
    _field(
        "target", "target_region_radius_mm", "ROI radius", "number", _scaled("target_region_radius", 1000.0), unit="mm"
    ),
    _field("target", "target_region_resolution", "ROI resolution", "integer", _plain("target_region_resolution", int)),
    _field(
        "target",
        "design_context",
        "Design context",
        "choice",
        _plain("design_context"),
        choices=("mr_gradient", "mr_shim", "free", "mr_rf"),
    ),
    _field("target", "b_0_direction", "B0 direction (x, y, z)", "vector", _vector("b_0_direction")),
    _field("target", "b_0_strength_T", "B0 strength", "number", _plain("b_0_strength_T", float), unit="T"),
    _field(
        "conductor",
        "conductor_width_mm",
        "Conductor width",
        "number",
        _scaled("conductor_cross_section_width", 1000.0),
        unit="mm",
    ),
    _field(
        "conductor",
        "conductor_height_mm",
        "Conductor height",
        "number",
        _scaled("conductor_cross_section_height", 1000.0),
        unit="mm",
    ),
    _field(
        "conductor",
        "insulation_mm",
        "Insulation thickness",
        "number",
        _scaled("sweep_insulation_thickness", 1000.0),
        unit="mm",
    ),
    _field(
        "conductor",
        "conductor_mounting_mode",
        "Mounting reference",
        "choice",
        _plain("conductor_mounting_mode"),
        choices=("centered", "surface_outer", "surface_inner"),
    ),
    _field(
        "conductor",
        "conductor_path_reference",
        "Sweep reference point",
        "choice",
        _plain("conductor_path_reference"),
        choices=("centroid", "profile_origin"),
        help="profile_origin sweeps about the profile's (0, 0) point, e.g. a litz wire centre in an omega holder.",
    ),
    _field("conductor", "return_path_lift", "Lift crossing return paths", "bool", _inverted("skip_normal_shift")),
    _field(
        "conductor", "normal_shift_mm", "Return-path lift", "number", _scaled("normal_shift_length", 1000.0), unit="mm"
    ),
    _field(
        "conductor",
        "support_body_mode",
        "Grooved support body",
        "choice",
        _plain("support_body_mode"),
        choices=("off", "inside", "outside"),
        help="Thicken the support into a printable former and cut the conductor grooves into it. "
        "inside grows the solid away from surface_outer conductors; surface_inner conductors sit fully recessed.",
    ),
    _field(
        "conductor",
        "support_body_thickness_mm",
        "Support body thickness",
        "number",
        _scaled("support_body_thickness", 1000.0),
        unit="mm",
    ),
    _field(
        "conductor",
        "support_body_groove_clearance_mm",
        "Groove clearance",
        "number",
        _scaled("support_body_groove_clearance", 1000.0),
        unit="mm",
        help="Gap around the insulated conductor in the groove (print tolerance).",
    ),
    _field(
        "conductor",
        "support_body_groove_extra_depth_mm",
        "Extra groove depth",
        "number",
        _scaled("support_body_groove_extra_depth", 1000.0),
        unit="mm",
    ),
    _field(
        "optimization",
        "sf_opt_method",
        "Stream-function method",
        "choice",
        _plain("sf_opt_method"),
        choices=("tikhonov", "target_error_constrained_efficiency", "constrained_quadratic"),
    ),
    _field(
        "optimization", "tikonov_reg_factor", "Regularization factor", "number", _plain("tikonov_reg_factor", float)
    ),
    _field(
        "optimization",
        "target_error_tolerance_percent",
        "Target error tolerance",
        "number",
        _plain("target_error_tolerance_percent", float),
        unit="%",
    ),
    _field("winding", "levels", "Contour levels", "integer", _plain("levels", int)),
    _field(
        "winding",
        "contour_treatment",
        "Contour path treatment",
        "choice",
        (_contour_treatment, _set_contour_treatment),
        choices=("off", "mesh_coupled", "legacy", "resample"),
        help="Smooth closed contours on the support before routing. If no fit passes the limits, keep the extracted contour.",
    ),
    _field(
        "winding",
        "contour_smoothing_mesh_gain",
        "Gain at mesh wavelength",
        "number",
        _plain("contour_smoothing_mesh_gain", float),
        help="Between 0 and 1. Lower values suppress mesh-scale kinks more strongly.",
    ),
    _field(
        "winding",
        "contour_smoothing_factor",
        "Legacy smoothing strength",
        "number",
        _plain("contour_smoothing_factor", float),
        help="Zero disables the legacy fit. Increase toward 1 for stronger smoothing.",
    ),
    _field(
        "winding",
        "contour_displacement_mm",
        "Maximum contour displacement",
        "number",
        _scaled("contour_smoothing_max_displacement", 1000),
        unit="mm",
        help="Maximum allowed departure from the extracted contour. Zero removes the displacement cap.",
    ),
    _field(
        "winding",
        "contour_smoothing_preserve_min_turn_distance",
        "Guard inter-turn spacing during smoothing",
        "bool",
        _plain("contour_smoothing_preserve_min_turn_distance", bool),
        help="Reject fits that violate the requested spacing. With zero spacing, or when the original contours already miss the limit, preserve their original minimum separation.",
    ),
    _field(
        "winding",
        "contour_smoothing_local_fallback",
        "Try local corner blends if a fit is rejected",
        "bool",
        _plain("contour_smoothing_local_fallback", bool),
        help="Round individual corners on the support when a whole-contour fit fails. The displacement cap and spacing guard must both be active.",
    ),
    _field(
        "winding",
        "contour_segment_mm",
        "Target contour segment length",
        "number",
        _scaled("contour_resample_target_segment_length", 1000),
        unit="mm",
        help="Zero uses automatic B-spline sampling and disables resampling-only mode.",
    ),
    _field(
        "winding",
        "min_turn_distance_mm",
        "Minimum centerline spacing",
        "number",
        _scaled("min_turn_distance", 1000.0),
        unit="mm",
        help="Distance between centerlines before routing; the final copper-and-insulation clearance is checked separately.",
    ),
    _field(
        "winding", "min_bend_mm", "Minimum bend radius", "number", _scaled("sweep_min_bend_radius", 1000.0), unit="mm"
    ),
    _field(
        "winding",
        "interconnection_axis",
        "Interconnection direction",
        "choice",
        _plain("interconnection_axis"),
        choices=("b0", "support_axis"),
        help="support_axis routes cuts and returns along the bore; try it when the bore is perpendicular to B0.",
    ),
    _field("winding", "sweep", "Generate conductor solids", "bool", _inverted("skip_sweep")),
    _field(
        "validation",
        "validation_resolution",
        "Independent grid resolution",
        "integer",
        _study("validation_resolution", int),
        study=True,
    ),
    _field("validation", "design_margin", "Design budget ratio", "number", _study("design_margin", float), study=True),
    _field("validation", "fabricate", "Fabrication review", "bool", _study("fabricate", bool), study=True),
)
SETTINGS_BY_KEY = {item.key: item for item in SETTINGS}


@dataclass
class _History:
    undo: list = field(default_factory=list)
    redo: list = field(default_factory=list)


class WorkbenchSession:
    """Qt-free state of one user's workbench: model, history, jobs, runs.

    ``jobs`` is a :class:`coilgen.research.jobs.JobManager` (created lazily in
    ``workspace/jobs``) or any object with ``submit``/``list``/``cancel``.
    Listeners receive ``(event, payload)`` after every state change.
    """

    def __init__(self, workspace, *, jobs=None, policy=None, service=None, project_loader=None):
        from .session_evidence import load_assembly

        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.jobs_directory = self.workspace / "jobs"
        self._jobs = jobs
        self._policy = policy
        self.service = service or DesignService()
        self.project_loader = project_loader or load_assembly
        self.assembly: ProjectAssembly | None = None
        self.active_channel_id: str | None = None
        self.project_path: Path | None = None
        self._save_paths: dict[str, tuple[Path, tuple]] = {}
        self.history = _History()
        self.job_ids: list[str] = []
        self.states: dict[str, dict] = {}
        self._embedded_job_ids: dict[str, str] = {}
        self._listeners: list[Callable[[str, Any], None]] = []
        if self._jobs is not None or any(self.jobs_directory.glob("*/job.json")):
            self.poll()
        from .session_evidence import restore_embedded

        self._add_embedded(restore_embedded(self.workspace / "embedded-runs"))

    # -- notifications -------------------------------------------------
    def subscribe(self, listener):
        self._listeners.append(listener)
        return listener

    def _emit(self, event, payload=None):
        for listener in tuple(self._listeners):
            listener(event, payload)

    # -- model ---------------------------------------------------------
    @property
    def project(self) -> ProjectDocument | None:
        if self.assembly is None or self.active_channel_id is None:
            return None
        return self.assembly.channel(self.active_channel_id).project

    @property
    def channels(self):
        return () if self.assembly is None else self.assembly.channels

    def _set_assembly(self, assembly, *, record=True, channel_id=None):
        if record and self.assembly is not None:
            self.history.undo.append((self.assembly, self.active_channel_id))
            self.history.redo.clear()
        self.assembly = assembly
        ids = [channel.id for channel in assembly.channels]
        self.active_channel_id = (
            channel_id
            if channel_id in ids
            else self.active_channel_id
            if self.active_channel_id in ids
            else ids[0]
            if ids
            else None
        )
        self._emit("model", self.assembly)

    def templates(self):
        from .templates import TEMPLATE_SPECS

        return TEMPLATE_SPECS

    def load_template(self, template_id: str):
        from .templates import template_assembly

        projects = self.workspace / "projects"
        projects.mkdir(exist_ok=True)
        self.project_path = None
        self._save_paths.clear()
        self.history = _History()
        self._set_assembly(template_assembly(template_id, project_root=projects), record=False)
        return self.assembly

    def new_project(self, name: str = "New design"):
        from .presets import preset_config, preset_study

        projects = self.workspace / "projects"
        projects.mkdir(exist_ok=True)
        document = ProjectDocument.from_config(preset_config("cylinder"), name=name, project_root=projects)
        self.project_path = None
        self._save_paths.clear()
        self.history = _History()
        self._set_assembly(ProjectAssembly.new(name, (document.update_study(preset_study("cylinder")),)), record=False)
        return self.assembly

    def open_project(self, path, *, loader=None):
        """Verify the project and its portable evidence before replacing the current model."""
        from .session_evidence import open_embedded, retain_revision

        path = Path(path)
        assembly = (loader or self.project_loader)(path)
        embedded = open_embedded(assembly, self.workspace / "embedded-runs")
        for channel in assembly.channels:
            retain_revision(self.workspace, channel.project)
        self._add_embedded(embedded)
        self.project_path = path
        self._save_paths.clear()
        # Browser format buttons own files in projects/. Imported originals
        # keep their legacy explicit path for normal programmatic saves.
        self._remember_save(path)
        self.history = _History()
        active_id = assembly.channels[0].project.session.get("active_channel_id")
        self._set_assembly(assembly, record=False, channel_id=active_id)
        self._emit("jobs", self.job_states())
        return assembly

    def _remember_save(self, path):
        path = path.resolve()
        if path.parent == self.workspace / "projects":
            kind = "bundle" if path.suffix.lower() == ".coilgen" else "json"
            self._save_paths[kind] = (path, _file_version(path))

    def save_project(self, path=None, *, format=None, selected_run=...) -> Path:
        """Keep an existing portable selection unless the client supplies its current run or None."""
        from .session_evidence import portable_assembly, save_new_assembly

        if self.assembly is None:
            raise ValueError("Create or open a project before saving.")
        extensions = {"bundle": ".coilgen", "json": ".coilgen.json"}
        if format is not None and format not in extensions:
            raise ValueError("Project format must be 'bundle' or 'json'.")
        target = self.project_path
        if path is not None:
            target = Path(path)
        elif format is not None:
            target = None
            if format in self._save_paths:
                previous, version = self._save_paths[format]
                if previous.resolve() != previous:
                    raise ValueError("The previous project save now redirects to another path.")
                if previous.is_file() and _file_version(previous) == version:
                    target = previous
        if path is not None and format is not None:
            expected = ".coilgen" if format == "bundle" else ".json"
            if target.suffix.lower() != expected:
                raise ValueError(f"An explicit {format} destination must end in {expected}.")
        projects = self.workspace / "projects"
        if path is None and (format is not None or target is None):
            if projects.resolve() != projects or (target is not None and target.resolve().parent != projects):
                raise ValueError("Browser project saves must stay in this workspace's projects directory.")
        self.poll()
        portable = portable_assembly(
            self.assembly,
            self.completed_runs(),
            self.load_run,
            self.workspace,
            self.active_channel_id,
            self.project_loader,
            selected_run=selected_run,
        )
        saved = (
            save_new_assembly(portable, projects, _safe_name(portable.name), extensions[format or "json"])
            if target is None
            else Path(portable.save(target))
        )
        self.assembly = portable
        self.project_path = saved
        self._remember_save(saved)
        self._emit("saved", saved)
        return saved

    def select_channel(self, channel_id: str):
        self.assembly.channel(channel_id)
        self.active_channel_id = channel_id
        self._emit("model", self.assembly)

    def duplicate_channel(self, channel_id: str | None = None):
        self._require_project()
        source = self.assembly.channel(channel_id or self.active_channel_id)
        existing = {channel.name.strip().casefold() for channel in self.channels}
        index = 1
        while True:
            suffix = " copy" if index == 1 else f" copy {index}"
            name = source.name[: 120 - len(suffix)].rstrip() + suffix
            if name.casefold() not in existing:
                break
            index += 1
        project = source.project.duplicate(name)
        assembly = self.assembly.add_channel(project, name=name)
        self._set_assembly(assembly, channel_id=assembly.channels[-1].id)
        return project

    def remove_channel(self, channel_id: str | None = None):
        if len(self.channels) <= 1:
            raise ValueError("A project keeps at least one channel.")
        self._set_assembly(self.assembly.remove_channel(channel_id or self.active_channel_id))

    def set_channel_enabled(self, channel_id: str, enabled: bool):
        self._set_assembly(self.assembly.update_channel(channel_id, enabled=bool(enabled)))

    def undo(self):
        if not self.history.undo:
            return False
        self.history.redo.append((self.assembly, self.active_channel_id))
        assembly, channel = self.history.undo.pop()
        self._set_assembly(assembly, record=False, channel_id=channel)
        return True

    def redo(self):
        if not self.history.redo:
            return False
        self.history.undo.append((self.assembly, self.active_channel_id))
        assembly, channel = self.history.redo.pop()
        self._set_assembly(assembly, record=False, channel_id=channel)
        return True

    def _replace_project(self, project):
        self._set_assembly(self.assembly.update_channel(self.active_channel_id, project=project))

    # -- settings ------------------------------------------------------
    def settings_fields(self, node: str | None = None):
        """Form fields applicable to the current source; STL scaling only applies to raw STL input."""
        if self.project is None:
            return ()
        config = self.project.config
        source = str(config.coil_mesh_file)
        inactive = set()
        if config.skip_postprocessing or config.contour_only_sweep_mode:
            inactive.update(("interconnection_axis", "return_path_lift", "normal_shift_mm"))
        if str(getattr(config, "support_body_mode", "off")) == "off":
            inactive.update(
                (
                    "support_body_thickness_mm",
                    "support_body_groove_clearance_mm",
                    "support_body_groove_extra_depth_mm",
                )
            )
        if self.project.study.get("requirements"):
            inactive.update(
                (
                    "field_shape_function",
                    "target_field_component",
                    "target_gradient_normalization",
                    "target_gradient_axis",
                    "target_gradient_strength",
                    "target_region_radius_mm",
                    "target_region_resolution",
                )
            )
        elif str(config.target_field_definition_file).lower() not in {"none", ""}:
            inactive.update(("field_shape_function", "target_region_radius_mm", "target_region_resolution"))
        elif config.target_region_geometry is not None or str(config.target_mesh_file).lower() not in {"none", ""}:
            inactive.add("target_region_radius_mm")
        if config.design_context != "mr_gradient" and str(config.target_gradient_normalization).lower() in {
            "off",
            "none",
        }:
            inactive.update(("target_gradient_strength", "target_gradient_axis"))
        return tuple(
            item
            for item in SETTINGS
            if (node is None or item.node == node)
            and item.key not in inactive
            and (item.key not in _SUPPORT_FIELDS or _SUPPORT_FIELDS[item.key][0].source == source)
            and (item.key != "imported_mesh_length_scale_to_m" or source.lower().endswith(".stl"))
        )

    def settings_values(self, node: str | None = None) -> dict:
        project = self._require_project()
        config = project.config
        return {item.key: item.get(config, project.study) for item in self.settings_fields(node)}

    def apply_settings(self, values: dict):
        """Apply form values; raises ValueError (nothing changed) when invalid."""
        project = self._require_project()
        config = project.config
        geometry = {key: value for key, value in values.items() if key in _SUPPORT_FIELDS}
        source = values.get("coil_mesh_file", config.coil_mesh_file)
        config_changes, study_changes = _support_edits(config, geometry, source), {}
        for key, value in values.items():
            if key in geometry:
                continue
            item = SETTINGS_BY_KEY[key]
            changes = item.set(value, config, project.study)
            (study_changes if item.study else config_changes).update(changes)
        if source != config.coil_mesh_file:
            config_changes["surface_is_cylinder_flag"] = source == "create cylinder mesh"
        return self.apply_inputs(config_changes, study_changes)

    def apply_inputs(self, config_changes, study_changes=None):
        """Commit a single revision, preserving deliberately distinct mechanical backgrounds."""
        project = self._require_project()
        changes = dict(config_changes)
        if "research_physics" not in changes and {"b_0_direction", "b_0_strength_T"}.intersection(changes):
            changes.update(_linked_background_change(project.config, changes))
        updated = project.update_inputs(changes, study_changes)
        if updated is not project:
            self._replace_project(updated)
        return updated

    def apply_config(self, changes: dict):
        """Advanced settings: any CoilGenConfig field, validated by the engine."""
        return self.apply_inputs(changes)

    def apply_profile_preset(self, kind: str):
        from .conductor_profiles import omega_holder_profile, round_wire_profile

        project = self._require_project()
        width = float(project.config.conductor_cross_section_width)
        if kind == "rectangle":
            # The default rectangle sentinel follows conductor width and height.
            changes = {"cross_sectional_points": type(project.config)().cross_sectional_points}
        elif kind == "round":
            changes = {"cross_sectional_points": round_wire_profile(width)}
        elif kind == "omega":
            changes = {
                "cross_sectional_points": omega_holder_profile(width),
                "conductor_path_reference": "profile_origin",
            }
        else:
            raise ValueError(f"Unknown conductor profile preset {kind!r}.")
        self._replace_project(project.update_config(changes))

    def import_support_mesh(self, path, *, length_scale_to_m: float = 1.0):
        from ..geometry import read_stl, normalize_mesh

        project = self._require_project()
        mesh = read_stl(Path(path))
        mesh.vertices = mesh.vertices * float(length_scale_to_m)
        mesh.metadata = {**(getattr(mesh, "metadata", None) or {}), "source_file": Path(path).name, "units": "SI"}
        self._replace_project(
            project.with_mesh(normalize_mesh(mesh), field="coil_mesh_file", changes={"surface_is_cylinder_flag": False})
        )

    def preflight(self, pipeline_stage="evaluated"):
        return check_preflight(self._require_project(), pipeline_stage=pipeline_stage)

    # -- computation ---------------------------------------------------
    @property
    def jobs(self):
        if self._jobs is None:
            from ..research.jobs import JobManager, RuntimePolicy

            self._jobs = JobManager(self.jobs_directory, self._policy or RuntimePolicy())
        return self._jobs

    def controls(self, pipeline_stage="evaluated", project=None):
        project = project or self._require_project()
        study = project.study
        return request_controls(
            project,
            pipeline_stage=pipeline_stage,
            validation_resolution=study["validation_resolution"],
            design_margin=study["design_margin"],
            fabricate=study["fabricate"],
        )

    def _at_stage(self, project, stage):
        if stage not in dict(PIPELINE_STAGES):
            raise ValueError(f"Unknown pipeline stage {stage!r}.")
        current = project.study
        if stage == current["pipeline_stage"]:
            return project
        session = project.session
        full_review = (
            current["fabricate"]
            if current["pipeline_stage"] == "evaluated"
            else session.get("full_fabrication", current["fabricate"])
        )
        updated = project.update_study(stage_study_changes(stage, full_review))
        return updated.with_results(updated.saved_runs, session={**session, "full_fabrication": bool(full_review)})

    def set_stage(self, stage: str):
        """Select the pipeline boundary as a study edit, as the desktop does."""
        project = self._require_project()
        updated = self._at_stage(project, stage)
        if updated is not project:
            self._replace_project(updated)
        return self.project

    def compute(self, pipeline_stage=None, *, channel_id=None):
        """Queue the active channel at ``pipeline_stage`` (default: its study)."""
        if pipeline_stage is not None and channel_id is None:
            self.set_stage(pipeline_stage)
        project = self.assembly.channel(channel_id).project if channel_id is not None else self._require_project()
        pipeline_stage = project.study["pipeline_stage"]
        report = check_preflight(project, pipeline_stage=pipeline_stage)
        if not report.ready:
            raise ValueError("; ".join(issue.message for issue in report.blocking))
        from .session_evidence import retain_revision

        retain_revision(self.workspace, project)
        state = submit_design(project, self.jobs, self.controls(pipeline_stage, project), service=self.service)
        state.setdefault("channel_id", channel_id or self.active_channel_id)
        if state["id"] not in self.states:
            self.job_ids.append(state["id"])
        self.states[state["id"]] = state
        self._emit("jobs", self.job_states())
        return state

    def compute_channels(self, pipeline_stage=None):
        """Submit enabled channels at the requested boundary, or at their individually saved boundaries."""
        enabled = [channel for channel in self.channels if channel.enabled]
        if not enabled:
            raise ValueError("Enable at least one channel before computing the assembly.")
        assembly = self.assembly
        for channel in enabled:
            project = self._at_stage(channel.project, pipeline_stage) if pipeline_stage is not None else channel.project
            report = check_preflight(project, pipeline_stage=project.study["pipeline_stage"])
            if not report.ready:
                raise ValueError(f"{channel.name}: " + "; ".join(issue.message for issue in report.blocking))
            if project is not channel.project:
                assembly = assembly.update_channel(channel.id, project=project)
        if assembly is not self.assembly:
            self._set_assembly(assembly)
        from .session_evidence import retain_revision

        for channel in assembly.channels:
            if channel.enabled:
                retain_revision(self.workspace, channel.project)
        states = self.service.submit_assembly(assembly, self.jobs)
        for channel_id, state in states.items():
            state.setdefault("channel_id", channel_id)
            if state["id"] not in self.states:
                self.job_ids.append(state["id"])
            self.states[state["id"]] = state
        self._emit("jobs", self.job_states())
        return states

    def poll(self):
        if self._jobs is None and not self.jobs_directory.exists():
            return self.job_states()
        for state in sorted(self.jobs.list(), key=lambda job: job.get("created_at", "")):
            job_id = state["id"]
            channel = self.states.get(job_id, {}).get("channel_id")
            if job_id not in self.states:
                self.job_ids.append(job_id)
            self.states[job_id] = {**state, **({"channel_id": channel} if channel else {})}
        states = self.job_states()
        self._emit("jobs", states)
        return states

    def job_states(self):
        return [self.states[id_] for id_ in reversed(self.job_ids) if id_ in self.states]

    def active(self) -> bool:
        return any(state.get("status") in ACTIVE_STATUSES for state in self.states.values())

    def cancel_job(self, job_id: str):
        state = self.jobs.cancel(job_id)
        self.states[job_id] = {**self.states.get(job_id, {}), **state}
        self._emit("jobs", self.job_states())
        return state

    # -- results -------------------------------------------------------
    def _add_embedded(self, states):
        from .evidence import verified_run_metadata
        from .session_evidence import run_digest

        for state in states:
            # Two executions can share a request. Match immutable result metadata
            # before suppressing the local copy of an exported execution.
            matches = (
                item
                for item in self.states.values()
                if not item.get("embedded")
                and item.get("status") == "completed"
                and item.get("request_sha256") == state.get("request_sha256")
            )
            for match in matches:
                try:
                    metadata, _, _ = verified_run_metadata(self.jobs_directory / match["id"])
                except (OSError, TypeError, ValueError, RuntimeError):
                    continue
                if match.get("project_id") == metadata.get("project_id") == state.get("project_id") and run_digest(
                    metadata
                ) == state.get("evidence_digest"):
                    self._embedded_job_ids[state["id"]] = match["id"]
                    break
            else:
                if state["id"] not in self.states:
                    self.job_ids.append(state["id"])
                self.states[state["id"]] = state
                self._embedded_job_ids[state["id"]] = state["id"]

    def saved_run_job(self):
        """Resolve this channel's explicit portable selection to its verified workspace job."""
        project = self.project
        if project is None:
            return None
        archive_id = project.session.get("selected_run_id")
        if not isinstance(archive_id, str) or not any(
            archive.id == archive_id and archive.project_id == project.id for archive in project.saved_runs
        ):
            return None
        job_id = self._embedded_job_ids.get(archive_id)
        state = self.states.get(job_id, {})
        return job_id if state.get("status") == "completed" and state.get("project_id") == project.id else None

    def completed_runs(self):
        return [state for state in self.job_states() if state.get("status") == "completed"]

    def load_run(self, job_id: str):
        state = self.states.get(job_id)
        if state is None or state.get("status") != "completed":
            raise ValueError("Select a completed run from this workspace.")
        root = self.workspace / "embedded-runs" if state.get("embedded") else self.jobs_directory
        return DesignService.load_run(root / job_id)

    def run_summary(self, record) -> dict:
        from .diagnostics import run_verdicts

        metadata = record.metadata
        summary = dict(metadata.get("summary") or {})
        fabrication = metadata.get("fabrication") or {}
        return {
            "accepted": summary.get("accepted"),
            "construction_released": summary.get("construction_released") is True,
            "verdicts": run_verdicts(metadata),
            "failure_reasons": list(fabrication.get("failure_reasons") or ()),
            "warnings": list(metadata.get("warnings") or ())[:20],
            "metrics": {
                key: value
                for key, value in summary.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            },
        }

    def export_options(self, record):
        from .cad import export_catalog

        return export_catalog(record)

    def export(self, record, names, *, unit="mm", allow_draft=False, include_project=False, destination=None):
        from .cad import export_geometry
        from .session_evidence import source_project

        original = (
            source_project(self.workspace, record, self.channels, self.project_loader, search_saved=True)
            if include_project
            else None
        )
        if include_project and original is None:
            raise ValueError(
                "The selected run's exact source project revision is unavailable. Open that revision or export geometry without a project companion."
            )
        exports = self.workspace / "exports"
        exports.mkdir(exist_ok=True)
        if destination is None:
            stamp = record.metadata.get("project_revision", "run")
            metadata = record.metadata_view
            name = _safe_name(
                metadata.get("name")
                or (metadata.get("job") or {}).get("name")
                or (original.name if original else "design")
            )
            index = 1
            while (exports / f"{name}-r{stamp}-{index}").exists():
                index += 1
            destination = exports / f"{name}-r{stamp}-{index}"
        return export_geometry(
            record,
            destination,
            names,
            unit=unit,
            project=original,
            allow_draft=allow_draft,
        )

    def close(self):
        if self._jobs is not None and hasattr(self._jobs, "close"):
            self._jobs.close()

    def _require_project(self) -> ProjectDocument:
        if self.project is None:
            raise ValueError("Create a project or load a template first.")
        return self.project


def _safe_name(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "-" for character in str(value))
    stem = cleaned.strip("-")[:60] or "design"
    return f"design-{stem}" if PureWindowsPath(stem).is_reserved() else stem


def _file_version(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
