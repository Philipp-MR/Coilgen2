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
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Callable
from .session_model import SessionModel, ModelHistory
from .assembly import ProjectAssembly
from . import cylinder_pair as _pair
from .execution import plan_execution
from .preflight import check_preflight
from .project import ProjectDocument
from .service import DesignService
from .targets import ORIGIN_KEYS, recognize_solid_harmonic_target, translated_expression
from .settings import (
    SettingField as SettingField,
    _scaled as _scaled,
    _plain as _plain,
    _inverted as _inverted,
    _vector as _vector,
    target_scaling_mode as target_scaling_mode,
    _target_region as _target_region,
    _origin_parameter as _origin_parameter,
    _target_region_values as _target_region_values,
    _target_region_edits as _target_region_edits,
    _target_region_parameter as _target_region_parameter,
    _contour_treatment as _contour_treatment,
    _set_contour_treatment as _set_contour_treatment,
    _support_edits as _support_edits,
    _support_parameter as _support_parameter,
    _linked_background_change as _linked_background_change,
    _study as _study,
    _field as _field,
    _pair_fields as _pair_fields,
    _TARGET_CENTER_FIELDS as _TARGET_CENTER_FIELDS,
    _TARGET_RADIUS_FIELD as _TARGET_RADIUS_FIELD,
    _TARGET_REGION_FIELDS as _TARGET_REGION_FIELDS,
    _CYLINDER_KEYS as _CYLINDER_KEYS,
    _CYLINDER_LABELS as _CYLINDER_LABELS,
    _SUPPORT_PREFIXES as _SUPPORT_PREFIXES,
    _SUPPORT_FIELDS as _SUPPORT_FIELDS,
    SETTINGS_NODES as SETTINGS_NODES,
    MESH_SOURCES as MESH_SOURCES,
    TERMINAL_STRATEGIES as TERMINAL_STRATEGIES,
    SETTINGS as SETTINGS,
    SETTINGS_BY_KEY as SETTINGS_BY_KEY,
)


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


def submit_recorded_design(project, jobs, controls, *, workspace, states, channel_id=None, **options):
    """Retain the submitted revision and register an admitted job for either client.

    Failed submissions never enter the session's state map. Reused jobs retain
    their identity, so clients can distinguish following a run from new work.
    """
    from .session_evidence import retain_revision

    retain_revision(workspace, project)
    state = submit_design(project, jobs, controls, **options)
    if channel_id is not None:
        state.setdefault("channel_id", channel_id)
    is_new = state["id"] not in states
    states[state["id"]] = state
    return state, is_new


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


_History = ModelHistory


class ProjectSettings:
    """Pure project settings adapter; clients own history and computation."""

    def __init__(self, project):
        self.project = project
        self.allow_incomplete = True

    def _require_project(self):
        if self.project is None:
            raise ValueError("Open a project first.")
        return self.project

    def _replace_project(self, project):
        self.project = project

    def settings_fields(self, node: str | None = None, *, support_source=None, former_mode=None):
        """Return fields applicable to the selected support and former, including drafts."""
        if self.project is None:
            return ()
        config = self.project.config
        source = str(_pair.displayed_support_source(self.project) if support_source is None else support_source)
        # Preserve the existing mT/m setter for scripted form callers. The
        # interactive amplitude field uses the expression's native units.
        inactive = {"target_gradient_strength", *_pair.hidden_keys(self.project, source)}
        if config.skip_postprocessing or config.contour_only_sweep_mode:
            inactive.update(("interconnection_axis", "return_path_lift", "normal_shift_mm"))
        if (config.support_body_mode if former_mode is None else former_mode) == "off":
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
                    "target_field_units",
                    "target_amplitude_native",
                    *_TARGET_REGION_FIELDS,
                    "target_region_resolution",
                )
            )
        elif str(config.target_field_definition_file).lower() not in {"none", ""}:
            inactive.update(("field_shape_function", *_TARGET_REGION_FIELDS, "target_region_resolution"))
        else:
            try:
                region = _target_region(config)
            except (TypeError, ValueError):
                inactive.update(_TARGET_REGION_FIELDS)
            else:
                if region is not None:
                    if region.radius_m is None:
                        inactive.add(_TARGET_RADIUS_FIELD)
                    if region.shape == "enclosing_mesh":
                        inactive.update(_TARGET_CENTER_FIELDS)
                elif str(config.target_mesh_file).lower() not in {"none", ""}:
                    inactive.update(_TARGET_REGION_FIELDS)
                elif config.set_roi_into_mesh_center:
                    inactive.update(_TARGET_CENTER_FIELDS)
        return tuple(
            item
            for item in SETTINGS
            if (
                node is None
                or item.node == node
                or (node == "winding" and item.key in {"return_path_lift", "normal_shift_mm"})
            )
            and item.key not in inactive
            and (
                item.key not in ORIGIN_KEYS
                or (
                    "field_shape_function" not in inactive
                    and recognize_solid_harmonic_target(config.field_shape_function) is not None
                )
            )
            and (item.key not in _SUPPORT_FIELDS or _SUPPORT_FIELDS[item.key][0].source == source)
            and (item.key != "imported_mesh_length_scale_to_m" or source.lower().endswith(".stl"))
        )

    def settings_values(self, node: str | None = None, *, support_source=None, former_mode=None) -> dict:
        project = self._require_project()
        config = project.config
        values = {
            item.key: item.get(config, project.study)
            for item in self.settings_fields(node, support_source=support_source, former_mode=former_mode)
        }
        return _pair.form_values(project, values)

    def apply_settings(self, values: dict, extra_changes=None):
        """Apply form values; raises ValueError (nothing changed) when invalid.

        ``extra_changes`` are config edits committed in the same revision, such
        as clearing the shielding target after a confirmed support switch.
        """
        values, project = _pair.apply_pair_settings(self._require_project(), values)
        config = project.config
        if any(key in values for key in ORIGIN_KEYS):
            expression = translated_expression(config.field_shape_function, values)
            values = {key: value for key, value in values.items() if key not in ORIGIN_KEYS}
            values["field_shape_function"] = expression
        geometry = {key: value for key, value in values.items() if key in _SUPPORT_FIELDS}
        target_region = {key: value for key, value in values.items() if key in _TARGET_REGION_FIELDS}
        source = values.get("coil_mesh_file", config.coil_mesh_file)
        config_changes, study_changes = _support_edits(config, geometry, source), {}
        config_changes.update(_target_region_edits(config, target_region))
        for key, value in values.items():
            if key in geometry or key in target_region:
                continue
            if key == "target_amplitude_native" and target_scaling_mode(config, values) == "none":
                continue
            item = SETTINGS_BY_KEY[key]
            changes = item.set(value, config, project.study)
            (study_changes if item.study else config_changes).update(changes)
        if source != config.coil_mesh_file:
            config_changes["surface_is_cylinder_flag"] = source == "create cylinder mesh"
        config_changes.update(extra_changes or {})
        return self.apply_inputs(config_changes, study_changes, base=project)

    def apply_inputs(self, config_changes, study_changes=None, *, base=None):
        """Commit a single revision, preserving deliberately distinct mechanical backgrounds.

        ``base`` is a revision of the current project already prepared by the caller.
        """
        project = self._require_project()
        start = project if base is None else base
        changes = dict(config_changes)
        if "research_physics" not in changes and {"b_0_direction", "b_0_strength_T"}.intersection(changes):
            changes.update(_linked_background_change(start.config, changes))
        updated = start.update_inputs(changes, study_changes, allow_incomplete=getattr(self, "allow_incomplete", False))
        if updated is not project:
            self._replace_project(updated)
        return updated


class WorkbenchSession(ProjectSettings):
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
        self.model = SessionModel()
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
        return self.model.project

    @property
    def assembly(self):
        return self.model.assembly

    @assembly.setter
    def assembly(self, value):
        self.model.install(value)

    @property
    def active_channel_id(self):
        return self.model.channel_id

    @property
    def channels(self):
        return () if self.assembly is None else self.assembly.channels

    def _set_assembly(self, assembly, *, record=True, channel_id=None):
        edit = self.model.install(assembly, channel_id)
        if record:
            self.history.record(edit)
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
        self.model.select(channel_id)
        self._emit("model", self.assembly)

    def duplicate_channel(self, channel_id: str | None = None):
        self._require_project()
        edit = self.model.duplicate(channel_id=channel_id)
        self._set_assembly(edit.after.assembly, channel_id=edit.after.channel_id)
        return edit.after.project

    def remove_channel(self, channel_id: str | None = None):
        edit = self.model.removal(channel_id)
        self._set_assembly(edit.after.assembly, channel_id=edit.after.channel_id)

    def set_channel_enabled(self, channel_id: str, enabled: bool):
        self._set_assembly(self.assembly.update_channel(channel_id, enabled=bool(enabled)))

    def undo(self):
        snapshot = self.history.step()
        if snapshot is None:
            return False
        self._set_assembly(snapshot.assembly, record=False, channel_id=snapshot.channel_id)
        return True

    def redo(self):
        snapshot = self.history.step(redo=True)
        if snapshot is None:
            return False
        self._set_assembly(snapshot.assembly, record=False, channel_id=snapshot.channel_id)
        return True

    def _replace_project(self, project):
        self._set_assembly(self.assembly.update_channel(self.active_channel_id, project=project))

    # -- settings ------------------------------------------------------
    def apply_config(self, changes: dict):
        """Advanced settings: any CoilGenConfig field, validated by the engine."""
        return self.apply_inputs(changes)

    def apply_fix(self, key: str):
        """Apply the one-click fix of the current preflight tip for ``key``."""
        project = self._require_project()
        report = self.preflight(project.study["pipeline_stage"])
        issue = next((item for item in report.issues if item.key == key and item.fix), None)
        if issue is None:
            raise ValueError("This tip has no automatic fix any more.")
        return self.apply_inputs(dict(issue.fix))

    def apply_profile_preset(
        self, kind: str, *, segments: int = 16, equal_area: bool = False, diameter_mm: float | str | None = None
    ):
        """Replace the cross-section with a preset built from conductor width/height.

        ``segments`` (points around a full circle) applies to the round wire
        and the U-shape half circle; ``equal_area`` keeps the round wire area.
        An explicit round diameter updates both nominal dimensions in the same revision.
        """
        from .conductor_profiles import omega_holder_profile, round_wire_profile, u_shape_profile

        project = self._require_project()
        width = float(project.config.conductor_cross_section_width)
        height = float(project.config.conductor_cross_section_height)
        if kind == "rectangle":
            # The default rectangle sentinel follows conductor width and height.
            changes = {"cross_sectional_points": type(project.config)().cross_sectional_points}
        elif kind == "round":
            if diameter_mm is not None:
                try:
                    if isinstance(diameter_mm, bool):
                        raise ValueError
                    millimetres = float(diameter_mm)
                    if millimetres != width * 1e3:
                        width = millimetres * 1e-3
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("Enter a positive round wire diameter in mm.") from exc
            changes = {"cross_sectional_points": round_wire_profile(width, segments, equal_area=equal_area)}
            if diameter_mm is not None:
                changes.update(conductor_cross_section_width=width, conductor_cross_section_height=width)
        elif kind == "u_shape":
            changes = {"cross_sectional_points": u_shape_profile(width, height, segments)}
        elif kind == "omega":
            changes = {
                "cross_sectional_points": omega_holder_profile(width),
                "conductor_path_reference": "profile_origin",
            }
        else:
            raise ValueError(f"Unknown conductor profile preset {kind!r}.")
        return self.apply_config(changes)

    def import_profile_table(self, path, *, length_scale_to_m: float = 1e-3):
        """Replace the cross-section with a two-column (u, n) table, millimetres by default."""
        from .conductor_profiles import read_profile_table, validate_conductor_profile

        project = self._require_project()
        points = validate_conductor_profile(read_profile_table(path, length_scale_to_m=length_scale_to_m)).points_m
        self._replace_project(project.update_config({"cross_sectional_points": points}))

    def import_support_mesh(self, path, *, length_scale_to_m: float = 1.0, part_roles=None):
        """Own the scaled STL; ``part_roles`` gives one role per body in pipeline order.

        Without ``part_roles`` a support of two or more bodies gets the default
        roles (inner radius group primary, outer group active shield), so a
        shielded pair never loses its primary/shield split on import.
        """
        from .support_import import default_part_roles, read_support_stl, support_bodies, with_part_roles

        project = self._require_project()
        mesh = read_support_stl(path, length_scale_to_m)
        if part_roles is None:
            bodies = support_bodies(mesh)
            part_roles = default_part_roles(bodies) if len(bodies) >= 2 else None
        mesh = with_part_roles(mesh, part_roles)
        self._replace_project(
            project.with_mesh(mesh, field="coil_mesh_file", changes={"surface_is_cylinder_flag": False})
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

    def compute(self, pipeline_stage=None, *, channel_id=None, temporary_stage=False):
        """Queue the active channel at ``pipeline_stage`` (default: its study)."""
        if pipeline_stage is not None and channel_id is None and not temporary_stage:
            self.set_stage(pipeline_stage)
        project = self.assembly.channel(channel_id).project if channel_id is not None else self._require_project()
        pipeline_stage = pipeline_stage if temporary_stage else project.study["pipeline_stage"]
        report = check_preflight(project, pipeline_stage=pipeline_stage)
        if not report.ready:
            raise ValueError("; ".join(issue.message for issue in report.blocking))
        controls = self.controls(pipeline_stage, project)
        if temporary_stage:
            controls = {**controls, "fabricate": False}
        state, is_new = submit_recorded_design(
            project,
            self.jobs,
            controls,
            workspace=self.workspace,
            states=self.states,
            channel_id=channel_id or self.active_channel_id,
            service=self.service,
        )
        if is_new:
            self.job_ids.append(state["id"])
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
