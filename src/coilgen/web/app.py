"""One user's browser workbench (trame + Vuetify), over the shared session.

Every action calls :class:`WorkbenchSession`; this module only translates
between browser state and that session. Rendering uses
:func:`coilgen.application.scene_spec.scene_layers`, the dataset menu and the
B0 marker of the desktop workbench.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from functools import partial
from collections.abc import Mapping
from hashlib import sha256
import json
from io import BytesIO
from pathlib import Path
import shutil
import tempfile
from threading import Event
from urllib.parse import quote
import zipfile

import numpy as np

from ..application.cad import (
    GEOMETRY_FORMAT_LABELS,
    default_export_selection,
    draft_export_notice,
    geometry_option_label,
    is_draft_artifact,
)
from ..application.scene_spec import (
    B0_INDICATOR_COLOR,
    DATASET_GROUPS,
    DATASET_LABELS,
    PREFERRED_RUN_PLOTS,
    DENSITY_CMAP,
    FIELD_CMAP,
    b0_indicator_mesh,
    model_preview,
    scene_layers,
)
from ..application.diagnostics import recorded_stage
from ..application.evidence import FIELD_KEYS, load_saved_request, resolve_dataset
from ..application.project import _json_object
from ..application.session import (
    ACTIVE_STATUSES,
    PIPELINE_STAGES,
    SETTINGS_NODES,
    WorkbenchSession,
)
from ..attribution import AUTHOR_CREDIT, PROJECT_URL
from ..application.support_parameters import SUPPORT_EDITORS
from ..field_components import normalized_b0_direction
from ..geometry import geometry_factory
from ..persistence import atomic_text
from .input_policy import (
    PROJECT_UPLOAD_LIMIT,
    check_browser_inputs,
    checked_upload_path,
    claim_project_upload,
    load_browser_assembly,
    upload_name,
)

# Browser handlers by session action id; tests check that each id is bound.
BOUND_ACTIONS: dict[str, str] = {}
INLINE_REQUEST_BYTES = 256 * 1024


def _bind(action_id):
    def decorate(function):
        BOUND_ACTIONS[action_id] = function.__name__
        return function

    return decorate


def _json_safe(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _upload_bytes(value):
    """Decode a trame file-input value into (name, bytes)."""
    if isinstance(value, list):
        value = value[0] if value else None
    if not isinstance(value, dict):
        return None, None
    content = value.get("content")
    if isinstance(content, str):
        encoded = content.split(",", 1)[-1]
        if len(encoded) > 4 * ((PROJECT_UPLOAD_LIMIT + 2) // 3):
            raise ValueError("Mesh exceeds the 512 MiB upload limit.")
        content = base64.b64decode(encoded, validate=True)
    elif isinstance(content, list):
        if len(content) > PROJECT_UPLOAD_LIMIT:
            raise ValueError("Mesh exceeds the 512 MiB upload limit.")
        content = bytes(content)
    name = upload_name(value.get("name"))
    if not isinstance(content, (bytes, bytearray)):
        raise ValueError("Choose a readable file to upload.")
    if len(content) > PROJECT_UPLOAD_LIMIT:
        raise ValueError("Mesh exceeds the 512 MiB upload limit.")
    return name, content


class BrowserWorkbench:
    """Binds one trame server to one :class:`WorkbenchSession`."""

    def __init__(self, server, session: WorkbenchSession, *, user: str = "local", poll_seconds: float = 1.0):
        import pyvista as pv

        self.server, self.session, self.user = server, session, user
        session.project_loader = partial(load_browser_assembly, workspace=session.workspace)
        self.poll_seconds = poll_seconds
        self.pv = pv
        self.plotter = pv.Plotter(off_screen=True, window_size=(900, 700))
        self.plotter.set_background("#fafafa")
        self.camera_preset = "Isometric"
        self.axis_widget = self.b0_widget = None
        self.record = None
        self.shielding_render_lock = asyncio.Lock()
        self.shielding_fields = {}
        self.shielding_cancel = Event()
        self.verified_plan_records = {}
        self.plan_errors = {}
        self.follow_job_id = None
        self.editing_identity = None
        self.editing_fingerprint = None
        self.view = None
        self.exports = session.workspace / "exports"
        self.exports.mkdir(parents=True, exist_ok=True)
        server.serve["downloads"] = str(self.exports)
        server.enable_module(
            {
                "serve": {
                    "coilgen-client": str(Path(__file__).with_name("assets")),
                    "coilgen-brand": str(Path(__file__).resolve().parents[1] / "gui" / "assets"),
                },
                "styles": ["coilgen-client/workbench.css"],
                "scripts": ["coilgen-client/project_upload.js"],
                "vue_use": ["CoilGenProjectUpload"],
            }
        )
        self._initial_state()
        session.subscribe(self._session_event)
        self._build_ui()
        self._refresh_jobs()
        server.controller.on_server_ready.add(self._start_polling)

    # -- state ---------------------------------------------------------
    def _initial_state(self):
        state = self.server.state
        state.update(
            {
                "user": self.user,
                "trame__title": "CoilGen2 workbench",
                "templates": [
                    {"title": f"{spec.category} · {spec.title}", "value": spec.id, "description": spec.description}
                    for spec in self.session.templates()
                ],
                "template": None,
                "project_name": "No project",
                "channel_enabled": False,
                "channels": [],
                "active_channel": None,
                "nodes": [{"title": label, "value": key} for key, label in SETTINGS_NODES],
                "node": "geometry",
                "settings_fields": [],
                "settings_values": {},
                "applied_support": "",
                "geometry_help": "",
                "target_help": "",
                "winding_help": "",
                "settings_dirty": False,
                "settings_dialog": False,
                "profile_preset": None,
                "profile_segments": 16,
                "profile_equal_area": False,
                "profile_upload": None,
                "file_dialog": False,
                "file_section": "examples",
                "workspace_panel": "center",
                "center_section": "results",
                "history_open": False,
                "plan_summary": "",
                "plan_channels": [],
                "stages": [{"title": label, "value": key} for key, label in PIPELINE_STAGES],
                "stage": "evaluated",
                "jobs": [],
                "older_jobs": [],
                "runs": [],
                "selected_run": None,
                "displayed_run": None,
                "workspace_history": False,
                "run_source": "",
                "run_context": "",
                "design_review": None,
                "review_dialog": False,
                "review_busy": False,
                "field_plot": False,
                "shielding_sources": [],
                "shielding_source": "",
                "shielding_component": "magnitude",
                "shielding_detail": False,
                "shielding_normal": False,
                "shielding_image": "",
                "shielding_note": "",
                "shielding_busy": False,
                "field_legend": None,
                "outer_winding_support": False,
                "see_through_support": False,
                "current_job_active": False,
                "attempt_notice": None,
                "batch_busy_channels": [],
                "editing_label": "",
                "datasets": [],
                "available_plots": [],
                "dataset": "support",
                "result_plot": None,
                "component": "Bz",
                "scene_caption": "Load a template or open a project.",
                "summary": None,
                "metrics": [],
                "export_options": [],
                "export_selected": [],
                "export_selection_note": "",
                "export_unit": "mm",
                "draft_ack": False,
                "draft_notice": "",
                "draft_selected": False,
                "download_url": "",
                "download_label": "",
                "project_download": "",
                "workspace_file": None,
                "opening_workspace_file": None,
                "saved_projects": list(self._workspace_files()),
                "include_project": True,
                "message": "",
                "message_color": "info",
                "show_message": False,
                "advanced_dialog": False,
                "advanced_config": "",
                "advanced_study": "",
                "project_open_result": None,
                "mesh_upload": None,
                "mesh_scale": 0.001,
                "can_undo": False,
                "can_redo": False,
                "preflight": [],
                "b0_text": "B0 ∥ (0, 0, 1)",
                "author": AUTHOR_CREDIT,
                "project_url": PROJECT_URL,
            }
        )
        state.change("node")(lambda **_: self._refresh_settings())
        # All dismissal paths (Cancel, Escape and backdrop) update the dialog model.
        state.change("settings_dialog")(lambda **_: self._refresh_settings())
        state.change("advanced_dialog")(self._advanced_dialog_changed)
        state.change("settings_values", "profile_preset", "mesh_upload", "profile_upload")(
            lambda **_: self._check_edits()
        )
        state.change("active_channel")(lambda active_channel, **_: self._select_channel(active_channel))
        state.change("dataset", "component", "see_through_support")(lambda **_: self._render())
        state.change("selected_run")(lambda selected_run, **_: self._open_run(selected_run))
        state.change("workspace_history")(lambda **_: self._refresh_jobs())
        state.change("stage")(lambda stage, **_: self._select_stage(stage))
        state.change("export_selected", "draft_ack")(lambda **_: self._refresh_draft_state())
        state.change("center_section")(self._shielding_section_changed)

    def _notify(self, message, color="info"):
        state = self.server.state
        state.message, state.message_color, state.show_message = str(message), color, True

    def _guard(self, operation, *args, success=None, **kwargs):
        try:
            result = operation(*args, **kwargs)
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
            self._notify(str(exc), "error")
            return None
        if success:
            self._notify(success, "success")
        return result

    def _session_event(self, event, payload):
        if event == "model":
            self._refresh_model()
        elif event == "jobs":
            self._refresh_jobs()

    # -- model ---------------------------------------------------------
    def _refresh_model(self):
        from .workbench_shell import model_tree_nodes

        state, session = self.server.state, self.session
        assembly = session.assembly
        state.project_name = assembly.name if assembly else "No project"
        state.channels = [
            {"title": channel.name, "value": channel.id, "enabled": channel.enabled} for channel in session.channels
        ]
        state.active_channel = session.active_channel_id
        state.channel_enabled = bool(
            assembly and session.active_channel_id and assembly.channel(session.active_channel_id).enabled
        )
        state.can_undo, state.can_redo = bool(session.history.undo), bool(session.history.redo)
        project = session.project
        self.editing_fingerprint = project.fingerprint if project is not None else None
        state.editing_label = f"Editing {project.name} · revision {project.revision}" if project else ""
        identity = (assembly.id, session.active_channel_id) if assembly else None
        switched = identity != self.editing_identity
        state.project_download = ""
        if switched:
            self.follow_job_id = None
            self.editing_identity = identity
            self._clear_result()
        if project is not None:
            state.nodes = model_tree_nodes(project.config)
            state.stage = project.study["pipeline_stage"]
            state.preflight = [
                {
                    "title": (
                        "This winding uses independent closed turns. Connected routing and feeds are outside this study."
                        if issue.key == "skip_postprocessing" and project.values["contour_only_sweep_mode"]
                        else issue.message
                    ),
                    "blocking": issue.blocking,
                    "informational": issue.key == "skip_postprocessing" and project.values["contour_only_sweep_mode"],
                }
                for issue in session.preflight(project.study["pipeline_stage"]).issues
            ]
        self._refresh_settings()
        self._refresh_jobs(select_matching=switched, restore_saved=switched)
        self._refresh_run_context()
        if self.record is None or state.dataset in {"support", "preview_target"}:
            self._render()

    def _refresh_settings(self):
        from ..application.submitted_design import choice_title

        state, project = self.server.state, self.session.project
        state.settings_dirty = False
        state.profile_preset, state.mesh_upload, state.mesh_scale = None, None, 0.001
        state.profile_upload = None
        fields = self.session.settings_fields(state.node)
        state.settings_fields = [
            {
                "key": item.key,
                "label": item.label + (f" [{item.unit}]" if item.unit else ""),
                "kind": item.kind,
                "choices": list(item.choices),
                "help": item.help,
            }
            for item in fields
        ]
        if project is None:
            state.settings_values = {}
            state.applied_support, state.geometry_help = "", ""
            return
        values = self.session.settings_values(state.node)
        if state.node == "target":
            config = project.config
            if project.study.get("requirements"):
                state.target_help = "This study uses registered fields in named regions. Inspect or edit the requirements under Advanced."
            elif str(config.target_field_definition_file).lower() not in {"none", ""}:
                state.target_help = "An imported field map supplies the samples and desired field. See Advanced for the saved input reference."
            else:
                state.target_help = ""
        if state.node == "winding":
            state.winding_help = (
                "This winding uses independent closed turns. Inter-turn connectors and return lifts do not apply."
                if project.config.contour_only_sweep_mode
                else "Connected routing is disabled in this model."
                if project.config.skip_postprocessing
                else ""
            )
        if state.node == "geometry":
            source = str(values["coil_mesh_file"])
            state.applied_support = source
            editor = next((editor for editor in SUPPORT_EDITORS if editor.source == source), None)
            factory = geometry_factory(source)
            if editor is not None and len(fields) > 1:
                state.geometry_help = editor.description
            elif factory is not None:
                state.geometry_help = f"Edit {factory.parameter_attribute} in Advanced settings for this support."
            elif source.lower().endswith(".npz"):
                state.geometry_help = (
                    "The project owns this mesh in metres. Import a replacement STL to change its shape."
                )
            elif source.lower().endswith(".stl"):
                state.geometry_help = "Imported mesh unit scale applies to this saved STL. Set the scale below before importing a replacement."
            else:
                state.geometry_help = "Select a generated support or import an STL."
        entries = state.settings_fields
        for entry in entries:
            if entry["kind"] != "choice":
                continue
            key = entry["key"]
            choices = entry["choices"]
            if values[key] not in choices:
                choices.insert(0, values[key])
            if key == "coil_mesh_file":
                labels = {editor.source: editor.label for editor in SUPPORT_EDITORS}
                entry["choices"] = [
                    {
                        "title": labels.get(value) or ("Project-owned support" if Path(value).suffix else value),
                        "value": value,
                    }
                    for value in choices
                ]
            if key == "contour_treatment":
                labels = {
                    "off": "Original contours",
                    "mesh_coupled": "Surface B-spline (mesh scale)",
                    "legacy": "Surface B-spline (legacy strength)",
                    "resample": "Resample without smoothing",
                }
                entry["choices"] = [{"title": labels[value], "value": value} for value in choices]
            elif key != "coil_mesh_file" and any(choice_title(key, value) != value for value in choices):
                entry["choices"] = [{"title": choice_title(key, value), "value": value} for value in choices]
        state.settings_fields = entries
        state.dirty("settings_fields")
        state.settings_values = _json_safe(values)

    def _settings_fields_changed(self):
        state = self.server.state
        if self.session.project is None:
            return False
        saved = self.session.settings_values(state.node)
        edited = state.settings_values or {}

        # Numeric fields arrive as strings after browser entry, even when unchanged.
        def differs(key, value):
            other = edited.get(key)
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                try:
                    return float(other) != value
                except (TypeError, ValueError):
                    return True
            return other != value

        return any(differs(key, value) for key, value in saved.items())

    def _check_edits(self):
        state = self.server.state
        state.settings_dirty = self._settings_fields_changed() or bool(
            state.profile_preset or state.mesh_upload or state.profile_upload
        )
        return state.settings_dirty

    def _ready_to_submit(self):
        if self._check_edits():
            self._notify("Apply or discard the pending edits in Model settings first.", "warning")
            return False
        return True

    def _select_channel(self, channel_id):
        if channel_id and channel_id != self.session.active_channel_id:
            self._guard(self.session.select_channel, channel_id)

    def _select_stage(self, stage):
        project = self.session.project
        if project and stage and stage != project.study["pipeline_stage"]:
            self._guard(self.session.set_stage, stage)

    # -- actions -------------------------------------------------------
    def _show_project(self):
        state = self.server.state
        state.file_dialog, state.settings_dialog = False, False
        state.workspace_panel, state.center_section = "center", "results"
        state.history_open = False

    @_bind("load_template")
    def load_template(self, template=None):
        template = self.server.state.template if template is None else template
        if template and self._ready_to_submit():
            if self._guard(self.session.load_template, template, success="Template loaded.") is not None:
                self.server.state.template = template
                self._show_project()

    @_bind("new_project")
    def new_project(self):
        if self._ready_to_submit() and self._guard(self.session.new_project, success="New project created."):
            self.server.state.template = None
            self._show_project()

    @_bind("open_project")
    def open_uploaded_project(self, token):
        if not isinstance(token, str) or len(token) > 64:
            token = ""
        state = self.server.state
        project = None
        path = self._guard(claim_project_upload, self.session.workspace, token)
        if path is not None and self._ready_to_submit():
            project = self._open_stored_project(path)
        state.project_open_result = {"token": token, "ok": project is not None, "message": state.message}

    def _open_stored_project(self, path):
        path = self._guard(checked_upload_path, self.session.workspace, path)
        if path is not None:
            return self._open_document(path)
        return None

    def _workspace_files(self):
        root = self.session.workspace / "projects"
        if root.resolve() != root or not root.is_dir():
            return {}
        return {
            path.name: path
            for path in sorted(root.iterdir())
            if path.suffix.lower() in {".coilgen", ".json"} and path.resolve() == path and path.is_file()
        }

    async def open_workspace_project(self):
        state = self.server.state
        if state.opening_workspace_file or not self._ready_to_submit():
            return
        path = self._workspace_files().get(state.workspace_file)
        if path is None:
            self._notify("Choose a project saved in this workspace.", "warning")
            return
        state.opening_workspace_file = path.name
        state.flush()
        try:
            # Keep verification off the trame event loop; publish the verified
            # assembly and its embedded runs on the session's owning thread.
            assembly = await asyncio.to_thread(self.session.project_loader, path)
            self._open_document(path, loader=lambda _path: assembly)
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
            self._notify(str(exc), "error")
        finally:
            state.opening_workspace_file = None
            state.flush()

    def _open_document(self, path, *, loader=None):
        previous_identity = self.editing_identity
        self.editing_identity = None
        assembly = self._guard(self.session.open_project, path, loader=loader, success=f"Opened {path.name}.")
        if assembly is not None:
            self.server.state.template = None
            self._show_project()
        else:
            self.editing_identity = previous_identity
        return assembly

    def _store_upload(self, upload, suffixes=(".stl",), kind="an STL mesh file"):
        name, content = _upload_bytes(upload)
        if name is None or content is None:
            raise ValueError("Choose a readable file to upload.")
        if not name.lower().endswith(tuple(suffixes)):
            raise ValueError(f"Choose {kind} to import.")
        uploads = self.session.workspace / "uploads"
        if uploads.resolve() != uploads:
            raise ValueError("Invalid uploads directory.")
        uploads.mkdir(exist_ok=True)
        path = Path(tempfile.mkdtemp(dir=uploads)) / name
        path.write_bytes(content)
        return path

    @_bind("save_project")
    def save_project(self, file_format="bundle"):
        if not self._ready_to_submit():
            return
        self.server.state.project_download = ""
        saved = self._guard(self.session.save_project, format=file_format, selected_run=self.record)
        if saved and self._guard(self._publish_project, saved):
            self.server.state.project_download = f"downloads/{quote(saved.name)}"
            self.server.state.saved_projects = list(self._workspace_files())
            self.server.state.workspace_file = saved.name
            self._notify(f"Saved {saved.name}. Use Download project to keep a portable copy.", "success")

    def _publish_project(self, saved):
        target = self.exports / saved.name
        with tempfile.TemporaryDirectory(dir=self.exports) as staging:
            staged = Path(staging) / saved.name
            shutil.copyfile(saved, staged)
            return staged.replace(target)

    @_bind("import_support_mesh")
    def _import_mesh(self, upload):
        scale = self.server.state.mesh_scale
        try:
            scale = float(scale)
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError
        except (TypeError, ValueError):
            self._notify("Set a positive, finite STL scale before importing (0.001 for millimetres).", "error")
            return
        path = self._guard(self._store_upload, upload)
        if path:
            self._guard(
                self.session.import_support_mesh,
                path,
                length_scale_to_m=scale,
                success=f"Imported support mesh {path.name}.",
            )

    def _import_profile_table(self, upload):
        path = self._guard(self._store_upload, upload, (".csv", ".tsv", ".txt"), "a CSV, TSV or TXT point table")
        if path:
            self._guard(
                self.session.import_profile_table,
                path,
                success=f"Imported the cross-section table {path.name}.",
            )

    def _profile_segments(self):
        from ..application.conductor_profiles import MAX_CIRCLE_SEGMENTS, MIN_CIRCLE_SEGMENTS

        try:
            value = float(self.server.state.profile_segments)
        except (TypeError, ValueError):
            value = float("nan")
        if not (value.is_integer() and MIN_CIRCLE_SEGMENTS <= value <= MAX_CIRCLE_SEGMENTS):
            raise ValueError(
                f"Circle points must be a whole number from {MIN_CIRCLE_SEGMENTS} to {MAX_CIRCLE_SEGMENTS}."
            )
        return int(value)

    @_bind("undo")
    def undo(self):
        self._guard(self.session.undo)

    @_bind("redo")
    def redo(self):
        self._guard(self.session.redo)

    @_bind("apply_settings")
    def apply_settings(self):
        state = self.server.state
        pending = [bool(item) for item in (state.mesh_upload, state.profile_preset, state.profile_upload)]
        if any(pending) and (self._settings_fields_changed() or sum(pending) > 1):
            self._notify(
                "Apply field edits, conductor presets, profile tables and mesh replacements separately.", "warning"
            )
            return
        if state.mesh_upload:
            self._import_mesh(state.mesh_upload)
            return
        if state.profile_upload:
            self._import_profile_table(state.profile_upload)
            return
        if state.profile_preset:
            try:
                segments = self._profile_segments()
            except ValueError as exc:
                self._notify(str(exc), "error")
                return
            title = {"u_shape": "U-shape", "round": "round wire"}.get(state.profile_preset, state.profile_preset)
            self._guard(
                self.session.apply_profile_preset,
                state.profile_preset,
                segments=segments,
                equal_area=bool(state.profile_equal_area),
                success=f"Applied the {title} conductor profile.",
            )
            return
        values = dict(state.settings_values or {})
        try:
            check_browser_inputs(values, current_project=self.session.project)
        except (TypeError, ValueError) as exc:
            self._notify(str(exc), "error")
            return
        if self._guard(self.session.apply_settings, values, success="Settings applied.") is not None:
            self._refresh_settings()

    @_bind("advanced_settings")
    def open_advanced(self):
        project = self.session.project
        if project is None:
            return
        state = self.server.state
        state.advanced_config = json.dumps(project.values, indent=2)
        state.advanced_study = json.dumps(_json_safe(project.study), indent=2, default=str)
        state.advanced_dialog = True

    def _advanced_dialog_changed(self, advanced_dialog, **_):
        if not advanced_dialog:
            self.server.state.advanced_config = self.server.state.advanced_study = ""

    def apply_advanced(self):
        project = self.session.project
        state = self.server.state
        try:
            config = json.loads(state.advanced_config, object_pairs_hook=_json_object)
            study = json.loads(state.advanced_study, object_pairs_hook=_json_object)
        except (TypeError, ValueError) as exc:
            self._notify(f"Invalid JSON: {exc}", "error")
            return
        if not isinstance(config, dict) or not isinstance(study, dict):
            self._notify("Configuration and Study must each be a JSON object enclosed in { }.", "error")
            return
        current = project.values
        if current.keys() - config.keys() or project.study.keys() - study.keys():
            self._notify(
                "Keep the existing field names when editing these JSON objects. Change their values instead.", "error"
            )
            return
        changes = {key: value for key, value in config.items() if current.get(key) != value}
        study_changes = {key: value for key, value in study.items() if project.study.get(key) != value}
        try:
            check_browser_inputs(config, study, workspace=self.session.workspace)
            self.session.apply_inputs(changes, study_changes)
        except (ValueError, TypeError, KeyError) as exc:
            self._notify(str(exc), "error")
            return
        state.advanced_dialog = False
        self._notify(f"Applied {len(changes) + len(study_changes)} advanced changes.", "success")

    @_bind("duplicate_channel")
    def duplicate_channel(self):
        self._guard(self.session.duplicate_channel)

    @_bind("remove_channel")
    def remove_channel(self):
        self._guard(self.session.remove_channel)

    @_bind("select_channel")
    def toggle_channel(self, channel_id, enabled):
        self._guard(self.session.set_channel_enabled, channel_id, bool(enabled))

    @_bind("compute")
    def compute(self):
        if not self._ready_to_submit():
            return
        state = self._guard(self.session.compute, self.server.state.stage)
        if state:
            self.follow_job_id = state["id"]
            self._notify(f"Queued revision {state.get('project_revision')} · {state.get('pipeline_stage')}.", "info")

    @_bind("compute_channels")
    def compute_channels(self):
        if not self._ready_to_submit():
            return
        self._refresh_jobs()
        if self.server.state.batch_busy_channels:
            self._notify(
                "This batch would repeat an active channel. Wait for it to finish or compute another channel individually.",
                "info",
            )
            return
        states = self._guard(self.session.compute_channels, self.server.state.stage)
        if states:
            channel_job = states.get(self.session.active_channel_id)
            self.follow_job_id = channel_job["id"] if channel_job else None
            self._notify(f"Queued {len(states)} channel studies.", "info")

    @_bind("cancel_job")
    def cancel_job(self, job_id):
        job = self._guard(self.session.cancel_job, job_id)
        if job and job.get("status") in {"cancelled", "cancelling"}:
            message = "Cancelled" if job["status"] == "cancelled" else "Stopping"
            self._notify(f"{message} job {job_id[:8]}.", "info")

    # -- jobs and runs -------------------------------------------------
    def _refresh_jobs(self, *, select_matching=False, restore_saved=False):
        state = self.server.state
        project = self.session.project
        project_ids = {channel.project.id for channel in self.session.channels}
        show_all = state.workspace_history or project is None
        job_states = self.session.job_states()
        self._refresh_plan(job_states)

        def label(job):
            return f"{job.get('name') or job.get('project_id') or 'Saved design'} · r{job.get('project_revision')} · {job.get('pipeline_stage')} · {job['id'][:8]}"

        submissions = [
            {
                "id": job["id"],
                "status": job.get("status"),
                "stage": job.get("stage", ""),
                "progress": 100 * min(1.0, max(0.0, float(job.get("progress") or 0.0))),
                "boundary": dict(PIPELINE_STAGES).get(job.get("pipeline_stage"), "Stage unrecorded"),
                "revision": job.get("project_revision"),
                "channel": job.get("name") or job.get("project_id") or "Saved design",
                "error": (job.get("error") or "")[:400],
            }
            for job in job_states
            if (show_all or job.get("project_id") in project_ids)
            and (not job.get("embedded") or job.get("status") != "completed")
        ]
        active_jobs = [job for job in submissions if job["status"] in ACTIVE_STATUSES]
        finished = [job for job in submissions if job["status"] not in ACTIVE_STATUSES]
        state.jobs, state.older_jobs = active_jobs + finished[:3], finished[3:]
        fingerprint = self.editing_fingerprint

        def identity(job):
            return tuple(
                job.get(key) for key in ("project_id", "project_revision", "project_fingerprint", "pipeline_stage")
            )

        current = (project.id, project.revision, fingerprint, state.stage) if project else None
        active = {identity(job) for job in job_states if job.get("status") in ACTIVE_STATUSES}
        state.current_job_active = current in active
        # Imported archives have no submission ordering in this workspace.
        last = next((job for job in job_states if not job.get("embedded") and identity(job) == current), {})
        stopped = {
            "failed": "Read the worker error in Design plan, then choose Compute to retry.",
            "cancelled": "Choose Compute to start another run when ready.",
            "interrupted": "Check Design plan, then choose Compute to retry.",
        }
        status = last.get("status")
        state.attempt_notice = (
            {
                "title": f"Job {last['id'][:8]} {status} for {project.name} r{project.revision}.",
                "action": f"This attempt produced no completed result. {stopped[status]}",
                "color": "error" if status == "failed" else "warning",
            }
            if status in stopped and not state.current_job_active
            else None
        )
        busy = []
        if len(self.session.channels) > 1:
            for channel in self.session.channels:
                candidate = channel.project
                if channel.enabled and any(
                    owner == candidate.id and stage == state.stage for owner, _, _, stage in active
                ):
                    try:
                        planned = self.session._at_stage(candidate, state.stage)
                    except ValueError:
                        # This combination cannot submit. Normal batch preflight explains why on click.
                        continue
                    if (planned.id, planned.revision, planned.fingerprint, state.stage) in active:
                        busy.append(channel.name)
        state.batch_busy_channels = busy
        jobs = [job for job in self.session.completed_runs() if show_all or job.get("project_id") == project.id]
        runs = [{"title": label(job), "value": job["id"]} for job in jobs]
        if runs != state.runs:
            state.runs = runs
            if state.selected_run and all(job["id"] != state.selected_run for job in jobs):
                self._clear_result()
            select_matching = True
        if self.follow_job_id:
            pending = self.session.states.get(self.follow_job_id, {})
            if pending.get("status") in {"completed", "failed", "cancelled", "interrupted"}:
                if pending.get("status") == "completed" and any(job["id"] == self.follow_job_id for job in jobs):
                    state.selected_run = self.follow_job_id
                self.follow_job_id = None
        if restore_saved and state.selected_run is None:
            saved_id = self.session.saved_run_job()
            if saved_id is not None and any(job["id"] == saved_id for job in jobs):
                state.selected_run = saved_id
        if select_matching and project and state.selected_run is None:
            match = next(
                (
                    job
                    for job in jobs
                    if job.get("project_id") == project.id
                    and job.get("project_revision") == project.revision
                    and job.get("project_fingerprint") == fingerprint
                ),
                None,
            )
            if match:
                state.selected_run = match["id"]

    def _refresh_plan(self, jobs):
        # Use the desktop's evidence rules. Opening a historical run never
        # awards completed steps to a different applied project or revision.
        from ..gui.design_plan import STAGE_STATUS_LABELS, build_design_plan

        state, session = self.server.state, self.session
        if session.assembly is None:
            state.plan_summary, state.plan_channels = "", []
            return
        project_ids = {channel.project.id for channel in session.channels}
        jobs = [job for job in jobs if job.get("project_id") in project_ids]
        # Restored embedded states already carry the verified archive identity.
        # Keep the same small receipt after a live run has been opened, so
        # changing the plot or channel does not make recorded steps disappear.
        records = {job["id"]: job for job in jobs if job.get("embedded") and job.get("status") == "completed"}
        records.update(
            (key, row) for key, row in self.verified_plan_records.items() if row["project_id"] in project_ids
        )
        plan = build_design_plan(
            session.assembly,
            active_channel_id=session.active_channel_id,
            states=jobs,
            records=records,
            verification_errors=self.plan_errors,
            requested_stage=state.stage,
        )
        state.plan_summary = (
            f"{plan.completed_count}/{plan.requested_count} steps recorded · "
            f"{plan.completed_channels}/{plan.enabled_channels} channels verified"
        )
        state.plan_channels = [
            {
                "id": channel.channel_id,
                "name": channel.name,
                "enabled": channel.enabled,
                "activity": (
                    "Execution finished · choose View result for this channel in Jobs to verify the saved result."
                    if channel.verification == "pending"
                    else channel.activity
                    if channel.verification == "failed" or channel.status not in {"idle", "completed"}
                    else ""
                ),
                "error": channel.verification == "failed" or channel.status in {"failed", "interrupted"},
                "steps": [
                    {
                        "key": step.key,
                        "label": step.label,
                        "code": step.status,
                        "status": STAGE_STATUS_LABELS[step.status],
                        "help": step.tooltip,
                    }
                    for step in channel.stages
                ],
            }
            for channel in plan.channels
        ]

    def view_job_result(self, job_id):
        """Inspect an exact completed job without switching the editing channel."""
        state = self.server.state
        if self.session.states.get(job_id, {}).get("status") != "completed":
            self._notify("This job has no completed result.", "warning")
            return
        if all(run["value"] != job_id for run in state.runs):
            state.workspace_history = True
            self._refresh_jobs()
        self.follow_job_id = None
        state.workspace_panel, state.center_section = "center", "results"
        if state.displayed_run == job_id and state.result_plot:
            state.dataset = state.result_plot
        state.selected_run = job_id

    def _clear_result(self):
        self.record = None
        self.shielding_cancel.set()
        self.shielding_fields = {}
        state = self.server.state
        if state.center_section in {"checks", "metrics", "export", "shielding"}:
            state.center_section = "results"
        state.update(
            {
                "selected_run": None,
                "displayed_run": None,
                "summary": None,
                "shielding_sources": [],
                "shielding_image": "",
                "shielding_note": "",
                "shielding_busy": False,
                "result_plot": None,
                "metrics": [],
                "export_options": [],
                "export_selected": [],
                "draft_ack": False,
                "draft_notice": "",
                "draft_selected": False,
                "download_url": "",
                "download_label": "",
                "run_source": "",
                "run_context": "",
                "design_review": None,
                "review_dialog": False,
                "review_busy": False,
                "include_project": True,
            }
        )
        if state.dataset not in {"support", "preview_target"}:
            state.dataset = "support"

    @_bind("open_run")
    def _open_run(self, job_id):
        from ..application.shielding import shielding_sources
        from ..application.submitted_design import contour_fit_caption
        from ..gui.shielding_plot import SHIELDING_SOURCES
        from .result_display import problem_groups, result_metrics, result_overview, reviewed_wire_metric

        if not job_id:
            self._clear_result()
            self._render()
            return
        record = self._guard(self.session.load_run, job_id)
        if record is None:
            self.plan_errors[job_id] = self.server.state.message
            self.verified_plan_records.pop(job_id, None)
            self._clear_result()
            self._refresh_plan(self.session.job_states())
            self._render()
            return
        self.record = record
        self.shielding_cancel.set()
        self.shielding_fields = {}
        state = self.server.state
        if state.center_section == "shielding":
            state.center_section = "results"
        sources = shielding_sources(record)
        state.shielding_sources = [{"value": key, "title": SHIELDING_SOURCES[key]} for key in sources]
        state.shielding_source = next(
            (key for key in ("routed", "contour", "sheet", "fabricated") if key in sources), ""
        )
        state.shielding_component, state.shielding_normal, state.shielding_detail = "magnitude", False, False
        state.shielding_image, state.shielding_note, state.shielding_busy = "", "", False
        state.review_dialog, state.design_review, state.review_busy = False, None, False
        state.download_url, state.download_label = "", ""
        meta = record.metadata_view
        self.plan_errors.pop(job_id, None)
        self.verified_plan_records[job_id] = {
            key: meta.get(key) or (meta.get("job") or {}).get(key)
            for key in ("project_id", "project_revision", "project_fingerprint", "pipeline_stage", "request_sha256")
        }
        boundary = recorded_stage(meta)
        state.run_source = f"{meta.get('name', 'Saved design')} · r{meta.get('project_revision', '?')} · {boundary or 'stage unrecorded'} · {job_id[:8]}"
        self._refresh_run_context()
        summary = self.session.run_summary(record)
        summary["contour_fit_caption"] = contour_fit_caption(meta)
        summary["reviewed_wire_metric"] = reviewed_wire_metric(meta)
        summary["overview"] = result_overview(record)
        summary["constraints"] = [
            f"{'Failed' if check.get('passed') is False else 'Unverified'} · {check.get('name', 'Unnamed constraint')}"
            for check in meta.get("constraints") or ()
            if check.get("passed") is not True
        ]
        summary["suppression"] = (meta.get("fabricated_physics") or {}).get("sampled_suppression")
        summary["problems"] = problem_groups(record)
        summary["check_rows"] = [
            {"name": name, "value": summary["verdicts"][key]}
            for key, name in (
                ("sheet", "Stream function model"),
                ("sweeps", "Pipeline conductor sweep"),
                ("fabrication", "Strict fabrication review"),
            )
        ]
        for key, name in (
            ("validation", "Independent stream function field"),
            ("winding_validation", "Independent winding field"),
        ):
            check = meta.get(key) or {}
            if check.get("available") is True:
                passed = check.get("passed")
                value = "Passed" if passed is True else "Failed" if passed is False else "Unverified"
                summary["check_rows"].append({"name": name, "value": value})
        verdicts = [row["value"] for row in summary["check_rows"]]
        suppression = summary["suppression"] or {}
        failed = (
            "Failed" in verdicts
            or (boundary == "evaluated" and summary["accepted"] is False)
            or any(check.get("passed") is False for check in meta.get("constraints") or ())
            or any(
                (suppression.get(source) or {}).get(key) is False
                for source in ("sheet", "fabricated")
                for key in ("rms_passed", "peak_passed")
            )
        )
        if boundary is None:
            summary["label"], summary["color"] = "Historical result · stage unrecorded", "warning"
        elif boundary != "evaluated":
            summary["label"] = f"{dict(PIPELINE_STAGES).get(boundary, boundary)} completed"
            summary["color"] = "error" if failed else "info"
            if failed:
                summary["label"] += " · checks failed"
        elif failed:
            summary["label"], summary["color"] = "Engineering checks failed", "error"
        elif summary["accepted"] is True and all(v == "Passed" for v in verdicts) and not summary["constraints"]:
            summary["label"], summary["color"] = "Engineering checks passed", "success"
        else:
            summary["label"], summary["color"] = "Engineering review incomplete", "warning"
        state.summary = _json_safe(
            {key: value for key, value in summary.items() if key not in {"metrics", "failure_reasons", "constraints"}}
        )
        state.metrics = result_metrics(meta)
        options = self._guard(self.session.export_options, record) or ()
        state.export_options = [
            {
                "value": option.name,
                "title": geometry_option_label(option),
                "format": option.format,
                "status": ("Inspection · " if option.inspection_only else "") + option.check_status,
                "draft": is_draft_artifact(option.name),
            }
            for option in options
        ]
        state.export_selected = list(default_export_selection(options))
        fabrication_issues = [
            item["title"]
            for group in summary["problems"]
            if group["title"] == "Strict fabrication review"
            for item in group["items"]
        ]
        state.draft_notice = draft_export_notice(list(dict.fromkeys(fabrication_issues)))
        state.draft_ack = False
        arrays = record.select_arrays(copy=False)
        if state.dataset in {"support", "preview_target"} or not resolve_dataset(arrays, state.dataset).available:
            state.dataset = next(
                (key for key in PREFERRED_RUN_PLOTS if resolve_dataset(arrays, key).available),
                "saved_unavailable",
            )
        state.result_plot = state.dataset
        self._render()
        state.displayed_run = job_id
        self._refresh_plan(self.session.job_states())

    def _shielding_section_changed(self, center_section, **_):
        if center_section != "shielding":
            self.shielding_cancel.set()

    async def show_shielding(self, source, component, net_detail=False):
        from ..gui.shielding_plot import SHIELDING_COMPONENTS

        record, state = self.record, self.server.state
        if record is None or source not in [item["value"] for item in state.shielding_sources]:
            return
        if not isinstance(component, str) or component not in SHIELDING_COMPONENTS or not isinstance(net_detail, bool):
            return
        self.shielding_cancel.set()
        cancel = self.shielding_cancel = Event()
        state.center_section = "shielding"
        state.shielding_source, state.shielding_component, state.shielding_detail = source, component, net_detail
        state.shielding_image, state.shielding_note, state.shielding_busy = "", "", True
        state.flush()
        async with self.shielding_render_lock:
            if cancel.is_set():
                return
            try:
                image, note, has_normal = await asyncio.to_thread(
                    self._shielding_image, record, source, component, net_detail, self.shielding_fields, cancel
                )
            except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
                image, note, has_normal = "", f"Surface plot unavailable · {exc}", False
            if not cancel.is_set():
                state.shielding_image, state.shielding_note, state.shielding_normal = image, note, has_normal
                state.shielding_busy = False
                state.flush()

    @staticmethod
    def _shielding_image(record, source, component, net_detail, cache, cancel):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        from ..application.shielding import shielding_surface_fields
        from ..gui.shielding_plot import SHIELDING_SOURCES, draw_shielding_surface

        fields = cache.get(source)
        if fields is None:
            fields = shielding_surface_fields(record, source, cancel_check=cancel.is_set)
            if source.endswith("_interior"):
                # Recolor the same evaluated vectors when changing component or scale.
                cache[source] = fields
        figure = Figure(figsize=(11, 6), dpi=115, layout="constrained")
        canvas = FigureCanvasAgg(figure)
        title = f"{record.metadata_view.get('name', 'Saved run')} · {SHIELDING_SOURCES[source]}"
        note = draw_shielding_surface(figure, fields, component, title=title, net_detail=net_detail)
        buffer = BytesIO()
        canvas.print_png(buffer)
        image = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{image}", note, fields.normals is not None

    async def review_submitted_design(self):
        """Read the original request for the selected run without blocking the workbench."""
        record, state = self.record, self.server.state
        if record is None:
            return
        state.review_dialog, state.review_busy, state.design_review = True, True, None
        state.flush()
        try:
            review = await asyncio.to_thread(self._read_design_review, record)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            review = {"groups": [], "request_json": "", "smoothing": [], "error": str(exc)}
        if self.record is record:
            state.design_review, state.review_busy = review, False
            state.flush()

    def _read_design_review(self, record):
        from ..application.submitted_design import design_choice_groups, displacement_label

        request = load_saved_request(record)
        source = json.dumps(request, ensure_ascii=False, indent=2, sort_keys=True)
        encoded = source.encode("utf-8")
        download = ""
        if len(encoded) > INLINE_REQUEST_BYTES:
            name = f"submitted-{sha256(encoded).hexdigest()}.json"
            atomic_text(self.exports / name, source)
            download, source = f"downloads/{name}", ""
        return {
            "groups": design_choice_groups(request),
            "request_json": source,
            "download": download,
            "smoothing": [
                {**_json_safe(part), "displacement_label": displacement_label(part.get("max_displacement_m"))}
                for part in record.metadata_view.get("contour_smoothing") or ()
                if isinstance(part, Mapping)
            ],
            "error": "",
        }

    def _refresh_run_context(self):
        if self.record is None:
            return
        project = self.session.project
        meta = self.record.metadata_view
        if project is None:
            label = "Reviewing a saved run · no project open for editing"
        elif meta.get("project_id") != project.id:
            source = next(
                (channel for channel in self.session.channels if channel.project.id == meta.get("project_id")), None
            )
            if source is None:
                label = "Historical run from a different model"
            else:
                label = f"Viewing channel {source.name} · editing {project.name}"
                if (source.project.revision, source.project.fingerprint) != (
                    meta.get("project_revision"),
                    meta.get("project_fingerprint"),
                ):
                    label += " · earlier or different revision of the viewed channel"
        elif meta.get("project_revision") != project.revision or meta.get("project_fingerprint") != project.fingerprint:
            label = "Earlier or different revision of this model"
        else:
            label = "Matches the editing model"
        self.server.state.run_context = label

    def _refresh_draft_state(self):
        state = self.server.state
        selected = set(state.export_selected or ())
        state.draft_selected = any(is_draft_artifact(name) for name in selected)
        kinds = [option.get("format") for option in state.export_options or () if option["value"] in selected]
        state.export_selection_note = (
            "Selected: "
            + ", ".join(
                f"{label} × {kinds.count(key)}" for key, label in GEOMETRY_FORMAT_LABELS.items() if key in kinds
            )
            if kinds
            else "Choose files and formats below to enable export."
        )

    @_bind("export_geometry")
    def export_geometry(self):
        state = self.server.state
        if self.record is None or not state.export_selected:
            self._notify("Select a completed run and at least one file.", "warning")
            return
        if state.draft_selected and not state.draft_ack:
            self._notify(state.draft_notice + " Tick the DRAFT confirmation to export.", "warning")
            return
        package = self._guard(
            self.session.export,
            self.record,
            list(state.export_selected),
            unit=state.export_unit,
            allow_draft=bool(state.draft_ack),
            include_project=bool(state.include_project),
        )
        if package is None:
            return
        archive = package.directory.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(package.directory.rglob("*")):
                if path.is_file():
                    bundle.write(path, path.relative_to(package.directory))
        drafts = any(row.get("draft") for row in package.manifest["geometry"])
        self._offer_download(archive, f"Download {archive.name}" + (" (DRAFT)" if drafts else ""))
        self._notify(f"Exported {len(package.manifest['geometry'])} files.", "warning" if drafts else "success")

    def _offer_download(self, path: Path, label: str):
        state = self.server.state
        state.download_url = f"downloads/{quote(path.relative_to(self.exports).as_posix())}"
        state.download_label = label

    # -- rendering -----------------------------------------------------
    def _orientation_widgets(self, visible, b0):
        from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget

        pv = self.pv

        def widget(actor, viewport):
            marker = vtkOrientationMarkerWidget()
            marker.SetOrientationMarker(actor)
            marker.SetInteractor(self.plotter.iren.interactor)
            marker.SetDefaultRenderer(self.plotter.renderer)
            marker.SetViewport(*viewport)
            return marker

        if visible and self.axis_widget is None:
            axes = []
            for axis, color in zip(np.eye(3), ("#d62728", "#2ca02c", "#1f77b4")):
                arrow = pv.Arrow(direction=axis, shaft_radius=0.025)
                arrow.cell_data["rgb"] = np.tile(pv.Color(color).int_rgb, (arrow.n_cells, 1)).astype(np.uint8)
                axes.append(arrow)
            mapper = pv.DataSetMapper(pv.merge(axes, merge_points=False))
            mapper.scalar_map_mode = "cell"
            mapper.color_mode = "direct"
            self.axis_widget = widget(pv.Actor(mapper=mapper), (0.01, 0.02, 0.16, 0.17))
        if visible and b0 is not None:
            arrow = b0_indicator_mesh(b0, pv, label=False)
            if self.b0_widget is None:
                actor = pv.Actor(mapper=pv.DataSetMapper(arrow))
                actor.prop.color = B0_INDICATOR_COLOR
                self.b0_widget = widget(actor, (0.01, 0.18, 0.16, 0.33))
            else:
                self.b0_widget.GetOrientationMarker().GetMapper().SetInputData(arrow)
        active = []
        for marker, enabled in ((self.axis_widget, visible), (self.b0_widget, visible and b0 is not None)):
            if marker is not None:
                marker.SetEnabled(int(enabled))
                if enabled:
                    marker.InteractiveOff()
                    active.append(marker)
        return active

    def set_camera_view(self, direction, *, push=True):
        preset = {
            "Isometric": self.plotter.view_isometric,
            "XY": self.plotter.view_xy,
            "XZ": self.plotter.view_xz,
            "YZ": self.plotter.view_yz,
        }.get(direction)
        if preset is not None:
            self.camera_preset = direction
            preset()
            if push and self.view is not None:
                self.view.push_camera(camera=self.plotter.camera)

    @_bind("show_dataset")
    def _render(self):
        state, plotter, pv = self.server.state, self.plotter, self.pv
        project = self.session.project
        plotter.clear()
        arrays, reason = {}, ""
        current_model = state.dataset in {"support", "preview_target"}
        state.field_plot = state.dataset in FIELD_KEYS
        state.field_legend = None
        saved_arrays = self.record.select_arrays(copy=False) if self.record is not None else {}
        if current_model and project is not None:
            try:
                mesh, target, message, _diagnostics = model_preview(
                    project.config, project.project_root, requirements=project.study.get("requirements")
                )
                if mesh is not None:
                    arrays = {"vertices_m": mesh.vertices, "faces": mesh.faces, **target}
                reason = message
            except Exception as exc:
                reason = f"Model preview unavailable: {type(exc).__name__}: {exc}"
        elif not current_model:
            arrays = saved_arrays
        state.datasets = [
            {
                "title": label,
                "value": key,
                "props": {
                    "disabled": project is None
                    if key in {"support", "preview_target"}
                    else not resolve_dataset(saved_arrays, key).available,
                    "subtitle": group.title(),
                },
            }
            for group, entries in DATASET_GROUPS
            for label, key in entries
        ]
        groups = {}
        for item in state.datasets:
            if not item["props"]["disabled"]:
                groups.setdefault(item["props"]["subtitle"], []).append(item)
        state.available_plots = [{"label": label, "items": items} for label, items in groups.items()]
        if not current_model and self.record is not None:
            b0 = (self.record.metadata_view.get("config") or {}).get("b_0_direction")
            source = f"Saved run · {state.run_source}"
        else:
            b0 = project.config.b_0_direction if project is not None else None
            source = f"Editing model · {project.name} · r{project.revision}" if project else ""
        try:
            b0 = normalized_b0_direction(b0) if b0 is not None else None
        except (TypeError, ValueError):
            b0 = None
        state.b0_text = (
            "B0 ∥ (" + ", ".join(f"{v:.3g}" for v in b0) + ")"
            if b0 is not None
            else ("B0 unavailable" if source else "")
        )
        if state.dataset == "saved_unavailable":
            layers, missing = (
                [],
                "This run has no viewable plot data. Review its recorded design choices and engineering checks.",
            )
        elif b0 is None and state.component == "BdotB0" and state.dataset in FIELD_KEYS:
            layers, missing = [], "This run has no recorded B0 direction. Select Bx, By, Bz or Bmag."
        else:
            layers, missing = (
                scene_layers(arrays, state.dataset, component=state.component, b0_direction=b0) if arrays else ([], "")
            )
        if not current_model and layers:
            state.result_plot = state.dataset
        state.outer_winding_support = bool(
            state.dataset == "review_conductor"
            and self.record is not None
            and (self.record.metadata_view.get("config") or {}).get("conductor_mounting_mode") == "surface_outer"
            and any(layer.name == "support" for layer in layers)
        )
        caption = "Coordinates [m] · " + DATASET_LABELS.get(state.dataset, state.dataset)
        for layer in layers:
            if layer.kind == "surface":
                faces = np.column_stack((np.full(len(layer.faces), 3), layer.faces)).ravel()
                mesh = pv.PolyData(layer.points, faces)
            elif layer.kind == "lines":
                paths = [pv.lines_from_points(path) for path in layer.lines]
                mesh = paths[0] if len(paths) == 1 else pv.merge(paths, merge_points=False)
            else:
                mesh = pv.PolyData(layer.points)
            options = {"opacity": layer.opacity}
            if layer.scalars is not None and len(layer.scalars) == mesh.n_points:
                limit = (float(np.max(np.abs(layer.scalars))) or 1e-12) if layer.symmetric else None
                options.update(
                    scalars=np.asarray(layer.scalars, dtype=float),
                    cmap=DENSITY_CMAP if layer.name == "current_density" else FIELD_CMAP,
                    clim=(-limit, limit) if limit else None,
                    show_scalar_bar=False,
                )
                finite = layer.scalars[np.isfinite(layer.scalars)]
                if len(finite):
                    caption += f" · {layer.scalar_label} {finite.min():.4g} … {finite.max():.4g}"
            else:
                options["color"] = layer.color
            if layer.name == "support" and state.outer_winding_support and not state.see_through_support:
                # A light opaque former separates front and rear turns in WebGL.
                # Only apply this to centerlines recorded as mounted outside.
                options.update(opacity=1.0, color="#d2dcdf", ambient=0.62, diffuse=0.38)
            if layer.kind == "points":
                options.update(point_size=8 if state.field_plot else 6, render_points_as_spheres=False)
            elif layer.kind == "lines":
                options["line_width"] = 2
            actor = plotter.add_mesh(mesh, name=layer.name, **options)
            if "scalars" in options:
                # Read the same VTK lookup table and limits that color this actor.
                lut = actor.mapper.lookup_table
                low, high = actor.mapper.scalar_range
                indices = np.linspace(0, lut.n_values - 1, 17, dtype=int) if low != high else [0, 0]
                stops = ["#%02x%02x%02x" % tuple(rgb) for rgb in lut.values[indices, :3]]
                state.field_legend = {
                    "label": layer.scalar_label,
                    "low": f"{low:.4g}",
                    "high": f"{high:.4g}",
                    "gradient": f"linear-gradient(to right, {', '.join(stops)})",
                }
        markers = self._orientation_widgets(bool(layers), b0)
        if layers:
            caption += " · corner arrows: x red, y green, z blue"
            if b0 is not None:
                caption += "; B0 orange"
            plotter.enable_parallel_projection()
            self.set_camera_view(self.camera_preset, push=False)
        state.scene_caption = (
            f"{source} · {caption}"
            if layers
            else (
                missing
                or reason
                or (
                    "Choose a completed run in Results."
                    if state.runs
                    else "Choose a template or open a project to preview the support."
                )
            )
        )
        if self.view is not None:
            # set_widgets sends the geometry and camera together. The first
            # view following the welcome screen must already be isometric.
            self.view.set_widgets(markers)
            if layers:
                self.view.push_camera(camera=plotter.camera)

    # -- polling -------------------------------------------------------
    def _start_polling(self, **_kwargs):
        asyncio.ensure_future(self._poll_loop())

    async def _poll_loop(self):
        while True:
            await asyncio.sleep(self.poll_seconds)
            if self.session.active():
                with self.server.state:
                    self.session.poll()

    # -- layout --------------------------------------------------------
    def _build_ui(self):
        from trame.ui.vuetify3 import VAppLayout
        from trame.widgets import html, vuetify3 as v3
        from .center_panel import build_center_panel
        from .workbench_shell import (
            build_design_plan,
            build_file_dialog,
            build_model_builder,
            build_toolbar,
            build_welcome,
        )

        server, ctrl = self.server, self.server.controller
        ctrl.load_template = self.load_template
        ctrl.open_workspace_project = self.open_workspace_project
        ctrl.discard_edits = self._refresh_settings
        ctrl.new_project = self.new_project
        ctrl.save_project = self.save_project
        ctrl.open_uploaded_project = self.open_uploaded_project
        ctrl.undo, ctrl.redo = self.undo, self.redo
        ctrl.apply_settings = self.apply_settings
        ctrl.open_advanced, ctrl.apply_advanced = self.open_advanced, self.apply_advanced
        ctrl.duplicate_channel, ctrl.remove_channel = self.duplicate_channel, self.remove_channel
        ctrl.toggle_channel = self.toggle_channel
        ctrl.compute, ctrl.compute_channels, ctrl.cancel_job = self.compute, self.compute_channels, self.cancel_job
        ctrl.export_geometry = self.export_geometry
        ctrl.view_job_result = self.view_job_result
        ctrl.review_submitted_design = self.review_submitted_design
        ctrl.show_shielding = self.show_shielding

        with VAppLayout(
            server,
            vuetify_config={
                "theme": {
                    "defaultTheme": "coilgen",
                    "themes": {
                        "coilgen": {
                            "dark": False,
                            "colors": {"primary": "#075fac", "surface": "#ffffff", "background": "#f4f4f4"},
                        }
                    },
                }
            },
        ):
            with html.Div(classes="cg-workbench"):
                build_toolbar(ctrl)
                build_welcome(ctrl)
                with html.Div(
                    v_show="channels.length || history_open",
                    classes="cg-work-area",
                    __properties=[("data_panel", "data-panel")],
                    data_panel=("workspace_panel",),
                ):
                    with html.Nav(
                        classes="cg-panel-switcher",
                        __properties=[("aria_label", "aria-label")],
                        aria_label="Workbench panels",
                    ):
                        for value, label in (
                            ("model", "Model Builder"),
                            ("center", "Plot and results"),
                            ("plan", "Design plan"),
                        ):
                            html.Button(
                                label,
                                type="button",
                                __properties=[("aria_pressed", "aria-pressed")],
                                aria_pressed=(f"workspace_panel === '{value}'",),
                                click=f"workspace_panel = '{value}'",
                            )
                    with html.Aside(
                        classes="cg-model", __properties=[("aria_label", "aria-label")], aria_label="Model Builder"
                    ):
                        build_model_builder(ctrl)
                    with html.Main(
                        classes="cg-center", __properties=[("aria_label", "aria-label")], aria_label="Plot and results"
                    ):
                        build_center_panel(self)
                    with html.Aside(
                        classes="cg-plan", __properties=[("aria_label", "aria-label")], aria_label="Design plan"
                    ):
                        build_design_plan(ctrl)
                with html.Footer(classes="cg-footer"):
                    html.Div(
                        "{{ settings_dirty ? 'Pending edits · open Model settings to apply or discard' : "
                        "current_job_active ? 'Computing · see Design plan' : "
                        "channels.length ? editing_label : 'No project open' }}",
                        role="status",
                    )
                    with html.Div(classes="cg-credits"):
                        html.Span("{{ author }} · ")
                        html.A("Project source", href=("project_url",), target="_blank", rel="noopener")
                build_file_dialog(ctrl)
            with v3.VDialog(
                v_model=("settings_dialog",),
                max_width=660,
                scrollable=True,
                __properties=[("aria_label", "aria-label")],
                aria_label="Model settings",
            ):
                with v3.VCard():
                    v3.VCardTitle("Model settings")
                    with v3.VCardText():
                        v3.VSelect(
                            v_model=("node",),
                            items=("nodes",),
                            label="Settings section",
                            density="compact",
                            classes="mt-3",
                            disabled=("settings_dirty",),
                        )
                        html.Div(
                            "{{ target_help }}", v_if="node === 'target' && target_help", classes="text-caption mb-2"
                        )
                        html.Div(
                            "{{ winding_help }}", v_if="node === 'winding' && winding_help", classes="text-caption mb-2"
                        )
                        with html.Div(
                            v_for="field in settings_fields",
                            key="field.key",
                            v_show="(node !== 'geometry' || field.key === 'coil_mesh_file' || settings_values.coil_mesh_file === applied_support) && "
                            "(field.key !== 'contour_smoothing_mesh_gain' || settings_values.contour_treatment === 'mesh_coupled') && "
                            "(field.key !== 'contour_smoothing_factor' || settings_values.contour_treatment === 'legacy') && "
                            "(field.key !== 'contour_smoothing_local_fallback' || "
                            "(settings_values.contour_treatment === 'mesh_coupled' && settings_values.contour_displacement_mm > 0 && "
                            "settings_values.contour_smoothing_preserve_min_turn_distance)) && "
                            "(!['contour_displacement_mm', 'contour_smoothing_preserve_min_turn_distance'].includes(field.key) || ['mesh_coupled', 'legacy'].includes(settings_values.contour_treatment)) && "
                            "(field.key !== 'contour_segment_mm' || settings_values.contour_treatment !== 'off')",
                        ):
                            v3.VTextField(
                                v_if="['number', 'integer', 'text', 'vector'].includes(field.kind)",
                                v_model="settings_values[field.key]",
                                label=("field.label",),
                                type=("['number', 'integer'].includes(field.kind) ? 'number' : 'text'",),
                                hint=("field.help",),
                                disabled=(
                                    "!!profile_preset || !!mesh_upload?.length || !!mesh_upload?.name || !!profile_upload",
                                ),
                                persistent_hint=("!!field.help",),
                                density="compact",
                                update_modelValue="flushState('settings_values')",
                            )
                            v3.VSelect(
                                v_if="field.kind === 'choice'",
                                v_model="settings_values[field.key]",
                                items=("field.choices",),
                                label=("field.label",),
                                hint=("field.help",),
                                disabled=(
                                    "!!profile_preset || !!mesh_upload?.length || !!mesh_upload?.name || !!profile_upload",
                                ),
                                persistent_hint=("!!field.help",),
                                density="compact",
                                update_modelValue="flushState('settings_values')",
                            )
                            v3.VCheckbox(
                                v_if="field.kind === 'bool'",
                                v_model="settings_values[field.key]",
                                label=("field.label",),
                                density="compact",
                                hint=("field.help",),
                                disabled=(
                                    "!!profile_preset || !!mesh_upload?.length || !!mesh_upload?.name || !!profile_upload",
                                ),
                                persistent_hint=("!!field.help",),
                                hide_details=("!field.help",),
                                update_modelValue="flushState('settings_values')",
                            )
                        with html.Div(v_if="node === 'conductor'", classes="mb-2"):
                            html.Div("Cross-section preset", classes="text-caption")
                            for value, title in (
                                ("rectangle", "Rectangle"),
                                ("round", "Round wire"),
                                ("u_shape", "U-shape"),
                                ("omega", "Omega holder"),
                            ):
                                v3.VBtn(
                                    title,
                                    size="small",
                                    classes="mr-1",
                                    active=(f"profile_preset === '{value}'",),
                                    click=f"profile_preset = '{value}'",
                                    disabled=("settings_dirty && !profile_preset",),
                                )
                            with html.Div(classes="d-flex flex-wrap ga-2 mt-2"):
                                v3.VTextField(
                                    v_model=("profile_segments",),
                                    label="Circle points",
                                    type="number",
                                    density="compact",
                                    style="max-width:10rem",
                                    hint="Round wire and U-shape (half circle uses half). Fewer points: smaller STL.",
                                    persistent_hint=True,
                                    disabled=("settings_dirty && !profile_preset",),
                                )
                                v3.VCheckbox(
                                    v_model=("profile_equal_area",),
                                    label="Keep circle area",
                                    density="compact",
                                    hint="Enlarge the round polygon so area and resistance match the round wire.",
                                    persistent_hint=True,
                                    disabled=("settings_dirty && !profile_preset",),
                                )
                            with html.Details(classes="mt-2"):
                                html.Summary("Import a cross-section table", style="cursor:pointer;font-weight:600")
                                html.Div(
                                    "Two columns u, n in millimetres, one point per row; separators ; tab , or "
                                    "spaces; an optional header line.",
                                    classes="text-caption",
                                )
                                v3.VFileInput(
                                    v_model=("profile_upload",),
                                    label="Import u, n table (mm)",
                                    accept=".csv,.tsv,.txt",
                                    density="compact",
                                    disabled=("(settings_dirty && !profile_upload) || !channels.length",),
                                )
                        with html.Div(v_if="node === 'geometry'", classes="mb-2"):
                            html.Div(
                                "{{ settings_values.coil_mesh_file === applied_support ? geometry_help : 'Apply the support selection to show its dimensions.' }}",
                                classes="text-caption mb-2",
                            )
                            with html.Details():
                                html.Summary("Import a support STL", style="cursor:pointer;font-weight:600")
                                v3.VTextField(
                                    v_model=("mesh_scale",),
                                    label="STL scale to metres",
                                    type="number",
                                    density="compact",
                                    hint="Set before choosing the STL. Use 0.001 for millimetres or 1 for metres.",
                                    persistent_hint=True,
                                )
                                v3.VFileInput(
                                    v_model=("mesh_upload",),
                                    label="Import support STL",
                                    accept=".stl",
                                    density="compact",
                                    disabled=("settings_dirty || !channels.length",),
                                )
                        html.Div(
                            "Pending edits · apply or discard before continuing.",
                            v_if="settings_dirty",
                            classes="text-caption text-warning",
                        )
                        with html.Div(classes="d-flex flex-wrap ga-1 mt-2"):
                            v3.VBtn(
                                "Advanced…",
                                variant="text",
                                click=ctrl.open_advanced,
                                disabled=("!channels.length || settings_dirty",),
                            )
                        with html.Div(v_if="preflight.length", classes="mt-3"):
                            html.Div("Readiness", classes="text-caption")
                            v3.VAlert(
                                "{{ issue.title }}",
                                v_for="issue in preflight",
                                type=("issue.blocking ? 'error' : issue.informational ? 'info' : 'warning'",),
                                density="compact",
                                variant="tonal",
                                classes="mb-1 text-caption",
                            )
                    with v3.VCardActions(classes="flex-wrap"):
                        v3.VBtn(
                            "Apply",
                            color="primary",
                            variant="flat",
                            click=ctrl.apply_settings,
                            disabled=("!settings_dirty",),
                        )
                        v3.VBtn("Discard edits", variant="text", click=ctrl.discard_edits, v_if="settings_dirty")
                        v3.VSpacer()
                        v3.VBtn("Cancel", click="settings_dialog = false")

            # A native select updates immediately in the client. Cover the previous
            # wire and checks until the selected archive and its scene are ready.
            with v3.VDialog(
                model_value=("!opening_workspace_file && !!selected_run && selected_run !== displayed_run",),
                persistent=True,
                transition=False,
                max_width=440,
            ):
                with v3.VCard(classes="pa-4"):
                    html.Strong("Loading saved result…")
                    html.Div(
                        "{{ runs.find(run => run.value === selected_run)?.title }}",
                        role="status",
                        classes="my-2",
                    )
                    v3.VProgressLinear(indeterminate=True)
            with v3.VDialog(model_value=("!!opening_workspace_file",), persistent=True, max_width=420):
                with v3.VCard(classes="pa-4"):
                    html.Strong("Opening {{ opening_workspace_file }}…")
                    html.Div("Verifying the project and its saved results.", role="status", classes="my-2")
                    v3.VProgressLinear(indeterminate=True)
            with v3.VDialog(
                v_model=("review_dialog",),
                max_width=780,
                scrollable=True,
                __properties=[("aria_label", "aria-label")],
                aria_label="Submitted design",
            ):
                with v3.VCard():
                    v3.VCardTitle("Submitted design")
                    with html.Div(classes="px-6 pb-2", style="overflow-wrap:anywhere"):
                        html.Strong("{{ run_source }}", classes="d-block")
                        html.Div("{{ run_context }} · read-only", classes="text-caption")
                    with v3.VCardText(style="overflow-wrap:anywhere"):
                        html.Div("Checking the original request…", v_if="review_busy", role="status")
                        with html.Div(v_if="design_review"):
                            v3.VAlert(
                                "Original settings unavailable. {{ design_review.error }}",
                                v_if="design_review.error",
                                type="warning",
                                variant="tonal",
                                density="compact",
                            )
                            with html.Div(v_else=True):
                                html.P(
                                    "These choices come from this run’s verified request. Expand the full request for every recorded field and study control.",
                                    classes="mb-2",
                                )
                                with html.Div(v_for="group in design_review.groups", key="group.title", classes="mb-3"):
                                    html.H3("{{ group.title }}", classes="text-subtitle-1 font-weight-bold")
                                    with html.Dl(classes="text-body-2"):
                                        with html.Div(
                                            v_for="row in group.rows",
                                            key="row.name",
                                            classes="d-flex flex-wrap align-baseline ga-2",
                                        ):
                                            html.Dt("{{ row.name }}", style="flex:1 1 12em")
                                            html.Dd("{{ row.value }}", style="flex:2 1 12em;min-width:0")
                                html.H3("Recorded smoothing outcome", classes="text-subtitle-1 font-weight-bold")
                                html.P(
                                    "No per-support smoothing outcome is recorded for this run.",
                                    v_if="!design_review.smoothing.length",
                                    classes="text-body-2 mb-2",
                                )
                                with html.Div(
                                    v_for="part in design_review.smoothing",
                                    key="part.part_index",
                                    classes="text-body-2 mb-2",
                                ):
                                    html.Strong("Support {{ part.part_index }}")
                                    html.Div(
                                        "{{ part.smoothed_count }} contours smoothed · {{ part.fit_rejected_count }} fits rejected · {{ part.spacing_rejected_count }} retained for spacing"
                                    )
                                    html.Div(
                                        "Of the accepted fits, {{ part.local_fit_count }} used local corner blends.",
                                        v_if="Number.isInteger(part.local_fit_count)",
                                    )
                                    html.Div(
                                        "Largest applied displacement {{ part.displacement_label }}",
                                    )
                                    html.Div(
                                        "{{ reason }} ({{ count }})", v_for="(count, reason) in part.failure_reasons"
                                    )
                                with html.Details(v_if="design_review.request_json", classes="my-2"):
                                    html.Summary(
                                        "Full submitted request (JSON)", style="cursor:pointer;font-weight:600"
                                    )
                                    html.Pre(
                                        "{{ design_review.request_json }}",
                                        style="white-space:pre-wrap;overflow-wrap:anywhere;font-size:0.8rem",
                                    )
                                html.A(
                                    "Download full submitted request (JSON)",
                                    v_if="design_review.download",
                                    href=("design_review.download",),
                                    download=True,
                                )
                    with v3.VCardActions():
                        v3.VSpacer()
                        v3.VBtn("Close", click="review_dialog = false", variant="tonal")
            with v3.VDialog(
                v_model=("advanced_dialog",),
                max_width=900,
                scrollable=True,
                __properties=[("aria_label", "aria-label")],
                aria_label="Advanced settings",
            ):
                with v3.VCard():
                    v3.VCardTitle("Advanced settings — every configuration and study field")
                    with v3.VCardText():
                        v3.VTextarea(
                            v_model=("advanced_config",), label="Configuration (JSON)", rows=16, classes="text-caption"
                        )
                        v3.VTextarea(v_model=("advanced_study",), label="Study (JSON)", rows=6, classes="text-caption")
                    with v3.VCardActions():
                        v3.VSpacer()
                        v3.VBtn("Cancel", click="advanced_dialog = false")
                        v3.VBtn("Apply", color="primary", click=ctrl.apply_advanced)
            v3.VSnackbar("{{ message }}", v_model=("show_message",), color=("message_color",), timeout=6000)


def _exit_with_parent(parent_pid: int, interval: float = 5.0):
    """A hub that crashes must not leave orphaned per-user workbenches."""
    import os
    import threading
    import time

    import psutil

    def watch():
        while psutil.pid_exists(parent_pid):
            time.sleep(interval)
        os._exit(0)

    threading.Thread(target=watch, name="coilgen-parent-watch", daemon=True).start()


def create_app(workspace, *, user="local", server_name=None, policy=None):
    """Build a trame server bound to a new session in ``workspace``."""
    from trame.app import get_server

    server = get_server(server_name or f"coilgen-{user}", client_type="vue3")
    session = WorkbenchSession(workspace, policy=policy)
    workbench = BrowserWorkbench(server, session, user=user)
    return server, workbench


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serve one CoilGen2 browser workbench session.")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--user", default="local")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1, help="Native threads per computation.")
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--parent-pid", type=int, default=None, help="Exit when this (hub) process ends.")
    options = parser.parse_args(argv)
    from ..research.jobs import RuntimePolicy

    if options.parent_pid:
        _exit_with_parent(options.parent_pid)

    server, workbench = create_app(
        options.workspace, user=options.user, policy=RuntimePolicy(max_workers=1, threads_per_worker=options.threads)
    )
    try:
        server.start(
            host=options.host,
            port=options.port,
            open_browser=options.open_browser,
            show_connection_info=True,
            timeout=0,
        )
    finally:
        workbench.session.close()


if __name__ == "__main__":
    main()
