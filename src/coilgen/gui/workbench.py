"""Native scientific workbench over the shared headless application services.

The controller edits ProjectDocuments and displays immutable RunRecords.
All numerical computation executes through the same process runtime as scripts.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

import numpy as np

from PySide6.QtCore import QEvent, QSignalBlocker, QSize, QStandardPaths, Qt, QThreadPool, QTimer
from PySide6.QtGui import QAction, QKeySequence, QPalette, QUndoCommand, QUndoStack
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..application import ASSEMBLY_SCHEMA, DesignService, ProjectAssembly, ProjectDocument, dependency_plan
from ..application.templates import TEMPLATE_SPECS, template_assembly, template_spec
from ..application.acceptance import (
    field_acceptance_slack_T,
    field_acceptance_slack_overrides,
    independent_field_acceptance_slack_T,
)
from ..application.targets import recognize_solid_harmonic_target, solid_harmonic_target
from ..application.reuse import plan_reuse, plan_stopped_reuse, ReusePlan
from ..application.evidence import SourceHealth
from ..application.execution import (
    STOPPED_JOB_STATES,
    execution_reuse,
    matches_active_request,
    matches_project,
    plan_execution,
    run_actions,
)
from ..application.history import HistoryEntry, RunHistoryStore
from ..application.diagnostics import run_verdicts
from ..application.bundles import RunArchive
from ..application.design_studies import (
    DesignStudyState,
    design_study_state,
    pin_baseline,
    create_variant,
    set_study_notes,
    set_study_notes_batch,
)
from ..application.assembly import LEGACY_ASSEMBLY_SCHEMAS
from ..attribution import AUTHOR_CREDIT, AUTHOR_ROLE, PROJECT_URL, software_attribution
from ..persistence import atomic_text
from ..research.jobs import JobManager, RuntimePolicy
from ..research.storage import result_metadata_digest
from .viewport import create_viewport
from .controls import ScientificSpinBox, SettingsPanel
from .editors import SUPPORT_EDITORS
from .history import HistoryPanel
from .problems import ProblemsPanel
from .readiness import ReadinessIssue, ReadinessPanel, check_readiness
from .colors import apply_tone, item_foreground, tone_color
from .guidance import CONTINUATION_STAGES, assess_guidance, resolved_model_summary
from .overview import DesignOverview
from .design_brief import DesignBriefPanel
from .acceptance_panel import AcceptancePanel
from .design_tasks import DesignStudyTask
from .saved_settings import SavedSettingsDialog, SavedSettingsTask
from .dataset_capabilities import resolve_dataset
from .export_panel import (
    ArtifactExportTask,
    ArtifactVerificationError,
    ExportReceiptPanel,
    GeometryExportPanel,
    GeometryExportState,
    GeometryExportTask,
    export_record_key,
    export_verified_artifacts,
    geometry_export_description,
)
from .job_history import job_origin, format_job_timestamp
from .winding_intent import WINDING_INTENTS, winding_intent, winding_overrides
from .icons import TARGET_MODE_ICONS, WINDING_ICONS, icon
from .visual_selectors import CircuitSelector, SurfaceGallery
from .channels_panel import ChannelOverview
from .target_regions import TargetRegionEditor, default_region
from .focused_editors import control_form, open_focused_control
from .choice_dialogs import GraphicalChoiceCombo
from .settings_sequences import GRAPHICAL_SETTINGS, build_settings_sequence
from .workspace_windows import EvidenceWindows, PanelWindow
from .view_context import ViewContextHeader, build_view_context
from .conductor_preview import ConductorPreview
from .conductor_profile import ConductorProfileEditor
from .design_progress import ComputationIndicator, DesignProgress
from .live_preview import LivePreviewController
from .job_outcomes import JobOutcomeNotice, build_job_outcome
from .about import AboutDialog


STAGES = (
    ("Prepare model", "prepared"),
    ("Optimize current sheet", "sheet"),
    ("Generate contours", "contours"),
    ("Connect / sweep conductors", "routed"),
    ("Full design evaluation", "evaluated"),
)
NODES = (
    ("geometry", "Geometry and mesh"),
    ("target", "Target field and regions"),
    ("material", "Conductor and material"),
    ("optimization", "Continuous-sheet optimization"),
    ("winding", "Winding and routing"),
    ("validation", "Validation settings"),
    ("results", "Results and provenance"),
    ("export", "Export conductor geometry"),
)
WORKFLOW_PAGES = (
    "geometry",
    "target",
    "material",
    "winding",
    "optimization",
    "validation",
    "compute",
    "results",
    "export",
)
NODE_ICONS = {
    "geometry": "mesh",
    "target": "target-gradient",
    "material": "conductor-profile",
    "optimization": "stream-function",
    "winding": "connected-winding",
    "validation": "independent-validation",
    "results": "field-error",
    "export": "export-conductor",
}
STAGE_ICONS = {
    "prepared": "mesh",
    "sheet": "stream-function",
    "contours": "contours",
    "routed": "conductor-sweep",
    "evaluated": "independent-validation",
}
PRIMARY_ACTION_STYLE = """
QPushButton { background: #075aaa; color: white; border: 1px solid #075aaa;
              border-radius: 5px; padding: 8px 14px; font-weight: 600; }
QPushButton:hover { background: #064b8e; }
QPushButton:pressed { background: #04386d; }
QPushButton:focus { border: 2px solid #002d5c; }
QPushButton:disabled { background: #dce3eb; color: #596777; border-color: #b9c5d2; }
"""
CONTROL_FIELDS = {
    "resistivity": "specific_conductivity_conductor",
    "sheet_thickness": "conductor_thickness",
    "wire_width": "conductor_cross_section_width",
    "wire_height": "conductor_cross_section_height",
    "insulation": "sweep_insulation_thickness",
    "objective": "research_physics",
    "current_limit": "surface_current_density_limit_A_per_m",
    "min_bend": "sweep_min_bend_radius",
    "route": "skip_postprocessing",
    "sweep": "skip_sweep",
    "independent_turns": "contour_only_sweep_mode",
    "return_path_lift": "skip_normal_shift",
    "normal_shift_mm": "normal_shift_length",
    "min_wire_distance_mm": "min_wire_distance_m",
}


def _target_icon(config):
    """Use a signed mode pictogram only for a recognized positive Bz mode."""
    if config.target_field_definition_file != "none":
        return "target-map"
    if config.target_field_component != "Bz":
        return "field-component"
    recipe = recognize_solid_harmonic_target(config.field_shape_function)
    if recipe is not None:
        if any(getattr(config, key) != value for key, value in recipe.to_config_overrides().items()):
            return "target-harmonic"
        modes = {0: ("Z2", 1), 1: ("XZ", -1), -1: ("YZ", -1), 2: ("X2-Y2", 1), -2: ("XY", 1)}
        if recipe.degree == 2 and recipe.order in modes:
            mode, sign = modes[recipe.order]
            if sign * recipe.coefficient_T > 0:
                return TARGET_MODE_ICONS[mode]
        return "target-harmonic"
    expression = str(config.field_shape_function).strip()
    if expression in {"x", "y", "z"} and config.target_gradient_strength > 0:
        return TARGET_MODE_ICONS[f"G{expression}"]
    try:
        if float(expression) > 0:
            return TARGET_MODE_ICONS["Z0"]
    except ValueError:
        pass
    return "field-component"


class ProjectEdit(QUndoCommand):
    def __init__(self, window, before, after, label):
        super().__init__(label)
        self.window, self.before, self.after = window, before, after
        self.channel_id = window._active_channel_id

    def redo(self):
        self.window._set_project(self.after, channel_id=self.channel_id)
        self.window._acknowledge_inputs("Applied" if self.window._applying else "Redid " + self.text())

    def undo(self):
        self.window._set_project(self.before, channel_id=self.channel_id)
        self.window._acknowledge_inputs("Undid " + self.text())


class AssemblyEdit(QUndoCommand):
    def __init__(self, window, after, selected, label):
        super().__init__(label)
        self.window, self.before, self.after = window, window.assembly, after
        self.before_selected, self.after_selected = window._active_channel_id, selected

    def redo(self):
        self.window._set_assembly(self.after, self.after_selected)

    def undo(self):
        self.window._set_assembly(self.before, self.before_selected)


class CoilGenWorkbench(QMainWindow):
    """Thin native client, with injectable runtime and viewport for testing."""

    def __init__(
        self,
        project=None,
        *,
        jobs_directory=None,
        jobs=None,
        service=None,
        viewport_factory=None,
        parent=None,
        startup_progress=None,
    ):
        super().__init__(parent)
        self.setObjectName("coilgen_workbench")
        self.resize(1440, 930)
        self.project = project
        self.assembly = (
            self.project
            if isinstance(self.project, ProjectAssembly)
            else ProjectAssembly.new(self.project.name, (self.project,))
            if self.project is not None
            else None
        )
        self._collection_mode = isinstance(project, ProjectAssembly)
        if self.assembly is not None and not self.assembly.channels:
            raise ValueError("The native workbench needs at least one channel.")
        self._active_channel_id = self._saved_active_channel(self.assembly) if self.assembly is not None else None
        self.project = self.assembly.channel(self._active_channel_id).project if self.assembly is not None else None
        self.service = service or DesignService()
        self.jobs = jobs
        self._own_jobs = False
        self.jobs_directory = (
            Path(jobs_directory)
            if jobs_directory
            else Path(jobs.directory)
            if jobs is not None
            else Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)) / "jobs"
        )
        self.live_preview = LivePreviewController(self.jobs_directory, self)
        self.history = None
        self._history_error = ""
        try:
            self.history = RunHistoryStore(self.jobs_directory.parent / "workbench-history.json")
        except (OSError, ValueError) as exc:
            self._history_error = str(exc)
        self._history_entries = ()
        self._history_ids = {}
        self._history_key = None
        self._restore_diagnostics = {}
        self.project_path = None
        self.runs = {}
        self.states = {}
        self._session_started_at = datetime.now(timezone.utc)
        self._session_job_ids = set()
        self._previous_job_ids = set()
        self._runtime_snapshot_taken = False
        self._last_job_states = ()
        self._observed_job_statuses = {}
        self._notified_job_outcomes = set()
        self._loaded_ids = set()
        self._selected_run = None
        self._draft_controls = {}
        self._focused_sections = []
        self._target_region_editor = None
        self._draft_initial = {}
        self._widget_initial = {}
        self._changed_controls = set()
        self._draft_builder = None
        self._draft_fields = {}
        self._field_errors = {}
        self._draft_error = ""
        self.preflight_label = None
        self._reuse_plan = ReusePlan((), (), ())
        self._preflight = None
        self._full_fabrication = False
        self._fabrication_preferences = {}
        self._draft_dirty = False
        self._applying = False
        self._current_node = None
        self._design_node = "material"
        self._last_runtime_error = None
        self._submitting = False
        self._expression_targets = {}
        self._expression_editor_fingerprint = None
        self._guidance_readiness_key = None
        self._guidance_readiness = None
        self._guidance_state = None
        self._model_change_note = ""
        self._geometry_export_task = None
        self._geometry_export_states = {}
        self._artifact_export_task = None
        self._design_study_task = None
        self._design_study_task_label = ""
        self._saved_settings_task = None
        self._acceptance_refresh_key = None
        self._restoring_panels = False
        self._closing = False
        self._panel_views = {}
        self.validation_resolution = 12
        self.design_margin = 1.0
        self.fabricate = False
        self.undo_stack = QUndoStack(self)
        self.undo_stack.cleanChanged.connect(self._update_title)
        if startup_progress is not None:
            startup_progress("Initializing 3D rendering...")
        self.viewport = (viewport_factory or create_viewport)(self)
        if startup_progress is not None:
            startup_progress("Building the workspace...")
        self._build_central()
        self._build_model_tree()
        self._build_settings_dock()
        self._build_evidence_windows()
        self._build_actions()
        self.computation_indicator = ComputationIndicator(self)
        self.statusBar().addPermanentWidget(self.computation_indicator, 1)
        self.statusBar().messageChanged.connect(self._refresh_computation_indicator)
        self.live_preview.preview_ready.connect(self._show_live_preview)
        self.live_preview.changed.connect(self._refresh_design_progress)
        for name in ("dataset", "inspection_task"):
            control = getattr(self.viewport, name, None)
            if control is not None:
                control.activated.connect(self._pause_live_follow)
        actions = getattr(self.viewport, "inspection_actions", {})
        for action in actions.values():
            action.triggered.connect(self._pause_live_follow)
        self.timer = QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.poll_jobs)
        self.timer.start()
        if hasattr(self.viewport, "selected"):
            self.viewport.selected.connect(self._inspect_node)
        if hasattr(self.viewport, "error"):
            self.viewport.error.connect(self._error)
        if hasattr(self.viewport, "preflight_ready"):
            self.viewport.preflight_ready.connect(self._show_preflight)
        if self.project is not None:
            self.select_node("geometry")
            self._set_project(self.project)
        else:
            self._set_workspace_active(False)
        self._fill_jobs(())
        self.poll_jobs()
        if self.screen() is not None:
            bounds = self.screen().availableGeometry()
            self.resize(min(1440, bounds.width()), min(930, bounds.height()))

    def _build_central(self):
        center = self.workspace = QWidget(self)
        layout = QVBoxLayout(center)
        self.model_label = QLabel()
        self.model_label.setWordWrap(True)
        self.model_label.setTextFormat(Qt.TextFormat.PlainText)
        self.model_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.model_label)
        self.design_brief = DesignBriefPanel(self)
        self.design_brief.section_requested.connect(self._open_overview_section)
        self.brief_window = PanelWindow("Physical design brief — CoilGen2", self.design_brief, self, size=(760, 560))
        self.result_state = QLabel("No run selected · current model geometry")
        self.result_state.setWordWrap(True)
        self.result_state.hide()
        self.model_label.hide()
        self.view_context = ViewContextHeader(self)
        self.view_context.history_requested.connect(lambda: self._open_evidence(self.history_panel))
        self.view_context.model_requested.connect(self._view_applied_model)
        layout.addWidget(self.view_context)
        if hasattr(self.viewport, "context_changed"):
            self.viewport.context_changed.connect(self._refresh_view_context)
        else:
            for name in ("dataset", "wire_current", "slice_axis", "component", "units"):
                control = getattr(self.viewport, name, None)
                if control is not None:
                    control.currentIndexChanged.connect(self._refresh_view_context)
            if hasattr(self.viewport, "slice_position"):
                self.viewport.slice_position.valueChanged.connect(self._refresh_view_context)
        self.readiness_panel = ReadinessPanel(self)
        self.readiness_panel.issue_requested.connect(self._open_readiness_issue)
        layout.addWidget(self.readiness_panel)
        self.modeling_notes_panel = ReadinessPanel(self)
        self.modeling_notes_panel.issue_requested.connect(self._open_readiness_issue)
        self.modeling_notes_window = PanelWindow(
            "Modeling notes — CoilGen2", self.modeling_notes_panel, self, size=(780, 350)
        )
        self.design_overview = DesignOverview(self)
        self.design_overview.action_requested.connect(self._guidance_action)
        self.design_overview.section_requested.connect(self._open_overview_section)
        self.design_overview.expand_button.setText("Pipeline progress and evidence")
        self.design_overview.expand_button.setAccessibleName("Show pipeline progress and evidence")
        self.design_overview.expand_button.setChecked(True)
        self.design_brief.expand_button.setChecked(True)
        self.overview_window = PanelWindow("Design status — CoilGen2", self.design_overview, self, size=(860, 640))
        destinations = QHBoxLayout()
        self.design_status_label = QLabel("Prepare your coil model")
        self.design_status_label.setObjectName("compact_design_status")
        self.design_status_label.setWordWrap(True)
        destinations.addWidget(self.design_status_label, 1)
        self.guidance_action_button = QPushButton(self)
        self.guidance_action_button.setObjectName("compact_guidance_action")
        self.guidance_action_button.clicked.connect(self._trigger_guidance_action)
        destinations.addWidget(self.guidance_action_button)
        self.modeling_notes_button = QPushButton(self)
        self.modeling_notes_button.setObjectName("modeling_notes_button")
        self.modeling_notes_button.setIcon(icon("information"))
        self.modeling_notes_button.clicked.connect(self.modeling_notes_window.open_panel)
        self.modeling_notes_button.hide()
        destinations.addWidget(self.modeling_notes_button)
        layout.addLayout(destinations)
        self.output_scope_label = QLabel()
        self.output_scope_label.setObjectName("requested_outputs")
        self.output_scope_label.setWordWrap(True)
        self.output_scope_label.hide()
        self.design_progress = DesignProgress(self)
        self.design_progress.follow_requested.connect(self._set_live_follow)
        layout.addWidget(self.design_progress)
        layout.addWidget(self.viewport, 1)
        self.workspace_stack = QStackedWidget(self)
        self.welcome = QWidget(self)
        self.welcome.setObjectName("empty_workspace")
        welcome = QVBoxLayout(self.welcome)
        welcome.addStretch(1)
        heading = QLabel("CoilGen2", self.welcome)
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heading.setStyleSheet("font-size: 26px; font-weight: 600;")
        welcome.addWidget(heading)
        subtitle = QLabel("Create a coil design, open a saved project, or start from a template.", self.welcome)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)
        welcome.addWidget(subtitle)
        for label, symbol, callback, name in (
            ("New project…", "project-json", self._new_project_dialog, "welcome_new_project"),
            ("Open project…", "open", self._open_dialog, "welcome_open_project"),
            ("Load template…", "load-template", self._welcome_template_menu, "welcome_load_template"),
        ):
            button = QPushButton(label, self.welcome)
            button.setObjectName(name)
            button.setIcon(icon(symbol))
            button.setIconSize(QSize(26, 26))
            button.setMinimumHeight(48)
            button.setMaximumWidth(340)
            button.clicked.connect(callback)
            welcome.addWidget(button, alignment=Qt.AlignmentFlag.AlignHCenter)
        welcome.addSpacing(16)
        credits = QLabel(f"{AUTHOR_CREDIT}\n{AUTHOR_ROLE}", self.welcome)
        credits.setObjectName("welcome_author_credit")
        credits.setAlignment(Qt.AlignmentFlag.AlignCenter)
        credits.setTextFormat(Qt.TextFormat.PlainText)
        credits.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        credits.setWordWrap(True)
        welcome.addWidget(credits)
        welcome.addStretch(1)
        self.workspace_stack.addWidget(self.welcome)
        self.workspace_stack.addWidget(center)
        central = QWidget(self)
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.addWidget(self.workspace_stack, 1)
        self.job_outcomes = JobOutcomeNotice(self)
        self.job_outcomes.action_requested.connect(self._open_job_outcome)
        central_layout.addWidget(self.job_outcomes)
        self.setCentralWidget(central)

    def _build_model_tree(self):
        self.tree = QTreeWidget()
        self.tree.setObjectName("model_tree")
        self.tree.setIconSize(QSize(22, 22))
        self.tree.setHeaderLabels(["Model Builder", "Solution state"])
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.setMinimumWidth(245)
        self.tree.setToolTip("Click a model or study item, or select it and press Enter, to open its settings.")
        self.tree_items = {}
        model = QTreeWidgetItem(["Model"])
        model.setIcon(0, icon("project-json"))
        model.setData(0, Qt.ItemDataRole.UserRole, "model")
        self.tree.addTopLevelItem(model)
        study = QTreeWidgetItem(["Study"])
        study.setIcon(0, icon("compute-study"))
        study.setData(0, Qt.ItemDataRole.UserRole, "study")
        self.tree.addTopLevelItem(study)
        for key, label in NODES:
            item = QTreeWidgetItem([label])
            item.setIcon(0, icon(NODE_ICONS[key]))
            item.setData(0, Qt.ItemDataRole.UserRole, key)
            if key in {"optimization", "validation"}:
                study.addChild(item)
            elif key == "results":
                self.tree.addTopLevelItem(item)
            elif key == "export":
                self.tree_items["results"].addChild(item)
            else:
                model.addChild(item)
            self.tree_items[key] = item
        self.stage_items = {}
        for label, stage in STAGES:
            item = QTreeWidgetItem([label, "Not computed"])
            item.setIcon(0, icon(STAGE_ICONS[stage], badge="unverified"))
            item.setData(0, Qt.ItemDataRole.UserRole, f"stage:{stage}")
            study.addChild(item)
            self.stage_items[stage] = item
        model.setExpanded(True)
        study.setExpanded(True)
        self.tree_items["results"].setExpanded(True)
        self.tree.currentItemChanged.connect(self._node_changed)
        self.tree.itemClicked.connect(lambda *_: self._show_dock("settings_dock"))
        self.tree.itemActivated.connect(lambda *_: self._show_dock("settings_dock"))
        dock = QDockWidget("Model Builder", self)
        dock.setObjectName("model_dock")
        dock.setWidget(self.tree)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self.resizeDocks([dock], [280], Qt.Orientation.Horizontal)

    def _view_applied_model(self):
        self._pause_live_follow()
        if hasattr(self.viewport, "clear_stage_preview"):
            self.viewport.clear_stage_preview()
        self._refresh_view_context()

    def _pause_live_follow(self, *_args):
        self.live_preview.pause()

    def _set_live_follow(self, following):
        if following:
            self.live_preview.resume()
        else:
            self.live_preview.pause()

    def _begin_live_follow(self, state):
        if not self._live_owner_matches(state):
            return
        self._view_applied_model()
        self.live_preview.start(state)

    def _live_owner_matches(self, state):
        return matches_project(state, self.project, revision=True)

    def _show_live_preview(self, job_id, preview):
        if (
            job_id != self.live_preview.job_id
            or not self.live_preview.following
            or not self._live_owner_matches(self.live_preview.state)
            or job_id in self.runs
        ):
            return
        if hasattr(self.viewport, "display_stage_preview"):
            self.viewport.display_stage_preview(preview.arrays, preview.metadata)
        self._refresh_view_context()
        self._refresh_design_progress()
        self._refresh_stage_status(refresh_reuse=False)

    def _refresh_design_progress(self, *_args):
        if not hasattr(self, "design_progress") or self.project is None:
            return
        state = self.live_preview.state
        if not self._live_owner_matches(state):
            state = {}
        completed = state.get("completed_stages", ())
        preview = self.live_preview.preview
        if preview is not None and state:
            completed = tuple(dict.fromkeys((*completed, *preview.metadata.get("completed_stages", ()))))
        source_id = state.get("id") if state else self._selected_run
        verification = "failed" if self._run_verification_error(source_id) else None
        record = self.runs.get(state.get("id")) if state else self.runs.get(self._selected_run)
        if record is not None and self._live_owner_matches(record.metadata):
            boundary = record.metadata.get("pipeline_stage")
            keys = tuple(key for _label, key in STAGES)
            if boundary in keys:
                completed = keys[: keys.index(boundary) + 1]
            if not state:
                state = {**self.states.get(self._selected_run, {}), **record.metadata, "status": "completed"}
            if verification != "failed":
                verification = "verified"
        elif state.get("status") == "completed" and verification is None:
            verification = "pending"
        visible = getattr(self.viewport, "stage_preview_metadata", {})
        if visible and hasattr(self.viewport, "set_stage_preview_status"):
            self.viewport.set_stage_preview_status(self.live_preview.state.get("status"))
        if hasattr(self.viewport, "set_verification_error"):
            self.viewport.set_verification_error(
                self._run_verification_error(self._selected_run) if not visible else None
            )
        self.design_progress.set_state(
            state.get("pipeline_stage", self.stage_selector.currentData()),
            state,
            completed=completed,
            reusable=(
                tuple(entry.stage.value for entry in self._reuse_plan.available)
                if getattr(self, "_reuse_plan_fingerprint", None) == self.project.fingerprint
                and verification != "failed"
                else ()
            ),
            preview_stage=visible.get("pipeline_stage") if visible else None,
            following=self.live_preview.following,
            pending=self._has_input_drafts(),
            verification=verification,
            unavailable=self._last_runtime_error,
        )
        if self.live_preview.error:
            self.design_progress.setToolTip("Live preview unavailable: " + self.live_preview.error)
        else:
            self.design_progress.setToolTip(self.output_scope_label.toolTip())

    def _refresh_view_context(self, *_args):
        if not hasattr(self, "view_context") or self.project is None:
            return
        record = self.runs.get(self._selected_run)
        entry = next(
            (
                entry
                for entry in self._history_entries
                if self._history_ids.get(entry.id, entry.id) == self._selected_run
            ),
            None,
        )
        dataset = getattr(self.viewport, "dataset", None)
        current = getattr(self.viewport, "wire_current", None)
        slab = getattr(self.viewport, "slice_label", None)
        context = build_view_context(
            self.project,
            record,
            channel_label=self.assembly.channel(self._active_channel_id).name,
            dataset_key=dataset.currentData() if dataset is not None else "support",
            pending=self._has_input_drafts(),
            alias=entry.alias if entry is not None else None,
            state=self.live_preview.state
            if getattr(self.viewport, "stage_preview_metadata", {})
            else self.states.get(self._selected_run),
            wire_current="original" if current is not None and current.currentIndex() == 1 else "stored",
            sample_slice=slab.text() if slab is not None else "",
            stage_preview=(self.live_preview.preview if getattr(self.viewport, "stage_preview_metadata", {}) else None),
            verification_error=self._run_verification_error(self._selected_run),
            dataset_capability=getattr(self.viewport, "current_capability", None),
        )
        draft_note = self._expert_draft_note()
        if draft_note:
            context = replace(
                context,
                editing_text=context.editing_text + " · " + draft_note,
                editing_tooltip=context.editing_tooltip + "\n" + draft_note,
            )
        if context.source == "run":
            context = replace(
                context,
                viewing_tooltip=context.viewing_tooltip
                + "\nModel/study compatibility compares saved parameters. Checkpoint reuse also verifies engine source, dependencies and saved artifact integrity; inspect Study or Recompute plan.",
            )
        self.view_context.set_context(context)
        self._refresh_design_progress()

    def _build_settings_dock(self):
        self.settings = QScrollArea()
        self.settings.setWidgetResizable(True)
        self.settings.setMinimumWidth(285)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.settings, 1)
        footer = QHBoxLayout()
        self.apply_settings_button = QPushButton("Apply & preview")
        self.apply_settings_button.setObjectName("apply_settings")
        self.apply_settings_button.setIcon(icon("check"))
        self.apply_settings_button.clicked.connect(self.apply_settings)
        self.revert_settings_button = QPushButton("Revert")
        self.revert_settings_button.setIcon(icon("undo"))
        self.revert_settings_button.clicked.connect(self.revert_draft)
        footer.addWidget(self.apply_settings_button, 1)
        footer.addWidget(self.revert_settings_button)
        layout.addLayout(footer)
        self.page_label = QLabel()
        self.page_label.setObjectName("workflow_page")
        layout.addWidget(self.page_label)
        navigation = QHBoxLayout()
        self.back_button = QPushButton("← Back")
        self.back_button.setIcon(icon("back"))
        self.back_button.setObjectName("workflow_back")
        self.back_button.clicked.connect(lambda: self.navigate_page(-1))
        self.next_button = QPushButton("Next →")
        self.next_button.setIcon(icon("next"))
        self.next_button.setObjectName("workflow_next")
        self.next_button.clicked.connect(lambda: self.navigate_page(1))
        navigation.addWidget(self.back_button)
        navigation.addWidget(self.next_button)
        layout.addLayout(navigation)
        dock = QDockWidget("Settings", self)
        dock.setObjectName("settings_dock")
        dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.resizeDocks([dock], [360], Qt.Orientation.Horizontal)
        dock.setFloating(True)
        bounds = self.screen().availableGeometry() if self.screen() else None
        dock.resize(min(520, bounds.width() - 20) if bounds else 520, min(760, bounds.height() - 50) if bounds else 760)
        dock.setAllowedAreas(Qt.DockWidgetArea.NoDockWidgetArea)
        dock.hide()

    @staticmethod
    def _table(headers):
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        table.horizontalHeader().setStretchLastSection(True)
        return table

    def _build_evidence_windows(self):
        self.evidence = EvidenceWindows(self)
        self.evidence.setObjectName("evidence_tabs")
        self.evidence.setTabsClosable(True)
        self.evidence.setDocumentMode(True)
        self.evidence.tabCloseRequested.connect(self._close_evidence)
        self.jobs_table = self._table(
            ["Design / job", "Session", "Submitted (local)", "Execution", "Activity", "Revision", "Progress", "Elapsed"]
        )
        self.jobs_table.setObjectName("jobs_table")
        self.jobs_table.cellClicked.connect(self._job_selected)
        self.jobs_table.itemSelectionChanged.connect(self._update_execution_feedback)
        self.jobs_table.setWordWrap(False)
        jobs_header = self.jobs_table.horizontalHeader()
        jobs_header.setStretchLastSection(False)
        jobs_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        jobs_header.resizeSection(0, 160)
        jobs_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        jobs_header.resizeSection(2, 190)
        jobs_header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        for column in (1, 3, 5, 6, 7):
            jobs_header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.jobs_table.horizontalHeaderItem(1).setToolTip(
            "This session means a job submitted from this workbench window. Earlier jobs retain their original identity after reopening."
        )
        self.jobs_table.horizontalHeaderItem(2).setToolTip(
            "Original submission time in your computer's local timezone, including jobs from previous days. Hover for exact UTC time."
        )
        self.checks_table = self._table(["Check", "Observed", "Requested limit", "Effective limit", "Unit", "Verdict"])
        self.checks_table.cellClicked.connect(self._check_selected)
        self.runs_table = self._table(
            [
                "Run",
                "Sheet checks",
                "Sweep checks",
                "Fabrication review",
                "Sheet P [W]",
                "Sheet Kmax [A/m]",
                "Independent sheet RMS [%]",
                "Revision",
            ]
        )
        self.runs_table.cellClicked.connect(self._run_selected)
        self.wire_fields_table = self._table(
            [
                "Wire field",
                "I [A]",
                "Refitted current",
                "RMS error [µT]",
                "Relative RMS [%]",
                "Peak error [µT]",
                "Samples",
                "Sampling scope",
            ]
        )
        self.wire_fields_table.setObjectName("wire_fields_table")
        self.messages = QPlainTextEdit()
        self.messages.setReadOnly(True)
        self.messages.document().setMaximumBlockCount(1000)
        self.recompute = QPlainTextEdit()
        self.recompute.setReadOnly(True)
        self.evidence.addTab(self.jobs_table, "Jobs")
        self.evidence.addTab(self.checks_table, "Checks")
        self.evidence.addTab(self.runs_table, "Run comparison")
        self.evidence.addTab(self.wire_fields_table, "Wire fields")
        self.evidence.addTab(self.recompute, "Recompute plan")
        self.evidence.addTab(self.messages, "Messages")
        self.acceptance_panel = AcceptancePanel(self)
        self.acceptance_panel.action_requested.connect(self._acceptance_action)
        self.acceptance_panel.saved_settings_requested.connect(self._inspect_saved_settings)
        self.acceptance_panel.witness_requested.connect(self._acceptance_witness)
        self.evidence.insertTab(1, self.acceptance_panel, "Design acceptance")
        self.problems_panel = ProblemsPanel(self)
        self.problems_panel.locate_requested.connect(self._locate_problem)
        self.problems_panel.navigate_requested.connect(self._inspect_selected_saved_settings)
        self.evidence.addTab(self.problems_panel, "Problems")
        self.history_panel = HistoryPanel(self)
        self.history_panel.selected.connect(self._history_selected)
        self.history_panel.alias_changed.connect(self._rename_history)
        self.history_panel.compare_requested.connect(self._compare_runs)
        self.history_panel.baseline_requested.connect(self._pin_history_baseline)
        self.history_panel.clear_baseline_requested.connect(self.clear_design_baseline)
        self.history_panel.variant_requested.connect(self._create_history_variant)
        self.history_panel.notes_changed.connect(self._save_design_notes)
        self.evidence.addTab(self.history_panel, "Project history")
        if self._history_error:
            self.messages.appendPlainText(self._history_error)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 4, 6, 4)
        feedback = QHBoxLayout()
        self.execution_label = QLabel("Ready to compute")
        self.execution_label.setObjectName("execution_feedback")
        self.execution_label.setWordWrap(True)
        self.execution_label.setMinimumWidth(180)
        feedback.addWidget(self.execution_label, 1)
        self.execution_progress = QProgressBar()
        self.execution_progress.setObjectName("execution_progress")
        self.execution_progress.setFixedWidth(160)
        self.execution_progress.setRange(0, 100)
        self.execution_progress.setValue(0)
        self.execution_progress.setToolTip(
            "Reported pipeline progress. Stage percentages describe work boundaries, not a time estimate."
        )
        self.execution_progress.hide()
        feedback.addWidget(self.execution_progress)
        self.execution_elapsed = QLabel()
        self.execution_elapsed.setObjectName("execution_elapsed")
        self.execution_elapsed.setToolTip("Time since submission, including time waiting for a worker.")
        feedback.addWidget(self.execution_elapsed)
        self.view_results_button = QPushButton("View results")
        self.view_results_button.setIcon(icon("field-error"))
        self.view_results_button.setObjectName("view_completed_results")
        self.view_results_button.setEnabled(False)
        self.view_results_button.setToolTip(
            "Open verified evidence in Review. Pending model edits are validated before navigation."
        )
        self.view_results_button.clicked.connect(self.view_results)
        feedback.addWidget(self.view_results_button)
        layout.addLayout(feedback)
        self.jobs_filter_bar = QWidget()
        jobs_context_layout = QHBoxLayout(self.jobs_filter_bar)
        jobs_context_layout.setContentsMargins(0, 0, 0, 0)
        self.jobs_context = QLabel()
        self.jobs_context.setObjectName("jobs_session_context")
        self.jobs_context.setTextFormat(Qt.TextFormat.PlainText)
        self.jobs_context.setWordWrap(True)
        jobs_context_layout.addWidget(self.jobs_context, 1)
        jobs_context_layout.addWidget(QLabel("Show:"))
        self.jobs_filter = QComboBox()
        self.jobs_filter.setObjectName("jobs_session_filter")
        self.jobs_filter.setAccessibleName("Show jobs by session")
        self.jobs_filter.addItem("All jobs", "all")
        self.jobs_filter.addItem("This session only", "current")
        self.jobs_filter.addItem("Earlier / other sessions", "previous")
        for index in range(self.jobs_filter.count()):
            self.jobs_filter.setItemIcon(index, icon("compute-study" if index == 1 else "run-history"))
        self._size_combo(self.jobs_filter)
        self.jobs_filter.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.jobs_filter.currentIndexChanged.connect(lambda: self._fill_jobs(self._last_job_states))
        jobs_context_layout.addWidget(self.jobs_filter)
        layout.addWidget(self.jobs_filter_bar)
        layout.addWidget(self.jobs_table, 1)
        self.evidence.set_content(self.jobs_table, container)
        evidence_icons = {
            "Jobs": "compute-study",
            "Design acceptance": "constraint",
            "Checks": "constraint",
            "Run comparison": "compare-designs",
            "Wire fields": "field-error",
            "Recompute plan": "design-variant",
            "Messages": "information",
            "Problems": "collision",
            "Project history": "run-history",
        }
        for index in range(self.evidence.count()):
            self.evidence.setTabIcon(index, icon(evidence_icons[self.evidence.tabText(index)]))
        self._job_verification_errors = {}
        self._source_health = SourceHealth()
        self._run_verification_paths = {}
        self._loading_job_id = None
        self._feedback_job_id = None
        self._restore_panel_layout({})

    def _action(self, text, slot, shortcut=None, *, icon_name=None):
        action = QAction(text, self)
        if icon_name:
            action.setIcon(icon(icon_name))
        action.triggered.connect(slot)
        if shortcut:
            action.setShortcut(shortcut)
        return action

    def _build_actions(self):
        file_menu = self.file_menu = self.menuBar().addMenu("File")
        self.new_action = self._action(
            "New project…", self._new_project_dialog, QKeySequence.StandardKey.New, icon_name="project-json"
        )
        file_menu.addAction(self.new_action)
        open_action = self._action("Open project…", self._open_dialog, QKeySequence.StandardKey.Open, icon_name="open")
        file_menu.addAction(open_action)
        self.template_menu = file_menu.addMenu(icon("load-template"), "Load template")
        self.template_menu.setObjectName("load_template_menu")
        self.template_menu.setToolTipsVisible(True)
        categories = {}
        for template in TEMPLATE_SPECS:
            if template.category not in categories:
                category_menu = self.template_menu.addMenu(template.category)
                category_menu.setToolTipsVisible(True)
                categories[template.category] = category_menu
            action = self._action(
                template.title,
                lambda _checked=False, selected=template.id: self.load_template(selected),
                icon_name=template.icon,
            )
            action.setData(template.id)
            action.setStatusTip(template.description)
            action.setToolTip(template.description)
            categories[template.category].addAction(action)
        save_action = self._action("Save project", self._save_dialog, QKeySequence.StandardKey.Save, icon_name="save")
        file_menu.addSeparator()
        file_menu.addAction(save_action)
        file_menu.addAction(
            self._action(
                "Save project as…", lambda: self._save_dialog(True), QKeySequence.StandardKey.SaveAs, icon_name="save"
            )
        )
        file_menu.addAction(self._action("Export active channel Python…", self._export_dialog, icon_name="export"))
        file_menu.addAction(
            self._action("Export enabled channels Python…", self._export_assembly_dialog, icon_name="export")
        )
        file_menu.addAction(self._action("Open completed run…", self._open_run_dialog, icon_name="run-history"))
        file_menu.addAction(self._action("Import support mesh…", self._import_mesh_dialog, icon_name="support-import"))
        file_menu.addAction(self._action("Import target map…", self._import_target_dialog, icon_name="target-map"))
        file_menu.addAction(self._action("Export selected run artifacts…", self._export_run_dialog, icon_name="export"))
        file_menu.addAction(
            self._action(
                "Export conductor geometry…", lambda: self.navigate_workflow("export"), icon_name="export-conductor"
            )
        )
        file_menu.addSeparator()
        self.close_project_action = self._action("Close project", self.close_project)
        file_menu.addAction(self.close_project_action)
        file_menu.addAction(self._action("Exit", self.close, QKeySequence.StandardKey.Quit))
        edit_menu = self.edit_menu = self.menuBar().addMenu("Edit")
        self.undo_action = self.undo_stack.createUndoAction(self, "Undo")
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.setIcon(icon("undo"))
        self.redo_action = self.undo_stack.createRedoAction(self, "Redo")
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self.redo_action.setIcon(icon("redo"))
        edit_menu.addActions([self.undo_action, self.redo_action])
        edit_menu.addAction(self._action("Advanced settings…", self._advanced_dialog, icon_name="settings"))
        edit_menu.addAction(self._action("Rename project…", self._rename_dialog, icon_name="edit"))
        self.gradient_menu = self.menuBar().addMenu("Gradient engineering")
        self.gradient_actions = {}
        for title, handler, symbol in (
            ("Gradient requirements…", self._gradient_requirements_dialog, "target-gradient"),
            ("Primary, shield and external region…", self._gradient_geometry_dialog, "mesh"),
            ("Queue convergence study…", self._gradient_convergence_dialog, "compute-study"),
            ("Review convergence study…", self._gradient_convergence_review, "compare-designs"),
            ("Terminals, feeds and drive waveform…", self._gradient_circuit_dialog, "connected-winding"),
            ("Compare measured field…", self._gradient_measured_dialog, "field-error"),
        ):
            action = self._action(title, handler, icon_name=symbol)
            self.gradient_actions[handler.__name__] = action
            self.gradient_menu.addAction(action)
        channels_menu = self.channels_menu = self.menuBar().addMenu("Channels")
        self.channel_overview_action = self._action("Channel overview…", self._channels_dialog, icon_name="channels")
        channels_menu.addAction(self.channel_overview_action)
        channels_menu.addAction(
            self._action("Duplicate active channel…", self._add_channel_dialog, icon_name="design-variant")
        )
        channels_menu.addAction(self._action("Remove active channel", self.remove_active_channel, icon_name="remove"))
        channels_menu.addAction(
            self._action("Compute enabled channels independently", self.compute_channels, icon_name="compute-study")
        )
        self.research_menu = self.menuBar().addMenu("Research studies")
        self.research_actions = {}
        for title, handler, symbol in (
            ("Named targets, validation and poses…", self._research_requirements_dialog, "target-gradient"),
            ("Review regional / scenario results…", self._research_requirements_review, "field-error"),
            ("Queue nominal and robust comparison", self._research_robust_comparison, "compare-designs"),
            ("Review nominal and robust comparison…", self._research_robust_review, "compare-designs"),
            ("Spatial scene and route anchors…", self._research_scene_dialog, "mesh"),
            ("Review spatial constraints…", self._research_scene_review, "constraint"),
            ("Array currents and cohort design…", self._research_array_dialog, "channels"),
            ("Circuit references and coupled physics…", self._research_dynamic_dialog, "connected-winding"),
            ("Review waveform and coupled physics…", self._research_dynamic_review, "field-error"),
        ):
            action = self._action(title, handler, icon_name=symbol)
            self.research_actions[handler.__name__] = action
            self.research_menu.addAction(action)
        toolbar = self.addToolBar("Project and study")
        toolbar.setObjectName("study_toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(20, 20))
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.channel_selector = QComboBox()
        self._size_combo(self.channel_selector)
        self.channel_selector.setMinimumContentsLength(25)
        self.channel_selector.setMaximumWidth(320)
        self.channel_selector.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.channel_selector.setObjectName("active_channel")
        self.channel_selector.setToolTip("Independent channel in the shared SI world frame")
        self.channel_selector.currentIndexChanged.connect(self._channel_changed)
        self.channel_selector_action = toolbar.addWidget(self.channel_selector)
        self.channel_enabled = QCheckBox("Enabled")
        self.channel_enabled.toggled.connect(self._channel_enabled_changed)
        self.channel_enabled_action = toolbar.addWidget(self.channel_enabled)
        self._fill_channels()
        self.resume_action = self._action("Resume selected run", self.resume_selected_run, icon_name="compute-study")
        self.resume_action.setEnabled(False)
        toolbar.addAction(self.resume_action)
        self.cancel_action = self._action("Cancel selected job", self.cancel_selected_job, icon_name="cancel")
        self.cancel_action.setEnabled(False)
        toolbar.addAction(self.cancel_action)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        self.stage_selector = QComboBox()
        self.stage_selector.setObjectName("compute_stage")
        self._size_combo(self.stage_selector)
        self.stage_selector.setMinimumContentsLength(24)
        self.stage_selector.setMaximumWidth(360)
        self.stage_selector.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        for label, key in STAGES:
            self.stage_selector.addItem(icon(STAGE_ICONS[key]), label, key)
        self.stage_selector.setToolTip(
            "Compute executes all required steps through this boundary. Winding options determine whether routing and solid sweeping run."
        )
        self.stage_selector.setCurrentIndex(1)
        self.stage_selector.currentIndexChanged.connect(self._stage_changed)
        self.stage_selector.hide()
        self.compute_button = QPushButton("Compute")
        self.compute_button.setIcon(icon("compute-study", foreground="primary"))
        self.compute_button.setIconSize(QSize(22, 22))
        self.compute_button.setObjectName("primary_compute")
        self.compute_button.setStyleSheet(PRIMARY_ACTION_STYLE)
        self.compute_button.setMinimumHeight(38)
        self.compute_button.setMinimumWidth(125)
        self.compute_button.setAutoDefault(False)
        self.compute_button.setAccessibleName("Compute selected study step")
        self.compute_button.setToolTip(
            "Compute through the selected boundary (Ctrl+Enter). Pending settings are validated first."
        )
        self.compute_button.clicked.connect(self.compute)
        toolbar.addWidget(self.compute_button)
        self.compute_action = self._action("Compute selected stage", self.compute, "Ctrl+Return")
        self.addAction(self.compute_action)
        view_menu = self.view_menu = self.menuBar().addMenu("View")
        self.field_analysis_action = self._action(
            "Detailed field analysis…", self._field_analysis_dialog, icon_name="field-error"
        )
        self.field_analysis_action.setEnabled(False)
        view_menu.addAction(self.field_analysis_action)
        inspection_menu = getattr(self.viewport, "inspection_menu", None)
        if inspection_menu is not None:
            inspection_menu.addSeparator()
            inspection_menu.addAction(self.field_analysis_action)
        panel_icons = {"model_dock": "model-tree", "settings_dock": "inspector", "evidence_dock": "evidence-panel"}
        for dock in self.findChildren(QDockWidget):
            action = dock.toggleViewAction()
            action.setIcon(icon(panel_icons[dock.objectName()]))
            view_menu.addAction(action)
        view_menu.addAction(self._action("Design status…", self.overview_window.open_panel, icon_name="constraint"))
        view_menu.addAction(
            self._action("Physical design brief…", self.brief_window.open_panel, icon_name="target-gradient")
        )
        evidence_menu = view_menu.addMenu("Open evidence window")
        evidence_menu.setIcon(icon("evidence-panel"))
        self.evidence_actions = {}
        details_menu = evidence_menu.addMenu("Detailed evidence")
        for index in range(self.evidence.count()):
            page = self.evidence.widget(index)
            action = self._action(
                self.evidence.tabText(index), lambda _checked=False, selected=page: self._open_evidence(selected)
            )
            action.setIcon(self.evidence.tabIcon(index))
            (
                evidence_menu
                if self.evidence.tabText(index) in {"Jobs", "Design acceptance", "Project history"}
                else details_menu
            ).addAction(action)
            action.setText(
                {"Design acceptance": "Results", "Project history": "History"}.get(
                    self.evidence.tabText(index), self.evidence.tabText(index)
                )
            )
            self.evidence_actions[self.evidence.tabText(index)] = action
        help_menu = self.menuBar().addMenu("Help")
        self.about_action = self._action("About CoilGen2…", self._about_dialog)
        help_menu.addAction(self.about_action)
        self._adapt_toolbars()

    def _about_dialog(self):
        dialog = AboutDialog(self)
        dialog.exec()
        dialog.deleteLater()

    def _update_title(self, *_args):
        self.setWindowTitle(f"{self.project.name} — CoilGen2[*]" if self.project is not None else "CoilGen2")
        self.setWindowModified(not self.undo_stack.isClean() or self._has_input_drafts())

    def _set_workspace_active(self, active):
        if not active:
            scene_editor = getattr(self, "_scene_editor", None)
            if scene_editor is not None:
                scene_editor.reject()
        self.workspace_stack.setCurrentWidget(self.workspace if active else self.welcome)
        self.job_outcomes.set_result_context(active)
        self.findChild(QDockWidget, "model_dock").setVisible(active)
        if not active:
            self.findChild(QDockWidget, "settings_dock").hide()
        self.tree.setEnabled(active)
        self.findChild(QToolBar, "study_toolbar").setVisible(active)
        self.edit_menu.setEnabled(active)
        self.gradient_menu.setEnabled(active)
        self.research_menu.setEnabled(active)
        self.channels_menu.setEnabled(active)
        self.view_menu.setEnabled(active)
        for action in self.file_menu.actions():
            if action.isSeparator():
                continue
            action.setEnabled(active or action.text() in {"New project…", "Open project…", "Load template", "Exit"})
        self.compute_button.setEnabled(active)
        self.compute_action.setEnabled(active)
        if not active:
            self._update_title()
            self.statusBar().showMessage("No project open")

    def _welcome_template_menu(self):
        button = self.welcome.findChild(QPushButton, "welcome_load_template")
        self.template_menu.popup(button.mapToGlobal(button.rect().bottomLeft()))

    def _new_project_dialog(self):
        from .project_start import NewProjectDialog

        if not self._can_discard():
            return
        dialog = NewProjectDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.new_project(name=dialog.project.name, project=dialog.project, discard_checked=True)

    def close_project(self):
        if not self._can_discard():
            return False
        self._clear_project()
        return True

    def _clear_project(self):
        self._save_session()
        self.live_preview.stop()
        self._discard_project_editors()
        self.evidence.close_all()
        for dialog in self.findChildren(QDialog):
            dialog.hide()
        for section in self._focused_sections:
            section.dispose()
        self._focused_sections = []
        self._target_region_editor = None
        panel = self.settings.takeWidget()
        if panel is not None:
            panel.deleteLater()
        self._draft_controls.clear()
        self._draft_initial.clear()
        self._widget_initial.clear()
        self._draft_fields.clear()
        self._field_errors.clear()
        self._draft_builder = None
        self._draft_error = ""
        self._settings_decisions = {}
        self._focused_node = None
        self.preflight_label = None
        self.draft_error_label = None
        self._draft_dirty = False
        self._changed_controls.clear()
        self._current_node = None
        self._history_key = None
        self._history_entries = ()
        self._selected_run = None
        self._preflight = None
        self._reuse_plan = ReusePlan((), (), ())
        self._guidance_readiness_key = None
        self._guidance_readiness = None
        self._expression_targets.clear()
        self._expression_editor_fingerprint = None
        self._model_change_note = ""
        self.runs.clear()
        self.states.clear()
        self.project = self.assembly = None
        self._active_channel_id = None
        self._collection_mode = False
        self.project_path = None
        self._panel_views.clear()
        self.undo_stack.clear()
        self.undo_stack.setClean()
        self._fill_channels()
        if hasattr(self.viewport, "clear_model"):
            self.viewport.clear_model()
        elif hasattr(self.viewport, "clear_run"):
            self.viewport.clear_run()
        self._set_workspace_active(False)
        return True

    def _fill_channels(self):
        with QSignalBlocker(self.channel_selector), QSignalBlocker(self.channel_enabled):
            self.channel_selector.clear()
            if self.assembly is None:
                self.channel_enabled.setChecked(False)
                return
            for channel in self.assembly.channels:
                self.channel_selector.addItem(icon(_target_icon(channel.project.config)), channel.name, channel.id)
                self.channel_selector.setItemData(
                    self.channel_selector.count() - 1, channel.name, Qt.ItemDataRole.ToolTipRole
                )
            self.channel_selector.setCurrentIndex(self.channel_selector.findData(self._active_channel_id))
            self.channel_enabled.setChecked(self.assembly.channel(self._active_channel_id).enabled)
        if hasattr(self, "channel_selector_action"):
            show_channels = (
                len(self.assembly.channels) > 1 or not self.assembly.channel(self._active_channel_id).enabled
            )
            self.channel_selector_action.setVisible(show_channels)
            self.channel_enabled_action.setVisible(show_channels)

    def _set_assembly(self, assembly, selected=None):
        self.assembly = assembly
        self._active_channel_id = selected or self._saved_active_channel(assembly)
        self._fill_channels()
        self._set_project(assembly.channel(self._active_channel_id).project)

    @staticmethod
    def _saved_active_channel(assembly):
        return next(
            (
                channel.id
                for channel in assembly.channels
                if channel.project.session.get("active_channel_id") == channel.id
            ),
            assembly.channels[0].id,
        )

    def _channel_changed(self, *_args):
        id_ = self.channel_selector.currentData()
        if id_ is None or id_ == self._active_channel_id:
            return
        if not self.apply_settings():
            self._fill_channels()
            return
        self._active_channel_id = id_
        self._fill_channels()
        self._set_project(self.assembly.channel(id_).project)

    def _channels_dialog(self):
        if not self.apply_settings():
            self._focus_draft_error()
            return
        previous = getattr(self, "channels_window", None)
        if previous is not None:
            previous.set_snapshot(
                self.assembly,
                self._active_channel_id,
                states=tuple(self.states.values()),
                records=self._channel_records(),
            )
            previous.show()
            previous.raise_()
            return
        dialog = ChannelOverview(
            self.assembly,
            self._active_channel_id,
            self,
            states=tuple(self.states.values()),
            records=self._channel_records(),
        )
        dialog.compute_button.setStyleSheet(PRIMARY_ACTION_STYLE)
        dialog.compute_button.setIcon(icon("compute-study", foreground="primary"))
        dialog.channel_requested.connect(self._open_channel)
        dialog.compute_requested.connect(self.compute_channels)
        self.channels_window = dialog
        dialog.setModal(False)
        dialog.show()
        dialog.raise_()

    def _refresh_channel_overview(self):
        if self.assembly is None:
            return
        dialog = getattr(self, "channels_window", None)
        if dialog is not None:
            dialog.set_snapshot(self.assembly, states=tuple(self.states.values()), records=self._channel_records())

    def _channel_records(self):
        # RunArchive validates its original metadata and artifact hashes when
        # loading the JSON. Inspecting metadata needs no array materialization.
        saved = tuple(archive for channel in self.assembly.channels for archive in channel.project.saved_runs)
        return (*saved, *(record for record in self.runs.values() if not self._run_verification_error(record)))

    def _open_channel(self, channel_id):
        index = self.channel_selector.findData(channel_id)
        if index >= 0:
            self.channel_selector.setCurrentIndex(index)
            self._show_dock("settings_dock")

    def _channel_enabled_changed(self, enabled):
        if not self.apply_settings():
            self._fill_channels()
            return
        try:
            self._enter_collection_mode()
            after = self.assembly.update_channel(self._active_channel_id, enabled=enabled)
            self.undo_stack.push(AssemblyEdit(self, after, self._active_channel_id, "Change enabled channel"))
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            self._fill_channels()
            self._error(f"Cannot change channel: {exc}")

    def _enter_collection_mode(self):
        if self._collection_mode:
            return
        self._save_session()
        channel = self.assembly.channel(self._active_channel_id)
        if self.history is not None:
            self.history.link_scope(channel.project.id, target_channel_id=channel.id)
        preference = self._fabrication_preferences.get((channel.project.id, None))
        if preference is not None:
            self._fabrication_preferences[(channel.project.id, channel.id)] = preference
        self._collection_mode = True

    def add_channel(self, name):
        if not self.apply_settings():
            raise ValueError("Correct pending settings before adding a channel.")
        # Detached revisions and asset bytes are safely shareable. A fresh
        # ProjectDocument ID distinguishes future independent run ownership.
        project = self.project.duplicate(name)
        after = self.assembly.add_channel(project, name=name)
        self._enter_collection_mode()
        self.undo_stack.push(AssemblyEdit(self, after, after.channels[-1].id, "Add independent channel"))
        return project

    def _add_channel_dialog(self):
        name, accepted = QInputDialog.getText(self, "Duplicate active channel", "New independent channel name")
        if accepted:
            try:
                self.add_channel(name)
            except (ValueError, TypeError, OSError) as exc:
                self._error(f"Cannot add channel: {exc}")

    def remove_active_channel(self):
        if len(self.assembly.channels) < 2:
            self._error("A workbench project must retain at least one channel.")
            return False
        if not self.apply_settings():
            return False
        after = self.assembly.remove_channel(self._active_channel_id)
        self.undo_stack.push(AssemblyEdit(self, after, after.channels[0].id, "Remove independent channel"))
        return True

    def _set_project(self, project, *, channel_id=None):
        previous = getattr(self, "project", None)
        scene_editor = getattr(self, "_scene_editor", None)
        if scene_editor is not None and scene_editor.property("sourceProjectId") != project.id:
            scene_editor.reject()
        next_channel = channel_id or self._active_channel_id
        key = (project.id, next_channel if self._collection_mode else None)
        if self._history_key is not None and self._history_key != key:
            self._save_session()
        if channel_id is not None:
            self._active_channel_id = channel_id
        channel = self.assembly.channel(self._active_channel_id)
        if channel.project is not project:
            self.assembly = self.assembly.update_channel(channel.id, project=project)
        self._fill_channels()
        self.project = project
        if self.live_preview.job_id and not self._live_owner_matches(self.live_preview.state):
            self.live_preview.stop()
            if hasattr(self.viewport, "clear_stage_preview"):
                self.viewport.clear_stage_preview()
        self._set_workspace_active(True)
        self._refresh_channel_overview()
        if previous is not None and previous.id != project.id:
            self._model_change_note = ""
        self.readiness_panel.set_issues(())
        self._draft_dirty = False
        self._preflight = None
        self._update_title()
        label = project.name
        if self._collection_mode:
            label = f"{self.assembly.name} · Channel {channel.name} ({len(self.assembly.channels)} channels)"
        self.model_label.setText(f"{label} · revision {project.revision}")
        self.model_label.setToolTip(f"SI model in world coordinates · numerical identity {project.fingerprint}")
        if hasattr(project, "study"):
            study = project.study
            with QSignalBlocker(self.stage_selector):
                self.stage_selector.setCurrentIndex(self.stage_selector.findData(study["pipeline_stage"]))
            current = self.tree.currentItem()
            selected_node = current.data(0, Qt.ItemDataRole.UserRole) if current is not None else ""
            if selected_node and selected_node.startswith("stage:"):
                # Restoring a study revision must restore its selected tree
                # boundary before building settings; otherwise undo would
                # interpret the old node as a new user stage selection.
                with QSignalBlocker(self.tree):
                    self.tree.setCurrentItem(self.stage_items[study["pipeline_stage"]])
            self.validation_resolution = study["validation_resolution"]
            self.design_margin = study["design_margin"]
            self.fabricate = study["fabricate"]
            if study["pipeline_stage"] == "evaluated":
                self._fabrication_preferences[key] = self.fabricate
            self._full_fabrication = self._fabrication_preferences.get(key, self.fabricate)
        if previous is not None and previous.fingerprint != project.fingerprint:
            plan = dependency_plan(previous.config, project.config)
            self._model_change_note = (
                "Changed: "
                + ", ".join(plan.changed_fields)
                + ". Recompute: "
                + (" → ".join(plan.stages) or "none")
                + "."
                if previous.id == project.id and plan.changed_fields
                else ""
            )
            self.recompute.setPlainText(
                "Changed inputs: "
                + ", ".join(plan.changed_fields)
                + "\n\nRequired computation: "
                + (" → ".join(plan.stages) or "none")
                + "\n\n"
                + "\n".join(plan.reasons)
                + "\n\nExisting runs remain immutable. Compute creates a new run."
            )
        if hasattr(self.viewport, "render_project"):
            self.viewport.render_project(project)
        else:
            self.viewport.render_config(project.config)
        self._node_changed(self.tree.currentItem())
        self._update_result_state()
        if self._history_key != key:
            self._restore_history(key)
        self._refresh_stage_status()
        self._refresh_history_panel()

    def _stage_changed(self, *_args):
        if self.project is None:
            return
        stage = self.stage_selector.currentData()
        if not self.apply_settings():
            if hasattr(self.project, "study"):
                with QSignalBlocker(self.stage_selector):
                    self.stage_selector.setCurrentIndex(
                        self.stage_selector.findData(self.project.study["pipeline_stage"])
                    )
            return
        current = self._current_node.data(0, Qt.ItemDataRole.UserRole) if self._current_node else ""
        if current and current.startswith("stage:") and current[6:] != stage:
            with QSignalBlocker(self.tree):
                self.tree.setCurrentItem(self.stage_items[stage])
        if hasattr(self.project, "update_study"):
            changes = {"pipeline_stage": stage, "fabricate": self._full_fabrication if stage == "evaluated" else False}
            self.apply_study(changes)

    def apply_study(self, changes):
        return self.apply_inputs(study_changes=changes, label="Edit study controls")

    def apply_config(self, changes, label="Edit scientific model"):
        """Atomic validated model edit, shared by native and advanced controls."""
        return self.apply_inputs(config_changes=changes, label=label)

    def apply_inputs(
        self, config_changes=None, study_changes=None, *, label="Edit model and study", expected_fingerprint=None
    ):
        """Commit a complete input transaction as one undoable revision."""
        if self.project is None:
            return False
        try:
            next_project = self.project.update_inputs(
                config_changes,
                study_changes,
                allow_incomplete=True,
                expected_fingerprint=expected_fingerprint,
            )
        except (ValueError, TypeError, OSError, KeyError) as exc:
            self._show_draft_error(str(exc))
            self._error(f"Input edit rejected: {exc}", open_window=False)
            return False
        if next_project.fingerprint != self.project.fingerprint:
            self._applying = True
            try:
                self.undo_stack.push(ProjectEdit(self, self.project, next_project, label))
            finally:
                self._applying = False
        return True

    def select_node(self, key):
        if self.project is None:
            return
        if key in self.tree_items:
            self.tree.setCurrentItem(self.tree_items[key])
        elif isinstance(key, str) and key.startswith("stage:") and key[6:] in self.stage_items:
            self.tree.setCurrentItem(self.stage_items[key[6:]])

    def _inspect_node(self, key):
        if self.project is None:
            return
        self.select_node(key)
        self._show_dock("settings_dock")

    def navigate_workflow(self, step):
        if self.project is None:
            return
        key = {
            "geometry": "geometry",
            "target": "target",
            "design": self._design_node,
            "compute": f"stage:{self.stage_selector.currentData()}",
            "review": "results",
            "export": "export",
        }.get(step)
        if key is not None:
            self.select_node(key)
            self._sync_workflow_navigation()
            self._show_dock("settings_dock")
            if self._draft_dirty and self._draft_error:
                self._focus_draft_error()
                return
            if self._workflow_page() == "compute":
                self.evidence.setCurrentWidget(self.jobs_table)
                self._show_dock("evidence_dock")
            if self._current_node and self._current_node.data(0, Qt.ItemDataRole.UserRole) == "results":
                self.evidence.setCurrentWidget(self.acceptance_panel)
                self._show_dock("evidence_dock")
                self._size_review_evidence()

    def _show_dock(self, name):
        if name == "evidence_dock":
            self.evidence.open_current()
            return
        dock = self.findChild(QDockWidget, name)
        dock.show()
        dock.raise_()

    def _open_evidence(self, page):
        self.evidence.open_page(page)

    def _close_evidence(self, index):
        """Close a view without altering completed records or active jobs."""
        if not 0 <= index < self.evidence.count():
            return
        self.evidence.close_page(self.evidence.widget(index))

    def _workbench_view_state(self):
        view = dict(self.viewport.view_state()) if hasattr(self.viewport, "view_state") else {}
        if view.get("source") == "live":
            view.update(source="model", dataset="preview_target")
        view["model_node"] = self._current_node.data(0, Qt.ItemDataRole.UserRole) if self._current_node else "geometry"
        view["evidence_tab"] = self.evidence.tabText(self.evidence.currentIndex())
        view["open_evidence_tabs"] = [
            self.evidence.tabText(i) for i in range(self.evidence.count()) if self.evidence.isTabVisible(i)
        ]
        for panel in ("model", "settings"):
            view[f"{panel}_panel_visible"] = not self.findChild(QDockWidget, f"{panel}_dock").isHidden()
        view["evidence_panel_visible"] = self.evidence.any_open()
        return view

    def _restore_panel_layout(self, view):
        """Restore display preferences; old projects start with a clear plot area."""
        view = view if isinstance(view, dict) else {}
        opened = view.get("open_evidence_tabs", [])
        opened = opened if isinstance(opened, list) else []
        self._restoring_panels = True
        try:
            self.evidence.close_all()
            with QSignalBlocker(self.evidence):
                for i in range(self.evidence.count()):
                    self.evidence.setTabVisible(i, self.evidence.tabText(i) in opened)
                visible = [i for i in range(self.evidence.count()) if self.evidence.isTabVisible(i)]
                selected = next(
                    (i for i in range(self.evidence.count()) if self.evidence.tabText(i) == view.get("evidence_tab")),
                    0,
                )
                if visible and selected not in visible:
                    selected = visible[0]
                self.evidence.setCurrentIndex(selected)
            for panel in ("model", "settings"):
                requested = view.get(f"{panel}_panel_visible", panel == "model") is True
                self.findChild(QDockWidget, f"{panel}_dock").setVisible(requested)
            if view.get("evidence_panel_visible") is True:
                for index in visible or [selected]:
                    self.evidence.open_page(self.evidence.widget(index))
                self.evidence.setCurrentIndex(selected)
            else:
                self.evidence.close_all()
        finally:
            self._restoring_panels = False

    def _size_review_evidence(self):
        self.evidence.open_current()

    def _focus_draft_error(self):
        control = next(
            (self._draft_controls[key] for key, error in self._field_errors.items() if not error.isHidden()), None
        )
        if control is not None:
            self._show_dock("settings_dock")
            if not open_focused_control(self, control):
                self.settings.ensureWidgetVisible(control)
            control.setFocus(Qt.FocusReason.OtherFocusReason)
            if isinstance(control, ConductorProfileEditor):
                control.focus_error()

    def _open_readiness_issue(self, issue):
        if issue.key == "expert_draft":
            self._advanced_dialog()
            return
        if issue.key == "channel_enabled":
            self.channel_enabled.setFocus(Qt.FocusReason.OtherFocusReason)
            return
        self.select_node(issue.node)
        self._show_dock("settings_dock")
        key = next((key for key, field in CONTROL_FIELDS.items() if field == issue.key), issue.key)
        control = self._draft_controls.get(key)
        if control is None and issue.key in {"coil_mesh_file", "field_shape_function"}:
            selector_name = "support_type" if issue.key == "coil_mesh_file" else "target_mode"
            control = self.settings.findChild(QComboBox, selector_name)
        if control is not None:
            if not open_focused_control(self, control):
                self.settings.ensureWidgetVisible(control)
            control.setFocus(Qt.FocusReason.OtherFocusReason)
            if isinstance(control, ConductorProfileEditor):
                control.focus_error()

    def _check_compute_readiness(self, project, *, channel_name=None):
        report = check_readiness(project, pipeline_stage=project.study["pipeline_stage"])
        if self.project is not None and project.id == self.project.id:
            self._refresh_modeling_notes(report)
        if report.ready:
            self.readiness_panel.set_issues(())
            return True
        self.readiness_panel.set_issues(report.blocking)
        prefix = f"Channel {channel_name}: " if channel_name else ""
        self.messages.appendPlainText(
            prefix + "Cannot compute yet:\n" + "\n".join(issue.message for issue in report.blocking)
        )
        self.statusBar().showMessage(prefix + "Cannot compute yet: " + report.blocking[0].title, 15000)
        if project.id == self.project.id:
            self._open_readiness_issue(report.blocking[0])
        return False

    def _report_compute_draft_error(self):
        key = self._current_node.data(0, Qt.ItemDataRole.UserRole) if self._current_node else "geometry"
        self.readiness_panel.set_issues(
            (
                ReadinessIssue(
                    title="Correct pending settings before computing",
                    message=self._draft_error or "Review the highlighted input errors.",
                    node=key,
                    key="",
                    action="Open settings",
                    severity="error",
                ),
            )
        )
        self._focus_draft_error()

    def _workflow_page(self):
        key = self._current_node.data(0, Qt.ItemDataRole.UserRole) if self._current_node else "geometry"
        return "compute" if key and key.startswith("stage:") else key

    def _sync_workflow_navigation(self):
        # The Model Builder is the sole section navigation.
        self._adapt_toolbars()

    def _adapt_toolbars(self):
        """Reserve the full Compute caption; optional channels use the remainder."""
        if not hasattr(self, "compute_button"):
            return
        self.compute_button.setMinimumWidth(max(125, self.compute_button.minimumSizeHint().width()))
        if hasattr(self, "channel_selector"):
            remaining = (
                self.width() - self.compute_button.minimumWidth() - self.channel_enabled.sizeHint().width() - 100
            )
            self.channel_selector.setMaximumWidth(max(100, min(320, remaining)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._adapt_toolbars()

    def navigate_page(self, offset):
        key = self._workflow_page()
        if key in WORKFLOW_PAGES:
            index = WORKFLOW_PAGES.index(key) + offset
            if 0 <= index < len(WORKFLOW_PAGES):
                target = WORKFLOW_PAGES[index]
                if target == "compute":
                    self.navigate_workflow("compute")
                else:
                    self.select_node(target)
                    self._show_dock("settings_dock")
                    if self._draft_dirty and self._draft_error:
                        self._focus_draft_error()

    def _node_changed(self, item, *_args):
        if item is None or self.project is None:
            return
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if key is None:
            return
        if key in {"model", "study"}:
            self.select_node("geometry" if key == "model" else f"stage:{self.stage_selector.currentData()}")
            return
        if self._draft_dirty and not self._applying:
            if not self.apply_settings():
                with QSignalBlocker(self.tree):
                    self.tree.setCurrentItem(self._current_node)
                self._sync_workflow_navigation()
                self._focus_draft_error()
                return
        self._current_node = item
        if key.startswith("stage:") and key[6:] != self.stage_selector.currentData():
            # Selecting a solution step makes the persistent Compute action
            # execute the step whose settings are visible.
            self.stage_selector.setCurrentIndex(self.stage_selector.findData(key[6:]))
            return
        if key in {"material", "optimization", "winding", "validation"}:
            if key in {"material", "winding"}:
                self._design_node = key
        self._sync_workflow_navigation()
        same_node = getattr(self, "_focused_node", None) == key
        old_sections = list(self._focused_sections)
        old_sections.extend(getattr(self._target_region_editor, "focused_sections", ()))
        reopened = {
            section.dialog.windowTitle(): section.dialog.geometry()
            for section in old_sections
            if same_node and section.dialog.isVisible()
        }
        for section in self._focused_sections:
            section.dispose()
        self._focused_sections = []
        self._target_region_editor = None
        self._focused_node = key
        container = SettingsPanel()
        self.findChild(QDockWidget, "settings_dock").setWindowTitle(item.text(0) + " — " + self.project.name)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        title = QLabel(item.text(0))
        title.setTextFormat(Qt.TextFormat.PlainText)
        title.setWordWrap(True)
        title.setStyleSheet("font-weight: 600; font-size: 15px")
        layout.addWidget(title)
        page = "compute" if key.startswith("stage:") else key
        if page in WORKFLOW_PAGES:
            position = WORKFLOW_PAGES.index(page)
            self.page_label.setText(f"Section: {item.text(0)}")
            self.back_button.setEnabled(position > 0)
            self.next_button.setEnabled(position < len(WORKFLOW_PAGES) - 1)
            if position < len(WORKFLOW_PAGES) - 1:
                destination = WORKFLOW_PAGES[position + 1]
                label = "Study" if destination == "compute" else dict(NODES)[destination]
                short_label = {
                    "target": "Target",
                    "material": "Material",
                    "optimization": "Optimization",
                    "winding": "Winding",
                    "validation": "Validation",
                    "compute": "Study",
                    "results": "Results",
                    "export": "Export",
                }[destination]
                self.next_button.setText(f"Next: {short_label} →")
                self.next_button.setToolTip(f"Open {label}. Pending edits are validated before navigation.")
            else:
                self.next_button.setText("Next →")
                self.next_button.setToolTip(
                    "Export is the last section; model, study and results remain directly accessible."
                )
            self.back_button.setToolTip(
                f"Open {dict(NODES).get(WORKFLOW_PAGES[position - 1], 'Study')}."
                if position > 0
                else "Geometry is the first section."
            )
        pending = QWidget()
        pending_layout = QVBoxLayout(pending)
        pending_layout.setContentsMargins(0, 0, 0, 0)
        self.pending_label = QLabel("Pending changes · applied on navigation, save or compute")
        self.pending_label.setObjectName("pending_changes")
        self.pending_label.setWordWrap(True)
        pending_layout.addWidget(self.pending_label)
        revert = QPushButton("Revert draft")
        revert.setIcon(icon("undo"))
        revert.setObjectName("revert_draft")
        revert.clicked.connect(self.revert_draft)
        pending_layout.addWidget(revert)
        pending.hide()
        self.pending_panel = pending
        layout.addWidget(pending)
        self.draft_error_label = QLabel()
        self.draft_error_label.setObjectName("draft_error")
        self.draft_error_label.setWordWrap(True)
        apply_tone(self.draft_error_label, "error")
        self.draft_error_label.hide()
        layout.addWidget(self.draft_error_label)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        container.form = form
        layout.addLayout(form)
        self._draft_controls = {}
        self._draft_initial = {}
        self._widget_initial = {}
        self._changed_controls = set()
        self._draft_builder = None
        self._draft_fields = {}
        self._field_errors = {}
        self._draft_error = ""
        self.preflight_label = None
        self._draft_dirty = False
        config = self.project.config

        notes = []

        def note(text):
            notes.append(text)

        def details(text):
            label = QLabel(text)
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setWordWrap(True)
            button = QToolButton()
            button.setText("Geometry details")
            button.setObjectName("geometry_definition")
            button.setIcon(icon("information"))
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            button.setAccessibleName("Show geometry coordinate and boundary details")
            window = PanelWindow("Geometry coordinates and boundaries", label, container, size=(620, 330))
            button.clicked.connect(window.open_panel)
            layout.addWidget(button)

        if key.startswith("stage:"):
            stage = key.split(":", 1)[1]
            candidate = self._reuse_plan.through(stage)
            note(
                "Compute runs all required steps through this boundary. "
                + (
                    f"Reuse verified {dict((key, label) for label, key in STAGES)[candidate.stage.value].lower()} evidence, then run the remaining steps."
                    if candidate
                    else "No compatible saved solution is available; computation starts from preparation."
                )
            )
            scope = QLabel(self._study_scope(stage, config))
            scope.setObjectName("study_scope")
            scope.setWordWrap(True)
            descriptions = {
                "prepared": "Inspect support meshes, target samples and field operators.",
                "sheet": "Optimize the continuous current sheet and check its field independently.",
                "contours": "Discretize the stream function into wire-current contours using the chosen policy.",
                "routed": "Run the requested interconnections and conductor sweeping. Circuit and solids settings determine the geometry.",
                "evaluated": "Evaluate the requested winding fields and electrical estimates. Review geometry and validation before export.",
            }
            selector = GraphicalChoiceCombo(
                choices={label: (STAGE_ICONS[boundary], label, descriptions[boundary]) for label, boundary in STAGES},
                title="Choose study output",
                columns=2,
            )
            selector.setObjectName("study_boundary")
            for label, boundary in STAGES:
                selector.addItem(label, boundary)
            selector.setCurrentIndex(selector.findData(stage))
            form.addRow("Requested output", selector)
            selector.currentIndexChanged.connect(
                lambda: self.stage_selector.setCurrentIndex(self.stage_selector.findData(selector.currentData()))
            )
            scope_window = PanelWindow("Requested study operations", scope, container, size=(640, 420))
            scope_button = QPushButton("Requested operations…")
            scope_button.setIcon(icon("compute-study"))
            scope_button.clicked.connect(scope_window.open_panel)
            layout.addWidget(scope_button)
            button = QPushButton("Use this study step")
            button.setIcon(icon("compute-study"))
            button.setObjectName("compute_through_stage")
            button.clicked.connect(
                lambda _checked=False, selected=stage: self.stage_selector.setCurrentIndex(
                    self.stage_selector.findData(selected)
                )
            )
            # Retain the programmatic action inside the detailed scope window.
            scope_window.layout().insertWidget(scope_window.layout().count() - 1, button)
            note(
                "Saved solutions retain their settings and checks. Execution completion does not imply engineering acceptance."
            )
        elif key == "geometry":
            note(
                f"Source: {config.coil_mesh_file}. Generated geometry remains editable. Imported assets are owned by the saved project."
            )
            selector = SurfaceGallery()
            selector.setObjectName("support_type")
            selected = selector.findData(config.coil_mesh_file)
            missing = str(config.coil_mesh_file).strip().lower() in {"", "none"}
            if selected < 0 and not missing:
                selector.addItem("Imported / custom support", config.coil_mesh_file)
                selected = selector.count() - 1
            selector.setCurrentIndex(selected)
            form.addRow(QLabel("Winding support surface"))
            form.addRow(selector)
            selector.currentIndexChanged.connect(lambda: self._switch_support(selector.currentData()))
            editor = next((entry for entry in SUPPORT_EDITORS if entry.source == config.coil_mesh_file), None)
            if editor:
                values = list(getattr(config, editor.parameter_field))
                for field in editor.fields:
                    value = editor.display_value(field, values)
                    if field.integer:
                        self._integer(form, field.key, field.label, int(value), int(field.minimum), int(field.maximum))
                    else:
                        self._number(form, field.key, field.label, value, field.suffix, field.minimum, field.maximum)
                    self._draft_fields[field.key] = (editor.parameter_field,)

                def geometry_changes():
                    return {
                        editor.parameter_field: editor.apply_edits(
                            values, {key: self._value(key) for key in self._changed_controls}
                        )
                    }

                self._draft_builder = geometry_changes
                details(editor.description)
            elif Path(config.coil_mesh_file).suffix.lower() == ".stl":
                self._number(
                    form,
                    "imported_mesh_length_scale_to_m",
                    "Source unit → metres",
                    config.imported_mesh_length_scale_to_m,
                    "",
                    1e-12,
                    1e9,
                )
                self._draft_builder = lambda: {
                    "imported_mesh_length_scale_to_m": self._value("imported_mesh_length_scale_to_m")
                }
            import_button = QPushButton("Import support mesh…")
            import_button.setObjectName("import_support")
            import_button.setIcon(icon("support-import"))
            import_button.clicked.connect(self._import_mesh_dialog)
            layout.addWidget(import_button)
            note(
                "Choose support contains the primitive gallery. Specialized mesh settings remain in Advanced settings. Preview uses the solver's geometry generator."
            )
        elif key == "target":
            self._build_target_editor(config, form, layout, note)
        elif key == "material":
            self._number(
                form, "resistivity", "Resistivity", config.specific_conductivity_conductor * 1e9, " nΩ·m", 0.000001, 1e9
            )
            self._number(
                form,
                "sheet_thickness",
                "Effective sheet thickness",
                config.conductor_thickness * 1000,
                " mm",
                0.000001,
                10000,
            )
            self._number(
                form,
                "wire_width",
                "Conductor width",
                config.conductor_cross_section_width * 1000,
                " mm",
                0.000001,
                10000,
            )
            self._number(
                form,
                "wire_height",
                "Conductor height",
                config.conductor_cross_section_height * 1000,
                " mm",
                0.000001,
                10000,
            )
            self._number(
                form, "insulation", "Insulation thickness", config.sweep_insulation_thickness * 1000, " mm", 0, 10000
            )
            self._combo(
                form,
                "conductor_mounting_mode",
                "Mounting reference",
                ["centered", "surface_outer", "surface_inner"],
                config.conductor_mounting_mode,
            )
            self._draft_controls["conductor_mounting_mode"].setToolTip(
                "Centered keeps the profile relative to the routed path. Outer/inner mounting offsets the conductor along the support normal. The saved fabrication centroid and field source must be reviewed."
            )
            self._check(form, "return_path_lift", "Lift crossing return paths", not config.skip_normal_shift)
            self._number(
                form, "normal_shift_mm", "Return-path lift", config.normal_shift_length * 1000, " mm", 0, 10000
            )
            self._draft_controls["normal_shift_mm"].setToolTip(
                "Normal displacement used by connected return-path routing. This is separate from conductor mounting and profile coordinates."
            )
            profile_editor = ConductorProfileEditor()
            profile_editor.set_profile(
                config.cross_sectional_points,
                width_mm=config.conductor_cross_section_width * 1000,
                height_mm=config.conductor_cross_section_height * 1000,
            )
            self._register(
                form, "cross_sectional_points", "Cross-section", profile_editor, initial=config.cross_sectional_points
            )
            self._draft_builder = lambda: {
                "specific_conductivity_conductor": self._value("resistivity") * 1e-9,
                "conductor_thickness": self._value("sheet_thickness") / 1000,
                "conductor_cross_section_width": self._value("wire_width") / 1000,
                "conductor_cross_section_height": self._value("wire_height") / 1000,
                "sweep_insulation_thickness": self._value("insulation") / 1000,
                "cross_sectional_points": self._value("cross_sectional_points"),
                "conductor_mounting_mode": self._value("conductor_mounting_mode"),
                "skip_normal_shift": not self._value("return_path_lift"),
                "normal_shift_length": self._value("normal_shift_mm") * 1e-3,
            }
            conductor_preview = ConductorPreview(container)
            layout.addWidget(conductor_preview)

            def show_cross_section(*_args):
                width = self._draft_controls["wire_width"]
                height = self._draft_controls["wire_height"]
                insulation = self._draft_controls["insulation"]
                polygon = profile_editor.mode_combo.currentData() == "polygon"
                for control, name in ((width, "width"), (height, "height")):
                    label = f"Nominal {name}" if polygon else f"Conductor {name}"
                    control_form(self, control, form).labelForField(control.parentWidget()).setText(label)
                    control.setAccessibleName(label)
                width.setToolTip(
                    "The polygon sets swept geometry, copper area and physical clearance. "
                    "Nominal width still sets PCB track width and some routing thresholds; "
                    "nominal dimensions remain in legacy reports."
                    if polygon
                    else "Rectangular conducting width, transverse to the wire within the support surface. Insulation is separate."
                )
                height.setToolTip(
                    "The polygon sets its physical height. Nominal height remains a rectangle input and legacy report setting."
                    if polygon
                    else "Rectangular conducting thickness along the support-surface normal. Insulation is separate."
                )
                w, h, t = (
                    control.valueFromText(control.text()) if control.hasAcceptableInput() else None
                    for control in (width, height, insulation)
                )
                profile_editor.set_rectangle_dimensions(w, h)
                if t is None or (not polygon and (w is None or h is None)):
                    conductor_preview.set_dimensions(None, None, None)
                    return
                try:
                    profile = profile_editor.profile()
                except ValueError as exc:
                    conductor_preview.set_invalid_profile(str(exc))
                    return
                if np.asarray(profile).shape == (1, 2) and np.all(np.asarray(profile) == 0):
                    conductor_preview.set_dimensions(w, h, t)
                else:
                    conductor_preview.set_profile(profile, t)

            for field in ("wire_width", "wire_height", "insulation"):
                self._draft_controls[field].valueChanged.connect(show_cross_section)
                self._draft_controls[field].lineEdit().textEdited.connect(show_cross_section)
            profile_editor.changed.connect(show_cross_section)
            show_cross_section()
            note(
                "Sheet thickness belongs to the continuous optimization model. Copper area sets resistance and Joule loss; larger profiles need more routing clearance. Custom points define the swept shape. Nominal width also affects PCB and legacy routing settings, so use a value consistent with the polygon extent."
            )
        elif key == "optimization":
            native_objective = "Native objective (see below)"
            objective = (
                native_objective
                if config.research_physics is None
                else config.research_physics.get("objective", "power")
            )
            self._combo(form, "objective", "Physical objective", ["power", "minimax", "magnetic_energy"], objective)
            self._draft_controls["objective"].setToolTip(
                "power: minimum sheet Joule loss [W]. minimax: minimum peak surface current density [A/m]. magnetic_energy: minimum stored magnetic energy [J]."
            )
            effective_method = "constrained_quadratic" if config.research_physics is not None else config.sf_opt_method
            self._combo(
                form, "sf_opt_method", "Solver formulation", ["constrained_quadratic", "tikhonov"], effective_method
            )
            self._number(
                form, "tikonov_reg_factor", "Tikhonov regularization factor", config.tikonov_reg_factor, "", 1e-12, 1e15
            )
            mode_info = QLabel()
            mode_info.setObjectName("optimization_mode_info")
            mode_info.setWordWrap(True)
            layout.addWidget(mode_info)

            def selected_physics():
                if (
                    self._value("sf_opt_method") != "constrained_quadratic"
                    or self._value("objective") == native_objective
                ):
                    return None
                return {**(config.research_physics or {}), "objective": self._value("objective")}

            def show_mode(_text=None):
                method = self._raw_value("sf_opt_method")
                tikhonov = method == "tikhonov"
                self._draft_controls["objective"].setEnabled(method == "constrained_quadratic")
                self._draft_controls["tikonov_reg_factor"].setEnabled(tikhonov)
                if tikhonov:
                    message = "Tikhonov fits a soft field residual with regularization. Field budgets are checked after solving. Physical force/torque, power/energy, field or scenario constraints must be explicitly removed in Advanced settings before switching. The hardware-limit policy can reject or rescale the solution; inspect independent validation."
                elif method == "constrained_quadratic" and self._raw_value("objective") == native_objective:
                    message = f"Native quadratic objective: mode={config.constrained_quadratic_objective_mode!r}; cost function={config.stream_function_cost_function!r}. Existing objective weights and regularization remain active; no research physical objective is configured. Selecting a physical objective switches to the physical convex solver. Full controls are available in Advanced settings."
                elif method == "constrained_quadratic":
                    message = "The physical convex solver enforces the field budget and configured physical bounds while minimizing the selected objective. Solver feasibility and independent field validation are reported separately. Magnetic energy is in joules; Joule power is in watts; peak surface current is in A/m."
                else:
                    message = f"The imported formulation {method!r} remains active. Its controls are available in Advanced settings. Choose constrained_quadratic to configure physical convex objectives."
                mode_info.setText(message)
                if "acceptance_slack" in self._draft_controls:
                    candidate = replace(config, research_physics=selected_physics(), sf_opt_method=method)
                    if "acceptance_slack" not in self._changed_controls:
                        allowance = field_acceptance_slack_T(candidate) * 1e9
                        widget = self._draft_controls["acceptance_slack"]
                        with QSignalBlocker(widget):
                            widget.setValue(allowance)
                        self._widget_initial["acceptance_slack"] = widget.value()
                        self._draft_initial["acceptance_slack"] = allowance
                    else:
                        candidate = replace(
                            candidate,
                            **field_acceptance_slack_overrides(candidate, self._value("acceptance_slack") * 1e-9),
                        )
                    independent = independent_field_acceptance_slack_T(candidate) * 1e9
                    acceptance_info.setText(
                        "This control edits the selected solver's absolute field allowance. "
                        + (
                            "The physical solver and independent check use the same policy. "
                            if candidate.research_physics is not None
                            else "The native solver allowance and independent research check use separate existing policies. "
                        )
                        + f"Selected-policy independent-grid allowance: {independent:.6g} nT. Relative budgets remain additional constraints."
                    )

            self._draft_controls["sf_opt_method"].currentTextChanged.connect(show_mode)
            self._draft_controls["objective"].currentTextChanged.connect(show_mode)
            show_mode()
            self._number(
                form,
                "target_error_tolerance_percent",
                "Relative field budget",
                config.target_error_tolerance_percent,
                " %",
                0.000001,
                1000,
            )
            self._number(
                form,
                "acceptance_slack",
                "Solver absolute acceptance slack",
                field_acceptance_slack_T(config) * 1e9,
                " nT",
                0,
                1e12,
            )
            self._draft_fields["acceptance_slack"] = (
                "research_physics",
                "target_error_constraint_acceptance_absolute_T",
            )
            acceptance_info = QLabel()
            acceptance_info.setObjectName("acceptance_policy_info")
            acceptance_info.setWordWrap(True)
            layout.addWidget(acceptance_info)
            self._draft_controls["acceptance_slack"].valueChanged.connect(show_mode)
            show_mode()
            self._number(
                form,
                "current_limit",
                "Maximum surface current density K",
                config.surface_current_density_limit_A_per_m,
                " A/m",
                0,
                1e12,
            )

            def optimization_changes():
                physics = dict(config.research_physics or {})
                objective = self._value("objective")
                if objective != native_objective:
                    physics["objective"] = objective
                method = self._value("sf_opt_method")
                if method == "tikhonov" and "sf_opt_method" in self._changed_controls:
                    constrained = [
                        name
                        for name in (
                            "sheet_power_limit_W",
                            "magnetic_energy_limit_J",
                            "force_limit_N",
                            "torque_limit_Nm",
                            "force_equality_N",
                            "torque_equality_Nm",
                        )
                        if physics.get(name) is not None
                    ]
                    constrained.extend(name for name in ("field_constraints", "scenarios") if physics.get(name))
                    if constrained:
                        raise ValueError(
                            "Tikhonov does not enforce these research constraints: "
                            + ", ".join(constrained)
                            + ". Explicitly remove them in Advanced settings before switching formulations; the current model is unchanged."
                        )
                changes = {
                    "research_physics": selected_physics(),
                    "sf_opt_method": method,
                    "tikonov_reg_factor": self._value("tikonov_reg_factor"),
                    "target_error_tolerance_percent": self._value("target_error_tolerance_percent"),
                    "surface_current_density_limit_A_per_m": self._value("current_limit"),
                }
                if "acceptance_slack" in self._changed_controls:
                    candidate = replace(config, research_physics=changes["research_physics"], sf_opt_method=method)
                    changes.update(field_acceptance_slack_overrides(candidate, self._value("acceptance_slack") * 1e-9))
                return changes

            self._draft_builder = optimization_changes
            note(
                f"Error metric: {config.target_error_constraint_metric}; fallback policy: {config.constrained_quadratic_fallback_policy}. Zero surface-current limit disables that bound. Force/torque, scenarios, regularization, basis and solver controls are in Advanced settings."
            )
        elif key == "winding":
            circuit = CircuitSelector()
            circuit.setCurrentText(winding_intent(config))
            self._register(form, "winding_intent", "Circuit intention", circuit)
            self._draft_fields["winding_intent"] = (
                "skip_postprocessing",
                "contour_only_sweep_mode",
                "routing_circuit_scope",
                "skip_sweep",
            )
            self._integer(form, "levels", "Contour levels", config.levels, 2, 1000)
            self._combo(
                form,
                "contour_discretization_method",
                "Discretization",
                ["level_count", "engineering", "auto"],
                config.contour_discretization_method,
            )
            self._number(form, "max_wire_current_A", "Wire current limit", config.max_wire_current_A, " A", 0, 1e12)
            self._draft_controls["max_wire_current_A"].setToolTip(
                f"Maximum Δψ per wire in engineering contouring. Zero retains the general max_current_A fallback ({config.max_current_A:g} A); it does not remove that bound."
            )
            self._number(
                form,
                "min_wire_distance_mm",
                "Engineering centerline distance",
                config.min_wire_distance_m * 1000,
                " mm",
                0,
                10000,
            )
            self._draft_controls["min_wire_distance_mm"].setToolTip(
                "Minimum centerline distance in engineering contouring. Zero uses the general minimum turn spacing. This is not an edge-to-edge copper clearance."
            )
            self._number(
                form,
                "min_turn_distance",
                "Minimum centerline spacing",
                config.min_turn_distance * 1000,
                " mm",
                0,
                10000,
            )
            self._number(form, "min_bend", "Minimum bend radius", config.sweep_min_bend_radius * 1000, " mm", 0, 10000)
            self._check(form, "sweep", "Generate conductor solids", not config.skip_sweep)
            winding_info = QLabel()
            winding_info.setObjectName("winding_scope")
            winding_info.setWordWrap(True)
            layout.addWidget(winding_info)

            def show_winding_scope(_value=None):
                mode = WINDING_INTENTS[self._raw_value("winding_intent")]
                sweep = self._raw_value("sweep")
                self._draft_controls["sweep"].setEnabled(mode != "contours")
                if mode == "independent":
                    text = "Each retained contour is an independent closed turn; interconnections are skipped. "
                    text += (
                        "A separate solid is swept for each retained turn."
                        if sweep
                        else "Solid generation is disabled; only contour geometry and field evidence will be available."
                    )
                elif mode in {"single", "per_part"}:
                    text = (
                        "Contours are cut and interconnected separately on each support part. "
                        if mode == "per_part"
                        else "Contours are cut and interconnected using the single-circuit routing scope. "
                    )
                    text += (
                        "The connected paths are swept into conductor solids."
                        if sweep
                        else "Solid generation is disabled; the connected centerlines remain available."
                    )
                else:
                    text = "Retain unconnected contour centerlines and their available field evidence. Solid generation requires a connected circuit or independent closed turns."
                    if sweep:
                        text += " The saved solid preference has no route to sweep in this mode."
                winding_info.setText(text)

            def change_intent(_value):
                if self._raw_value("winding_intent") == "Contours only":
                    self._draft_controls["sweep"].setChecked(False)
                show_winding_scope()

            self._draft_controls["winding_intent"].currentTextChanged.connect(change_intent)
            self._draft_controls["sweep"].toggled.connect(show_winding_scope)
            show_winding_scope()
            self._draft_builder = lambda: {
                "levels": self._value("levels"),
                "contour_discretization_method": self._value("contour_discretization_method"),
                "max_wire_current_A": self._value("max_wire_current_A"),
                "min_wire_distance_m": self._value("min_wire_distance_mm") * 1e-3,
                "min_turn_distance": self._value("min_turn_distance") / 1000,
                "sweep_min_bend_radius": self._value("min_bend") / 1000,
                "skip_sweep": not self._value("sweep"),
                **winding_overrides(config, self._value("winding_intent")),
            }
            note(
                "Contour step Δψ gives the wire current [A]; sheet current K is [A/m]. Feed/return conductors and drive terminals require explicit definition. Geometry closure alone does not establish a complete driven circuit."
            )
        elif key == "validation":
            self._integer(
                form, "validation_resolution", "Independent grid resolution", self.validation_resolution, 2, 128
            )
            self._number(form, "design_margin", "Design / validation budget ratio", self.design_margin, "", 0.001, 1.0)
            self._check(form, "fabricate", "Strict fabrication review", self.fabricate)
            self._draft_controls["fabricate"].setEnabled(self.stage_selector.currentData() == "evaluated")
            note(
                "These settings apply to the next study. Validation limit = design limit ÷ budget ratio. A ratio of 1 uses the same limit; 0.8 permits 1.25× the design limit. This does not change optimization tolerance. Independent sampling checks field generalization; mesh convergence requires a separate refinement study."
            )
        elif key == "export":
            record = self.runs.get(self._selected_run)
            options, diagnostic = (), self._run_verification_error(self._selected_run)
            if record is not None and not diagnostic:
                from ..application.cad import geometry_catalog

                try:
                    options = geometry_catalog(record)
                except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
                    diagnostic = f"Saved geometry verification failed: {exc}"
            self.geometry_export_panel = GeometryExportPanel(
                record,
                options,
                diagnostic=diagnostic,
                state=self._geometry_export_states.setdefault(export_record_key(record), GeometryExportState()),
            )
            self.geometry_export_panel.set_context(
                self.project, assembly=self.assembly, pending=self._has_input_drafts()
            )
            if record is not None and not self._owns_export_run(record):
                self.geometry_export_panel.include_project.setChecked(False)
                self.geometry_export_panel.include_project.setEnabled(False)
                self.geometry_export_panel.include_project.setToolTip(
                    "Open the run's owning project to include its inputs and results in the export package."
                )
                self.geometry_export_panel.scope.setText(
                    self.geometry_export_panel.scope.text()
                    + " Open its owning project to include a project JSON with this run."
                )
            self.geometry_export_panel.export_button.setStyleSheet(PRIMARY_ACTION_STYLE)
            self.geometry_export_panel.export_button.setIcon(icon("export-conductor", foreground="primary"))
            self.geometry_export_panel.set_exporting(self._geometry_export_task is not None)
            self.geometry_export_panel.export_requested.connect(self._geometry_export_dialog)
            self.geometry_export_panel.review_requested.connect(lambda: self._guidance_action("review_fabrication"))
            self.geometry_export_panel.problems_requested.connect(lambda: self._guidance_action("inspect_problems"))
            layout.addWidget(self.geometry_export_panel)
        else:
            note(
                "Select a completed job or run below to inspect its immutable geometry, constraints and provenance. Previous runs remain visible after model edits and are marked stale when their design fingerprint differs."
            )
            if self._selected_run is not None:
                self._build_run_inspector(layout, note)
        if key in {"geometry", "target"}:
            self.preflight_label = QLabel()
            self.preflight_label.setObjectName("preflight_summary")
            self.preflight_label.setWordWrap(True)
            preflight_window = PanelWindow(
                "Current model preview details", self.preflight_label, container, size=(680, 460)
            )
            preflight_button = QPushButton("Preview details…")
            preflight_button.setIcon(icon("mesh"))
            preflight_button.clicked.connect(preflight_window.open_panel)
            layout.addWidget(preflight_button)
            self._show_preflight(self._preflight)
        self._group_settings(key, form, layout)
        self.apply_settings_button.setVisible(bool(self._draft_controls))
        self.revert_settings_button.setVisible(bool(self._draft_controls))
        if notes:
            help_text = QLabel("\n\n".join(notes))
            help_text.setWordWrap(True)
            help_text.setTextFormat(Qt.TextFormat.PlainText)
            help_window = PanelWindow(item.text(0) + " — engineering notes", help_text, container, size=(680, 520))
            help_button = QPushButton("Engineering notes…")
            help_button.setIcon(icon("information"))
            help_button.clicked.connect(help_window.open_panel)
            layout.addWidget(help_button)
        engineering_actions = {
            "geometry": ("Primary and shield dimensions…", self._gradient_geometry_dialog),
            "target": ("Gradient requirements…", self._gradient_requirements_dialog),
            "winding": ("Define terminals, feeds and drive…", self._gradient_circuit_dialog),
            "validation": ("Convergence study…", self._gradient_convergence_dialog),
        }
        if key in engineering_actions:
            title, operation = engineering_actions[key]
            engineering_button = QPushButton(title)
            engineering_button.setObjectName("gradient_engineering_" + key)
            engineering_button.clicked.connect(operation)
            if key == "winding":
                record = self.runs.get(self._selected_run)
                engineering_button.setEnabled(
                    bool(record and record.metadata.get("winding_steps") and not self._run_verification_error(record))
                )
                engineering_button.setToolTip("Requires a verified Fabrication review for the applied numerical model.")
            layout.addWidget(engineering_button)
        advanced = QPushButton("Advanced settings…")
        advanced.setIcon(icon("settings"))
        advanced.clicked.connect(self._advanced_dialog)
        layout.addWidget(advanced)
        layout.addStretch()
        old = self.settings.takeWidget()
        self.settings.setWidget(container)
        if old:
            old.deleteLater()
        new_sections = list(self._focused_sections)
        new_sections.extend(getattr(self._target_region_editor, "focused_sections", ()))
        for section in new_sections:
            geometry = reopened.get(section.dialog.windowTitle())
            if geometry is not None:
                section.dialog.setGeometry(geometry)
                section.open()
        self._refresh_guidance()

    def _group_settings(self, key, form, layout):
        build_settings_sequence(self, key, form, layout)

    def _study_scope(self, stage, config):
        """Describe actual requested operations, without predicting acceptance."""
        index = [key for _label, key in STAGES].index(stage)
        lines = ["Support mesh, target samples and field operators."]
        if index >= 1:
            lines.append("Current-sheet optimization and independent sheet-field check.")
        if index >= 2:
            from ..engineering_limits import engineering_contouring_enabled, max_wire_current_A, min_wire_distance_m

            if engineering_contouring_enabled(config):
                lines.append(
                    f"Contour extraction: engineering current ≤{max_wire_current_A(config):g} A, "
                    f"centerline spacing ≥{min_wire_distance_m(config) * 1e3:g} mm; wire current comes from Δψ."
                )
            else:
                lines.append(f"Contour extraction: {config.levels} configured levels; wire current comes from Δψ.")
        if index >= 3:
            if config.contour_only_sweep_mode:
                lines.append("Interconnections skipped: independent closed turns.")
            elif config.skip_postprocessing:
                lines.append("Interconnections skipped by winding settings.")
            else:
                lines.append("Within-group and between-group interconnections, return paths and conductor centerlines.")
            if config.skip_sweep:
                lines.append("Conductor solids skipped by winding settings.")
            elif config.contour_only_sweep_mode:
                lines.append("Separate swept solid for each retained contour.")
            elif config.skip_postprocessing:
                lines.append("No connected route is available for solid sweeping.")
            else:
                lines.append("Swept conductor solids and native geometry checks.")
        if index >= 4:
            lines.append(
                "Final winding fields and electrical estimates; inspect each saved current and sampling policy."
            )
            if config.skip_inductance_calculation:
                lines.append("Inductance calculation skipped by model settings.")
            lines.append(
                "Strict fabrication review requested."
                if self.fabricate
                else "Strict fabrication review is not requested."
            )
        else:
            lines.append("Partial solution: final winding acceptance remains unverified.")
        return "Requested outputs and scope:\n" + "\n".join(f"• {line}" for line in lines)

    def _switch_support(self, source):
        if source == self.project.config.coil_mesh_file:
            return
        if not self.apply_settings():
            self._node_changed(self._current_node)
            return
        editor = next((entry for entry in SUPPORT_EDITORS if entry.source == source), None)
        if editor:
            self.apply_config(
                {"coil_mesh_file": source, "surface_is_cylinder_flag": editor.cylindrical},
                f"Use {editor.label.lower()} support",
            )

    def _new_target_region(self, shape, sampling="volume"):
        config = self.project.config
        region = default_region(config, shape, sampling)
        if (
            config.target_region_geometry is None
            and config.set_roi_into_mesh_center
            and shape != "enclosing_mesh"
            and str(config.coil_mesh_file).strip().lower() not in {"", "none"}
        ):
            from ..geometry import load_mesh_from_config

            mesh = load_mesh_from_config(config, self.project.project_root)
            region["center_m"] = np.mean(mesh.vertices, axis=0).tolist()
        return region

    def _reset_region_choices(self):
        editor = self._target_region_editor
        config = self.project.config
        region = config.target_region_geometry if isinstance(config.target_region_geometry, dict) else {}
        shape = region.get("shape", "sphere")
        if config.target_field_definition_file != "none":
            shape = "imported_map"
        elif config.target_region_geometry is None and config.target_mesh_file != "none":
            shape = "legacy_mesh"
        sampling = region.get("sampling", "volume")
        for selector, value in ((editor.gallery, shape), (editor.sampling, sampling)):
            with QSignalBlocker(selector):
                selector.setCurrentIndex(selector.findData(value))

    def _switch_target_region(self, shape):
        if self.project.config.target_field_definition_file != "none":
            self._reset_region_choices()
            return
        if shape in {"imported_map", "legacy_mesh"}:
            return
        if not self.apply_settings():
            self._reset_region_choices()
            return
        config = self.project.config
        previous = config.target_region_geometry if isinstance(config.target_region_geometry, dict) else {}
        if shape == "enclosing_mesh" and config.target_mesh_file == "none":
            self._import_region_dialog()
            self._reset_region_choices()
            return
        sampling = previous.get("sampling", "volume")
        if not isinstance(sampling, str) or sampling not in {"volume", "surface"}:
            sampling = "volume"
        try:
            region = self._new_target_region(shape, sampling)
            changes = {"target_region_geometry": region, "set_roi_into_mesh_center": False}
            if shape != "enclosing_mesh":
                changes["target_mesh_file"] = "none"
            else:
                from ..geometry import load_optional_target_mesh
                from ..target_regions import validate_target_region_mesh

                validate_target_region_mesh(
                    load_optional_target_mesh(config.target_mesh_file, config, self.project.project_root), sampling
                )
            self.apply_config(changes, f"Use {shape.replace('_', ' ')} target region")
        except (ValueError, TypeError, OSError, KeyError) as exc:
            self._error(f"Target region rejected: {exc}")
        self._reset_region_choices()

    def _switch_target_region_sampling(self, sampling):
        if self.project.config.target_field_definition_file != "none":
            self._reset_region_choices()
            return
        if not self.apply_settings():
            self._reset_region_choices()
            return
        config = self.project.config
        try:
            region = (
                dict(config.target_region_geometry)
                if config.target_region_geometry
                else self._new_target_region("sphere")
            )
            if region.get("sampling") == sampling:
                return
            region["sampling"] = sampling
            changes = {"target_region_geometry": region, "set_roi_into_mesh_center": False}
            if region["shape"] == "enclosing_mesh":
                from ..geometry import load_optional_target_mesh
                from ..target_regions import validate_target_region_mesh

                validate_target_region_mesh(
                    load_optional_target_mesh(config.target_mesh_file, config, self.project.project_root), sampling
                )
            self.apply_config(changes, f"Sample target region {sampling}")
        except (ValueError, TypeError, OSError, KeyError) as exc:
            self._error(f"Target sampling rejected: {exc}")
        self._reset_region_choices()

    def _build_run_inspector(self, layout, note):
        metadata = self.runs[self._selected_run].metadata
        summary = metadata.get("summary", {})
        fabrication = metadata.get("fabrication") or {}
        sweeps = metadata.get("pipeline_sweeps") or {}
        verdicts = run_verdicts(metadata)
        note(f"Continuous-sheet acceptance: {verdicts['sheet']}. Scope: {summary.get('scope', 'unverified')}.")
        note(
            f"Pipeline sweep geometry: {verdicts['sweeps']}. "
            + (
                f"{sweeps['body_count']} saved conductor {'body' if sweeps['body_count'] == 1 else 'bodies'}."
                if "body_count" in sweeps
                else "No sweep-check evidence was saved in this run."
            )
        )
        rows = [("Continuous sheet", "Power [W]", summary.get("power_w"))]
        independent = summary.get("validated_relative_field_error")
        rows.append(
            ("Independent sheet grid", "Relative RMS error [%]", None if independent is None else independent * 100)
        )
        contour = metadata.get("wire_fields", {}).get("contour", {})
        if contour.get("current_A") is not None:
            rows.append(("Original contours", "Current [A]", contour["current_A"]))
        for body in fabrication.get("electrical_metrics", {}).get("bodies", []):
            scope = f"Review: {body.get('body_id', 'conductor')}"
            for key, label, scale in (
                ("current_A", "Current [A]", 1),
                ("resistance_ohm", "Resistance [Ω]", 1),
                ("inductance_H", "Inductance [µH]", 1e6),
                ("joule_power_W", "Joule power [W]", 1),
            ):
                value = body.get(key)
                rows.append((scope, label, None if value is None else value * scale))
        metrics = self._table(["Model / body", "Quantity", "Value"])
        metrics.setObjectName("selected_run_metrics")
        metrics.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                item = QTableWidgetItem(self._format(value))
                if column == 1:
                    symbol = {
                        "Power [W]": "power-loss",
                        "Joule power [W]": "power-loss",
                        "Relative RMS error [%]": "field-error",
                        "Current [A]": "current-direction",
                        "Resistance [Ω]": "resistance",
                        "Inductance [µH]": "inductance",
                    }.get(value)
                    if symbol:
                        item.setIcon(icon(symbol))
                item.setToolTip(
                    self._format(value)
                    + "\nValues describe the explicitly labelled model. Review inductance retains the report's self/mutual and circuit assumptions; full details are in the verified run report."
                )
                metrics.setItem(row, column, item)
        metrics.horizontalHeader().setStretchLastSection(False)
        metrics.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        metrics.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        metrics.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        metrics.setFixedHeight(
            min(
                260,
                metrics.horizontalHeader().height()
                + sum(metrics.rowHeight(row) for row in range(min(7, len(rows))))
                + 4,
            )
        )
        layout.addWidget(metrics)
        reasons = list(fabrication.get("failure_reasons", []))
        for body in sweeps.get("bodies", []):
            codes = (body.get("diagnostics") or {}).get("manufacturing_failure_reasons", "")
            if codes:
                reasons.append(f"Pipeline {body.get('body_id', 'conductor')}:{codes}")
        note("Fabrication review: " + verdicts["fabrication"])
        note(
            "Construction release: " + ("Released" if summary.get("construction_released") is True else "Not released")
        )
        diagnostics = QPlainTextEdit()
        diagnostics.setObjectName("fabrication_failure_reasons")
        diagnostics.setReadOnly(True)
        lines = []
        for reason in reasons:
            scope, separator, codes = str(reason).partition(":")
            if separator:
                lines.append(scope + ":")
            else:
                codes = scope
            lines.extend(
                "  " + code.replace("_", " ").capitalize() + " [" + code + "]" for code in codes.split(";") if code
            )
        diagnostics.setPlainText(
            "\n".join(lines) or "No stored failure reasons. Missing or unverified checks remain unverified."
        )
        diagnostics.setMaximumHeight(180)
        layout.addWidget(diagnostics)
        note(
            "Failures identify the affected physical conductor. A sheet-field pass does not certify routing, conductor solids or the fabricated field. Review current and electrical metrics retain the report's geometry and circuit assumptions."
        )
        terminal_assumptions = []
        for step in metadata.get("winding_steps", []):
            if step.get("drive_feed_designed") is not True:
                terminal_assumptions.append(
                    f"{step.get('body_id', 'Conductor')}: {step.get('terminal_assumption', 'Drive feed and terminal connection are unverified.')}"
                )
        if terminal_assumptions:
            terminal_note = QLabel("Feed / terminal scope:\n" + "\n".join(dict.fromkeys(terminal_assumptions)))
            terminal_note.setObjectName("drive_feed_scope")
            terminal_note.setWordWrap(True)
            terminal_note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addWidget(terminal_note)

    def _build_target_editor(self, config, form, layout, note):
        requirements = self.project.study.get("requirements")
        if requirements is not None:
            note(
                "Named regional requirements supply this model's registered SI target vectors, independent validation data and finite pose bounds."
            )
            for region in requirements["regions"]:
                note(
                    f"{region['name']} · {region['component']} · RMS limit {region.get('rms_limit_T')} T · peak limit {region.get('peak_limit_T')} T"
                )
            edit = QPushButton("Edit named targets, validation and poses…")
            edit.clicked.connect(self._research_requirements_dialog)
            layout.addWidget(edit)
            restore = QPushButton("Use the single target definition")
            restore.clicked.connect(
                lambda: self.apply_inputs(study_changes={"requirements": None}, label="Use single target definition")
            )
            layout.addWidget(restore)
            self._draft_builder = lambda: {}
            return
        from ..target import _target_gradient_normalization_method

        recipe = recognize_solid_harmonic_target(config.field_shape_function)
        imported = config.target_field_definition_file != "none"
        missing = not imported and str(config.field_shape_function).strip().lower() in {"", "none"}
        canonical = recipe and all(getattr(config, key) == value for key, value in recipe.to_config_overrides().items())
        expression_override = self._expression_editor_fingerprint == (
            self._active_channel_id,
            config.field_shape_function,
        )
        harmonic = bool(canonical and not expression_override)
        linear = (
            not imported
            and not expression_override
            and str(config.field_shape_function).strip() in {"x", "y", "z"}
            and config.target_gradient_axis in {str(config.field_shape_function).strip(), "auto"}
            and _target_gradient_normalization_method(config, source="expression") == "linear_fit"
            and config.target_field_units == "T"
            and config.target_field_component == "Bz"
            and config.design_context == "mr_gradient"
        )
        mode = (
            "Imported map"
            if imported
            else "Spherical harmonic"
            if harmonic
            else "Linear gradient"
            if linear
            else "Analytic expression"
        )
        selector = GraphicalChoiceCombo(
            title="Choose target field",
            choices={
                "Linear gradient": (
                    "target-gradient",
                    "Linear gradient",
                    "Define dBz/dx, dBz/dy or dBz/dz and its physical amplitude.",
                ),
                "Analytic expression": (
                    "field-component",
                    "Analytic expression",
                    "Define a scalar Bz expression in world coordinates with explicit units and scaling.",
                ),
                "Spherical harmonic": (
                    "target-harmonic",
                    "Spherical harmonic",
                    "Choose a real solid harmonic, coefficient, reference radius and origin.",
                ),
                "Imported map": (
                    "target-map",
                    "Imported map",
                    "Keep the loaded coordinates and scalar/vector field values authoritative.",
                ),
            },
        )
        selector.setObjectName("target_mode")
        selector.addItems(["Linear gradient", "Analytic expression", "Spherical harmonic"])
        if imported:
            selector.addItem("Imported map")
        target_icons = {
            "Linear gradient": "target-gradient",
            "Analytic expression": "field-component",
            "Spherical harmonic": "target-harmonic",
            "Imported map": "target-map",
        }
        for index in range(selector.count()):
            selector.setItemIcon(index, icon(target_icons[selector.itemText(index)]))
        if missing:
            selector.setCurrentIndex(-1)
        else:
            selector.setCurrentText(mode)
        form.addRow("Target definition", selector)
        selector.currentTextChanged.connect(self._switch_target_mode)
        import_button = QPushButton("Import target map…")
        import_button.setObjectName("import_target")
        import_button.setIcon(icon("target-map"))
        import_button.clicked.connect(self._import_target_dialog)
        layout.addWidget(import_button)
        if missing:
            note("Choose a target field or import a measured field map.")
            return
        region = (
            config.target_region_geometry if not imported and isinstance(config.target_region_geometry, dict) else {}
        )
        radius = region.get("radius_m", config.target_region_radius)
        if (
            not isinstance(radius, (int, float, np.number))
            or isinstance(radius, (bool, np.bool_))
            or not np.isfinite(radius)
            or radius <= 0
        ):
            radius = config.target_region_radius
        self._number(form, "target_region_radius", "ROI radius", radius * 1000, " mm", 0.001, 10000)
        self._integer(form, "target_region_resolution", "Fitting resolution", config.target_region_resolution, 2, 128)
        self._draft_controls["target_region_resolution"].setToolTip(
            "Sampling resolution for the generated design grid, not the surface mesh. The preview reports the actual retained point count. Independent validation has its own grid under Study → Validation settings."
        )
        if harmonic:
            self._integer(form, "harmonic_degree", "Degree l", recipe.degree, 0, 16)
            self._integer(form, "harmonic_order", "Order m", recipe.order, -recipe.degree, recipe.degree)
            self._number(
                form, "harmonic_coefficient_uT", "Real Ylm coefficient", recipe.coefficient_T * 1e6, " µT", -1e12, 1e12
            )
            self._number(
                form,
                "harmonic_reference_radius_mm",
                "Harmonic reference radius",
                recipe.reference_radius_m * 1000,
                " mm",
                0.001,
                10000,
            )
            for index, axis in enumerate("xyz"):
                self._number(
                    form,
                    f"harmonic_center_{axis}_mm",
                    f"Harmonic center {axis}",
                    recipe.center_m[index] * 1000,
                    " mm",
                    -10000,
                    10000,
                )
            self._draft_controls["harmonic_degree"].valueChanged.connect(
                lambda degree: self._draft_controls["harmonic_order"].setRange(-degree, degree)
            )
            target_fields = tuple(recipe.to_config_overrides())
            for control in self._draft_controls:
                if control.startswith("harmonic_"):
                    self._draft_fields[control] = target_fields

            def target_changes():
                def edited(key, original, scale=1):
                    return self._value(key) * scale if key in self._changed_controls else original

                target = solid_harmonic_target(
                    edited("harmonic_degree", recipe.degree),
                    edited("harmonic_order", recipe.order),
                    edited("harmonic_coefficient_uT", recipe.coefficient_T, 1e-6),
                    edited("harmonic_reference_radius_mm", recipe.reference_radius_m, 1e-3),
                    tuple(
                        edited(f"harmonic_center_{axis}_mm", recipe.center_m[index], 1e-3)
                        for index, axis in enumerate("xyz")
                    ),
                )
                return {
                    **target.to_config_overrides(),
                    "target_region_radius": self._value("target_region_radius") * 1e-3,
                    "target_region_resolution": self._value("target_region_resolution"),
                }

            self._draft_builder = target_changes
            note(
                "Bz = coefficient × (r/R)^l × real Ylm, using CoilGen's normalized real harmonic and Condon–Shortley phase. The coefficient is not a pole amplitude: l=2, m=0 gives 0.630783 × coefficient on the +z pole at r=R; l=0 gives coefficient/√(4π). Center and R use world coordinates. Changing the ROI does not change R or the target amplitude."
            )
            note(
                "Absolute tesla target · Bz · MR shim context · gradient normalization off. Gradient strength and gradient axis do not rescale this analytic target. Independent validation evaluates the same expression on its own grid."
            )
        else:
            initial_method = _target_gradient_normalization_method(config, source="expression")
            initial_multiplier = initial_method == "scale"
            initial_hz = str(config.target_field_units).lower() == "hz"
            strength_scale = {"value": 1.0 if initial_multiplier or initial_hz else 1000.0}
            self._text(form, "field_shape_function", "Expression → Bz", config.field_shape_function)
            components = ["Bx", "By", "Bz", "BdotB0", "vector"] if imported else ["Bz"]
            self._combo(form, "target_field_component", "Fitting component", components, config.target_field_component)
            self._number(
                form,
                "target_gradient_strength",
                "Amplitude multiplier" if initial_method == "scale" else "Gradient amplitude",
                config.target_gradient_strength * strength_scale["value"],
                "" if initial_multiplier else " Hz/m" if initial_hz else " mT/m",
                0.000001,
                100000,
            )
            self._combo(
                form,
                "target_gradient_normalization",
                "Normalization",
                ["auto", "off", "none", "linear_fit", "legacy"],
                config.target_gradient_normalization,
            )
            self._combo(
                form, "target_gradient_axis", "Gradient axis", ["x", "y", "z", "auto"], config.target_gradient_axis
            )
            self._combo(
                form, "target_field_units", "Field expression/map units", ["T", "Hz"], config.target_field_units
            )
            self._combo(
                form,
                "design_context",
                "Design context",
                ["mr_gradient", "mr_shim", "free", "mr_rf"],
                config.design_context,
            )
            self._draft_builder = lambda: {
                "field_shape_function": self._value("field_shape_function"),
                "target_field_component": self._value("target_field_component"),
                "target_gradient_strength": self._value("target_gradient_strength") / strength_scale["value"],
                "target_region_radius": self._value("target_region_radius") / 1000,
                "target_region_resolution": self._value("target_region_resolution"),
                "target_gradient_normalization": self._value("target_gradient_normalization"),
                "target_gradient_axis": self._value("target_gradient_axis"),
                "target_field_units": self._value("target_field_units"),
                "design_context": self._value("design_context"),
            }
            if imported:
                for key in (
                    "field_shape_function",
                    "target_gradient_strength",
                    "target_gradient_normalization",
                    "target_gradient_axis",
                    "target_field_units",
                    "target_region_radius",
                    "target_region_resolution",
                ):
                    self._draft_controls[key].setEnabled(False)
                note(
                    f"Imported target: {config.target_field_definition_file}. Map values and coordinates are authoritative. Choosing an analytic definition explicitly replaces this map."
                )
            else:
                gradient_summary = QLabel()
                gradient_summary.setObjectName("linear_gradient_summary")
                gradient_summary.setTextFormat(Qt.TextFormat.PlainText)
                gradient_summary.setWordWrap(True)
                gradient_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                gradient_summary.setStyleSheet(
                    "background: palette(alternate-base); border-left: 3px solid palette(highlight); padding: 8px;"
                )
                layout.insertWidget(1, gradient_summary)
                policy_info = QLabel()
                policy_info.setObjectName("target_scaling_info")
                policy_info.setWordWrap(True)
                layout.addWidget(policy_info)

                def show_target_policy(_text=None):
                    candidate = replace(
                        config,
                        target_gradient_normalization=self._value("target_gradient_normalization"),
                        design_context=self._value("design_context"),
                    )
                    method = _target_gradient_normalization_method(candidate, source="expression")
                    multiplier = method == "scale"
                    units = self._value("target_field_units")
                    hz = str(units).lower() == "hz"
                    scale = 1.0 if multiplier or hz else 1000.0
                    widget = self._draft_controls["target_gradient_strength"]
                    if not widget.hasAcceptableInput():
                        policy_info.setText(
                            "Finish or revert the amplitude entry before changing its displayed units. The incomplete value is retained in its previous units."
                        )
                        return
                    if scale != strength_scale["value"]:
                        original = self._draft_initial["target_gradient_strength"] / strength_scale["value"]
                        edited = self._raw_value("target_gradient_strength") / strength_scale["value"]
                        with QSignalBlocker(widget):
                            widget.setRange(
                                min(1e-9 * scale, original * scale, edited * scale),
                                max(1e12 if multiplier or hz else 1e5, original * scale, edited * scale),
                            )
                            widget.setValue(edited * scale)
                        self._draft_initial["target_gradient_strength"] = original * scale
                        self._widget_initial["target_gradient_strength"] = original * scale
                        strength_scale["value"] = scale
                    label = "Amplitude multiplier" if multiplier else "Gradient amplitude"
                    control_form(self, widget, form).labelForField(widget.parentWidget()).setText(label)
                    widget.setAccessibleName(label)
                    widget.setSuffix("" if multiplier else " Hz/m" if hz else " mT/m")
                    widget.setEnabled(method != "none")
                    self._draft_controls["target_gradient_axis"].setEnabled(method == "linear_fit")
                    if method == "none":
                        text = f"Absolute field: the expression supplies Bz in {units}, evaluated with x, y, z in metres. Gradient amplitude and axis are inactive. For example, 1e-4 in T means a constant 100 µT field."
                    elif method == "scale":
                        text = f"Amplitude scaling: the expression supplies Bz in {units} with x, y, z in metres. A dimensionless multiplier scales it before conversion to tesla. Gradient axis is inactive. Use off/none for an absolute field."
                    elif method == "linear_fit":
                        text = "Gradient normalization: x, y, z are world coordinates in metres. The backend rescales Bz to the requested linear-fit gradient magnitude along the selected axis. The expression retains its sign. "
                        text += (
                            "Strength is in Hz/m before conversion to tesla using the proton gyromagnetic ratio."
                            if hz
                            else "For example, y with 20 mT/m gives Bz = 0.02·y T."
                        )
                    else:
                        text = "Legacy extrema scaling uses the target extrema and coordinate span to scale the expression. Inspect the resolved target amplitude in the preview; use linear_fit for gradient examples or off/none for an absolute field."
                    policy_info.setText(text)
                    widget.setToolTip(text)

                def show_gradient_summary(*_args):
                    candidate = replace(
                        config,
                        target_gradient_normalization=self._value("target_gradient_normalization"),
                        design_context=self._value("design_context"),
                    )
                    amplitude = self._draft_controls["target_gradient_strength"]
                    radius = self._draft_controls["target_region_radius"]
                    simple_y = (
                        self._value("field_shape_function").strip() == "y"
                        and self._value("target_gradient_axis") == "y"
                        and self._value("target_field_units") == "T"
                        and self._value("target_field_component") == "Bz"
                        and _target_gradient_normalization_method(candidate, source="expression") == "linear_fit"
                        and config.target_mesh_file == "none"
                        and config.target_region_geometry is None
                        and not config.set_roi_into_mesh_center
                        and amplitude.hasAcceptableInput()
                        and radius.hasAcceptableInput()
                    )
                    gradient_summary.setVisible(simple_y)
                    if not simple_y:
                        return
                    g = amplitude.valueFromText(amplitude.text()) / strength_scale["value"] * 1000
                    r = radius.valueFromText(radius.text())
                    edge_mT = g * r / 1000
                    gradient_summary.setText(
                        f"Y gradient (Gy): dBz/dy = {g:g} mT/m.\n"
                        "Gy specifies the y dependence of the z field component: ΔBz = Gy × y.\n"
                        f"Spherical ROI: {2 * r:g} mm diameter, centered at the world origin.\n"
                        f"At y = −{r:g} / 0 / +{r:g} mm, ΔBz = −{edge_mT:g} / 0 / +{edge_mT:g} mT.\n"
                        "The main B₀ field is separate from this gradient perturbation."
                    )

                for field in ("target_gradient_normalization", "design_context", "target_field_units"):
                    self._draft_controls[field].currentTextChanged.connect(show_target_policy)
                self._draft_controls["target_gradient_strength"].valueChanged.connect(show_target_policy)
                show_target_policy()
                for field in (
                    "field_shape_function",
                    "target_gradient_axis",
                    "target_gradient_normalization",
                    "design_context",
                    "target_field_units",
                    "target_field_component",
                ):
                    control = self._draft_controls[field]
                    signal = control.textChanged if isinstance(control, QLineEdit) else control.currentTextChanged
                    signal.connect(show_gradient_summary)
                for field in ("target_gradient_strength", "target_region_radius"):
                    self._draft_controls[field].valueChanged.connect(show_gradient_summary)
                    self._draft_controls[field].lineEdit().textEdited.connect(show_gradient_summary)
                show_gradient_summary()
            if linear:
                direction = GraphicalChoiceCombo(
                    title="Choose gradient direction",
                    choices={
                        f"G{axis}: dBz/d{axis}": (
                            TARGET_MODE_ICONS[f"G{axis}"],
                            f"G{axis}",
                            f"The {axis} dependence of the z field component: dBz/d{axis}.",
                        )
                        for axis in "xyz"
                    },
                )
                direction.setObjectName("linear_gradient_direction")
                direction.setAccessibleName("Gradient direction")
                for axis in "xyz":
                    direction.addItem(f"G{axis}: dBz/d{axis}", axis)
                direction.setCurrentIndex(direction.findData(str(config.field_shape_function).strip()))
                form.addRow("Gradient direction", direction)

                def change_direction(_index):
                    axis = direction.currentData()
                    self._draft_controls["field_shape_function"].setText(axis)
                    self._mark_draft_dirty("field_shape_function")
                    self._draft_controls["target_gradient_axis"].setCurrentText(axis)

                direction.currentIndexChanged.connect(change_direction)
                advanced_target = QCheckBox("Expert target controls")
                advanced_target.setToolTip(
                    "Show the target expression, component, normalization, units and design context."
                )
                advanced_target.setObjectName("advanced_target_controls")
                advanced_fields = (
                    "field_shape_function",
                    "target_field_component",
                    "target_gradient_normalization",
                    "target_gradient_axis",
                    "target_field_units",
                    "design_context",
                )

                def reveal_target_controls(expanded):
                    candidate = replace(
                        config,
                        target_gradient_normalization=self._value("target_gradient_normalization"),
                        design_context=self._value("design_context"),
                    )
                    method = _target_gradient_normalization_method(candidate, source="expression")
                    for field in advanced_fields:
                        control = self._draft_controls[field]
                        control_form(self, control, form).setRowVisible(
                            control.parentWidget(),
                            expanded and (field != "target_gradient_axis" or method == "linear_fit"),
                        )

                reveal_target_controls(False)
                advanced_target.toggled.connect(reveal_target_controls)
                layout.addWidget(advanced_target)
                note(
                    "Choose the gradient direction and amplitude, then set the ROI. The expression and scaling controls expose the same saved target definition."
                )
        field_builder = self._draft_builder
        self._target_region_editor = TargetRegionEditor(self, config, form, layout, note)
        self._draft_builder = lambda: {**field_builder(), **self._target_region_editor.changes()}
        note(
            "Suppression regions, B₀ direction and coordinate transforms are in Advanced settings. Import an Nx3 field map for Bx, By or vector targets; selecting a component does not move scalar expression values between components."
        )

    def _switch_target_mode(self, mode):
        if not self.apply_settings():
            self._node_changed(self._current_node)
            return
        config = self.project.config
        if mode == "Linear gradient":
            axis = str(config.field_shape_function).strip()
            is_linear = axis in {"x", "y", "z"} and config.target_gradient_normalization in {"auto", "linear_fit"}
            if not is_linear:
                self._expression_targets[self._active_channel_id] = {
                    key: getattr(config, key)
                    for key in (
                        "field_shape_function",
                        "target_gradient_normalization",
                        "target_gradient_strength",
                        "target_gradient_axis",
                        "target_field_units",
                        "target_field_component",
                        "design_context",
                    )
                }
            axis = axis if axis in {"x", "y", "z"} else "y"
            self._expression_editor_fingerprint = None
            self.apply_config(
                {
                    "field_shape_function": axis,
                    "target_gradient_axis": axis,
                    "target_field_definition_file": "none",
                    "target_field_component": "Bz",
                    "target_field_units": "T",
                    "target_gradient_normalization": "linear_fit",
                    "design_context": "mr_gradient",
                    "target_gradient_strength": config.target_gradient_strength
                    if is_linear and config.target_field_units == "T"
                    else 0.02,
                },
                "Use linear gradient target",
            )
        elif mode == "Spherical harmonic":
            previous_recipe = recognize_solid_harmonic_target(config.field_shape_function)
            if not previous_recipe or any(
                getattr(config, key) != value for key, value in previous_recipe.to_config_overrides().items()
            ):
                self._expression_targets[self._active_channel_id] = {
                    key: getattr(config, key)
                    for key in (
                        "field_shape_function",
                        "target_gradient_normalization",
                        "target_gradient_strength",
                        "target_gradient_axis",
                        "target_field_units",
                        "target_field_component",
                        "design_context",
                    )
                }
            recipe = recognize_solid_harmonic_target(config.field_shape_function) or solid_harmonic_target(
                2, 0, 100e-6, config.target_region_radius
            )
            self._expression_editor_fingerprint = None
            self.apply_config(recipe.to_config_overrides(), "Use spherical harmonic target")
        elif mode == "Analytic expression":
            changes = {
                "target_field_definition_file": "none",
                **self._expression_targets.get(self._active_channel_id, {}),
            }
            if str(changes.get("field_shape_function", config.field_shape_function)).strip().lower() in {"", "none"}:
                # An explicit choice creates an editable absolute-field example.
                # Its units and scaling are shown in the target controls.
                changes.update(
                    field_shape_function="1e-4",
                    target_field_units="T",
                    target_field_component="Bz",
                    target_gradient_normalization="none",
                    design_context="mr_shim",
                )
            self._expression_editor_fingerprint = (
                self._active_channel_id,
                changes.get("field_shape_function", config.field_shape_function),
            )
            self.apply_config(changes, "Use analytic expression target")
        self._node_changed(self._current_node)

    def _number(self, form, key, label, value, suffix, minimum, maximum):
        widget = ScientificSpinBox()
        widget.setMinimumWidth(160)
        widget.setDecimals(12)
        widget.setRange(min(minimum, value), max(maximum, value))
        widget.setSuffix(suffix)
        widget.setValue(float(value))
        widget.setKeyboardTracking(False)
        self._register(form, key, label, widget, initial=float(value))

    def _integer(self, form, key, label, value, minimum, maximum):
        widget = QSpinBox()
        widget.setRange(min(minimum, value), max(maximum, value))
        widget.setValue(int(value))
        widget.setKeyboardTracking(False)
        self._register(form, key, label, widget, initial=int(value))

    def _text(self, form, key, label, value):
        self._register(form, key, label, QLineEdit(str(value)))

    def _combo(self, form, key, label, choices, value):
        widget = (
            GraphicalChoiceCombo(title=label, choices=GRAPHICAL_SETTINGS[key])
            if key in GRAPHICAL_SETTINGS
            else QComboBox()
        )
        if not isinstance(widget, GraphicalChoiceCombo):
            self._size_combo(widget)
        if str(value) not in choices:
            choices = [str(value), *choices]
        widget.addItems(choices)
        symbol = {
            "target_kind": "target-map",
            "target_mode": "target-gradient",
            "sf_opt_method": "stream-function",
        }.get(key)
        if symbol:
            for index in range(widget.count()):
                widget.setItemIcon(index, icon(symbol))
        if key == "objective":
            for index in range(widget.count()):
                symbol = {"power": "power-loss", "minimax": "current-density", "magnetic_energy": "inductance"}.get(
                    widget.itemText(index), "stream-function"
                )
                widget.setItemIcon(index, icon(symbol))
        widget.setCurrentText(str(value))
        self._register(form, key, label, widget)

    @staticmethod
    def _size_combo(widget):
        widget.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        widget.setMinimumContentsLength(12)
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _check(self, form, key, label, value):
        widget = QCheckBox()
        symbol = {"sweep": "conductor-sweep", "fabricate": "independent-validation"}.get(key)
        if symbol:
            widget.setIcon(icon(symbol))
        widget.setChecked(bool(value))
        self._register(form, key, label, widget)

    def _register(self, form, key, label, widget, *, initial=None):
        widget.setObjectName(key)
        self._draft_controls[key] = widget
        widget.setProperty("baseDraftStyle", widget.styleSheet())
        self._widget_initial[key] = self._raw_value(key)
        self._draft_initial[key] = self._value(key) if initial is None else initial
        label_widget = QLabel(label)
        label_widget.setWordWrap(True)
        label_widget.setBuddy(widget)
        widget.setAccessibleName(label)
        label_widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        field = QWidget()
        field_layout = QVBoxLayout(field)
        field_layout.setContentsMargins(0, 0, 0, 0)
        field_layout.setSpacing(2)
        field_layout.addWidget(widget)
        error = QLabel()
        error.setObjectName(f"error_{key}")
        error.setWordWrap(True)
        apply_tone(error, "error")
        error.hide()
        field_layout.addWidget(error)
        self._field_errors[key] = error
        form.addRow(label_widget, field)
        signal = (
            widget.toggled
            if isinstance(widget, QCheckBox)
            else widget.currentTextChanged
            if isinstance(widget, QComboBox)
            else widget.textEdited
            if isinstance(widget, QLineEdit)
            else widget.changed
            if isinstance(widget, ConductorProfileEditor)
            else widget.valueChanged
        )
        signal.connect(lambda *_args, _key=key: self._mark_draft_dirty(_key))
        if isinstance(widget, QAbstractSpinBox):

            def mark_numeric_edit(_text, _key=key, _widget=widget):
                # Qt can emit the first textEdited signal before updating its
                # modified flag. A one-character draft must still be tracked.
                _widget.lineEdit().setModified(True)
                self._mark_draft_dirty(_key)

            widget.lineEdit().textEdited.connect(mark_numeric_edit)

    def _mark_draft_dirty(self, key):
        widget = self._draft_controls[key]
        pending_text = isinstance(widget, QAbstractSpinBox) and widget.lineEdit().isModified()
        if pending_text or self._raw_value(key) != self._widget_initial[key]:
            self._changed_controls.add(key)
        else:
            self._changed_controls.discard(key)
        self._draft_dirty = bool(self._changed_controls)
        self._draft_error = ""
        self.draft_error_label.hide()
        for error in self._field_errors.values():
            error.hide()
        self._update_pending()

    def _update_pending(self):
        self.pending_panel.setVisible(self._draft_dirty)
        for key, widget in self._draft_controls.items():
            base = widget.property("baseDraftStyle") or ""
            border = f"border: 1px solid {tone_color('warning', self.palette()).name()}"
            if isinstance(widget, (ConductorProfileEditor, GraphicalChoiceCombo, CircuitSelector)):
                border = f"QWidget#{widget.objectName()} {{ {border}; }}"
            widget.setStyleSheet(base + (border if key in self._changed_controls else ""))
        self._refresh_decision_drafts()
        if self._draft_dirty:
            self.statusBar().showMessage("Pending settings · applied on navigation, save or compute")
        elif self.statusBar().currentMessage().startswith("Pending settings"):
            self.statusBar().clearMessage()
        self._update_title()
        self._refresh_guidance()
        self._refresh_view_context()

    def _acknowledge_inputs(self, action):
        if self.project is not None:
            text = (
                f"Applied revision {self.project.revision}"
                if action == "Applied"
                else f"{action} · applied revision {self.project.revision}"
            )
            self.statusBar().showMessage(text, 7000)

    def revert_draft(self):
        self._draft_dirty = False
        self._node_changed(self._current_node)
        self.statusBar().showMessage("Draft reverted; model revision unchanged", 5000)

    def _show_draft_error(self, message):
        self._draft_error = message
        self.draft_error_label.setText(message)
        self.draft_error_label.show()
        fields = set(re.findall(r"\[ERROR\]\s+(\w+)", message))
        matching = [
            key
            for key in self._changed_controls
            if fields.intersection(self._draft_fields.get(key, (CONTROL_FIELDS.get(key, key),)))
        ]
        failed = matching or sorted(self._changed_controls)[:1]
        for key in failed:
            if key in self._field_errors:
                self._field_errors[key].setText(message)
                self._field_errors[key].show()
                control = self._draft_controls[key]
                border = f"border: 1px solid {tone_color('error', self.palette()).name()}"
                if isinstance(control, (ConductorProfileEditor, GraphicalChoiceCombo, CircuitSelector)):
                    border = f"QWidget#{control.objectName()} {{ {border}; }}"
                control.setStyleSheet((control.property("baseDraftStyle") or "") + border)
        self._refresh_decision_drafts(failed)
        self._focus_draft_error()

    def _refresh_decision_drafts(self, error_keys=()):
        sections = list(self._focused_sections)
        sections.extend(getattr(self._target_region_editor, "focused_sections", ()))
        for section in sections:
            keys = {control.objectName() for control in section.controls}
            section.set_draft_state(
                pending=bool(keys.intersection(self._changed_controls)),
                error=bool(keys.intersection(error_keys)),
            )

    def _show_preflight(self, info):
        self._refresh_computation_indicator()
        if self.project is None:
            return
        self._preflight = info
        if self._target_region_editor is not None:
            self._target_region_editor.set_preflight(info)
        label = getattr(self, "preflight_label", None)
        if label is None:
            return
        if not info:
            label.setText("Preflight: extracting current model geometry and target…")
            return
        geometry, target = info.get("geometry", {}), info.get("target", {})
        lines = ["Current model preflight · world coordinates · preview only"]
        if geometry:
            lines.append(
                f"Support: {geometry.get('vertex_count', '—')} vertices, {geometry.get('triangle_count', '—')} triangles."
            )
            bounds = zip(geometry.get("bounds_min_m") or (), geometry.get("bounds_max_m") or ())
            lines.append(
                "Support bounds [mm]: "
                + ", ".join(f"{axis} {lo * 1e3:.5g}…{hi * 1e3:.5g}" for axis, (lo, hi) in zip("xyz", bounds))
            )
        if target.get("available"):
            from ..target_regions import target_region_summary

            if self.project.study.get("requirements"):
                lines.append(
                    "Named regions: "
                    + ", ".join(region["name"] for region in self.project.study["requirements"]["regions"])
                )
            else:
                lines.append(target_region_summary(self.project.config))
            lines.append(
                f"Target: {target.get('sample_count')} samples; {target.get('active_sample_count')} active, {target.get('suppressed_count')} suppression."
            )
            statistics = [target.get(key) for key in ("Bz_min_T", "Bz_max_T", "Bz_rms_T")]
            if all(isinstance(value, (int, float)) and np.isfinite(value) for value in statistics):
                lines.append(
                    "Bz [µT]: min {:.6g}, max {:.6g}, weighted RMS {:.6g}.".format(
                        *(value * 1e6 for value in statistics)
                    )
                )
            else:
                lines.append("Bz statistics unavailable: no active positive-weight samples.")
            bounds = zip(target.get("bounds_min_m") or (), target.get("bounds_max_m") or ())
            lines.append(
                "Sample bounds [mm]: "
                + "; ".join(f"{axis} {lo * 1e3:.4g}…{hi * 1e3:.4g}" for axis, (lo, hi) in zip("xyz", bounds))
            )
            lines.append(
                f"Normalization: {target.get('normalization_requested', '—')}; resolved scale {target.get('normalization_scale', '—')}. Sampling bounds do not certify mesh convergence."
            )
            recipe = recognize_solid_harmonic_target(self.project.config.field_shape_function)
            if recipe is not None and recipe.degree == 2 and recipe.order == 0:
                pole = recipe.coefficient_T * np.sqrt(5 / (4 * np.pi)) * 1e6
                lines.append(f"Normalized Y20 at r=R: axial pole {pole:+.6g} µT; equator {-pole / 2:+.6g} µT.")
        if info.get("message"):
            lines.append(info["message"])
        label.setText("\n".join(lines))

    def _value(self, key):
        if key not in self._changed_controls and key in self._draft_initial:
            return self._draft_initial[key]
        if isinstance(self._draft_controls[key], ConductorProfileEditor):
            try:
                return self._draft_controls[key].profile()
            except ValueError as exc:
                raise ValueError(f"[ERROR] cross_sectional_points: {exc}") from exc
        return self._raw_value(key)

    def _raw_value(self, key):
        widget = self._draft_controls[key]
        if isinstance(widget, ConductorProfileEditor):
            return widget.draft_value()
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QComboBox):
            return widget.currentText()
        if isinstance(widget, QLineEdit):
            return widget.text()
        return widget.value()

    def apply_settings(self):
        if self.project is None:
            return True
        # Interpret only edited text: the concise display deliberately has
        # fewer digits than a saved model value, so parsing untouched controls
        # would introduce rounding into otherwise unrelated settings edits.
        for key, widget in tuple(self._draft_controls.items()):
            if not isinstance(widget, QAbstractSpinBox):
                continue
            if not widget.isEnabled() and key not in self._changed_controls and not widget.lineEdit().isModified():
                continue
            if not widget.hasAcceptableInput():
                self._changed_controls.add(key)
                self._draft_dirty = True
                self._update_pending()
                self._show_draft_error(
                    f"[ERROR] {CONTROL_FIELDS.get(key, key)}: Enter a complete numeric value in {widget.suffix().strip() or 'the displayed units'}."
                )
                return False
            if widget.lineEdit().isModified():
                widget.interpretText()
                widget.lineEdit().setModified(False)
                self._mark_draft_dirty(key)
        if not self._draft_dirty:
            return True
        self._draft_dirty = False
        self._changed_controls = {
            key for key in self._changed_controls if self._raw_value(key) != self._widget_initial[key]
        }
        if not self._changed_controls:
            self._update_pending()
            return True
        if self._draft_builder is not None:
            try:
                fields = {
                    field
                    for key in self._changed_controls
                    for field in self._draft_fields.get(key, (CONTROL_FIELDS.get(key, key),))
                }
                if "sf_opt_method" in fields:
                    fields.add("research_physics")
                changes = {key: value for key, value in self._draft_builder().items() if key in fields}
                accepted = self.apply_config(changes)
            except (ValueError, TypeError, KeyError) as exc:
                self._show_draft_error(str(exc))
                self._error(f"Model edit rejected: {exc}", open_window=False)
                accepted = False
            self._draft_dirty = not accepted
            if not accepted:
                self.pending_panel.show()
            return accepted
        if "validation_resolution" in self._draft_controls:
            changes = {key: self._value(key) for key in ("validation_resolution", "design_margin", "fabricate")}
            if hasattr(self.project, "update_study"):
                accepted = self.apply_study(changes)
                self._draft_dirty = not accepted
                if not accepted:
                    self.pending_panel.show()
                return accepted
            self.validation_resolution, self.design_margin, self.fabricate = changes.values()
            self.statusBar().showMessage("Independent-validation run controls updated", 5000)
        return True

    def _advanced_dialog(self):
        if self.project is None:
            return
        from .expert_settings import ExpertSettingsDialog

        existing = getattr(self, "expert_window", None)
        if existing is not None:
            existing.show()
            existing.raise_()
            return
        if not self.apply_settings():
            self._focus_draft_error()
            return
        owner, identity = self.project.id, self.project.model_fingerprint

        def apply_expert(changes, label):
            nonlocal identity
            if self._draft_dirty or self.project.id != owner or self.project.model_fingerprint != identity:
                self._error(
                    "The model changed while Advanced settings was open. Its draft is retained. Apply or revert model edits, or discard the expert draft and reload the current model."
                )
                return False
            accepted = self.apply_config(changes, label)
            if accepted:
                identity = self.project.model_fingerprint
                self.expert_window.setWindowTitle(
                    f"{self.project.name} · revision {self.project.revision} — Advanced settings"
                )
                self._expert_model_fingerprint = identity
            return accepted

        self.expert_window = ExpertSettingsDialog(self.project.config, apply_expert, self)
        self._expert_project_id = owner
        self._expert_model_fingerprint = identity
        self.expert_window.draft_changed.connect(self._refresh_input_drafts)
        self.expert_window.setWindowTitle(f"{self.project.name} · revision {self.project.revision} — Advanced settings")
        for buttons in self.expert_window.findChildren(QDialogButtonBox):
            cancel = buttons.button(QDialogButtonBox.StandardButton.Cancel)
            if cancel is not None:
                cancel.setText("Close")
        reload_button = QPushButton("Discard expert draft & reload current model", self.expert_window)
        reload_button.clicked.connect(self._reload_expert)
        self.expert_window.layout().addWidget(reload_button)
        self.expert_window.setModal(False)
        self.expert_window.show()
        self.expert_window.raise_()

    def _dispose_expert_window(self):
        expert = getattr(self, "expert_window", None)
        if expert is not None:
            expert.hide()
            expert.deleteLater()
            self.expert_window = None

    def _discard_project_editors(self):
        if self.assembly is not None:
            for channel in self.assembly.channels:
                self.history_panel.discard_project_notes(channel.project.id)
        self._dispose_expert_window()

    def _reload_expert(self):
        previous = self.expert_window
        previous.hide()
        self.expert_window = None
        previous.deleteLater()
        self._advanced_dialog()
        self._refresh_input_drafts()

    def _check_expert_draft(self):
        expert = getattr(self, "expert_window", None)
        if expert is None or not expert.pending_changes:
            return True
        self.readiness_panel.set_issues(
            (
                ReadinessIssue(
                    title="Apply or discard Advanced settings edits",
                    message="Advanced settings contains unapplied edits. Apply them or discard its draft before computing.",
                    node="optimization",
                    key="expert_draft",
                    action="Open Advanced settings",
                ),
            )
        )
        self.statusBar().showMessage("Apply or discard Advanced settings edits before computing.", 15000)
        self._advanced_dialog()
        return False

    def _has_input_drafts(self):
        expert = getattr(self, "expert_window", None)
        return self._draft_dirty or (expert is not None and expert.pending_changes)

    def _expert_draft_note(self):
        expert = getattr(self, "expert_window", None)
        if expert is None or not expert.pending_changes:
            return ""
        if self.project is None or self.project.id != getattr(self, "_expert_project_id", None):
            return "Advanced edits pending for another channel"
        if self.project.model_fingerprint != getattr(self, "_expert_model_fingerprint", None) or self._draft_dirty:
            return "Advanced edits pending · model conflict"
        return "Advanced edits pending"

    def _refresh_input_drafts(self, *_args):
        self._update_title()
        self._refresh_guidance()

    def _apply_document_inputs(self):
        """Save/export the visible input drafts through their existing validators."""
        if self.project is None:
            return False
        if not self.apply_settings():
            return False
        expert = getattr(self, "expert_window", None)
        if expert is not None and expert.pending_changes and not expert.apply_changes():
            expert.show()
            expert.raise_()
            return False
        self._apply_note_drafts()
        return True

    def _apply_note_drafts(self):
        """Validate notes for every channel before publishing a document snapshot."""
        updated = self.assembly
        for channel in self.assembly.channels:
            state = design_study_state(channel.project)
            if self.history_panel.has_note_drafts(channel.project.id, state):
                hypothesis, decisions = self.history_panel.note_drafts(channel.project.id, state)
                project = set_study_notes_batch(
                    channel.project,
                    hypothesis=hypothesis,
                    decisions=tuple((entry.run, rationale) for entry, rationale in decisions),
                )
                updated = updated.update_channel(channel.id, project=project)
        if updated is not self.assembly:
            self.undo_stack.push(AssemblyEdit(self, updated, self._active_channel_id, "Save pending design decisions"))

    def _refresh_stage_status(self, *, refresh_reuse=True):
        if self.project is None:
            return
        if refresh_reuse:
            threads = getattr(getattr(self.jobs, "policy", None), "threads_per_worker", 1)
            try:
                records = [record for record in self.runs.values() if not self._run_verification_error(record)]
                self._reuse_plan = execution_reuse(
                    self.project,
                    records,
                    (state for id_, state in self.states.items() if not self._run_verification_error(id_)),
                    jobs_directory=self.jobs_directory,
                    threads=threads,
                )
            except (OSError, ValueError, TypeError, KeyError) as exc:
                self._reuse_plan = ReusePlan((), (), (str(exc),))
            self._reuse_plan_fingerprint = self.project.fingerprint
        available = {entry.stage.value for entry in self._reuse_plan.available}
        active = [state for state in self.states.values() if self._live_owner_matches(state)]
        live = self.live_preview.state
        if not self._live_owner_matches(live) or live.get("id") in self.runs:
            live = {}
        completed = tuple(live.get("completed_stages", ()))
        if live and self.live_preview.preview is not None:
            completed += tuple(self.live_preview.preview.metadata.get("completed_stages", ()))
        for stage, item in self.stage_items.items():
            status = "Cached" if stage in available else "Stale" if stage in self._reuse_plan.stale else "Not computed"
            tooltip = "Compute through this boundary; source, inputs and dependencies are verified before reuse."
            candidate = self._reuse_plan.through(stage)
            if candidate:
                tooltip += f"\nReusable {candidate.stage.value}: {candidate.path}"
            # A previous execution failure cannot erase a verified compatible
            # checkpoint. Current active execution is shown independently.
            stage_jobs = sorted(
                (state for state in active if state.get("pipeline_stage") == stage),
                key=lambda state: format_job_timestamp(state.get("created_at")).timestamp or float("-inf"),
            )
            for state in stage_jobs:
                if state.get("pipeline_stage") == stage and state.get("status") in {
                    "queued",
                    "running",
                    "cancelling",
                    "failed",
                    "interrupted",
                }:
                    if stage in available and state.get("status") in {"failed", "interrupted"}:
                        continue
                    status = {
                        "queued": "Queued",
                        "running": "Running to",
                        "cancelling": "Cancelling",
                        "failed": "Execution failed",
                        "interrupted": "Interrupted",
                    }[state["status"]]
                    tooltip += f"\n{state.get('stage', '')}"
            if stage in completed:
                status = "Computed"
                tooltip += (
                    "\nRecorded completed checkpoint of the submitted revision. "
                    "Live plot availability, final result integrity and engineering acceptance are separate."
                )
            if stage == "evaluated" and status == "Cached":
                matching = [
                    record
                    for record in self.runs.values()
                    if record.metadata.get("project_fingerprint") == self.project.fingerprint
                ]
                if matching:
                    metadata = matching[-1].metadata
                    verdict = metadata.get("summary", {}).get("accepted")
                    checks = run_verdicts(metadata)
                    status = (
                        "Checks failed"
                        if verdict is False or "Failed" in checks.values()
                        else "Complete"
                        if verdict is True
                        else "Unverified"
                    )
                    tooltip += f"\nSheet: {checks['sheet']}; pipeline sweeps: {checks['sweeps']}; fabrication review: {checks['fabrication']}."
                else:
                    status = "Review needed"
            item.setText(1, status)
            badge = (
                None
                if status == "Computed"
                else "failed"
                if status in {"Execution failed", "Checks failed"}
                else "stale"
                if status == "Stale"
                else "pending"
                if status in {"Queued", "Running to", "Cancelling"}
                else "evaluated"
                if status in {"Cached", "Complete"}
                else "unverified"
            )
            item.setIcon(0, icon(STAGE_ICONS[stage], badge=badge))
            item.setForeground(
                1,
                item_foreground(
                    "error"
                    if status in {"Execution failed", "Checks failed"}
                    else "warning"
                    if status in {"Stale", "Interrupted", "Review needed"}
                    else "info"
                    if status in {"Cached", "Complete", "Computed", "Queued", "Running to", "Cancelling"}
                    else "neutral",
                    self.palette(),
                ),
            )
            item.setToolTip(0, tooltip)
            item.setToolTip(1, tooltip)
        if refresh_reuse and self._reuse_plan.diagnostics:
            self.recompute.appendPlainText("\nUnavailable cached evidence:\n" + "\n".join(self._reuse_plan.diagnostics))
        current = self._current_node.data(0, Qt.ItemDataRole.UserRole) if self._current_node else ""
        if refresh_reuse and current and current.startswith("stage:") and not self._draft_dirty:
            self._node_changed(self._current_node)
        self._refresh_design_progress()

    def compute_through(self, stage):
        if not self.apply_settings():
            self._report_compute_draft_error()
            return None
        self.stage_selector.setCurrentIndex(self.stage_selector.findData(stage))
        if self.stage_selector.currentData() != self.project.study["pipeline_stage"]:
            return None
        return self.compute()

    def _save_session(self):
        if self._history_key is None:
            return
        view = self._workbench_view_state()
        self._panel_views[self._history_key] = view
        if self.history is None:
            return
        selected = next(
            (entry.id for entry in self._history_entries if self._history_ids.get(entry.id) == self._selected_run), None
        )
        try:
            self.history.save_session(
                self._history_key[0],
                channel_id=self._history_key[1],
                selected_id=selected,
                view_state=view,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            self.messages.appendPlainText(f"Session could not be saved: {exc}")

    def _restore_history(self, key):
        panel_view = {}
        self._restore_panel_layout(panel_view)
        self._history_key = key
        self._history_entries = ()
        self._selected_run = None
        self.checks_table.setRowCount(0)
        self.wire_fields_table.setRowCount(0)
        self.messages.clear()
        if self._history_error:
            self.messages.appendPlainText(self._history_error)
        if hasattr(self.viewport, "clear_run"):
            self.viewport.clear_run()
        self.problems_panel.set_run(None)
        embedded_ids = {}
        embedded_entries = []
        diagnostics = []
        for archive in self.project.saved_runs:
            try:
                record = archive.materialize(self.jobs_directory.parent / "embedded-runs")
                self.runs[archive.id] = record
                embedded_ids[archive.id] = archive.id
                entry = HistoryEntry(
                    archive.id,
                    key[0],
                    key[1],
                    archive.alias,
                    record.directory,
                    record.metadata["request_sha256"],
                    record,
                )
                if self.history is not None:
                    try:
                        entry = self.history.add_run(key[0], record, channel_id=key[1], alias=archive.alias)
                        self.history.rename_run(entry.id, archive.alias)
                        embedded_ids[archive.id] = entry.id
                    except (OSError, ValueError, TypeError, RuntimeError) as exc:
                        diagnostics.append(f"Local history cache unavailable; embedded evidence is restored: {exc}")
                self._history_ids[entry.id] = embedded_ids[archive.id]
                embedded_entries.append(entry)
            except (OSError, ValueError, TypeError, RuntimeError) as exc:
                message = f"Embedded run {archive.alias!r} could not be restored: {exc}"
                diagnostics.append(message)
                embedded_entries.append(
                    HistoryEntry(
                        archive.id,
                        key[0],
                        key[1],
                        archive.alias,
                        self.jobs_directory.parent / "embedded-runs" / archive.id,
                        archive.metadata["request_sha256"],
                        diagnostic=message,
                    )
                )
        if self.history is not None:
            try:
                self._history_entries = self.history.restore(key[0], channel_id=key[1])
                verified = {entry.id: entry for entry in self._history_entries if entry.verified}
                unavailable = {entry.directory for entry in self._history_entries if not entry.verified}
                for id_, record in list(self.runs.items()):
                    if record.directory in unavailable:
                        self.runs.pop(id_)
                for entry in self._history_entries:
                    if entry.verified:
                        self.runs[entry.id] = entry.run
                        self._history_ids[entry.id] = entry.id
                session = self.history.session(key[0], channel_id=key[1])
                selected = session.get("selected_id")
                if selected in verified:
                    self.show_run(self._history_ids[selected])
                view = session.get("view_state", {})
                view = view if isinstance(view, dict) else {}
                if hasattr(self.viewport, "restore_view_state"):
                    self.viewport.restore_view_state(view)
                self.select_node(view.get("model_node", "geometry"))
                panel_view = view
            except (OSError, ValueError, TypeError, RuntimeError) as exc:
                diagnostics.append(f"History could not be restored: {exc}")
        present = {entry.id for entry in self._history_entries}
        self._history_entries += tuple(entry for entry in embedded_entries if entry.id not in present)
        if self.project.saved_runs or self.project.session:
            session = self.project.session
            selected = session.get(
                "selected_run_id", self.project.saved_runs[-1].id if self.project.saved_runs else None
            )
            selected_id = embedded_ids.get(selected)
            if selected_id in self.runs:
                self.show_run(selected_id)
            if "view_state" in session:
                view = session["view_state"]
                view = view if isinstance(view, dict) else {}
                if hasattr(self.viewport, "restore_view_state"):
                    self.viewport.restore_view_state(view)
                self.select_node(view.get("model_node", "results" if self.project.saved_runs else "geometry"))
                panel_view = view
            elif selected_id in self.runs:
                self.select_node("results")
        self._restore_diagnostics[key] = tuple(diagnostics)
        self._refresh_history_panel()
        self._fill_runs()
        self._update_result_state()
        self._node_changed(self._current_node)
        if key in self._panel_views:
            panel_view = self._panel_views[key]
            if hasattr(self.viewport, "restore_view_state"):
                self.viewport.restore_view_state(panel_view)
            self.select_node(panel_view.get("model_node", "geometry"))
        self._restore_panel_layout(panel_view)
        for message in diagnostics:
            self.messages.appendPlainText(message)

    def _refresh_history_panel(self):
        diagnostics = self._restore_diagnostics.get(self._history_key, ())
        if self._history_error:
            diagnostics = (self._history_error, *diagnostics)
        try:
            study = design_study_state(self.project)
        except (ValueError, TypeError, KeyError) as exc:
            diagnostics = (*diagnostics, f"Design decisions unavailable: {exc}")
            study = DesignStudyState()
        self.history_panel.set_study_state(study, project_id=self.project.id)
        entries = tuple(
            replace(entry, diagnostic=self._run_verification_error(entry.run) or entry.diagnostic)
            if entry.run is not None
            else entry
            for entry in self._history_entries
        )
        self.history_panel.set_records(entries, diagnostics, project=self.project)
        self._refresh_view_context()
        self._refresh_channel_overview()

    def _apply_design_document(self, project, label):
        before = self.project
        if project.id != before.id:
            raise ValueError("A design variant must retain the active project owner.")
        # Compare detached session/state without encoding potentially large
        # portable array archives into base64 on the UI thread.
        if (project.revision, project.fingerprint, project.session, project.saved_runs) != (
            before.revision,
            before.fingerprint,
            before.session,
            before.saved_runs,
        ):
            self.undo_stack.push(ProjectEdit(self, before, project, label))
        self._refresh_history_panel()
        return project

    def pin_design_baseline(self, record=None):
        """Synchronous application API; GUI buttons verify in a background task."""
        record = record or self.runs.get(self._selected_run)
        if self._run_verification_error(record):
            raise ValueError("Saved evidence failed verification; reopen and verify it before pinning a baseline.")
        if not self.apply_settings():
            raise ValueError("Correct pending settings before pinning a baseline.")
        if record is None:
            raise ValueError("Select a verified owned run to pin as the baseline.")
        return self._apply_design_document(pin_baseline(self.project, record), "Pin design baseline")

    def clear_design_baseline(self):
        if self._design_study_task is not None:
            return
        if not self.apply_settings():
            return
        try:
            self._apply_design_document(pin_baseline(self.project, None), "Unpin design baseline")
        except (ValueError, TypeError, OSError) as exc:
            self._error(f"Cannot unpin baseline: {exc}")

    def create_design_variant(self, record=None, *, hypothesis=""):
        """Restore one owned run as an undoable revision, preserving its evidence."""
        record = record or self.runs.get(self._selected_run)
        if self._run_verification_error(record):
            raise ValueError("Saved evidence failed verification; reopen and verify it before creating a variant.")
        if not self.apply_settings():
            raise ValueError("Correct pending settings before creating a variant.")
        if record is None:
            raise ValueError("Select a verified owned run to create a variant.")
        return self._apply_design_document(
            create_variant(self.project, record, hypothesis=hypothesis), "Create controlled design variant"
        )

    def _start_design_task(self, operation, label, record=None, **kwargs):
        if self._design_study_task is not None:
            self.statusBar().showMessage("A design action is already being verified", 5000)
            return
        if operation in {pin_baseline, create_variant} and self._run_verification_error(record):
            self._error("Saved evidence failed verification; reopen and verify it before using this design action.")
            return
        if not self.apply_settings():
            self._report_compute_draft_error()
            return
        task = (
            DesignStudyTask(operation, self.project, record.directory if record is not None else None, **kwargs)
            if operation in {pin_baseline, create_variant}
            else DesignStudyTask(
                operation, self.project, record=record.directory if record is not None else None, **kwargs
            )
        )
        task.signals.finished.connect(self._design_task_finished)
        task.signals.progress.connect(self._background_work_progress)
        self._design_study_task = task
        self._design_study_task_label = label
        self.history_panel.setEnabled(False)
        self.statusBar().showMessage(f"{label}: verifying saved inputs and evidence…")
        QThreadPool.globalInstance().start(task)

    def _pin_history_baseline(self, entry):
        self._start_design_task(pin_baseline, "Pin design baseline", entry.run)

    def _create_history_variant(self, entry, hypothesis):
        self._start_design_task(create_variant, "Create controlled design variant", entry.run, hypothesis=hypothesis)

    def _save_design_notes(self, hypothesis, rationale):
        entry = self.history_panel.owned_selection
        record = entry.run if entry is not None else None
        self._start_design_task(
            set_study_notes, "Save design decisions", record, hypothesis=hypothesis, rationale=rationale
        )

    def _design_task_finished(self, project, error):
        task = self._design_study_task
        getattr(self, "_local_progress", {}).pop(getattr(task, "signals", None), None)
        label = self._design_study_task_label
        self._design_study_task = None
        self._design_study_task_label = ""
        self.history_panel.setEnabled(True)
        if error:
            self._error(f"{label} failed: {error}")
        elif task is None or task.project is not self.project or self._draft_dirty:
            self.messages.appendPlainText(
                f"{label}: model or draft changed during verification; the returned document was not applied. Repeat the action on the intended revision."
            )
            self.statusBar().showMessage("Design action needs repeating on the current revision", 15000)
        else:
            self._apply_design_document(project, label)
            self.statusBar().showMessage(f"{label} saved in this project", 7000)
            if label == "Create controlled design variant":
                self.select_node("winding")
                self.evidence.setCurrentWidget(self.history_panel)
                self._size_review_evidence()

    def _acceptance_action(self, key):
        if key == "problems":
            self.evidence.setCurrentWidget(self.problems_panel)
            self._show_dock("evidence_dock")
        else:
            record = self.acceptance_panel._record
            if self._run_verification_error(record):
                self._error("Saved evidence failed verification. Reopen and verify its source before using this check.")
                return
            owner = (
                record.metadata.get("project_id") or (record.metadata.get("job") or {}).get("project_id")
                if record is not None
                else None
            )
            if record is not None and owner != self.project.id:
                self._error(
                    "This check belongs to another project. Inspect its saved settings or open its owning project."
                )
                return
            self.select_node(key)
            self._show_dock("settings_dock")

    def _inspect_saved_settings(self, record, node):
        if self._saved_settings_task is not None:
            self.statusBar().showMessage("Verifying saved settings…", 5000)
            return
        if self._run_verification_error(record):
            self._error("Saved settings inspection is unavailable while this run fails integrity verification.")
            return
        task = SavedSettingsTask(record, self.project, node)
        self._saved_settings_task = task
        task.signals.finished.connect(self._saved_settings_finished)
        self.statusBar().showMessage("Verifying original run settings…")
        QThreadPool.globalInstance().start(task)

    def _saved_settings_finished(self, request, error):
        task, self._saved_settings_task = self._saved_settings_task, None
        self._refresh_computation_indicator()
        if error:
            if task is not None and task.verification_error:
                source = next(
                    (id_ for id_, record in self.runs.items() if record is task.record), str(task.record.directory)
                )
                self._mark_run_verification_error(source, error, directory=task.record.directory)
                if self.project is not None:
                    self._refresh_history_panel()
                self._refresh_stage_status()
                self._update_result_state()
            self._error("Saved settings inspection failed: " + error)
        elif task is not None and self.project is not None:
            dialog = SavedSettingsDialog(request, task.project, task.node, self)
            dialog.show()
            self.statusBar().showMessage("Verified saved settings opened read-only", 5000)

    def _acceptance_witness(self, issue):
        self._locate_problem(issue)
        if issue in self.problems_panel.issues:
            self.problems_panel.table.selectRow(self.problems_panel.issues.index(issue))

    def _history_selected(self, entry):
        if entry.verified:
            id_ = self._history_ids.setdefault(entry.id, entry.id)
            self.runs[id_] = entry.run
            self.show_run(id_)

    def _rename_history(self, id_, alias):
        if self.history is None:
            return
        try:
            self.history.rename_run(id_, alias)
            self._history_entries = self.history.restore(self._history_key[0], channel_id=self._history_key[1])
            self._refresh_history_panel()
            self.history_panel.select_record(id_)
        except (OSError, ValueError, TypeError) as exc:
            self._error(f"Cannot rename run display: {exc}")

    def _compare_runs(self, left, right):
        if self._run_verification_error(left) or self._run_verification_error(right):
            self._error("Comparison is unavailable while a saved source fails integrity verification.")
            return
        if hasattr(self.viewport, "display_comparison"):
            try:
                self.viewport.display_comparison(left, right)
            except (OSError, ValueError, TypeError, RuntimeError) as exc:
                self._error(f"Comparison view unavailable: {exc}")

    def _locate_problem(self, issue):
        if hasattr(self.viewport, "locate_issue"):
            self.viewport.locate_issue(issue)

    def save_project(self, path):
        if self.project is None:
            raise ValueError("Create or open a project before saving.")
        if not self._apply_document_inputs():
            raise ValueError(
                "Correct pending settings, or apply/discard the retained Advanced settings draft before saving."
            )
        portable = self._portable_assembly()
        self.project_path = Path(
            portable.save(path)
            if self._collection_mode or len(self.assembly.channels) > 1
            else portable.channel(self._active_channel_id).project.save(path)
        )
        self.assembly = portable
        self.project = portable.channel(self._active_channel_id).project
        self.undo_stack.setClean()
        self._save_session()
        self.statusBar().showMessage(f"Saved {self.project_path}", 5000)
        return self.project_path

    def _portable_assembly(self):
        """Embed all available completed evidence, retaining per-channel ownership."""
        self._save_session()
        channels = []
        for channel in self.assembly.channels:
            project = channel.project
            scope = channel.id if self._collection_mode else None
            try:
                entries = self.history.restore(project.id, channel_id=scope) if self.history is not None else ()
            except (OSError, ValueError, TypeError, RuntimeError) as exc:
                entries = ()
                self.messages.appendPlainText(f"Local history cache unavailable; saving owned project evidence: {exc}")
            aliases = {entry.directory.resolve(): entry.alias for entry in entries}
            records = {
                entry.run.directory.resolve(): entry.run
                for entry in entries
                if entry.verified and not self._run_verification_error(entry.run)
            }
            records.update(
                {
                    record.directory.resolve(): record
                    for id_, record in self.runs.items()
                    if not self._run_verification_error(record)
                    and (
                        record.metadata.get("project_id") == project.id
                        or (
                            not record.metadata.get("project_id")
                            and project.id == self.project.id
                            and id_ == self._selected_run
                        )
                    )
                }
            )
            saved = {archive.id: archive for archive in project.saved_runs}
            paths = {}
            for directory, record in records.items():
                metadata = {key: value for key, value in record.metadata.items() if key != "job"}
                previous = next(
                    (
                        archive
                        for archive in saved.values()
                        if archive.metadata.get("request_sha256") == metadata.get("request_sha256")
                        and result_metadata_digest(archive.metadata) == result_metadata_digest(metadata)
                    ),
                    None,
                )
                archive = (
                    replace(previous, alias=aliases.get(directory, previous.alias))
                    if previous is not None
                    else RunArchive.capture(record, alias=aliases.get(directory))
                )
                saved[archive.id] = archive
                paths[directory] = archive.id
            session = project.session
            if project.id == self.project.id:
                selected = self.runs.get(self._selected_run)
                session = {
                    **session,
                    "selected_run_id": paths.get(selected.directory.resolve()) if selected else None,
                    "view_state": self._workbench_view_state(),
                }
            elif self.history is not None:
                try:
                    state = self.history.session(project.id, channel_id=scope)
                except (OSError, ValueError, TypeError, RuntimeError):
                    state = session
                selected = next((entry for entry in entries if entry.id == state.get("selected_id")), None)
                session = {
                    **session,
                    "selected_run_id": paths.get(selected.directory.resolve()) if selected else None,
                    "view_state": state.get("view_state", {}),
                }
            if project.id != self.project.id and (project.id, scope) in self._panel_views:
                session = {**session, "view_state": self._panel_views[(project.id, scope)]}
            if any(not entry.verified for entry in entries):
                session["unavailable_runs"] = [
                    {"alias": entry.alias, "diagnostic": entry.diagnostic} for entry in entries if not entry.verified
                ]
            session["active_channel_id"] = self._active_channel_id if self._collection_mode else None
            channels.append(replace(channel, project=project.with_results(tuple(saved.values()), session=session)))
        return replace(self.assembly, channels=tuple(channels))

    def open_project(self, path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Project file must contain a JSON object.")
        schema = value.get("schema")
        assembly = ProjectAssembly.load(path) if schema in {ASSEMBLY_SCHEMA, *LEGACY_ASSEMBLY_SCHEMAS} else None
        if assembly is not None and not assembly.channels:
            raise ValueError(
                "This collection has no channels. Add a channel in Python before opening it in the workbench."
            )
        project = None if assembly is not None else ProjectDocument.load(path)
        # The file is valid before discarding the previous owner's editor.
        # A discarded expert draft must not block the newly opened project.
        self._discard_project_editors()
        self._draft_dirty = False
        self.undo_stack.clear()
        self.project_path = Path(path)
        self._history_key = None
        self._panel_views.clear()
        self._collection_mode = assembly is not None
        self._set_assembly(assembly or ProjectAssembly.new(project.name, (project,)))
        self.undo_stack.setClean()
        self.statusBar().showMessage(f"Opened {self.project.name} · revision {self.project.revision}", 7000)
        return assembly or project

    def new_project(self, kind=None, *, name="New coil design", project=None, discard_checked=False):
        """Create an unconfigured document; explicit legacy kinds load templates."""
        from .project_start import blank_project

        if kind is not None:
            return self.load_template(kind)
        if not discard_checked and not self._can_discard():
            return None
        document = project if project is not None else blank_project(name)
        self._clear_project()
        self._set_assembly(ProjectAssembly.new(document.name, (document,)))
        self.undo_stack.setClean()
        self._inspect_node("geometry")
        self.statusBar().showMessage("New project · choose a support and target field", 5000)
        return document

    def load_template(self, kind="cylinder"):
        """Load configured inputs ready to compute; settings remain optional."""
        template = template_spec(kind)
        if not self._can_discard():
            return
        try:
            assembly = template_assembly(kind)
        except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
            self._error(f"Cannot load template: {exc}")
            return
        self._discard_project_editors()
        self.undo_stack.clear()
        self._draft_dirty = False
        self.project_path = None
        self._history_key = None
        self._panel_views.clear()
        self._collection_mode = len(assembly.channels) > 1
        self._set_assembly(assembly)
        self.messages.appendPlainText(template.description)
        self.select_node(f"stage:{self.project.study['pipeline_stage']}")
        self.findChild(QDockWidget, "settings_dock").hide()
        self.undo_stack.setClean()
        self.statusBar().showMessage(
            f"Loaded template: {template.title} · Settings loaded. Click Compute full design; editing is optional.",
            10000,
        )
        return assembly

    def export_python(self, path):
        """Export a script plus owned project assets, avoiding duplicated inputs."""
        if not self._apply_document_inputs():
            raise ValueError("Correct the pending settings before exporting.")
        return self._portable_assembly().channel(self._active_channel_id).project.export_python(path)

    def export_assembly_python(self, path):
        if not self._apply_document_inputs():
            raise ValueError("Correct pending settings before exporting.")
        path = Path(path)
        project_path = path.with_suffix(".coilgen-assembly.json")
        self._portable_assembly().save(project_path)
        source = (
            '"""Independent CoilGen2 channels; coupling and assembly checks require separate evaluation."""\n'
            f"# CoilGen2 software: {AUTHOR_CREDIT} ({AUTHOR_ROLE}).\n"
            f"# Project: {PROJECT_URL}\n"
            "from pathlib import Path\n"
            "from coilgen.application import DesignService, ProjectAssembly\n\n"
            "if __name__ == '__main__':\n"
            f"    project = ProjectAssembly.load(Path(__file__).with_name({project_path.name!r}))\n"
            "    results = DesignService.run_assembly(project, Path(__file__).with_suffix('') / 'runs')\n"
            "    print(results)\n"
        )
        atomic_text(path, source)
        return path

    def _can_discard(self):
        if self.project is None:
            return True
        if (
            self.undo_stack.isClean()
            and not self._draft_dirty
            and not any(
                self.history_panel.has_note_drafts(channel.project.id, design_study_state(channel.project))
                for channel in self.assembly.channels
            )
            and not (getattr(self, "expert_window", None) is not None and self.expert_window.pending_changes)
        ):
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved project changes",
            "Save this project before continuing?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        return self._save_dialog() if answer == QMessageBox.StandardButton.Save else True

    def _open_dialog(self):
        if not self._can_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open CoilGen project", "", "CoilGen project (*.json)")
        if path:
            try:
                self.open_project(path)
            except (ValueError, TypeError, OSError, KeyError) as exc:
                self._error(f"Cannot open project: {exc}")

    def _save_dialog(self, save_as=False):
        if self.project is None:
            self.statusBar().showMessage("Create or open a project before saving.", 5000)
            return False
        path = None if save_as else self.project_path
        if path is None:
            name, _ = QFileDialog.getSaveFileName(
                self, "Save CoilGen project", f"{self.project.name}.coilgen.json", "CoilGen project (*.json)"
            )
            if not name:
                return False
            path = Path(name)
        try:
            self.save_project(path)
            return True
        except (ValueError, OSError, TypeError) as exc:
            self._error(f"Cannot save project: {exc}")
            return False

    def _export_dialog(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export reproducible script", "coilgen_design.py", "Python script (*.py)"
        )
        if path:
            try:
                self.export_python(path)
                self.statusBar().showMessage(f"Exported {path} and owned project inputs", 5000)
            except (ValueError, TypeError, OSError) as exc:
                self._error(f"Cannot export script: {exc}")

    def _export_assembly_dialog(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export independent channels", "coilgen_channels.py", "Python script (*.py)"
        )
        if path:
            try:
                self.export_assembly_python(path)
                self.statusBar().showMessage("Exported independent channels and owned project inputs", 5000)
            except (ValueError, TypeError, OSError) as exc:
                self._error(f"Cannot export channels: {exc}")

    def _open_run_dialog(self):
        path = QFileDialog.getExistingDirectory(self, "Open completed run directory")
        if path:
            try:
                self.load_run(path)
            except (ValueError, TypeError, OSError, KeyError) as exc:
                self._error(f"Cannot open run: {exc}")

    def _rename_dialog(self):
        name, accepted = QInputDialog.getText(self, "Rename project", "Scientific project name", text=self.project.name)
        if accepted and self.apply_settings():
            try:
                project = self.project.rename(name)
                if project is not self.project:
                    after = self.assembly.update_channel(self._active_channel_id, name=name, project=project)
                    self.undo_stack.push(AssemblyEdit(self, after, self._active_channel_id, "Rename channel"))
            except (ValueError, TypeError) as exc:
                self._error(f"Cannot rename project: {exc}")

    def import_mesh(self, path, *, length_scale_to_m=1.0):
        """Use the shared support loader, then capture the validated input bytes."""
        from ..geometry import load_mesh_from_config

        if self.project is None:
            raise ValueError("Create or open a project before importing a support.")
        path = Path(path).expanduser().resolve()
        if path.suffix.lower() not in {".stl", ".npz"}:
            raise ValueError("Support meshes must be STL or CoilGen NPZ; coordinates must have explicit units.")
        if not self.apply_settings():
            raise ValueError("Correct pending settings before importing.")
        changes = {
            "coil_mesh_file": str(path),
            "imported_mesh_length_scale_to_m": length_scale_to_m,
            "surface_is_cylinder_flag": False,
        }
        candidate = self.project.update_config(changes, allow_incomplete=True)
        load_mesh_from_config(candidate.config, candidate.project_root)
        self.undo_stack.push(ProjectEdit(self, self.project, candidate, "Import support mesh"))
        self.select_node("geometry")
        return candidate

    def _import_mesh_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import support mesh", "", "Support mesh (*.stl *.npz)")
        if not path:
            return
        scale = 1.0
        if Path(path).suffix.lower() == ".stl":
            unit, accepted = QInputDialog.getItem(
                self, "STL coordinate units", "Source coordinate unit", ["mm", "m", "cm", "µm"], 0, False
            )
            if not accepted:
                return
            scale = {"mm": 1e-3, "m": 1.0, "cm": 1e-2, "µm": 1e-6}[unit]
        try:
            self.import_mesh(path, length_scale_to_m=scale)
        except (ValueError, TypeError, OSError, KeyError, RuntimeError) as exc:
            self._error(f"Support import rejected: {exc}")

    def import_region_mesh(self, path, *, length_scale_to_m=1.0, sampling="surface"):
        """Own an SI target surface independently of the winding-support units."""
        from ..geometry import read_mesh_npz, read_stl
        from ..models import Mesh
        from ..target_regions import validate_target_region_mesh

        if self.project is None:
            raise ValueError("Create or open a project before importing a target surface.")
        path = Path(path).expanduser().resolve()
        if path.suffix.lower() not in {".stl", ".npz"}:
            raise ValueError("Target surfaces must be STL or CoilGen SI mesh NPZ.")
        if isinstance(length_scale_to_m, bool) or not np.isfinite(length_scale_to_m) or length_scale_to_m <= 0:
            raise ValueError("Source coordinate scale must be a finite positive number of metres per source unit.")
        if path.suffix.lower() == ".npz" and length_scale_to_m != 1.0:
            raise ValueError("CoilGen mesh NPZ coordinates are already metres; use source scale 1.")
        if not self.apply_settings():
            raise ValueError("Correct pending settings before importing a target surface.")
        if self.project.config.target_field_definition_file != "none":
            raise ValueError(
                "Choose an analytic target before importing a generated field region; map coordinates are authoritative."
            )
        source = read_mesh_npz(path) if path.suffix.lower() == ".npz" else read_stl(path)
        mesh = Mesh(
            vertices=np.array(source.vertices, copy=True) * length_scale_to_m,
            faces=np.array(source.faces, copy=True),
            metadata={**source.metadata, "coordinate_units": "m", "coordinate_frame": "world"},
        )
        region = {"shape": "enclosing_mesh", "sampling": sampling}
        mesh = validate_target_region_mesh(mesh, sampling)
        candidate = self.project.with_mesh(
            mesh,
            field="target_mesh_file",
            changes={"target_region_geometry": region, "set_roi_into_mesh_center": False},
        )
        self.undo_stack.push(ProjectEdit(self, self.project, candidate, "Import target region surface"))
        self.select_node("target")
        return candidate

    def _import_region_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import target region surface", "", "Target surface (*.stl *.npz)")
        if not path:
            return
        scale = 1.0
        if Path(path).suffix.lower() == ".stl":
            unit, accepted = QInputDialog.getItem(
                self, "Target STL coordinate units", "Source coordinate unit", ["mm", "m", "cm", "µm"], 0, False
            )
            if not accepted:
                return
            scale = {"mm": 1e-3, "m": 1.0, "cm": 1e-2, "µm": 1e-6}[unit]
        choice, accepted = QInputDialog.getItem(
            self,
            "Target sampling",
            "Constrain the interior or the imported surface?",
            ["Volume interior (closed enclosure required)", "Surface only (open surface allowed)"],
            0,
            False,
        )
        if not accepted:
            return
        try:
            self.import_region_mesh(
                path, length_scale_to_m=scale, sampling="volume" if choice.startswith("Volume") else "surface"
            )
        except (ValueError, TypeError, OSError, KeyError, RuntimeError) as exc:
            self._error(f"Target region import rejected: {exc}")

    def import_target(self, path, *, field_units="T"):
        """Keep supplied field amplitudes; normalized maps require explicit edits."""
        from ..target import load_target_field_definition

        if self.project is None:
            raise ValueError("Create or open a project before importing a target field.")
        path = Path(path).expanduser().resolve()
        if not self.apply_settings():
            raise ValueError("Correct pending settings before importing.")
        changes = {
            "target_field_definition_file": str(path),
            "target_field_units": field_units,
            "target_gradient_normalization": "none",
        }
        candidate = self.project.update_config(changes, allow_incomplete=True)
        target = load_target_field_definition(candidate.config.target_field_definition_file, None, candidate.config)
        for name in ("coords", "b", "weights", "target_gradient_dbdxyz"):
            values = getattr(target, name, None)
            if values is not None and not np.all(np.isfinite(values)):
                raise ValueError(f"Imported target {name} contains nonfinite values.")
        self.undo_stack.push(ProjectEdit(self, self.project, candidate, "Import target map"))
        self.select_node("target")
        return candidate

    def _import_target_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import target map — coordinates in metres", "", "Field map (*.npz *.csv *.tsv *.json)"
        )
        if not path:
            return
        unit, accepted = QInputDialog.getItem(
            self,
            "Target field units",
            "Field values (coordinate columns remain metres)",
            ["T", "Hz"],
            0,
            False,
        )
        if accepted:
            try:
                self.import_target(path, field_units=unit)
            except (ValueError, TypeError, OSError, KeyError, RuntimeError, NotImplementedError) as exc:
                self._error(f"Target import rejected: {exc}")

    def export_run_artifacts(self, directory):
        """Copy only verified outputs; never recompute another representation."""
        if self._selected_run is None:
            raise ValueError("Select a completed run before exporting artifacts.")
        if self._run_verification_error(self._selected_run):
            raise ValueError("Saved evidence failed verification; reopen and verify it before exporting artifacts.")
        source = self.runs[self._selected_run]
        try:
            return export_verified_artifacts(source, directory, loader=self.service.load_run)
        except ArtifactVerificationError as exc:
            self._mark_run_verification_error(self._selected_run, exc, directory=source.directory)
            self._update_result_state()
            raise

    def _export_run_dialog(self):
        if self._artifact_export_task is not None:
            self.statusBar().showMessage("Run artifact export is already running", 5000)
            return
        source_id = self._selected_run
        record = self.runs.get(source_id)
        if record is None:
            self.statusBar().showMessage("Select a completed run before exporting artifacts.", 5000)
            return
        source_name = record.metadata.get("name", record.directory.name)
        receipt = ExportReceiptPanel(retry=True)
        window = PanelWindow(f"Run artifact export — {source_name}", receipt, self, size=(620, 240))

        def submit():
            if self._artifact_export_task is not None:
                self.statusBar().showMessage("Run artifact export is already running", 5000)
                return
            directory = QFileDialog.getExistingDirectory(self, f"Export {source_name} artifacts to an empty directory")
            if not directory:
                return
            diagnostic = self._run_verification_error(record)
            if diagnostic:
                receipt.set_failed(diagnostic, directory=directory)
                window.open_panel()
                return
            task = ArtifactExportTask(record, directory, loader=self.service.load_run)
            self._artifact_export_task = task
            if hasattr(task.signals, "progress"):
                task.signals.progress.connect(self._background_work_progress)
            receipt.set_pending(directory, f"Exporting verified saved artifacts for {source_name}…")
            window.open_panel()
            self.statusBar().showMessage(f"Exporting verified artifacts for {source_name}…")

            def finished(paths, error):
                self._artifact_export_task = None
                getattr(self, "_local_progress", {}).pop(task.signals, None)
                if self._closing:
                    return
                if error:
                    receipt.set_failed(error, directory=task.directory)
                    if task.verification_error:
                        owner_id = (
                            source_id
                            if export_record_key(self.runs.get(source_id)) == export_record_key(record)
                            else str(record.directory)
                        )
                        self._mark_run_verification_error(owner_id, task.verification_error, directory=record.directory)
                        self._update_result_state()
                    message = f"Artifact export for {source_name} failed: {error}"
                else:
                    formats = ", ".join(dict.fromkeys(path.suffix.lstrip(".").upper() or "file" for path in paths))
                    description = (
                        f"Exported {len(paths)} verified {'artifact' if len(paths) == 1 else 'artifacts'} · {formats}. "
                        "Saved file units preserved; no coordinate conversion."
                    )
                    receipt.set_finished(task.directory, description)
                    message = f"Artifact export for {source_name} saved: {task.directory}"
                self.messages.appendPlainText(message)
                self.statusBar().showMessage(message, 15000)
                self._refresh_computation_indicator()

            task.signals.finished.connect(finished)
            QThreadPool.globalInstance().start(task)

        receipt.retry_requested.connect(submit)
        submit()

    def resume_selected_run(self):
        if not self._check_expert_draft():
            return None
        if not self.apply_settings():
            self._report_compute_draft_error()
            return None
        try:
            plan = self._selected_resume_plan(refresh=True)
            selected = plan.through(self.stage_selector.currentData())
            if selected is None:
                raise ValueError(
                    "No verified compatible boundary exists at or before the selected stage. "
                    + (plan.diagnostics[0] if plan.diagnostics else "Compute will rebuild affected stages.")
                )
            return self.compute(resume_checkpoint=str(selected.path))
        except (ValueError, TypeError, OSError, KeyError) as exc:
            self._error(f"Cannot resume run: {exc}")
            self._update_result_state()
            return None

    def _selected_resume_plan(self, *, refresh=False):
        """Cache menu capabilities; execution re-verifies the chosen source."""
        selected_job = self._selected_job_id()
        state = self.states.get(selected_job, {})
        stopped = state.get("status") in {"failed", "cancelled", "interrupted"}
        source_id = selected_job if stopped else self._selected_run
        record = None if stopped else self.runs.get(source_id)
        threads = getattr(getattr(self.jobs, "policy", None), "threads_per_worker", 1)
        error = self._run_verification_error(source_id)
        key = (
            self.project.fingerprint,
            self.stage_selector.currentData(),
            threads,
            source_id,
            id(record),
            state.get("status") if stopped else None,
            tuple(state.get("completed_stages", ())) if stopped else (),
            error,
        )
        if not refresh and key == getattr(self, "_resume_capability_key", None):
            return self._resume_capability_plan
        try:
            if error:
                plan = ReusePlan((), (), ("Saved evidence failed verification: " + error,))
            elif stopped:
                plan = plan_stopped_reuse(self.project, [state], self.jobs_directory, threads=threads)
            elif record is not None:
                if refresh:
                    try:
                        record = self.service.load_run(record.directory)
                    except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
                        self._mark_run_verification_error(source_id, exc, directory=record.directory)
                        raise
                plan = plan_reuse(self.project, [record], threads=threads)
            else:
                plan = ReusePlan((), (), ("Select a run or stopped job containing a verified checkpoint.",))
        except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
            plan = ReusePlan((), (), (str(exc),))
        self._resume_capability_key, self._resume_capability_plan = key, plan
        return plan

    def compute(self, _checked=False, *, resume_checkpoint=None):
        if self.project is None:
            self.statusBar().showMessage("Create or open a project before computing.", 5000)
            return None
        if not self._check_expert_draft():
            return None
        if self._submitting:
            return None
        if not self.apply_settings():
            self._report_compute_draft_error()
            return None

        self._submitting = True
        self._update_execution_feedback()
        try:
            if not self._check_compute_readiness(self.project):
                return None
            self.poll_jobs()
            plan = plan_execution(self.project, states=self.states.values(), **self._request_controls())
            existing = self.states.get(plan.active_job_id)
            if existing is not None and resume_checkpoint is None and not isinstance(self.service, DesignService):
                self._begin_live_follow(existing)
                self._focus_job(existing["id"], open_window=False)
                self.statusBar().showMessage(
                    f"This revision is already {existing['status']}; showing its design progress.", 10000
                )
                return existing
            if self.jobs is None:
                self.jobs = JobManager(self.jobs_directory, RuntimePolicy())
                self._own_jobs = True
            self._snapshot_existing_jobs()
            self._refresh_stage_status()
            plan = plan_execution(
                self.project,
                reuse=self._reuse_plan,
                **self._request_controls(),
                **({"resume_checkpoint": resume_checkpoint} if resume_checkpoint else {}),
            )
            plan.require_current(self.project)
            if isinstance(self.service, DesignService):
                state = self.service.submit_plan(self.project, self.jobs, plan)
            else:
                state = self.service.submit(self.project, self.jobs, **plan.request_controls())
            state.setdefault("pipeline_stage", self.stage_selector.currentData())
            state.setdefault("project_id", self.project.id)
            state.setdefault("project_fingerprint", self.project.fingerprint)
            state.setdefault("project_revision", self.project.revision)
            for key, value in self._request_controls().items():
                state.setdefault(key, value)
            reused = state["id"] in self.states
            self.states[state["id"]] = state
            if not reused:
                self._session_job_ids.add(state["id"])
            self._begin_live_follow(state)
            self.statusBar().showMessage(
                f"This revision is already {state['status']}; showing its design progress."
                if reused
                else f"Queued immutable revision {self.project.revision}",
                5000,
            )
            self.poll_jobs()
            self._focus_job(state["id"], open_window=False)
            return state
        except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
            self._error(f"Cannot submit run: {exc}")
            return None
        finally:
            self._submitting = False
            self._update_execution_feedback()

    def _same_active_job(self, state):
        return matches_active_request(state, self.project, self._request_controls())

    def _request_controls(self):
        return {
            **({"engineering": self.project.study["engineering"]} if "engineering" in self.project.study else {}),
            "pipeline_stage": self.stage_selector.currentData(),
            "validation_resolution": self.validation_resolution,
            "design_margin": self.design_margin,
            "fabricate": self.fabricate,
            "kernel_backend": self.project.study["kernel_backend"],
        }

    def _gradient_requirements_dialog(self):
        from .gradient_tools import GradientRequirementsDialog

        if self.project is None or not self._apply_document_inputs():
            return
        project = self.project
        dialog = GradientRequirementsDialog(project, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            config, study = dialog.result_value
            self.apply_inputs(
                config, study, label="Edit gradient requirements", expected_fingerprint=project.fingerprint
            )

    def _gradient_geometry_dialog(self):
        from .gradient_tools import GradientGeometryDialog

        if self.project is None or not self._apply_document_inputs():
            return
        project = self.project
        try:
            dialog = GradientGeometryDialog(project, self)
        except (ValueError, TypeError, OSError, KeyError) as exc:
            self._error(f"Gradient support geometry: {exc}")
            return
        if dialog.exec() == QDialog.DialogCode.Accepted:
            geometry = dialog.result_value
            self._gradient_task(
                lambda captured: geometry.apply(captured, expected_fingerprint=project.fingerprint),
                lambda result: self._apply_design_document(result, "Edit cylindrical gradient supports"),
                "Build gradient support meshes",
            )

    def _gradient_task(self, operation, callback, label, *, retain_receipt=False):
        """Run file verification and engineering analysis off the UI thread."""
        task = DesignStudyTask(operation, self.project)
        task.retain_receipt = retain_receipt
        if not hasattr(self, "_gradient_tasks"):
            self._gradient_tasks = {}
        self._gradient_tasks[task.signals] = (task, callback, label)
        task.signals.progress.connect(self._gradient_action_progress)
        task.signals.finished.connect(self._gradient_action_finished)
        self.statusBar().showMessage(f"{label}…")
        self._refresh_computation_indicator()
        QThreadPool.globalInstance().start(task)

    def _research_requirements_dialog(self):
        from .research_requirements import ResearchRequirementsDialog

        if self.project is None or not self._apply_document_inputs():
            return
        project = self.project
        dialog = ResearchRequirementsDialog(project.study.get("requirements"), self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.apply_inputs(
                study_changes={"requirements": dialog.result_value},
                label="Edit named research requirements",
                expected_fingerprint=project.fingerprint,
            )

    def _field_analysis_dialog(self):
        """Inspect a pinned saved result; analysis never writes the active model."""
        from .field_analysis import FieldAnalysisDialog

        record = self.runs.get(self._selected_run)
        if record is None or self._run_verification_error(record):
            self._error("Select verified computed field evidence to analyze.")
            return
        expected = result_metadata_digest({k: v for k, v in record.metadata.items() if k != "job"})
        key = (str(record.directory), expected)

        def verify(_project):
            saved = DesignService.load_run(record.directory)
            actual = result_metadata_digest({k: v for k, v in saved.metadata.items() if k != "job"})
            if actual != expected:
                raise ValueError("The selected result changed on disk. Reload its evidence before analyzing it.")
            return saved

        def display(saved):
            if not hasattr(self, "_field_analysis_windows"):
                self._field_analysis_windows = {}
            previous = self._field_analysis_windows.get(key)
            if previous is not None:
                previous.show()
                previous.raise_()
                return
            view_state = self.viewport.view_state() if hasattr(self.viewport, "view_state") else {}
            initial_source = view_state.get("dataset")
            if view_state.get("wire_current") == "Original contour current" and isinstance(initial_source, str):
                for source in ("contour", "routed"):
                    if initial_source.startswith(source):
                        initial_source = source + "_at_stream_current"
                        break
            dialog = FieldAnalysisDialog(saved, self, initial_source=initial_source)
            dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            self._field_analysis_windows[key] = dialog
            label = f"Field analysis · {saved.metadata.get('name', 'saved result')} · {expected[:8]}"
            dialog.busyChanged.connect(lambda busy: self._research_busy(label, busy, key=key))
            dialog.progressChanged.connect(lambda phase: self.statusBar().showMessage(f"{label}: {phase}"))
            dialog.progressDetails.connect(lambda event: self._research_progress(label, event, key=key))

            def closed():
                self._field_analysis_windows.pop(key, None)
                self._research_busy(label, False, key=key)

            dialog.closed.connect(closed)
            dialog.show()

        self._gradient_task(verify, display, "Verify saved field analysis inputs", retain_receipt=True)

    def _research_verified_review(self, key, display, label):
        if self.project is None or not self._apply_document_inputs():
            return
        record = self.runs.get(self._selected_run)
        if record is None or self._run_verification_error(record):
            self._error("Select a verified completed run to review its research evidence.")
            return

        def verify(_project):
            saved = DesignService.load_run(record.directory)
            if key not in saved.metadata:
                raise ValueError(
                    "This run contains no evidence for this research study. Apply its inputs and compute the design."
                )
            return saved

        self._gradient_task(verify, display, label)

    def _research_requirements_review(self):
        from .research_requirements import ResearchRequirementsReviewDialog

        self._research_verified_review(
            "requirements_review",
            lambda saved: ResearchRequirementsReviewDialog(
                saved.metadata["requirements_review"], saved.select_arrays(copy=False), self
            ).exec(),
            "Verify regional and pose evidence",
        )

    def _research_robust_comparison(self):
        from ..application.research_requirements import ResearchRequirements, nominal_requirements

        if self.project is None or not self._apply_document_inputs():
            return
        spec = self.project.study.get("requirements")
        if spec is None or not any(p.membership == "design" for p in ResearchRequirements.from_dict(spec).scenarios):
            self._error("Define at least one design pose before queueing nominal and robust candidates.")
            return
        if self.jobs is None:
            try:
                self.jobs = JobManager(self.jobs_directory, RuntimePolicy())
                self._own_jobs = True
            except (ValueError, OSError, RuntimeError) as exc:
                self._error(str(exc))
                return
        jobs = self.jobs
        owner_id = self.project.id
        self._snapshot_existing_jobs()

        def submit(project):
            from ..persistence import atomic_json
            from uuid import uuid4

            nominal = project.duplicate((project.name[:100] + " · nominal")[:120]).update_study(
                {"requirements": nominal_requirements(spec).to_dict()}
            )
            robust = project.duplicate((project.name[:100] + " · robust")[:120])
            states = []
            try:
                for candidate in (nominal, robust):
                    self.service.preflight(candidate).raise_for_errors()
                for candidate in (nominal, robust):
                    states.append(self.service.submit(candidate, jobs))
                path = self.jobs_directory.parent / "robust_comparison" / f"{uuid4()}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(
                    path,
                    {
                        "schema": "coilgen-nominal-robust-pair-v1",
                        "source_project_id": project.id,
                        "source_fingerprint": project.fingerprint,
                        "coverage": "same desired datasets and listed poses; nominal excludes additional design-pose bounds",
                        "candidates": [
                            {"kind": name, "job_id": state["id"], "project": candidate.to_dict()}
                            for name, state, candidate in zip(
                                ("nominal", "robust"), states, (nominal, robust), strict=True
                            )
                        ],
                    },
                )
            except Exception:
                for state in states:
                    jobs.cancel(state["id"])
                raise
            return states, path

        def queued(value):
            states, path = value
            self._research_robust_path = path
            self._research_robust_owner = owner_id
            self._session_job_ids.update(state["id"] for state in states)
            self.messages.appendPlainText(
                f"Queued nominal and robust candidates with common evaluation poses. Comparison manifest: {path}"
            )
            self.poll_jobs()
            self._focus_job(states[0]["id"])

        self._gradient_task(submit, queued, "Queue nominal and robust designs", retain_receipt=True)

    def _research_robust_review(self):
        from ..application.comparison import compare_runs
        from .research_review import ResearchComparisonDialog

        if self.project is None or not self._apply_document_inputs():
            return
        path = (
            getattr(self, "_research_robust_path", None)
            if getattr(self, "_research_robust_owner", None) == self.project.id
            else None
        )
        if path is None:
            name, _ = QFileDialog.getOpenFileName(
                self,
                "Open nominal / robust comparison manifest",
                str(self.jobs_directory.parent / "robust_comparison"),
                "Comparison manifest (*.json)",
            )
            if not name:
                return
            path = Path(name)
        directory = self.jobs_directory.resolve()

        def review(_project):
            import re

            value = json.loads(Path(path).read_text(encoding="utf-8"))
            if value.get("schema") != "coilgen-nominal-robust-pair-v1" or len(value.get("candidates", ())) != 2:
                raise ValueError("Malformed nominal/robust comparison manifest.")
            records = []
            for candidate, kind in zip(value["candidates"], ("nominal", "robust"), strict=True):
                id_ = candidate.get("job_id")
                if (
                    candidate.get("kind") != kind
                    or not isinstance(id_, str)
                    or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", id_) is None
                ):
                    raise ValueError("Invalid comparison candidate identity.")
                project = ProjectDocument.from_dict(candidate["project"])
                saved = DesignService.load_run(directory / id_)
                if any(
                    saved.metadata.get(key) != expected
                    for key, expected in (
                        ("project_id", project.id),
                        ("project_revision", project.revision),
                        ("project_fingerprint", project.fingerprint),
                    )
                ):
                    raise ValueError("Comparison result does not belong to its saved candidate contract.")
                records.append(saved)
            return compare_runs(*records), value.get("source_fingerprint", "Unavailable")

        self._gradient_task(
            review,
            lambda value: ResearchComparisonDialog(value[0], self, source_fingerprint=value[1]).exec(),
            "Verify nominal and robust comparison",
        )

    def _research_scene_dialog(self):
        from .spatial_scene import SpatialSceneDialog
        from ..application.spatial_scene import SpatialScene

        if self.project is None or not self._apply_document_inputs():
            return
        previous = getattr(self, "_scene_editor", None)
        if previous is not None:
            if previous.property("sourceProjectId") == self.project.id:
                previous.show()
                previous.raise_()
                return
            previous.reject()
        project = self.project
        record = self.runs.get(self._selected_run)
        if record is not None and self._run_verification_error(record):
            record = None

        def edit(saved):
            previous = getattr(self, "_scene_editor", None)
            if previous is not None:
                if previous.property("sourceProjectId") == project.id:
                    previous.show()
                    previous.raise_()
                    return
                previous.reject()
            if saved is not None and saved.metadata.get("project_model_fingerprint") != project.model_fingerprint:
                raise ValueError("Select winding evidence for the current applied model to bind spatial anchors.")
            dialog = SpatialSceneDialog(project, saved, self)
            dialog.setProperty("sourceProjectId", project.id)
            self._scene_editor = dialog
            if hasattr(self.viewport, "set_scene"):
                dialog.sceneChanged.connect(self.viewport.set_scene)
            if hasattr(self.viewport, "set_point_pick_mode"):
                dialog.pickRequested.connect(lambda: self.viewport.set_point_pick_mode(True))
                self.viewport.point_picked.connect(dialog.picked_point)

            def finish(code):
                if hasattr(self.viewport, "set_point_pick_mode"):
                    self.viewport.set_point_pick_mode(False)
                    self.viewport.point_picked.disconnect(dialog.picked_point)
                if code == QDialog.DialogCode.Accepted:
                    if self.project is None or self.project.id != dialog.property("sourceProjectId"):
                        self._error("The spatial scene belongs to another project. Reopen it for the active channel.")
                    else:
                        self.apply_inputs(
                            study_changes={"scene": dialog.result_value},
                            label="Edit spatial scene and bound anchors",
                            expected_fingerprint=project.fingerprint,
                        )
                if hasattr(self.viewport, "set_scene"):
                    current = self.project.study.get("scene") if self.project else None
                    self.viewport.set_scene(SpatialScene.from_dict(current) if current else None)
                self._scene_editor = None
                dialog.deleteLater()

            dialog.finished.connect(finish)
            dialog.show()
            if hasattr(self.viewport, "set_scene"):
                self.viewport.set_scene(dialog.scene)

        if record is None:
            edit(None)
        else:
            self._gradient_task(
                lambda _project: DesignService.load_run(record.directory), edit, "Verify spatial anchor geometry"
            )

    def _research_scene_review(self):
        from .research_review import SpatialReviewDialog

        self._research_verified_review(
            "scene_review",
            lambda saved: SpatialReviewDialog(saved.metadata["scene_review"], self).exec(),
            "Verify spatial clearance evidence",
        )

    def _research_array_dialog(self):
        from .array_studies import ArrayStudyDialog

        if self.project is None or not self._apply_document_inputs():
            return
        project = self.project
        dialog = ArrayStudyDialog(
            self.assembly,
            self,
            saved_spec=project.study.get("array_study"),
            run_records=tuple(self.runs.values()),
            active_project=project,
        )
        if hasattr(dialog, "busyChanged"):
            dialog.busyChanged.connect(lambda busy: self._research_busy("Fit array / synthesize cohort", busy))
        if hasattr(dialog, "progressDetails"):
            dialog.progressDetails.connect(
                lambda event: self._research_progress("Fit array / synthesize cohort", event)
            )
        dialog.generatedProjects.connect(self._research_generated_channels)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.apply_inputs(
                study_changes={"array_study": dialog.result_value},
                label="Edit array study contract",
                expected_fingerprint=project.fingerprint,
            )
        self._research_busy("Fit array / synthesize cohort", False)

    def _research_busy(self, label, busy, *, key=None):
        key = label if key is None else key
        if not hasattr(self, "_research_active_tasks"):
            self._research_active_tasks = set()
            self._research_task_labels = {}
        if busy:
            self._research_active_tasks.add(key)
            self._research_task_labels[key] = label
        else:
            self._research_active_tasks.discard(key)
            self._research_task_labels.pop(key, None)
            getattr(self, "_local_progress", {}).pop(key, None)
        self._refresh_computation_indicator()
        if not self._research_active_tasks and getattr(self, "_research_close_pending", False):
            self._research_close_pending = False
            QTimer.singleShot(0, self, self.close)

    def _research_progress(self, label, event, *, key=None):
        key = label if key is None else key
        if key in getattr(self, "_research_active_tasks", ()):
            self._local_work_progress(key, label, event)

    def _gradient_action_progress(self, event):
        key = self.sender()
        task = getattr(self, "_gradient_tasks", {}).get(key)
        if task is not None:
            self._local_work_progress(key, task[2], event)

    def _background_work_progress(self, event):
        key = self.sender()
        for attribute, label in (
            ("_geometry_export_task", "Exporting conductor geometry"),
            ("_artifact_export_task", "Exporting saved artifacts"),
            ("_design_study_task", self._design_study_task_label),
        ):
            task = getattr(self, attribute, None)
            if task is not None and getattr(task, "signals", None) is key:
                self._local_work_progress(key, label, event)
                return

    def _local_work_progress(self, key, label, event):
        from ..progress import update_work_progress

        if self._closing:
            return
        if not hasattr(self, "_local_progress"):
            self._local_progress = {}
        state = self._local_progress.setdefault(key, {"name": label})
        # Local tools may supply a directly measured total without a pipeline
        # allocation. Keep their current operation distinct from job progress.
        value = event.get("overall_percent", event.get("percent"))
        update_work_progress(state, {**event, "overall_percent": value}, completion_limit=1.0)
        if value is None:
            state.pop("progress", None)
        self.statusBar().showMessage(f"{label}: {state['stage']}")
        self._refresh_computation_indicator()

    def _research_generated_channels(self, projects):
        if self.assembly is None:
            return
        before, after = self.assembly, self.assembly
        for project in projects:
            name, counter = project.name, 2
            while any(channel.name.casefold() == name.casefold() for channel in after.channels):
                name = f"{project.name[:100]} {counter}"
                counter += 1
            after = after.add_channel(project, name=name)
        if after is not before:
            self._enter_collection_mode()
            self.undo_stack.push(AssemblyEdit(self, after, self._active_channel_id, "Add cohort-designed channels"))

    def _research_dynamic_dialog(self):
        from .dynamic_studies import DynamicStudyDialog

        if self.project is None or not self._apply_document_inputs():
            return
        project = self.project
        record = self.runs.get(self._selected_run)
        if record is not None and self._run_verification_error(record):
            record = None

        def edit(saved):
            if saved is not None and (
                saved.metadata.get("project_id") != project.id
                or saved.metadata.get("project_model_fingerprint") != project.model_fingerprint
            ):
                raise ValueError("Select evidence for this channel's current applied model to bind circuit references.")
            dialog = DynamicStudyDialog(project, saved, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                changes = {"dynamic_analysis": dialog.result_value}
                engineering = dialog.engineering_value()
                if engineering is not None:
                    changes["engineering"] = engineering
                self.apply_inputs(
                    study_changes=changes,
                    label="Edit circuit references and coupled evidence",
                    expected_fingerprint=project.fingerprint,
                )

        if record is None:
            edit(None)
        else:
            self._gradient_task(
                lambda _project: DesignService.load_run(record.directory), edit, "Verify coupled-physics geometry"
            )

    def _research_dynamic_review(self):
        from .dynamic_studies import DynamicReviewDialog

        if self.project is None or not self._apply_document_inputs():
            return
        record = self.runs.get(self._selected_run)
        if record is None or self._run_verification_error(record):
            self._error("Select a verified run with waveform or coupled-physics evidence.")
            return

        def display(saved):
            report = saved.metadata.get("dynamic_analysis_review")
            if report is None and saved.metadata.get("engineering_review"):
                report = {"engineering_review": saved.metadata["engineering_review"]}
            if report is None:
                raise ValueError("This run has no waveform or coupled-physics evidence.")
            DynamicReviewDialog(report, self).exec()

        self._gradient_task(
            lambda _project: DesignService.load_run(record.directory),
            display,
            "Verify waveform and coupled-physics evidence",
        )

    def _gradient_action_finished(self, value, error):
        key = self.sender()
        task, callback, label = self._gradient_tasks.pop(key)
        getattr(self, "_local_progress", {}).pop(key, None)
        if self._closing:
            return
        self._refresh_computation_indicator()
        if error:
            self._error(f"{label}: {error}")
        elif not getattr(task, "retain_receipt", False) and (
            self.project is None
            or (task.project.id, task.project.revision, task.project.fingerprint)
            != (self.project.id, self.project.revision, self.project.fingerprint)
            or self._has_input_drafts()
        ):
            self.messages.appendPlainText(
                f"{label}: the project or draft changed during the operation. Repeat the action on the intended revision; submitted jobs retain their original inputs."
            )
        else:
            try:
                callback(value)
            except (ValueError, TypeError, OSError, KeyError, RuntimeError) as exc:
                self._error(f"{label}: {exc}")

    def _gradient_convergence_dialog(self):
        from .gradient_tools import ConvergenceDialog
        from ..application.gradient_studies import convergence_candidates, submit_convergence

        if self.project is None or not self._apply_document_inputs():
            return
        dialog = ConvergenceDialog(self.project, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        levels, criteria = dialog.result_value
        if self.jobs is None:
            try:
                self.jobs = JobManager(self.jobs_directory, RuntimePolicy())
                self._own_jobs = True
            except (ValueError, OSError, RuntimeError) as exc:
                self._error(f"Cannot start convergence runtime: {exc}")
                return
        self._snapshot_existing_jobs()
        jobs = self.jobs

        def submit(project):
            candidates = convergence_candidates(project, **levels)
            submitted = submit_convergence(candidates, jobs)
            # Candidate requests already own all scientific inputs; this manifest
            # records their axis and criterion without duplicating mesh payloads.
            from ..persistence import atomic_json
            from uuid import uuid4

            path = self.jobs_directory.parent / "convergence" / f"{uuid4()}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(
                path,
                {
                    "baseline": candidates[0].project.with_results((), session={}).to_dict(),
                    "source_fingerprint": project.fingerprint,
                    "criteria": criteria,
                    "candidates": [
                        {
                            "axis": item.candidate.axis,
                            "value": item.candidate.value,
                            "id": item.candidate.project.id,
                            "revision": item.candidate.project.revision,
                            "fingerprint": item.candidate.project.fingerprint,
                            "job_id": item.job_id,
                        }
                        for item in submitted
                    ],
                },
            )
            return submitted, path

        def queued(value):
            submitted, path = value
            self._gradient_convergence_path = path
            self._session_job_ids.update(item.job_id for item in submitted)
            self.messages.appendPlainText(f"Queued {len(submitted)} convergence candidates. Manifest: {path}")
            self.poll_jobs()
            self._focus_job(submitted[0].job_id)

        self._gradient_task(submit, queued, "Queue convergence study")

    def _gradient_convergence_review(self):
        from ..application.gradient_studies import ConvergenceCriterion, StudyCandidate, summarize_convergence

        if self.project is None or not self._apply_document_inputs():
            return
        path = getattr(self, "_gradient_convergence_path", None)
        if path is None:
            name, _ = QFileDialog.getOpenFileName(
                self,
                "Open convergence manifest",
                str(self.jobs_directory.parent / "convergence"),
                "Study manifest (*.json)",
            )
            if not name:
                return
            path = Path(name)
        jobs_directory = self.jobs_directory

        def review(_project):
            descriptor = json.loads(Path(path).read_text(encoding="utf-8"))
            baseline = ProjectDocument.from_dict(descriptor["baseline"])
            candidates, records = [], {}
            for item in descriptor["candidates"]:
                axis, value = item["axis"], item["value"]
                candidate = replace(baseline, id=item["id"], revision=1)
                if axis != "baseline":
                    fields = {
                        "mesh": "iteration_num_mesh_refinement",
                        "quadrature": "gauss_order",
                        "independent_sampling": "validation_resolution",
                    }
                    candidate = candidate.update_inputs(
                        study_changes={fields[axis]: value} if axis == "independent_sampling" else None,
                        config_changes=None if axis == "independent_sampling" else {fields[axis]: value},
                    )
                candidate = replace(candidate, revision=item["revision"])
                if candidate.fingerprint != item["fingerprint"]:
                    raise ValueError("Convergence manifest input fingerprint does not match its candidate.")
                candidates.append(StudyCandidate(candidate, axis, value, descriptor["source_fingerprint"]))
                records[candidate.id] = jobs_directory / item["job_id"]
            criteria = descriptor["criteria"]
            report = summarize_convergence(
                candidates,
                records,
                criteria=(
                    ConvergenceCriterion(
                        "Sheet independent RMS error",
                        criteria["relative_tolerance"],
                        criteria["absolute_tolerance_T"] * 1e6,
                    ),
                ),
            )
            from ..persistence import atomic_json

            atomic_json(
                Path(path).with_suffix(".review.json"), {**report.to_dict(), "software": software_attribution()}
            )
            return report

        def display(report):
            dialog = QDialog(self)
            dialog.setWindowTitle("Verified convergence review")
            layout = QVBoxLayout(dialog)
            label = QLabel(
                "Observed stable axes: "
                + (", ".join(report.stable_axes) or "none established")
                + "\n"
                + "\n".join(report.notes)
            )
            label.setWordWrap(True)
            layout.addWidget(label)
            table = QTableWidget(len(report.changes), 7)
            table.setHorizontalHeaderLabels(["Axis", "From", "To", "Metric", "Change", "Criterion", "Scope / gap"])
            for index, row in enumerate(report.changes):
                for column, text in enumerate(
                    (
                        row.axis,
                        row.earlier,
                        row.later,
                        row.metric,
                        f"{row.absolute_change:.6g} {row.unit}" if row.absolute_change is not None else "Unavailable",
                        "Met"
                        if row.within_tolerance is True
                        else "Not met"
                        if row.within_tolerance is False
                        else "Unverified",
                        row.reason,
                    )
                ):
                    cell = QTableWidgetItem(text)
                    cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    table.setItem(index, column, cell)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            layout.addWidget(table)
            dialog.resize(1000, 580)
            dialog.exec()

        self._gradient_task(review, display, "Verify convergence evidence")

    def _gradient_circuit_dialog(self):
        from .gradient_tools import CircuitDriveDialog

        if self.project is None or not self._apply_document_inputs():
            return
        record = self.runs.get(self._selected_run)
        if record is None or self._run_verification_error(record):
            self._error("Select a verified Fabrication review run before authoring its circuit.")
            return

        def verify(project):
            saved = DesignService.load_run(record.directory)
            if (
                saved.metadata.get("project_id") != project.id
                or saved.metadata.get("project_model_fingerprint") != project.model_fingerprint
            ):
                raise ValueError("Select a fabrication review for this channel's applied numerical model.")
            return saved

        def edit(saved):
            project = self.project
            try:
                dialog = CircuitDriveDialog(project, saved, self)
            except (ValueError, TypeError, KeyError) as exc:
                self._error(str(exc))
                return
            if dialog.exec() == QDialog.DialogCode.Accepted:
                from ..persistence import json_value

                if dialog.result_value == json_value(project.study.get("engineering")):
                    return
                self.apply_inputs(
                    study_changes={
                        "engineering": dialog.result_value,
                        "fabricate": True,
                        "pipeline_stage": "evaluated",
                    },
                    label="Define gradient circuit and drive",
                    expected_fingerprint=project.fingerprint,
                )

        self._gradient_task(verify, edit, "Verify winding for circuit authoring")

    def _gradient_measured_dialog(self):
        from .gradient_tools import MeasuredFieldDialog
        from ..application.gradient_studies import saved_winding_currents

        if self.project is None or not self._apply_document_inputs():
            return
        record = self.runs.get(self._selected_run)
        if record is None or self._run_verification_error(record):
            self._error("Select verified winding evidence before comparing measurements.")
            return

        def verify(_project):
            saved = DesignService.load_run(record.directory)
            return saved, saved_winding_currents(saved)

        def edit(value):
            saved, currents = value
            dialog = MeasuredFieldDialog(self, body_ids=tuple(currents["body_currents_A"]))
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self._compare_gradient_measurement(saved, dialog.result_value)

        self._gradient_task(verify, edit, "Verify winding measurement reference")

    def _compare_gradient_measurement(self, record, inputs):
        from ..application.gradient_studies import MeasuredFieldMap, RigidTransform, compare_measured_winding
        from ..persistence import atomic_json
        from uuid import uuid4

        output = self.jobs_directory.parent / "measurements" / f"{uuid4()}.json"

        def compare(_project):
            values = np.asarray(inputs["values"], dtype=float)
            measurement = MeasuredFieldMap(
                values[:, :3],
                values[:, 3:],
                inputs["length_unit"],
                inputs["field_unit"],
                independent=True,
                source=inputs["source_file"] + " · " + inputs["source"],
                source_bytes=inputs["source_bytes"],
            )
            transform = RigidTransform(inputs["rotation"], inputs["translation_m"])
            report = compare_measured_winding(
                record.directory,
                measurement,
                transform=transform,
                body_currents_A=inputs["currents_A"],
                rms_limit_T=inputs["rms_limit_T"],
                peak_limit_T=inputs["peak_limit_T"],
            )
            atomic_json(output, {**report.to_dict(), "software": software_attribution()})
            return report

        def show(report):
            dialog = QDialog(self)
            dialog.setWindowTitle("Measured winding comparison")
            layout = QVBoxLayout(dialog)
            status = (
                "Within declared limits"
                if report.passed is True
                else "Outside declared limits"
                if report.passed is False
                else "Descriptive · acceptance unverified"
            )
            label = QLabel(
                f"{status}\nRMS vector error: {report.rms_error_T * 1e6:.6g} µT\nPeak vector error: {report.peak_error_T * 1e6:.6g} µT\nSaved comparison: {output}\nOriginal measurement, world transform and signed current calibration are retained. Thin-wire model; construction release remains separate."
            )
            label.setWordWrap(True)
            layout.addWidget(label)
            indices = np.arange(len(report.points_m))
            if len(indices) > 1000:
                indices = np.argsort(np.linalg.norm(report.residual_T, axis=1))[-1000:][::-1]
                layout.addWidget(
                    QLabel("Showing the 1000 largest residual samples. The archive retains every measurement.")
                )
            samples = QTableWidget(len(indices), 5)
            samples.setHorizontalHeaderLabels(
                ["World x [m]", "World y [m]", "World z [m]", "Measured |B| [µT]", "Residual |ΔB| [µT]"]
            )
            for index, (point, measured, residual) in enumerate(
                zip(report.points_m[indices], report.measured_T[indices], report.residual_T[indices], strict=True)
            ):
                for column, value in enumerate(
                    (*point, np.linalg.norm(measured) * 1e6, np.linalg.norm(residual) * 1e6)
                ):
                    cell = QTableWidgetItem(f"{value:.7g}")
                    cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    samples.setItem(index, column, cell)
            layout.addWidget(samples)
            export = QPushButton("Export comparison archive…")

            def save():
                filename, _ = QFileDialog.getSaveFileName(
                    dialog, "Export measured comparison", "measured-comparison.json", "Comparison archive (*.json)"
                )
                if filename:
                    atomic_json(filename, {**report.to_dict(), "software": software_attribution()})

            export.clicked.connect(save)
            layout.addWidget(export)
            dialog.resize(860, 620)
            dialog.exec()
            self.messages.appendPlainText(f"Measured winding comparison saved: {output}")

        self._gradient_task(compare, show, "Compare measured winding field")

    def _focus_job(self, id_, *, open_window=True):
        if not any(
            self.jobs_table.item(row, 0).data(Qt.ItemDataRole.UserRole) == id_
            for row in range(self.jobs_table.rowCount())
        ):
            self.jobs_filter.setCurrentIndex(self.jobs_filter.findData("all"))
            self._fill_jobs(self._last_job_states)
        self._feedback_job_id = id_
        self.evidence.setCurrentWidget(self.jobs_table)
        if open_window:
            self._show_dock("evidence_dock")
        for row in range(self.jobs_table.rowCount()):
            item = self.jobs_table.item(row, 0)
            if item and item.data(Qt.ItemDataRole.UserRole) == id_:
                self.jobs_table.selectRow(row)
                self.jobs_table.scrollToItem(item)
                break
        self._update_execution_feedback()

    def compute_channels(self):
        if self.assembly is None:
            self.statusBar().showMessage("Create or open a project before computing.", 5000)
            return None
        if not self._check_expert_draft():
            return None
        if not self.apply_settings():
            self._report_compute_draft_error()
            return None
        try:
            enabled = [channel for channel in self.assembly.channels if channel.enabled]
            if not enabled:
                self.readiness_panel.set_issues(
                    (
                        ReadinessIssue(
                            title="No enabled channel",
                            message="Enable at least one channel before computing the assembly.",
                            node="geometry",
                            key="channel_enabled",
                            action="Use Enabled beside the active channel selector.",
                        ),
                    )
                )
                self.statusBar().showMessage("Enable at least one channel before computing.", 15000)
                return None
            for channel in enabled:
                if not self._check_compute_readiness(channel.project, channel_name=channel.name):
                    self.channel_selector.setCurrentIndex(self.channel_selector.findData(channel.id))
                    self._check_compute_readiness(channel.project, channel_name=channel.name)
                    return None
            if self.jobs is None:
                self.jobs = JobManager(self.jobs_directory, RuntimePolicy())
                self._own_jobs = True
            self._snapshot_existing_jobs()
            states = self.service.submit_assembly(self.assembly, self.jobs)
            self._session_job_ids.update(state["id"] for state in states.values())
            self.states.update({state["id"]: state for state in states.values()})
            active_state = next((state for state in states.values() if self._live_owner_matches(state)), None)
            if active_state is not None:
                self._begin_live_follow(active_state)
            self.messages.appendPlainText(
                "Submitted independent channels. Coupled fields, circuits, mutual inductance and assembly clearance remain unverified."
            )
            self.poll_jobs()
            if states:
                self._focus_job((active_state or next(iter(states.values())))["id"], open_window=False)
            return states
        except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
            self._error(f"Cannot submit channels: {exc}")
            return None

    def poll_jobs(self):
        if self.jobs is None:
            self._update_execution_feedback()
            return
        try:
            states = self.jobs.list()
            self._last_job_states = tuple(states)
            if self.project is None:
                self.states.update((state["id"], dict(state)) for state in states)
                self._last_runtime_error = None
                self._fill_jobs(states)
                self._record_job_outcomes(states)
                self._refresh_computation_indicator()
                return
            self._refresh_computation_indicator()
            self._snapshot_existing_jobs(states)
            merged_states = []
            stopped_changed = False
            for state in states:
                previous = self.states.get(state["id"], {})
                # A test/legacy runtime may omit service annotations; preserve them.
                state = {**previous, **state}
                self.states[state["id"]] = state
                stopped_changed |= state.get("status") in {"failed", "cancelled", "interrupted"} and state.get(
                    "status"
                ) != previous.get("status")
                merged_states.append(state)
                if state["status"] == "completed" and state["id"] not in self._loaded_ids:
                    self._loading_job_id = state["id"]
                    self._update_execution_feedback()
                    self.execution_label.repaint()
                    self.execution_progress.repaint()
                    try:
                        self.load_run(self.jobs.directory / state["id"], id_=state["id"], select=None)
                        self._loaded_ids.add(state["id"])
                        self._clear_run_verification_error(self.runs[state["id"]], id_=state["id"])
                    except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
                        self._loaded_ids.add(state["id"])
                        self._mark_run_verification_error(state["id"], exc, directory=self.jobs.directory / state["id"])
                        self._error(f"Completed run could not be verified: {exc}", open_window=False)
                    finally:
                        self._loading_job_id = None
                if state["id"] == self.live_preview.job_id:
                    self.live_preview.refresh(state)
            self._fill_jobs(merged_states)
            self._record_job_outcomes(merged_states)
            self._refresh_stage_status(refresh_reuse=stopped_changed)
            self._last_runtime_error = None
            self._refresh_channel_overview()
            self._update_execution_feedback()
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            message = f"Job runtime unavailable: {exc}"
            if message != self._last_runtime_error:
                self._error(message)
                self._last_runtime_error = message
            self._update_execution_feedback()

    def _record_job_outcomes(self, states):
        unresolved = {outcome.job_id: outcome for outcome in self.job_outcomes.outcomes}
        for state in states:
            id_, status = state["id"], state.get("status")
            previous = self._observed_job_statuses.get(id_)
            self._observed_job_statuses[id_] = status
            if status not in {"completed", "failed", "cancelled", "interrupted"}:
                continue
            diagnostic = self._run_verification_error(id_)
            key = (id_, status)
            observed = previous in {"queued", "running", "cancelling"} or id_ in self._session_job_ids
            if not observed and not diagnostic and id_ not in unresolved:
                continue
            record = self.runs.get(id_)
            if status == "completed" and record is None and not diagnostic and id_ in unresolved:
                continue
            outcome = build_job_outcome(state, record, diagnostic)
            if key not in self._notified_job_outcomes:
                self.messages.appendPlainText(outcome.title + "\n" + outcome.detail)
                self._notified_job_outcomes.add(key)
                self.job_outcomes.publish(outcome)
            elif unresolved.get(id_) is not None and unresolved[id_] != outcome:
                self.job_outcomes.publish(outcome)

    def _open_job_outcome(self, id_):
        state = self.states.get(id_) or next((state for state in self._last_job_states if state["id"] == id_), None)
        if state is None:
            self.statusBar().showMessage("This job is no longer in the runtime; inspect saved History.", 10000)
            return
        self.states.setdefault(id_, dict(state))
        self._focus_job(id_)
        diagnostic = self._run_verification_error(id_)
        record = self.runs.get(id_)
        if state.get("status") == "completed" and not diagnostic and self.project is not None:
            directory = record.directory if record is not None else self.jobs.directory / id_
            self._loading_job_id = id_
            self._update_execution_feedback()
            try:
                record = self.load_run(directory, id_=id_, select=False, expected_record=record)
                self._loaded_ids.add(id_)
            except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
                self._mark_run_verification_error(id_, exc, directory=directory)
                diagnostic = self._run_verification_error(id_)
                self._error(f"Completed run could not be verified: {exc}", open_window=False)
            finally:
                self._loading_job_id = None
                self._record_job_outcomes((state,))
                self._update_execution_feedback()
        if state.get("status") == "completed" and record is not None and not diagnostic and self.project is not None:
            self.show_run(id_)
            self._open_evidence(self.acceptance_panel)
            if self.job_outcomes.selected and self.job_outcomes.selected.action_label == "Review checks":
                self._open_evidence(self.problems_panel)
        else:
            self.messages.setPlainText(
                "\n".join(state.get("log", ()))
                + ("\n" + str(diagnostic or state.get("error")) if diagnostic or state.get("error") else "")
            )
            self._open_evidence(self.messages)

    def _snapshot_existing_jobs(self, states=None):
        if not self._runtime_snapshot_taken and self.jobs is not None:
            states = self.jobs.list() if states is None else states
            self._previous_job_ids.update(state["id"] for state in states if state["id"] not in self._session_job_ids)
            self._runtime_snapshot_taken = True

    def _job_origin(self, id_):
        return job_origin(id_, initial_ids=self._previous_job_ids, session_ids=self._session_job_ids)

    def _fill_jobs(self, states):
        self._last_job_states = tuple(states)
        counts = {
            origin: sum(self._job_origin(state["id"]) == origin for state in states)
            for origin in ("current", "previous", "other")
        }
        opened = format_job_timestamp(self._session_started_at.isoformat())
        self.jobs_context.setText(
            f"Window opened {opened.text} · This session: {counts['current']} · Earlier: {counts['previous']}"
            + (f" · Other: {counts['other']}" if counts["other"] else "")
        )
        self.jobs_context.setToolTip(
            opened.tooltip
            + "\nSession labels describe submission origin; they do not establish model compatibility or check acceptance."
        )
        selected_filter = self.jobs_filter.currentData()

        def order(state):
            submitted = format_job_timestamp(state.get("created_at"))
            origin = self._job_origin(state["id"])
            return (
                0 if state.get("status") in {"queued", "running", "cancelling"} else 1,
                0 if origin == "current" else 1,
                -(submitted.timestamp if submitted.timestamp is not None else float("-inf")),
            )

        visible = sorted(
            (
                state
                for state in states
                if selected_filter == "all"
                or (self._job_origin(state["id"]) == "current") == (selected_filter == "current")
            ),
            key=order,
        )
        selected_id = self._selected_job_id()
        with QSignalBlocker(self.jobs_table):
            self.jobs_table.clearSelection()
            self.jobs_table.setRowCount(len(visible))
            for row, state in enumerate(visible):
                progress = self._job_progress(state)
                origin = self._job_origin(state["id"])
                origin_label = {"current": "This session", "previous": "Earlier session", "other": "Other session"}[
                    origin
                ]
                origin_detail = {
                    "current": "Submitted from this workbench window; the label resets when the window is reopened.",
                    "previous": "Already present when this workbench connected to the job runtime, including any ongoing computation.",
                    "other": "Discovered later and not recorded as submitted from this window; its session origin is unverified.",
                }[origin]
                submitted = format_job_timestamp(state.get("created_at"))
                values = [
                    state.get("name", state["id"][:10]),
                    origin_label,
                    submitted.text,
                    state["status"],
                    state.get("stage", ""),
                    state.get("project_revision", "—"),
                    "—" if progress is None else f"{progress}%",
                    self._job_elapsed(state),
                ]
                for column, value in enumerate(values):
                    item = QTableWidgetItem(str(value))
                    item.setData(Qt.ItemDataRole.UserRole, state["id"])
                    item.setToolTip(
                        (self._run_verification_error(state["id"]) or state.get("error") or str(value))
                        + f"\nJob ID: {state['id']}\n{origin_label}: {origin_detail}\nSubmitted: {submitted.tooltip}"
                    )
                    if column == 1:
                        item.setIcon(icon("compute-study" if origin == "current" else "run-history"))
                        item.setForeground(
                            item_foreground("info" if origin == "current" else "neutral", self.palette())
                        )
                        font = item.font()
                        font.setBold(origin == "current")
                        item.setFont(font)
                    elif column == 3:
                        item.setIcon(
                            icon(
                                "compute-study",
                                badge="failed"
                                if state["status"] == "failed"
                                else "pending"
                                if state["status"] in {"queued", "running", "cancelling"}
                                else "evaluated"
                                if state["status"] == "completed"
                                else "unverified",
                            )
                        )
                        item.setForeground(
                            item_foreground(
                                "error"
                                if state["status"] == "failed"
                                else "warning"
                                if state["status"] in {"cancelled", "interrupted"}
                                else "info"
                                if state["status"] in {"queued", "running", "cancelling"}
                                else "neutral",
                                self.palette(),
                            )
                        )
                    self.jobs_table.setItem(row, column, item)
                if state["id"] == selected_id:
                    self.jobs_table.selectRow(row)
        self._update_execution_feedback()

    def _job_selected(self, row, *_args):
        item = self.jobs_table.item(row, 0)
        if not item:
            return
        id_ = item.data(Qt.ItemDataRole.UserRole)
        state = self.states.get(id_, {})
        self.messages.setPlainText(
            "\n".join(state.get("log", [])) + (f"\n{state['error']}" if state.get("error") else "")
        )
        if id_ in self.runs:
            self.show_run(id_)
        self._feedback_job_id = id_
        self._update_execution_feedback()

    def _selected_job_id(self):
        if not self.jobs_table.selectionModel().hasSelection():
            return None
        item = self.jobs_table.item(self.jobs_table.currentRow(), 0)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    @staticmethod
    def _job_progress(state):
        if state.get("status") == "completed":
            return 100
        value = state.get("progress")
        if isinstance(value, bool):
            return None
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return round(max(0.0, min(1.0, value)) * 100) if math.isfinite(value) else None

    @staticmethod
    def _job_elapsed(state, *, now=None):
        def timestamp(value):
            if not isinstance(value, str):
                return None
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

        started = timestamp(state.get("created_at"))
        if started is None:
            return "—"
        end = (
            now or datetime.now(timezone.utc)
            if state.get("status") in {"queued", "running", "cancelling"}
            else timestamp(state.get("updated_at"))
        )
        if end is None:
            return "—"
        elapsed = max(0, int((end - started).total_seconds()))
        minutes, seconds = divmod(elapsed, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"

    def _execution_job_id(self):
        selected = self._selected_job_id()
        if self._loading_job_id is not None:
            return self._loading_job_id
        if selected in self.states:
            return selected
        if self.project is None:
            return self._feedback_job_id if self._feedback_job_id in self.states else None
        previous = self.states.get(self._feedback_job_id, {})
        if previous and previous.get("project_id") in {None, self.project.id}:
            return self._feedback_job_id
        matching = [state for state in self.states.values() if state.get("project_id") in {None, self.project.id}]
        active = [state for state in matching if state.get("status") in {"queued", "running", "cancelling"}]
        return next(iter(active or matching), {}).get("id")

    def _run_verification_directory(self, source):
        record = source if hasattr(source, "directory") else self.runs.get(source)
        if record is not None:
            return Path(record.directory).expanduser().resolve()
        path = getattr(self, "_run_verification_paths", {}).get(source)
        if path is None and source in self.states:
            directory = getattr(self.jobs, "directory", self.jobs_directory)
            path = (Path(directory) / source).expanduser().resolve()
        return path

    def _mark_run_verification_error(self, id_, error, *, directory=None):
        """Retain failed source health independently of changing run-link IDs."""
        self._job_verification_errors[id_] = str(error)
        path = (
            Path(directory).expanduser().resolve() if directory is not None else self._run_verification_directory(id_)
        )
        if path is not None:
            self._run_verification_paths[id_] = path
            self._source_health.mark_failed(path, error)

    def _run_verification_error(self, source):
        """Every alias of a known failed source shares the same evidence health."""
        errors = getattr(self, "_job_verification_errors", {})
        direct = errors.get(source, "") if not hasattr(source, "directory") else ""
        path = self._run_verification_directory(source)
        if direct:
            if path is not None:
                self._run_verification_paths[source] = path
                self._source_health.mark_failed(path, direct)
            return direct
        if path is None:
            return ""
        error = self._source_health.error(path)
        if not error:
            for id_, detail in errors.items():
                if self._run_verification_directory(id_) == path:
                    self._run_verification_paths[id_] = path
                    self._source_health.mark_failed(path, detail)
                    error = detail
                    break
        return error

    def _clear_run_verification_error(self, source, *, id_=None):
        """Only a successful explicit load clears health for its verified directory."""
        path = self._run_verification_directory(source)
        aliases = [
            key
            for key in self._job_verification_errors
            if path is not None and self._run_verification_directory(key) == path
        ]
        if id_ is not None:
            aliases.append(id_)
            if path is not None:
                self._run_verification_paths[id_] = path
        for alias in aliases:
            self._job_verification_errors.pop(alias, None)
        if path is not None:
            self._source_health.mark_verified(source if hasattr(source, "directory") else path)

    def _view_results_id(self):
        id_ = self._execution_job_id()
        if id_ is not None:
            state = self.states.get(id_, {})
            actions = run_actions(
                self.stage_selector.currentData(),
                state=state,
                record=self.runs.get(id_),
                source_error=self._run_verification_error(id_),
            )
            return id_ if actions.can_view_results else None
        actions = run_actions(
            self.stage_selector.currentData(),
            record=self.runs.get(self._selected_run),
            source_error=self._run_verification_error(self._selected_run),
        )
        return self._selected_run if actions.can_view_results else None

    def _refresh_computation_indicator(self, *_args):
        if not hasattr(self, "computation_indicator"):
            return
        working = []
        local_progress = getattr(self, "_local_progress", {})
        for active, label in (
            (self._submitting, "Submitting study"),
            (self._loading_job_id is not None, "Verifying completed results"),
            (self._geometry_export_task, "Exporting conductor geometry"),
            (self._artifact_export_task, "Exporting saved artifacts"),
            (self._design_study_task, self._design_study_task_label),
            (self._saved_settings_task is not None, "Verifying saved settings"),
            (getattr(self.viewport, "_preview_worker", None) is not None, "Preparing model preview"),
        ):
            if active:
                working.append(local_progress.get(getattr(active, "signals", None), label))
        working.extend(local_progress.get(key, task[2]) for key, task in getattr(self, "_gradient_tasks", {}).items())
        labels = getattr(self, "_research_task_labels", {})
        working.extend(
            local_progress.get(key, labels.get(key, key)) for key in getattr(self, "_research_active_tasks", ())
        )
        self.computation_indicator.message.setText(self.statusBar().currentMessage())
        self.computation_indicator.set_activity(
            self._last_job_states, working=working, unavailable=self._last_runtime_error
        )

    def _update_execution_feedback(self):
        self._refresh_computation_indicator()
        if hasattr(self, "field_analysis_action"):
            from ..application.field_analysis import field_analysis_available

            record = self.runs.get(self._selected_run)
            verified = record is not None and not self._run_verification_error(record)
            cached = getattr(self, "_field_analysis_capability", None)
            if cached is None or cached[0] is not record:
                cached = (record, bool(record is not None and field_analysis_available(record)))
                self._field_analysis_capability = cached
            available = verified and cached[1]
            self.field_analysis_action.setEnabled(available)
            self.field_analysis_action.setToolTip(
                "Analyze this selected run's saved fields, slice grids, error maps and spherical harmonics."
                if available
                else "Select a verified saved run containing computed field samples."
            )
        if hasattr(self, "research_actions"):
            record = self.runs.get(self._selected_run)
            verified = record is not None and not self._run_verification_error(record)
            for handler, key in (
                ("_research_requirements_review", "requirements_review"),
                ("_research_scene_review", "scene_review"),
            ):
                available = bool(verified and record.metadata.get(key))
                self.research_actions[handler].setEnabled(available)
                self.research_actions[handler].setToolTip(
                    "Review this selected run's verified evidence."
                    if available
                    else "Select a verified run computed with these research inputs."
                )
            available = bool(
                verified
                and (record.metadata.get("dynamic_analysis_review") or record.metadata.get("engineering_review"))
            )
            self.research_actions["_research_dynamic_review"].setEnabled(available)
            spec = self.project.study.get("requirements") if self.project else None
            self.research_actions["_research_robust_comparison"].setEnabled(
                bool(spec and any(p["membership"] == "design" for p in spec["scenarios"]))
            )
        if hasattr(self, "gradient_actions"):
            record = self.runs.get(self._selected_run)
            verified = record is not None and not self._run_verification_error(record)
            self.gradient_actions["_gradient_circuit_dialog"].setEnabled(
                bool(verified and record.metadata.get("winding_steps"))
            )
            self.gradient_actions["_gradient_measured_dialog"].setEnabled(
                bool(
                    verified
                    and (
                        record.metadata.get("winding_steps")
                        or (record.metadata.get("wire_fields") or {}).get("routed", {}).get("available")
                    )
                )
            )
        if not hasattr(self, "execution_label"):
            return
        id_ = self._execution_job_id()
        state = self.states.get(id_, {})
        selected_state = self.states.get(self._selected_job_id(), {})
        if hasattr(self, "cancel_action"):
            resume_plan = self._selected_resume_plan() if self.project is not None else ReusePlan((), (), ())
            actions = run_actions(
                self.stage_selector.currentData(),
                state=selected_state,
                reuse=resume_plan,
                source_error=self._run_verification_error(
                    self._selected_job_id()
                    if selected_state.get("status") in STOPPED_JOB_STATES
                    else self._selected_run
                ),
                submitting=self._submitting,
                runtime_available=self.jobs is not None and not self._last_runtime_error,
            )
            resume = actions.resume_stage
            can_cancel, can_resume = actions.can_cancel, actions.can_resume
            # Qt treats hidden actions as unavailable. Derive visibility from
            # job state so an action can reappear when its operation is valid.
            self.cancel_action.setVisible(can_cancel)
            self.cancel_action.setEnabled(can_cancel)
            self.resume_action.setVisible(
                self.project is not None and (self._selected_run is not None or bool(selected_state))
            )
            self.resume_action.setEnabled(can_resume)
            self.resume_action.setText(f"Resume from {resume.value}" if resume is not None else "Resume selected run")
            self.resume_action.setToolTip(
                f"Continue using the verified {resume.value} checkpoint; worker compatibility is checked again."
                if resume is not None
                else (
                    resume_plan.diagnostics[0]
                    if resume_plan.diagnostics
                    else "No compatible verified checkpoint is available."
                )
            )
        if hasattr(self, "compute_button"):
            self.compute_button.setEnabled(self.project is not None and not self._submitting)
            self.compute_action.setEnabled(self.project is not None and not self._submitting)
        results_id = self._view_results_id()
        self.view_results_button.setEnabled(
            self.project is not None
            and results_id is not None
            and self._loading_job_id is None
            and not self._submitting
        )
        self.execution_progress.setVisible(bool(state))
        self.execution_elapsed.setText(f"Elapsed {self._job_elapsed(state)}" if state else "")
        progress = self._job_progress(state)
        if self._submitting:
            text = "Submitting immutable model revision…"
        elif self._loading_job_id is not None:
            text = "Execution completed · verifying saved evidence…"
        elif self._last_runtime_error:
            text = "Job runtime unavailable · see Messages"
        elif self._run_verification_error(id_):
            text = "Execution completed · evidence verification failed · see Messages"
        else:
            status = state.get("status")
            text = {
                "queued": "Queued · waiting for an available worker",
                "running": f"Computing · {state.get('stage') or 'worker running'}",
                "cancelling": "Cancelling · waiting for the worker to stop",
                "completed": "Execution completed · review engineering checks",
                "failed": f"Execution failed · {state.get('error') or state.get('stage') or 'see Messages'}",
                "cancelled": "Cancelled · no completed result",
                "interrupted": "Interrupted · compute again or resume a verified checkpoint",
            }.get(
                status,
                "Verified saved run · review engineering checks"
                if results_id
                else "Ready to compute"
                if self.project is not None
                else "No project open",
            )
        name = state.get("name", "")
        revision = state.get("project_revision")
        prefix = f"{name} · " if name else ""
        if revision is not None:
            prefix += f"revision {revision} · "
        if state:
            origin = {"current": "This session", "previous": "Earlier session", "other": "Other session"}[
                self._job_origin(id_)
            ]
            submitted = format_job_timestamp(state.get("created_at"))
            prefix += origin + (f" · submitted {submitted.text}" if submitted.timestamp is not None else "") + " · "
        self.execution_label.setText(prefix + text)
        self.execution_label.setToolTip(
            self._run_verification_error(id_) or state.get("error") or state.get("stage") or text
        )
        status = state.get("status")
        busy = (
            self._submitting
            or self._loading_job_id is not None
            or (
                not self._last_runtime_error
                and status in {"running", "cancelling"}
                and (progress is None or status == "cancelling")
            )
        )
        self.execution_progress.setRange(0, 0 if busy else 100)
        self.execution_progress.setFormat(
            "Last reported: %p%"
            if self._last_runtime_error and progress is not None
            else "Unavailable"
            if self._last_runtime_error
            else "Queued"
            if status == "queued"
            else "%p%"
            if progress is not None
            else "Unreported"
        )
        if not busy:
            self.execution_progress.setValue(progress if progress is not None else 0)
        apply_tone(
            self.execution_label,
            "error"
            if status == "failed" or self._run_verification_error(id_)
            else "warning"
            if self._last_runtime_error or status in {"cancelling", "cancelled", "interrupted"}
            else "info"
            if state or self._submitting or self._loading_job_id is not None
            else "neutral",
        )
        self._refresh_guidance()

    def _refresh_modeling_notes(self, report):
        warnings = tuple(report.warnings)
        if warnings != self.modeling_notes_panel.issues:
            self.modeling_notes_panel.set_issues(warnings)
        count = len(warnings)
        self.modeling_notes_button.setText(f"{count} modeling {'note' if count == 1 else 'notes'}")
        self.modeling_notes_button.setToolTip("\n".join(f"{issue.title}: {issue.message}" for issue in warnings))
        self.modeling_notes_button.setVisible(bool(warnings))
        self.modeling_notes_window.setWindowTitle(
            f"Modeling notes — {self.project.name} · revision {self.project.revision}"
            if self.project
            else "Modeling notes"
        )

    def _trigger_guidance_action(self):
        if self._guidance_state is not None:
            self._guidance_action(self._guidance_state.action_key)

    def _refresh_guidance(self):
        if not hasattr(self, "design_overview") or self.project is None:
            return
        # Readiness is advisory and cached for the requested study boundary.
        # Compute always performs its own fresh authoritative check.
        stage = self.project.study["pipeline_stage"]
        key = (self.project.fingerprint, stage)
        if key != self._guidance_readiness_key:
            self._guidance_readiness = check_readiness(self.project, pipeline_stage=stage)
            self._guidance_readiness_key = key
        self._refresh_modeling_notes(self._guidance_readiness)
        record = self.runs.get(self._selected_run)
        active = next(
            (
                state
                for state in reversed(tuple(self.states.values()))
                if state.get("status") in {"queued", "running", "cancelling"}
                and state.get("project_id") in {None, self.project.id}
            ),
            None,
        )
        state = assess_guidance(
            self.project,
            record,
            issues=getattr(getattr(self, "problems_panel", None), "issues", ()),
            readiness=self._guidance_readiness,
            active_job=active,
            pending=self._has_input_drafts(),
        )
        if self._last_runtime_error and state.action_key == "show_job":
            state = replace(
                state,
                title="Computation activity unavailable",
                detail=self._last_runtime_error
                + " Last recorded job state is retained; activity resumes when the runtime reconnects.",
                action_label="View last known job",
                tone="warning",
            )
        if self._model_change_note and state.action_key == "recompute":
            state = replace(state, detail=self._model_change_note + " " + state.detail)
        operation = dict((key, label) for label, key in STAGES)[stage]
        if state.action_key in {"compute_full", "recompute"}:
            state = replace(
                state,
                action_label="Compute full design" if stage == "evaluated" else operation,
                detail=(self._model_change_note + " " if state.action_key == "recompute" else "")
                + f"Compute through {operation.lower()}. Settings determine the requested outputs; saved runs remain immutable.",
            )
        draft_note = self._expert_draft_note()
        diagnostic = self._run_verification_error(self._selected_run)
        health_key = (id(record), diagnostic)
        if health_key != getattr(self, "_problems_health_key", None):
            self.problems_panel.set_run(record, diagnostic=diagnostic)
            self._problems_health_key = health_key
            self._refresh_history_panel()
            panel = getattr(self, "geometry_export_panel", None)
            if self._workflow_page() == "export" and panel is not None and panel._record is record:
                panel._diagnostic = diagnostic
                if diagnostic:
                    panel.status.setText(diagnostic)
                    apply_tone(panel.status, "error")
                panel.set_context(self.project, assembly=self.assembly, pending=self._has_input_drafts())
                panel._selection_changed()
        if diagnostic:
            state = replace(
                state,
                title="Saved evidence verification failed",
                detail=diagnostic,
                action_label="Inspect verification failure",
                action_key="inspect_verification",
                tone="error",
            )
        if draft_note:
            state = replace(
                state,
                title=draft_note,
                detail="Apply or discard the retained Advanced draft. Resolve changes to its opening model before computing.",
                action_label="Resolve Advanced draft",
                action_key="resolve_expert",
                tone="warning",
            )
        self.design_status_label.setText(state.title)
        self.design_status_label.setToolTip(state.detail)
        apply_tone(self.design_status_label, state.tone)
        self._guidance_state = state
        self.guidance_action_button.setText(state.action_label)
        self.guidance_action_button.setToolTip(state.detail)
        self.guidance_action_button.setAccessibleDescription(state.detail)
        self.guidance_action_button.setVisible(
            bool(state.action_key) and state.action_key not in {"compute_full", "recompute"}
        )
        outputs = ["Mesh", "Target samples"]
        boundary = [key for _label, key in STAGES].index(stage)
        if boundary >= 1:
            outputs += ["Sheet", "Independent check"]
        if boundary >= 2:
            outputs += ["Contours"]
        if boundary >= 3:
            config = self.project.config
            if not config.skip_postprocessing and not config.contour_only_sweep_mode:
                outputs += ["Connected winding"]
            if not config.skip_sweep and (not config.skip_postprocessing or config.contour_only_sweep_mode):
                outputs += ["Conductor solids"]
        if boundary >= 4:
            outputs += ["Winding fields"]
            if self.fabricate:
                outputs += ["Fabrication review"]
        self.output_scope_label.setText("Requested: " + "  ›  ".join(outputs))
        self.output_scope_label.setToolTip(self._study_scope(stage, self.project.config))
        self._refresh_design_progress()
        if hasattr(self, "compute_button"):
            self.compute_button.setText(
                {
                    "prepared": "Prepare model",
                    "sheet": "Optimize sheet",
                    "contours": "Generate contours",
                    "routed": "Connect / sweep",
                    "evaluated": "Compute full design",
                }[stage]
            )
            self.compute_button.setToolTip(
                self._study_scope(stage, self.project.config) + "\nCtrl+Enter · pending settings are validated first."
            )
            if draft_note:
                self.compute_button.setText("Resolve Advanced draft")
                self.compute_button.setToolTip(draft_note + ". Apply or discard that draft before computing.")
            self._adapt_toolbars()
        self.design_overview.set_state(state, resolved_model_summary(self.project))
        self.design_brief.set_project(self.project, pending=self._has_input_drafts())
        evidence_key = (
            id(record),
            self.project.id,
            self.project.revision,
            key,
            self._has_input_drafts(),
            diagnostic,
            id(self.problems_panel.issues),
        )
        if evidence_key != self._acceptance_refresh_key:
            self.acceptance_panel.set_record(
                record,
                project=self.project,
                issues=self.problems_panel.issues,
                pending=self._has_input_drafts(),
                diagnostic=diagnostic,
            )
            self._acceptance_refresh_key = evidence_key
        action_style = (
            PRIMARY_ACTION_STYLE
            if state.action_key in {"compute_full", "continue_full", "recompute", "review_fabrication"}
            else ""
        )
        if self.design_overview.action_button.styleSheet() != action_style:
            self.design_overview.action_button.setStyleSheet(action_style)
        action_icon = {
            "correct_input": "constraint",
            "compute_full": "compute-study",
            "continue_full": "compute-study",
            "recompute": "compute-study",
            "review_fabrication": "independent-validation",
            "show_job": "run-history",
            "inspect_problems": "collision",
            "export_geometry": "export-conductor",
        }.get(state.action_key, "information")
        self.design_overview.action_button.setIcon(
            icon(action_icon, foreground="primary" if action_style else "default")
        )
        blocking_nodes = {issue.node for issue in self._guidance_readiness.blocking}
        node_icons = dict(NODE_ICONS)
        support_editor = next(
            (entry for entry in SUPPORT_EDITORS if entry.source == self.project.config.coil_mesh_file), None
        )
        node_icons["geometry"] = support_editor.icon_name if support_editor else "mesh"
        if Path(self.project.config.coil_mesh_file).suffix.lower() in {".stl", ".npz"}:
            node_icons["geometry"] = "support-import"
        node_icons["winding"] = WINDING_ICONS[WINDING_INTENTS[winding_intent(self.project.config)]]
        node_icons["target"] = _target_icon(self.project.config)
        for node, item in self.tree_items.items():
            item.setIcon(0, icon(node_icons[node], badge="failed" if node in blocking_nodes else None))
        self._refresh_view_context()

    def _open_overview_section(self, key):
        self.select_node(
            {"support": "geometry", "conductor": "material", "solve": "optimization", "check": "results"}.get(key, key)
        )
        self._show_dock("settings_dock")
        if key == "check":
            self.navigate_workflow("review")

    def _guidance_action(self, key):
        if key == "resolve_expert":
            self._advanced_dialog()
        elif key == "inspect_verification":
            self._error(self._run_verification_error(self._selected_run) or "Saved evidence failed verification.")
        elif key == "correct_input":
            if self._draft_dirty and not self.apply_settings():
                self._report_compute_draft_error()
                return
            if self._guidance_readiness and self._guidance_readiness.blocking:
                self.readiness_panel.set_issues(self._guidance_readiness.blocking)
                self._open_readiness_issue(self._guidance_readiness.blocking[0])
        elif key in {"compute_full", "continue_full", "recompute", "review_fabrication"}:
            if not self._check_expert_draft():
                return
            if not self.apply_settings():
                self._report_compute_draft_error()
                return
            if key == "review_fabrication":
                if not self.apply_study({"pipeline_stage": "evaluated", "fabricate": True}):
                    return
            elif key == "continue_full":
                record = self.runs.get(self._selected_run)
                stage = record.metadata.get("pipeline_stage") if record is not None else None
                if stage not in CONTINUATION_STAGES or record.metadata.get("project_id") != self.project.id:
                    self._error("Select an owned completed study boundary to continue.")
                    return
                if record.metadata.get("project_fingerprint") != self.project.fingerprint:
                    self._error("The model changed; recompute the selected boundary before continuing this result.")
                    return
                if not self.apply_study({"pipeline_stage": CONTINUATION_STAGES[stage][0]}):
                    return
            if key != "review_fabrication" or self.project.study["fabricate"]:
                self.compute()
        elif key == "show_job":
            id_ = self._guidance_state.job_id if self._guidance_state is not None else None
            if id_ is not None:
                self._focus_job(id_)
            else:
                self._open_evidence(self.jobs_table)
        elif key == "inspect_problems":
            self.navigate_workflow("review")
            self._open_evidence(self.problems_panel)
            witness = next((issue for issue in self.problems_panel.issues if issue.locatable), None)
            if witness is not None:
                self.problems_panel.table.selectRow(self.problems_panel.issues.index(witness))
                self._locate_problem(witness)
        elif key == "export_geometry":
            self.navigate_workflow("export")

    def export_selected_geometry(self, directory, names=None, *, unit="mm", include_project=True):
        """Programmatic equivalent of the GUI export using the saved run geometry."""
        from ..application.cad import export_geometry

        if self._selected_run not in self.runs:
            raise ValueError("Select a saved run before exporting conductor geometry.")
        if self._run_verification_error(self._selected_run):
            raise ValueError("Saved evidence failed verification; reopen and verify the run before exporting.")
        if include_project and not self._apply_document_inputs():
            raise ValueError("Correct pending settings before exporting the project and geometry.")
        record = self.runs[self._selected_run]
        project = self._export_companion(record) if include_project else None
        return export_geometry(record, directory, names, unit=unit, project=project)

    def _owns_export_run(self, record):
        owner = record.metadata.get("project_id", (record.metadata.get("job") or {}).get("project_id")) or None
        return owner is None or owner in {channel.project.id for channel in self.assembly.channels}

    def _export_companion(self, record):
        if not self._owns_export_run(record):
            raise ValueError(
                "Open this run's owning project to include its inputs and completed results, or export its CAD files alone."
            )
        return self._portable_assembly()

    def _geometry_export_dialog(self, names, unit, include_project):
        if self._geometry_export_task is not None:
            self.statusBar().showMessage("Geometry export is already running", 5000)
            return
        source_id = self._selected_run
        record = self.runs.get(source_id)
        key = export_record_key(record)
        panel = getattr(self, "geometry_export_panel", None) if self._workflow_page() == "export" else None
        feedback = (
            panel.state
            if panel is not None and panel.record_key == key
            else self._geometry_export_states.setdefault(key, GeometryExportState())
        )
        names = tuple(names)
        feedback.names, feedback.unit, feedback.include_project = names, unit, include_project
        name = re.sub(r"[^A-Za-z0-9_.-]+", "-", self.project.name).strip("-") or "coilgen"
        directory, _filter = QFileDialog.getSaveFileName(self, "Create a new CAD export folder", name + "-cad")
        if not directory:
            return
        try:
            if include_project and not self._apply_document_inputs():
                return
            if record is None:
                raise ValueError("Select a saved run before exporting.")
            if self._run_verification_error(record):
                raise ValueError("Saved evidence failed verification; reopen and verify the run before exporting.")
            project = self._export_companion(record) if include_project else None
            task = GeometryExportTask(record, directory, names, unit, project)
            task.feedback_state, task.source_id = feedback, source_id
            task.signals.finished.connect(self._geometry_export_finished)
            if hasattr(task.signals, "progress"):
                task.signals.progress.connect(self._background_work_progress)
            self._geometry_export_task = task
            feedback.receipt.directory = Path(directory).expanduser().resolve()
            feedback.receipt.text = "Writing verified saved geometry and provenance…"
            feedback.receipt.error, feedback.receipt.complete = "", False
            panel = getattr(self, "geometry_export_panel", None) if self._workflow_page() == "export" else None
            if panel is not None and panel.record_key == key:
                panel.set_exporting(True)
                panel.receipt.refresh()
            self.statusBar().showMessage("Writing verified geometry and provenance…")
            QThreadPool.globalInstance().start(task)
        except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
            feedback.receipt.directory = Path(directory).expanduser().resolve()
            feedback.receipt.text, feedback.receipt.error, feedback.receipt.complete = "", str(exc), False
            panel = getattr(self, "geometry_export_panel", None) if self._workflow_page() == "export" else None
            if panel is not None and panel.record_key == key:
                panel.receipt.refresh()
            self.messages.appendPlainText(f"Geometry export failed: {exc}")
            self.statusBar().showMessage(f"Geometry export failed: {exc}", 15000)

    def _geometry_export_finished(self, package, error):
        task = self._geometry_export_task
        self._geometry_export_task = None
        getattr(self, "_local_progress", {}).pop(getattr(task, "signals", None), None)
        if task is None:
            return
        feedback = task.feedback_state
        source_name = task.record.metadata.get("name", task.record.directory.name)
        if error:
            if task.verification_error:
                owner_id = (
                    task.source_id
                    if export_record_key(self.runs.get(task.source_id)) == task.record_key
                    else str(task.record.directory)
                )
                self._mark_run_verification_error(owner_id, task.verification_error, directory=task.record.directory)
                self._update_result_state()
            feedback.receipt.directory = Path(task.directory).expanduser().resolve()
            feedback.receipt.text, feedback.receipt.error, feedback.receipt.complete = "", error, False
            message = f"Geometry export for {source_name} failed: {error}"
        else:
            feedback.receipt.directory = package.directory
            feedback.receipt.text = geometry_export_description(package)
            feedback.receipt.error, feedback.receipt.complete = "", True
            message = f"Geometry export for {source_name} saved: {package.directory}"
        self.messages.appendPlainText(message)
        self.statusBar().showMessage(message, 15000)
        panel = getattr(self, "geometry_export_panel", None) if self._workflow_page() == "export" else None
        if panel is not None:
            panel.set_exporting(False)
            if panel.record_key == task.record_key:
                panel.receipt.refresh()
        self._refresh_computation_indicator()

    def view_results(self, _checked=False):
        id_ = self._view_results_id()
        if id_ is None:
            return False
        if not self.apply_settings():
            self._report_compute_draft_error()
            return False
        try:
            self.load_run(self.runs[id_].directory, id_=id_, expected_record=self.runs[id_])
            self._clear_run_verification_error(self.runs[id_], id_=id_)
            self.navigate_workflow("review")
            self._update_execution_feedback()
            return True
        except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
            self._mark_run_verification_error(id_, exc)
            self._error(f"Run evidence could not be verified: {exc}")
            self._update_execution_feedback()
            return False

    def cancel_selected_job(self):
        id_ = self._selected_job_id()
        if self.jobs is not None and self.states.get(id_, {}).get("status") in {"queued", "running"}:
            try:
                self.jobs.cancel(id_)
                self.poll_jobs()
            except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
                self._error(f"Cannot cancel job: {exc}")

    def load_run(self, path, *, id_=None, select=True, expected_record=None):
        try:
            record = self.service.load_run(path)
            if expected_record is not None:
                old = {key: value for key, value in expected_record.metadata.items() if key != "job"}
                current = {key: value for key, value in record.metadata.items() if key != "job"}
                if result_metadata_digest(old) != result_metadata_digest(current):
                    raise ValueError("Saved run metadata changed after selection. Reopen and verify the run.")
        except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
            self._mark_run_verification_error(id_ or str(Path(path).expanduser().resolve()), exc, directory=path)
            raise
        id_ = id_ or str(record.directory)
        self._clear_run_verification_error(record, id_=id_)
        self.runs[id_] = record
        owner = record.metadata.get("project_id")
        channel = next((channel for channel in self.assembly.channels if channel.project.id == owner), None)
        scope = (owner, channel.id if self._collection_mode else None) if channel is not None else None
        if self.history is not None and scope is not None:
            try:
                entry = self.history.add_run(scope[0], record, channel_id=scope[1])
                self._history_ids[entry.id] = id_
                if scope == self._history_key:
                    self._history_entries = self.history.restore(scope[0], channel_id=scope[1])
                    self._refresh_history_panel()
            except (OSError, ValueError, TypeError, RuntimeError) as exc:
                self.messages.appendPlainText(f"Run history link could not be saved: {exc}")
        if select is None:
            followed = self.live_preview.state
            select = (
                self.live_preview.following
                and self.live_preview.job_id == id_
                and self._live_owner_matches(followed)
                and self._live_owner_matches(record.metadata)
            )
        self._fill_runs()
        if select:
            self.show_run(id_)
        self._refresh_stage_status()
        if select:
            self._save_session()
        return record

    def _fill_runs(self):
        records = [
            (id_, record)
            for id_, record in self.runs.items()
            if record.metadata.get("project_id") in {None, self.project.id} or id_ == self._selected_run
        ]
        self.runs_table.setRowCount(len(records))
        for row, (id_, record) in enumerate(records):
            metadata = record.metadata
            summary = metadata.get("summary", {})
            state = self.states.get(id_, {})
            error = summary.get("validated_relative_field_error")
            verdicts = run_verdicts(metadata)
            values = [
                metadata.get("name", id_[:10]),
                verdicts["sheet"],
                verdicts["sweeps"],
                verdicts["fabrication"],
                self._format(summary.get("power_w")),
                self._format(summary.get("max_current_density_a_per_m")),
                self._format(None if error is None else error * 100),
                state.get("project_revision", metadata.get("project_revision", "—")),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.ItemDataRole.UserRole, id_)
                if column in {1, 2, 3}:
                    item.setForeground(
                        item_foreground(
                            "success" if value == "Passed" else "error" if value == "Failed" else "neutral",
                            self.palette(),
                        )
                    )
                self.runs_table.setItem(row, column, item)
        self.runs_table.resizeColumnsToContents()

    @staticmethod
    def _verdict(value):
        return "Passed" if value is True else "Failed" if value is False else "Unverified"

    @staticmethod
    def _format(value):
        return "—" if value is None else f"{value:.6g}" if isinstance(value, (float, int)) else str(value)

    def _run_selected(self, row, *_args):
        item = self.runs_table.item(row, 0)
        if item:
            self.show_run(item.data(Qt.ItemDataRole.UserRole))

    def _check_selected(self, row, *_args):
        item = self.checks_table.item(row, 0)
        if item:
            self._inspect_selected_saved_settings(item.data(Qt.ItemDataRole.UserRole) or "results")

    def _inspect_selected_saved_settings(self, node):
        record = self.runs.get(self._selected_run)
        if record is not None:
            self._inspect_saved_settings(record, node)

    def show_run(self, id_):
        if self.live_preview.job_id != id_ and self.live_preview.state.get("status") in {
            "queued",
            "running",
            "cancelling",
        }:
            self.live_preview.pause()
        else:
            self.live_preview.stop()
        self._selected_run = id_
        record = self.runs[id_]
        metadata = record.metadata
        self.problems_panel.set_run(record, diagnostic=self._run_verification_error(id_))
        self._problems_health_key = (id(record), self._run_verification_error(id_))
        run_arrays = record.select_arrays(copy=False) if hasattr(record, "select_arrays") else record.arrays
        self.viewport.display_run(run_arrays, metadata=metadata)
        wire_fields = metadata.get("wire_fields", {})
        self.wire_fields_table.setRowCount(len(wire_fields))
        for row, (kind, evidence) in enumerate(wire_fields.items()):
            capability = resolve_dataset(run_arrays, "contour_field" if kind == "contour" else "routed_field")
            available = evidence.get("available") is True or (
                evidence.get("available") is not False and capability.available
            )
            source = "Original contours" if kind == "contour" else "Connected conductor" if kind == "routed" else kind
            values = [
                source if available else source + " · unavailable",
                evidence.get("current_A"),
                "Yes"
                if evidence.get("current_refitted") is True
                else "No"
                if evidence.get("current_refitted") is False
                else "Unverified",
                None if evidence.get("rms_error_T") is None else evidence["rms_error_T"] * 1e6,
                None if evidence.get("relative_rms_error") is None else evidence["relative_rms_error"] * 100,
                None if evidence.get("peak_error_T") is None else evidence["peak_error_T"] * 1e6,
                evidence.get("sample_count"),
                "Design grid · original current"
                if evidence.get("current_refitted") is False
                else "Design grid · stored current",
            ]
            tooltip = f"Geometry: {evidence.get('geometry_source', evidence.get('geometry_sources', 'unverified'))}; current: {evidence.get('current_source', 'unverified')}. Scope: design-grid wire field. This evidence does not claim independent engineering acceptance."
            if not available:
                reason = evidence.get("reason") or capability.explanation or "This wire field was not calculated."
                values = [values[0], *([None] * 6), "Unavailable · " + reason]
                tooltip = reason + "\n" + tooltip
            for column, value in enumerate(values):
                item = QTableWidgetItem(self._format(value))
                item.setToolTip(tooltip)
                if not available:
                    item.setForeground(self.palette().color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text))
                self.wire_fields_table.setItem(row, column, item)
        self.wire_fields_table.resizeColumnsToContents()
        constraints = metadata.get("constraints", [])
        self.checks_table.setRowCount(len(constraints))
        for row, check in enumerate(constraints):
            values = [
                check.get("name", ""),
                self._format(check.get("value")),
                self._format(check.get("limit")),
                self._format(check.get("effective_limit")),
                check.get("unit", ""),
                self._verdict(check.get("passed")),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                name = check.get("name", "").lower()
                node = (
                    "winding"
                    if any(word in name for word in ("bend", "clearance", "spacing", "routing", "fabrication"))
                    else "target"
                    if "field" in name or "error" in name
                    else "optimization"
                )
                item.setData(Qt.ItemDataRole.UserRole, node)
                item.setToolTip(
                    f"Scope: {check.get('scope', metadata.get('summary', {}).get('scope', 'unverified'))}; "
                    f"click to inspect verified saved {node} settings. The applied model is shown separately."
                )
                self.checks_table.setItem(row, column, item)
                if column == 5:
                    item.setForeground(
                        item_foreground(
                            "success"
                            if check.get("passed") is True
                            else "error"
                            if check.get("passed") is False
                            else "neutral",
                            self.palette(),
                        )
                    )
        self.checks_table.resizeColumnsToContents()
        self.messages.setPlainText("\n".join(metadata.get("warnings", [])))
        self._update_result_state()
        if self._current_node and self._current_node.data(0, Qt.ItemDataRole.UserRole) in {"results", "export"}:
            self._node_changed(self._current_node)
        self._update_execution_feedback()

    def _update_result_state(self):
        if self.project is None:
            return
        self._refresh_view_context()
        if self._selected_run is None:
            self.result_state.setText("No run selected · current model geometry")
            self.result_state.setToolTip("Compute a model or open a saved run to inspect engineering evidence.")
            apply_tone(self.result_state, "neutral")
            self._update_execution_feedback()
            return
        record = self.runs[self._selected_run]
        metadata = record.metadata
        verdicts = run_verdicts(metadata)
        state = self.states.get(self._selected_run, {})
        fingerprint = metadata.get(
            "project_fingerprint", state.get("project_fingerprint", metadata.get("job", {}).get("project_fingerprint"))
        )
        model_fingerprint = metadata.get(
            "project_model_fingerprint",
            state.get("project_model_fingerprint", metadata.get("job", {}).get("project_model_fingerprint")),
        )
        owner = metadata.get("project_id") or state.get("project_id") or (metadata.get("job") or {}).get("project_id")
        diagnostic = self._run_verification_error(record)
        if diagnostic:
            status = "Evidence verification failed"
        elif owner and owner != self.project.id:
            status = "Another channel · saved evidence"
        elif not owner:
            status = "Model identity unverified"
        elif fingerprint == self.project.fingerprint:
            status = "Current model/study"
        elif model_fingerprint and model_fingerprint == self.project.model_fingerprint:
            status = "Current numerical model · study/display settings differ"
        else:
            status = "STALE for current model/study" if fingerprint else "Model identity unverified"
        self.result_state.setText(
            f"{status} · immutable run\n"
            f"Sheet checks: {verdicts['sheet']} · Sweep geometry: {verdicts['sweeps']} · Fabrication review: {verdicts['fabrication']}"
        )
        self.result_state.setToolTip(
            "Sheet checks evaluate the optimized surface-current model. Sweep checks evaluate recorded pipeline solids. "
            "Fabrication review checks its own conductor geometry and field; it does not establish construction release."
        )
        apply_tone(
            self.result_state,
            "error"
            if diagnostic or "Failed" in verdicts.values()
            else "warning"
            if status.startswith(("STALE", "Another channel")) or "differ" in status
            else "neutral"
            if status == "Model identity unverified"
            else "info",
        )
        self._update_execution_feedback()

    def _error(self, message, *, open_window=True):
        self.messages.appendPlainText(message)
        self.statusBar().showMessage(message, 15000)
        if open_window:
            self.evidence.setCurrentWidget(self.messages)
            self._show_dock("evidence_dock")

    def _refresh_status_colors(self):
        if not hasattr(self, "execution_label"):
            return
        self._update_result_state()
        self._refresh_stage_status(refresh_reuse=False)
        self._fill_jobs(self._last_job_states)
        for table, columns in ((self.runs_table, (1, 2, 3)), (self.checks_table, (5,))):
            for row in range(table.rowCount()):
                for column in columns:
                    item = table.item(row, column)
                    if item is not None:
                        item.setForeground(
                            item_foreground(
                                "success"
                                if item.text() == "Passed"
                                else "error"
                                if item.text() == "Failed"
                                else "neutral",
                                self.palette(),
                            )
                        )
        if getattr(self, "draft_error_label", None) is not None:
            apply_tone(self.draft_error_label, "error")
        for label in self._field_errors.values():
            apply_tone(label, "error")

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in {QEvent.Type.FontChange, QEvent.Type.ApplicationFontChange, QEvent.Type.StyleChange}:
            if hasattr(self, "compute_button"):
                QTimer.singleShot(0, self, self._adapt_toolbars)
        if event.type() in {QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange}:
            if hasattr(self, "execution_label"):
                QTimer.singleShot(0, self, self._refresh_status_colors)

    def closeEvent(self, event):
        if getattr(self, "_research_active_tasks", None):
            from .array_studies import ArrayStudyDialog

            self._research_close_pending = True
            for dialog in self.findChildren(ArrayStudyDialog):
                dialog.reject()
            for dialog in tuple(getattr(self, "_field_analysis_windows", {}).values()):
                dialog.request_close()
            self.statusBar().showMessage("Cancelling the active research study before closing…")
            event.ignore()
            return
        if not self._can_discard():
            event.ignore()
            return
        self._closing = True
        for dialog in tuple(getattr(self, "_field_analysis_windows", {}).values()):
            dialog.request_close()
        scene_editor = getattr(self, "_scene_editor", None)
        if scene_editor is not None:
            scene_editor.reject()
        self._save_session()
        self.evidence.close_all()
        for dialog in self.findChildren(QDialog):
            dialog.hide()
        self.findChild(QDockWidget, "settings_dock").hide()
        self.timer.stop()
        self.live_preview.close()
        if self._own_jobs and self.jobs is not None:
            self.jobs.close()
        self.viewport.close()
        super().closeEvent(event)
