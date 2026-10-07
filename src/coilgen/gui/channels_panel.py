"""Compact, read-only overview of the configured independent coil channels."""

from collections.abc import Mapping

from PySide6.QtCore import QSize, Qt, Signal, QSignalBlocker
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QFormLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import CoilGenConfig
from ..application.targets import recognize_solid_harmonic_target
from ..application.assembly import ProjectAssembly
from ..application.diagnostics import run_verdicts
from .colors import item_foreground
from .guidance import resolved_model_summary
from .icons import TARGET_MODE_ICONS, icon
from .job_history import format_job_timestamp
from .run_context import run_compatibility, run_timestamp


_STUDY_LABELS = {
    "prepared": "Mesh / target",
    "sheet": "Sheet solve",
    "contours": "Contours",
    "routed": "Routing",
    "evaluated": "Wire evaluation",
}


def _target_icon(config, summary):
    if str(config.target_field_definition_file).lower() != "none":
        return "target-map"
    if config.target_field_component != "Bz":
        return "field-component"
    for axis in ("x", "y", "z"):
        if summary.startswith(f"G{axis} =") and config.target_gradient_strength > 0:
            return TARGET_MODE_ICONS[f"G{axis}"]
    recipe = recognize_solid_harmonic_target(config.field_shape_function)
    if recipe is not None:
        modes = {0: ("Z2", 1), 1: ("XZ", -1), -1: ("YZ", -1), 2: ("X2-Y2", 1), -2: ("XY", 1)}
        canonical = all(getattr(config, key) == value for key, value in recipe.to_config_overrides().items())
        if canonical and recipe.degree == 2 and recipe.order in modes:
            name, sign = modes[recipe.order]
            if sign * recipe.coefficient_T > 0:
                return TARGET_MODE_ICONS[name]
        return "target-harmonic"
    if summary.startswith("Uniform Bz"):
        try:
            if float(config.field_shape_function) > 0:
                return TARGET_MODE_ICONS["Z0"]
        except (TypeError, ValueError):
            pass
        return "target-uniform"
    return "field-component"


def _physical_cells(project, summaries):
    """Use stored values only: this overview never resolves assets or runs physics."""
    config = CoilGenConfig(**project.values)
    support = summaries["support"].split(";")[0]
    if config.coil_mesh_file == "create bi-planary mesh":
        height, width, *rest = config.biplanar_mesh_parameter_list
        support = f"{width * 1000:g} × {height * 1000:g} mm · gap {rest[8] * 1000:g} mm"
    profile = config.cross_sectional_points
    rectangle = len(profile) == 1 and len(profile[0]) == 2 and all(value == 0 for value in profile[0])
    copper = (
        f"{config.conductor_cross_section_width * 1000:g} × {config.conductor_cross_section_height * 1000:g} mm"
        if rectangle
        else "Custom profile"
    )
    target = summaries["target"].split(";")[0].removesuffix(".")
    study = _STUDY_LABELS.get(project.study["pipeline_stage"], project.study["pipeline_stage"])
    if project.study["fabricate"]:
        study += " · strict review"
    return config, (target, support, copper, study)


class ChannelOverview(QDialog):
    """Request parent-owned actions; keep this view free of numerical/file actions."""

    channel_requested = Signal(str)
    compute_requested = Signal()

    def __init__(self, assembly, active_channel_id, parent=None, *, states=(), records=()):
        super().__init__(parent)
        self.setObjectName("channel_overview")
        self.setWindowTitle("Channel overview — CoilGen2")
        self.setWindowIcon(icon("channels"))
        self.assembly = assembly
        self._configuration_windows = {}
        self._configuration_keys = {}
        self._enabled_count = sum(channel.enabled for channel in assembly.channels)
        layout = QVBoxLayout(self)
        self.heading = QLabel(f"{assembly.name} · {len(assembly.channels)} channels · {self._enabled_count} enabled")
        self.heading.setTextFormat(Qt.TextFormat.PlainText)
        self.heading.setWordWrap(True)
        layout.addWidget(self.heading)
        self.table = QTableWidget(len(assembly.channels), 10)
        self.table.setObjectName("channel_overview_table")
        self.table.setAccessibleName("Configured independent coil channels")
        self.table.setHorizontalHeaderLabels(
            [
                "Mode",
                "Enabled",
                "Target",
                "Support",
                "Copper",
                "Study",
                "Execution",
                "Saved result",
                "Sheet checks",
                "Geometry checks",
            ]
        )
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setWordWrap(False)
        self.table.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.table.setIconSize(QSize(24, 24))
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(34)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.resizeSection(0, 88)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        for column in range(2, 6):
            self.table.setColumnHidden(column, True)
        for column in range(6, 10):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        self._descriptions = (
            "Stored channel name. The pictogram follows the current target, not earlier example metadata.",
            "Whether the parent workbench includes this channel in an independent-channel compute request.",
            "Configured physical target and scaling. Hover for the complete definition, ROI and stored expression.",
            "Configured support dimensions in mm. Stored geometry uses SI metres in the shared world frame.",
            "Conducting profile, separate from insulation and homogenised sheet thickness. Dimensions displayed in mm.",
            "Requested study boundary and fabrication-review settings. Configured requests are not recorded passes.",
            "Recorded execution state is separate from engineering acceptance. Hover for revision and submission date.",
            "Prefer verified evidence matching the current model/study fingerprint. Earlier evidence remains clearly identified.",
            "Saved continuous-sheet acceptance only. Wire metrics retain their separate design-grid/current scope in Results.",
            "Pipeline sweep geometry and strict fabrication review are distinct saved scopes. This is not combined hardware qualification.",
        )
        for column, description in enumerate(self._descriptions):
            self.table.horizontalHeaderItem(column).setToolTip(description)
        self._populate_configuration()
        layout.addWidget(self.table, 1)
        self.next_action = QLabel()
        self.next_action.setObjectName("channel_overview_next_action")
        self.next_action.setTextFormat(Qt.TextFormat.PlainText)
        self.next_action.setWordWrap(True)
        layout.addWidget(self.next_action)
        self.scope = QLabel(
            "Independent channel studies. External circuit feeds, inter-channel clearance/coupling "
            "and combined hardware qualification remain unverified."
        )
        self.scope.setObjectName("channel_overview_scope")
        self.scope.setWordWrap(True)
        layout.addWidget(self.scope)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.configuration_button = QPushButton("Channel configuration…")
        self.configuration_button.setIcon(icon("settings"))
        self.configuration_button.setAutoDefault(False)
        self.configuration_button.clicked.connect(self._show_configuration)
        buttons.addButton(self.configuration_button, QDialogButtonBox.ButtonRole.ActionRole)
        self.open_button = QPushButton("Open selected channel")
        self.open_button.setIcon(icon("channels"))
        self.open_button.setDefault(True)
        self.open_button.clicked.connect(self._open_selected)
        buttons.addButton(self.open_button, QDialogButtonBox.ButtonRole.ActionRole)
        self.compute_button = QPushButton("Compute enabled channels")
        self.compute_button.setIcon(icon("compute-study"))
        self.compute_button.setAutoDefault(False)
        self.compute_button.setEnabled(self._enabled_count > 0)
        self.compute_button.clicked.connect(self._compute_enabled)
        buttons.addButton(self.compute_button, QDialogButtonBox.ButtonRole.ActionRole)
        self.close_button = buttons.button(QDialogButtonBox.StandardButton.Close)
        self.close_button.setIcon(icon("cancel"))
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.cellDoubleClicked.connect(lambda *_: self._open_selected())
        row = next((index for index, channel in enumerate(assembly.channels) if channel.id == active_channel_id), 0)
        if assembly.channels:
            self.table.selectRow(row)
        else:
            self.open_button.setEnabled(False)
        self.refresh_evidence(states=states, records=records)
        self._selection_changed()
        self.resize(1060, min(590, max(290, 180 + 34 * len(assembly.channels))))

    def _populate_configuration(self):
        """Refresh presentation from stored SI values without resolving project assets."""
        self.table.setRowCount(len(self.assembly.channels))
        for row, channel in enumerate(self.assembly.channels):
            summary = resolved_model_summary(channel.project)
            try:
                config, cells = _physical_cells(channel.project, summary)
                mode_icon = _target_icon(config, summary["target"])
                support_detail = summary["support"]
                if config.coil_mesh_file == "create bi-planary mesh":
                    parameters = config.biplanar_mesh_parameter_list
                    support_detail += (
                        f"\nPrimitive normal: {tuple(parameters[4:7])}; primitive centre [m]: {tuple(parameters[7:10])}."
                        f"\nAdditional generated-mesh rotation: {config.generated_mesh_rotation_angle:g} rad"
                        f" about axis {tuple(config.generated_mesh_rotation_axis)} (around the world origin)."
                        f"\nStored SI mesh parameters: {tuple(parameters)}."
                    )
            except (ValueError, TypeError, IndexError, OverflowError):
                cells = tuple(summary[key].split(";")[0] for key in ("target", "support", "conductor", "study"))
                mode_icon = "field-component"
                support_detail = summary["support"]
            values = (channel.name, "Enabled" if channel.enabled else "Disabled", *cells)
            target_detail = f"{summary['target']}\n{summary['roi']}\nStored expression: {channel.project.values['field_shape_function']}"
            details = (
                channel.name,
                self._descriptions[1],
                target_detail,
                support_detail,
                summary["conductor"],
                summary["study"],
            )
            for column, (value, detail) in enumerate(zip(values, details)):
                item = QTableWidgetItem(value)
                item.setToolTip(detail)
                item.setData(Qt.ItemDataRole.AccessibleTextRole, value)
                item.setData(Qt.ItemDataRole.AccessibleDescriptionRole, detail)
                if column == 0:
                    item.setIcon(icon(mode_icon))
                    item.setData(Qt.ItemDataRole.UserRole, channel.id)
                self.table.setItem(row, column, item)

    @staticmethod
    def _configuration_key(channel):
        return channel.name, channel.project.id, channel.project.revision, channel.project.fingerprint

    def set_snapshot(self, assembly, active_channel_id=None, *, states=(), records=()):
        """Update an existing overview from caller-owned model and evidence snapshots.

        Omit active_channel_id when polling to preserve the designer's selection.
        A configuration window is an applied-revision snapshot: input changes or
        channel removal close it, while job-only updates keep it open.
        """
        if not isinstance(assembly, ProjectAssembly):
            raise TypeError("Channel overview requires a ProjectAssembly snapshot.")
        row = self.table.currentRow()
        selected = active_channel_id or (
            self.assembly.channels[row].id if 0 <= row < len(self.assembly.channels) else None
        )
        channels = {channel.id: channel for channel in assembly.channels}
        for channel_id, window in tuple(self._configuration_windows.items()):
            channel = channels.get(channel_id)
            if channel is None or self._configuration_keys[channel_id] != self._configuration_key(channel):
                window.close()
                window.deleteLater()
                self._configuration_windows.pop(channel_id)
                self._configuration_keys.pop(channel_id)
        self.assembly = assembly
        self._enabled_count = sum(channel.enabled for channel in assembly.channels)
        self.heading.setText(f"{assembly.name} · {len(assembly.channels)} channels · {self._enabled_count} enabled")
        with QSignalBlocker(self.table):
            self.table.clearSelection()
            self._populate_configuration()
            if assembly.channels:
                row = next((index for index, channel in enumerate(assembly.channels) if channel.id == selected), 0)
                self.table.selectRow(row)
            else:
                self.table.setCurrentCell(-1, -1)
        self.compute_button.setEnabled(self._enabled_count > 0)
        self.refresh_evidence(states=states, records=records)

    @staticmethod
    def _channel_items(items, channel):
        if isinstance(items, Mapping):
            items = items.get(channel.id, items.get(channel.project.id, ()))
        if isinstance(items, Mapping) or hasattr(items, "metadata") or hasattr(items, "run"):
            return (items,)
        return tuple(items or ())

    def refresh_evidence(self, *, states=(), records=()):
        """Accept already loaded snapshots; never load runs, discover jobs, or evaluate physics."""
        self._next_actions = {}
        for row, channel in enumerate(self.assembly.channels):
            jobs = [
                state
                for state in self._channel_items(states, channel)
                if isinstance(state, Mapping) and state.get("project_id") == channel.project.id
            ]
            job = max(
                jobs,
                key=lambda state: (
                    state.get("status") in {"queued", "running", "cancelling"},
                    format_job_timestamp(state.get("created_at")).timestamp or float("-inf"),
                ),
                default=None,
            )
            candidates = []
            for value in self._channel_items(records, channel):
                if hasattr(value, "run"):
                    if not value.verified:
                        continue
                    value = value.run
                metadata = getattr(value, "metadata", None)
                if (
                    isinstance(metadata, Mapping)
                    and (metadata.get("project_id") or (metadata.get("job") or {}).get("project_id"))
                    == channel.project.id
                ):
                    candidates.append(value)
            matching = [
                record for record in candidates if run_compatibility(record.metadata, channel.project)[0] == "current"
            ]
            pool = matching or candidates
            record = max(
                enumerate(pool),
                key=lambda pair: (run_timestamp(pair[1].metadata).timestamp or float("-inf"), pair[0]),
                default=(None, None),
            )[1]
            active = job is not None and job.get("status") in {"queued", "running", "cancelling"}
            execution = (
                str(job.get("status", "unknown")).capitalize()
                if job
                else "Saved result only"
                if record
                else "No job record"
            )
            if job:
                execution += f" · rev {job.get('project_revision', '?')}"
            execution_detail = (
                f"{job.get('stage', '')}\nSubmitted: {format_job_timestamp(job.get('created_at')).text}\n{run_compatibility(job, channel.project)[1]}"
                if job
                else "No job-state snapshot supplied for this channel. Saved results alone do not establish present execution."
            )
            compatibility, context = (
                run_compatibility(record.metadata, channel.project) if record else ("unknown", "No saved result")
            )
            verdicts = (
                run_verdicts(record.metadata)
                if record
                else {key: "Unverified" for key in ("sheet", "sweeps", "fabrication")}
            )
            saved = (
                f"{'Matching' if compatibility == 'current' else 'Earlier' if compatibility == 'earlier' else 'Unverified'} · rev {record.metadata.get('project_revision', '?')}"
                if record
                else "No saved result"
            )
            record_detail = (
                f"{record.metadata.get('name', 'Saved run')}\n{context}\nSubmitted: {run_timestamp(record.metadata).text}\nPipeline: {record.metadata.get('pipeline_stage', 'unrecorded')}\n{run_timestamp(record.metadata).tooltip}"
                if record
                else "Compute this channel to save model-linked evidence."
            )
            values = (
                execution,
                saved,
                verdicts["sheet"],
                f"Sweep {verdicts['sweeps']} · Review {verdicts['fabrication']}",
            )
            details = (
                execution_detail,
                record_detail,
                record_detail + "\nContinuous sheet only; original and refitted wire fields are separate evidence.",
                record_detail
                + "\nPipeline sweep screen and strict fabrication review remain separate from field acceptance and inter-channel checks.",
            )
            for column, (value, detail) in enumerate(zip(values, details), 6):
                item = QTableWidgetItem(value)
                item.setToolTip(detail)
                item.setData(Qt.ItemDataRole.AccessibleTextRole, value)
                item.setData(Qt.ItemDataRole.AccessibleDescriptionRole, detail)
                if column == 7:
                    item.setIcon(
                        icon(
                            "run-history",
                            badge="evaluated"
                            if compatibility == "current"
                            else "stale"
                            if compatibility == "earlier"
                            else "unverified",
                        )
                    )
                if column >= 8:
                    failed = "Failed" in value
                    passed = column == 8 and value == "Passed"
                    item.setForeground(
                        item_foreground("error" if failed else "success" if passed else "neutral", self.table.palette())
                    )
                self.table.setItem(row, column, item)
            self._next_actions[channel.id] = (
                "Wait for the active study or open its jobs window."
                if active
                else "Compute this channel's current model/study."
                if compatibility != "current"
                else "Open Results to inspect recorded failures and their witnesses."
                if any(value == "Failed" for value in verdicts.values())
                else "Open Results to review evidence gaps and the saved evaluation scopes."
            )
        self._selection_changed()

    def _selection_changed(self):
        row = self.table.currentRow()
        valid = 0 <= row < len(self.assembly.channels)
        self.open_button.setEnabled(valid)
        self.configuration_button.setEnabled(valid)
        channel = self.assembly.channels[row] if valid else None
        self.next_action.setText(
            f"{channel.name}: {getattr(self, '_next_actions', {}).get(channel.id, 'Open the channel to review its model.')}"
            if channel
            else "Select a channel."
        )

    def _show_configuration(self):
        row = self.table.currentRow()
        if row < 0:
            return
        channel = self.assembly.channels[row]
        existing = self._configuration_windows.get(channel.id)
        if existing is not None:
            existing.show()
            existing.raise_()
            return
        window = QDialog(self)
        window.setObjectName("channel_configuration_window")
        window.setWindowTitle(f"{channel.name} · applied revision {channel.project.revision} — Configuration")
        window.setWindowIcon(icon("settings"))
        layout = QVBoxLayout(window)
        scroll = QScrollArea(window)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget(scroll)
        form = QFormLayout(body)
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)
        for column, title in ((2, "Target and region"), (3, "Support"), (4, "Conductor"), (5, "Requested study")):
            text = QLabel(self.table.item(row, column).toolTip())
            text.setTextFormat(Qt.TextFormat.PlainText)
            text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            text.setWordWrap(True)
            form.addRow(title, text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(window.close)
        layout.addWidget(buttons)
        self._configuration_windows[channel.id] = window
        self._configuration_keys[channel.id] = self._configuration_key(channel)
        window.resize(680, 420)
        window.show()

    def _open_selected(self):
        item = self.table.item(self.table.currentRow(), 0)
        if item is not None:
            selected = item.data(Qt.ItemDataRole.UserRole)
            self.accept()
            self.channel_requested.emit(selected)

    def _compute_enabled(self):
        if self._enabled_count:
            self.accept()
            self.compute_requested.emit()
