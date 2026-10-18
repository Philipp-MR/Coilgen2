"""Named verified run links and scope-aware comparison for the Qt workbench."""

from __future__ import annotations

import json

from PySide6.QtCore import QEvent, QTimer, Qt, Signal, QSignalBlocker, QSize
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QScrollArea,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..application.comparison import compare_runs
from ..application.design_studies import DesignStudyState
from ..application.diagnostics import run_verdicts
from ..research.storage import result_metadata_digest
from .colors import apply_tone, item_foreground
from .editors import SUPPORT_EDITORS
from .icons import icon
from .run_context import run_compatibility, run_timestamp


def _text(value):
    if value is None:
        return "Unavailable"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def _item(value, tooltip=""):
    text = _text(value)
    item = QTableWidgetItem(text if len(text) <= 180 else text[:177] + "…")
    item.setToolTip(tooltip or (text if len(text) <= 2048 else text[:2048] + "…"))
    return item


def _table(headers, name):
    table = QTableWidget(0, len(headers))
    table.setObjectName(name)
    table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    return table


def _record_key(record):
    metadata = {key: value for key, value in record.metadata.items() if key != "job"}
    return metadata.get("project_id"), metadata.get("request_sha256"), result_metadata_digest(metadata)


def _parameter(field, value):
    """Display common engineering inputs in useful units; retain exact keys."""
    if isinstance(value, (list, tuple)):
        editor = next((item for item in SUPPORT_EDITORS if field == f"config.{item.parameter_field}"), None)
        if editor and len(value) > max(item.index for item in editor.fields):
            try:
                parts = []
                for item in editor.fields:
                    number = editor.display_value(item, value)
                    parts.append(f"{item.label}: {number:.6g}{item.suffix}")
                return f"{editor.label} support", "; ".join(parts)
            except (TypeError, ValueError, OverflowError):
                # Readable legacy evidence can lack a valid editor contract.
                # Preserve the recorded vector rather than inventing units.
                return field, value
    known = {
        "config.levels": ("Contour level count", 1, ""),
        "config.target_region_radius": ("ROI radius", 1e3, "mm"),
        "config.target_gradient_strength": ("Target gradient", 1e3, "mT/m"),
        "config.conductor_cross_section_width": ("Conductor width", 1e3, "mm"),
        "config.conductor_cross_section_height": ("Conductor height", 1e3, "mm"),
        "config.conductor_thickness": ("Effective sheet thickness", 1e3, "mm"),
        "config.specific_conductivity_conductor": ("Conductor resistivity", 1e9, "nΩ·m"),
        "config.sweep_insulation_thickness": ("Insulation thickness", 1e3, "mm"),
        "config.sweep_min_bend_radius": ("Minimum bend radius", 1e3, "mm"),
        "config.min_wire_distance_m": ("Minimum wire clearance", 1e3, "mm"),
        "config.interconnection_cut_width": ("Interconnection cut width", 1e3, "mm"),
        "config.target_error_tolerance_percent": ("Design field error budget", 1, "%"),
        "config.stream_function_cost_function": ("Sheet objective", 1, ""),
        "config.sf_opt_method": ("Sheet solver", 1, ""),
        "config.coil_mesh_file": ("Support source", 1, ""),
        "config.field_shape_function": ("Target expression", 1, ""),
        "config.target_field_component": ("Target field component", 1, ""),
        "config.b_0_direction": ("B₀ world direction", 1, ""),
        "study.validation_resolution": ("Independent validation resolution", 1, ""),
        "study.pipeline_stage": ("Study boundary", 1, ""),
        "study.design_margin": ("Design / validation budget ratio", 1, ""),
    }
    label, factor, unit = known.get(field, (field, 1, ""))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = f"{value * factor:.6g}" + (f" {unit}" if unit else "")
    return label, value


class HistoryPanel(QWidget):
    selected = Signal(object)  # HistoryEntry
    alias_changed = Signal(str, str)
    compare_requested = Signal(object, object)  # Verified RunRecord pair.
    baseline_requested = Signal(object)  # Selected verified HistoryEntry.
    clear_baseline_requested = Signal()
    variant_requested = Signal(object, str)  # Source HistoryEntry, hypothesis.
    notes_changed = Signal(str, str)  # Design hypothesis, selected run rationale.

    def __init__(self, parent=None):
        super().__init__(parent)
        self.entries = ()
        self.comparison = None
        self.study_state = DesignStudyState()
        self._study_project_id = None
        self._comparison_keys = None
        self._hypothesis_drafts = {}
        self._rationale_drafts = {}
        self._rationale_entries = {}
        self._notes_context = None
        self._current_project = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea(self)
        self.scroll.setObjectName("history_scroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget(self.scroll)
        layout = QVBoxLayout(body)
        self.scroll.setWidget(body)
        outer.addWidget(self.scroll)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        self.diagnostics = QLabel()
        self.diagnostics.setTextFormat(Qt.TextFormat.PlainText)
        self.diagnostics.setObjectName("history_diagnostics")
        self.diagnostics.setWordWrap(True)
        layout.addWidget(self.diagnostics)
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Show runs"))
        self.compatibility_filter = QComboBox()
        self.compatibility_filter.setObjectName("history_compatibility_filter")
        self.compatibility_filter.setAccessibleName("Run model and study compatibility filter")
        for title, key in (
            ("All history", "all"),
            ("Matching model/study", "current"),
            ("Earlier model/study", "earlier"),
            ("Unverified identity", "unknown"),
        ):
            self.compatibility_filter.addItem(title, key)
        self.compatibility_filter.currentIndexChanged.connect(self._filter_runs)
        filter_layout.addWidget(self.compatibility_filter)
        self.filter_count = QLabel()
        self.filter_count.setObjectName("history_filter_count")
        filter_layout.addWidget(self.filter_count)
        filter_layout.addStretch()
        layout.addLayout(filter_layout)
        self.runs_table = _table(
            [
                "Named run",
                "Stage",
                "Validation scope",
                "Verdict",
                "Revision",
                "Independent error [%]",
                "Link",
                "Submitted",
                "Model / study",
            ],
            "history_runs",
        )
        self.runs_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.runs_table.setMinimumHeight(84)
        header = self.runs_table.horizontalHeader()
        header.moveSection(header.visualIndex(7), 1)
        header.moveSection(header.visualIndex(8), 2)
        self.runs_table.horizontalHeaderItem(7).setToolTip(
            "Recorded job submission date; missing execution timestamps remain unrecorded."
        )
        self.runs_table.horizontalHeaderItem(8).setToolTip(
            "Compatibility with current applied inputs and study settings, established by saved full fingerprints."
        )
        self.runs_table.itemSelectionChanged.connect(self._selection_changed)
        layout.addWidget(self.runs_table)
        controls = QHBoxLayout()
        self.alias = QLineEdit()
        self.alias.setObjectName("history_alias")
        self.alias.setMaxLength(120)
        self.alias.setPlaceholderText("Display alias; scientific metadata stays unchanged")
        controls.addWidget(self.alias)
        self.rename_button = QPushButton("Rename display")
        self.rename_button.setObjectName("history_rename")
        self.rename_button.clicked.connect(self._rename_selected)
        controls.addWidget(self.rename_button)
        self.compare_button = QPushButton("Compare two runs")
        self.compare_button.setObjectName("history_compare")
        self.compare_button.clicked.connect(self.compare_selected)
        controls.addWidget(self.compare_button)
        self.decisions_button = QPushButton("Design decisions…")
        self.decisions_button.setObjectName("history_design_decisions")
        self.decisions_button.clicked.connect(lambda: self._open_details(2))
        controls.addWidget(self.decisions_button)
        self.run_details_button = QPushButton("Run details…")
        self.run_details_button.setObjectName("history_run_details")
        self.run_details_button.setIcon(icon("information"))
        self.run_details_button.clicked.connect(self._open_run_details)
        controls.addWidget(self.run_details_button)
        layout.addLayout(controls)
        self.selected_summary = QLabel()
        self.selected_summary.setObjectName("history_selected_summary")
        self.selected_summary.setTextFormat(Qt.TextFormat.PlainText)
        self.selected_summary.setWordWrap(True)
        layout.addWidget(self.selected_summary)
        study_controls = QHBoxLayout()
        self.baseline_label = QLabel("Baseline: choose a verified run and pin it for comparison.")
        self.baseline_label.setObjectName("history_baseline")
        self.baseline_label.setWordWrap(True)
        self.baseline_label.setMaximumHeight(36)
        self.baseline_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.baseline_label)
        self.pin_baseline_button = QPushButton("Pin baseline")
        self.pin_baseline_button.setObjectName("history_pin_baseline")
        self.pin_baseline_button.setToolTip("Keep this verified result as the reference for design experiments.")
        self.pin_baseline_button.clicked.connect(self._pin_baseline)
        study_controls.addWidget(self.pin_baseline_button)
        self.clear_baseline_button = QPushButton("Unpin")
        self.clear_baseline_button.setObjectName("history_unpin_baseline")
        self.clear_baseline_button.clicked.connect(self.clear_baseline_requested.emit)
        study_controls.addWidget(self.clear_baseline_button)
        self.compare_baseline_button = QPushButton("Compare to baseline")
        self.compare_baseline_button.setObjectName("history_compare_baseline")
        self.compare_baseline_button.clicked.connect(self.compare_to_baseline)
        study_controls.addWidget(self.compare_baseline_button)
        self.create_variant_button = QPushButton("Create variant from run")
        self.create_variant_button.setObjectName("history_create_variant")
        self.create_variant_button.setToolTip(
            "Restore this run's physical settings and owned inputs as a new revision. The source result remains immutable."
        )
        self.create_variant_button.clicked.connect(self._create_variant)
        study_controls.addWidget(self.create_variant_button)
        study_controls.addStretch()
        layout.addLayout(study_controls)
        self.details_window = QDialog(self)
        self.details_window.setObjectName("history_details_window")
        self.details_window.setWindowTitle("Run comparison and design decisions — CoilGen2")
        self.details_window.setWindowIcon(icon("compare-designs"))
        self.details_window.resize(980, 620)
        details_layout = QVBoxLayout(self.details_window)
        tabs = QTabWidget()
        self.comparison_tabs = tabs
        self.metrics_table = _table(["Quantity", "A", "B", "Unit", "Contract", "B − A"], "history_metrics")
        self.parameters_table = _table(["Changed setting", "A", "B"], "history_parameters")
        tabs.addTab(self.metrics_table, "Comparison evidence")
        tabs.addTab(self.parameters_table, "Parameter differences")
        notes = QWidget()
        notes_layout = QFormLayout(notes)
        notes_layout.setContentsMargins(6, 6, 6, 6)
        notes_layout.setVerticalSpacing(4)
        self.hypothesis = QPlainTextEdit()
        self.hypothesis.setObjectName("history_hypothesis")
        self.hypothesis.setPlaceholderText("What change do you want to test, and what improvement do you expect?")
        self.hypothesis.setMaximumHeight(46)
        self.rationale = QPlainTextEdit()
        self.rationale.setObjectName("history_rationale")
        self.rationale.setPlaceholderText("Why select, retain or reject the selected run? Include relevant tradeoffs.")
        self.rationale.setMaximumHeight(46)
        notes_layout.addRow("Design hypothesis", self.hypothesis)
        notes_layout.addRow("Selected run decision", self.rationale)
        self.save_notes_button = QPushButton("Save design notes")
        self.save_notes_button.setObjectName("history_save_notes")
        self.save_notes_button.clicked.connect(self._save_notes)
        self.hypothesis.textChanged.connect(self._hypothesis_edited)
        self.rationale.textChanged.connect(self._rationale_edited)
        save_notes = QHBoxLayout()
        save_notes.addWidget(self.save_notes_button)
        self.notes_status = QLabel()
        self.notes_status.setObjectName("history_notes_status")
        self.notes_status.setWordWrap(True)
        save_notes.addWidget(self.notes_status, 1)
        notes_layout.addRow(save_notes)
        tabs.addTab(notes, "Design decisions")
        for button, name in (
            (self.rename_button, "edit"),
            (self.compare_button, "compare-designs"),
            (self.pin_baseline_button, "pin"),
            (self.clear_baseline_button, "unpin"),
            (self.compare_baseline_button, "compare-designs"),
            (self.create_variant_button, "design-variant"),
            (self.save_notes_button, "save"),
            (self.decisions_button, "edit"),
        ):
            button.setIcon(icon(name))
            button.setIconSize(QSize(18, 18))
            button.setAccessibleName(button.text())
        for index, name in enumerate(("compare-designs", "settings", "edit")):
            tabs.setTabIcon(index, icon(name))
        details_layout.addWidget(tabs, 1)
        self.comparison_notes = QLabel(
            "Select two verified runs to compare their original requests and scoped evidence."
        )
        self.comparison_notes.setTextFormat(Qt.TextFormat.PlainText)
        self.comparison_notes.setObjectName("history_comparison_notes")
        self.comparison_notes.setWordWrap(True)
        details_layout.addWidget(self.comparison_notes)
        close_details = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_details.rejected.connect(self.details_window.close)
        details_layout.addWidget(close_details)
        self.run_details_window = QDialog(self)
        self.run_details_window.setObjectName("history_run_details_window")
        self.run_details_window.setWindowTitle("Recorded run details — CoilGen2")
        self.run_details_window.setWindowIcon(icon("run-history"))
        self.run_details_window.resize(680, 460)
        run_layout = QVBoxLayout(self.run_details_window)
        run_scroll = QScrollArea()
        run_scroll.setWidgetResizable(True)
        run_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        run_body = QWidget()
        run_form = QFormLayout(run_body)
        self.run_details_rows = {}
        for title, column in (
            ("Named run", 0),
            ("Submitted", 7),
            ("Model / study", 8),
            ("Revision", 4),
            ("Stage", 1),
            ("Validation scope", 2),
            ("Scoped verdict", 3),
            ("Independent error [%]", 5),
            ("Evidence link", 6),
        ):
            label = QLabel()
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            label.setWordWrap(True)
            self.run_details_rows[column] = label
            run_form.addRow(title, label)
        run_scroll.setWidget(run_body)
        run_layout.addWidget(run_scroll, 1)
        run_close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        run_close.rejected.connect(self.run_details_window.close)
        run_layout.addWidget(run_close)
        self._selection_changed()

    def set_records(self, entries, diagnostics=(), *, project=None):
        selected_ids = {entry.id for entry in self.selected_entries}
        comparison_keys = self._comparison_keys
        self.entries = tuple(entries)
        self._current_project = project
        self.compatibility_filter.setEnabled(project is not None)
        if project is None:
            with QSignalBlocker(self.compatibility_filter):
                self.compatibility_filter.setCurrentIndex(0)
        messages = [*diagnostics, *(entry.diagnostic for entry in self.entries if entry.diagnostic)]
        self.diagnostics.setText("\n".join(messages))
        self.diagnostics.setVisible(bool(messages))
        with QSignalBlocker(self.runs_table):
            self.runs_table.clearSelection()
            self.runs_table.setRowCount(len(self.entries))
            for row, entry in enumerate(self.entries):
                metadata = entry.run.metadata if entry.run else {}
                summary = metadata.get("summary", {})
                accepted = summary.get("accepted")
                claimed = summary.get("scope") in {"continuous_sheet", "fabrication_review"}
                error = summary.get("validated_relative_field_error")
                date = run_timestamp(entry.run or metadata)
                compatibility, compatibility_label = run_compatibility(metadata, project)
                values = (
                    (
                        "Baseline · "
                        if self.study_state.baseline and entry.run and self.study_state.baseline.matches(entry.run)
                        else ""
                    )
                    + entry.label,
                    metadata.get("pipeline_stage", summary.get("pipeline_stage", "unknown")),
                    summary.get("scope", "unavailable"),
                    "Passed"
                    if claimed and accepted is True
                    else "Failed"
                    if claimed and accepted is False
                    else "Unverified",
                    metadata.get("project_revision"),
                    None if error is None else error * 100,
                    "Verified" if entry.verified else "Unavailable",
                    date.text,
                    compatibility_label if entry.verified else "Evidence unavailable",
                )
                for column, value in enumerate(values):
                    item = _item(value, entry.diagnostic or str(entry.directory))
                    if entry.run and column == 3:
                        verdicts = run_verdicts(metadata)
                        item.setToolTip(
                            f"Saved validation scope: {summary.get('scope', 'unrecorded')}. This verdict applies only to that scope.\n"
                            f"Sheet checks: {verdicts['sheet']}\n"
                            f"Sweep geometry: {verdicts['sweeps']}\n"
                            f"Fabrication review: {verdicts['fabrication']}\n"
                            "A passed validation scope does not release the coil for construction."
                        )
                    if column == 7:
                        item.setToolTip(
                            "Recorded job submission time. "
                            + date.tooltip
                            + "\nRun-link creation and file modification times are never used."
                        )
                    if column == 8:
                        item.setToolTip(
                            "Compatibility with the current applied model and study settings. Revision number alone cannot establish compatibility."
                        )
                        item.setData(Qt.ItemDataRole.UserRole + 1, compatibility if entry.verified else "unknown")
                    item.setData(Qt.ItemDataRole.UserRole, entry.id)
                    if column == 0:
                        item.setToolTip(
                            f"{entry.label}\nSaved revision: {metadata.get('project_revision', 'unrecorded')}\nEvidence link: {'Verified' if entry.verified else 'Unavailable'}\n{entry.diagnostic or entry.directory}"
                        )
                        baseline = bool(
                            self.study_state.baseline and entry.run and self.study_state.baseline.matches(entry.run)
                        )
                        item.setIcon(
                            icon(
                                "run-history",
                                badge="baseline" if baseline else None if entry.verified else "unverified",
                            )
                        )
                    self.runs_table.setItem(row, column, item)
                if entry.id in selected_ids:
                    flags = (
                        self.runs_table.selectionModel().SelectionFlag.Select
                        | self.runs_table.selectionModel().SelectionFlag.Rows
                    )
                    self.runs_table.selectionModel().select(self.runs_table.model().index(row, 0), flags)
        self.runs_table.resizeColumnsToContents()
        self._responsive_columns()
        self._filter_runs(emit=False)
        self._refresh_colors()
        self._selection_changed(emit=False)
        if comparison_keys:
            pair = tuple(
                next((entry for entry in self.entries if entry.verified and _record_key(entry.run) == key), None)
                for key in comparison_keys
            )
            if all(pair):
                self._compare_entries(*pair, emit=False)
            else:
                self._clear_comparison()
                self.comparison_notes.setText(
                    "Previous comparison is unavailable because a referenced run is missing or failed verification."
                )
        self._update_study_controls()

    def discard_project_notes(self, project_id):
        """Discard this owner's unsaved text; retain other owners' drafts."""
        self._hypothesis_drafts.pop(project_id, None)
        contexts = set(self._rationale_drafts) | set(self._rationale_entries)
        for context in contexts:
            if context[0] == project_id:
                self._rationale_drafts.pop(context, None)
                self._rationale_entries.pop(context, None)
        if self._study_project_id == project_id:
            self.set_study_state(self.study_state, project_id=project_id)

    def has_note_drafts(self, project_id, state: DesignStudyState):
        """Check every pending run decision without changing visible selection."""
        hypothesis, decisions = self.note_drafts(project_id, state)
        return hypothesis != state.hypothesis or bool(decisions)

    def note_drafts(self, project_id, state: DesignStudyState):
        """Return an owner's drafts for validation by the shared application."""
        hypothesis = self._hypothesis_drafts.get(project_id, state.hypothesis).strip()
        decisions = []
        for context, entry in self._rationale_entries.items():
            if context[0] == project_id:
                note = state.note_for(entry.run)
                rationale = self._rationale_drafts.get(context, "").strip()
                if rationale != (note.rationale if note else ""):
                    decisions.append((entry, rationale))
        return hypothesis, tuple(decisions)

    def set_study_state(self, state: DesignStudyState, *, project_id=None):
        if not isinstance(state, DesignStudyState):
            raise TypeError("History design state must be a DesignStudyState.")
        self.study_state = state
        self._study_project_id = project_id
        for context, entry in tuple(self._rationale_entries.items()):
            if context[0] != project_id:
                continue
            note = state.note_for(entry.run)
            if self._rationale_drafts.get(context, "").strip() == (note.rationale if note else ""):
                self._rationale_drafts.pop(context, None)
                self._rationale_entries.pop(context, None)
        draft = self._hypothesis_drafts.get(project_id)
        if draft is not None and draft.strip() == state.hypothesis:
            self._hypothesis_drafts.pop(project_id, None)
            draft = None
        with QSignalBlocker(self.hypothesis):
            self.hypothesis.setPlainText(state.hypothesis if draft is None else draft)
        self._selection_changed(emit=False)
        # Refresh identity labels without disturbing selection or comparison.
        for row, entry in enumerate(self.entries):
            item = self.runs_table.item(row, 0)
            if item:
                baseline = bool(state.baseline and entry.run and state.baseline.matches(entry.run))
                item.setText(("Baseline · " if baseline else "") + entry.label)
                item.setIcon(
                    icon("run-history", badge="baseline" if baseline else None if entry.verified else "unverified")
                )
        self._update_study_controls()

    def _baseline_entry(self):
        reference = self.study_state.baseline
        return next(
            (entry for entry in self.entries if entry.verified and reference and reference.matches(entry.run)), None
        )

    def _owned_selection(self):
        selected = self.selected_entries
        if len(selected) != 1 or not selected[0].verified:
            return None
        owner = selected[0].run.metadata.get("project_id")
        return selected[0] if owner and (self._study_project_id is None or owner == self._study_project_id) else None

    @property
    def owned_selection(self):
        """Selected verified result that belongs to the displayed project."""
        return self._owned_selection()

    @property
    def pending_notes(self):
        """Current unsaved intent and decision, for atomic project saving.

        The selected entry is explicit because the viewport may show another
        result. Length validation remains the application service's authority.
        """
        selected = self._owned_selection()
        hypothesis = self.hypothesis.toPlainText().strip()
        rationale = self.rationale.toPlainText().strip() if selected else ""
        note = self.study_state.note_for(selected.run) if selected else None
        if hypothesis == self.study_state.hypothesis and rationale == (note.rationale if note else ""):
            return None
        return hypothesis, rationale, selected

    @property
    def pending_note_updates(self):
        """All dirty decisions in this project, including nonselected runs.

        Keep the entry last verified when the note was edited. Application
        saving rechecks its disk evidence and reports corruption or removal
        instead of silently losing the designer's pending decision.
        """
        updates = []
        for context, entry in self._rationale_entries.items():
            if context[0] != self._study_project_id:
                continue
            note = self.study_state.note_for(entry.run)
            rationale = self._rationale_drafts.get(context, "").strip()
            if rationale != (note.rationale if note else ""):
                updates.append((entry, rationale))
        return tuple(updates)

    def _update_study_controls(self):
        selected = self._owned_selection()
        baseline = self._baseline_entry()
        if baseline:
            self.baseline_label.setText(
                f"Baseline: {baseline.label} · revision {baseline.run.metadata.get('project_revision', 'unrecorded')}"
            )
            apply_tone(self.baseline_label, "info")
        elif self.study_state.baseline:
            self.baseline_label.setText(
                "Pinned baseline is unavailable. Restore its verified evidence or pin another run."
            )
            apply_tone(self.baseline_label, "warning")
        else:
            self.baseline_label.setText("Baseline: choose a verified owned run and pin it for comparison.")
            apply_tone(self.baseline_label, "neutral")
        self.pin_baseline_button.setEnabled(selected is not None)
        self.baseline_label.setToolTip(self.baseline_label.text())
        self.clear_baseline_button.setEnabled(self.study_state.baseline is not None)
        self.compare_baseline_button.setEnabled(bool(selected and baseline and selected.id != baseline.id))
        self.create_variant_button.setEnabled(selected is not None and len(self.hypothesis.toPlainText()) <= 4000)
        self.rationale.setEnabled(selected is not None)
        self.save_notes_button.setEnabled(
            len(self.hypothesis.toPlainText()) <= 4000 and len(self.rationale.toPlainText()) <= 4000
        )
        hypothesis_dirty = self.hypothesis.toPlainText().strip() != self.study_state.hypothesis
        note = self.study_state.note_for(selected.run) if selected else None
        rationale_dirty = bool(selected and self.rationale.toPlainText().strip() != (note.rationale if note else ""))
        oversized = len(self.hypothesis.toPlainText()) > 4000 or len(self.rationale.toPlainText()) > 4000
        pending_decisions = len(self.pending_note_updates)
        self.notes_status.setText(
            "Limit each note to 4000 characters."
            if oversized
            else "Unsaved design notes"
            if hypothesis_dirty or rationale_dirty
            else f"Unsaved decisions for {pending_decisions} other run(s)"
            if pending_decisions
            else "Notes match project state"
        )
        apply_tone(
            self.notes_status,
            "warning" if oversized or hypothesis_dirty or rationale_dirty or pending_decisions else "neutral",
        )

    def _hypothesis_edited(self):
        self._hypothesis_drafts[self._study_project_id] = self.hypothesis.toPlainText()
        self._update_study_controls()

    def _rationale_edited(self):
        if self._notes_context is not None:
            self._rationale_drafts[self._notes_context] = self.rationale.toPlainText()
            entry = self._owned_selection()
            if entry is not None:
                self._rationale_entries[self._notes_context] = entry
        self._update_study_controls()

    def _pin_baseline(self):
        selected = self._owned_selection()
        if selected:
            self.baseline_requested.emit(selected)

    def _create_variant(self):
        selected = self._owned_selection()
        if selected and len(self.hypothesis.toPlainText()) <= 4000:
            self.variant_requested.emit(selected, self.hypothesis.toPlainText().strip())

    def _save_notes(self):
        if self.save_notes_button.isEnabled():
            self.notes_changed.emit(
                self.hypothesis.toPlainText().strip(),
                self.rationale.toPlainText().strip() if self._owned_selection() else "",
            )

    @property
    def selected_entries(self):
        rows = sorted(
            index.row()
            for index in self.runs_table.selectionModel().selectedRows()
            if not self.runs_table.isRowHidden(index.row())
        )
        return tuple(self.entries[row] for row in rows)

    def select_record(self, record_id):
        for row, entry in enumerate(self.entries):
            if entry.id == record_id:
                if self.runs_table.isRowHidden(row):
                    self.compatibility_filter.setCurrentIndex(0)
                self.runs_table.clearSelection()
                self.runs_table.selectRow(row)
                return True
        return False

    def _filter_runs(self, *_args, emit=True):
        if not hasattr(self, "runs_table"):
            return
        key = self.compatibility_filter.currentData()
        count = 0
        with QSignalBlocker(self.runs_table):
            for row in range(self.runs_table.rowCount()):
                context = self.runs_table.item(row, 8)
                visible = key == "all" or context is not None and context.data(Qt.ItemDataRole.UserRole + 1) == key
                self.runs_table.setRowHidden(row, not visible)
                count += visible
        self.filter_count.setText(f"{count} / {len(self.entries)}")
        self._selection_changed(emit=emit)

    def _selection_changed(self, *, emit=True):
        selected = self.selected_entries
        single = len(selected) == 1
        self.run_details_button.setEnabled(single)
        if single:
            metadata = selected[0].run.metadata if selected[0].run else {}
            summary = metadata.get("summary") or {}
            self.selected_summary.setText(
                f"Revision {metadata.get('project_revision', 'unrecorded')} · {metadata.get('pipeline_stage', 'unrecorded')} · saved scope: {summary.get('scope', 'unrecorded')}"
            )
        else:
            self.selected_summary.setText(
                "Select a run to inspect its saved revision, study boundary and acceptance scope."
            )
        self.alias.setEnabled(single)
        self.rename_button.setEnabled(single)
        self.alias.setText(selected[0].alias if single else "")
        self.compare_button.setEnabled(len(selected) == 2 and all(entry.verified for entry in selected))
        note = self.study_state.note_for(selected[0].run) if single and selected[0].verified else None
        self._notes_context = (
            (self._study_project_id, _record_key(selected[0].run)) if single and selected[0].verified else None
        )
        rationale = note.rationale if note else ""
        draft = self._rationale_drafts.get(self._notes_context)
        if draft is not None and draft.strip() == rationale:
            self._rationale_drafts.pop(self._notes_context, None)
            self._rationale_entries.pop(self._notes_context, None)
            draft = None
        with QSignalBlocker(self.rationale):
            self.rationale.setPlainText(rationale if draft is None else draft)
        self._update_study_controls()
        if emit and single and selected[0].verified:
            self.selected.emit(selected[0])

    def _rename_selected(self):
        if len(self.selected_entries) == 1:
            self.alias_changed.emit(self.selected_entries[0].id, self.alias.text().strip())

    def _clear_comparison(self):
        self.comparison = None
        self._comparison_keys = None
        self.metrics_table.setRowCount(0)
        self.parameters_table.setRowCount(0)
        self.comparison_notes.setText(
            "Select two verified runs to compare their original requests and scoped evidence."
        )

    def compare_selected(self):
        selected = self.selected_entries
        if len(selected) != 2 or not all(entry.verified for entry in selected):
            return None
        return self._compare_entries(*selected)

    def compare_to_baseline(self):
        selected, baseline = self._owned_selection(), self._baseline_entry()
        if selected and baseline and selected.id != baseline.id:
            return self._compare_entries(baseline, selected)
        return None

    def _compare_entries(self, left, right, *, emit=True):
        self._clear_comparison()
        try:
            comparison = compare_runs(left.run, right.run)
        except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
            self.comparison_notes.setText(f"Comparison unavailable: {exc}")
            return None
        self.comparison = comparison
        self._comparison_keys = (_record_key(comparison.left), _record_key(comparison.right))
        self.metrics_table.setRowCount(len(comparison.metrics))
        for row, metric in enumerate(comparison.metrics):
            tooltip = f"A: {metric.left_scope}\nB: {metric.right_scope}\n{metric.reason}"
            for column, value in enumerate(
                (
                    metric.name,
                    metric.left,
                    metric.right,
                    metric.unit,
                    "Same scope" if metric.comparable else "Separate scopes",
                    metric.right - metric.left if metric.comparable else "Unavailable",
                )
            ):
                self.metrics_table.setItem(row, column, _item(value, tooltip))
        self.parameters_table.setRowCount(len(comparison.parameter_deltas))
        for row, delta in enumerate(comparison.parameter_deltas):
            label, first = _parameter(delta.field, delta.left)
            _, second = _parameter(delta.field, delta.right)
            for column, value in enumerate((label, first, second)):
                self.parameters_table.setItem(
                    row,
                    column,
                    _item(
                        value,
                        f"{delta.field}\nPhysical A: {_text(first)}\nPhysical B: {_text(second)}\nOriginal A: {_text(delta.left)}\nOriginal B: {_text(delta.right)}",
                    ),
                )
        self.metrics_table.resizeColumnsToContents()
        self.parameters_table.resizeColumnsToContents()
        self.comparison_notes.setText(
            f"A: {left.label}; B: {right.label}. B − A is shown only for comparable evidence; its sign does not select a better design. "
            + " ".join(comparison.notes)
        )
        if emit:
            self.comparison_tabs.setCurrentWidget(self.metrics_table)
            self._open_details(0)
            self.compare_requested.emit(comparison.left, comparison.right)
        return comparison

    def _open_details(self, tab=None):
        if tab is not None:
            self.comparison_tabs.setCurrentIndex(tab)
        self.details_window.show()
        self.details_window.raise_()
        self.details_window.activateWindow()

    def hideEvent(self, event):
        super().hideEvent(event)
        if hasattr(self, "details_window"):
            self.details_window.hide()
            self.run_details_window.hide()

    def _open_run_details(self):
        selected = self.selected_entries
        if len(selected) != 1:
            return
        row = self.entries.index(selected[0])
        self.run_details_window.setWindowTitle(f"{selected[0].label} — Recorded run details")
        for column, label in self.run_details_rows.items():
            item = self.runs_table.item(row, column)
            label.setText(item.text())
            label.setToolTip(item.toolTip())
        self.run_details_window.show()
        self.run_details_window.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "runs_table"):
            QTimer.singleShot(0, self, self._responsive_columns)

    def _responsive_columns(self):
        """Date, compatibility, scoped verdict and independent error stay visible together."""
        compact = self.width() < 1050
        table = self.runs_table
        header = table.horizontalHeader()
        for column in (1, 2, 4, 6):
            table.setColumnHidden(column, compact)
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        widths = {1: 90, 2: 155, 3: 85, 4: 70, 5: 145, 6: 85, 7: 155, 8: 160}
        for column, width in widths.items():
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
            table.setColumnWidth(column, width)

    def _refresh_colors(self):
        apply_tone(self.diagnostics, "warning" if self.diagnostics.text() else "neutral")
        self._update_study_controls()
        for row, entry in enumerate(self.entries):
            for column in range(self.runs_table.columnCount()):
                item = self.runs_table.item(row, column)
                if item is None:
                    continue
                tone = "neutral"
                if entry.verified and column == 3:
                    tone = {"Passed": "success", "Failed": "error"}.get(item.text(), "neutral")
                elif entry.verified and column in {1, 6}:
                    tone = "info"
                item.setForeground(item_foreground(tone, self.runs_table.palette()))

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange, QEvent.Type.StyleChange):
            if hasattr(self, "runs_table"):
                QTimer.singleShot(0, self, self._refresh_colors)
