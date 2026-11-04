"""Read-only inspection of a verified run's original request and applied inputs."""

from __future__ import annotations

from collections.abc import Mapping
import json

from PySide6.QtCore import QObject, QRunnable, Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from ..application.evidence import EvidenceIntegrityError as SavedEvidenceError
from ..application.evidence import load_saved_request as load_saved_request
from ..application.submitted_design import design_choice_groups, displacement_label
from ..contours import _normalise_contour_path_regularization_mode


class _Signals(QObject):
    finished = Signal(object, str)


class SavedSettingsTask(QRunnable):
    def __init__(self, record, project, node):
        super().__init__()
        self.record, self.project, self.node = record, project, node
        self.signals = _Signals()
        self.verification_error = False

    def run(self):
        from .design_tasks import deliver_task_signal

        try:
            request = load_saved_request(self.record)
        except Exception as exc:
            self.verification_error = isinstance(exc, SavedEvidenceError)
            deliver_task_signal(self.signals, "finished", None, f"{type(exc).__name__}: {exc}")
        else:
            deliver_task_signal(self.signals, "finished", request, "")


class SavedSettingsDialog(QDialog):
    def __init__(self, request, project, node, parent=None, *, smoothing=()):
        super().__init__(parent)
        self.setWindowTitle("Run design choices — CoilGen2")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        bounds = self.screen().availableGeometry()
        self.resize(min(940, bounds.width() - 20), min(650, bounds.height() - 50))
        layout = QVBoxLayout(self)
        owner = request.get("project_id")
        relation = (
            "owner unrecorded"
            if not owner
            else "another project"
            if owner != project.id
            else "matching submitted inputs"
            if request.get("project_fingerprint") == project.fingerprint
            else "same project, different or unrecorded input fingerprint"
        )
        label = QLabel(
            f"Submitted {request.get('name') or 'run'} · revision {request.get('project_revision') or 'unrecorded'}\n"
            f"Editor at inspection: {project.name} · applied revision {project.revision} · {relation}. Pending edits are excluded.\n"
            "Read-only. Expand a section to inspect its submitted values. Complete request contains every recorded input "
            "at its original precision. An absent setting is unrecorded."
        )
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setWordWrap(True)
        label.setToolTip(f"Submitted project ID: {owner or 'unrecorded'}\nEditor project ID: {project.id}")
        layout.addWidget(label)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        choices = QTreeWidget()
        choices.setObjectName("submitted_design_choices")
        choices.setAccessibleName("Submitted design choices and recorded smoothing results")
        choices.setTabKeyNavigation(False)
        choices.setHeaderLabels(["Design choice or recorded outcome", "Value"])
        choices.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        choices.setWordWrap(True)
        groups = design_choice_groups(request)
        focused_group = {
            "geometry": "Support and mesh",
            "target": "Target and magnetic field",
            "optimization": "Stream function optimization",
            "material": "Winding and conductor",
            "winding": "Winding and conductor",
            "validation": "Study and validation",
        }.get(node)
        expanded_groups = {focused_group} if focused_group else {"Contour smoothing", "Study and validation"}
        for group in groups:
            parent_item = QTreeWidgetItem(choices, [group["title"]])
            first = group["rows"][0]
            parent_item.setText(1, f"{first['name']} · {first['value']}")
            for row in group["rows"]:
                item = QTreeWidgetItem(parent_item, [row["name"], row["value"]])
                item.setToolTip(1, row["value"])
            parent_item.setExpanded(group["title"] in expanded_groups)
        if not groups:
            QTreeWidgetItem(choices, ["No summary fields recorded. See Complete request."])
        smoothing = (
            tuple(part for part in smoothing if isinstance(part, Mapping))
            if isinstance(smoothing, (list, tuple))
            else ()
        )
        if smoothing:
            outcome = QTreeWidgetItem(choices, ["Recorded smoothing outcome", "From this run"])
            for row in smoothing:
                support = QTreeWidgetItem(
                    outcome,
                    [
                        f"Support {row.get('part_index', 'unrecorded')}",
                        f"{row.get('smoothed_count', 'unrecorded')} contours smoothed",
                    ],
                )
                for name, key in (
                    ("Original paths kept after fit checks", "fit_rejected_count"),
                    ("Original paths kept after spacing checks", "spacing_rejected_count"),
                ):
                    QTreeWidgetItem(support, [name, str(row.get(key, "unrecorded"))])
                if "local_fit_count" in row:
                    QTreeWidgetItem(
                        support, ["Locally fitted (included in smoothed count)", str(row["local_fit_count"])]
                    )
                distance = displacement_label(row.get("max_displacement_m"))
                QTreeWidgetItem(support, ["Maximum accepted displacement", distance])
                reasons = row.get("failure_reasons")
                if isinstance(reasons, Mapping):
                    for reason, count in reasons.items():
                        QTreeWidgetItem(support, [str(reason).replace("_", " "), str(count)])
                support.setExpanded(len(smoothing) == 1)
            outcome.setExpanded(True)
        elif (
            _normalise_contour_path_regularization_mode(
                request.get("config", {}).get("contour_path_regularization_mode")
            )
            == "parameterized_bspline"
        ):
            QTreeWidgetItem(choices, ["Smoothing outcome", "No per-support outcome recorded for this run."])
        tabs.addTab(choices, "Design choices")
        table = QTableWidget(0, 3)
        table.setObjectName("saved_settings_differences")
        table.setTabKeyNavigation(False)
        table.setHorizontalHeaderLabels(
            ["Native parameter", "Submitted request", f"Applied model · r{project.revision}"]
        )
        table.setToolTip(
            f"Compared with {project.name} (project {project.id}, revision {project.revision}) when inspection opened. "
            "Owned input references and saved worker file paths may differ after relocation."
        )
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setStretchLastSection(True)
        tabs.addTab(table, "Compare with editor")
        raw_tabs = {}
        for title in ("Complete request", "Applied model"):
            editor = QPlainTextEdit()
            editor.setReadOnly(True)
            tabs.addTab(editor, title)
            raw_tabs[editor] = title

        populated = set()

        def show_tab(index):
            widget = tabs.widget(index)
            if widget in populated:
                return
            if widget is table:
                count = self._compare(table, request, project)
                tabs.setTabText(index, f"Compare with editor ({count})")
            elif widget in raw_tabs:
                payload = (
                    request if raw_tabs[widget] == "Complete request" else {"config": project.values, **project.study}
                )
                widget.setPlainText(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
            populated.add(widget)

        tabs.currentChanged.connect(show_tab)
        if node:
            label.setText(label.text() + f" Selected check section: {node}.")
        footer = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        footer.rejected.connect(self.reject)
        layout.addWidget(footer)

    @staticmethod
    def _compare(table, request, project):
        saved = {"config." + key: value for key, value in request.get("config", {}).items()}
        applied = {"config." + key: value for key, value in project.values.items()}
        study_keys = set(project.study) | {"requirements", "scene", "dynamic_analysis", "array_study", "engineering"}
        saved.update({"study." + key: request[key] for key in study_keys if key in request})
        applied.update({"study." + key: value for key, value in project.study.items()})
        missing = object()

        def encode(value):
            return "Unrecorded" if value is missing else json.dumps(value, ensure_ascii=False, sort_keys=True)

        differences = [
            key for key in sorted(saved.keys() | applied.keys()) if saved.get(key, missing) != applied.get(key, missing)
        ]
        table.setRowCount(len(differences))
        for row, key in enumerate(differences):
            for column, value in enumerate((key, encode(saved.get(key, missing)), encode(applied.get(key, missing)))):
                item = QTableWidgetItem(value if len(value) < 200 else value[:197] + "…")
                item.setToolTip(value)
                table.setItem(row, column, item)
        return len(differences)
