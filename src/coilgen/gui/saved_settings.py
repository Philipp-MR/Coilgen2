"""Read-only inspection of a verified run's original request and applied inputs."""

from __future__ import annotations

import json

from PySide6.QtCore import QObject, QRunnable, Qt, Signal
from PySide6.QtWidgets import QDialog, QLabel, QPlainTextEdit, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout

from ..application.evidence import EvidenceIntegrityError as SavedEvidenceError
from ..application.evidence import load_saved_request as load_saved_request


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
    def __init__(self, request, project, node, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Inspect saved settings — CoilGen2")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.resize(940, 650)
        layout = QVBoxLayout(self)
        label = QLabel(
            f"Saved {request.get('name', 'run')} · project {request.get('project_id')} · revision {request.get('project_revision')}\n"
            f"Compared with applied {project.name} · project {project.id} · revision {project.revision} at inspection request.\n"
            f"Selected check settings: {node}. Read-only; values retain native parameter units and exact precision. "
            "Owned input references and saved worker file paths can differ after relocation."
        )
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setWordWrap(True)
        layout.addWidget(label)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        current = {"config": project.values, **project.study}
        saved = {"config." + key: value for key, value in request["config"].items()}
        applied = {"config." + key: value for key, value in project.values.items()}
        saved.update({"study." + key: request.get(key) for key in project.study})
        applied.update({"study." + key: value for key, value in project.study.items()})
        table = QTableWidget(0, 3)
        table.setObjectName("saved_settings_differences")
        table.setHorizontalHeaderLabels(["Native parameter", "Saved request", "Applied model"])
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setStretchLastSection(True)

        def encode(value):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)

        differences = [
            key for key in sorted(saved.keys() | applied.keys()) if encode(saved.get(key)) != encode(applied.get(key))
        ]
        table.setRowCount(len(differences))
        for row, key in enumerate(differences):
            for column, value in enumerate((key, encode(saved.get(key)), encode(applied.get(key)))):
                item = QTableWidgetItem(value if len(value) < 200 else value[:197] + "…")
                item.setToolTip(value)
                table.setItem(row, column, item)
        tabs.addTab(table, f"Differences ({len(differences)})")
        for title, payload in (("Verified saved request", request), ("Applied model at inspection request", current)):
            editor = QPlainTextEdit()
            editor.setReadOnly(True)
            editor.setPlainText(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
            tabs.addTab(editor, title)
