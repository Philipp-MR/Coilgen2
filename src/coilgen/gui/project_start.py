"""Explicit creation of an empty design; examples remain separate templates."""

from pathlib import Path

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QVBoxLayout

from ..application.project import ProjectDocument
from ..config import CoilGenConfig
from .colors import apply_tone


def blank_project(name: str = "New coil design", *, project_root: str | Path | None = None) -> ProjectDocument:
    """Create a saveable document with neither support nor field selected.

    Retain native configuration and study defaults, including inactive parameter
    values. The explicit missing target complements the native missing support;
    execution readiness remains the shared validator's responsibility.
    """
    return ProjectDocument.from_config(CoilGenConfig(field_shape_function="none"), name=name, project_root=project_root)


class NewProjectDialog(QDialog):
    """Name a blank project without constructing one on open or cancel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New project — CoilGen2")
        self.project = None
        layout = QVBoxLayout(self)
        note = QLabel("Start with an empty design. Choose the support and target in the model tree.")
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.name_edit = QLineEdit("New coil design")
        self.name_edit.setObjectName("new_project_name")
        self.name_edit.setMaxLength(120)
        self.name_edit.setAccessibleName("Project name")
        form.addRow("Project name", self.name_edit)
        layout.addLayout(form)
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setAccessibleName("Project name error")
        apply_tone(self.error_label, "error")
        self.error_label.hide()
        layout.addWidget(self.error_label)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        create = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        create.setText("Create project")
        create.setAccessibleName("Create blank project")
        create.setDefault(True)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.name_edit.textEdited.connect(self.error_label.hide)
        self.name_edit.selectAll()
        self.name_edit.setFocus()
        self.resize(430, self.sizeHint().height())

    def accept(self):
        try:
            project = blank_project(self.name_edit.text().strip())
        except ValueError as exc:
            self.error_label.setText(str(exc))
            self.error_label.show()
            self.name_edit.setFocus()
            return
        self.project = project
        super().accept()
