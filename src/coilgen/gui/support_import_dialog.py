"""Confirm what a support STL holds and which body is the primary or the shield."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)

from ..application.support_import import ROLE_CHOICES


class SupportImportDialog(QDialog):
    """Bodies, size, boundary loops and unit warnings, then one role per body."""

    def __init__(self, summary, name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Import support · {name}")
        self.setObjectName("support_import_dialog")
        layout = QVBoxLayout(self)
        self.summary = QLabel(summary["text"], self)
        self.summary.setObjectName("support_import_summary")
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.summary)
        self.warnings = QLabel("\n".join(summary["warnings"]), self)
        self.warnings.setObjectName("support_import_warnings")
        self.warnings.setWordWrap(True)
        self.warnings.setTextFormat(Qt.TextFormat.PlainText)
        self.warnings.setVisible(bool(summary["warnings"]))
        layout.addWidget(self.warnings)
        self.role_boxes = []
        if summary["needs_roles"]:
            layout.addWidget(QLabel("Role of each body. The inner radius group starts as the primary.", self))
            form = QFormLayout()
            for body, role in zip(summary["bodies"], summary["roles"]):
                box = QComboBox(self)
                for choice in ROLE_CHOICES:
                    box.addItem(choice["title"], choice["value"])
                box.setCurrentIndex(max(0, box.findData(role)))
                box.setToolTip(body["text"])
                box.setAccessibleName(f"{body['name']} role")
                form.addRow(f"{body['name']} · {body['text']}", box)
                self.role_boxes.append(box)
            layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Import")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def roles(self):
        """One role per body in pipeline order, or ``None`` for a single body."""
        return [box.currentData() for box in self.role_boxes] or None
