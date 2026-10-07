"""Compact navigation for scientific model construction and evidence review."""

from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QSizePolicy, QToolButton, QWidget

from .colors import tone_color
from .icons import icon


WORKFLOW_STEPS = (
    ("geometry", "Geometry", "Define the support surface, coordinates and mesh."),
    ("target", "Target", "Define the requested field, sampling region and amplitude."),
    ("design", "Conductor", "Define the physical conductor and winding constraints."),
    ("compute", "Study", "Set optimization, validation and requested outputs; compute the current model."),
    ("review", "Results", "Inspect saved solutions, engineering checks and comparisons."),
    ("export", "Export", "Export saved conductor solid geometry with explicit units and recorded model provenance."),
)
_STEP_ICONS = {
    "geometry": "support-cylinder",
    "target": "target-gradient",
    "design": "conductor-profile",
    "compute": "compute-study",
    "review": "field-error",
    "export": "export-conductor",
}


class WorkflowNavigation(QWidget):
    """Active step is navigation state; it makes no completion or acceptance claim."""

    step_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("workflow_navigation")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.buttons = {}
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)
        self._current_step = "geometry"
        self._updating_style = False
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for index, (step, label, explanation) in enumerate(WORKFLOW_STEPS, 1):
            button = QToolButton(self)
            button.setObjectName(f"workflow_step_{step}")
            button.setText(label)
            button.setIcon(icon(_STEP_ICONS[step]))
            button.setIconSize(QSize(20, 20))
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            button.setAccessibleName(f"Model section: {label}")
            button.setToolTip(explanation + " Active means selected, not accepted.")
            button.setAccessibleDescription(button.toolTip())
            button.setCheckable(True)
            button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            button.setMinimumHeight(30)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.buttons[step] = button
            self.button_group.addButton(button, index - 1)
            layout.addWidget(button, 1)
        self._apply_style()
        self.button_group.idClicked.connect(self._request_step)
        self.set_current(self._current_step)

    @property
    def current_step(self):
        return self._current_step

    def set_current(self, step):
        """Synchronize with the workbench without emitting a navigation request."""
        if not isinstance(step, str) or step not in self.buttons:
            raise ValueError(f"Unknown workflow step: {step!r}.")
        self._current_step = step
        self.buttons[step].setChecked(True)

    def _request_step(self, index):
        step = WORKFLOW_STEPS[index][0]
        self.set_current(step)
        self.step_requested.emit(step)

    def _apply_style(self):
        if self._updating_style:
            return
        self._updating_style = True
        try:
            palette = self.palette()
            accent = tone_color("info", palette).name()
            background = palette.color(QPalette.ColorRole.Window).name()
            foreground = palette.color(QPalette.ColorRole.WindowText).name()
            button = palette.color(QPalette.ColorRole.Button).name()
            button_text = palette.color(QPalette.ColorRole.ButtonText).name()
            border = palette.color(QPalette.ColorRole.Mid).name()
            self.setStyleSheet(
                f"QToolButton {{ background: {button}; color: {button_text}; border: 1px solid {border};"
                "border-radius: 4px; padding: 4px 6px; }"
                f"QToolButton:hover {{ background: {background}; color: {foreground}; border-color: {accent}; }}"
                f"QToolButton:checked {{ background: {background}; color: {accent}; border-color: {accent};"
                "font-weight: 600; }"
                f"QToolButton:focus {{ border: 2px solid {accent}; padding: 3px 5px; }}"
            )
        finally:
            self._updating_style = False

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange, QEvent.Type.StyleChange):
            if hasattr(self, "_updating_style"):
                self._apply_style()
