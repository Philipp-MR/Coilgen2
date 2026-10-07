"""Small scientific Qt controls with concise display and explicit units."""

from PySide6.QtGui import QDoubleValidator, QValidator
from PySide6.QtCore import QLocale, QSignalBlocker
from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout, QWidget


class SettingsPanel(QWidget):
    """Keep controls usable when a settings dock becomes narrow."""

    form = None

    def resizeEvent(self, event):
        if self.form is not None:
            policy = (
                QFormLayout.RowWrapPolicy.WrapAllRows
                if event.size().width() < 340
                else QFormLayout.RowWrapPolicy.WrapLongRows
            )
            if self.form.rowWrapPolicy() != policy:
                self.form.setRowWrapPolicy(policy)
        super().resizeEvent(event)


class ScientificSpinBox(QDoubleSpinBox):
    """Display significant digits without reducing the stored decimal precision."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._validator = QDoubleValidator(self)
        self._validator.setNotation(QDoubleValidator.Notation.ScientificNotation)
        locale = QLocale.c()
        locale.setNumberOptions(QLocale.NumberOption.OmitGroupSeparator | QLocale.NumberOption.RejectGroupSeparator)
        self.setLocale(locale)
        self._incomplete_text = None
        self.lineEdit().textEdited.connect(self._remember_incomplete_text)
        self.lineEdit().textChanged.connect(self._restore_incomplete_text)

    def textFromValue(self, value):
        return self.locale().toString(value, "g", 12)

    def _number_text(self, text):
        if self.prefix() and text.startswith(self.prefix()):
            text = text[len(self.prefix()) :]
        if self.suffix() and text.endswith(self.suffix()):
            text = text[: -len(self.suffix())]
        return text.strip()

    def valueFromText(self, text):
        value, accepted = self.locale().toDouble(self._number_text(text))
        return value if accepted else self.value()

    def validate(self, text, position):
        self._validator.setLocale(self.locale())
        self._validator.setRange(self.minimum(), self.maximum(), self.decimals())
        state, _text, _position = self._validator.validate(self._number_text(text), position)
        return state, text, position

    def _remember_incomplete_text(self, text):
        state, _text, _position = self.validate(text, len(text))
        self._incomplete_text = text if state != QValidator.State.Acceptable else None

    def _restore_incomplete_text(self, *_args):
        # Qt may reformat on focus or style changes. Keep incomplete user text
        # so Apply/Compute can report it rather than submit the previous value.
        if self._incomplete_text is not None and self.lineEdit().text() != self._incomplete_text:
            edit = self.lineEdit()
            with QSignalBlocker(edit):
                edit.setText(self._incomplete_text)
                edit.setModified(True)

    def event(self, event):
        handled = super().event(event)
        if getattr(self, "_incomplete_text", None) is not None:
            self._restore_incomplete_text()
        return handled

    def setValue(self, value):
        self._incomplete_text = None
        super().setValue(value)

    def stepBy(self, steps):
        pending = self._incomplete_text is not None
        self._incomplete_text = None
        if pending:
            super().setValue(self.value())
        super().stepBy(steps)
