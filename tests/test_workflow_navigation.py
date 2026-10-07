"""Workflow navigation is accessible, reversible and separate from acceptance."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from coilgen.gui.workflow import WorkflowNavigation, WORKFLOW_STEPS


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["workflow-navigation-test"])


@pytest.fixture
def navigation(application):
    widget = WorkflowNavigation()
    yield widget
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_direct_controls_have_stable_keys_and_accessible_navigation_scope(navigation):
    assert list(navigation.buttons) == [step for step, *_ in WORKFLOW_STEPS]
    for index, (step, label, _explanation) in enumerate(WORKFLOW_STEPS, 1):
        button = navigation.buttons[step]
        assert button.text() == label
        assert button.isCheckable()
        assert button.accessibleName() == f"Model section: {label}"
        assert "not accepted" in button.accessibleDescription()
        assert button.focusPolicy() == Qt.FocusPolicy.StrongFocus
    assert navigation.current_step == "geometry"
    assert navigation.buttons["geometry"].isChecked()


@pytest.mark.parametrize("step", [step for step, *_ in WORKFLOW_STEPS])
def test_click_requests_exact_step_and_keeps_one_active_control(navigation, step):
    requests = []
    navigation.step_requested.connect(requests.append)
    navigation.buttons[step].click()
    assert requests == [step]
    assert navigation.current_step == step
    assert [key for key, button in navigation.buttons.items() if button.isChecked()] == [step]
    assert all(button.isEnabled() for button in navigation.buttons.values())


def test_programmatic_navigation_can_skip_and_revert_without_request_signals(navigation):
    requests = []
    navigation.step_requested.connect(requests.append)
    navigation.set_current("review")
    assert navigation.current_step == "review"
    navigation.set_current("target")
    assert navigation.current_step == "target"
    assert navigation.buttons["target"].isChecked()
    assert not navigation.buttons["review"].isChecked()
    assert requests == []


def test_unknown_step_does_not_mutate_navigation(navigation):
    navigation.set_current("design")
    with pytest.raises(ValueError, match="Unknown workflow step"):
        navigation.set_current("accepted")
    assert navigation.current_step == "design"
    assert navigation.buttons["design"].isChecked()


def test_keyboard_activation_uses_native_qt_focus_and_signal(navigation, application):
    requests = []
    navigation.step_requested.connect(requests.append)
    navigation.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    navigation.show()
    button = navigation.buttons["compute"]
    button.setFocus()
    application.processEvents()
    QTest.keyClick(button, Qt.Key.Key_Space)
    assert requests == ["compute"]
    assert navigation.current_step == "compute"


def test_full_width_navigation_fits_1280_without_global_styles(navigation, application):
    stylesheet = application.styleSheet()
    navigation.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    navigation.resize(1280, 40)
    navigation.show()
    application.processEvents()
    assert navigation.width() == 1280
    assert navigation.minimumSizeHint().width() <= 1280
    for button in navigation.buttons.values():
        assert navigation.rect().contains(button.geometry())
        assert button.width() >= button.minimumSizeHint().width()
    assert application.styleSheet() == stylesheet
