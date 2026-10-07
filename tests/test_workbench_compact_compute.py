"""Primary study controls remain readable in the real scientific workspace."""

from pathlib import Path
import tempfile

import pytest
import test_native_workbench as native

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QPushButton

from coilgen.application import ProjectDocument
from coilgen.application.presets import EXAMPLE_TITLES, preset_config, preset_study
from coilgen.gui.viewport import ScientificViewport
from coilgen.gui.workbench import CoilGenWorkbench


application = native.application


@pytest.fixture
def window(application, tmp_path, monkeypatch):
    pyvistaqt = pytest.importorskip("pyvistaqt")
    monkeypatch.setattr(pyvistaqt, "QtInteractor", native.RenderlessPlotter)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    original_font = QFont(application.font())
    font_path = Path("C:/Windows/Fonts/segoeui.ttf")
    if font_path.is_file():
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            application.setFont(QFont(families[0], 9))
    project = ProjectDocument.from_config(preset_config(), name=EXAMPLE_TITLES["cylinder"]).update_study(preset_study())
    instance = CoilGenWorkbench(
        project,
        jobs=native.FakeJobs(tmp_path / "jobs"),
        service=native.FakeService(),
        viewport_factory=lambda parent: ScientificViewport(parent, synchronous_preview=True),
    )
    instance.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    instance.timer.stop()
    yield instance
    instance._draft_dirty = False
    instance.undo_stack.setClean()
    instance.close()
    instance.undo_stack.clear()
    instance.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    application.setFont(original_font)


def _assert_visible_controls(window, application):
    for _ in range(3):
        application.processEvents()
    assert window.compute_button.isVisible(), {
        "window": window.width(),
        "toolbar": window.compute_button.parentWidget().width(),
        "compute": (window.compute_button.width(), window.compute_button.minimumSizeHint().width()),
        "font": window.font().toString(),
    }
    assert window.compute_button.width() >= window.compute_button.minimumSizeHint().width()
    toolbar = window.compute_button.parentWidget()
    assert toolbar.contentsRect().contains(window.compute_button.geometry())
    assert window.tree.isVisible()
    assert window.stage_selector.isHidden()
    assert not hasattr(window, "study_caption")
    assert not hasattr(window, "workflow")
    assert len(window.findChildren(QPushButton, "primary_compute")) == 1


@pytest.mark.parametrize("width", (800, 900, 1440))
def test_real_workspace_keeps_complete_compute_label_visible(window, application, width):
    window.resize(width, 800)
    window.show()
    for stage, expected in (("prepared", "Prepare model"), ("evaluated", "Compute full design")):
        window.tree.setCurrentItem(window.stage_items[stage])
        for step in ("geometry", "material", "export"):
            window.select_node(step)
            window._adapt_toolbars()
            _assert_visible_controls(window, application)
            assert window.compute_button.text() == expected


def test_visible_workspace_adapts_after_font_change(window, application):
    window.resize(900, 800)
    window.show()
    _assert_visible_controls(window, application)
    larger_font = QFont(window.font())
    larger_font.setPointSize(14)
    window.setFont(larger_font)
    _assert_visible_controls(window, application)
    assert window.service.submissions == []
