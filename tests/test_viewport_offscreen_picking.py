"""Headless VTK needs no picking; interactive viewports retain actor picking."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtWidgets import QApplication

from test_native_workbench import RenderlessPlotter


@pytest.mark.parametrize("has_interactor", [False, True])
def test_viewport_initialization_only_registers_picking_with_vtk_interactor(monkeypatch, has_interactor):
    pyvistaqt = pytest.importorskip("pyvistaqt")
    from coilgen.gui.viewport import ScientificViewport

    application = QApplication.instance() or QApplication([])

    class PickingPlotter(RenderlessPlotter):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.iren = object() if has_interactor else None
            self.picking_calls = []

        def enable_mesh_picking(self, **kwargs):
            if self.iren is None:
                raise AssertionError("Offscreen VTK has no picker interactor")
            self.picking_calls.append(kwargs)

    monkeypatch.setattr(pyvistaqt, "QtInteractor", PickingPlotter)
    viewport = ScientificViewport(synchronous_preview=True)
    viewport.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    try:
        assert len(viewport.plotter.picking_calls) == int(has_interactor)
        if has_interactor:
            options = viewport.plotter.picking_calls[0]
            assert options["callback"] == viewport._picked_actor
            assert options["left_clicking"] and options["use_actor"]
            assert options["show"] is False and options["show_message"] is False
    finally:
        viewport.close()
        viewport.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        application.processEvents()
