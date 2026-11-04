"""Capture the real Qt workbench and its VTK framebuffer without opening a desktop window.

Pass a saved single-channel source .coilgen, its verified run directory and a
PNG destination. Qt's offscreen platform supplies no native display handle.
VTK renders into its own hidden framebuffer, captured at the widget's exact size.
"""

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["PYVISTA_OFF_SCREEN"] = "true"

import numpy as np
import pyvistaqt
from pyvista.plotting.render_window_interactor import RenderWindowInteractor
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QFont, QFontDatabase, QImage, QPainter
from PySide6.QtWidgets import QApplication, QTreeWidget
from vtkmodules.vtkRenderingCore import vtkRenderWindow

from coilgen.application import ProjectDocument
from coilgen.gui.saved_settings import SavedSettingsDialog
from coilgen.gui.workbench import CoilGenWorkbench


def capture(source, run, output, scratch=None):
    from pyvistaqt import QtInteractor

    class HiddenFramebuffer(type(vtkRenderWindow())):
        # QVTK normally binds the provided surface to QWidget.winId(). Qt's
        # offscreen handle has no WGL pixel format. Keep VTK's own offscreen DC.
        def SetWindowInfo(self, _qt_id):
            pass

    def interactor(parent):
        surface = HiddenFramebuffer()
        surface.SetOffScreenRendering(1)
        surface.SetShowWindow(False)
        plotter = QtInteractor(parent=parent, rw=surface, off_screen=True, auto_update=0)
        # The B0 compass and world triad attach to a VTK interactor. Attach
        # them to this hidden surface so the capture matches normal operation.
        plotter.iren = RenderWindowInteractor(plotter, interactor=surface.GetInteractor())
        plotter.iren.initialize()
        return plotter

    app = QApplication.instance() or QApplication([])
    app.setOrganizationName("CoilGen QA")
    app.setApplicationName("Winding review")
    font_path = Path("C:/Windows/Fonts/segoeui.ttf")
    if font_path.is_file():
        font = QFontDatabase.addApplicationFont(str(font_path))
        if font >= 0:
            app.setFont(QFont(QFontDatabase.applicationFontFamilies(font)[0], 9))
    project = ProjectDocument.load(source)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch, ignore_cleanup_errors=True) as private:
        pyvistaqt.QtInteractor = interactor
        try:
            window = CoilGenWorkbench(project, jobs_directory=Path(private) / "jobs")
        finally:
            pyvistaqt.QtInteractor = QtInteractor
        try:
            window.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
            window.resize(1600, 1020)
            window.show()
            app.processEvents()
            window.load_run(run)
            deadline = time.monotonic() + 30
            while window.viewport._preview_worker is not None and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(0.01)
            app.processEvents()
            viewport = window.viewport
            plotter = viewport.plotter
            surface = plotter.ren_win
            assert surface.GetOffScreenRendering() and not surface.GetShowWindow()
            assert window.project.fingerprint == project.fingerprint
            assert viewport._metadata["project_fingerprint"] == project.fingerprint
            assert viewport.dataset.currentData() == "review_conductor"
            assert sum(key == "winding" for key in viewport._actors.values()) == 1
            source_points = viewport._arrays["review_conductor_0_points_m"]
            rendered = viewport._fit_meshes[-1].points
            np.testing.assert_array_equal(source_points, rendered)
            viewport._fit_view()
            app.processEvents()
            viewport_width, viewport_height = plotter.width(), plotter.height()
            assert viewport_width > 0 and viewport_height > 0
            frame = np.ascontiguousarray(plotter.screenshot(return_img=True)[:, :, :3])
            assert float(np.ptp(frame)) > 80
            pixels = []
            for point in rendered:
                plotter.renderer.SetWorldPoint(*point, 1)
                plotter.renderer.WorldToDisplay()
                pixels.append(plotter.renderer.GetDisplayPoint()[:2])
            pixels = np.asarray(pixels)
            assert pixels.min() >= 0 and np.all(pixels.max(axis=0) < frame.shape[1::-1])
            captured_ui = window.grab()
            painter = QPainter(captured_ui)
            origin = plotter.mapTo(window, QPoint())
            raster = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format.Format_RGB888)
            painter.drawImage(origin.x(), origin.y(), raster.scaled(viewport_width, viewport_height))
            painter.end()
            assert captured_ui.save(str(output))
            window.view_context.design_button.click()
            deadline = time.monotonic() + 30
            while not window.findChildren(SavedSettingsDialog) and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(0.01)
            reader = window.findChildren(SavedSettingsDialog)[-1]
            tree = reader.findChild(QTreeWidget, "submitted_design_choices")
            local = tree.findItems(
                "Locally fitted (included in smoothed count)",
                Qt.MatchFlag.MatchExactly | Qt.MatchFlag.MatchRecursive,
                0,
            )
            reader_values = [item.text(1) for item in local]
            assert reader_values == [
                str(part["local_fit_count"])
                for part in viewport._metadata["contour_smoothing"]
                if "local_fit_count" in part
            ]
            assert reader.grab().save(str(output.with_stem(output.stem + "-choices")))
            reader.close()
            window.select_node("winding")
            control = window._draft_controls["contour_smoothing_local_fallback"]
            assert control.isChecked() == project.config.contour_smoothing_local_fallback
            stages = {key: item.text(1) for key, item in window.stage_items.items()}
            assert all(
                value in {"Cached", "Computed", "Complete", "Checks failed", "Unverified"} for value in stages.values()
            )
            print(
                json.dumps(
                    {
                        "case": project.name,
                        "request_sha256": viewport._metadata["request_sha256"],
                        "plot": viewport.caption.text(),
                        "result_badge": window.results_overview.badge.text(),
                        "stages": stages,
                        "points": len(rendered),
                        "viewport_px": [frame.shape[1], frame.shape[0]],
                        "centerline_px": [pixels.min(axis=0).tolist(), pixels.max(axis=0).tolist()],
                        "reader_local_counts": reader_values,
                        "editor_local_enabled": control.isEnabled(),
                        "editor_local_checked": control.isChecked(),
                    },
                    ensure_ascii=True,
                )
            )
        finally:
            window.undo_stack.setClean()
            window.close()
            app.processEvents()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scratch", type=Path)
    args = parser.parse_args()
    capture(args.source, args.run, args.output, args.scratch)
