"""Capture the production desktop GUI on white, using a verified saved example.

python docs/frontpage/capture_desktop_tour.py --project example.coilgen --run RUN

Qt widgets and their VTK framebuffers are captured at the same native bounds.
The framebuffer registration restores OpenGL pixels omitted by QWidget.grab();
it does not alter plotted geometry, numerical values or application controls.
"""

# Configure the Qt and plotting backends before importing their modules.
# ruff: noqa: E402

from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_SCALE_FACTOR"] = "2"
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pyvistaqt
from pyvista.plotting.render_window_interactor import RenderWindowInteractor
from vtkmodules.vtkRenderingCore import vtkRenderWindow
from PySide6.QtCore import QPoint, QRect, Qt, QThreadPool
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPalette
from PySide6.QtWidgets import QApplication, QCheckBox, QDockWidget, QScrollArea
from coilgen.gui.workbench import CoilGenWorkbench


class PrivateFramebuffer(type(vtkRenderWindow())):
    def SetWindowInfo(self, _qt_id):
        pass


_interactor = pyvistaqt.QtInteractor


def isolated_interactor(*args, **kwargs):
    surface = PrivateFramebuffer()
    surface.SetOffScreenRendering(1)
    surface.SetShowWindow(False)
    kwargs.update(rw=surface, off_screen=True, auto_update=0)
    plotter = _interactor(*args, **kwargs)
    plotter.iren = RenderWindowInteractor(plotter, interactor=surface.GetInteractor())
    plotter.iren.initialize()
    return plotter


class DesktopTour:
    def __init__(self, args):
        self.args = args
        self.output = args.output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.scratch = ROOT / "tmp/desktop-tour-capture"
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.app = QApplication.instance() or QApplication([])
        self.app.setOrganizationName("CoilGen2 documentation")
        self.app.setApplicationName("Desktop feature tour")
        self.app.setStyle("Fusion")
        if Path("C:/Windows/Fonts/segoeui.ttf").exists():
            QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
        self.app.setFont(QFont("Segoe UI", 11))
        palette = QPalette()
        for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base, QPalette.ColorRole.Button):
            palette.setColor(role, QColor("#ffffff"))
        for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
            palette.setColor(role, QColor("#213b49"))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#f2f6f8"))
        palette.setColor(QPalette.ColorRole.Highlight, QColor("#24697d"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
        self.app.setPalette(palette)
        self.app.setStyleSheet("QToolTip {color:#213b49;background:white;} QGroupBox {font-weight:600;}")
        pyvistaqt.QtInteractor = isolated_interactor
        manifest = self.output / "capture-manifest.json"
        self.records = json.loads(manifest.read_text(encoding="utf-8"))["screenshots"] if manifest.exists() else []
        self.window = CoilGenWorkbench(jobs_directory=self.scratch / "jobs")
        self.window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        self.window.resize(1440, 960)
        self.window.timer.stop()
        self.window.show()
        self.errors = []
        self.window._error = self.errors.append

    def settle(self, predicate=None, timeout=90):
        deadline = time.monotonic() + timeout
        stable = 0
        while time.monotonic() < deadline:
            self.app.processEvents()
            ready = predicate is None or predicate()
            stable = stable + 1 if ready else 0
            if stable >= 10:
                return
            time.sleep(0.03)
        raise TimeoutError("The requested native view did not finish: " + str(self.errors))

    def capture(self, name, widget, title, *, width=1440, height=960):
        widget.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        widget.resize(width, height)
        widget.show()
        self.settle()
        plotters = [p for p in widget.findChildren(_interactor) if p.isVisibleTo(widget)]
        for plotter in plotters:
            plotter.set_background("white")
            plotter.render()
        self.settle()
        pixmap = widget.grab()
        painter = QPainter(pixmap)
        registered = []
        for plotter in plotters:
            child = plotter.interactor
            top = child.mapTo(widget, QPoint(0, 0))
            ratio = widget.devicePixelRatioF()
            size = (round(child.width() * ratio), round(child.height() * ratio))
            if min(size) <= 0:
                continue
            pixels = np.ascontiguousarray(plotter.screenshot(return_img=True, window_size=size)[:, :, :3])
            image = QImage(
                pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0], QImage.Format.Format_RGB888
            ).copy()
            painter.drawImage(QRect(top, child.size()), image)
            registered.append(dict(x=top.x(), y=top.y(), width=child.width(), height=child.height()))
        painter.end()
        path = self.output / (name + ".png")
        assert pixmap.save(str(path))
        entry = dict(
            file=path.name,
            title=title,
            pixels=[pixmap.width(), pixmap.height()],
            logical_size=[widget.width(), widget.height()],
            scale=widget.devicePixelRatioF(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            vtk_registration=registered,
        )
        self.records = sorted(
            [r for r in self.records if r["file"] != entry["file"]] + [entry], key=lambda r: r["file"]
        )
        print("CAPTURED", path.name, entry["pixels"], flush=True)
        self.manifest()

    def manifest(self):
        payload = dict(
            style="White background; Fusion light theme; Segoe UI 11 pt; native 2× pixel density",
            scope="Production desktop widgets; actual verified saved result; no invented numerical state",
            source_run=self.args.run.name,
            screenshots=self.records,
            errors=self.errors,
        )
        (self.output / "capture-manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def open_result(self):
        self.window.open_project(self.args.project)
        self.settle()
        self.record = self.window.load_run(self.args.run, id_="desktop-tour")
        self.window.show_run("desktop-tour")
        self.settle(lambda: not getattr(self.window.viewport, "_preview_worker", None))
        if self.errors:
            raise RuntimeError(self.errors)

    def workbench(self):
        self.capture("01-welcome", self.window, "Start a design or open a portable project")
        self.open_result()
        self.window.design_plan_dock.hide()
        self.window.viewport.plotter.reset_camera()
        self.window.results_overview.metrics_button.setChecked(False)
        self.capture("02-workbench", self.window, "Design setup, saved results and the 3D winding")
        self.window.results_overview.metrics_button.setChecked(False)
        self.window.viewport.dataset.setCurrentIndex(self.window.viewport.dataset.findData("stream_function"))
        self.capture("03-stream-function", self.window, "Inspect the optimized surface-current stream function")
        self.window.viewport.dataset.setCurrentIndex(self.window.viewport.dataset.findData("swept"))
        self.capture("04-swept-conductor", self.window, "Inspect the swept conductor volume")
        self.window.design_plan_dock.show()
        self.capture("05-design-plan", self.window, "Follow saved stages and reproducible computations")
        self.window.design_plan_dock.hide()

    def pcb(self):
        from coilgen.application.pcb import PcbSession
        from coilgen.gui.pcb_layout import PcbLayoutDialog

        session = PcbSession.from_record(self.record, workspace=self.scratch)
        session.change("settings", mode="spirals", width_m=0.0015, seam_deg=0)
        dialog = PcbLayoutDialog(session)
        self.capture("12-pcb-layout", dialog, "Develop flat PCB artwork with a linked installed-coil preview")
        dialog.open_tab(2)
        dialog.end_a.setCurrentIndex(dialog.end_a.findData("g1-in:end"))
        dialog.end_b.setCurrentIndex(dialog.end_b.findData("g1-out:start"))
        self.capture(
            "13-pcb-connections", dialog, "Preview endpoint connections, layer transitions and clearance concerns"
        )
        dialog.open_tab(3)
        groups = dialog.scene["issue_groups"]
        row = next((i for i, g in enumerate(groups) if g["key"] == "off_surface"), 0)
        dialog.checks.setCurrentRow(row)
        self.capture("14-pcb-checks", dialog, "Inspect grouped PCB checks and saved-path displacement")
        dialog.close()

    def settings(self):
        for key, name, title in (
            ("geometry", "06-supports", "Choose support geometry and mesh resolution"),
            ("target", "07-target", "Define the target field and its physical region"),
            ("optimization", "08-optimization", "Set objectives, regularization and field-error constraints"),
            ("material", "09-conductor", "Define conductor dimensions and cross section"),
            ("winding", "10-winding", "Control contour spacing, smoothing and routing"),
        ):
            self.window._inspect_node(key)
            self.settle()
            dock = self.window.findChild(QDockWidget, "settings_dock")
            dock.setFloating(True)
            self.capture(name, dock, title, width=1000, height=720)
            dock.hide()

    def analysis(self):
        from coilgen.gui.field_analysis import FieldAnalysisDialog

        dialog = FieldAnalysisDialog(self.record, initial_view="metrics")
        dialog.section_resolution.setCurrentIndex(dialog.section_resolution.findData(64))
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        self.settle(lambda: dialog.state is not None and dialog.worker is None and not dialog.debounce.isActive(), 180)
        self.capture("15-field-maps", dialog, "Compare target, calculated field and residual on matched scales")
        dialog.focus_view("slices")
        self.settle(lambda: dialog.worker is None and not dialog.debounce.isActive(), 120)
        self.capture("16-field-slices", dialog, "Inspect physical field sections through the target region")
        dialog.focus_view("errors")
        self.settle(lambda: dialog.worker is None and not dialog.debounce.isActive())
        self.capture("17-field-errors", dialog, "Inspect the spatial distribution of field error")
        dialog.fit_harmonics()
        self.settle(lambda: dialog.worker is None and not dialog.debounce.isActive())
        self.capture("18-harmonics", dialog, "Resolve field errors into solid-harmonic components")
        dialog.focus_view("coil")
        self.settle(lambda: dialog.worker is None and not dialog.debounce.isActive())
        self.capture(
            "19-coil-statistics", dialog, "Review recorded current, resistance, inductance and conductor properties"
        )
        dialog.close()

    def routing(self):
        from coilgen.application.interconnection_editor import WindingEditorSession
        from coilgen.gui.interconnection_editor import InterconnectionEditorDialog

        dialog = InterconnectionEditorDialog(WindingEditorSession.from_record(self.record))
        dialog.view_layout.setCurrentIndex(0)
        self.capture("11-interconnections", dialog, "Edit interconnections with linked 3D and flat views")
        dialog.close()

    def groups(self):
        from coilgen.pipeline_stages import PipelineCheckpoint
        from coilgen.application.group_editor import GroupEditor
        from coilgen.gui.group_circuits import GroupsCircuitsEditor

        # Inspect the owned contour snapshot at its original stage, before later routing edits.
        checkpoint = PipelineCheckpoint.load(self.record.directory / "checkpoints/contours.cgstage")
        groups = GroupsCircuitsEditor()
        groups.set_editor(GroupEditor.from_parts(checkpoint.parts, checkpoint.config), "")
        groups.note.setText(
            "Groups refer to the verified contour snapshot. Spiral order preserves current circulation."
        )
        self.capture("21-groups-circuits", groups, "Assign contour groups to circuits and choose spiral order")
        groups.close()

    def geometry(self):
        from coilgen.gui.surface_editor import SurfaceEditorDialog

        dialog = SurfaceEditorDialog(None)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        self.settle(lambda: dialog.task is None)
        evidence = json.loads((ROOT / "docs/images/welcome-geometry-evidence.json").read_text(encoding="utf-8"))
        recipe = evidence["figures"][0]["model"]
        dialog._start(lambda: dialog.editor.edit(recipe), "Open windowed-cylinder CAD example")
        self.settle(lambda: dialog.task is None)
        dialog._start(dialog.editor.generate, "Mesh retained surface")
        self.settle(lambda: dialog.task is None)
        dialog.show_mesh.setChecked(True)
        dialog.view.camera_position = [(0.32, -0.5, 0.28), (0, 0, 0.01), (0, 0, 1)]
        dialog.view.reset_camera()
        self.capture("20-geometry-studio", dialog, "Build, subtract, select faces and mesh support surfaces")
        dialog.close()

    def engineering(self):
        self.window._results_action("checks")
        self.settle()
        panel = self.window.evidence.window_for(self.window.acceptance_panel)
        for control in panel.findChildren(QCheckBox):
            if control.text() == "Show all checks":
                control.setChecked(True)
        self.capture("22-engineering-checks", panel, "Review saved engineering evidence and acceptance checks")
        panel.close()
        self.window.navigate_workflow("export")
        self.settle()
        dock = self.window.findChild(QDockWidget, "settings_dock")
        dock.setFloating(True)
        self.capture(
            "23-geometry-export",
            dock,
            "Export conductor geometry from the selected saved result",
            width=1100,
            height=850,
        )
        dock.hide()
        self.window._inspect_node("print")
        self.settle()
        for control in dock.findChildren(QCheckBox):
            if control.isVisibleTo(dock):
                control.setChecked(True)
                break
        self.settle()
        dock.setFloating(True)
        self.capture(
            "24-print-former", dock, "Configure a printable winding former and wire grooves", width=1000, height=720
        )
        dock.hide()

    def systems(self):
        self.window._draft_dirty = False
        self.window.undo_stack.setClean()
        self.window.load_template("shielded_xyz_gradient")
        self.settle(lambda: not getattr(self.window.viewport, "_preview_worker", None))
        self.window.design_plan_dock.hide()
        for area in self.window.setup_panel.findChildren(QScrollArea):
            area.verticalScrollBar().setValue(0)
        if isinstance(self.window.setup_panel, QScrollArea):
            self.window.setup_panel.verticalScrollBar().setValue(0)
        self.window.viewport.plotter.reset_camera()
        self.capture("25-shielded-system", self.window, "Configure the actively shielded XYZ gradient example")
        self.window._channels_dialog()
        self.capture(
            "26-channel-overview",
            self.window.channels_window,
            "Manage independent channels and inspect each execution state",
            height=520,
        )
        self.window.channels_window.close()

    def close(self):
        self.window._draft_dirty = False
        self.window.undo_stack.setClean()
        self.window.close()
        self.app.processEvents()
        QThreadPool.globalInstance().waitForDone(15000)
        self.manifest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/images/desktop-tour")
    parser.add_argument(
        "--sections",
        nargs="+",
        default=["workbench", "settings", "pcb", "analysis", "routing", "groups", "geometry", "engineering", "systems"],
    )
    args = parser.parse_args()
    tour = DesktopTour(args)
    try:
        if "workbench" not in args.sections:
            tour.open_result()
        for section in args.sections:
            getattr(tour, section)()
    finally:
        tour.close()


if __name__ == "__main__":
    main()
