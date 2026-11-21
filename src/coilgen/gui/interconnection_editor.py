"""Native 3D winding editor. Camera interaction never changes the circuit."""

from __future__ import annotations

from pathlib import Path
from threading import Event

import numpy as np
from PySide6.QtCore import QEvent, QObject, QRunnable, QThreadPool, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTextBrowser,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..models import WirePathGeometry
from ..winding_visuals import group_color, direction_markers
from ..winding_review import local_frame, review_issues, winding_sequence


def _icon(name):
    from PySide6.QtSvg import QSvgRenderer

    paths = {
        "select": "M5 3 L19 13 L12 14 L9 21 Z",
        "draw": "M4 18 C7 2 14 23 20 5 M17 5 L20 5 L20 8",
        "seam": "M5 5 L19 19 M5 19 L19 5",
        "shape": "M3 7 L15 3 L21 17 L9 21 Z M12 2 L12 22",
        "connect": "M4 12 L20 12 M15 7 L20 12 L15 17",
        "bridge": "M3 16 L8 16 C8 3 16 3 16 16 L21 16 M12 3 L12 7 M12 17 L12 21",
        "undo": "M8 5 L3 10 L8 15 M3 10 L15 10 C23 10 23 20 15 20",
        "redo": "M16 5 L21 10 L16 15 M21 10 L9 10 C1 10 1 20 9 20",
        "lock": "M7 10 L7 6 C7 0 17 0 17 6 L17 10 M5 10 L19 10 L19 21 L5 21 Z",
        "cut": "M6 6 L20 20 M6 18 L20 4 M3 6 A3 3 0 1 0 3 5 M3 19 A3 3 0 1 0 3 18",
        "save": "M12 3 L12 16 M7 11 L12 16 L17 11 M4 18 L4 21 L20 21 L20 18",
        "open": "M3 7 L10 7 L12 10 L21 10 L19 21 L3 21 Z",
        "generate": "M4 20 L18 6 M15 3 L21 9 M4 3 L4 9 M1 6 L7 6 M19 15 L19 21 M16 18 L22 18",
        "details": "M4 5 L20 5 M4 12 L20 12 M4 19 L20 19",
    }
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="{paths[name]}" fill="none" stroke="#36576c" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    pixmap = QPixmap(24, 24)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    QSvgRenderer(svg.encode()).render(painter)
    painter.end()
    return QIcon(pixmap)


class _Signals(QObject):
    finished = Signal(object, str)
    progress = Signal(dict)


class _Task(QRunnable):
    def __init__(self, operation):
        super().__init__()
        self.operation, self.signals = operation, _Signals()
        self.cancel_requested = Event()

    def run(self):
        from ..progress import DesignCancelled, ProgressRelay, progress_context

        relay = ProgressRelay(self.signals.progress.emit)
        try:
            with progress_context(relay.report, cancel_check=self.cancel_requested.is_set):
                if self.cancel_requested.is_set():
                    raise DesignCancelled("Operation cancelled.")
                value = self.operation()
            relay.flush()
            self.signals.finished.emit(value, "")
        except Exception as error:
            relay.flush()
            self.signals.finished.emit(None, str(error))


class WindingViewport(QWidget):
    selected = Signal(str)
    cut_moved = Signal(str, object)
    guide_moved = Signal(str, int, object)
    message = Signal(str)
    seam_drawn = Signal(object, bool)
    connection_drawn = Signal(str, str, object, bool)
    port_picked = Signal(str, str)
    ports_connected = Signal(str, str)
    bridge_target = Signal(str, object)
    shape_moved = Signal(str, object)
    bridge_height_moved = Signal(str, object)
    object_selected = Signal(str)
    opening_selected = Signal(str, int)
    shape_placed = Signal(object)
    issue_selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        import pyvista as pv
        from pyvistaqt import QtInteractor

        self.pv = pv
        self.plotter = QtInteractor(self, off_screen=QApplication.platformName() == "offscreen")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plotter.interactor)
        self.plotter.interactor.installEventFilter(self)
        self.plotter.add_axes()
        self.editor = None
        self.current_key = ""
        self.busy = False
        self._drag = None
        self._actors = {}
        self._support = None
        self._handles = {}
        self._first = True
        self.radius = 0.001
        self.mode = "select"
        self.preview = None
        self._sketch = None
        self._shape_drag = None
        self.hide_rear = False
        self.isolate_group = None
        self.place_on_surface = False
        self.issues = []
        self._object_actors = {}
        self._issue_actors = {}

    def _line(self, points, name, color, width=3, opacity=1, pickable=False):
        return self.plotter.add_mesh(
            self.pv.lines_from_points(points),
            name=name,
            color=color,
            line_width=width,
            opacity=opacity,
            pickable=pickable,
            reset_camera=False,
            render=False,
        )

    def display(self, editor, key, *, baseline=False, physical_part=None):
        self.editor, self.current_key = editor, key
        camera = self.plotter.camera_position if not self._first else None
        self.plotter.clear()
        self._actors, self._handles = {}, {}
        self._object_actors, self._issue_actors = {}, {}
        support = editor.support
        dark = self.palette().window().color().lightness() < 128
        self.plotter.set_background("#17212b" if dark else "#f4f7f9")
        mesh = self.pv.PolyData(support.vertices, np.column_stack((np.full(len(support.faces), 3), support.faces)))
        self._support = self.plotter.add_mesh(
            mesh, color="#dce5eb", opacity=1 if self.hide_rear else 0.16, pickable=True, render=False
        )
        # Display-only depth bias keeps surface-coincident wires visible while
        # an opaque support occludes rear wires. Physical vertices stay exact.
        self._support.GetMapper().SetResolveCoincidentTopologyToPolygonOffset()
        self._support.GetMapper().SetRelativeCoincidentTopologyPolygonOffsetParameters(2.0, 2.0)
        if physical_part is not None and physical_part.swept_body is not None:
            body = physical_part.swept_body
            solid = self.pv.PolyData(body.vertices, np.column_stack((np.full(len(body.faces), 3), body.faces)))
            self.plotter.add_mesh(solid, color="#c39666", opacity=0.4, pickable=False, render=False)
        self.radius = max(float(np.linalg.norm(np.ptp(support.vertices, axis=0))) * 0.008, 1e-6)
        if baseline:
            self._line(editor.automatic_path.v, "automatic", "#91a3af", 1, 0.35)
        for section in editor.sections:
            if self.isolate_group is not None and section.owner.group_index != self.isolate_group:
                continue
            chosen = section.key == key
            color = group_color(section.owner.group_index)
            actor = self._line(
                section.points, section.key, color, 5 if chosen else 2.5, 0.25 if section.disconnected else 1, True
            )
            self._actors[actor.GetAddressAsString("")] = section.key
            if not section.editable and section.owner.loop_index >= 0:
                markers = direction_markers(section.points, self.radius * 20)
                if markers:
                    glyphs = self.pv.PolyData(np.array([p for p, _ in markers]))
                    glyphs["direction"] = np.array([v for _, v in markers])
                    arrows = glyphs.glyph(orient="direction", scale=False, factor=self.radius * 0.9)
                    self.plotter.add_mesh(arrows, color=color, pickable=False, render=False)
            if chosen:
                if self.mode in {"select", "bridge"} and section.bridge and not section.locked:
                    self._handle(editor.bridge_handle(key)["point"], (key, "bridge_height", -1))
                elif self.mode == "select" and section.cut is not None and not section.locked:
                    self._handle(editor.cut_point(key), (key, "cut", -1))
                elif (
                    self.mode == "select"
                    and section.editable
                    and not section.disconnected
                    and not section.bridge
                    and not section.locked
                ):
                    guides = section.guides or (tuple(section.points[len(section.points) // 2]),)
                    for index, point in enumerate(guides):
                        self._handle(point, (key, "guide", index))
                mid = max(0, (len(section.points) - 2) // 2)
                vector = section.points[mid + 1] - section.points[mid]
                arrow = self.pv.Arrow(start=section.points[mid], direction=vector, scale=self.radius * 3)
                self.plotter.add_mesh(arrow, color=color, pickable=False, render=False)
        if self.mode in {"connect", "draw"}:
            for port in editor.drawing_ports(key) if self.mode == "draw" else editor.ports:
                self._handle(port["point"], (port["key"], port["role"], -1))
        if self.preview and self.preview["revision"] == editor.revision:
            for route in self.preview.get("routes", []):
                self._line(np.asarray(route["points"]), f"proposal-{route['key']}", "#0b9b91", 4)
            if "shape" in self.preview:
                for i, line in enumerate(self.preview["display"]["lines"]):
                    self._line(np.asarray(line), f"cut-shape-{i}", "#a766c1", 2)
                for cut in self.preview["cuts"]:
                    self._line(np.asarray(cut["opening"]), f"cut-opening-{cut['key']}", "#e05b47", 6)
                for cut in self.preview.get("alternatives", []):
                    if not cut["selected"]:
                        self._line(np.asarray(cut["opening"]), f"alternative-{cut['key']}-{cut['index']}", "#e5a38f", 4)
                        self._handle(cut["point"], ("choice", cut["key"], cut["index"]))
                if self.mode == "shape":
                    for handle in self.preview["display"]["handles"]:
                        self._handle(handle["point"], ("shape", handle["role"], -1))
                    labels = [
                        (h["point"], f"{self.preview['shape']['size'][int(h['role'][4:])] * 2000:.2f} mm")
                        for h in self.preview["display"]["handles"]
                        if h["role"].startswith("size")
                    ]
                    if labels:
                        self.plotter.add_point_labels(
                            np.array([p for p, _ in labels]),
                            [s for _, s in labels],
                            font_size=11,
                            point_size=0,
                            always_visible=True,
                            render=False,
                        )
            elif "stroke" in self.preview:
                points = np.asarray([editor.chart.pick(uv) for uv in self.preview["stroke"]])
                self._line(points, "seam-preview", "#007f83", 4)
        for obj in editor.cut_objects:
            if self.preview and obj["id"] == self.preview.get("object_id"):
                continue
            for i, line in enumerate(obj["display"]["lines"]):
                actor = self._line(np.asarray(line), f"object-{obj['id']}-{i}", "#bba4c6", 2, pickable=True)
                self._object_actors[actor.GetAddressAsString("")] = obj["id"]
        for i, issue in enumerate(self.issues):
            if issue.get("point") is not None:
                actor = self.plotter.add_points(
                    np.asarray(issue["point"])[None],
                    point_size=10,
                    render_points_as_spheres=True,
                    color="#c48a2d",
                    pickable=True,
                    render=False,
                )
                self._issue_actors[actor.GetAddressAsString("")] = i
        self.plotter.add_axes()
        if camera is not None:
            self.plotter.camera_position = camera
        else:
            self.plotter.view_isometric()
            self.plotter.reset_camera()
            self._first = False
        self.plotter.render()

    def face_selection(self, local=False):
        frame = local_frame(self.editor, self.current_key)
        center, normal = np.array(frame["center"]), np.array(frame["normal"])
        if normal @ (np.array(self.plotter.camera.position) - center) < 0:
            normal = -normal
        up = np.eye(3)[np.argmin(abs(normal))]
        distance = np.linalg.norm(np.ptp(self.editor.support.vertices, axis=0))
        self.plotter.camera_position = [center + normal * distance * 2, center, up]
        self.plotter.enable_parallel_projection()
        self.plotter.camera.parallel_scale = distance * (0.16 if local else 0.4)
        if local:
            self.plotter.add_text("Local projection - distances distorted", name="local-label", font_size=10)
        else:
            self.plotter.remove_actor("local-label", render=False)
        self.plotter.render()

    def focus_issue(self, issue):
        if issue.get("point") is None:
            return
        center = np.asarray(issue["point"])
        offset = np.array(self.plotter.camera.position) - self.plotter.camera.focal_point
        self.plotter.camera.focal_point = center
        self.plotter.camera.position = center + offset
        self.plotter.camera.zoom(1.6)
        for i, (a, b) in enumerate(issue.get("segments", [])):
            self._line(np.asarray([issue["points"][a], issue["points"][b]]), f"issue-witness-{i}", "#c84938", 6)
        self.plotter.add_point_labels(
            center[None], [issue["label"]], name="focused-issue", always_visible=True, font_size=12
        )

    def playback(self, key, fraction):
        rows = {s.key: s for s in self.editor.sections}
        for actor in self.plotter.renderer.actors.values():
            name = self._actors.get(actor.GetAddressAsString(""))
            if name and hasattr(actor, "GetProperty"):
                actor.GetProperty().SetOpacity(1 if not key or name == key else 0.18)
        self.plotter.remove_actor("playback-dot", render=False)
        if key in rows:
            points = rows[key].points
            station = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            p = np.array([np.interp(fraction * station[-1], station, points[:, i]) for i in range(3)])
            self.plotter.add_points(
                p[None],
                point_size=16,
                render_points_as_spheres=True,
                name="playback-dot",
                color="#007f83",
                pickable=False,
                render=False,
            )
        self.plotter.render()

    def _handle(self, point, identity):
        color = "#d97724" if identity[1] == "in" else "#159884"
        if identity[0] == "shape":
            color = {"center": "#a766c1", "normal": "#307caf"}.get(identity[1], "#ffffff")
        actor = self.plotter.add_points(
            np.asarray(point)[None],
            point_size=15,
            render_points_as_spheres=True,
            color=color,
            pickable=True,
            render=False,
        )
        self._handles[actor.GetAddressAsString("")] = identity

    def _pick(self, position, actors):
        from vtkmodules.vtkRenderingCore import vtkCellPicker

        picker = vtkCellPicker()
        # VTK expresses tolerance as a fraction of the window diagonal.
        # Keep edit targets usable without swallowing neighbouring turns when
        # the user zooms in on a millimetre-scale cut or connection.
        picker.SetTolerance(8 / max(np.hypot(self.plotter.interactor.width(), self.plotter.interactor.height()), 1))
        picker.PickFromListOn()
        for actor in actors:
            picker.AddPickList(actor)
        ratio = self.plotter.interactor.devicePixelRatioF()
        picker.Pick(
            position.x() * ratio,
            (self.plotter.interactor.height() - position.y() - 1) * ratio,
            0,
            self.plotter.renderer,
        )
        actor, point = picker.GetActor(), np.asarray(picker.GetPickPosition())
        if actor is not None and self.hide_rear and actor is not self._support:
            front, surface = self._pick(position, [self._support])
            eye = np.array(self.plotter.camera.position)
            if front is not None and np.linalg.norm(point - eye) > np.linalg.norm(surface - eye) + self.radius * 1.5:
                return None, point
        return actor, point

    def _drag_point(self, event):
        section = self.editor.section(self._drag[0])
        if self._drag[1] != "bridge_height" and (
            self._drag[1] == "cut" or WirePathGeometry.FREE_3D not in section.geometry
        ):
            actor, point = self._pick(event.position(), [self._support])
            return point if actor is not None else None
        # Explicit free-space connections move in the screen plane at the picked
        # depth. This does not silently turn a surface route into a 3D jump.
        renderer = self.plotter.renderer
        renderer.SetWorldPoint(*self._drag[3], 1)
        renderer.WorldToDisplay()
        depth = renderer.GetDisplayPoint()[2]
        ratio = self.plotter.interactor.devicePixelRatioF()
        renderer.SetDisplayPoint(
            event.position().x() * ratio, (self.plotter.interactor.height() - event.position().y() - 1) * ratio, depth
        )
        renderer.DisplayToWorld()
        point = np.asarray(renderer.GetWorldPoint())
        return point[:3] / point[3] if point[3] else None

    def eventFilter(self, watched, event):
        if watched is not self.plotter.interactor or self.editor is None or self.busy:
            return super().eventFilter(watched, event)
        if (
            self.mode == "bridge"
            and event.type() == QEvent.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.LeftButton
        ):
            handles = [
                a
                for a in self.plotter.renderer.actors.values()
                if self._handles.get(a.GetAddressAsString(""), (None, None))[1] == "bridge_height"
            ]
            actor, point = self._pick(event.position(), handles)
            if actor:
                key, kind, index = self._handles[actor.GetAddressAsString("")]
                self._drag = (key, kind, index, point.copy(), event.position(), None)
                return True
        if (
            event.type() == QEvent.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.LeftButton
            and not event.modifiers()
            and self.mode in {"select", "shape"}
        ):
            actors = list(self.plotter.renderer.actors.values())
            handle, _ = self._pick(
                event.position(),
                [
                    a
                    for a in actors
                    if a.GetAddressAsString("") in self._handles
                    and self._handles[a.GetAddressAsString("")][0] != "choice"
                ],
            )
            # Active handles and drawing tools take priority over cut-object outlines.
            if handle is None:
                choice_actors = [
                    a for a in actors if self._handles.get(a.GetAddressAsString(""), (None,))[0] == "choice"
                ]
                actor, _ = self._pick(event.position(), choice_actors)
                if actor:
                    _, key, index = self._handles[actor.GetAddressAsString("")]
                    self.opening_selected.emit(key, index)
                    return True
                actor, _ = self._pick(
                    event.position(), [a for a in actors if a.GetAddressAsString("") in self._object_actors]
                )
                if actor:
                    self.object_selected.emit(self._object_actors[actor.GetAddressAsString("")])
                    return True
                if self.mode == "select":
                    actor, _ = self._pick(
                        event.position(), [a for a in actors if a.GetAddressAsString("") in self._issue_actors]
                    )
                    if actor:
                        self.issue_selected.emit(self._issue_actors[actor.GetAddressAsString("")])
                        return True
            if self.mode == "shape" and self.place_on_surface and handle is None:
                actor, point = self._pick(event.position(), [self._support])
                if actor:
                    self.shape_placed.emit(point)
                return True
        if self.mode == "shape":
            if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
                self._shape_drag = None
                return True
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                actors = [a for a in self.plotter.renderer.actors.values() if a.GetAddressAsString("") in self._handles]
                actor, point = self._pick(event.position(), actors)
                if actor:
                    role = self._handles[actor.GetAddressAsString("")][1]
                    self._shape_drag = (role, point, None)
                    return True
            if event.type() == QEvent.Type.MouseMove and self._shape_drag:
                role, start, _ = self._shape_drag
                if role == "center":
                    actor, point = self._pick(event.position(), [self._support])
                    point = point if actor else None
                else:
                    renderer = self.plotter.renderer
                    renderer.SetWorldPoint(*start, 1)
                    renderer.WorldToDisplay()
                    depth = renderer.GetDisplayPoint()[2]
                    ratio = self.plotter.interactor.devicePixelRatioF()
                    renderer.SetDisplayPoint(
                        event.position().x() * ratio,
                        (self.plotter.interactor.height() - event.position().y() - 1) * ratio,
                        depth,
                    )
                    renderer.DisplayToWorld()
                    world = np.asarray(renderer.GetWorldPoint())
                    point = world[:3] / world[3]
                self._shape_drag = (role, start, point)
                if point is not None:
                    self._line(np.array([start, point]), "shape-drag", "#a766c1", 4)
                    self.plotter.render()
                return True
            if event.type() == QEvent.Type.MouseButtonRelease and self._shape_drag:
                role, _, point = self._shape_drag
                self._shape_drag = None
                self.plotter.remove_actor("shape-drag", render=True)
                if point is not None:
                    self.shape_moved.emit(role, point)
                return True
        if self.mode in {"seam", "connect", "bridge", "draw"} and not self._drag:
            if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
                self._sketch = None
                self.plotter.remove_actor("sketch-preview", render=True)
                return True
            if (
                event.type() == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
                and not event.modifiers()
            ):
                if self.mode == "seam":
                    actor, point = self._pick(event.position(), [self._support])
                    self._sketch = {"stroke": [point], "press": event.position()} if actor else None
                else:
                    actors = [
                        a
                        for a in self.plotter.renderer.actors.values()
                        if a.GetAddressAsString("")
                        in (self._handles if self.mode in {"connect", "draw"} else self._actors)
                        and (self.mode != "bridge" or self._actors.get(a.GetAddressAsString("")) != self.current_key)
                    ]
                    actor, point = self._pick(event.position(), actors)
                    if actor:
                        address = actor.GetAddressAsString("")
                        if self.mode == "bridge":
                            self.bridge_target.emit(self._actors[address], point)
                        else:
                            self._sketch = {"port": self._handles[address], "press": event.position()}
                            if self.mode == "draw" and self._handles[address][1] == "out":
                                self._sketch["stroke"] = [self.editor.section(self._handles[address][0]).points[0]]
                    elif self.mode == "draw":
                        editable = {s.key for s in self.editor.sections if s.editable}
                        actors = [
                            a
                            for a in self.plotter.renderer.actors.values()
                            if self._actors.get(a.GetAddressAsString("")) in editable
                        ]
                        actor, _ = self._pick(event.position(), actors)
                        if actor:
                            self.selected.emit(self._actors[actor.GetAddressAsString("")])
                return True
            if event.type() == QEvent.Type.MouseMove and self._sketch:
                if "stroke" in self._sketch:
                    actor, point = self._pick(event.position(), [self._support])
                    stroke = self._sketch["stroke"]
                    if actor and len(stroke) < 512 and np.linalg.norm(point - stroke[-1]) > self.radius * 0.1:
                        stroke.append(point)
                        self._line(np.asarray(stroke), "sketch-preview", "#007f83", 4)
                        self.plotter.render()
                return True
            if event.type() == QEvent.Type.MouseButtonRelease and self._sketch:
                sketch, self._sketch = self._sketch, None
                self.plotter.remove_actor("sketch-preview", render=True)
                if "stroke" in sketch and "port" in sketch:
                    actors = [
                        a for a in self.plotter.renderer.actors.values() if a.GetAddressAsString("") in self._handles
                    ]
                    actor, _ = self._pick(event.position(), actors)
                    target = self._handles.get(actor.GetAddressAsString("")) if actor else None
                    if target and target[1] == "in" and len(sketch["stroke"]) >= 2:
                        stroke = [*sketch["stroke"], self.editor.section(target[0]).points[0]]
                        self.connection_drawn.emit(sketch["port"][0], target[0], np.asarray(stroke).tolist(), False)
                    else:
                        self.message.emit("Finish on an input endpoint.")
                elif "stroke" in sketch and len(sketch["stroke"]) >= 2:
                    actor, end = self._pick(event.position(), [self._support])
                    if actor and np.linalg.norm(end - sketch["stroke"][-1]) > 1e-12:
                        if len(sketch["stroke"]) >= 512:
                            sketch["stroke"][-1] = end
                        else:
                            sketch["stroke"].append(end)
                    self.seam_drawn.emit(np.asarray(sketch["stroke"]).tolist(), False)
                elif "port" in sketch:
                    actors = [
                        a for a in self.plotter.renderer.actors.values() if a.GetAddressAsString("") in self._handles
                    ]
                    actor, _ = self._pick(event.position(), actors)
                    target = self._handles.get(actor.GetAddressAsString("")) if actor else None
                    source = sketch["port"]
                    if (
                        source[1] == "out"
                        and target
                        and target[1] == "in"
                        and (event.position() - sketch["press"]).manhattanLength() >= 3
                    ):
                        self.ports_connected.emit(source[0], target[0])
                    else:
                        self.port_picked.emit(source[0], source[1])
                return True
        if (
            event.type() == QEvent.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.LeftButton
            and not event.modifiers()
        ):
            # A visible edit handle takes precedence over winding segments at
            # the same screen position, especially on doubly curved supports.
            actors = list(self.plotter.renderer.actors.values())
            handles = [a for a in actors if a.GetAddressAsString("") in self._handles]
            actor, point = self._pick(event.position(), handles)
            if actor is None:
                actor, point = self._pick(
                    event.position(), [a for a in actors if a.GetAddressAsString("") in self._actors]
                )
            if actor is not None:
                address = actor.GetAddressAsString("")
                identity = self._handles.get(address)
                key = identity[0] if identity else self._actors[address]
                section = self.editor.section(key)
                self.selected.emit(key)
                if section.locked or section.disconnected:
                    self.message.emit("Unlock and reconnect this section before dragging.")
                elif (not section.bridge or identity and identity[1] == "bridge_height") and (
                    identity or section.editable
                ):
                    kind, index = (identity[1], identity[2]) if identity else ("guide", 0)
                    if kind == "guide" and len(set(section.geometry)) != 1:
                        self.message.emit("Select a connection with one explicit attachment contract.")
                    else:
                        self._drag = (key, kind, index, point.copy(), event.position(), None)
                return True
        if event.type() == QEvent.Type.MouseMove and self._drag is not None:
            point = self._drag_point(event)
            if point is not None:
                self._drag = (*self._drag[:5], point.copy())
                key, kind, index = self._drag[:3]
                section = self.editor.section(key)
                if kind == "bridge_height":
                    preview = np.vstack((self.editor.bridge_handle(key)["base"], point))
                elif kind == "cut":
                    station = self.editor.nearest_cut_position(key, point)
                    point = self.editor.cut_at_position(key, station)
                    preview = np.vstack((section.points[0], point, section.points[-1]))
                else:
                    guides = list(section.guides) or [tuple(point)]
                    guides[index] = tuple(point)
                    preview = np.vstack((section.points[0], guides, section.points[-1]))
                self._line(preview, "drag-preview", "#35cbbb", 5, 0.8)
                self.plotter.render()
                self.message.emit("Preview · release to route and commit one undo step; Esc cancels.")
            return True
        if event.type() == QEvent.Type.MouseButtonRelease and self._drag is not None:
            key, kind, index, _start, press, point = self._drag
            self._drag = None
            self.plotter.remove_actor("drag-preview", render=True)
            if point is not None and (event.position() - press).manhattanLength() >= 3:
                if kind == "bridge_height":
                    self.bridge_height_moved.emit(key, point)
                elif kind == "cut":
                    self.cut_moved.emit(key, point)
                else:
                    self.guide_moved.emit(key, index, point)
            return True
        if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape and self._drag is not None:
            self._drag = None
            self.plotter.remove_actor("drag-preview", render=True)
            self.message.emit("Edit cancelled.")
            return True
        return super().eventFilter(watched, event)

    def closeEvent(self, event):
        self.plotter.close()
        super().closeEvent(event)


class InterconnectionEditorDialog(QDialog):
    def __init__(self, session, parent=None, *, apply_plan=None, viewport_factory=WindingViewport):
        super().__init__(parent)
        self.session, self.apply_plan = session, apply_plan
        self.setWindowTitle("CoilGen · Interconnection editor")
        self.setObjectName("interconnection_editor")
        self.resize(1320, 850)
        self._busy, self._refreshing = False, False
        self._saved_identity = session.identity
        self._key = ""
        layout = QVBoxLayout(self)
        heading = QLabel("<b>Interconnections</b>")
        layout.addWidget(heading)
        self.toolbar = QToolBar()
        self.toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        layout.addWidget(self.toolbar)
        self.undo_action = self._action("Undo", self._undo, QKeySequence.StandardKey.Undo)
        self.redo_action = self._action("Redo", self._redo, QKeySequence.StandardKey.Redo)
        self.toolbar.addSeparator()
        self.lock_action = self._action("Lock", self._lock)
        self.lock_action.setCheckable(True)
        self.cut_action = self._action("Cut connection", self._disconnect)
        self.route_action = self._action("Generate connections", self._generate)
        self.toolbar.widgetForAction(self.route_action).setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toolbar.addSeparator()
        self._action("Save draft…", self._save, QKeySequence.StandardKey.Save)
        self._action("Open draft…", self._load)
        self.mode = QComboBox()
        for title, mode in [
            ("Select / drag", "select"),
            ("Draw seam", "seam"),
            ("Connect endpoints", "connect"),
            ("Over / under", "bridge"),
            ("Draw connection", "draw"),
            ("Cut shapes and planes", "shape"),
        ]:
            self.mode.addItem(title, mode)
        self.mode.setObjectName("winding_tool")
        self.mode.hide()
        self.tool_actions = []
        for index in [0, 4, 5, 1, 2, 3]:
            action = self._action(
                self.mode.itemText(index), lambda checked=False, i=index: self.mode.setCurrentIndex(i)
            )
            action.setIcon(_icon(self.mode.itemData(index)))
            action.setCheckable(True)
            action.setData(index)
            self.tool_actions.append(action)
        self.details_action = self._action("Details", lambda checked: self.inspector.setVisible(checked))
        self.details_action.setCheckable(True)
        self._seam_preview = None
        splitter = QSplitter()
        layout.addWidget(splitter, 1)
        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Circuit"))
        self.circuit = QComboBox()
        for index in range(len(session.editors)):
            self.circuit.addItem(f"Circuit {index + 1}", index)
        controls.addWidget(self.circuit)
        self.ghost = QCheckBox("Automatic route")
        controls.addWidget(self.ghost)
        self.envelope = QCheckBox("Conductor envelope")
        self.envelope.setEnabled(False)
        self.envelope.setToolTip(
            "Show the physical conductor generated by the current validation, including failed geometry."
        )
        controls.addWidget(self.envelope)
        controls.addStretch(1)
        self.view_layout = QComboBox()
        self.view_layout.addItems(["3D + flat", "3D", "Flat"])
        controls.addWidget(self.view_layout)
        fit = QPushButton("Fit views")
        controls.addWidget(fit)
        main_layout.addLayout(controls)
        self.group_legend = QLabel()
        self.group_legend.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse)
        self.group_legend.linkActivated.connect(self._isolate_group)
        main_layout.addWidget(self.group_legend)
        review_controls = QHBoxLayout()
        self.hide_rear = QCheckBox("Hide rear wires")
        self.hide_rear.toggled.connect(self._draw)
        review_controls.addWidget(self.hide_rear)
        for title, callback in [
            ("Face selected area", lambda: self._face_selected(False)),
            ("Local view", lambda: self._face_selected(True)),
        ]:
            button = QPushButton(title)
            button.clicked.connect(callback)
            review_controls.addWidget(button)
        self.play_toggle = QPushButton("Winding preview")
        self.play_toggle.setCheckable(True)
        self.play_toggle.toggled.connect(self._toggle_playback)
        review_controls.addWidget(self.play_toggle)
        review_controls.addStretch()
        main_layout.addLayout(review_controls)
        self._isolation = None
        self.cut_objects = QComboBox()
        self.cut_objects.setAccessibleName("Saved cut objects")
        self.cut_objects.activated.connect(self._select_cut_object)
        main_layout.addWidget(self.cut_objects)
        self.play_bar = QWidget()
        play_layout = QHBoxLayout(self.play_bar)
        self.play_button = QPushButton("Play")
        for title, callback in [
            ("‹", lambda: self._step_playback(-1)),
            ("Play", self._play_pause),
            ("›", lambda: self._step_playback(1)),
        ]:
            button = self.play_button if title == "Play" else QPushButton(title)
            button.clicked.connect(callback)
            play_layout.addWidget(button)
        self.play_label = QLabel()
        play_layout.addWidget(self.play_label, 1)
        main_layout.addWidget(self.play_bar)
        self.play_bar.hide()
        self._play_index, self._play_fraction = 0, 0.0
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(50)
        self.play_timer.timeout.connect(self._play_tick)
        self.shape_bar = QWidget()
        shape_layout = QHBoxLayout(self.shape_bar)
        shape_layout.setContentsMargins(0, 0, 0, 0)
        self.shape_kind, self.shape_group, self.shape_axis = QComboBox(), QComboBox(), QComboBox()
        self.shape_kind.setAccessibleName("Cut shape")
        self.shape_group.setAccessibleName("Cut shape contour group")
        self.shape_axis.setAccessibleName("Cut plane normal")
        for label, value in [("Cut plane", "plane"), ("Sphere", "sphere"), ("Box", "box")]:
            self.shape_kind.addItem(label, value)
        for label, value in [
            ("Tilt handle", None),
            ("Normal X", [1, 0, 0]),
            ("Normal Y", [0, 1, 0]),
            ("Normal Z", [0, 0, 1]),
        ]:
            self.shape_axis.addItem(label, value)
        for control in (self.shape_kind, self.shape_group, self.shape_axis):
            shape_layout.addWidget(control)
        self.shape_dimensions, self.shape_labels = [], []
        for i in range(3):
            label, control = QLabel(), self._number(0.01)
            control.setSuffix(" mm")
            control.setAccessibleName(f"Cut shape dimension {i + 1} in millimetres")
            control.editingFinished.connect(self._shape_dimensions_changed)
            shape_layout.addWidget(label)
            shape_layout.addWidget(control)
            self.shape_labels.append(label)
            self.shape_dimensions.append(control)
        place = QPushButton("Place at selection")
        place.clicked.connect(lambda: self._preview_shape())
        shape_layout.addWidget(place)
        self.place_on_coil = QPushButton("Place on coil")
        self.place_on_coil.setCheckable(True)
        self.place_on_coil.toggled.connect(lambda value: setattr(self.viewport, "place_on_surface", value))
        shape_layout.addWidget(self.place_on_coil)
        discard = QPushButton("×")
        discard.setAccessibleName("Discard cut preview")
        discard.setToolTip("Discard preview")
        discard.clicked.connect(self._clear_cut_preview)
        shape_layout.addWidget(discard)
        self.shape_summary = QLabel()
        shape_layout.addWidget(self.shape_summary)
        self.shape_kind.currentIndexChanged.connect(lambda: self._preview_shape())
        self.shape_group.currentIndexChanged.connect(self._shape_scope_changed)
        self.shape_axis.currentIndexChanged.connect(self._shape_dimensions_changed)
        main_layout.addWidget(self.shape_bar)
        self.viewport = viewport_factory(self)
        from .winding_flat_view import WindingFlatView

        self.flat_view = WindingFlatView(self)
        self.views = QSplitter(Qt.Orientation.Horizontal)
        self.views.addWidget(self.viewport)
        self.views.addWidget(self.flat_view)
        self.views.setSizes([400, 350])
        main_layout.addWidget(self.views, 1)
        self.view_layout.currentIndexChanged.connect(self._layout_changed)
        fit.clicked.connect(self._fit_views)
        self.hint = QLabel("Click a line · drag its handles")
        self.hint.setWordWrap(True)
        main_layout.addWidget(self.hint)
        splitter.addWidget(main)
        self.inspector = QWidget()
        side = QVBoxLayout(self.inspector)
        self.sketch_tools = QStackedWidget()
        side.addWidget(self.sketch_tools)
        self.sketch_tools.addWidget(QLabel(""))
        seam_panel = QWidget()
        seam_form = QFormLayout(seam_panel)
        self.seam_group = QComboBox()
        seam_form.addRow("Contour group", self.seam_group)
        self.seam_summary = QLabel("Draw across turns")
        self.seam_summary.setWordWrap(True)
        seam_form.addRow(self.seam_summary)
        self.seam_apply = QPushButton("Apply seam")
        self.seam_apply.clicked.connect(self._apply_seam)
        seam_form.addRow(self.seam_apply)
        self.seam_apply.hide()  # Generation is the single visible commit action.
        self.sketch_tools.addWidget(seam_panel)
        connection_panel = QWidget()
        connection_form = QFormLayout(connection_panel)
        self.port_source, self.port_target = QComboBox(), QComboBox()
        connection_form.addRow("OUT endpoint", self.port_source)
        connection_form.addRow("IN endpoint", self.port_target)
        self.connect_button = QPushButton("Connect endpoints")
        self.connect_button.clicked.connect(
            lambda: self._connect_ports(self.port_source.currentData(), self.port_target.currentData())
        )
        connection_form.addRow(self.connect_button)
        self.sketch_tools.addWidget(connection_panel)
        bridge_panel = QWidget()
        self.bridge_bar = bridge_panel
        bridge_form = QFormLayout(bridge_panel)
        self.bridge_other = QComboBox()
        self._bridge_point = None
        self.bridge_other.currentIndexChanged.connect(lambda: setattr(self, "_bridge_point", None))
        bridge_form.addRow("Crossing partner", self.bridge_other)
        self.bridge_separation, self.bridge_span = self._number(0.001), self._number(0.001)
        self.bridge_separation.setValue(3)
        self.bridge_span.setValue(20)
        bridge_form.addRow("Centreline separation [mm]", self.bridge_separation)
        bridge_form.addRow("Ramp span [mm]", self.bridge_span)
        bridge_buttons = QHBoxLayout()
        for label, relation in [("Over", "over"), ("Under", "under")]:
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, relation=relation: self._bridge(relation))
            bridge_buttons.addWidget(button)
        bridge_form.addRow(bridge_buttons)
        self.remove_bridge = QPushButton("Remove bridge")
        self.remove_bridge.clicked.connect(
            lambda: self._run(lambda: self.editor.remove_bridge(self._key), "Removing bridge")
        )
        bridge_form.addRow(self.remove_bridge)
        main_layout.insertWidget(main_layout.indexOf(self.views), bridge_panel)
        self.sketch_tools.addWidget(QLabel("Select a crossing in the view"))
        bridge_panel.hide()
        self.sketch_tools.addWidget(QLabel("Draw from → to •"))
        self.sketch_tools.addWidget(QLabel("Drag the cut volume in 3D"))
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.sections = QListWidget()
        self.sections.setObjectName("winding_sections")
        side.addWidget(self.sections, 2)
        self.selection_label = QLabel()
        self.selection_label.setWordWrap(True)
        side.addWidget(self.selection_label)
        self.properties = QStackedWidget()
        side.addWidget(self.properties)
        empty = QLabel("Select a connection or a contour cut.")
        empty.setWordWrap(True)
        self.properties.addWidget(empty)
        cut_panel = QWidget()
        cut_form = QFormLayout(cut_panel)
        self.cut_position = self._number()
        self.cut_width = self._number()
        cut_form.addRow("Position along contour [mm]", self.cut_position)
        cut_form.addRow("Opening length [mm]", self.cut_width)
        move = QPushButton("Move cut")
        move.clicked.connect(self._move_cut_numeric)
        cut_form.addRow(move)
        self.properties.addWidget(cut_panel)
        guide_panel = QWidget()
        guide_form = QFormLayout(guide_panel)
        self.guide_selection = QComboBox()
        guide_form.addRow("Guide point", self.guide_selection)
        self.xyz = [self._number(-1e9) for _ in range(3)]
        for name, control in zip(("X [mm]", "Y [mm]", "Z [mm]"), self.xyz):
            guide_form.addRow(name, control)
        buttons = QHBoxLayout()
        for title, callback in (
            ("Add", self._add_guide),
            ("Remove", self._remove_guide),
            ("Set", self._set_guide_numeric),
        ):
            button = QPushButton(title)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        guide_form.addRow(buttons)
        self.properties.addWidget(guide_panel)
        self.validation_text = QTextBrowser()
        self.validation_text.setObjectName("winding_validation")
        self.validation_text.setMinimumHeight(150)
        side.addWidget(self.validation_text, 1)
        self.issue_list = QListWidget()
        self.issue_list.setAccessibleName("Winding issues")
        self.issue_list.itemClicked.connect(lambda item: self._focus_issue(item.data(Qt.ItemDataRole.UserRole)))
        side.addWidget(self.issue_list, 1)
        splitter.addWidget(self.inspector)
        splitter.setSizes([930, 350])
        splitter.setStretchFactor(0, 1)
        self.inspector.setMinimumWidth(300)
        self.inspector.hide()
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)
        footer = QHBoxLayout()
        self.status = QLabel()
        self.status.setWordWrap(True)
        footer.addWidget(self.status, 1)
        self.check_status = QPushButton("Not checked")
        self.check_status.setFlat(True)
        self.check_status.clicked.connect(lambda: self.details_action.setChecked(True) or self.inspector.show())
        footer.addWidget(self.check_status)
        self.cancel_button = QPushButton("Cancel operation")
        self.cancel_button.hide()
        self.cancel_button.clicked.connect(self._cancel)
        footer.addWidget(self.cancel_button)
        self.validate_button = QPushButton("Check winding")
        self.validate_button.setObjectName("winding_validate")
        self.validate_button.clicked.connect(self._validate)
        footer.addWidget(self.validate_button)
        self.apply_button = QPushButton("Apply to design")
        self.apply_button.setObjectName("winding_apply")
        self.apply_button.clicked.connect(self._apply)
        footer.addWidget(self.apply_button)
        layout.addLayout(footer)
        self.circuit.currentIndexChanged.connect(self._circuit_changed)
        self.sections.currentItemChanged.connect(self._selection_changed)
        self.guide_selection.currentIndexChanged.connect(self._guide_fields)
        self.ghost.toggled.connect(self._draw)
        self.envelope.toggled.connect(self._draw)
        self.viewport.selected.connect(self.select_section)
        self.viewport.cut_moved.connect(self._drag_cut)
        self.viewport.guide_moved.connect(self._drag_guide)
        self.viewport.message.connect(self.hint.setText)
        if hasattr(self.viewport, "shape_moved"):
            self.viewport.shape_moved.connect(self._move_shape_handle)
        if hasattr(self.viewport, "object_selected"):
            self.viewport.object_selected.connect(self._preview_object)
            self.viewport.opening_selected.connect(self._choose_opening)
            self.viewport.shape_placed.connect(self._place_shape)
            self.viewport.issue_selected.connect(self._focus_issue)
            self.viewport.bridge_height_moved.connect(
                lambda key, point: self._run(
                    lambda: self.editor.set_bridge_height(key, point), "Changing bridge height"
                )
            )
        for view in [self.viewport, self.flat_view]:
            if hasattr(view, "seam_drawn"):
                view.seam_drawn.connect(self._preview_seam)
                view.port_picked.connect(self._port_picked)
                view.ports_connected.connect(self._connect_ports)
                view.bridge_target.connect(self._pick_crossing)
            if hasattr(view, "connection_drawn"):
                view.connection_drawn.connect(self._draw_connection)
        self.flat_view.selected.connect(self.select_section)
        self.flat_view.cut_moved.connect(self._drag_cut)
        self.flat_view.guide_moved.connect(lambda key, index, point: self._drag_guide(key, index, point, flat=True))
        self.flat_view.message.connect(self.hint.setText)
        self._refresh()

    def _isolate_group(self, group):
        group = int(group)
        self._isolation = None if self._isolation == group else group
        self.shape_group.blockSignals(True)
        self.shape_group.setCurrentIndex(self.shape_group.findData(group))
        self.shape_group.blockSignals(False)
        self._draw()

    def _face_selected(self, local):
        self.view_layout.setCurrentIndex(1)
        if hasattr(self.viewport, "face_selection"):
            self.viewport.face_selection(local)

    def _select_cut_object(self, index):
        identity = self.cut_objects.itemData(index)
        if identity:
            self._preview_object(identity)

    def _preview_object(self, identity):
        obj = next(o for o in self.editor.cut_objects if o["id"] == identity)
        self._key = obj["keys"][0]
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(self.mode.findData("shape"))
        self.mode.blockSignals(False)
        self.viewport.mode = "shape"
        self.view_layout.setCurrentIndex(1)
        self._preview_shape(obj["shape"], object_id=identity, keys=obj["keys"])

    def _choose_opening(self, key, index):
        p = self._seam_preview
        if p and "shape" in p:
            self._preview_shape(
                p["shape"], object_id=p["object_id"], keys=p["keys"], choices={**p.get("choices", {}), key: index}
            )

    def _place_shape(self, point):
        from ..winding_cut_shapes import CutShape

        self.place_on_coil.setChecked(False)
        shape = CutShape.from_dict(self.editor.default_cut_shape(self.shape_kind.currentData(), self._key))
        self._preview_shape(shape.moved("center", self.editor.project(point)).to_dict(), object_id="new")

    def _focus_issue(self, index):
        issue = self._issues[index]
        if issue["key"]:
            self.select_section(issue["key"])
        self.hint.setText(issue["detail"])
        if hasattr(self.viewport, "focus_issue"):
            self.viewport.focus_issue(issue)

    def _toggle_playback(self, enabled):
        self.play_bar.setVisible(enabled)
        if not enabled:
            self.play_timer.stop()
            self.play_button.setText("Play")
        self._play_display()

    def _play_pause(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_button.setText("Play")
        elif winding_sequence(self.editor):
            self.play_timer.start()
            self.play_button.setText("Pause")

    def _step_playback(self, delta):
        self.play_timer.stop()
        self.play_button.setText("Play")
        sequence = winding_sequence(self.editor)
        self._play_index = min(max(0, self._play_index + delta), max(0, len(sequence) - 1))
        self._play_fraction = 0.0
        self._play_display()

    def _play_tick(self):
        if self._busy:
            return
        self._play_fraction += 0.025
        if self._play_fraction >= 1:
            self._play_fraction = 0.0
            if self._play_index + 1 >= len(winding_sequence(self.editor)):
                self.play_timer.stop()
                self.play_button.setText("Play")
                self._play_fraction = 1.0
            else:
                self._play_index += 1
        self._play_display()

    def _play_display(self):
        sequence = winding_sequence(self.editor)
        step = sequence[min(self._play_index, len(sequence) - 1)] if sequence else None
        label = (
            f"{self._play_index + 1} / {len(sequence)} · {step['label']}"
            if step
            else "Complete the circuit to preview winding"
        )
        if step and step["bridge"]:
            label += f" · Over {step['bridge']['other_label']} · {step['bridge']['order_note']}"
        self.play_label.setText(label)
        if hasattr(self.viewport, "playback") and self.viewport.editor:
            self.viewport.playback(step["key"] if step and self.play_toggle.isChecked() else None, self._play_fraction)

    @staticmethod
    def _number(minimum=0):
        number = QDoubleSpinBox()
        number.setRange(minimum, 1e9)
        number.setDecimals(5)
        number.setSingleStep(0.1)
        number.setKeyboardTracking(False)
        return number

    def _action(self, title, callback, shortcut=None):
        action = QAction(title, self)
        icon = title.split()[0].lower()
        if icon in {"undo", "redo", "lock", "cut", "save", "open", "generate", "details"}:
            action.setIcon(_icon(icon))
        action.setToolTip(title)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
            action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        action.triggered.connect(callback)
        self.toolbar.addAction(action)
        self.addAction(action)
        return action

    @property
    def editor(self):
        return self.session.editors[max(0, self.circuit.currentIndex())]

    def _circuit_changed(self):
        self._key = ""
        self._seam_preview = None
        self.viewport._first = True
        self._refresh()

    def select_section(self, key):
        if self._busy:
            return
        for index in range(self.sections.count()):
            item = self.sections.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == key:
                self.sections.setCurrentItem(item)
                return

    def _selection_changed(self, current, previous=None):
        if self._refreshing or current is None:
            return
        self._key = current.data(Qt.ItemDataRole.UserRole)
        self._refresh()

    def _refresh(self):
        self._refreshing = True
        rows = self.editor.sections
        if not self._key or self._key not in {s.key for s in rows}:
            self._key = next((s.key for s in rows if s.editable), rows[0].key)
        self.sections.clear()
        groups = sorted({s.owner.group_index for s in rows if s.owner.group_index >= 0})
        self.group_legend.setText(
            " &nbsp; ".join(f'<a href="{g}" style="color:{group_color(g)}">● Group {g + 1}</a>' for g in groups)
        )
        self.cut_objects.clear()
        self.cut_objects.addItem("Cut objects", None)
        for obj in self.editor.cut_objects:
            self.cut_objects.addItem(f"◇ {obj['label']} · {len(obj['keys'])} cuts", obj["id"])
        identity = (self._seam_preview or {}).get("object_id")
        self.cut_objects.setCurrentIndex(max(0, self.cut_objects.findData(identity)))
        self.cut_objects.setVisible(bool(self.editor.cut_objects))
        if getattr(self, "_play_identity", None) != self.session.identity:
            self.play_timer.stop()
            self.play_button.setText("Play")
            self._play_index, self._play_fraction = 0, 0.0
            self._play_identity = self.session.identity
        for section in rows:
            suffix = " · locked" if section.locked else " · open" if section.disconnected else ""
            item = QListWidgetItem(section.label + suffix)
            item.setData(Qt.ItemDataRole.UserRole, section.key)
            item.setForeground(QColor(group_color(section.owner.group_index)))
            self.sections.addItem(item)
            if section.key == self._key:
                self.sections.setCurrentItem(item)
        row = self.editor.section(self._key)
        self.selection_label.setText(
            row.label + "\n" + ("Locked" if row.locked else " · ".join(sorted({g.value for g in row.geometry})))
        )
        self.undo_action.setEnabled(bool(self.editor.undo_label))
        self.undo_action.setToolTip("Undo " + self.editor.undo_label if self.editor.undo_label else "Undo")
        self.redo_action.setEnabled(bool(self.editor.redo_label))
        self.lock_action.setChecked(row.locked)
        self.lock_action.setText("Unlock" if row.locked else "Lock")
        self.cut_action.setEnabled(row.editable and not row.locked)
        self.cut_action.setText("Reconnect" if row.disconnected else "Cut connection")
        self.properties.setEnabled(not row.locked and not row.disconnected and not row.bridge)
        self.properties.setCurrentIndex(1 if row.cut is not None else 2 if row.editable else 0)
        self.properties.setFixedHeight(self.properties.currentWidget().sizeHint().height())
        if row.cut is not None:
            self.cut_position.setValue(row.cut[0] * 1e3)
            self.cut_width.setValue(row.cut[1] * 1e3)
        self.guide_selection.clear()
        for index in range(max(1, len(row.guides))):
            self.guide_selection.addItem(f"Guide {index + 1}" if row.guides else "New midpoint guide")
        self._guide_fields()
        for combo, items in [
            (
                self.seam_group,
                [("All movable cuts", None)]
                + [(f"Group {g + 1}", g) for g in sorted({s.owner.group_index for s in rows if s.cut is not None})],
            ),
            (
                self.shape_group,
                [("All groups", None), ("Selected turn", "selected")]
                + [(f"Group {g + 1}", g) for g in sorted({s.owner.group_index for s in rows if s.cut is not None})],
            ),
            (
                self.port_source,
                [("Choose OUT", None)] + [(p["label"], p["key"]) for p in self.editor.ports if p["role"] == "out"],
            ),
            (
                self.port_target,
                [("Choose IN", None)] + [(p["label"], p["key"]) for p in self.editor.ports if p["role"] == "in"],
            ),
            (
                self.bridge_other,
                [("Choose crossing partner", None)]
                + [(s.label, s.key) for s in rows if s.key != self._key and not s.disconnected],
            ),
        ]:
            selected = combo.currentData()
            combo.clear()
            for label, value in items:
                combo.addItem(label, value)
            combo.setCurrentIndex(max(0, combo.findData(selected)))
        self.remove_bridge.setEnabled(row.bridge is not None and not row.locked)
        if row.bridge:
            self.bridge_other.setCurrentIndex(self.bridge_other.findData(row.bridge["other"]))
            self.bridge_separation.setValue(row.bridge["separation_m"] * 1000)
            self.bridge_span.setValue(row.bridge["span_m"] * 1000)
        if self._seam_preview and self._seam_preview["revision"] != self.editor.revision:
            self._seam_preview = None
        self.seam_apply.setEnabled(self._seam_preview is not None)
        self.seam_summary.setText(
            f"{len(self._seam_preview['cuts'])} cuts · Generate to apply" if self._seam_preview else "Draw across turns"
        )
        self.shape_bar.setVisible(self.mode.currentData() == "shape")
        self.bridge_bar.setVisible(self.mode.currentData() == "bridge")
        shape = self._seam_preview.get("shape") if self._seam_preview else None
        if shape:
            self.shape_kind.setCurrentIndex(self.shape_kind.findData(shape["kind"]))
            for i, (label, control) in enumerate(zip(self.shape_labels, self.shape_dimensions)):
                visible = i == 0 or shape["kind"] == "box"
                label.setVisible(visible)
                control.setVisible(visible)
                label.setText(
                    "Diameter"
                    if shape["kind"] == "sphere"
                    else "Gap"
                    if shape["kind"] == "plane"
                    else ["Width", "Height", "Depth"][i]
                )
                control.setValue(shape["size"][i] * 2000)
            self.shape_axis.setVisible(shape["kind"] != "sphere")
        self.shape_summary.setText(
            self.seam_summary.text()
            + (" · nearest existing openings" if self._seam_preview and self._seam_preview.get("multiple") else "")
        )
        self.route_action.setEnabled(
            bool(self._seam_preview and self._seam_preview["cuts"])
            if self.mode.currentData() == "shape"
            else not self._seam_preview or bool(self._seam_preview["cuts"])
        )
        if self._seam_preview and "drawing" in self._seam_preview:
            self.route_action.setEnabled(True)
        if self._seam_preview and self._seam_preview.get("error"):
            self.route_action.setEnabled(False)
            self.hint.setText(self._seam_preview["error"])
        self.sketch_tools.setFixedHeight(self.sketch_tools.currentWidget().sizeHint().height())
        connection = self.editor.connectivity
        topology = "Continuous"
        if not connection["complete"]:
            topology = f"{connection['free_ends']} open ends"
        self.status.setText(topology)
        receipt = self.session.validation if self.session.validated else None
        self._issues = review_issues(
            self.editor, receipt, self.circuit.currentIndex(), self.session.config.sweep_min_bend_radius
        )
        if self._seam_preview and not self._seam_preview.get("error"):
            self._issues = [i for i in self._issues if i["category"] != "advisory"] + self._seam_preview.get(
                "warnings", []
            )
        self.issue_list.clear()
        for index, issue in enumerate(self._issues):
            prefix = {"advisory": "Estimate", "unverified": "Not certified", "failed": "Failed"}.get(
                issue["category"], "Open"
            )
            item = QListWidgetItem(f"{prefix} · {issue['label']}" + (" ↗" if issue["point"] else ""))
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setToolTip(issue["detail"])
            self.issue_list.addItem(item)
        self.check_status.setText(
            "Not checked"
            if receipt is None
            else "Checks passed"
            if receipt.accepted
            else f"{len(receipt.reasons)} checks failed"
        )
        self.check_status.setStyleSheet("color: #a35b2d" if receipt and not receipt.accepted else "")
        for action in self.tool_actions:
            action.setChecked(action.data() == self.mode.currentIndex())
        self.apply_button.setEnabled(self.session.can_apply and self.apply_plan is not None and not self._seam_preview)
        self.envelope.setEnabled(
            self.session.validated
            and len(self.session.validation.parts) > self.circuit.currentIndex()
            and self.session.validation.parts[self.circuit.currentIndex()].swept_body is not None
        )
        self.validate_button.setEnabled(
            all(e.connectivity["complete"] for e in self.session.editors) and not self._seam_preview
        )
        if not self.session.validated:
            self.validation_text.setPlainText(
                "Engineering validation pending\n\nGeometry edits invalidate previous checks. Validate the conductor envelope, bend radius, assembly clearance and fixed-current field change before applying."
            )
        else:
            report = self.session.validation
            lines = ["Engineering checks passed" if report.accepted else "Engineering checks need attention"]
            physical = report.diagnostics.get("physical", {})
            delta = physical.get("longitudinal_change", {})
            if "rms_T" in delta:
                lines += [f"RMS ΔB₀: {delta['rms_T'] * 1e6:.5g} µT", f"Peak ΔB₀: {delta['peak_T'] * 1e6:.5g} µT"]
            lines += ["Current held fixed; no refit.", *report.reasons]
            circuits = report.diagnostics.get("circuits", ())
            if self.circuit.currentIndex() < len(circuits):
                circuit = circuits[self.circuit.currentIndex()]
                metrics = circuit["manufacturing"]
                lines.insert(1, f"Circuit current: {circuit['current_A']:.6g} A")
                radius = metrics.get("sweep_final_min_bend_radius_m")
                if radius is not None:
                    lines.insert(
                        2,
                        f"Minimum bend radius: {radius * 1e3:.5g} mm\nRequired: {self.session.config.sweep_min_bend_radius * 1e3:.5g} mm",
                    )
            self.validation_text.setPlainText("\n\n".join(lines))
        self._refreshing = False
        self._draw()

    def _draw(self):
        if not self._busy:
            physical = None
            if self.envelope.isChecked() and self.envelope.isEnabled():
                physical = self.session.validation.parts[self.circuit.currentIndex()]
            self.viewport.preview = self._seam_preview
            self.viewport.hide_rear = self.hide_rear.isChecked()
            self.viewport.isolate_group = self._isolation
            self.viewport.issues = getattr(self, "_issues", [])
            self.viewport.display(self.editor, self._key, baseline=self.ghost.isChecked(), physical_part=physical)
            self.flat_view.mode = self.mode.currentData()
            self.flat_view.display(self.editor, self._key, preview=self._seam_preview)
            if self.play_toggle.isChecked():
                self._play_display()
            chart_available = self.flat_view.chart is not None
            for index in (0, 2):
                self.view_layout.model().item(index).setEnabled(chart_available)
            self.view_layout.setToolTip(self.flat_view.error)
            self.mode.model().item(1).setEnabled(chart_available)
            for action in self.tool_actions:
                if action.data() == 1:
                    action.setEnabled(chart_available)
                    action.setToolTip("Draw seam" if chart_available else self.flat_view.error)
            if not chart_available:
                self.view_layout.setCurrentIndex(1)
                if self.mode.currentData() == "seam":
                    self.mode.setCurrentIndex(0)

    def _guide_fields(self):
        if not self._key:
            return
        row = self.editor.section(self._key)
        index = max(0, self.guide_selection.currentIndex())
        point = row.guides[min(index, len(row.guides) - 1)] if row.guides else row.points[len(row.points) // 2]
        for control, value in zip(self.xyz, point):
            control.setValue(float(value) * 1e3)

    def _run(self, operation, label, callback=None):
        if self._busy:
            return
        self._busy = self.viewport.busy = self.flat_view.busy = True
        self.toolbar.setEnabled(False)
        self.inspector.setEnabled(False)
        self.circuit.setEnabled(False)
        self.validate_button.setEnabled(False)
        self.apply_button.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.show()
        self.cancel_button.show()
        self.cancel_button.setEnabled(True)
        self.hint.setText(label + "…")
        task = self._task = _Task(operation)
        self._completion, self._task_label = callback, label
        task.signals.finished.connect(self._finished, Qt.ConnectionType.QueuedConnection)
        task.signals.progress.connect(self._progress, Qt.ConnectionType.QueuedConnection)
        QThreadPool.globalInstance().start(task)

    @Slot(object, str)
    def _finished(self, value, error):
        self._busy = self.viewport.busy = self.flat_view.busy = False
        self.toolbar.setEnabled(True)
        self.inspector.setEnabled(True)
        self.circuit.setEnabled(True)
        self.progress.hide()
        self.cancel_button.hide()
        if not error and self._completion:
            self._completion(value)
        self._refresh()
        if error and self._task_label.startswith("Previewing"):
            self._seam_preview = None
            self._refresh()
        self.hint.setText(error or self._task_label + " completed.")
        self._task = None

    @Slot(dict)
    def _progress(self, event):
        if self._busy and not self._task.cancel_requested.is_set():
            self.hint.setText(str(event.get("message", "Computing…")))
            percent = event.get("phase_percent", event.get("percent"))
            if isinstance(percent, (int, float)) and np.isfinite(percent):
                self.progress.setRange(0, 100)
                self.progress.setValue(int(np.clip(percent, 0, 100)))
            else:
                self.progress.setRange(0, 0)

    def _cancel(self):
        if self._busy:
            self._task.cancel_requested.set()
            self.cancel_button.setEnabled(False)
            self.hint.setText("Cancellation requested; waiting for the next numerical checkpoint.")

    def _undo(self):
        self.editor.undo()
        self._refresh()

    def _redo(self):
        self.editor.redo()
        self._refresh()

    def _lock(self):
        obj = next(
            (
                o
                for o in self.editor.cut_objects
                if self.mode.currentData() == "shape"
                and (o["id"] == (self._seam_preview or {}).get("object_id") or self._key in o["keys"])
            ),
            None,
        )
        if obj:
            self.editor.set_cut_object_locked(obj["id"], not obj["locked"])
        else:
            self.editor.set_locked(self._key, not self.editor.section(self._key).locked)
        self._refresh()

    def _disconnect(self):
        row = self.editor.section(self._key)
        try:
            self.editor.disconnect(self._key, not row.disconnected)
            self._refresh()
        except ValueError as error:
            self.hint.setText(str(error))

    def _drag_cut(self, key, point):
        editor = self.editor
        position = editor.nearest_cut_position(key, point)
        self._run(lambda: editor.move_cut(key, position), "Moving contour cut")

    def _move_cut_numeric(self):
        key, position, width = self._key, self.cut_position.value() / 1e3, self.cut_width.value() / 1e3
        editor = self.editor
        original = editor.section(key).cut
        if self.cut_width.value() == round(original[1] * 1e3, self.cut_width.decimals()):
            width = original[1]
        if self.cut_position.value() == round(original[0] * 1e3, self.cut_position.decimals()):
            position = original[0]
        self._run(lambda: editor.move_cut(key, position, width), "Moving contour cut")

    def _mode_changed(self):
        self.viewport.mode = self.mode.currentData()
        self.sketch_tools.setCurrentIndex(self.mode.currentIndex())
        self._refresh()
        if self.mode.currentData() == "shape":
            self.view_layout.setCurrentIndex(1)
            if not self._seam_preview or "shape" not in self._seam_preview:
                self._preview_shape()
        self.hint.setText(
            {
                "select": "Click a line · drag its handles",
                "seam": "Draw across turns · Generate to apply",
                "connect": "Drag → to •",
                "draw": "Select a connection · draw from → to •",
                "bridge": "Click the crossing partner",
                "shape": "Drag center · white handles resize · blue handle tilts",
            }[self.mode.currentData()]
        )

    def _generate(self):
        if self._seam_preview:
            self._apply_seam()
        else:
            self._run(self.editor.route_unlocked, "Generating connections")

    def _draw_connection(self, source, target, stroke, flat):
        self._run(
            lambda: self.editor.preview_drawing(
                source, target, stroke, flat=flat, minimum_radius=self.session.config.sweep_min_bend_radius
            ),
            "Previewing connection",
            lambda p: setattr(self, "_seam_preview", p),
        )

    def _layout_changed(self, index):
        self.viewport.setVisible(index != 2)
        self.flat_view.setVisible(index != 1)

    def _fit_views(self):
        self.flat_view.fit()
        if hasattr(self.viewport, "plotter"):
            self.viewport.plotter.reset_camera()

    def _preview_seam(self, stroke, flat):
        editor, group = self.editor, self.seam_group.currentData()
        keys = [s.key for s in editor.sections if s.cut is not None and (group is None or s.owner.group_index == group)]

        def ready(preview):
            self._seam_preview = preview
            self.viewport.preview = preview

        self._run(
            lambda: editor.complete_preview(
                editor.preview_seam(stroke, keys, flat=flat), self.session.config.sweep_min_bend_radius
            ),
            "Previewing seam",
            ready,
        )

    def _preview_shape(self, value=None, *, object_id=None, keys=None, choices=None):
        if self._refreshing or self._busy or self.mode.currentData() != "shape":
            return
        editor, key, group = self.editor, self._key, self.shape_group.currentData()
        kind = self.shape_kind.currentData()
        if keys is None:
            keys = [
                s.key
                for s in editor.sections
                if s.cut is not None
                and (s.key == key if group == "selected" else group is None or s.owner.group_index == group)
            ]
        if object_id is None and value and self._seam_preview:
            object_id = self._seam_preview.get("object_id")
            keys = self._seam_preview.get("keys", keys)
        if object_id not in {o["id"] for o in editor.cut_objects}:
            object_id = None

        def work():
            preview = editor.preview_cut_shape(
                value or editor.default_cut_shape(kind, key), keys, object_id=object_id, choices=choices
            )
            return editor.complete_preview(preview, self.session.config.sweep_min_bend_radius)

        self._run(work, "Previewing cut shape", lambda p: setattr(self, "_seam_preview", p))

    def _clear_cut_preview(self):
        if not self._busy:
            self._seam_preview = None
            self._refresh()

    def _shape_dimensions_changed(self):
        if self._refreshing or self._busy or not self._seam_preview or "shape" not in self._seam_preview:
            return
        value = dict(self._seam_preview["shape"])
        value["size"] = [c.value() / 2000 for c in self.shape_dimensions]
        if self.shape_axis.currentData() is not None:
            value["normal"] = self.shape_axis.currentData()
            value.pop("tangent", None)
        self._preview_shape(value)

    def _shape_scope_changed(self):
        if self._refreshing or self._busy:
            return
        value = self._seam_preview.get("shape") if self._seam_preview else None
        self._seam_preview = None
        self._preview_shape(value)

    def _move_shape_handle(self, role, point):
        from ..winding_cut_shapes import CutShape

        if self._seam_preview and "shape" in self._seam_preview:
            try:
                shape = CutShape.from_dict(self._seam_preview["shape"]).moved(role, point)
                self.shape_axis.blockSignals(True)
                self.shape_axis.setCurrentIndex(0)
                self.shape_axis.blockSignals(False)
                self._preview_shape(shape.to_dict())
            except ValueError as error:
                self.hint.setText(str(error))

    def _apply_seam(self):
        if self._seam_preview:
            editor, preview = self.editor, self._seam_preview
            self._run(
                lambda: editor.apply_preview(preview),
                "Generating connections",
                (lambda _: self.mode.setCurrentIndex(0)) if "drawing" in preview else None,
            )

    def _port_picked(self, key, role):
        combo = self.port_source if role == "out" else self.port_target
        combo.setCurrentIndex(combo.findData(key))
        if self.port_source.currentData() and self.port_target.currentData():
            self._connect_ports(self.port_source.currentData(), self.port_target.currentData())

    def _connect_ports(self, source, target):
        if source and target:
            editor = self.editor
            self._run(lambda: editor.reconnect_endpoints(source, target), "Reconnecting endpoints")

    def _bridge(self, relation):
        editor, key, other = self.editor, self._key, self.bridge_other.currentData()
        if not other:
            self.hint.setText("Choose the crossing partner first.")
            return
        separation, span = self.bridge_separation.value() / 1000, self.bridge_span.value() / 1000
        point = self._bridge_point
        self._run(
            lambda: editor.bridge_crossing(key, other, relation, separation, span, point),
            "Constructing crossing ramps",
            lambda raised: setattr(self, "_key", raised),
        )

    def _pick_crossing(self, key, point):
        self.bridge_other.setCurrentIndex(self.bridge_other.findData(key))
        self._bridge_point = point

    def _drag_guide(self, key, index, point, *, flat=False):
        editor = self.editor
        self._run(lambda: editor.drag_guide(key, index, point, flat=flat), "Routing guided connection")

    def _set_guide(self, key, index, point):
        editor = self.editor
        row = editor.section(key)
        guides = list(row.guides) or [tuple(point)]
        guides[index] = tuple(point)
        self._run(lambda: editor.set_guides(key, guides), "Routing guided connection")

    def _set_guide_numeric(self):
        self._set_guide(
            self._key,
            max(0, self.guide_selection.currentIndex()),
            np.array([control.value() / 1e3 for control in self.xyz]),
        )

    def _add_guide(self):
        editor = self.editor
        row = editor.section(self._key)
        guides = list(row.guides)
        guides.append(tuple(row.points[max(1, int(len(row.points) * 0.65))]))
        key = self._key
        self._run(lambda: editor.set_guides(key, guides), "Adding guide")

    def _remove_guide(self):
        editor = self.editor
        row = editor.section(self._key)
        guides = list(row.guides)
        if guides:
            guides.pop(max(0, self.guide_selection.currentIndex()))
        key = self._key
        self._run(lambda: editor.set_guides(key, guides), "Removing guide")

    def _validate(self):
        self._run(self.session.validation_job(), "Checking engineering", self.session.accept_validation)

    def _apply(self):
        if self.session.can_apply and self.apply_plan and self.apply_plan(self.session.plan()):
            self._saved_identity = self.session.identity
            self.hint.setText(
                "Applied as a new design revision. Compute the winding to update results and fabrication exports."
            )

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save winding draft",
            str(Path(self.session.source_directory).parent / "winding.cgwinding"),
            "Winding draft (*.cgwinding)",
        )
        if path:
            try:
                self.session.save_draft(path)
                self._saved_identity = self.session.identity
                self.hint.setText("Winding draft saved.")
            except (OSError, ValueError) as error:
                self.hint.setText(str(error))

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open winding draft", self.session.source_directory, "Winding draft (*.cgwinding)"
        )
        if path:
            self._seam_preview = None
            self.viewport.preview = None
            self._run(
                lambda: self.session.load_draft(path),
                "Opening winding draft",
                lambda _: setattr(self, "_saved_identity", self.session.identity),
            )

    def closeEvent(self, event):
        if self._busy:
            self.hint.setText("Wait for the current routing or validation operation to finish.")
            event.ignore()
            return
        if self.session.identity != self._saved_identity:
            choice = QMessageBox.question(
                self,
                "Unsaved winding draft",
                "Discard the unsaved winding edits?",
                QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if choice != QMessageBox.StandardButton.Discard:
                event.ignore()
                return
        self.play_timer.stop()
        self.viewport.close()
        super().closeEvent(event)
