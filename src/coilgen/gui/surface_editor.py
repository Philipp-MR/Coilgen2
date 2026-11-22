"""Native CAD modeller: object bar, direct transforms, face picking and meshing."""

from copy import deepcopy
import json
import math

import numpy as np
from PySide6.QtCore import QEvent, QObject, QRunnable, QThreadPool, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..application.surface_model import PRIMITIVES, SurfaceEditor, SurfaceModel, primitive
from ..geometry import rotation_matrix_by_vector
from ..persistence import atomic_json


COLORS = ("#3d8cc4", "#d98038", "#4fa680", "#a175c4", "#c29640", "#45a3b0")


class _Signals(QObject):
    finished = Signal(object, str)


class _Task(QRunnable):
    def __init__(self, operation):
        super().__init__()
        self.operation, self.signals = operation, _Signals()

    def run(self):
        try:
            self.signals.finished.emit(self.operation(), "")
        except Exception as exc:
            self.signals.finished.emit(None, str(exc))


class SurfaceEditorDialog(QDialog):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        import pyvista as pv
        from pyvistaqt import QtInteractor

        self.pv = pv
        self.setWindowTitle("Geometry studio — CoilGen2")
        self.resize(1180, 850)
        self.editor = SurfaceEditor(SurfaceModel.from_project(project))
        self.doc = self.editor.model.to_dict()
        self.selected = self.doc["nodes"][0]["id"]
        self.face, self.drag, self.task, self.result_value = None, None, None, None
        self.controls, self.fields, self.actors = [], {}, {}
        self._syncing, self._has_camera = False, False
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.kind = QComboBox()
        self.kind.addItems(list(PRIMITIVES))
        row.addWidget(self.kind)
        self._button(row, "Add object", self._add)
        self.mode = QComboBox()
        for text, value in (
            ("Select object", "select"),
            ("Move", "move"),
            ("Rotate", "rotate"),
            ("Orbit", "orbit"),
            ("Select face", "face"),
        ):
            self.mode.addItem(text, value)
        self.mode.currentIndexChanged.connect(self._draw)
        row.addWidget(self.mode)
        self.axis = QComboBox()
        self.axis.addItems(["X axis", "Y axis", "Z axis"])
        row.addWidget(self.axis)
        self.undo_button = self._button(row, "Undo", lambda: self._start(lambda: self.editor.travel(), "Undo"))
        self.redo_button = self._button(row, "Redo", lambda: self._start(lambda: self.editor.travel(True), "Redo"))
        self._button(row, "Fit view", self._fit)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Body A"))
        self.base = QComboBox()
        row.addWidget(self.base)
        row.addWidget(QLabel("Cutter B"))
        self.tool = QComboBox()
        row.addWidget(self.tool)
        self._button(row, "Subtract A − B", self._subtract)
        self.delete_face_button = self._button(row, "Delete face", self._delete_face)
        self.restore_button = self._button(row, "Restore deleted faces", self._restore_faces)
        self.show_deleted = QCheckBox("Show deleted faces")
        self.show_deleted.toggled.connect(self._draw)
        row.addWidget(self.show_deleted)
        layout.addLayout(row)

        self.view = QtInteractor(self, off_screen=QApplication.platformName() == "offscreen")
        self.view.set_background("#f3f6f9")
        self.view.interactor.installEventFilter(self)
        self.view.enable_parallel_projection()
        layout.addWidget(self.view.interactor, 1)

        self.bar_area = QScrollArea()
        self.bar_area.setWidgetResizable(True)
        self.bar_area.setFixedHeight(135)
        self.bar = QWidget()
        self.bar_layout = QHBoxLayout(self.bar)
        self.bar_area.setWidget(self.bar)
        layout.addWidget(self.bar_area)
        self.parameter_area = QWidget()
        self.parameter_layout = QHBoxLayout(self.parameter_area)
        layout.addWidget(self.parameter_area)
        row = QHBoxLayout()
        self._button(row, "Duplicate", self._duplicate)
        self._button(row, "Remove object / operation", self._remove)
        self._button(row, "Save recipe…", self._save)
        self._button(row, "Load recipe…", self._load)
        row.addStretch()
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Target edge length (mm)"))
        self.edge = QDoubleSpinBox()
        self.edge.setRange(0.01, 1000)
        self.edge.setDecimals(4)
        self.edge.setValue(self.doc["edge_length_m"] * 1000)
        self.edge.editingFinished.connect(self._edge_changed)
        row.addWidget(self.edge)
        self.mesh_button = self._button(
            row, "Mesh retained surface", lambda: self._start(self.editor.generate, "Meshing surface")
        )
        self.show_mesh = QCheckBox("Show mesh")
        self.show_mesh.toggled.connect(self._draw)
        row.addWidget(self.show_mesh)
        self.apply_button = self._button(row, "Apply as winding surface", self._apply)
        self._button(row, "Close", self.reject)
        layout.addLayout(row)
        self.status = QLabel("Opening CAD model…")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        note = QLabel(
            "Apply replaces the winding support; project Undo restores it. Target fields and saved runs are retained."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.controls += [
            self.kind,
            self.mode,
            self.axis,
            self.base,
            self.tool,
            self.edge,
            self.bar_area,
            self.parameter_area,
            self.show_deleted,
            self.show_mesh,
        ]
        self._sync()
        QTimer.singleShot(0, lambda: self._start(self.editor.preview, "Opening CAD model"))

    def _button(self, layout, text, action):
        button = QPushButton(text)
        button.setAutoDefault(False)
        button.clicked.connect(action)
        layout.addWidget(button)
        self.controls.append(button)
        return button

    @property
    def node(self):
        return next(n for n in self.doc["nodes"] if n["id"] == self.selected)

    def _start(self, operation, label):
        if self.task:
            return
        self.status.setText(label + "…")
        for control in self.controls:
            control.setEnabled(False)
        self.task = _Task(operation)
        self.task.signals.finished.connect(self._completed)
        QThreadPool.globalInstance().start(self.task)

    def _completed(self, _result, error):
        self.task = None
        self.doc = self.editor.model.to_dict()
        for control in self.controls:
            control.setEnabled(True)
        if self.selected not in {n["id"] for n in self.doc["nodes"]}:
            self.selected = self.doc["nodes"][0]["id"]
        self.face = None
        self._sync()
        self._draw()
        if error:
            self.status.setText(error)

    def _edit(self):
        value = deepcopy(self.doc)
        self._start(lambda: self.editor.edit(value), "Rebuilding CAD surface")

    def _id(self, prefix):
        index = 1
        while any(n["id"] == prefix + str(index) for n in self.doc["nodes"]):
            index += 1
        return prefix + str(index)

    def _add(self):
        if len(self.doc["nodes"]) >= 24:
            self.status.setText("This editor supports up to 24 objects and operations.")
            return
        node = primitive(self.kind.currentText(), self._id("object"), center_m=(0.08, -0.04, 0.04))
        self.doc["nodes"].append(node)
        self.selected = node["id"]
        self._edit()

    def _duplicate(self):
        if self.node["kind"] == "subtract":
            self.status.setText("Duplicate a primitive; subtraction operands remain editable.")
            return
        node = deepcopy(self.node)
        node["id"], node["name"] = self._id("object"), node["name"] + " copy"
        node["center_m"][0] += 0.02
        self.doc["nodes"].append(node)
        self.selected = node["id"]
        self._edit()

    def _remove(self):
        if len(self.doc["nodes"]) == 1:
            self.status.setText("Keep at least one object.")
            return
        if any(self.selected in (n.get("base"), n.get("tool")) for n in self.doc["nodes"]):
            self.status.setText("Remove dependent subtractions first. Their original operands will reappear.")
            return
        self.doc["nodes"] = [n for n in self.doc["nodes"] if n["id"] != self.selected]
        self._edit()

    def _subtract(self):
        base, tool = self.base.currentData(), self.tool.currentData()
        if not base or not tool or base == tool:
            self.status.setText("Choose two distinct result objects for A and B.")
            return
        key = self._id("cut")
        self.doc["nodes"].append(dict(id=key, name="Subtraction " + key, kind="subtract", base=base, tool=tool))
        self.selected = key
        self._edit()

    def _delete_face(self):
        if self.face and self.face not in self.doc["excluded_faces"]:
            self.doc["excluded_faces"].append(self.face)
            self._edit()

    def _restore_faces(self):
        self.doc["excluded_faces"] = []
        self._edit()

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save surface recipe", "surface.cgsurface", "Surface recipe (*.cgsurface)"
        )
        if path:
            atomic_json(path, self.doc)

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load surface recipe", "", "Surface recipe (*.cgsurface *.json)")
        if path:
            try:
                from pathlib import Path

                if Path(path).stat().st_size > 128_000:
                    raise ValueError("Surface recipe exceeds 128 kB.")
                model = SurfaceModel(json.loads(Path(path).read_text(encoding="utf-8")))
                self.doc = model.to_dict()
                self._edit()
            except (ValueError, OSError) as exc:
                self.status.setText(str(exc))

    def _edge_changed(self):
        value = self.edge.value() / 1000
        if not self._syncing and value != self.doc["edge_length_m"]:
            self.doc["edge_length_m"] = value
            self._edit()

    def _select(self, key):
        self.selected, self.face = key, None
        self._sync()
        self._draw()

    @staticmethod
    def _clear(layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _sync(self):
        self._syncing = True
        self._clear(self.bar_layout)
        self.bar_buttons = {}
        roots = self.editor.model.roots
        for i, node in enumerate(self.doc["nodes"]):
            button = QPushButton()
            button.setCheckable(True)
            button.setChecked(node["id"] == self.selected)
            button.setMinimumWidth(210)
            button.setAutoDefault(False)
            button.setStyleSheet(
                f"QPushButton {{border-left:5px solid {COLORS[i % len(COLORS)]};padding:8px;text-align:left;}} QPushButton:checked {{background:#e1effb;}}"
            )
            button.clicked.connect(lambda _checked=False, key=node["id"]: self._select(key))
            self.bar_layout.addWidget(button)
            self.bar_buttons[node["id"]] = button
        self._bar_values()
        old_base, old_tool = self.base.currentData(), self.tool.currentData()
        self.base.clear()
        self.tool.clear()
        for node in self.doc["nodes"]:
            if node["id"] in roots:
                self.base.addItem(node["name"], node["id"])
                if node["kind"] != "plane":
                    self.tool.addItem(node["name"], node["id"])
        self.base.setCurrentIndex(max(0, self.base.findData(old_base)))
        ti = self.tool.findData(old_tool)
        self.tool.setCurrentIndex(ti if ti >= 0 else min(1, self.tool.count() - 1))
        self._clear(self.parameter_layout)
        self.fields = {}
        node = self.node
        if node["kind"] != "subtract":
            definitions = [("parameters", k, k + " mm", 1000, 0.01, 10000) for k in node["parameters"]]
            definitions += [("center_m", i, "XYZ"[i] + " mm", 1000, -10000, 10000) for i in range(3)]
            definitions += [("rotation_rad", i, "R" + "xyz"[i] + " °", 180 / math.pi, -360, 360) for i in range(3)]
            for field, key, text, factor, low, high in definitions:
                container = QWidget()
                box = QVBoxLayout(container)
                box.setContentsMargins(0, 0, 0, 0)
                box.addWidget(QLabel(text))
                spin = QDoubleSpinBox()
                spin.setRange(low, high)
                spin.setDecimals(5)
                spin.setValue(node[field][key] * factor)
                spin.setMinimumWidth(75)
                spin.setKeyboardTracking(False)
                spin.editingFinished.connect(
                    lambda f=field, k=key, s=spin, scale=factor: self._field_changed(f, k, s, scale)
                )
                box.addWidget(spin)
                self.parameter_layout.addWidget(container)
                self.fields[(field, key)] = (spin, factor)
        self.edge.setValue(self.doc["edge_length_m"] * 1000)
        self.undo_button.setEnabled(bool(self.editor.undo))
        self.redo_button.setEnabled(bool(self.editor.redo))
        self.apply_button.setEnabled(self.editor.mesh is not None)
        self.show_mesh.setEnabled(self.editor.mesh is not None)
        if self.editor.mesh is None:
            self.show_mesh.setChecked(False)
        self.delete_face_button.setEnabled(self.face is not None)
        self.restore_button.setEnabled(bool(self.doc["excluded_faces"]))
        self.mesh_button.setEnabled(self.editor.scene is not None and not self.editor.scene["unresolved_faces"])
        self._syncing = False

    def _field_changed(self, field, key, spin, factor):
        if self._syncing or self.task:
            return
        value = spin.value() / factor
        if abs(self.node[field][key] - value) > 1e-12:
            self.node[field][key] = value
            self._edit()

    def _bar_values(self):
        for node in self.doc["nodes"]:
            if node["kind"] == "subtract":
                text = f"{node['name']}\n{node['base']} − {node['tool']}\nResult"
            else:
                dimensions = " · ".join(f"{k} {v * 1000:g}" for k, v in node["parameters"].items())
                centre = " / ".join(f"{v * 1000:.2f}" for v in node["center_m"])
                angles = " / ".join(f"{math.degrees(v):.2f}" for v in node["rotation_rad"])
                text = f"{node['name']}\n{dimensions} mm\nXYZ {centre} mm\nRotation {angles}°"
            self.bar_buttons[node["id"]].setText(text)

    def _ghost(self):
        node = self.node
        if node["kind"] == "subtract":
            return
        p, kind, pv = node["parameters"], node["kind"], self.pv
        if kind == "cylinder":
            poly = pv.Cylinder(
                center=(0, 0, 0), direction=(0, 0, 1), radius=p["radius"], height=p["height"], resolution=64
            )
        elif kind == "sphere":
            poly = pv.Sphere(radius=p["radius"], theta_resolution=48, phi_resolution=32)
        elif kind == "box":
            poly = pv.Box(
                bounds=(
                    -p["width"] / 2,
                    p["width"] / 2,
                    -p["depth"] / 2,
                    p["depth"] / 2,
                    -p["height"] / 2,
                    p["height"] / 2,
                )
            )
        else:
            poly = pv.Plane(i_size=p["width"], j_size=p["height"], i_resolution=1, j_resolution=1)
        matrix = np.eye(3)
        for axis, angle in zip(np.eye(3), node["rotation_rad"]):
            matrix = rotation_matrix_by_vector(axis, angle) @ matrix
        poly.points = poly.points @ matrix.T + node["center_m"]
        i = self.doc["nodes"].index(node)
        self.view.add_mesh(
            poly, color=COLORS[i % len(COLORS)], opacity=0.5, smooth_shading=True, render=False, pickable=False
        )

    def _draw(self, *_args):
        if not hasattr(self, "view") or self.editor.scene is None:
            return
        scene = self.editor.mesh_scene if self.show_mesh.isChecked() and self.editor.mesh_scene else self.editor.scene
        camera = self.view.camera_position if self._has_camera else None
        self.view.clear()
        self.view.enable_lightkit()
        self.actors = {}
        vertices, triangles, indices = (
            np.asarray(scene["vertices"]),
            np.asarray(scene["triangles"]),
            np.asarray(scene["face_indices"]),
        )
        for index, face in enumerate(scene["faces"]):
            if face["excluded"] and not self.show_deleted.isChecked():
                continue
            cells = triangles[indices == index]
            poly = self.pv.PolyData(vertices, np.column_stack((np.full(len(cells), 3), cells)).ravel())
            color_index = next(i for i, n in enumerate(self.doc["nodes"]) if n["id"] == face["body"])
            actor = self.view.add_mesh(
                poly,
                color="#efb332" if face["key"] == self.face else COLORS[color_index % len(COLORS)],
                opacity=0.16 if face["excluded"] else 0.3 if self.drag else 1,
                smooth_shading=True,
                show_edges=self.show_mesh.isChecked(),
                render=False,
            )
            self.actors[actor.GetAddressAsString("")] = face
        if self.drag or self.selected not in self.editor.model.roots:
            self._ghost()
        self.view.add_axes()
        if camera is not None:
            self.view.camera_position = camera
        else:
            self.view.view_isometric()
            self.view.reset_camera()
            self._has_camera = True
        self.view.render()
        unresolved = self.editor.scene["unresolved_faces"]
        if unresolved:
            self.status.setText(
                f"{len(unresolved)} deleted faces changed. Restore deleted faces, then reselect them before meshing."
            )
        elif self.editor.mesh_scene:
            r = self.editor.mesh_scene["mesh_report"]
            self.status.setText(
                f"{r['vertices']} vertices · {r['triangles']} triangles · {r['boundary_loops']} boundary loops · {r['components']} components"
            )
        else:
            self.status.setText(
                "Smooth CAD preview. Drag in Move/Rotate mode; select faces to leave openings. Scroll to zoom."
            )

    def _fit(self):
        self.view.reset_camera()
        self.view.render()

    def eventFilter(self, watched, event):
        if watched is not self.view.interactor or self.editor.scene is None:
            return super().eventFilter(watched, event)
        if self.task:
            return event.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseMove, QEvent.Type.MouseButtonRelease)
        mode = self.mode.currentData()
        if mode == "orbit":
            return super().eventFilter(watched, event)
        if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            if mode in {"select", "face"}:
                from vtkmodules.vtkRenderingCore import vtkCellPicker

                picker = vtkCellPicker()
                picker.SetTolerance(0.005)
                ratio = self.view.interactor.devicePixelRatioF()
                picker.Pick(
                    event.position().x() * ratio,
                    (self.view.interactor.height() - event.position().y()) * ratio,
                    0,
                    self.view.renderer,
                )
                actor = picker.GetActor()
                face = self.actors.get(actor.GetAddressAsString("")) if actor else None
                if face:
                    if mode == "select":
                        self._select(face["body"])
                    else:
                        self.face = face["key"]
                        self.delete_face_button.setEnabled(not face["excluded"])
                        self._draw()
                        self.status.setText(face["label"])
                return True
            if self.node["kind"] == "subtract":
                self.status.setText("Select a construction primitive to move or rotate its Boolean result.")
                return True
            direction = np.asarray(self.view.camera.direction)
            up = np.asarray(self.view.camera.up)
            right = np.cross(direction, up)
            right /= np.linalg.norm(right)
            up = np.cross(right, direction)
            self.drag = dict(
                x=event.position().x(), y=event.position().y(), node=deepcopy(self.node), right=right, up=up
            )
            return True
        if event.type() == QEvent.Type.MouseMove and self.drag:
            dx, dy = event.position().x() - self.drag["x"], event.position().y() - self.drag["y"]
            axis = self.axis.currentIndex()
            if mode == "rotate":
                self.node["rotation_rad"][axis] = float(
                    np.clip(self.drag["node"]["rotation_rad"][axis] + dx * 0.009, -2 * math.pi, 2 * math.pi)
                )
            else:
                a, b = self.drag["right"][axis], -self.drag["up"][axis]
                delta = (
                    ((dx * a + dy * b) / (a * a + b * b) if a * a + b * b > 0.04 else dx)
                    * 2
                    * self.view.camera.parallel_scale
                    / max(self.view.interactor.height(), 1)
                )
                self.node["center_m"][axis] = float(np.clip(self.drag["node"]["center_m"][axis] + delta, -10, 10))
            self._syncing = True
            for (field, key), (spin, factor) in self.fields.items():
                spin.setValue(self.node[field][key] * factor)
            self._syncing = False
            self._bar_values()
            self._draw()
            self.status.setText("Release to rebuild the CAD result.")
            return True
        if event.type() == QEvent.Type.MouseButtonRelease and self.drag:
            self.drag = None
            self._edit()
            return True
        return super().eventFilter(watched, event)

    def _apply(self):
        if self.editor.mesh is not None:
            self.result_value = self.editor.model, self.editor.mesh
            self.accept()

    def done(self, result):
        if self.task:
            return
        self.view.close()
        super().done(result)
