"""Scene authoring with owned SI geometry and deliberate viewport picking."""

from dataclasses import replace

import numpy as np
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..application.spatial_scene import ScenePart, RouteAnchor, SpatialScene, ROLES
from ..application.gradient_studies import RigidTransform
from .gradient_tools import EngineeringDialog, _matrix, _number, _values


class ScenePartDialog(EngineeringDialog):
    def __init__(self, parent=None, *, import_mesh=False):
        super().__init__("Import scene mesh" if import_mesh else "Add exclusion / corridor box", parent)
        self.import_mesh = import_mesh
        self.path = QLineEdit()
        if import_mesh:
            row = QHBoxLayout()
            row.addWidget(self.path)
            button = QPushButton("Browse…")
            button.clicked.connect(self._browse)
            row.addWidget(button)
            self.form.addRow("Closed scene mesh", row)
        self.name = QLineEdit("Exclusion")
        self.role = QComboBox()
        for role in sorted(ROLES):
            self.role.addItem(role.replace("_", " ").capitalize(), role)
        self.role.setCurrentIndex(self.role.findData("keepout"))
        self.center = QLineEdit("0 0 0")
        self.size = QLineEdit("0.01 0.01 0.01")
        self.rotation = QLineEdit("1 0 0; 0 1 0; 0 0 1")
        self.translation = QLineEdit("0 0 0")
        self.unit = QComboBox()
        self.unit.addItems(["m", "mm", "cm", "um"])
        self.clearance = _number(0.001)
        self.material = QLineEdit("unspecified")
        self.form.addRow("Name", self.name)
        self.form.addRow("Physical role", self.role)
        if import_mesh:
            self.form.addRow("Source mesh length unit", self.unit)
        else:
            self.form.addRow("Center xyz [m]", self.center)
            self.form.addRow("Full dimensions xyz [m]", self.size)
        self.form.addRow("Local-to-world rotation · 3 rows", self.rotation)
        self.form.addRow("World translation xyz [m]", self.translation)
        self.form.addRow("Required envelope gap [m]", self.clearance)
        self.form.addRow("Material / evidence label", self.material)
        self.note(
            "Obstacle, keepout and metal exclude the full conductor envelope. Feed corridor confines explicitly modeled feeds. A metal role supplies scene provenance; dynamic metal response requires separate physics evidence."
        )

    def _browse(self):
        name, _ = QFileDialog.getOpenFileName(self, "Import scene mesh", "", "Meshes (*.stl *.npz)")
        if name:
            self.path.setText(name)

    def value(self):
        rotation = _matrix(self.rotation.text().replace(";", "\n"), 3, "Rotation")
        translation = _matrix(self.translation.text(), 3, "Translation")
        if len(rotation) != 3 or len(translation) != 1:
            raise ValueError("Use three rotation rows and one xyz translation row.")
        transform = RigidTransform(rotation, translation[0])
        common = {
            "name": self.name.text(),
            "role": self.role.currentData(),
            "clearance_m": _values(self.clearance),
            "material": self.material.text(),
        }
        if self.import_mesh:
            return ScenePart.import_mesh(
                self.path.text(), length_unit=self.unit.currentText(), transform=transform, **common
            )
        center = _matrix(self.center.text(), 3, "Center")
        size = _matrix(self.size.text(), 3, "Dimensions")
        if len(center) != 1 or len(size) != 1:
            raise ValueError("Use one xyz row for box center and dimensions.")
        return ScenePart.box(center_m=center[0], size_m=size[0], transform=transform, **common)


class SpatialSceneDialog(QDialog):
    pickRequested = Signal()
    sceneChanged = Signal(object)

    def __init__(self, project, record=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(
            f"Spatial scene, exclusions and route anchors — {project.name} · revision {project.revision}"
        )
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.scene = SpatialScene.from_dict(project.study["scene"]) if project.study.get("scene") else SpatialScene()
        self.result_value = None
        self.bindings = {}
        self.winding_points = {}
        if record:
            from ..gradient_engineering import winding_path_sha256

            arrays = record.select_arrays(copy=False)
            for index, plan in enumerate(record.metadata.get("winding_steps", ())):
                points = arrays.get(f"review_conductor_{index}_points_m")
                if points is not None:
                    self.bindings[plan["body_id"]] = winding_path_sha256(points)
                    self.winding_points[plan["body_id"]] = points
            if not self.bindings:
                for key, points in arrays.items():
                    if key.startswith("evaluated_winding_") and key.endswith("_points_m"):
                        id_ = key.removesuffix("_points_m")
                        self.bindings[id_] = winding_path_sha256(points)
                        self.winding_points[id_] = points
        layout = QVBoxLayout(self)
        note = QLabel(
            "Scene inputs are owned by this revision. Checks use continuous electrical segments and conservative conductor envelopes; explicit feeds participate when available. Rotating the view keeps navigation active. Use Pick point only while defining an anchor."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Name", "Role", "Gap [m]", "Geometry / range [m]", "Material"])
        layout.addWidget(self.table)
        row = QHBoxLayout()
        for title, callback in [
            ("Add box…", lambda: self._add(False)),
            ("Import mesh…", lambda: self._add(True)),
            ("Remove selected", self._remove),
        ]:
            button = QPushButton(title)
            button.clicked.connect(callback)
            row.addWidget(button)
        layout.addLayout(row)
        self.anchors = QTableWidget(0, 6)
        self.anchors.setHorizontalHeaderLabels(
            ["Name", "Kind", "Body", "Point xyz [m]", "Tolerance [m]", "Winding SHA-256"]
        )
        layout.addWidget(self.anchors)
        form = QFormLayout()
        self.anchor_name = QLineEdit("Terminal")
        self.anchor_kind = QComboBox()
        for kind in ["positive_terminal", "negative_terminal", "cut", "waypoint"]:
            self.anchor_kind.addItem(kind.replace("_", " "), kind)
        self.body = QComboBox()
        self.body.addItems(list(self.bindings))
        self.point = QLineEdit("0 0 0")
        self.tolerance = _number(0.001)
        form.addRow("Anchor name", self.anchor_name)
        form.addRow("Purpose", self.anchor_kind)
        form.addRow("Verified winding body", self.body)
        form.addRow("World xyz [m]", self.point)
        form.addRow("Position tolerance [m]", self.tolerance)
        layout.addLayout(form)
        row = QHBoxLayout()
        self.pick = QPushButton("Pick point in 3D view")
        self.pick.clicked.connect(self._pick)
        self.pick.setEnabled(bool(self.bindings))
        add = QPushButton("Add bound anchor")
        add.clicked.connect(self._anchor)
        add.setEnabled(bool(self.bindings))
        remove = QPushButton("Remove selected anchor")
        remove.clicked.connect(self._remove_anchor)
        for button in [self.pick, add, remove]:
            row.addWidget(button)
        layout.addLayout(row)
        self.feedback = QLabel(
            "Select verified winding evidence to bind terminal / cut positions."
            if not self.bindings
            else "Anchors check the realized circuit. Apply scene, then use Terminals, feeds and drive waveform to construct explicit supply/return geometry."
        )
        self.feedback.setWordWrap(True)
        layout.addWidget(self.feedback)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self._apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._refresh()
        self.resize(1050, 720)

    def _pick(self):
        self.feedback.setText(
            "Pick mode active: click a displayed surface or winding. Drag to rotate. The picked xyz is copied here; add the anchor after reviewing it."
        )
        self.pickRequested.emit()

    def picked_point(self, point):
        self.point.setText(" ".join(f"{float(v):.17g}" for v in point))
        self.feedback.setText("Picked world coordinates. Review body and purpose, then Add bound anchor.")

    def _add(self, import_mesh):
        if not self._capture_edits():
            return
        dialog = ScenePartDialog(self, import_mesh=import_mesh)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                self.scene = SpatialScene((*self.scene.parts, dialog.result_value), self.scene.anchors)
                self._refresh()
            except ValueError as exc:
                QMessageBox.warning(self, "Scene inputs", str(exc))

    def _remove(self):
        if not self._capture_edits():
            return
        index = self.table.currentRow()
        if index >= 0:
            self.scene = SpatialScene(
                tuple(p for i, p in enumerate(self.scene.parts) if i != index), self.scene.anchors
            )
            self._refresh()

    def _remove_anchor(self):
        if not self._capture_edits():
            return
        index = self.anchors.currentRow()
        if index >= 0:
            self.scene = SpatialScene(
                self.scene.parts, tuple(a for i, a in enumerate(self.scene.anchors) if i != index)
            )
            self._refresh()

    def _anchor(self):
        if not self._capture_edits():
            return
        try:
            points = _matrix(self.point.text(), 3, "Anchor xyz")
            if len(points) != 1:
                raise ValueError("Use one xyz point.")
            id_ = self.body.currentText()
            anchor = RouteAnchor(
                self.anchor_name.text(),
                self.anchor_kind.currentData(),
                points[0],
                id_,
                self.bindings[id_],
                _values(self.tolerance),
            )
            self.scene = SpatialScene(self.scene.parts, (*self.scene.anchors, anchor))
            self._refresh()
        except (ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Anchor inputs", str(exc))

    def _refresh(self):
        self.table.setRowCount(len(self.scene.parts))
        for index, part in enumerate(self.scene.parts):
            bounds = np.stack([part.vertices_m.min(axis=0), part.vertices_m.max(axis=0)])
            for column, value in enumerate(
                [
                    part.name,
                    part.role,
                    f"{part.clearance_m:.17g}",
                    f"{len(part.faces)} triangles; {bounds.tolist()}",
                    part.material,
                ]
            ):
                item = QTableWidgetItem(value)
                if column == 3:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(index, column, item)
        self.anchors.setRowCount(len(self.scene.anchors))
        for index, anchor in enumerate(self.scene.anchors):
            for column, value in enumerate(
                [
                    anchor.name,
                    anchor.kind,
                    anchor.body_id,
                    " ".join(f"{v:.17g}" for v in anchor.point_m),
                    f"{anchor.tolerance_m:.17g}",
                    anchor.expected_path_sha256,
                ]
            ):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.anchors.setItem(index, column, item)
        self.table.resizeColumnsToContents()
        self.anchors.resizeColumnsToContents()
        self.sceneChanged.emit(self.scene)

    def value(self):
        parts = []
        for index, part in enumerate(self.scene.parts):
            parts.append(
                replace(
                    part,
                    name=self.table.item(index, 0).text(),
                    role=self.table.item(index, 1).text(),
                    clearance_m=float(self.table.item(index, 2).text()),
                    material=self.table.item(index, 4).text(),
                )
            )
        return SpatialScene(tuple(parts), self.scene.anchors).to_dict()

    def _capture_edits(self):
        try:
            self.scene = SpatialScene.from_dict(self.value())
            return True
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Correct scene inputs", str(exc))
            return False

    def _apply(self):
        try:
            self.result_value = self.value()
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Correct scene inputs", str(exc))
            return
        self.accept()
