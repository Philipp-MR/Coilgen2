"""Native PCB workspace with linked, selectable flat and installed views."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal, QThreadPool, QSignalBlocker
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .interconnection_editor import _Task


class PcbViewport(QWidget):
    selected = Signal(str)

    def __init__(self, flat=False, parent=None):
        super().__init__(parent)
        self.flat, self.scene, self.selection, self.layer = flat, None, "", "both"
        self.yaw, self.pitch, self.zoom = 0.6, 0.45, 1.0
        self.pan = np.zeros(2)
        self.setMinimumSize(200, 220)
        self.setAccessibleName("Flat PCB" if flat else "Wrapped PCB; drag to rotate, wheel to zoom")
        self.hits, self._drag = [], None
        self.focus_point = None
        self.options = {}
        self.setMouseTracking(True)

    def display(self, scene, selection="", layer="both", **options):
        self.scene, self.selection, self.layer = scene, selection, layer
        self.options = options
        self.update()

    def fit(self):
        self.zoom, self.pan = 1.0, np.zeros(2)
        self.focus_point = None
        self.update()

    def focus(self, point):
        self.focus_point = point
        self.zoom = 2.5
        self.pan = np.zeros(2)
        self.update()

    def _project(self, points):
        points = np.asarray(points, float)
        if self.flat:
            return points[:, :2] * [1, -1]
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        right = np.array([c, -s, 0])
        up = np.array([s * np.sin(self.pitch), c * np.sin(self.pitch), np.cos(self.pitch)])
        return np.column_stack((points @ right, -points @ up))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.palette().base())
        self.hits = []
        scene = self.scene
        if not scene or "board" not in scene:
            painter.setPen(self.palette().text().color())
            painter.drawText(
                self.rect().adjusted(20, 20, -20, -20),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                "Select a supported winding part to preview its PCB layout.",
            )
            return
        reference = (
            np.array([[0, 0], scene["size_m"]]) * [1, -1]
            if self.flat
            else np.vstack([self._project(p) for p in scene["support"]])
        )
        low, high = reference.min(axis=0), reference.max(axis=0)
        center = (low + high) / 2
        if self.focus_point is not None:
            center = self._project([self.focus_point])[0]
        scale = (
            min((self.width() - 45) / max(high[0] - low[0], 1e-9), (self.height() - 45) / max(high[1] - low[1], 1e-9))
            * self.zoom
        )

        def mapped(points):
            return (self._project(points) - center) * scale + np.array([self.width() / 2, self.height() / 2]) + self.pan

        def line(points, color, width=1, dashed=False, key=""):
            coords = mapped(points)
            pen = QPen(color, width)
            if dashed:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolyline(QPolygonF([QPointF(*p) for p in coords]))
            if key:
                self.hits.append((key, coords))

        neutral = self.palette().mid().color()
        if self.flat:
            for group in scene["board"]:
                path = QPainterPath()
                path.setFillRule(Qt.FillRule.OddEvenFill)
                for ring in group:
                    pts = mapped(ring)
                    path.addPolygon(QPolygonF([QPointF(*p) for p in pts]))
                painter.setPen(QPen(neutral, 1))
                painter.setBrush(self.palette().alternateBase())
                painter.drawPath(path)
        else:
            for points in scene["support"]:
                line(points, neutral)
        colors = {"top": QColor("#1a9b91"), "bottom": QColor("#da8844")}
        chosen = self.selection.split(":")[0]
        for trace in scene["traces"]:
            if self.layer not in {"both", trace["layer"]}:
                continue
            color = QColor(colors[trace["layer"]])
            if self.selection and trace["key"] != chosen:
                color.setAlpha(100)
            selected = trace["key"] == chosen
            if self.flat:
                for group in trace["polygons"]:
                    shape = QPainterPath()
                    shape.setFillRule(Qt.FillRule.OddEvenFill)
                    for ring in group:
                        shape.addPolygon(QPolygonF([QPointF(*p) for p in mapped(ring)]))
                    painter.setPen(QPen(self.palette().text().color(), 1) if selected else QPen(Qt.PenStyle.NoPen))
                    painter.setBrush(color)
                    painter.drawPath(shape)
            for points in trace["flat" if self.flat else "wrapped"]:
                line(points, color, 3 if selected else 1.5, trace["layer"] == "bottom", trace["key"])
            if not self.flat and self.options.get("compare") and trace.get("original") and (not chosen or selected):
                purple = QColor("#8055b4")
                line(trace["original"], purple, 2, True)
                if selected and trace.get("comparison"):
                    c = trace["comparison"]
                    line([c["saved"], c["proposed"]], purple, 3)
                    p = mapped([c["saved"]])[0]
                    painter.setPen(self.palette().text().color())
                    painter.drawText(QPointF(p[0] + 8, p[1] - 8), f"{c['distance_mm']:.3f} mm")
        for end in scene["endpoints"]:
            if self.layer not in {"both", end["layer"]}:
                continue
            p = mapped([end["uv"] if self.flat else end["xyz"]])[0]
            painter.setPen(QPen(colors[end["layer"]], 2))
            painter.setBrush(self.palette().base())
            r = 6 if self.selection == end["key"] else 4
            painter.drawEllipse(QPointF(*p), r, r)
            self.hits.append((end["key"], np.array([p])))
            if end["key"] in scene["terminals"]:
                painter.setPen(self.palette().text().color())
                painter.drawText(QPointF(p[0] + 8, p[1] - 7), "AB"[scene["terminals"].index(end["key"])])
        for issue in scene["issues"]:
            points = issue["uv" if self.flat else "xyz"]
            if not points:
                continue
            p = mapped([points])[0]
            painter.setPen(QPen(QColor("#d34d3a"), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(*p), 6, 6)

        if scene.get("seam_line"):
            if self.flat:
                w, h = scene["size_m"]
                for x in (0, w):
                    line([[x, 0], [x, h]], QColor("#c44e38"), 1.5, True)
            else:
                line(scene["seam_line"], QColor("#c44e38"), 1.5, True)
        for via in scene["vias"]:
            p = mapped([via["uv"] if self.flat else via["xyz"]])[0]
            painter.setPen(QPen(self.palette().text().color(), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(*p), 5, 5)

        proposal = self.options.get("proposal")
        if proposal and not proposal.get("error"):
            line(proposal["flat" if self.flat else "wrapped"], QColor("#8055b4"), 3, True)
            if not self.flat:
                for points in proposal["via_lines"]:
                    line(points, QColor("#8055b4"), 3)
        for i, end in enumerate(self.options.get("pins", [])):
            if not end:
                continue
            p = mapped([end["uv" if self.flat else "xyz"]])[0]
            color = QColor("#2467ae" if i == 0 else "#8055b4")
            painter.setPen(QPen(color, 3))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(*p), 8, 8)
            dx = -25 if i == 0 else 12
            painter.fillRect(int(p[0] + dx - 2), int(p[1] - 25), 20, 21, self.palette().base())
            painter.drawText(QPointF(p[0] + dx, p[1] - 9), "AB"[i])

    def nearest(self, point):
        candidates, endpoints = [], []
        for key, points in self.hits:
            if len(points) == 1:
                distance = np.linalg.norm(points[0] - point)
                endpoints.append((distance, key))
            else:
                a, b = points[:-1], points[1:]
                d = b - a
                t = np.clip(np.sum((point - a) * d, axis=1) / np.maximum(np.sum(d * d, axis=1), 1e-12), 0, 1)
                distance = np.linalg.norm(point - (a + t[:, None] * d), axis=1).min()
            candidates.append((distance, key))
        if endpoints and min(endpoints)[0] < 11:
            return min(endpoints)[1]
        return min(candidates)[1] if candidates and min(candidates)[0] < 10 else ""

    def mousePressEvent(self, event):
        self._drag = (
            np.array([event.position().x(), event.position().y()]),
            self.pan.copy(),
            self.yaw,
            self.pitch,
            event.button(),
        )

    def mouseMoveEvent(self, event):
        if not self._drag:
            key = self.nearest(np.array([event.position().x(), event.position().y()]))
            row = next((e for e in (self.scene or {}).get("endpoints", []) if e["key"] == key), None)
            if row:
                self.setToolTip(row["label"] + " · " + row["layer"])
            else:
                trace = next((t for t in (self.scene or {}).get("traces", []) if t["key"] == key), None)
                self.setToolTip(trace["label"] if trace else "")
            return
        pos, pan, yaw, pitch, button = self._drag
        delta = np.array([event.position().x(), event.position().y()]) - pos
        if self.flat or button == Qt.MouseButton.RightButton:
            self.pan = pan + delta
        else:
            self.yaw = yaw + delta[0] * 0.01
            self.pitch = np.clip(pitch + delta[1] * 0.01, -1.5, 1.5)
        self.update()

    def mouseReleaseEvent(self, event):
        if self._drag is None:
            return
        point = np.array([event.position().x(), event.position().y()])
        if np.linalg.norm(point - self._drag[0]) < 4:
            key = self.nearest(point)
            if key:
                self.selected.emit(key)
        self._drag = None

    def wheelEvent(self, event):
        self.zoom = np.clip(self.zoom * np.exp(event.angleDelta().y() / 1200), 0.2, 50)
        self.update()
        event.accept()


class PcbLayoutDialog(QDialog):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session, self.selection, self.layer = session, "", "both"
        self._busy = False
        self._task = None
        self._refreshing = True
        self.preview = None
        self.location_index = 0
        self._compact = False
        self.setWindowTitle("PCB layout · " + session.source_label)
        self.resize(1250, 820)
        outer = QVBoxLayout(self)
        header = QHBoxLayout()
        outer.addLayout(header)
        self.part = QComboBox()
        self.part.addItems([f"Part {i + 1}" for i in range(len(session.parts))])
        header.addWidget(self.part)
        self.part.currentIndexChanged.connect(lambda i: self.edit("part", index=i))
        self.undo_button = self.button("Undo", lambda: self.run(lambda: self.session.undo()), header)
        self.redo_button = self.button("Redo", lambda: self.run(lambda: self.session.undo(True)), header)
        self.button("Save draft copy…", self.save_draft, header)
        self.button("Open PCB draft…", self.load_draft, header)
        header.addStretch()
        self.controls = QCheckBox("Show controls")
        self.controls.setChecked(True)
        header.addWidget(self.controls)
        self.button("Close", self.close, header)
        summary_row = QHBoxLayout()
        outer.addLayout(summary_row)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        summary_row.addWidget(self.summary, 1)
        self.button("Review issues", lambda: self.open_tab(3), summary_row)
        split = QSplitter()
        outer.addWidget(split, 1)
        self.tools = QTabWidget()
        self.tools.setMinimumWidth(265)
        self.tools.setMaximumWidth(370)
        split.addWidget(self.tools)
        self.controls.toggled.connect(self.tools.setVisible)
        self.tools.currentChanged.connect(lambda _: self.update_preview())
        board = self.page("Board")
        form = QFormLayout()
        board.addLayout(form)
        self.mode = QComboBox()
        self.mode.addItem("Saved winding", "wire")
        self.mode.addItem("Two-layer group spirals", "spirals")
        self.mode.currentIndexChanged.connect(lambda: self.edit("settings", mode=self.mode.currentData()))
        form.addRow("Layout source", self.mode)
        self.seam = QSlider(Qt.Orientation.Horizontal)
        self.seam.setRange(0, 359)
        self.seam_label = QLabel()
        self.seam.valueChanged.connect(lambda v: self.seam_label.setText(f"Opening seam: {v}°"))
        board.addWidget(self.seam_label)
        board.addWidget(self.seam)
        self.seam.sliderReleased.connect(lambda: self.edit("settings", seam_deg=float(self.seam.value())))
        self.auto_seam = self.button("Find seam between windings", lambda: self.edit("seam"), board)
        self.dimensions = QLabel()
        self.dimensions.setWordWrap(True)
        board.addWidget(self.dimensions)
        self.inputs = {}
        self.add_number(form, "edge_m", "Edge clearance [mm]", 1e3, 0, 100, 3)
        traces = self.page("Copper")
        copper = QFormLayout()
        traces.addLayout(copper)
        self.add_number(copper, "width_m", "Trace width [mm]", 1e3, 0.001, 100, 3)
        self.add_number(copper, "clearance_m", "Copper gap [mm]", 1e3, 0, 100, 3)
        self.add_number(copper, "copper_m", "Copper thickness [µm]", 1e6, 1, 10000, 1)
        self.add_number(copper, "layer_gap_m", "Layer separation [mm]", 1e3, 0, 100, 3)
        self.trace_list = QComboBox()
        copper.addRow("Selected trace", self.trace_list)
        self.trace_list.currentIndexChanged.connect(lambda: self.select(self.trace_list.currentData() or ""))
        self.assign = QComboBox()
        self.assign.addItems(["top", "bottom"])
        copper.addRow("Move trace to", self.assign)
        self.button(
            "Apply layer",
            lambda: self.edit("layer", key=self.trace_list.currentData(), layer=self.assign.currentText()),
            traces,
        )
        connections = self.page("Connect")
        cf = QFormLayout()
        connections.addLayout(cf)
        self.end_a, self.end_b = QComboBox(), QComboBox()
        cf.addRow("Endpoint A", self.end_a)
        cf.addRow("Endpoint B", self.end_b)
        self.pick_endpoint = QComboBox()
        self.pick_endpoint.addItems(["Endpoint A", "Endpoint B"])
        cf.addRow("Next click picks", self.pick_endpoint)
        self.end_a.currentIndexChanged.connect(lambda: self.endpoint_changed(self.end_a))
        self.end_b.currentIndexChanged.connect(lambda: self.endpoint_changed(self.end_b))
        self.connection_layer = QComboBox()
        self.connection_layer.addItems(["top", "bottom"])
        cf.addRow("Connection layer", self.connection_layer)
        self.connection_layer.currentIndexChanged.connect(lambda _: self.update_preview())
        self.proposal_label = QLabel()
        self.proposal_label.setWordWrap(True)
        connections.addWidget(self.proposal_label)
        self.connect_button = self.button(
            "Connect A → B",
            lambda: self.edit(
                "connect",
                a=self.end_a.currentData(),
                b=self.end_b.currentData(),
                layer=self.connection_layer.currentText(),
            ),
            connections,
        )
        self.button(
            "Use A / B as terminals",
            lambda: self.edit("terminals", a=self.end_a.currentData(), b=self.end_b.currentData()),
            connections,
        )
        note = QLabel(
            "A / B mark the selected endpoints. The purple line previews the straight link and layer transitions. Hover endpoints for their names."
        )
        note.setWordWrap(True)
        connections.addWidget(note)
        self.connection_list = QComboBox()
        connections.addWidget(self.connection_list)
        self.button(
            "Remove connection",
            lambda: self.edit("remove_connection", index=self.connection_list.currentIndex()),
            connections,
        )
        board.addStretch()
        traces.addStretch()
        connections.addStretch()
        for form_layout in (form, copper, cf):
            form_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
            form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        for combo in (self.mode, self.trace_list, self.end_a, self.end_b, self.connection_list):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
        self.check_page = self.page("Review")
        self.checks = QListWidget()
        self.checks.setWordWrap(True)
        self.checks.setMaximumHeight(190)
        self.check_page.addWidget(self.checks, 1)
        self.checks.currentRowChanged.connect(self.select_issue)
        self.issue_detail = QLabel()
        self.issue_detail.setWordWrap(True)
        self.check_page.addWidget(self.issue_detail)
        self.next_location = self.button("Next finding", self.next_issue_location, self.check_page)
        self.repair_button = self.button("Edit connections", self.repair, self.check_page)
        self.evaluate_button = self.button(
            "Evaluate PCB coil", lambda: self.run(self.session.evaluate), self.check_page
        )
        self.performance = QLabel()
        self.performance.setWordWrap(True)
        self.check_page.addWidget(self.performance)
        view = QWidget()
        vl = QVBoxLayout(view)
        split.addWidget(view)
        split.setStretchFactor(1, 1)
        toolbar = QHBoxLayout()
        vl.addLayout(toolbar)
        for label, key in (("Both layers", "both"), ("Top", "top"), ("Bottom", "bottom")):
            self.button(label, lambda _, k=key: self.set_layer(k), toolbar)
        self.compare = QCheckBox("Compare saved path")
        toolbar.addWidget(self.compare)
        self.compare.toggled.connect(lambda _: self.select(self.selection))
        toolbar.addStretch()
        self.button("Fit views", lambda: (self.flat.fit(), self.wrapped.fit()), toolbar)
        views = QSplitter()
        vl.addWidget(views, 1)
        self.flat, self.wrapped = PcbViewport(True), PcbViewport(False)
        for title, widget in (("Flat PCB · mm", self.flat), ("Installed coil · drag to rotate", self.wrapped)):
            pane = QWidget()
            layout = QVBoxLayout(pane)
            layout.addWidget(QLabel(title))
            layout.addWidget(widget, 1)
            views.addWidget(pane)
            widget.selected.connect(self.pick)
        views.setSizes([650, 480])
        split.setSizes([300, 1100])
        self.selection_label = QLabel("Click a trace or endpoint in either view. Wheel to zoom.")
        self.selection_label.setWordWrap(True)
        vl.addWidget(self.selection_label)
        self.persistence = QLabel()
        self.persistence.setWordWrap(True)
        outer.addWidget(self.persistence)
        self.status = QLabel()
        self.status.setWordWrap(True)
        outer.addWidget(self.status)
        footer = QHBoxLayout()
        outer.addLayout(footer)
        self.svg, self.dxf = QCheckBox("SVG copper"), QCheckBox("DXF outlines")
        for box in (self.svg, self.dxf):
            box.setChecked(True)
            footer.addWidget(box)
        footer.addStretch()
        self.export_button = self.button("Export PCB CAD package…", self.export, footer)
        self._refreshing = False
        self.refresh()

    def button(self, label, callback, layout):
        b = QPushButton(label)
        b.clicked.connect(callback)
        layout.addWidget(b)
        return b

    def page(self, name):
        content = QWidget()
        layout = QVBoxLayout(content)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        self.tools.addTab(scroll, name)
        return layout

    def add_number(self, form, key, label, factor, minimum, maximum, decimals):
        field = QDoubleSpinBox()
        field.setRange(minimum, maximum)
        field.setDecimals(decimals)
        field.setKeyboardTracking(False)
        field.editingFinished.connect(lambda: self.edit("settings", **{key: field.value() / factor}))
        self.inputs[key] = (field, factor)
        form.addRow(label, field)

    def edit(self, operation, **values):
        if not self._busy:
            self.run(lambda: self.session.change(operation, **values))

    def open_tab(self, index):
        self.controls.setChecked(True)
        self.tools.setCurrentIndex(index)

    def endpoint_changed(self, combo):
        if self._refreshing:
            return
        self.select(combo.currentData() or "")
        self.update_preview()

    def update_preview(self):
        if self._refreshing or not hasattr(self, "scene"):
            return
        self.preview = None
        if not self._busy and self.tools.currentIndex() == 2:
            try:
                self.preview = self.session.preview_connection(
                    self.end_a.currentData(), self.end_b.currentData(), self.connection_layer.currentText()
                )
                text = "Proposed connection\n" + self.preview["summary"]
                text += "\n" + "\n".join(self.preview["warnings"])
                text += "\n" + self.preview["scope"]
                self.proposal_label.setText(text)
            except (ValueError, KeyError) as error:
                self.proposal_label.setText(str(error))
        self.connect_button.setEnabled(bool(self.preview and not self.preview["duplicate"] and not self._busy))
        self.select(self.selection)

    def run(self, operation, completed=None):
        if self._busy:
            return
        self._busy = True
        self.preview = None
        self.select(self.selection)
        self.tools.setEnabled(False)
        self.export_button.setEnabled(False)
        self.part.setEnabled(False)
        self.undo_button.setEnabled(False)
        self.redo_button.setEnabled(False)
        self.status.setText("Updating PCB layout and checks…")
        task = self._task = _Task(operation)

        def done(value, error):
            self._busy = False
            self.tools.setEnabled(True)
            self.part.setEnabled(True)
            self.refresh()
            if error:
                self.status.setText("Action failed; layout unchanged. " + error)
            elif completed:
                completed(value)

        task.signals.finished.connect(done)
        QThreadPool.globalInstance().start(task)

    def refresh(self):
        self._refreshing = True
        previous_groups = getattr(self, "scene", {}).get("issue_groups", [])
        row = self.checks.currentRow()
        group_key = previous_groups[row]["key"] if 0 <= row < len(previous_groups) else ""
        scene = self.scene = self.session.scene()
        with QSignalBlocker(self.part):
            self.part.setCurrentIndex(self.session.part)
        with QSignalBlocker(self.mode):
            self.mode.setCurrentIndex(self.mode.findData(scene["settings"]["mode"]))
        self.seam.setValue(round(scene["settings"]["seam_deg"]))
        self.seam_label.setText(f"Opening seam: {scene['settings']['seam_deg']:g}°")
        self.seam.setEnabled(scene.get("kind") == "cylinder")
        self.auto_seam.setEnabled(self.seam.isEnabled())
        for key, (field, factor) in self.inputs.items():
            with QSignalBlocker(field):
                field.setValue(scene["settings"][key] * factor)
        for combo, rows in (
            (
                self.trace_list,
                [(t["label"], t["key"]) for t in scene.get("traces", []) if not t["key"].startswith("link")],
            ),
            (self.end_a, [(e["label"], e["key"]) for e in scene.get("endpoints", [])]),
            (self.end_b, [(e["label"], e["key"]) for e in scene.get("endpoints", [])]),
        ):
            previous = combo.currentData()
            with QSignalBlocker(combo):
                combo.clear()
                for label, value in rows:
                    combo.addItem(label, value)
                found = combo.findData(previous)
                combo.setCurrentIndex(found if found >= 0 else min(1, combo.count() - 1) if combo is self.end_b else 0)
        self.connection_list.clear()
        self.connection_list.addItems(scene["connection_labels"])
        groups = scene["issue_groups"]
        with QSignalBlocker(self.checks):
            self.checks.clear()
            self.checks.addItems([g["label"] for g in groups])
            self.checks.setCurrentRow(
                next((i for i, g in enumerate(groups) if g["key"] == group_key), 0) if groups else -1
            )
        self.location_index = 0
        self.show_issue(focus=False)
        self.evaluate_button.setEnabled(bool(scene.get("circuit_complete")))
        self.undo_button.setEnabled(scene["undo"])
        self.redo_button.setEnabled(scene["redo"])
        self.export_button.setEnabled(bool(self.session.design))
        self.export_button.setText(scene["review"]["export_label"] + "…")
        self.summary.setText(scene["review"]["summary"])
        self.persistence.setText(scene["persistence"])
        self.tools.setTabText(3, "Review" if scene["problem"] else f"Review · {scene['review']['errors']}")
        size = scene.get("size_m", [])
        self.dimensions.setText(
            " × ".join(f"{v * 1000:.2f}" for v in size) + " mm\nTop and bottom share one origin; no mirroring."
            if size
            else "No flat PCB development for this support."
        )
        evaluation = scene.get("evaluation")
        self.performance.setText(
            f"Target RMS: {evaluation['target_rms_uT']:.4g} µT\nChange from saved coil: {evaluation['delta_rms_uT']:.4g} µT RMS\n{evaluation['field_budget_summary']}\nTrace R: {evaluation['trace_resistance_ohm']:.4g} Ω · {evaluation['trace_power_W']:.4g} W at {evaluation['current_A']:.4g} A\n{evaluation['scope']}"
            if evaluation
            else "Field and electrical review have not been run for this layout."
        )
        self.status.setText(scene["problem"] or "PCB layout updated.")
        self._refreshing = False
        self.update_preview()

    def pick(self, key):
        if any(e["key"] == key for e in self.scene.get("endpoints", [])):
            combo = self.end_a if self.pick_endpoint.currentIndex() == 0 else self.end_b
            with QSignalBlocker(combo):
                combo.setCurrentIndex(combo.findData(key))
            self.pick_endpoint.setCurrentIndex(1)
        self.select(key)
        self.update_preview()

    def select(self, key):
        self.selection = key
        pins = (
            [
                next((e for e in self.scene.get("endpoints", []) if e["key"] == combo.currentData()), None)
                for combo in (self.end_a, self.end_b)
            ]
            if self.tools.currentIndex() == 2
            else []
        )
        for view in (self.flat, self.wrapped):
            view.display(
                self.scene, key, self.layer, compare=self.compare.isChecked(), pins=pins, proposal=self.preview
            )
        if key:
            row = next((e for e in self.scene.get("endpoints", []) if e["key"] == key), None)
            trace = next((t for t in self.scene.get("traces", []) if t["key"] == key.split(":")[0]), None)
            label = row["label"] if row else trace["label"] if trace else "Selected: " + key
            if self.compare.isChecked() and trace and trace.get("comparison"):
                label += (
                    f" · {trace['comparison']['distance_mm']:.3f} mm maximum saved-to-PCB displacement (purple in 3D)"
                )
            self.selection_label.setText(label)
            with QSignalBlocker(self.trace_list):
                index = self.trace_list.findData(key.split(":")[0])
                if index >= 0:
                    self.trace_list.setCurrentIndex(index)
        else:
            self.selection_label.setText("Click a trace or endpoint in either view. Wheel to zoom.")

    def set_layer(self, layer):
        self.layer = layer
        self.select(self.selection)

    def select_issue(self, index):
        self.location_index = 0
        self.show_issue()

    def next_issue_location(self):
        self.location_index += 1
        self.show_issue()

    def show_issue(self, *, focus=True):
        groups = self.scene.get("issue_groups", [])
        row = self.checks.currentRow()
        if not 0 <= row < len(groups):
            self.issue_detail.setText("Geometric PCB checks passed." if self.scene.get("board") else "")
            self.next_location.setEnabled(False)
            self.repair_button.setVisible(False)
            return
        group = groups[row]
        indices = group["indices"]
        location = self.location_index % len(indices)
        issue = self.scene["issues"][indices[location]]
        detail = issue["text"] + f"\nFinding {location + 1} / {len(indices)}"
        if issue["uv"]:
            detail += "\nBoard location: " + ", ".join(f"{v * 1000:.2f}" for v in issue["uv"]) + " mm"
        self.issue_detail.setText(detail + "\n\n" + group["hint"])
        self.next_location.setEnabled(len(indices) > 1)
        self.repair_button.setVisible(bool(group["action"]))
        self.repair_button.setText(group["action_label"])
        if focus:
            self.layer = "both"
            if issue["code"] == "off_surface":
                self.compare.setChecked(True)
            self.select(issue["trace"])
            if issue["uv"]:
                self.flat.focus(issue["uv"])
                self.wrapped.focus(issue["xyz"])

    def repair(self):
        group = self.scene["issue_groups"][self.checks.currentRow()]
        action = group["action"]
        if action == "seam":
            self.edit("seam")
        elif action == "compare":
            self.compare.setChecked(True)
            self.show_issue()
        elif action == "export":
            self.status.setText(
                "Finish pads, drills, plating and stackup in PCB CAD. " + self.scene["review"]["summary"]
            )
        else:
            self.open_tab({"board": 0, "copper": 1, "connect": 2}[action])

    def save_draft(self):
        if self._busy:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save PCB draft copy", self.session.filename() + ".cgpcb", "PCB draft (*.cgpcb)"
        )
        if path:
            try:
                from ..persistence import atomic_text

                atomic_text(path, json.dumps(self.session.document(), indent=2, allow_nan=False))
                self.status.setText("PCB draft saved: " + path)
            except OSError as e:
                self.status.setText(str(e))

    def load_draft(self):
        if self._busy:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open PCB draft", "", "PCB draft (*.cgpcb)")
        if path:

            def load():
                if Path(path).stat().st_size > 10 * 1024**2:
                    raise ValueError("PCB draft exceeds 10 MiB.")
                self.session.load(json.loads(Path(path).read_text(encoding="utf-8")))

            self.run(load)

    def export(self):
        if self._busy:
            return
        formats = tuple(f for f, box in (("svg", self.svg), ("dxf", self.dxf)) if box.isChecked())
        path, _ = QFileDialog.getSaveFileName(
            self, self.scene["review"]["export_label"], self.session.filename("artwork") + ".zip", "PCB package (*.zip)"
        )
        if path:
            self.run(
                lambda: self.session.export(path, formats),
                lambda value: self.status.setText("Exported PCB CAD artwork: " + str(value)),
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "controls"):
            compact = self.width() < 1000
            if compact != self._compact:
                self._compact = compact
                self.controls.setChecked(not compact)

    def closeEvent(self, event):
        if self._busy:
            event.ignore()
        else:
            event.accept()
