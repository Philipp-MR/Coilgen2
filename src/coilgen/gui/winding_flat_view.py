"""Linked surface development with direct, metric-preserving sketch gestures."""

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget
from ..winding_visuals import group_color, direction_markers


class WindingFlatView(QWidget):
    selected = Signal(str)
    cut_moved = Signal(str, object)
    guide_moved = Signal(str, int, object)
    seam_drawn = Signal(object, bool)
    connection_drawn = Signal(str, str, object, bool)
    port_picked = Signal(str, str)
    ports_connected = Signal(str, str)
    bridge_target = Signal(str, object)
    message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(250, 200)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setObjectName("winding_flat_view")
        self.editor = None
        self.chart = None
        self.key = ""
        self.mode = "select"
        self.busy = False
        self.preview = None
        self._drag = None
        self._center = np.zeros(2)
        self._extent = np.ones(2)
        self._pan = np.zeros(2)
        self._zoom = 1.0
        self._lines = []
        self._ports = []
        self._handles = []
        self.error = ""

    @property
    def scale(self):
        return (
            min((self.width() - 50) / max(self._extent[0], 1e-6), (self.height() - 65) / max(self._extent[1], 1e-6))
            * self._zoom
        )

    def screen(self, uv):
        p = (
            (np.asarray(uv) - self._center) * [self.scale, -self.scale]
            + [self.width() / 2, self.height() / 2]
            + self._pan
        )
        return QPointF(*p)

    def uv(self, point):
        return (np.array([point.x(), point.y()]) - [self.width() / 2, self.height() / 2] - self._pan) / [
            self.scale,
            -self.scale,
        ] + self._center

    def fit(self):
        self._pan[:] = 0
        self._zoom = 1.0
        self.update()

    def display(self, editor, key, *, preview=None, **kwargs):
        first = self.editor is not editor
        self.editor, self.key = editor, key
        self.preview = preview if preview and preview["revision"] == editor.revision else None
        try:
            self.chart = editor.chart
            self.error = ""
        except ValueError as error:
            self.chart, self.error = None, str(error)
            self.update()
            return
        bounds = self.chart.face_uv.reshape(-1, 2)
        self._center = (bounds.min(axis=0) + bounds.max(axis=0)) / 2
        self._extent = np.ptp(bounds, axis=0)
        if first:
            self.fit()
        self._lines = [(row, self.chart.polyline(row.points)) for row in editor.sections]
        ports = editor.drawing_ports(key) if self.mode == "draw" else editor.ports
        self._ports = [(p, self.chart.project([p["point"]])[0]) for p in ports]
        self._handles = []
        row = editor.section(key)
        if self.mode == "select" and not row.locked and not row.disconnected and not row.bridge:
            if row.cut:
                self._handles = [("cut", 0, self.chart.project([editor.cut_point(key)])[0])]
            elif row.editable:
                self._handles = [
                    ("guide", i, p)
                    for i, p in enumerate(self.chart.project(row.guides or [row.points[len(row.points) // 2]]))
                ]
        self.update()

    def _path(self, painter, points, color, width=2, dashed=False):
        pen = QPen(QColor(color), width)
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        path, opened = QPainterPath(), False
        for p in points:
            if p is None:
                opened = False
            elif not opened:
                path.moveTo(self.screen(p))
                opened = True
            else:
                path.lineTo(self.screen(p))
        painter.drawPath(path)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f3f7f9"))
        if self.chart is None:
            painter.setPen(QColor("#526b7a"))
            painter.drawText(self.rect().adjusted(15, 15, -15, -15), Qt.TextFlag.TextWordWrap, self.error)
            return
        # One path avoids seams appearing darker than the source mesh edges.
        for face in self.chart.face_uv:
            self._path(painter, [*face, face[0]], "#e3eaf0", 0.5)
        for row, points in self._lines:
            color = group_color(row.owner.group_index)
            if row.key == self.key:
                self._path(painter, points, "#ffffff", 6, row.disconnected)
            self._path(
                painter,
                points,
                color,
                3 if row.key == self.key else 1.4,
                row.disconnected,
            )
            if not row.editable and row.owner.loop_index >= 0:
                chunks, chunk = [], []
                for point in points + [None]:
                    if point is None:
                        if len(chunk) > 1:
                            chunks.append(np.array(chunk))
                        chunk = []
                    else:
                        p = self.screen(point)
                        chunk.append([p.x(), p.y()])
                painter.setPen(QPen(QColor(color), 1.5))
                for chunk in chunks:
                    for p, tangent in direction_markers(chunk, 120):
                        normal = np.array([-tangent[1], tangent[0]])
                        painter.drawLine(QPointF(*(p - 5 * tangent + 2.5 * normal)), QPointF(*p))
                        painter.drawLine(QPointF(*p), QPointF(*(p - 5 * tangent - 2.5 * normal)))
        painter.setBrush(QColor("white"))
        painter.setPen(QPen(QColor("#007f83"), 2))
        for kind, index, uv in self._handles:
            p = self.screen(uv)
            painter.drawEllipse(p, 7, 7)
        if self.mode in {"connect", "draw"}:
            for port, uv in self._ports:
                p = self.screen(uv)
                painter.setBrush(QColor("#007f83" if port["role"] == "out" else "#bd663d"))
                painter.drawEllipse(p, 7, 7)
                painter.setPen(QColor("white"))
                painter.drawText(p + QPointF(-5, 4), "→" if port["role"] == "out" else "•")
        if self.preview:
            for route in self.preview.get("routes", []):
                self._path(painter, self.chart.polyline(np.asarray(route["points"])), "#0b9b91", 3, True)
            if "stroke" in self.preview:
                self._path(painter, self.preview["stroke"], "#007f83", 2, True)
            for cut in self.preview["cuts"]:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(self.screen(self.chart.project([cut["point"]])[0]), 7, 7)
        if self._drag and "stroke" in self._drag:
            self._path(painter, self._drag["stroke"], "#007f83", 3, True)
        if self._drag and "point" in self._drag:
            painter.setBrush(QColor("#007f83"))
            painter.drawEllipse(self.screen(self._drag["point"]), 6, 6)
        painter.setPen(QColor("#526b7a"))
        painter.drawText(12, 20, "Flat")

    def _hit(self, position, points):
        best = (float("inf"), None)
        xy = np.array([position.x(), position.y()])
        for a, b in zip(points[:-1], points[1:]):
            if a is None or b is None:
                continue
            pa, pb = self.screen(a), self.screen(b)
            start = np.array([pa.x(), pa.y()])
            direction = np.array([pb.x(), pb.y()]) - start
            t = np.clip(np.dot(xy - start, direction) / max(np.dot(direction, direction), 1e-20), 0, 1)
            distance = np.linalg.norm(xy - start - t * direction)
            if distance < best[0]:
                best = (distance, np.asarray(a) + t * (np.asarray(b) - a))
        return best

    def _port(self, position, role=None):
        hits = [
            ((self.screen(uv) - position).manhattanLength(), port)
            for port, uv in self._ports
            if role is None or port["role"] == role
        ]
        distance, port = min(hits, key=lambda hit: hit[0], default=(99, None))
        return port if distance < 18 else None

    def mousePressEvent(self, event):
        if self.busy or self.chart is None or event.button() != Qt.MouseButton.LeftButton:
            return
        self.setFocus()
        p = event.position()
        self._drag = {"press": p, "pan": self._pan.copy()}
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            return
        if self.mode == "draw":
            port = self._port(p, "out")
            if port and port["role"] == "out":
                self._drag.update(port=port, stroke=[self.chart.project([port["point"]])[0]])
                return
        if self.mode == "seam":
            try:
                self.chart.pick(self.uv(p))
                self._drag["stroke"] = [self.uv(p)]
            except ValueError:
                self._drag = None
                self.message.emit("Start the seam on the support.")
        elif self.mode == "connect":
            self._drag["port"] = self._port(p)
        else:
            for handle in self._handles:
                if (self.screen(handle[2]) - p).manhattanLength() < 18:
                    self._drag["handle"] = handle
                    return
            best = min(
                (
                    (self._hit(p, points)[0], row.key)
                    for row, points in self._lines
                    if self.mode != "bridge" or row.key != self.key
                    if self.mode != "draw" or row.editable
                ),
                default=(99, ""),
            )
            if best[0] < 9:
                self._drag["select"] = best[1]

    def mouseMoveEvent(self, event):
        if not self._drag:
            return
        d, p = self._drag, event.position()
        uv = self.uv(p)
        if "stroke" in d:
            try:
                self.chart.pick(uv)
                if len(d["stroke"]) < 512 and (self.screen(d["stroke"][-1]) - p).manhattanLength() > 3:
                    d["stroke"].append(uv)
            except ValueError:
                pass
        elif "handle" in d:
            if d["handle"][0] == "cut":
                uv = self._hit(p, self.chart.polyline(self.editor.cut_curve(self.key)))[1]
            try:
                self.chart.pick(uv)
                d["point"] = uv
            except (ValueError, TypeError):
                pass
        elif "select" not in d and "port" not in d:
            delta = p - d["press"]
            self._pan = d["pan"] + [delta.x(), delta.y()]
        self.update()

    def mouseReleaseEvent(self, event):
        d, self._drag = self._drag, None
        if not d:
            return
        moved = (event.position() - d["press"]).manhattanLength() >= 3
        if "stroke" in d and d.get("port") and moved:
            target = self._port(event.position(), "in")
            if target and target["role"] == "in":
                stroke = [*d["stroke"], self.chart.project([target["point"]])[0]]
                self.connection_drawn.emit(d["port"]["key"], target["key"], np.asarray(stroke).tolist(), True)
            else:
                self.message.emit("Finish on an orange endpoint.")
        elif "stroke" in d and not d.get("port") and len(d["stroke"]) >= 2:
            # Keep the release point even when it is below the motion sampling
            # threshold; otherwise a seam can silently miss its last turn.
            end = self.uv(event.position())
            try:
                self.chart.pick(end)
                if np.linalg.norm(end - d["stroke"][-1]) > 1e-12:
                    if len(d["stroke"]) >= 512:
                        d["stroke"][-1] = end
                    else:
                        d["stroke"].append(end)
            except ValueError:
                pass
            self.seam_drawn.emit(np.asarray(d["stroke"]).tolist(), True)
        elif d.get("port"):
            target = self._port(event.position())
            if moved and d["port"]["role"] == "out" and target and target["role"] == "in":
                self.ports_connected.emit(d["port"]["key"], target["key"])
            else:
                self.port_picked.emit(d["port"]["key"], d["port"]["role"])
        elif "handle" in d and "point" in d and moved:
            point = self.chart.pick(d["point"])
            if d["handle"][0] == "cut":
                self.cut_moved.emit(self.key, point)
            else:
                self.guide_moved.emit(self.key, d["handle"][1], point)
        elif "select" in d and not moved:
            if self.mode == "bridge":
                points = next(points for row, points in self._lines if row.key == d["select"])
                self.bridge_target.emit(d["select"], self.chart.pick(self._hit(event.position(), points)[1]))
            else:
                self.selected.emit(d["select"])
        self.update()

    def wheelEvent(self, event):
        anchor = self.uv(event.position())
        self._zoom = float(np.clip(self._zoom * np.exp(event.angleDelta().y() * 0.001), 0.2, 30))
        delta = event.position() - self.screen(anchor)
        self._pan += [delta.x(), delta.y()]
        self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self._drag:
            self._drag = None
            self.message.emit("Sketch cancelled.")
            self.update()
            event.accept()
        else:
            super().keyPressEvent(event)
