"""Run-bound field maps, continuous slices and solid-harmonic diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
import json
from pathlib import Path
from threading import Event
import time

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..application.field_analysis import FieldDataset, analyze_harmonics, metric_map, source_catalog
from ..application.field_sampling import sample_slices
from ..attribution import AUTHOR_CREDIT, SOFTWARE_NAME, software_attribution
from ..persistence import atomic_text_writer, json_value
from ..progress import DesignCancelled


@dataclass(frozen=True, slots=True)
class AnalysisSelection:
    source_key: str
    component: str
    metric: str
    sampling: str
    normalization: str
    display_unit: str
    plane: str
    center_m: tuple
    size_m: tuple
    offsets_m: tuple
    resolution: int
    harmonic_order: int
    harmonic_center_m: tuple
    harmonic_radius_m: float


@dataclass(frozen=True, slots=True)
class AnalysisState:
    selection: AnalysisSelection
    dataset: FieldDataset
    mapped: object
    slices: object
    slice_values: tuple
    error_map: object = None
    harmonics: object = None


class _Signals(QObject):
    progress = Signal(object, str)
    finished = Signal(object, str)


class _AnalysisTask(QRunnable):
    def __init__(self, operation, generation):
        super().__init__()
        self.operation, self.generation = operation, generation
        self.signals, self.cancel = _Signals(), Event()
        self._last_update, self._last_percent, self._last_message = 0.0, None, ""

    def check(self):
        if self.cancel.is_set():
            raise DesignCancelled("Field analysis cancelled between numerical operations.")

    def report(self, percent, message):
        self.check()
        percent = None if percent is None else float(percent)
        now, message = time.monotonic(), str(message)
        changed_kind = (percent is None) != (self._last_percent is None)
        changed_percent = percent is not None and (
            self._last_percent is None or percent >= self._last_percent + 1 or percent >= 100
        )
        if changed_kind or changed_percent or message != self._last_message or now - self._last_update >= 0.1:
            self.signals.progress.emit(percent, message)
            self._last_update, self._last_percent, self._last_message = now, percent, message

    def run(self):
        try:
            self.check()
            result = self.operation(self)
            self.check()
        except Exception as exc:
            self.signals.finished.emit(
                None, "Cancelled" if isinstance(exc, (DesignCancelled, InterruptedError)) else str(exc)
            )
        else:
            self.signals.finished.emit(result, "")


class _FigurePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(8, 5), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas)

    def draw(self):
        self.canvas.draw_idle()

    def save_png(self, path):
        suffix = Path(path).suffix.lower()
        key = "Creator" if suffix in {".pdf", ".svg"} else "Software" if suffix in {"", ".png"} else None
        metadata = None if key is None else {key: f"{SOFTWARE_NAME} — {AUTHOR_CREDIT}"}
        self.figure.savefig(path, dpi=180, metadata=metadata)


class _UnavailablePlot(QLabel):
    def __init__(self, reason, parent=None):
        super().__init__(
            "Plot rendering unavailable: " + reason + "\nNumerical tables and data exports remain available.", parent
        )
        self.setWordWrap(True)
        self.figure = None


class _PointCloudView(QWidget):
    """A true world-coordinate 3D view, with bounded display-only sampling."""

    def __init__(self, parent=None):
        super().__init__(parent)
        if QGuiApplication.platformName() in {"offscreen", "minimal"}:
            raise RuntimeError("3D rendering requires a graphics-capable Qt session")
        import pyvista as pv
        from pyvistaqt import QtInteractor

        self.pv, self.plotter, self.first = pv, QtInteractor(self), True
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        controls = QHBoxLayout()
        fit, isometric = QPushButton("Fit 3D"), QPushButton("Isometric")
        fit.clicked.connect(self.plotter.reset_camera)
        isometric.clicked.connect(self.plotter.view_isometric)
        controls.addWidget(fit)
        controls.addWidget(isometric)
        controls.addStretch()
        layout.addLayout(controls)
        layout.addWidget(self.plotter)

    def set_field(self, points, values, label, unit, limits, signed):
        self.plotter.clear()
        count = len(points)
        indices = (
            np.arange(count)
            if count <= 50_000
            else np.unique(np.r_[np.linspace(0, count - 1, 50_000, dtype=int), np.argmax(np.abs(values))])
        )
        cloud = self.pv.PolyData(np.asarray(points)[indices])
        cloud["value"] = np.asarray(values)[indices]
        self.plotter.add_mesh(
            cloud,
            scalars="value",
            cmap="coolwarm" if signed else "viridis",
            clim=limits,
            point_size=5,
            render_points_as_spheres=True,
            scalar_bar_args={"title": f"{label} [{unit}]"},
            reset_camera=False,
        )
        self.plotter.show_axes()
        self.plotter.show_grid(xtitle="x [m]", ytitle="y [m]", ztitle="z [m]")
        self.plotter.render()

    def save_png(self, path):
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo

        metadata = PngInfo()
        metadata.add_text("Software", f"{SOFTWARE_NAME} — {AUTHOR_CREDIT}")
        Image.fromarray(self.plotter.screenshot(return_img=True)).save(path, pnginfo=metadata)

    def set_roi(self, polylines):
        for index, points in enumerate(polylines):
            self.plotter.add_mesh(
                self.pv.lines_from_points(points),
                color="#252525",
                line_width=1.5,
                name=f"owned_roi_{index}",
                reset_camera=False,
            )
        if self.first:
            self.plotter.view_isometric()
            self.plotter.reset_camera()
            self.first = False
        self.plotter.render()

    def closeEvent(self, event):
        self.plotter.close()
        super().closeEvent(event)


def _numbers(text, count, label):
    try:
        values = tuple(float(value) for value in text.replace(",", " ").split())
    except ValueError as exc:
        raise ValueError(f"{label} requires {count} finite values.") from exc
    if len(values) != count or not np.all(np.isfinite(values)):
        raise ValueError(f"{label} requires {count} finite values.")
    return values


def _portable(value):
    """Serialize frozen scientific records without deepcopying mapping proxies."""
    if is_dataclass(value):
        return {item.name: _portable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {key: _portable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_portable(item) for item in value]
    return json_value(value)


def owned_roi_polylines(geometry):
    """Small explicit primitive wireframes; never infer a volume from samples."""
    if geometry is None:
        return ()
    from ..application.gradient_studies import RigidTransform
    from ..target_regions import TargetRegion

    region = TargetRegion.from_dict(geometry["region"])
    angle = np.linspace(0, 2 * np.pi, 97)
    lines = []
    if region.shape in {"sphere", "ellipsoid"}:
        for normal in range(3):
            line = np.zeros((len(angle), 3))
            axes = [axis for axis in range(3) if axis != normal]
            line[:, axes] = np.column_stack((np.cos(angle), np.sin(angle)))
            lines.append(line * region.half_extents_m)
    elif region.shape == "box":
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        for first in range(8):
            for second in range(first + 1, 8):
                if np.count_nonzero(corners[first] != corners[second]) == 1:
                    lines.append(corners[[first, second]] * region.half_extents_m)
    elif region.shape == "cylinder":
        for sign in (-1, 1):
            lines.append(
                np.column_stack(
                    (
                        region.radius_m * np.cos(angle),
                        region.radius_m * np.sin(angle),
                        np.full(len(angle), sign * region.length_m / 2),
                    )
                )
            )
        for phi in np.arange(4) * np.pi / 2:
            lines.append(
                np.array(
                    [
                        [region.radius_m * np.cos(phi), region.radius_m * np.sin(phi), -region.length_m / 2],
                        [region.radius_m * np.cos(phi), region.radius_m * np.sin(phi), region.length_m / 2],
                    ]
                )
            )
    result = []
    for line in lines:
        points = line @ region.rotation_matrix.T + region.center_m
        for key in ("registration", "sample_transform"):
            transform = RigidTransform(**geometry[key])
            points = points @ np.asarray(transform.rotation).T + transform.translation_m
        points.setflags(write=False)
        result.append(points)
    return tuple(result)


def owned_roi_bounds(geometry):
    """Exact world AABB of an authored primitive, independent of sample density."""
    if geometry is None:
        return None
    from ..application.gradient_studies import RigidTransform
    from ..target_regions import TargetRegion

    region = TargetRegion.from_dict(geometry["region"])
    if region.shape == "enclosing_mesh":
        return None
    rotation, center = region.rotation_matrix, np.asarray(region.center_m)
    for key in ("registration", "sample_transform"):
        transform = RigidTransform(**geometry[key])
        pose = np.asarray(transform.rotation)
        rotation, center = pose @ rotation, center @ pose.T + transform.translation_m
    if region.shape in {"sphere", "ellipsoid"}:
        half = np.linalg.norm(rotation * region.half_extents_m, axis=1)
    elif region.shape == "box":
        half = np.abs(rotation) @ region.half_extents_m
    else:
        half = region.radius_m * np.linalg.norm(rotation[:, :2], axis=1) + region.length_m / 2 * np.abs(rotation[:, 2])
    return np.stack((center - half, center + half))


class FieldAnalysisDialog(QDialog):
    """One immutable verified run; new input snapshots cannot relabel old results."""

    busyChanged = Signal(bool)
    progressChanged = Signal(str)
    progressDetails = Signal(object)
    closed = Signal()

    def __init__(
        self, record, parent=None, *, plot_factory=None, cloud_factory=None, auto_start=True, initial_source=None
    ):
        super().__init__(parent)
        self.record, self.datasets, self.state, self.worker = record, {}, None, None
        self._generation, self._pending, self._close_pending, self._closed = 0, False, False, False
        self._last_source_key = None
        self._plot_factory, self._cloud_factory = plot_factory or _FigurePanel, cloud_factory or _PointCloudView
        self._initial_source = {
            "contour_field": "contour",
            "contour_residual": "contour",
            "routed_field": "routed",
            "routed_residual": "routed",
            "winding_validation_residual": "winding_validation",
        }.get(initial_source, initial_source)
        self.setWindowTitle(f"Field analysis — {record.metadata.get('name', 'Verified run')}")
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.resize(1250, 820)
        layout = QVBoxLayout(self)
        self.identity = QLabel(
            f"Verified run · revision {record.metadata.get('project_revision', '—')} · "
            f"request {record.metadata.get('request_sha256', '')[:16]}\n"
            "Coordinates are world metres. Analysis retains the recorded source and current policy."
        )
        self.identity.setWordWrap(True)
        self.identity.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.identity)
        controls = QGridLayout()
        self.source, self.current_policy, self.component, self.metric_selector = (QComboBox() for _ in range(4))
        self.source.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.source.setMinimumContentsLength(24)
        self.current_policy.addItem("Stored evaluation", "recorded")
        self.current_policy.addItem("Stream-function current", "stream")
        for name in ["Bx", "By", "Bz", "Bparallel", "magnitude"]:
            self.component.addItem("|B|" if name == "magnitude" else name, name)
        self.component.setCurrentIndex(2)
        for title, value in [
            ("Field", "field"),
            ("Target", "target"),
            ("Signed residual", "signed_residual"),
            ("Absolute component error", "absolute_error"),
            ("Vector error norm", "vector_error"),
        ]:
            self.metric_selector.addItem(title, value)
        self.units, self.sampling, self.normalization = (QComboBox() for _ in range(3))
        self.units.addItems(["µT", "mT", "T"])
        self.sampling.addItem("Equal samples", "equal")
        self.sampling.addItem("Recorded weights", "weighted")
        self.normalization.addItem("Physical field units", "none")
        self.normalization.addItem("% of global target RMS", "target_rms")
        self.normalization.addItem("% of global target peak", "target_peak")
        for col, (title, widget) in enumerate(
            [
                ("Source", self.source),
                ("Current policy", self.current_policy),
                ("Component", self.component),
                ("Map", self.metric_selector),
                ("Unit", self.units),
                ("Sampling", self.sampling),
                ("Error normalization", self.normalization),
            ]
        ):
            controls.addWidget(QLabel(title), 0, col)
            controls.addWidget(widget, 1, col)
        controls.setColumnStretch(0, 2)
        layout.addLayout(controls)
        self.scope = QLabel("Loading available field sources…")
        self.scope.setWordWrap(True)
        layout.addWidget(self.scope)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self._build_cloud_tab()
        self._build_slices_tab()
        self._build_distribution_tab()
        self._build_harmonics_tab()
        self.status = QLabel("Only calculated, finite field sources are selectable.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        footer = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.hide()
        self.refresh = QPushButton("Evaluate maps")
        self.refresh.clicked.connect(self.refresh_analysis)
        self.cancel = QPushButton("Cancel calculation")
        self.cancel.clicked.connect(self._cancel)
        self.cancel.hide()
        self.export_data = QPushButton("Export analysis JSON / CSV…")
        self.export_data.clicked.connect(self._choose_export)
        self.export_plot = QPushButton("Export current plot PNG…")
        self.export_plot.clicked.connect(self._choose_plot)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        for widget in [self.progress, self.refresh, self.cancel, self.export_data, self.export_plot, close]:
            footer.addWidget(widget)
        layout.addLayout(footer)
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(250)
        self.debounce.timeout.connect(self.refresh_analysis)
        self.source.currentIndexChanged.connect(self._source_changed)
        for widget in [
            self.current_policy,
            self.component,
            self.metric_selector,
            self.units,
            self.sampling,
            self.normalization,
        ]:
            widget.currentIndexChanged.connect(self._changed)
        self.plane.currentIndexChanged.connect(self._plane_changed)
        for widget in [self.count, self.resolution, self.order]:
            (widget.currentIndexChanged if isinstance(widget, QComboBox) else widget.valueChanged).connect(
                self._changed
            )
        for widget in [self.center, self.size, self.offsets, self.harmonic_center, self.harmonic_radius]:
            widget.textEdited.connect(self._changed)
        self.profile_axis.currentIndexChanged.connect(lambda: self._draw_distribution() if self._current() else None)
        self.harmonic_source.currentIndexChanged.connect(lambda: self._draw_harmonics() if self._current() else None)
        self._enable(False)
        if auto_start:
            QTimer.singleShot(0, self, self.load_sources)

    def _plot(self, parent):
        try:
            return self._plot_factory(parent)
        except Exception as exc:
            return _UnavailablePlot(str(exc), parent)

    def _build_cloud_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        split = QSplitter()
        try:
            self.cloud = self._cloud_factory(split)
        except Exception as exc:
            self.cloud = _UnavailablePlot(str(exc), split)
        split.addWidget(self.cloud)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.summary = QTableWidget(0, 2)
        self.summary.setHorizontalHeaderLabels(["Metric / convention", "Value"])
        right_layout.addWidget(self.summary)
        self.samples = QTableWidget(0, 4)
        self.samples.setHorizontalHeaderLabels(["x [m]", "y [m]", "z [m]", "Map value"])
        right_layout.addWidget(QLabel("Largest absolute native-sample values · at most 256 rows"))
        right_layout.addWidget(self.samples)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        layout.addWidget(split)
        self.tabs.addTab(page, "3D samples and metrics")

    def _build_slices_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        row = QHBoxLayout()
        self.plane, self.count, self.resolution = QComboBox(), QSpinBox(), QSpinBox()
        for plane in ["xy", "xz", "yz"]:
            self.plane.addItem(plane.upper(), plane)
        self.count.setRange(1, 9)
        self.count.setValue(5)
        self.resolution.setRange(16, 128)
        self.resolution.setValue(64)
        self.center, self.size, self.offsets = QLineEdit("0 0 0"), QLineEdit("0.1 0.1"), QLineEdit("-0.04 0.04")
        for title, widget in [
            ("Plane", self.plane),
            ("Center xyz [m]", self.center),
            ("Full span [m]", self.size),
            ("Offsets from center [m]", self.offsets),
            ("Slices", self.count),
            ("Pixels / axis", self.resolution),
        ]:
            row.addWidget(QLabel(title))
            row.addWidget(widget)
        layout.addLayout(row)
        self.slice_note = QLabel(
            "Grey pixels are unsupported. Dashed outlines identify the owned ROI or sampling hull."
        )
        self.slice_note.setWordWrap(True)
        fit_row = QHBoxLayout()
        fit_row.addWidget(self.slice_note, 1)
        self.fit_source = QPushButton("Fit source / ROI")
        self.fit_source.setToolTip(
            "Fit the selected owned ROI; use saved sample bounds when its geometry is unavailable."
        )
        self.fit_source.clicked.connect(self._fit_source)
        fit_row.addWidget(self.fit_source)
        layout.addLayout(fit_row)
        self.slice_plot = self._plot(page)
        layout.addWidget(self.slice_plot, 1)
        self.tabs.addTab(page, "Linked slice grid")

    def _build_distribution_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        row = QHBoxLayout()
        row.addWidget(QLabel("Residual sample trend along"))
        self.profile_axis = QComboBox()
        self.profile_axis.addItems(["x", "y", "z"])
        row.addWidget(self.profile_axis)
        row.addWidget(QLabel("Other coordinates vary; each plotted point is an actual saved sample."), 1)
        layout.addLayout(row)
        self.error_plot = self._plot(page)
        layout.addWidget(self.error_plot, 1)
        self.tabs.addTab(page, "Error histogram and profile")

    def _build_harmonics_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        row = QHBoxLayout()
        self.order, self.harmonic_center, self.harmonic_radius = QSpinBox(), QLineEdit("0 0 0"), QLineEdit("0.05")
        self.order.setRange(0, 12)
        self.order.setValue(4)
        self.fit = QPushButton("Fit field / target / residual")
        self.fit.clicked.connect(self.fit_harmonics)
        self.harmonic_source = QComboBox()
        for source in ["field", "target", "residual"]:
            self.harmonic_source.addItem(source.capitalize(), source)
        for title, widget in [
            ("Maximum degree", self.order),
            ("Expansion center [m]", self.harmonic_center),
            ("Reference radius [m]", self.harmonic_radius),
            ("Coefficients", self.harmonic_source),
        ]:
            row.addWidget(QLabel(title))
            row.addWidget(widget)
        row.addWidget(self.fit)
        layout.addLayout(row)
        self.harmonic_note = QLabel(
            "Real solid harmonics use the displayed center and reference radius. A scalar expansion requires a source-free region; a numerical fit alone does not verify this assumption."
        )
        self.harmonic_note.setWordWrap(True)
        layout.addWidget(self.harmonic_note)
        self.harmonic_plot = self._plot(page)
        layout.addWidget(self.harmonic_plot, 2)
        self.coefficients = QTableWidget(0, 4)
        self.coefficients.setHorizontalHeaderLabels(["Degree l", "Order m", "Real harmonic", "Coefficient [µT]"])
        layout.addWidget(self.coefficients, 1)
        self.tabs.addTab(page, "Solid harmonics")

    def _enable(self, enabled):
        for widget in [
            self.source,
            self.current_policy,
            self.component,
            self.metric_selector,
            self.units,
            self.sampling,
            self.normalization,
            self.refresh,
            self.fit,
        ]:
            widget.setEnabled(enabled)
        self.export_data.setEnabled(enabled and self._current())
        self.export_plot.setEnabled(enabled and self._current())

    def load_sources(self):
        if self.worker is None and not self._close_pending and not self._closed:

            def operation(task):
                from ..application.evidence import load_saved_request

                task.report(None, "Read verified field source arrays")
                has_request = any(
                    item.get("name") == "request.json" for item in self.record.metadata.get("artifacts", ())
                )
                request = load_saved_request(self.record) if has_request else None
                sources = source_catalog(self.record, request=request, progress=task.report, cancel_check=task.check)
                task.report(100, "Available field sources loaded")
                return ("catalog", sources)

            self._start(operation)

    def _source_changed(self):
        if not self.datasets:
            return
        key = self.source.currentData()
        dataset = self.datasets.get(key)
        if dataset is None:
            return
        self.current_policy.blockSignals(True)
        self.current_policy.setItemText(
            0,
            "Original saved current" if dataset.provenance.get("current_policy") == "original" else "Stored evaluation",
        )
        self.current_policy.setCurrentIndex(0)
        self.current_policy.setEnabled(key + "_at_stream_current" in self.datasets)
        self.current_policy.blockSignals(False)
        weighted = bool(dataset.weights is not None and np.any(dataset.weights > 0))
        self.sampling.setEnabled(weighted)
        if not weighted:
            self.sampling.setCurrentIndex(0)
        for index in range(1, self.metric_selector.count()):
            self.metric_selector.model().item(index).setEnabled(dataset.target_T is not None)
        for index in range(1, self.normalization.count()):
            self.normalization.model().item(index).setEnabled(dataset.target_T is not None)
        if dataset.target_T is None:
            self.metric_selector.setCurrentIndex(0)
            self.normalization.setCurrentIndex(0)
        self.scope.setText(dataset.label + " · " + dataset.scope)
        self.source.setToolTip(dataset.label)
        previous = self.datasets.get(self._last_source_key)
        if previous is None or previous.target_geometry != dataset.target_geometry:
            self._fit_source(initialize=previous is None)
        self._last_source_key = key
        self._changed()

    def _dataset(self):
        key = self.source.currentData()
        if self.current_policy.currentData() == "stream":
            key += "_at_stream_current"
        if key not in self.datasets:
            raise ValueError("The selected source/current policy is unavailable in this run.")
        return self.datasets[key]

    def _source_bounds(self):
        dataset = self._dataset()
        bounds = owned_roi_bounds(dataset.target_geometry)
        return bounds if bounds is not None else np.stack((dataset.points_m.min(axis=0), dataset.points_m.max(axis=0)))

    def _fit_source(self, _checked=False, *, initialize=False):
        if self.datasets:
            bounds = self._source_bounds()
            center, span = bounds.mean(axis=0), bounds[1] - bounds[0]
            horizontal, vertical, normal = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}[self.plane.currentData()]
            radius = max(float(span.max()) / 2, 1e-6)
            self.center.setText(" ".join(f"{value:.10g}" for value in center))
            self.size.setText(
                " ".join(f"{max(span[axis] * 1.05, radius * 0.2):.8g}" for axis in (horizontal, vertical))
            )
            self.offsets.setText(f"{-span[normal] * 0.4:.8g} {span[normal] * 0.4:.8g}")
            if initialize:
                self.harmonic_center.setText(self.center.text())
                geometry = self._dataset().target_geometry
                reference = (
                    geometry["region"]["radius_m"]
                    if geometry is not None and geometry["region"]["shape"] == "sphere"
                    else max(float(np.linalg.norm(span / 2)), 1e-6)
                )
                self.harmonic_radius.setText(f"{reference:.10g}")
        self._changed()

    def _plane_changed(self, *_):
        self._fit_source()

    def selection(self):
        center = _numbers(self.center.text(), 3, "Slice center")
        size = _numbers(self.size.text(), 2, "Slice span")
        bounds = _numbers(self.offsets.text(), 2, "Slice offsets")
        harmonic_center = _numbers(self.harmonic_center.text(), 3, "Harmonic center")
        radius = _numbers(self.harmonic_radius.text(), 1, "Harmonic radius")[0]
        if min(size) <= 0 or radius <= 0 or bounds[0] > bounds[1]:
            raise ValueError("Slice spans and harmonic radius must be positive; offset minimum cannot exceed maximum.")
        relative = (0.5 * sum(bounds),) if self.count.value() == 1 else tuple(np.linspace(*bounds, self.count.value()))
        normal = {"xy": 2, "xz": 1, "yz": 0}[self.plane.currentData()]
        offsets = tuple(value + center[normal] for value in relative)
        return AnalysisSelection(
            self._dataset().key,
            self.component.currentData(),
            self.metric_selector.currentData(),
            self.sampling.currentData(),
            self.normalization.currentData(),
            self.units.currentText(),
            self.plane.currentData(),
            center,
            size,
            offsets,
            self.resolution.value(),
            self.order.value(),
            harmonic_center,
            radius,
        )

    def _changed(self, *_):
        self._generation += 1
        self._pending = bool(self.datasets)
        if self.worker is not None:
            self.worker.cancel.set()
        self.export_data.setEnabled(False)
        self.export_plot.setEnabled(False)
        self.status.setText("Inputs changed. Previous views remain visible until the new captured analysis completes.")
        self.debounce.start()

    def _current(self):
        if self.state is None:
            return False
        try:
            return self.selection() == self.state.selection
        except ValueError:
            return False

    def refresh_analysis(self):
        if self._close_pending or self._closed or not self.datasets:
            return
        if self.worker is not None:
            self._pending = True
            return
        self._pending = False
        try:
            selection, dataset = self.selection(), self._dataset()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        previous = self.state

        def slice_controls(item):
            return (item.source_key, item.plane, item.center_m, item.size_m, item.offsets_m, item.resolution)

        reused = (
            previous.slices
            if previous is not None and slice_controls(previous.selection) == slice_controls(selection)
            else None
        )

        def operation(task):
            task.report(0, "Calculate native-sample metrics")
            mapped = metric_map(
                dataset,
                component=selection.component,
                metric=selection.metric,
                sampling=selection.sampling,
                normalization=selection.normalization,
                progress=lambda percent, message: task.report(None if percent is None else 0.05 * percent, message),
                cancel_check=task.check,
            )
            error = (
                mapped
                if selection.metric in {"signed_residual", "absolute_error", "vector_error"}
                else (
                    metric_map(
                        dataset,
                        component=selection.component,
                        metric="signed_residual",
                        sampling=selection.sampling,
                        normalization=selection.normalization,
                        progress=lambda percent, message: task.report(
                            None if percent is None else 5 + 0.05 * percent, message
                        ),
                        cancel_check=task.check,
                    )
                    if dataset.target_T is not None
                    else None
                )
            )
            task.report(10, "Reuse captured physical slice grid" if reused is not None else "Evaluate slice grid")
            slices = (
                reused
                if reused is not None
                else sample_slices(
                    self.record,
                    dataset,
                    plane=selection.plane,
                    offsets_m=selection.offsets_m,
                    centre_m=selection.center_m,
                    size_m=selection.size_m,
                    resolution=selection.resolution,
                    progress=lambda percent, message: task.report(
                        None if percent is None else 10 + 0.8 * percent, message
                    ),
                    cancel_check=task.check,
                )
            )
            values = []
            for index, piece in enumerate(slices.slices):
                task.report(None, f"Calculate scalar slice map {index + 1}/{len(slices.slices)}")
                values.append(self._slice_values(dataset, mapped, piece, selection))
                task.report(
                    90 + 10 * (index + 1) / len(slices.slices),
                    f"Scalar slice map {index + 1}/{len(slices.slices)} complete",
                )
            values = tuple(values)
            task.report(100, "Field views ready")

            def harmonic_controls(item):
                return (
                    item.source_key,
                    item.component,
                    item.sampling,
                    item.harmonic_order,
                    item.harmonic_center_m,
                    item.harmonic_radius_m,
                )

            harmonics = (
                previous.harmonics
                if previous is not None and harmonic_controls(previous.selection) == harmonic_controls(selection)
                else None
            )
            return ("analysis", AnalysisState(selection, dataset, mapped, slices, values, error, harmonics))

        self._start(operation)

    @staticmethod
    def _slice_values(dataset, mapped, piece, selection):
        mask = np.asarray(piece.available_mask, dtype=bool).copy()
        if selection.metric != "field":
            if piece.target_T is None:
                return np.full(mask.shape, np.nan)
            mask &= np.asarray(piece.target_valid_mask, dtype=bool)
        values = np.full(mask.shape, np.nan)
        if np.any(mask):
            target = None if piece.target_T is None else piece.target_T[mask]
            # Slice maps share the original native-sample normalization. Their own
            # target RMS would otherwise change the engineering color scale.
            valid_target = target is not None and np.all(np.isfinite(target))
            view = FieldDataset(
                dataset.key,
                dataset.label,
                piece.coordinates_m[mask],
                piece.field_T[mask],
                target_T=target if valid_target else None,
                scope=dataset.scope,
                provenance=dataset.provenance,
            )
            scalar = metric_map(view, component=selection.component, metric=selection.metric, normalization="none")
            values[mask] = scalar.values
            if selection.normalization != "none":
                values[mask] *= 100 / mapped.normalization_scale_T
        values.setflags(write=False)
        return values

    def fit_harmonics(self):
        if self.worker is not None or not self._current():
            self.status.setText("Evaluate the current field views before fitting harmonics.")
            return
        state = self.state
        if state.selection.component == "magnitude":
            self.status.setText("Select Bx, By, Bz or Bparallel for a source-free scalar harmonic expansion.")
            return

        def operation(task):
            task.report(None, "Fit common real solid-harmonic basis")
            harmonic = analyze_harmonics(
                state.dataset,
                component=state.selection.component,
                max_order=state.selection.harmonic_order,
                center_m=state.selection.harmonic_center_m,
                radius_m=state.selection.harmonic_radius_m,
                sampling=state.selection.sampling,
                progress=task.report,
                cancel_check=task.check,
            )
            task.report(100, "Harmonic diagnostics ready")
            return ("harmonics", replace(state, harmonics=harmonic))

        self._start(operation)

    def _start(self, operation):
        if self._closed or self._close_pending or self.worker is not None:
            return
        self.debounce.stop()
        self.worker = _AnalysisTask(operation, self._generation)
        worker = self.worker
        worker.signals.progress.connect(self._progress)
        worker.signals.finished.connect(lambda result, error: self._finished(worker, result, error))
        self.progress.setRange(0, 0)
        self.progress.show()
        self.cancel.show()
        self.refresh.setEnabled(False)
        self.fit.setEnabled(False)
        self.busyChanged.emit(True)
        QThreadPool.globalInstance().start(worker)

    def _progress(self, percent, message):
        self.progress.setRange(0, 0 if percent is None else 100)
        if percent is not None:
            self.progress.setValue(int(max(0, min(100, percent))))
        self.status.setText(message)
        percentage = "" if percent is None else f" ({int(max(0, min(100, percent)))}%)"
        self.progressChanged.emit(f"Computing field analysis{percentage} · {message}")
        self.progressDetails.emit({"percent": percent, "progress_indeterminate": percent is None, "message": message})

    def _finished(self, worker, result, error):
        if worker is not self.worker:
            return
        self.worker = None
        if self._close_pending:
            self.progress.hide()
            self.cancel.hide()
            self.busyChanged.emit(False)
            self._finish_close()
            return
        stale = worker.generation != self._generation
        if error and not stale:
            self.status.setText(
                "Field analysis cancelled." if error == "Cancelled" else "Field analysis failed: " + error
            )
        elif result is not None and not stale:
            kind, value = result
            if kind == "catalog":
                self._catalog_loaded(value)
            elif kind in {"analysis", "harmonics"}:
                self.state = value
                self._progress(None, "Build captured field plots and tables")
                try:
                    self._render()
                except Exception as exc:
                    self.status.setText("Field plot publication failed: " + str(exc))
                if kind == "harmonics":
                    self.tabs.setCurrentIndex(3)
            else:
                self.status.setText("Saved captured field analysis: " + str(value))
        self.progress.hide()
        self.cancel.hide()
        self.busyChanged.emit(False)
        self._enable(bool(self.datasets))
        if self.datasets:
            self.current_policy.setEnabled(self.source.currentData() + "_at_stream_current" in self.datasets)
            self.sampling.setEnabled(bool(self._dataset().weights is not None and np.any(self._dataset().weights > 0)))
            self.units.setEnabled(self.normalization.currentData() == "none")
        if self._pending:
            QTimer.singleShot(0, self, self.refresh_analysis)

    def _catalog_loaded(self, datasets):
        self.datasets = {dataset.key: dataset for dataset in datasets}
        self.source.blockSignals(True)
        for dataset in datasets:
            if not dataset.key.endswith("_at_stream_current"):
                self.source.addItem(dataset.label, dataset.key)
        initial = (
            self._initial_source.removesuffix("_at_stream_current") if isinstance(self._initial_source, str) else None
        )
        preferred = [initial, "fabricated", "winding_validation", "routed", "validation", "sheet", "target"]
        for key in preferred:
            index = self.source.findData(key)
            if index >= 0:
                self.source.setCurrentIndex(index)
                break
        self.source.blockSignals(False)
        if not self.datasets:
            self.scope.setText("This verified run has no finite field maps to analyze.")
            self.status.setText("Compute a field-bearing design stage and open its saved evidence.")
            return
        component = self.component.findData(self.record.metadata.get("target_field_component", "Bz"))
        if component >= 0:
            self.component.setCurrentIndex(component)
        dataset = self._dataset()
        if dataset.target_T is not None:
            self.metric_selector.setCurrentIndex(self.metric_selector.findData("signed_residual"))
        self._source_changed()
        if self._initial_source in self.datasets and self._initial_source.endswith("_at_stream_current"):
            self.current_policy.setCurrentIndex(1)

    @staticmethod
    def _display(state):
        normalized = state.selection.normalization != "none"
        factor = 1.0 if normalized else {"T": 1.0, "mT": 1e3, "µT": 1e6}[state.selection.display_unit]
        return factor, "%" if normalized else state.selection.display_unit

    @staticmethod
    def _signed(state):
        if state.selection.metric == "signed_residual":
            return True
        return (
            state.selection.metric not in {"absolute_error", "vector_error"}
            and state.selection.component != "magnitude"
        )

    @staticmethod
    def _set_table(table, rows):
        table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for col, value in enumerate(values):
                text = (
                    "Unavailable"
                    if value is None
                    else f"{value:.7g}"
                    if isinstance(value, (float, np.floating))
                    else str(value)
                )
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(row, col, item)
        table.resizeColumnsToContents()

    def _render(self):
        state = self.state
        factor, unit = self._display(state)
        values = np.asarray(state.mapped.values) * factor
        finite = [np.abs(values[np.isfinite(values)])]
        finite += [np.abs(piece[np.isfinite(piece)]) * factor for piece in state.slice_values]
        maximum = max((float(piece.max()) for piece in finite if len(piece)), default=1.0) or 1.0
        limits = (-maximum, maximum) if self._signed(state) else (0, maximum)
        self._color_limits = limits
        self.scope.setText(
            f"{state.dataset.label} · {state.dataset.scope}\nSource current policy: {state.dataset.provenance.get('current_policy', 'saved')} · no current refit · "
            f"{len(state.dataset.points_m):,} native samples · {state.mapped.sampling_convention}"
            + (" · 3D display capped at 50,000 points; statistics use all samples" if len(values) > 50_000 else "")
            + (
                f"\nSaved sampling includes {state.dataset.provenance['suppression_sample_count']} suppression points. Equal statistics include them; weighted fits use positive weights."
                if state.dataset.provenance.get("suppression_sample_count", 0)
                else ""
            )
            + (
                (
                    "\nOwned ROI wireframe is shown in 3D."
                    if hasattr(self.cloud, "set_roi")
                    else "\nOwned ROI geometry is recorded; 3D wireframe rendering is unavailable."
                )
                if state.dataset.target_geometry is not None
                and state.dataset.target_geometry.get("region", {}).get("shape") != "enclosing_mesh"
                else "\nNo inferred enclosing volume is drawn; native sample locations define the displayed coverage."
            )
        )
        summary = []
        for key, value in state.mapped.summary.items():
            if key in {"rms", "peak", "std", "mean", "p95"}:
                summary.append((f"{key.upper()} [{unit}]", value * factor))
            elif key.endswith("_T") and value is not None:
                summary.append((key.removesuffix("_T").replace("_", " ") + " [µT]", value * 1e6))
            else:
                summary.append((key.replace("_", " "), value))
        self._set_table(self.summary, summary)
        ids = np.argsort(np.abs(values))[::-1][:256]
        self.samples.setHorizontalHeaderLabels(["x [m]", "y [m]", "z [m]", f"Map [{unit}]"])
        self._set_table(self.samples, [(*state.dataset.points_m[index], values[index]) for index in ids])
        graphics_errors = []
        try:
            if hasattr(self.cloud, "set_field"):
                self.cloud.set_field(
                    state.dataset.points_m, values, state.mapped.label, unit, limits, self._signed(state)
                )
            if hasattr(self.cloud, "set_roi"):
                self.cloud.set_roi(owned_roi_polylines(state.dataset.target_geometry))
        except Exception as exc:
            graphics_errors.append("3D rendering failed: " + str(exc))
        for title, draw in [
            ("Slice grid", self._draw_slices),
            ("Error distribution", self._draw_distribution),
            ("Harmonics", self._draw_harmonics),
        ]:
            try:
                draw()
            except Exception as exc:
                graphics_errors.append(title + " rendering failed: " + str(exc))
        self.status.setText(
            "Analysis complete. Maps, tables and exports refer to the captured source and controls."
            + ("\n" + "; ".join(graphics_errors) + ". Numerical exports remain available." if graphics_errors else "")
        )

    def _draw_slices(self):
        if self.slice_plot.figure is None:
            return
        state = self.state
        factor, unit = self._display(state)
        figure = self.slice_plot.figure
        figure.clear()
        count = len(state.slice_values)
        columns = min(3, count)
        rows = (count + columns - 1) // columns
        axes = figure.subplots(rows, columns, squeeze=False)
        horizontal, vertical = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[state.selection.plane]
        from matplotlib import colormaps

        cmap = colormaps["coolwarm" if self._signed(state) else "viridis"].with_extremes(bad="#eeeeee")
        image, visible = None, []
        for index, axis in enumerate(axes.flat):
            if index >= count:
                axis.set_visible(False)
                continue
            piece, values = state.slices.slices[index], state.slice_values[index]
            points = piece.coordinates_m
            image = axis.pcolormesh(
                points[:, :, horizontal],
                points[:, :, vertical],
                np.ma.masked_invalid(values * factor),
                shading="auto",
                cmap=cmap,
                vmin=self._color_limits[0],
                vmax=self._color_limits[1],
                rasterized=True,
            )
            for outline in piece.outlines_m:
                axis.plot(outline[:, horizontal], outline[:, vertical], "k--", linewidth=0.8)
            for outline in getattr(piece, "support_outlines_m", ()):
                axis.plot(outline[:, horizontal], outline[:, vertical], color="#525252", linewidth=0.7)
            roi = piece.roi_mask
            if roi is not None and np.any(roi) and not np.all(roi):
                axis.contour(
                    points[:, :, horizontal],
                    points[:, :, vertical],
                    roi.astype(float),
                    levels=[0.5],
                    colors="black",
                    linewidths=0.6,
                )
            normal = {"xy": 2, "xz": 1, "yz": 0}[state.selection.plane]
            axis.set_title(f"{'xyz'[normal]} = {piece.offset_m:.5g} m", fontsize=10)
            axis.set_xlabel(f"{'xyz'[horizontal]} [m]")
            axis.set_ylabel(f"{'xyz'[vertical]} [m]")
            axis.set_aspect("equal")
            visible.append(axis)
        if image is not None:
            figure.colorbar(image, ax=visible, label=f"{state.mapped.label} [{unit}]", shrink=0.82)
        descriptions = {
            "direct_sheet": "saved sheet, directly evaluated",
            "direct_wire": "saved winding, directly evaluated",
            "display_interpolation": "interpolated saved samples",
            "owned_analytic": "authored expression, original scale",
            "owned_geometry": "authored target boundary",
            "sampling_hull": "sample coverage",
            "unavailable_owned_enclosure": "target boundary unavailable",
            "unavailable": "unavailable",
        }
        self.slice_note.setText(
            f"Field: {descriptions.get(state.slices.representation, state.slices.representation.replace('_', ' '))} · "
            f"target: {descriptions.get(state.slices.target_representation, state.slices.target_representation.replace('_', ' '))} · "
            f"ROI: {descriptions.get(state.slices.roi_representation, state.slices.roi_representation.replace('_', ' '))}. "
            "Grey pixels have no supported value; one color scale is shared by every slice."
            + (
                "\n" + state.slices.provenance["target_domain_scope"]
                if state.slices.provenance.get("target_domain_scope")
                else ""
            )
        )
        self.slice_note.setToolTip(
            "\n".join(
                message
                for message in (
                    state.slices.provenance.get("fallback_reason"),
                    state.slices.provenance.get("roi_unavailable_reason"),
                )
                if message
            )
        )
        self.slice_plot.draw()

    def _draw_distribution(self):
        if self.error_plot.figure is None or self.state is None:
            return
        figure = self.error_plot.figure
        figure.clear()
        if self.state.error_map is None:
            figure.text(0.1, 0.5, "A target field is required for residual distributions.")
            self.error_plot.draw()
            return
        state = self.state
        factor, unit = self._display(state)
        values = np.asarray(state.error_map.values) * factor
        axis, profile = figure.subplots(1, 2)
        weights = state.dataset.weights if state.selection.sampling == "weighted" else None
        low, high = float(values.min()), float(values.max())
        histogram_range = None
        if np.isclose(low, high, rtol=1e-10, atol=0):
            padding = max(abs(low) * 0.05, 1e-12)
            histogram_range = (low - padding, high + padding)
        axis.hist(
            values,
            bins=min(60, max(8, int(np.sqrt(len(values))))),
            weights=weights,
            color="#23698c",
            range=histogram_range,
        )
        axis.set_xlabel(f"{state.error_map.label} [{unit}]")
        axis.set_ylabel("Recorded sample weight" if weights is not None else "Samples")
        index = self.profile_axis.currentIndex()
        ids = (
            np.arange(len(values))
            if len(values) <= 50_000
            else np.unique(np.r_[np.linspace(0, len(values) - 1, 50_000, dtype=int), np.argmax(np.abs(values))])
        )
        profile.scatter(state.dataset.points_m[ids, index], values[ids], s=7, alpha=0.45, color="#23698c")
        profile.axhline(0, color="black", linewidth=0.7)
        profile.set_xlabel(f"{'xyz'[index]} [m] · other coordinates vary")
        profile.set_ylabel(f"{state.error_map.label} [{unit}]")
        for item in [axis, profile]:
            item.grid(alpha=0.2)
        self.error_plot.draw()

    def _draw_harmonics(self):
        state = self.state
        if state is None:
            return
        figure = self.harmonic_plot.figure
        if figure is not None:
            figure.clear()
        if not state.harmonics:
            self.coefficients.setRowCount(0)
            if figure is not None:
                figure.text(0.1, 0.5, "Use Fit field / target / residual to inspect the common harmonic basis.")
                self.harmonic_plot.draw()
            return
        result = state.harmonics
        source = self.harmonic_source.currentData()
        fit = result["fits"].get(source)
        if fit is None:
            self.coefficients.setRowCount(0)
            self.harmonic_note.setText("This source has no target/residual harmonic fit.")
            if figure is not None:
                self.harmonic_plot.draw()
            return
        rows = [
            (term.degree, term.order, term.display_label, coefficient * 1e6)
            for term, coefficient in zip(fit.terms, fit.coefficients, strict=True)
        ]
        self._set_table(self.coefficients, rows)
        condition = result.get("condition_number", result.get("condition"))
        diagnostics = result.get("diagnostics", ())
        self.harmonic_note.setText(
            f"Rank {fit.rank}/{len(fit.terms)} · {fit.sample_count} samples · "
            f"condition {condition if condition is not None else 'unavailable'} · fit residual RMS {fit.residual_rms * 1e6:.5g} µT\n"
            + str(result.get("sourcefree_caveat", "Source-free validity is not established by this fit."))
            + ("\n" + str(diagnostics) if diagnostics else "")
        )
        if figure is not None:
            axis = figure.subplots()
            ids = np.arange(len(fit.terms))
            axis.bar(ids, fit.coefficients * 1e6, color="#23698c")
            axis.set_xticks(ids)
            axis.set_xticklabels([f"{term.degree},{term.order}" for term in fit.terms], rotation=90, fontsize=7)
            axis.set_xlabel("Real solid harmonic (degree, order)")
            axis.set_ylabel(f"{source.capitalize()} coefficient [µT]")
            axis.axhline(0, color="black", linewidth=0.7)
            self.harmonic_plot.draw()

    def export_to(self, path, *, state=None, progress=None, cancel_check=None):
        if state is None and not self._current():
            raise ValueError("Recompute the current controls before exporting their analysis.")
        state = self.state if state is None else state
        path = Path(path)

        def report(percent, message):
            if cancel_check is not None and cancel_check():
                raise DesignCancelled("Field analysis export cancelled.")
            if progress is not None:
                progress(percent, message)

        report(None, "Prepare captured field analysis export")
        if path.suffix.lower() == ".csv":
            columns = [state.dataset.points_m, state.dataset.field_T]
            header = "x_m,y_m,z_m,Bx_T,By_T,Bz_T"
            if state.dataset.target_T is not None:
                columns += [state.dataset.target_T, state.dataset.field_T - state.dataset.target_T]
                header += ",target_Bx_T,target_By_T,target_Bz_T,residual_Bx_T,residual_By_T,residual_Bz_T"
            if state.dataset.weights is not None:
                columns.append(state.dataset.weights[:, None])
                header += ",saved_sample_weight"
            columns.append(np.asarray(state.mapped.values)[:, None])
            component = "vector" if state.selection.metric == "vector_error" else state.selection.component
            header += "," + "_".join(
                (
                    component,
                    state.selection.metric,
                    state.selection.normalization,
                    state.selection.sampling,
                    "percent" if state.selection.normalization != "none" else "T",
                )
            )
            values = np.column_stack(columns)
            with atomic_text_writer(path) as output:
                for start in range(0, len(values), 4096):
                    report(99 * start / len(values), f"Write native samples {start:,}/{len(values):,}")
                    np.savetxt(
                        output,
                        values[start : start + 4096],
                        delimiter=",",
                        header=header if start == 0 else "",
                        comments="",
                    )
                report(99, "Flush captured CSV export")
        elif path.suffix.lower() == ".json":
            payload = {
                "schema": "coilgen-field-analysis-v1",
                "software": software_attribution(),
                "run_request_sha256": self.record.metadata.get("request_sha256"),
                "run_project_fingerprint": self.record.metadata.get("project_fingerprint"),
                "selection": state.selection,
                "source": state.dataset,
                "metric": state.mapped,
                "slices": state.slices,
                "slice_scalar_values": state.slice_values,
                "harmonics": state.harmonics,
                "display_note": "Native fields and slice fields are SI tesla; normalized metrics are percentages of a global target scale.",
            }
            portable = {}
            for name, value in payload.items():
                report(None, "Serialize captured analysis · " + name.replace("_", " "))
                portable[name] = _portable(value)
            # Encoded size is unknown before writing. Report observed bytes rather
            # than inventing a percentage; checkpoints preserve atomic cancellation.
            with atomic_text_writer(path) as output:
                encoder = json.JSONEncoder(allow_nan=False, indent=2)
                buffered, length, written = [], 0, 0
                for chunk in encoder.iterencode(portable):
                    buffered.append(chunk)
                    length += len(chunk)
                    if length >= 256 * 1024:
                        report(None, f"Write captured analysis JSON · {written / 1024**2:.1f} MiB")
                        text = "".join(buffered)
                        output.write(text)
                        written += len(text.encode("utf-8"))
                        buffered, length = [], 0
                report(None, "Flush captured JSON export")
                output.write("".join(buffered) + "\n")
        else:
            raise ValueError("Choose a .json analysis or .csv native-sample export.")
        report(100, "Captured numerical export complete")

    def _choose_export(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export captured field analysis",
            "field_analysis.json",
            "Analysis JSON (*.json);;Native samples CSV (*.csv)",
        )
        if path:
            if not self._current() or self.worker is not None:
                return
            state = self.state

            def operation(task):
                self.export_to(path, state=state, progress=task.report, cancel_check=task.check)
                task.report(100, "Analysis export complete")
                return ("export", path)

            self._start(operation)

    def _choose_plot(self):
        if not self._current():
            return
        plot = [self.cloud, self.slice_plot, self.error_plot, self.harmonic_plot][self.tabs.currentIndex()]
        if not hasattr(plot, "save_png"):
            self.status.setText(
                "The current graphics view is unavailable; export numerical analysis or a different plot."
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export current field plot", "field_analysis.png", "PNG image (*.png)"
        )
        if path:
            try:
                plot.save_png(path)
                self.status.setText("Saved current field plot: " + path)
            except Exception as exc:
                self.status.setText("Plot export failed: " + str(exc))

    def _cancel(self):
        self._pending = False
        self.debounce.stop()
        if self.worker is not None:
            self.worker.cancel.set()
            self.status.setText("Cancellation requested. Waiting for the current numerical operation to finish.")

    def request_close(self):
        self.reject()

    def _finish_close(self):
        self.debounce.stop()
        self._close_pending, self._pending = True, False
        if not self._closed:
            self._closed = True
            self.cloud.close()
            super().reject()
            self.closed.emit()

    def reject(self):
        if self.worker is not None:
            self._close_pending = True
            self._cancel()
            return
        self._finish_close()

    def closeEvent(self, event):
        if self.worker is not None:
            self.reject()
            event.ignore()
        else:
            self._finish_close()
            event.accept()
