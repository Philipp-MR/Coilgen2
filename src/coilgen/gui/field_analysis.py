"""Run-bound field maps, continuous slices and solid-harmonic diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
import csv
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
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..application.field_analysis import FieldDataset, analyze_harmonics, metric_map, owned_roi_bounds
from ..application.field_presentation import field_provenance, paired_color_limits
from ..application.field_sampling import SliceInputCache, sample_slices
from ..application.shielding import shielding_sources, shielding_surface_fields
from ..attribution import AUTHOR_CREDIT, SOFTWARE_NAME, software_attribution
from ..field_components import field_component_values
from ..persistence import atomic_text_writer, json_value
from ..progress import DesignCancelled
from .coil_statistics import build_coil_statistics, display_quantity, mass_estimates
from .run_metadata import read_run_metadata
from .shielding_plot import (
    SHIELDING_COMPONENTS,
    SHIELDING_SOURCES,
    draw_shielding_surface,
    record_openings,
    shielding_limits,
)


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
    paired_peak: float = 0.0


class _Signals(QObject):
    progress = Signal(object, str)
    partial = Signal(object)
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
            from .design_tasks import deliver_task_signal

            deliver_task_signal(self.signals, "progress", percent, message)
            self._last_update, self._last_percent, self._last_message = now, percent, message

    def publish(self, result):
        from .design_tasks import deliver_task_signal

        self.check()
        deliver_task_signal(self.signals, "partial", result)

    def run(self):
        from .design_tasks import deliver_task_signal

        try:
            self.check()
            result = self.operation(self)
            self.check()
        except Exception as exc:
            deliver_task_signal(
                self.signals,
                "finished",
                None,
                "Cancelled" if isinstance(exc, (DesignCancelled, InterruptedError)) else str(exc),
            )
        else:
            deliver_task_signal(self.signals, "finished", result, "")


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


class _ControlSection(QWidget):
    """Keep plotting space available while exposing editable conventions."""

    def __init__(self, title, body):
        super().__init__()
        self.title, self.body = title, body
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.toggle = QToolButton()
        self.toggle.setCheckable(True)
        self.toggle.setAutoRaise(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle.setText(title)
        self.toggle.toggled.connect(self._expanded)
        layout.addWidget(self.toggle, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(body)
        body.hide()

    def set_summary(self, summary):
        self.toggle.setText(self.title + " · " + summary)

    def _expanded(self, expanded):
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.body.setVisible(expanded)


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


class FieldAnalysisDialog(QDialog):
    """One immutable verified run; new input snapshots cannot relabel old results."""

    busyChanged = Signal(bool)
    progressChanged = Signal(str)
    progressDetails = Signal(object)
    closed = Signal()

    _VIEW_TABS = {"field": 0, "metrics": 0, "slices": 1, "errors": 2, "harmonics": 3, "coil": 4, "shielding": 5}

    def __init__(
        self,
        record,
        parent=None,
        *,
        plot_factory=None,
        cloud_factory=None,
        auto_start=True,
        initial_source=None,
        initial_view="metrics",
    ):
        self._validate_view(initial_view)
        super().__init__(parent)
        self.record, self.datasets, self.state, self.worker = record, {}, None, None
        self._generation, self._pending, self._close_pending, self._closed = 0, False, False, False
        self._auto_start, self._catalog_ready, self._grid_requested = auto_start, False, False
        self._last_source_key = None
        self._slice_cache = SliceInputCache()
        self._showing_partial = False
        self._plot_factory, self._cloud_factory = plot_factory or _FigurePanel, cloud_factory or _PointCloudView
        self._initial_source = self._source_alias(initial_source)
        self._requested_view = initial_view
        metadata = read_run_metadata(record)
        context = metadata.get("design_context")
        for section in ("summary", "config"):
            nested = metadata.get(section)
            if context is None and isinstance(nested, Mapping):
                context = nested.get("design_context")
        self._is_shim = context == "mr_shim"
        self.setWindowTitle(f"Coil results — {metadata.get('name', 'Verified run')}")
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.resize(1250, 820)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(10)
        title = QLabel("Coil results")
        title_font = title.font()
        title_font.setPointSizeF(title_font.pointSizeF() * 1.6)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        self.identity = QLabel(
            f"{metadata.get('name', 'Verified run')} · revision {metadata.get('project_revision', '—')} · "
            f"request {metadata.get('request_sha256', '')[:16]}"
        )
        self.identity.setWordWrap(True)
        self.identity.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.identity)
        self.source, self.current_policy, self.component, self.metric_selector = (QComboBox() for _ in range(4))
        self.source.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.source.setMinimumContentsLength(16)
        self.current_policy.addItem("Recorded evaluation", "recorded")
        self.current_policy.addItem("Original contour current", "stream")
        for name in ["Bx", "By", "Bz", "Bparallel", "magnitude"]:
            self.component.addItem("|B|" if name == "magnitude" else name, name)
        self.component.setCurrentIndex(2)
        for title, value in [
            ("Field", "field"),
            ("Target", "target"),
            ("Signed component residual", "signed_residual"),
            ("Absolute component error", "absolute_error"),
            ("Vector residual magnitude |ΔB|", "vector_error"),
        ]:
            self.metric_selector.addItem(title, value)
        self.units, self.sampling, self.normalization = (QComboBox() for _ in range(3))
        self.units.addItems(["µT", "mT", "T"])
        self.sampling.addItem("Equal samples", "equal")
        self.sampling.addItem("Recorded weights", "weighted")
        self.normalization.addItem("Physical field units", "none")
        self.normalization.addItem("% of global target RMS", "target_rms")
        self.normalization.addItem("% of global target peak", "target_peak")
        self.current_policy.setToolTip(
            "Use the saved evaluation or the original stream-function current; no current refit."
        )
        self.metric_selector.setToolTip(
            "Signed residual uses field − target for the selected component. "
            "Vector residual magnitude uses the Euclidean norm of ΔB across all three components."
        )
        self.component.setToolTip("Bparallel projects onto the recorded B0 direction. |B| is the vector magnitude.")
        self.normalization.setToolTip(
            "One target scale from all native samples is shared by every slice; no pointwise division."
        )
        primary = QHBoxLayout()
        primary.addWidget(
            self._control_group(
                "Field source and current",
                [
                    ("Source", self.source),
                    ("Current policy", self.current_policy),
                ],
            ),
            3,
        )
        primary.addWidget(
            self._control_group(
                "Map to inspect",
                [
                    ("Component", self.component),
                    ("Map", self.metric_selector),
                ],
            ),
            2,
        )
        self.field_controls = QWidget()
        self.field_controls.setLayout(primary)
        layout.addWidget(self.field_controls)
        self.display_controls = _ControlSection(
            "Display conventions",
            self._control_group(
                "",
                [
                    ("Field unit", self.units),
                    ("Native-sample statistics", self.sampling),
                    ("Error normalization", self.normalization),
                ],
            ),
        )
        layout.addWidget(self.display_controls)
        self.scope = QLabel("Loading available field sources…")
        self.scope.setWordWrap(True)
        layout.addWidget(self.scope)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self._build_cloud_tab()
        self._build_slices_tab()
        self._build_distribution_tab()
        self._build_harmonics_tab()
        self._build_coil_tab()
        self._build_shielding_tab()
        self.status = QLabel("Analysis describes the saved field. Engineering acceptance remains in Review.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        footer = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.hide()
        layout.addWidget(self.progress)
        self.refresh = QPushButton("Evaluate slices")
        self.refresh.clicked.connect(self.refresh_analysis)
        self.cancel = QPushButton("Cancel calculation")
        self.cancel.clicked.connect(self._cancel)
        self.cancel.hide()
        self.export_data = QPushButton("Export data…")
        self.export_data.setToolTip("Export the captured analysis as JSON or native-sample CSV.")
        self.export_data.clicked.connect(self._choose_export)
        self.export_plot = QPushButton("Export plot…")
        self.export_plot.setToolTip("Save the selected page's plot as PNG.")
        self.export_plot.clicked.connect(self._choose_plot)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        for widget in [self.refresh, self.cancel]:
            footer.addWidget(widget)
        footer.addStretch()
        for widget in [self.export_data, self.export_plot, close]:
            footer.addWidget(widget)
        layout.addLayout(footer)
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(250)
        self.debounce.timeout.connect(self._automatic_analysis)
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
        for widget in [self.center, self.slice_span, self.offsets, self.harmonic_center, self.harmonic_radius]:
            widget.textEdited.connect(self._changed)
        for widget in [self.harmonic_center, self.harmonic_radius]:
            widget.textChanged.connect(self._control_summaries)
        for widget in [self.units, self.sampling, self.normalization, self.plane]:
            widget.currentIndexChanged.connect(self._control_summaries)
        for widget in [self.count, self.resolution, self.order]:
            widget.valueChanged.connect(self._control_summaries)
        self._control_summaries()
        self.tabs.currentChanged.connect(self._view_changed)
        self.profile_axis.currentIndexChanged.connect(lambda: self._draw_distribution() if self._current() else None)
        self.harmonic_source.currentIndexChanged.connect(lambda: self._draw_harmonics() if self._current() else None)
        self._enable(False)
        self.tabs.setCurrentIndex(self._VIEW_TABS[initial_view])
        self._view_changed()

    @staticmethod
    def _control_group(title, controls, columns=None):
        group = QGroupBox(title)
        grid = QGridLayout(group)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setHorizontalSpacing(12)
        columns = columns or len(controls)
        for index, (label, widget) in enumerate(controls):
            row, column = divmod(index, columns)
            caption = QLabel(label)
            caption.setWordWrap(True)
            grid.addWidget(caption, 2 * row, column)
            grid.addWidget(widget, 2 * row + 1, column)
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            if isinstance(widget, QComboBox):
                widget.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
                widget.setMinimumContentsLength(8 if label in {"Component", "Plane"} else 16)
            grid.setColumnStretch(column, 1)
        return group

    @classmethod
    def _validate_view(cls, view):
        if not isinstance(view, str) or view not in cls._VIEW_TABS:
            raise ValueError(
                "Field analysis view must be field, metrics, slices, errors, harmonics, coil, or shielding."
            )

    @staticmethod
    def _source_alias(source):
        return {
            "contour_field": "contour",
            "contour_residual": "contour",
            "routed_field": "routed",
            "routed_residual": "routed",
            "winding_validation_residual": "winding_validation",
        }.get(source, source)

    def focus_view(self, view, *, source=None):
        """Open a diagnostic page without resetting source, current, or grid settings.

        Requests made while the source catalog loads are applied to that catalog.
        An explicit source selects its recorded/original current variant; omitted
        sources preserve that choice. Error-facing pages select a signed residual
        when a target exists; shim results use Bz. Display and fit settings remain.
        """
        self._validate_view(view)
        if self._closed or self._close_pending:
            return False
        if view == "shielding" and not self.shield_source.count():
            return False
        unavailable = None
        if source is not None:
            source = self._source_alias(source)
            if not self._catalog_ready:
                self._initial_source = source
            elif source in self.datasets:
                original = source.endswith("_at_stream_current")
                self.source.setCurrentIndex(self.source.findData(source.removesuffix("_at_stream_current")))
                self.current_policy.setCurrentIndex(1 if original else 0)
            else:
                unavailable = str(source)
        self._requested_view = view
        self.tabs.setCurrentIndex(self._VIEW_TABS[view])
        if self.datasets and view not in {"harmonics", "coil", "shielding"}:
            dataset = self._dataset()
            metric = "field" if view == "field" or dataset.target_T is None else "signed_residual"
            if self._is_shim:
                self.component.setCurrentIndex(self.component.findData("Bz"))
            self.metric_selector.setCurrentIndex(self.metric_selector.findData(metric))
        elif view not in {"coil", "shielding"} and self._auto_start and not self._catalog_ready and self.worker is None:
            self.load_sources()
        if unavailable is not None:
            self.status.setText(
                f"Requested field source '{unavailable}' is unavailable. The selected saved source is retained."
            )
        return True

    def _control_summaries(self, *_):
        unit = self.units.currentText() if self.normalization.currentData() == "none" else "%"
        self.display_controls.set_summary(
            f"{unit} · {self.sampling.currentText()} · {self.normalization.currentText()}"
        )
        self.slice_controls.set_summary(
            f"{self.plane.currentText()} · {self.count.value()} slices · {self.resolution.value()} pixels / axis"
        )
        self.harmonic_controls.set_summary(
            f"Degree {self.order.value()} · center ({self.harmonic_center.text()}) m · R {self.harmonic_radius.text()} m"
        )

    def _view_changed(self, *_):
        if self._VIEW_TABS[self._requested_view] != self.tabs.currentIndex():
            self._requested_view = ("metrics", "slices", "errors", "harmonics", "coil", "shielding")[
                self.tabs.currentIndex()
            ]
        coil = self._requested_view == "coil"
        shielding = self._requested_view == "shielding"
        field = not (coil or shielding)
        for widget in [self.field_controls, self.display_controls, self.scope]:
            widget.setVisible(field)
        self.refresh.setVisible(self._requested_view == "slices")
        self.status.setVisible(not shielding)
        self.export_data.setVisible(not shielding)
        if shielding:
            self._draw_shielding()
        elif self._shield_worker_source is not None and self.worker is not None:
            self.worker.cancel.set()
        self.export_data.setText("Export stats…" if coil else "Export data…")
        self.export_data.setEnabled(coil or (self.worker is None and self._current()))
        self.export_plot.setEnabled(
            self._shield_plot_key is not None if shielding else field and self.worker is None and self._current()
        )
        if field and self._auto_start and not self._catalog_ready and self.worker is None:
            QTimer.singleShot(0, self, self.load_sources)
        if self.datasets and self.tabs.currentIndex() == self._VIEW_TABS["slices"]:
            self._grid_requested = True
            if self.state is None or self.state.slices is None or not self._current():
                self._pending = True
                self.debounce.start()
        elif field and self.datasets and not self._current():
            self._pending = True
            self.debounce.start()

    def _automatic_analysis(self):
        if self._requested_view in {"coil", "shielding"}:
            self._pending = False
            return
        self.refresh_analysis(sample_grid=self._grid_requested or self.tabs.currentIndex() == self._VIEW_TABS["slices"])

    def _plot(self, parent):
        try:
            return self._plot_factory(parent)
        except Exception as exc:
            return _UnavailablePlot(str(exc), parent)

    def _build_cloud_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        note = QLabel("Native sample metrics · every finite saved sample contributes to the selected convention.")
        note.setWordWrap(True)
        layout.addWidget(note)
        split = QSplitter()
        try:
            self.cloud = self._cloud_factory(split)
        except Exception as exc:
            self.cloud = _UnavailablePlot(str(exc), split)
        split.addWidget(self.cloud)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10, 0, 0, 0)
        right_layout.addWidget(QLabel("Field and residual metrics"))
        self.summary = QTableWidget(0, 2)
        self.summary.setHorizontalHeaderLabels(["Metric / convention", "Value"])
        self._style_table(self.summary)
        self.summary.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.summary.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        right_layout.addWidget(self.summary)
        self.samples = QTableWidget(0, 4)
        self.samples.setHorizontalHeaderLabels(["x [m]", "y [m]", "z [m]", "Map value"])
        self._style_table(self.samples)
        sample_note = QLabel("Largest absolute map values · up to 256 native samples")
        sample_note.setWordWrap(True)
        right_layout.addWidget(sample_note)
        right_layout.addWidget(self.samples)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        layout.addWidget(split)
        self.tabs.addTab(page, "Metrics / 3D")

    def _build_slices_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.plane, self.count, self.resolution = QComboBox(), QSpinBox(), QSpinBox()
        for plane in ["xy", "xz", "yz"]:
            self.plane.addItem(plane.upper(), plane)
        self.count.setRange(1, 9)
        self.count.setValue(5)
        self.resolution.setRange(16, 128)
        self.resolution.setValue(64)
        self.center, self.slice_span, self.offsets = QLineEdit("0 0 0"), QLineEdit("0.1 0.1"), QLineEdit("-0.04 0.04")
        self.slice_controls = _ControlSection(
            "Slice settings",
            self._control_group(
                "World coordinates",
                [
                    ("Center xyz [m]", self.center),
                    ("Full in-plane span [m]", self.slice_span),
                    ("Plane", self.plane),
                    ("Offsets from center [m]", self.offsets),
                    ("Slices", self.count),
                    ("Pixels / axis", self.resolution),
                ],
                columns=3,
            ),
        )
        layout.addWidget(self.slice_controls)
        self.center.setToolTip("World x y z, in metres.")
        self.slice_span.setToolTip(
            "Two full spans, in metres, along the selected plane's horizontal and vertical axes."
        )
        self.offsets.setToolTip("Minimum and maximum offsets from the world center along the plane normal, in metres.")
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
        self.tabs.addTab(page, "Slices")

    def _build_distribution_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        row = QHBoxLayout()
        row.addWidget(QLabel("Residual sample trend along"))
        self.profile_axis = QComboBox()
        self.profile_axis.addItems(["x", "y", "z"])
        row.addWidget(self.profile_axis)
        note = QLabel("Other coordinates vary; this is a sample trend, rather than a line scan.")
        note.setWordWrap(True)
        row.addWidget(note, 1)
        self.residual_slices = QPushButton("Residual slices")
        self.residual_slices.setToolTip(
            "Evaluate field − target on physical slices using the selected source and current."
        )
        self.residual_slices.setEnabled(False)
        self.residual_slices.clicked.connect(self._show_residual_slices)
        row.addWidget(self.residual_slices)
        layout.addLayout(row)
        self.error_plot = self._plot(page)
        layout.addWidget(self.error_plot, 1)
        self.tabs.addTab(page, "Errors")

    def _show_residual_slices(self):
        self.metric_selector.setCurrentIndex(self.metric_selector.findData("signed_residual"))
        self.tabs.setCurrentIndex(self._VIEW_TABS["slices"])

    def _build_harmonics_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.order, self.harmonic_center, self.harmonic_radius = QSpinBox(), QLineEdit("0 0 0"), QLineEdit("0.05")
        self.order.setRange(0, 12)
        self.order.setValue(4)
        self.fit = QPushButton("Fit field / target / residual")
        self.fit.clicked.connect(self.fit_harmonics)
        self.harmonic_source = QComboBox()
        for source in ["field", "target", "residual"]:
            self.harmonic_source.addItem(source.capitalize(), source)
        self.harmonic_controls = _ControlSection(
            "Expansion settings",
            self._control_group(
                "Scalar solid harmonics",
                [
                    ("Maximum degree", self.order),
                    ("Expansion center xyz [m]", self.harmonic_center),
                    ("Reference radius [m]", self.harmonic_radius),
                ],
            ),
        )
        layout.addWidget(self.harmonic_controls)
        row = QHBoxLayout()
        row.addWidget(QLabel("Display coefficients"))
        row.addWidget(self.harmonic_source)
        row.addStretch()
        row.addWidget(self.fit)
        layout.addLayout(row)
        self.harmonic_note = QLabel(
            "Scalar solid harmonics require a source-free region. The fit does not verify this assumption or engineering acceptance."
        )
        self.harmonic_note.setWordWrap(True)
        layout.addWidget(self.harmonic_note)
        self.harmonic_plot = self._plot(page)
        layout.addWidget(self.harmonic_plot, 2)
        self.coefficients = QTableWidget(0, 5)
        self.coefficients.setHorizontalHeaderLabels(
            ["Degree l", "Order m", "Real harmonic", "Basis coefficient [µT]", "Sample RMS contribution [µT]"]
        )
        self._style_table(self.coefficients)
        self.coefficient_table = _ControlSection("Coefficient table", self.coefficients)
        self.coefficients.setMaximumHeight(180)
        layout.addWidget(self.coefficient_table)
        self.tabs.addTab(page, "Harmonics")

    def _build_coil_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self._coil_report = build_coil_statistics(self.record)
        self.coil_note = QLabel(self._coil_report.message)
        self.coil_note.setWordWrap(True)
        layout.addWidget(self.coil_note)
        density_row = QHBoxLayout()
        density_row.addWidget(QLabel("Material density [kg/m³]"))
        self.material_density = QDoubleSpinBox()
        self.material_density.setDecimals(2)
        self.material_density.setRange(0, 1e6)
        self.material_density.setSpecialValueText("Unspecified")
        self.material_density.setToolTip(
            "Optional assumption for nominal conductor mass; saved material and design settings remain unchanged."
        )
        density_row.addWidget(self.material_density)
        self.mass_estimate_note = QLabel()
        self.mass_estimate_note.setWordWrap(True)
        density_row.addWidget(self.mass_estimate_note, 1)
        layout.addLayout(density_row)
        self.coil_stats = QTreeWidget()
        self.coil_stats.setColumnCount(5)
        self.coil_stats.setHeaderLabels(["Property / body", "Value", "Unit", "Scope", "Status"])
        self.coil_stats.setAlternatingRowColors(True)
        self.coil_stats.setUniformRowHeights(True)
        self.coil_stats.setColumnWidth(0, 285)
        self.coil_stats.setColumnWidth(1, 100)
        self.coil_stats.setColumnWidth(2, 65)
        self.coil_stats.setColumnWidth(3, 300)
        self.coil_stats.header().setStretchLastSection(False)
        self.coil_stats.header().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.coil_stats.header().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.coil_stats, 1)
        self.material_density.valueChanged.connect(self._update_coil_stats)
        self._update_coil_stats()
        self.tabs.addTab(page, "Coil stats")

    def _build_shielding_tab(self):
        page = QWidget()
        self.shield_layout = QVBoxLayout(page)
        self.shield_source, self.shield_component, self.shield_scale = QComboBox(), QComboBox(), QComboBox()
        for source in shielding_sources(self.record):
            self.shield_source.addItem(SHIELDING_SOURCES[source], source)
        routed = self.shield_source.findData("routed")
        if routed >= 0:
            self.shield_source.setCurrentIndex(routed)
        for component, label in SHIELDING_COMPONENTS.items():
            self.shield_component.addItem(label, component)
        self.shield_scale.addItem("Compare fields on one scale", False)
        self.shield_scale.addItem("Combined field detail", True)
        controls = QHBoxLayout()
        for label, control in (
            ("Field source", self.shield_source),
            ("Component", self.shield_component),
            ("View", self.shield_scale),
        ):
            controls.addWidget(QLabel(label))
            controls.addWidget(control)
            control.currentIndexChanged.connect(self._draw_shielding)
        controls.addStretch()
        self.shield_compute = QPushButton("Compute interior samples")
        self.shield_compute.clicked.connect(self._compute_shield_interior)
        controls.addWidget(self.shield_compute)
        self.shield_layout.addLayout(controls)
        self.shield_note = QLabel()
        self.shield_note.setWordWrap(True)
        self.shield_layout.addWidget(self.shield_note)
        self.shield_plot, self._shield_plot_key = None, None
        self._shield_interiors = {}
        self._shield_worker_source = None
        index = self.tabs.addTab(page, "Shielding surface")
        self.tabs.setTabVisible(index, bool(self.shield_source.count()))

    def _draw_shielding(self):
        source, component = self.shield_source.currentData(), self.shield_component.currentData()
        net_detail = self.shield_scale.currentData()
        key = source, component, net_detail
        needs_compute = source is not None and source.endswith("_interior") and source not in self._shield_interiors
        self.shield_compute.setVisible(needs_compute)
        self.shield_compute.setEnabled(self.worker is None)
        if self._shield_worker_source is not None and self.worker is not None and self._shield_worker_source != source:
            self.worker.cancel.set()
        if source is None or self._shield_plot_key == key:
            return
        self._shield_plot_key = None
        self.export_plot.setEnabled(False)
        if self.shield_plot is None:
            self.shield_plot = self._plot(self.tabs.widget(self._VIEW_TABS["shielding"]))
            self.shield_layout.addWidget(self.shield_plot, 1)
        figure = self.shield_plot.figure
        if figure is None:
            return
        if needs_compute:
            figure.clear()
            self.shield_plot.draw()
            self.shield_note.setText(
                "Computing the field at triangle-interior points…"
                if source == self._shield_worker_source
                else "Evaluate the saved current distribution at three interior points per suppression triangle."
            )
            return
        try:
            data = (
                self._shield_interiors[source]
                if source in self._shield_interiors
                else shielding_surface_fields(self.record, source)
            )
            self.shield_component.model().item(self.shield_component.findData("normal")).setEnabled(
                data.normals is not None
            )
            if component == "normal" and data.normals is None:
                self.shield_component.setCurrentIndex(self.shield_component.findData("magnitude"))
                return
            title = f"{read_run_metadata(self.record).get('name', 'Saved run')} · {SHIELDING_SOURCES[source]}"
            note = draw_shielding_surface(
                figure,
                data,
                component,
                title=title,
                net_detail=net_detail,
                limits=shielding_limits(read_run_metadata(self.record)),
                openings_m=record_openings(self.record),
            )
            self.shield_note.setText(note)
            self._shield_plot_key = key
            self.export_plot.setEnabled(True)
        except (ValueError, KeyError, TypeError) as exc:
            figure.clear()
            self.shield_note.setText(f"Shielding field map unavailable. {exc}")
        self.shield_plot.draw()

    def _compute_shield_interior(self):
        source = self.shield_source.currentData()
        if self.worker is not None or not source or not source.endswith("_interior"):
            return
        self._shield_worker_source = source
        self.shield_compute.setEnabled(False)
        self.shield_note.setText("Computing the field at triangle-interior points…")

        def operation(task):
            task.report(None, "Calculate triangle-interior fields")
            fields = shielding_surface_fields(self.record, source, cancel_check=task.cancel.is_set)
            return ("shielding", (source, fields))

        self._start(operation)

    def _update_coil_stats(self, *_):
        self._coil_estimates = mass_estimates(self._coil_report, self.material_density.value())
        self.coil_note.setText(
            self._coil_estimates.message if self.material_density.value() > 0 else self._coil_report.message
        )
        if self.material_density.value() > 0 and self._coil_estimates.mass_unavailable_reason:
            self.coil_note.setText(self.coil_note.text() + " " + self._coil_estimates.mass_unavailable_reason)
        mass = self._coil_estimates.row("coil:mass")
        if mass is None or mass.status != "Assumed density estimate":
            mass = self._coil_estimates.row("estimate:coil:mass")
        if self.material_density.value() == 0:
            self.mass_estimate_note.setText(
                "Enter density to estimate bare conductor mass; insulation, feeds and former are excluded."
            )
        elif mass is None or mass.value is None:
            self.mass_estimate_note.setText(
                "Mass estimate unavailable: no finite nominal conducting volume is recorded."
            )
        else:
            scale, unit = (1000, "g") if mass.value < 1 else (1, "kg")
            self.mass_estimate_note.setText(
                f"Estimated bare conductor mass: {mass.value * scale:.5g} {unit} · nominal conducting volume × entered density; insulation, feeds and former excluded."
            )
        self.coil_stats.clear()
        groups = {}
        rows = list(self._coil_report.rows)
        estimates = self._mass_estimate_rows()
        rows.extend(estimates)
        for row in rows:
            category = "Density estimates" if row in estimates else row.category
            group = groups.get(category)
            if group is None:
                group = QTreeWidgetItem(self.coil_stats, [category])
                font = group.font(0)
                font.setBold(True)
                group.setFont(0, font)
                group.setExpanded(True)
                groups[category] = group
            quantity = f"{row.body_id} · {row.quantity}" if row.body_id else row.quantity
            value, unit = ("Unavailable", row.unit) if row.value is None else display_quantity(row.value, row.unit)
            item = QTreeWidgetItem(group, [quantity, value, unit, row.scope, row.status])
            item.setData(0, Qt.ItemDataRole.UserRole, row.key)
            for column in range(5):
                item.setToolTip(column, f"{quantity}\n{row.scope}\nSaved source: {row.source_key or '—'}\n{row.status}")

    def _mass_estimate_rows(self):
        return tuple(
            replace(
                row, key="estimate:coil:mass", category="Density estimates", quantity="Nominal conductor mass estimate"
            )
            if row.key == "coil:mass"
            else row
            for row in self._coil_estimates.rows
            if row.key.startswith("estimate:") or row.status == "Assumed density estimate"
        )

    @staticmethod
    def _style_table(table):
        table.setAlternatingRowColors(True)
        table.verticalHeader().hide()
        table.horizontalHeader().setStretchLastSection(True)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setShowGrid(False)

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
        coil = self.tabs.currentIndex() == self._VIEW_TABS["coil"]
        self.export_data.setEnabled(coil or (enabled and self._current()))
        self.export_plot.setEnabled(not coil and enabled and self._current())
        if self._requested_view == "shielding":
            self.export_plot.setEnabled(self._shield_plot_key is not None)

    def load_sources(self):
        if (
            self._requested_view not in {"coil", "shielding"}
            and self.worker is None
            and not self._close_pending
            and not self._closed
        ):

            def operation(task):
                task.report(None, "Read verified field source arrays")
                sources = self._slice_cache.catalog(self.record, progress=task.report, cancel_check=task.check)
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
            "Original saved current"
            if dataset.provenance.get("current_policy") == "original"
            else "Recorded evaluation",
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
        self.residual_slices.setEnabled(dataset.target_T is not None)
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
            self.slice_span.setText(
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
            self._control_summaries()
        self._changed()

    def _plane_changed(self, *_):
        self._fit_source()

    def selection(self):
        center = _numbers(self.center.text(), 3, "Slice center")
        size = _numbers(self.slice_span.text(), 2, "Slice span")
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

    def refresh_analysis(self, *, sample_grid=True):
        if self._close_pending or self._closed or not self.datasets:
            return
        self._grid_requested |= sample_grid
        if self.worker is not None:
            self._pending = True
            return
        self._pending = False
        try:
            selection, dataset = self.selection(), self._dataset()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        sample_grid = self._grid_requested
        self._grid_requested = False
        previous = self.state

        def slice_controls(item):
            return (item.source_key, item.plane, item.center_m, item.size_m, item.offsets_m, item.resolution)

        reused = (
            previous.slices
            if previous is not None and slice_controls(previous.selection) == slice_controls(selection)
            else None
        )

        def operation(task):
            if reused is not None:
                self._slice_cache.request(self.record, cancel_check=task.check)
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
            task.report(
                10,
                "Reuse captured physical slice grid"
                if reused is not None
                else "Evaluate slice grid"
                if sample_grid
                else "Native sample metrics ready",
            )
            values = []

            def capture_slices(collection):
                for piece in collection.slices[len(values) :] if collection is not None else ():
                    task.check()
                    values.append(self._slice_values(dataset, mapped, piece, selection))
                return AnalysisState(
                    selection,
                    dataset,
                    mapped,
                    collection,
                    tuple(values),
                    error,
                    paired_peak=self._paired_peak(dataset, mapped, collection, selection),
                )

            def show_partial(collection):
                if len(collection.slices) < len(selection.offsets_m):
                    task.publish(capture_slices(collection))

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
                    cache=self._slice_cache,
                    on_slice=show_partial,
                )
                if sample_grid
                else None
            )
            captured = capture_slices(slices)
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
            return ("analysis", replace(captured, harmonics=harmonics))

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

    @staticmethod
    def _paired_peak(dataset, mapped, slices, selection):
        if selection.metric not in {"field", "target"} or dataset.target_T is None:
            return 0.0
        is_field = selection.metric == "field"
        vectors = [dataset.target_T if is_field else dataset.field_T]
        for piece in slices.slices if slices is not None else ():
            if is_field:
                if piece.target_T is not None:
                    vectors.append(piece.target_T[piece.available_mask & piece.target_valid_mask])
            else:
                vectors.append(piece.field_T[piece.available_mask])
        component = "BdotB0" if selection.component == "Bparallel" else selection.component
        b0 = dataset.provenance.get("b0_direction", (0, 0, 1))
        projections = (
            np.hypot.reduce(values, axis=1)
            if component == "magnitude"
            else field_component_values(values, component, b0_direction=b0)
            for values in vectors
        )
        peak = max(float(np.max(np.abs(values), initial=0.0)) for values in projections)
        if mapped.normalization_scale_T is not None:
            peak = peak / mapped.normalization_scale_T * 100
        return peak

    def fit_harmonics(self):
        if self.worker is not None or not self._current():
            self.status.setText("Evaluate the current field views before fitting harmonics.")
            return
        state = self.state
        if state.selection.component == "magnitude":
            self.status.setText("Select Bx, By, Bz or Bparallel for a source-free scalar harmonic expansion.")
            return
        self.focus_view("harmonics")

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
        worker.signals.partial.connect(lambda state: self._publish_slice_preview(worker, state))
        worker.signals.finished.connect(lambda result, error: self._finished(worker, result, error))
        self.progress.setRange(0, 0)
        self.progress.show()
        self.cancel.show()
        self.refresh.setEnabled(False)
        self.fit.setEnabled(False)
        if self._requested_view not in {"coil", "shielding"}:
            self.export_plot.setEnabled(False)
            self.export_data.setEnabled(False)
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

    def _publish_slice_preview(self, worker, state):
        if (
            worker is not self.worker
            or worker.generation != self._generation
            or worker.cancel.is_set()
            or self.tabs.currentIndex() != self._VIEW_TABS["slices"]
        ):
            return
        try:
            self._draw_slices(state, partial=True)
        except (RuntimeError, TypeError, ValueError) as exc:
            self.status.setText(f"Slice preview rendering failed: {exc}")
        else:
            self._showing_partial = True

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
        if self._showing_partial:
            self._showing_partial = False
            if error or stale:
                self._draw_slices()
        interior_source, self._shield_worker_source = self._shield_worker_source, None
        if error and not stale:
            self.status.setText(
                "Field analysis cancelled." if error == "Cancelled" else "Field analysis failed: " + error
            )
            if interior_source is not None and interior_source == self.shield_source.currentData():
                self.shield_note.setText(self.status.text())
        elif result is not None and not stale:
            kind, value = result
            if kind == "catalog":
                self._catalog_loaded(value)
            elif kind == "shielding":
                source, data = value
                self._shield_interiors[source] = data
                if self._requested_view == "shielding" and source == self.shield_source.currentData():
                    self._draw_shielding()
            elif kind in {"analysis", "harmonics"}:
                self.state = value
                self._progress(None, "Build captured field plots and tables")
                try:
                    self._render()
                except Exception as exc:
                    self.status.setText("Field plot publication failed: " + str(exc))
            else:
                self.status.setText("Saved captured field analysis: " + str(value))
        self.progress.hide()
        self.cancel.hide()
        self.busyChanged.emit(False)
        self._enable(bool(self.datasets))
        self.shield_compute.setEnabled(True)
        if self.datasets:
            self.current_policy.setEnabled(self.source.currentData() + "_at_stream_current" in self.datasets)
            self.sampling.setEnabled(bool(self._dataset().weights is not None and np.any(self._dataset().weights > 0)))
            self.units.setEnabled(self.normalization.currentData() == "none")
        if self._pending:
            QTimer.singleShot(0, self, self._automatic_analysis)
        if interior_source is not None and not self._catalog_ready and self._auto_start:
            QTimer.singleShot(0, self, self.load_sources)

    def _catalog_loaded(self, datasets):
        self._catalog_ready = True
        self.datasets = {dataset.key: dataset for dataset in datasets}
        self.source.blockSignals(True)
        for dataset in datasets:
            if not dataset.key.endswith("_at_stream_current"):
                label = dataset.label.removesuffix(" · recorded current").removesuffix(" · original current")
                self.source.addItem(label, dataset.key)
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
            self.status.setText(
                "No finite field maps are recorded. Saved coil statistics remain available in Coil stats."
            )
            return
        component = self.component.findData(read_run_metadata(self.record).get("target_field_component", "Bz"))
        if component >= 0:
            self.component.setCurrentIndex(component)
        dataset = self._dataset()
        if dataset.target_T is not None:
            self.metric_selector.setCurrentIndex(self.metric_selector.findData("signed_residual"))
        self._source_changed()
        original = self.source.currentData() + "_at_stream_current"
        if self._initial_source in self.datasets and self._initial_source.endswith("_at_stream_current"):
            self.current_policy.setCurrentIndex(1)
        elif self._initial_source is None and original in self.datasets:
            self.current_policy.setCurrentIndex(1)
        self.focus_view(self._requested_view, source=self._initial_source)

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

    def _display_limits(self, state):
        factor, _ = self._display(state)
        peaks = (
            float(np.max(np.abs(values[np.isfinite(values)]), initial=0.0))
            for values in (state.mapped.values, *state.slice_values)
        )
        return paired_color_limits((max(peaks) * factor,), (state.paired_peak * factor,), signed=self._signed(state))

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
                item.setToolTip(text)
                table.setItem(row, col, item)
        table.resizeColumnsToContents()

    def _render(self):
        state = self.state
        factor, unit = self._display(state)
        values = np.asarray(state.mapped.values) * factor
        limits = self._display_limits(state)
        self._color_limits = limits
        provenance = field_provenance(
            state.dataset.key,
            read_run_metadata(self.record),
            len(state.dataset.points_m),
            evidence=state.dataset.provenance,
        )
        definition = {
            "signed_residual": "Field − target for the selected component",
            "vector_error": "‖Bfield − Btarget‖ across all three components",
        }.get(state.selection.metric, "")
        if state.selection.metric in {"field", "target"} and state.dataset.target_T is not None:
            definition = "Field and target share the same color scale"
        self.scope.setText(
            f"{provenance}\n{state.mapped.label} [{unit}] · {state.mapped.sampling_convention}"
            + (f" · {definition}" if definition else "")
            + (" · 3D display capped at 50,000 points; statistics use all samples" if len(values) > 50_000 else "")
            + (
                f"\nSaved sampling includes {state.dataset.provenance['suppression_sample_count']} suppression points. Equal statistics include them; weighted fits use positive weights."
                if state.dataset.provenance.get("suppression_sample_count", 0)
                else ""
            )
        )
        self.scope.setToolTip(
            "Owned ROI geometry is shown when recorded; no enclosing volume is inferred from samples."
        )
        summary = []
        for key, value in state.mapped.summary.items():
            if key in {"rms", "peak", "std", "mean", "p95"}:
                summary.append((f"{key.upper()} [{unit}]", value * factor))
            elif key == "relative_error_rms":
                summary.append(("Relative RMS error [%]", None if value is None else value * 100))
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

    def _draw_slices(self, state=None, *, partial=False):
        if self.slice_plot.figure is None:
            return
        state = self.state if state is None else state
        figure = self.slice_plot.figure
        figure.clear()
        figure.set_layout_engine("compressed")
        if state is None or state.slices is None:
            self.slice_note.setText(
                "Open Slices or select Evaluate slices to sample the saved field. Native-sample metrics are available separately."
            )
            self.slice_plot.draw()
            return
        factor, unit = self._display(state)
        limits = self._display_limits(state)
        count = len(state.slice_values)
        total = len(state.selection.offsets_m) if partial else count
        columns = min(3, total)
        rows = (total + columns - 1) // columns
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
                vmin=limits[0],
                vmax=limits[1],
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
            # Context outlines must not silently enlarge the requested sampling
            # window and shrink the actual ROI into a few pixels.
            axis.set_xlim(float(points[:, :, horizontal].min()), float(points[:, :, horizontal].max()))
            axis.set_ylim(float(points[:, :, vertical].min()), float(points[:, :, vertical].max()))
            axis.set_aspect("equal")
            visible.append(axis)
        if image is not None:
            # Colorbar's default parent anchor pushes equal-aspect slices to
            # the right edge of wide windows. Keep maps centered in each cell.
            figure.colorbar(image, ax=visible, label=f"{state.mapped.label} [{unit}]", shrink=0.82, panchor=False)
        descriptions = {
            "direct_sheet": "saved stream function, directly evaluated",
            "direct_wire": "saved winding, directly evaluated",
            "display_interpolation": "interpolated saved samples",
            "owned_analytic": "authored expression, original scale",
            "owned_geometry": "authored target boundary",
            "sampling_hull": "sample coverage",
            "unavailable_owned_enclosure": "target boundary unavailable",
            "unavailable": "unavailable",
        }
        self.slice_note.setText(
            (
                f"Preview {count}/{total} slices · {state.dataset.label} · scale updates as slices finish.\n"
                if partial
                else ""
            )
            + f"Field: {descriptions.get(state.slices.representation, state.slices.representation.replace('_', ' '))} · "
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
            self.coefficient_table.set_summary("Fit harmonics to populate")
            self.harmonic_note.setText("Fit a scalar component on one common basis for field, target and residual.")
            self.harmonic_note.setToolTip("")
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
        ranked = result["coefficient_rows"][source]
        rows = [
            (row["degree"], row["order"], row["term_label"], row["coefficient"] * 1e6, row["contribution_rms"] * 1e6)
            for row in ranked
        ]
        self._set_table(self.coefficients, rows)
        self.coefficient_table.set_summary(f"{len(rows)} terms · signed coefficients and sample RMS [µT]")
        condition = result.get("condition_number", result.get("condition"))
        diagnostics = result.get("diagnostics", ())
        condition_text = (
            f"{condition:.4g}" if isinstance(condition, (float, np.floating)) else str(condition or "unavailable")
        )
        relative = f" ({100 * fit.residual_rms / fit.field_rms:.3g}% of {source} RMS)" if fit.field_rms > 0 else ""
        center = ", ".join(f"{1e3 * value:.6g}" for value in fit.center)
        fit_quality = (
            f"Rank {fit.rank}/{len(fit.terms)} · {fit.sample_count:,} samples · condition {condition_text} · "
            f"fit residual RMS {fit.residual_rms * 1e6:.5g} µT{relative}"
        )
        reference = (
            f"{result['component']} expansion · normalized real basis · center ({center}) mm · "
            f"reference radius {fit.radius * 1e3:.6g} mm"
        )
        caveat = str(result.get("sourcefree_caveat", "Source-free validity is not established by this fit."))
        self.harmonic_note.setToolTip("\n".join((fit_quality, reference, caveat, *diagnostics)))
        self.harmonic_note.setText(
            "\n".join(
                (
                    fit_quality,
                    reference,
                    "Term contributions are RMS on the selected samples. The fit does not establish source-free validity.",
                    *diagnostics,
                )
            )
        )
        if figure is not None:
            axis = figure.subplots()
            shown = ranked[:12]
            ids = np.arange(len(shown))
            axis.bar(ids, [row["contribution_rms"] * 1e6 for row in shown], color="#23698c")
            axis.set_xticks(ids)
            axis.set_xticklabels([f"{row['degree']},{row['order']}" for row in shown])
            axis.set_xlabel("Real solid harmonic (degree, order)")
            axis.set_ylabel(f"{source.capitalize()} sample RMS contribution [µT]")
            if len(shown) < len(ranked):
                axis.set_title(f"Largest {len(shown)} of {len(ranked)} terms by sample RMS")
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
        if self.tabs.currentIndex() == self._VIEW_TABS["coil"]:
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Export recorded coil statistics and explicit mass assumptions",
                "coil_statistics.csv",
                "Coil statistics CSV (*.csv);;Coil statistics JSON (*.json)",
            )
            if path:
                try:
                    self.export_coil_stats(path)
                    self.status.setText("Saved recorded coil statistics: " + str(path))
                except Exception as exc:
                    self.status.setText("Coil statistics export failed: " + str(exc))
            return
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

    def export_coil_stats(self, path):
        """Export saved scalars separately from the user's optional density estimate."""
        path = Path(path)
        estimates = self._mass_estimate_rows()
        if path.suffix.lower() == ".csv":
            with atomic_text_writer(path) as output:
                writer = csv.writer(output)
                columns = ["key", "category", "body_id", "quantity", "value", "unit", "scope", "status", "source_key"]
                writer.writerow(columns)
                for row in (*self._coil_report.rows, *estimates):
                    writer.writerow([getattr(row, column) for column in columns])
        elif path.suffix.lower() == ".json":
            payload = {
                "software": software_attribution(),
                "run_request_sha256": read_run_metadata(self.record).get("request_sha256"),
                "recorded_statistics": self._coil_report,
                "assumed_density_kg_per_m3": self.material_density.value() or None,
                "assumed_density_estimates": estimates,
                "acceptance_claim": False,
            }
            with atomic_text_writer(path) as output:
                json.dump(_portable(payload), output, allow_nan=False, indent=2)
                output.write("\n")
        else:
            raise ValueError("Choose a .json or .csv coil-statistics export.")

    def _choose_plot(self):
        if self._requested_view == "shielding" and self._shield_plot_key is not None:
            plot = self.shield_plot
        elif self.worker is not None or not self._current() or self._requested_view in {"coil", "shielding"}:
            return
        else:
            plot = [self.cloud, self.slice_plot, self.error_plot, self.harmonic_plot][self.tabs.currentIndex()]
        if not hasattr(plot, "save_png"):
            self.status.setText(
                "The current graphics view is unavailable; export numerical analysis or a different plot."
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export current field plot",
            "shielding_surface.png" if self._requested_view == "shielding" else "field_analysis.png",
            "PNG image (*.png)",
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
