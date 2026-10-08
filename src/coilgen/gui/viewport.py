"""Scientific rendering adapter. Camera and display edits never change physics."""

from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import json

import numpy as np
from PySide6.QtCore import QEvent, QObject, QRunnable, QSignalBlocker, QSize, QThreadPool, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSlider,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from .icons import INSPECTION_ICONS, icon
from .dataset_capabilities import (
    DatasetCapability,
    FIELD_KEYS,
    WIRE_FIELDS,
    dataset_capabilities,
    points_available,
    resolve_dataset,
)
from ..persistence import json_value


def _preview_identities(config, values=None):
    """Compare model intent, retaining owned asset IDs across project revisions."""
    values = config.as_dict() if values is None else values
    support_fields = {
        "coil_mesh_file",
        "geometry_source_path",
        "imported_mesh_length_scale_to_m",
        "surface_is_cylinder_flag",
        "generated_mesh_rotation_axis",
        "generated_mesh_rotation_angle",
    } | {name for name in values if name.endswith("mesh_parameter_list")}
    target_fields = {
        "field_shape_function",
        "target_gradient_strength",
        "target_gradient_normalization",
        "target_gradient_axis",
        "target_field_component",
        "design_context",
        "target_field_units",
        "target_field_definition_file",
        "target_field_definition_field_name",
        "target_mesh_file",
        "target_region_geometry",
        "target_region_radius",
        "target_region_resolution",
        "use_only_target_mesh_verts",
        "set_roi_into_mesh_center",
        "secondary_target_mesh_file",
        "secondary_target_weight",
        "mesh_topology_suppression_separation",
        "mesh_topology_suppression_source",
    } | {
        name
        for name in values
        if name.startswith("create_")
        and name.endswith("field_suppression_surface")
        or name.endswith("area_suppression_parameters")
    }

    def identity(names):
        return json.dumps(json_value({name: values[name] for name in names}), sort_keys=True, separators=(",", ":"))

    return identity(support_fields), identity(target_fields)


def _has_target(config):
    return any(
        str(value).strip().lower() not in {"", "none"}
        for value in (config.target_field_definition_file, config.field_shape_function)
    )


def _signed_residual_limits(values):
    """Center a signed display scale on zero without modifying its samples."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None
    extent = float(np.max(np.abs(finite))) or 1e-12
    return -extent, extent


def _preview_data(config, root):
    """Extract backend numpy data only; VTK/Qt rendering stays on the UI thread."""
    from ..geometry import load_mesh_from_config, load_optional_target_mesh, load_secondary_target_mesh
    from ..target import define_target_field
    from ..application.diagnostics import preview_diagnostics

    if str(config.coil_mesh_file).strip().lower() in {"", "none"}:
        message = "Choose a support surface to preview the model."
        return None, {}, message, preview_diagnostics(config, None, message=message)
    mesh = load_mesh_from_config(config, root)
    arrays, message, target = {}, "", None
    imported_target = str(config.target_field_definition_file).strip().lower() not in {"", "none"}
    if not imported_target and str(config.field_shape_function).strip().lower() in {"", "none"}:
        message = "Choose a target field to preview it on this support."
    elif config.target_region_resolution > 40 and config.target_field_definition_file == "none":
        message = "Target preview omitted above resolution 40; computations retain the full grid."
    else:
        try:
            target_mesh = load_optional_target_mesh(config.target_mesh_file, config, root)
            secondary = load_secondary_target_mesh(config, root, coil_meshes=[mesh])
            target = define_target_field([mesh], target_mesh, secondary, config)
            arrays = {
                "target_points_m": target.coords,
                "target_field_T": target.b,
                "target_suppressed": target.suppressed,
            }
        except Exception as exc:
            # A failed target should not erase a valid current-carrying support
            # or substitute samples from the previous model or saved result.
            message = f"Target preview unavailable: {type(exc).__name__}: {exc}"
    return mesh, arrays, message, preview_diagnostics(config, mesh, target, message=message)


class _PreviewSignals(QObject):
    finished = Signal(int, object)


class _PreviewTask(QRunnable):
    def __init__(self, generation, config, root, owner):
        super().__init__()
        self.signals = _PreviewSignals()
        self.generation, self.config, self.root, self.owner = generation, config, root, owner

    def run(self):
        try:
            result = _preview_data(self.config, self.root)
        except Exception as exc:
            message = f"Model preview unavailable: {type(exc).__name__}: {exc}"
            result = (
                None,
                {},
                message,
                {"available": False, "message": message, "coordinate_frame": "world", "target": {"available": False}},
            )
        self.signals.finished.emit(self.generation, result)


class ScientificViewport(QWidget):
    selected = Signal(str)
    error = Signal(str)
    preflight_ready = Signal(dict)
    context_changed = Signal()

    def __init__(self, parent=None, *, synchronous_preview=False):
        super().__init__(parent)
        import pyvista as pv
        from pyvistaqt import QtInteractor

        self.pv = pv
        self._arrays = {}
        self._metadata = {}
        self._stage_arrays = {}
        self._stage_metadata = {}
        self._live_stage_preview = False
        self._located_issue = None
        self._before_issue_view = None
        self._colour_limits = None
        self._comparison = None
        self._preview_arrays = {}
        self._preview_geometry = {}
        self._capabilities = {}
        self._comparison_peer = None
        self._last_valid_dataset = None
        self._last_wire_current = 0
        self._stage_execution_status = "running"
        self._verification_error = ""
        self._source_notice = ""
        self._support = None
        self._preview_error = ""
        self._first_view = True
        self._closed = False
        self._actors = {}
        self._dataset_mesh = None
        self._dataset_actor_key = None
        self._synchronous_preview = synchronous_preview
        self._preview_generation = 0
        self._preview_worker = None
        self._pending_preview = None
        self._support_identity = None
        self._target_identity = None
        self._model_owner = None
        self._model_complete = False
        self._pick_press = None
        self._pick_dragged = False
        self._pending_pick = None
        self._navigation_camera = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.plot_selection_bar = QFrame()
        self.plot_selection_bar.setObjectName("viewport_plot_bar")
        self.plot_selection_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.plot_selection_bar.setStyleSheet(
            "QFrame#viewport_plot_bar { background: palette(alternate-base); "
            "border: 1px solid palette(mid); border-left: 3px solid palette(highlight); border-radius: 4px; }"
            "QLabel#viewport_plot_label { font-weight: 600; }"
            "QComboBox#viewport_dataset { background: palette(base); color: palette(text); "
            "font-weight: 600; min-height: 30px; padding: 3px 6px; "
            "border: 1px solid palette(mid); border-radius: 3px; }"
            "QComboBox#viewport_dataset:focus { border-color: palette(highlight); }"
        )
        plot_layout = QVBoxLayout(self.plot_selection_bar)
        plot_layout.setContentsMargins(8, 6, 8, 6)
        plot_layout.setSpacing(6)
        self.inspection_task = QComboBox(self)
        self.inspection_task.setObjectName("viewport_inspection_task")
        self.inspection_task.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.inspection_task.setMinimumContentsLength(8)
        self.inspection_task.setMaximumWidth(180)
        self.inspection_task.setAccessibleName("Scientific inspection task")
        self.inspection_task.addItem("Choose a task…", None)
        self.inspection_button = QToolButton(self)
        self.inspection_button.setObjectName("viewport_inspection_button")
        self.inspection_button.setText("Inspect…")
        self.inspection_button.setIcon(icon("independent-validation"))
        self.inspection_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.inspection_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.inspection_button.setAccessibleName("Inspect available scientific evidence")
        self.inspection_button.setToolTip(
            "Choose a design question to display available evidence. Plot selects an exact dataset."
        )
        self.inspection_menu = QMenu(self.inspection_button)
        self.inspection_menu.addSection("Inspect available evidence")
        self.inspection_button.setMenu(self.inspection_menu)
        self.inspection_actions = {}
        self.inspection_buttons = {}
        for label, key, short_label in (
            ("Target and ROI", "target", "Target"),
            ("Field accuracy", "field", "Field"),
            ("Wire fit", "wire", "Wire"),
            ("Connections", "connections", "Connections"),
        ):
            symbol = INSPECTION_ICONS[key]
            self.inspection_task.addItem(icon(symbol), label, key)
            button = QToolButton(self)
            button.setObjectName(f"inspect_{key}")
            button.setText(short_label)
            button.setIcon(icon(symbol))
            button.setIconSize(QSize(20, 20))
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            button.setCheckable(True)
            button.setAccessibleName(label)
            button.setToolTip(label + ": inspect available data; the note identifies its scientific scope.")
            button.clicked.connect(lambda _checked=False, selected=key: self.inspect_task(selected))
            self.inspection_buttons[key] = button
            # Keep existing programmatic shortcuts on the authoritative task model.
            button.hide()
            action = self.inspection_menu.addAction(icon(symbol), label)
            action.setCheckable(True)
            action.setToolTip(button.toolTip())
            action.triggered.connect(lambda _checked=False, selected=key: self.inspect_task(selected))
            self.inspection_actions[key] = action
        self.inspection_task.activated.connect(lambda index: self.inspect_task(self.inspection_task.itemData(index)))
        self.inspection_task.currentIndexChanged.connect(self._sync_inspection_buttons)
        # Preserve the combo contract for saved view state and programmatic users.
        self.inspection_task.hide()
        self.inspection_note = QLabel()
        self.inspection_note.setObjectName("viewport_inspection_note")
        self.inspection_note.setTextFormat(Qt.TextFormat.PlainText)
        self.inspection_note.setWordWrap(True)
        self.inspection_note.hide()
        toolbar = QHBoxLayout()
        plot_label = QLabel("&Plot")
        plot_label.setObjectName("viewport_plot_label")
        toolbar.addWidget(plot_label)
        self.dataset = QComboBox()
        self.dataset.setObjectName("viewport_dataset")
        self.dataset.setAccessibleName("Displayed scientific dataset")
        self.dataset.setToolTip(
            "Choose current-model previews, saved conductor geometry, field maps, or fabrication-review evidence."
        )
        self.dataset.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.dataset.setMinimumContentsLength(24)
        self.dataset.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.dataset.view().setMinimumWidth(330)
        self._dataset_headers = {}
        for group, entries in (
            ("CURRENT MODEL", (("Model: support", "support"), ("Model: support + target field", "preview_target"))),
            (
                "SAVED RUN · GEOMETRY",
                (
                    ("Stream function [A]", "stream_function"),
                    ("Surface current density |K| [A/m]", "surface_current_density"),
                    ("Contours", "contours"),
                    ("Connected routes", "routes"),
                    ("Evaluated winding field paths", "evaluated_winding"),
                    ("Pipeline conductor centerlines", "fabrication"),
                    ("Pipeline swept conductor", "swept"),
                ),
            ),
            (
                "SAVED RUN · FIELD MAPS",
                (
                    ("Target field", "target"),
                    ("Continuous-sheet field", "sheet"),
                    ("Independent sheet validation", "validation"),
                    ("Contour wire field", "contour_field"),
                    ("Contour residual (field − target)", "contour_residual"),
                    ("Evaluated winding field", "routed_field"),
                    ("Winding residual (field − target)", "routed_residual"),
                ),
            ),
            (
                "FABRICATION REVIEW",
                (
                    ("Review conductor centerlines", "review_conductor"),
                    ("Review electrical field paths", "review_electrical"),
                    ("Review field", "fabricated"),
                    ("Independent winding validation", "winding_validation"),
                    ("Independent winding residual", "winding_validation_residual"),
                ),
            ),
        ):
            self.dataset.addItem(group)
            header = self.dataset.model().item(self.dataset.count() - 1)
            self._dataset_headers[group] = self.dataset.count() - 1
            header.setEnabled(False)
            font = header.font()
            font.setBold(True)
            header.setFont(font)
            for label, key in entries:
                self.dataset.addItem(label, key)
        self.dataset.setMaxVisibleItems(self.dataset.count())
        self.dataset.setCurrentIndex(self.dataset.findData("support"))
        dataset_icons = {
            "support": "mesh",
            "preview_target": "target-gradient",
            "target": "target-map",
            "stream_function": "stream-function",
            "contours": "contours",
            "routes": "connected-winding",
            "evaluated_winding": "connected-winding",
            "fabrication": "contours",
            "swept": "conductor-sweep",
            "sheet": "stream-function",
            "surface_current_density": "stream-function",
            "validation": "independent-validation",
            "contour_field": "contours",
            "contour_residual": "field-error",
            "routed_field": "field-error",
            "routed_residual": "field-error",
            "review_conductor": "conductor-profile",
            "review_electrical": "current-direction",
            "fabricated": "field-error",
            "winding_validation": "independent-validation",
            "winding_validation_residual": "field-error",
        }
        for index in range(self.dataset.count()):
            key = self.dataset.itemData(index)
            if key in dataset_icons:
                self.dataset.setItemIcon(index, icon(dataset_icons[key]))
        plot_label.setBuddy(self.dataset)
        toolbar.addWidget(self.dataset, 1)
        toolbar.addWidget(self.inspection_button)
        plot_layout.addLayout(toolbar)
        plot_layout.addWidget(self.inspection_note)
        self.field_controls = QWidget()
        field_layout = QVBoxLayout(self.field_controls)
        field_layout.setContentsMargins(0, 0, 0, 0)
        field_layout.setSpacing(4)
        self.field_options = QWidget()
        field_toolbar = QHBoxLayout(self.field_options)
        field_toolbar.setContentsMargins(0, 0, 0, 0)
        component_label = QLabel("Component")
        self.component = QComboBox()
        self.component.setAccessibleName("Field component")
        self.component.addItems(["Bz", "Bx", "By", "|B|"])
        for index in range(self.component.count()):
            self.component.setItemIcon(index, icon("field-component"))
        component_label.setBuddy(self.component)
        field_toolbar.addWidget(component_label)
        field_toolbar.addWidget(self.component)
        units_label = QLabel("Units")
        self.units = QComboBox()
        self.units.setAccessibleName("Field display units")
        self.units.addItems(["T", "mT", "µT", "nT"])
        units_label.setBuddy(self.units)
        field_toolbar.addWidget(units_label)
        field_toolbar.addWidget(self.units)
        field_toolbar.addStretch()
        field_layout.addWidget(self.field_options)
        self.current_controls = QWidget()
        current_toolbar = QHBoxLayout(self.current_controls)
        current_toolbar.setContentsMargins(0, 0, 0, 0)
        current_label = QLabel("Wire current policy")
        current_toolbar.addWidget(current_label)
        self.wire_current = QComboBox()
        self.wire_current.setObjectName("viewport_wire_current")
        self.wire_current.setAccessibleName("Saved wire field current policy")
        self.wire_current.addItems(["Stored evaluation", "Original contour current"])
        self.wire_current.setToolTip("Select already computed wire maps; no current fitting or field recalculation.")
        current_label.setBuddy(self.wire_current)
        current_toolbar.addWidget(self.wire_current)
        current_toolbar.addStretch()
        field_layout.addWidget(self.current_controls)
        plot_layout.addWidget(self.field_controls)
        self.field_controls.hide()
        self.wire_current.hide()
        layout.addWidget(self.plot_selection_bar)
        view_toolbar = QHBoxLayout()
        self.view = QComboBox()
        self.view.setAccessibleName("Camera orientation")
        self.view.setToolTip("Choose a world-coordinate camera view; the coil model is unchanged.")
        self.view.addItems(["Isometric", "XY", "XZ", "YZ", "Custom"])
        self.view.model().item(self.view.findText("Custom")).setEnabled(False)
        for index in range(self.view.count()):
            self.view.setItemIcon(index, icon("placement"))
        view_toolbar.addWidget(self.view)
        reset = QPushButton("Fit")
        reset.setIcon(icon("fit"))
        reset.setToolTip("Fit the visible geometry and samples in the viewport.")
        view_toolbar.addWidget(reset)
        view_toolbar.addStretch()

        def controls_popup(label, title, symbol, minimum_width):
            button = QToolButton(self)
            button.setText(label)
            button.setIcon(icon(symbol))
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            button.setAccessibleName(title)
            button.setToolTip(title + "; these controls change the view only.")
            menu = QMenu(button)
            menu.addSection(title)
            panel = QWidget()
            panel.setMinimumWidth(minimum_width)
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(10, 6, 10, 10)
            panel_layout.setSpacing(6)
            action = QWidgetAction(menu)
            action.setDefaultWidget(panel)
            menu.addAction(action)
            button.setMenu(menu)
            view_toolbar.addWidget(button)
            return button, menu, panel, panel_layout

        self.layers_button, self.layers_menu, self.layers_panel, layers = controls_popup(
            "Layers", "Visible layers", "visibility", 230
        )
        self.layers_button.setObjectName("viewport_layers_button")
        self.display_button, self.display_menu, self.display_panel, display = controls_popup(
            "Display", "Display and sampling", "settings", 330
        )
        self.display_button.setObjectName("viewport_display_button")
        layout.addLayout(view_toolbar)
        self.orthographic = QCheckBox("Orthographic")
        self.orthographic.setToolTip("Parallel camera projection; it changes viewing perspective only.")
        self.orthographic.setChecked(True)
        self.support_layer = QCheckBox("Support")
        self.support_layer.setToolTip("Show the coil support as spatial context.")
        self.support_layer.setChecked(True)
        self.roi_layer = QCheckBox("Target points")
        self.roi_layer.setToolTip(
            "Show target sample coordinates from the displayed model or saved run; no interpolation."
        )
        self.edges = QCheckBox("Edges")
        self.edges.setAccessibleName("Mesh edges")
        self.edges.setToolTip("Show the triangle edges of support and swept conductor meshes.")
        self.edges.setChecked(True)
        self.clip = QCheckBox("Clip")
        self.clip.setAccessibleName("Clip plane")
        self.clip.setToolTip("Inspect a cross-section of the displayed dataset with an interactive clipping plane.")
        for widget in (self.support_layer, self.roi_layer):
            layers.addWidget(widget)
        for widget in (self.orthographic, self.edges, self.clip):
            display.addWidget(widget)
        self.slice_controls = QWidget()
        slice_layout = QVBoxLayout(self.slice_controls)
        slice_layout.setContentsMargins(0, 4, 0, 0)
        slice_layout.setSpacing(4)
        slice_bar = QHBoxLayout()
        slice_title = QLabel("Sample slab")
        slice_bar.addWidget(slice_title)
        self.slice_axis = QComboBox()
        self.slice_axis.setAccessibleName("Sample slab axis")
        self.slice_axis.setToolTip("Select an existing sampled coordinate slab; no interpolation or field refit.")
        self.slice_axis.addItems(["All points", "x", "y", "z"])
        for index in range(self.slice_axis.count()):
            self.slice_axis.setItemIcon(index, icon("slice"))
        slice_title.setBuddy(self.slice_axis)
        slice_bar.addWidget(self.slice_axis)
        slice_layout.addLayout(slice_bar)
        self.slice_position = QSlider(Qt.Orientation.Horizontal)
        self.slice_position.setAccessibleName("Sample slab position")
        self.slice_position.setToolTip("Choose the nearest stored sample slab along the selected axis.")
        self.slice_position.setRange(0, 100)
        self.slice_position.setValue(50)
        self.slice_position.setEnabled(False)
        slice_layout.addWidget(self.slice_position)
        self.slice_label = QLabel("No interpolation")
        self.slice_label.setWordWrap(True)
        slice_layout.addWidget(self.slice_label)
        self.slice_controls.setVisible(False)
        display.addWidget(self.slice_controls)
        self._popup_controls = {
            self.layers_menu: (self.support_layer, self.roi_layer),
            self.display_menu: (self.orthographic, self.edges, self.clip, self.slice_axis, self.slice_position),
        }
        for controls in self._popup_controls.values():
            for control in controls:
                control.installEventFilter(self)
        for menu, first_control in (
            (self.layers_menu, self.support_layer),
            (self.display_menu, self.orthographic),
        ):
            menu.aboutToShow.connect(
                lambda selected=menu, control=first_control: QTimer.singleShot(
                    0,
                    self,
                    lambda: control.setFocus(Qt.FocusReason.PopupFocusReason) if selected.isVisible() else None,
                )
            )
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#fafafa")
        self.plotter.enable_parallel_projection()
        self.plotter.interactor.installEventFilter(self)
        self.plotter.interactor.setToolTip(
            "Drag to rotate; click and release to inspect an object and probe its sample."
        )
        layout.addWidget(self.plotter.interactor, 1)
        self.hidden_geometry = QWidget()
        self.hidden_geometry.setObjectName("viewport_hidden_geometry")
        hidden_layout = QHBoxLayout(self.hidden_geometry)
        hidden_layout.setContentsMargins(4, 0, 4, 0)
        hidden_note = QLabel("Support hidden. Its geometry is available; enable the support layer to display it.")
        hidden_note.setWordWrap(True)
        hidden_layout.addWidget(hidden_note, 1)
        self.show_support = QPushButton("Show support")
        self.show_support.setObjectName("viewport_show_support")
        self.show_support.clicked.connect(lambda: self.support_layer.setChecked(True))
        hidden_layout.addWidget(self.show_support)
        self.hidden_geometry.hide()
        layout.addWidget(self.hidden_geometry)
        self.caption = QLabel("Coordinates [m] · drag to rotate; click and release to inspect a model object")
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)
        self.probe = QLabel("Probe: —")
        self.probe.setWordWrap(True)
        self.probe.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        probe_bar = QHBoxLayout()
        probe_bar.addWidget(self.probe, 1)
        self.clear_highlight = QPushButton("Clear highlight")
        self.clear_highlight.setIcon(icon("clear"))
        self.clear_highlight.setObjectName("viewport_clear_highlight")
        self.clear_highlight.setToolTip("Remove the recorded evidence overlay and return to the previous view.")
        self.clear_highlight.setVisible(False)
        self.clear_highlight.clicked.connect(self._clear_highlight)
        probe_bar.addWidget(self.clear_highlight)
        layout.addLayout(probe_bar)
        self.dataset.currentIndexChanged.connect(self._dataset_changed)
        for widget in (self.component, self.units, self.slice_axis):
            widget.currentIndexChanged.connect(self._render)
        self.wire_current.currentIndexChanged.connect(self._wire_current_changed)
        self.slice_position.valueChanged.connect(self._render)
        for widget in (self.clip, self.support_layer, self.roi_layer, self.edges):
            widget.toggled.connect(self._render)
        self.view.currentIndexChanged.connect(self._set_view)
        # Activating the current preset must work after manually moving its camera.
        self.view.activated.connect(self._set_view)
        self.orthographic.toggled.connect(self._set_projection)
        reset.clicked.connect(self._fit_view)
        self.plotter.enable_mesh_picking(
            callback=self._picked_actor, show=False, show_message=False, left_clicking=True, use_actor=True
        )
        self._rebuild_capabilities()
        self._refresh_availability()

    def eventFilter(self, watched, event):
        if watched is getattr(getattr(self, "plotter", None), "interactor", None):
            kind = event.type()
            if kind == QEvent.Type.MouseButtonPress:
                if self._navigation_camera is None:
                    self._navigation_camera = deepcopy(self.plotter.camera_position)
                if event.button() == Qt.MouseButton.LeftButton:
                    self._pick_press = event.globalPosition()
                    self._pick_dragged = bool(event.modifiers()) or event.buttons() != Qt.MouseButton.LeftButton
                    self._pending_pick = None
                elif self._pick_press is not None:
                    self._pick_dragged = True
            elif kind == QEvent.Type.MouseMove and self._pick_press is not None:
                self._pick_dragged |= (
                    (event.globalPosition() - self._pick_press).manhattanLength() >= QApplication.startDragDistance()
                    or bool(event.modifiers())
                    or event.buttons() != Qt.MouseButton.LeftButton
                )
            elif kind == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                actor = self._pending_pick
                inspect = (
                    self._pick_press is not None
                    and not self._pick_dragged
                    and not event.modifiers()
                    and not event.buttons()
                    and (event.globalPosition() - self._pick_press).manhattanLength() < QApplication.startDragDistance()
                )
                self._pick_press = self._pending_pick = None
                if inspect and actor is not None:
                    self._picked_actor(actor)
            elif kind in {QEvent.Type.Leave, QEvent.Type.FocusOut, QEvent.Type.Hide}:
                self._pick_press = self._pending_pick = None
            if (
                kind == QEvent.Type.MouseButtonRelease
                and not event.buttons()
                or kind in {QEvent.Type.Leave, QEvent.Type.FocusOut, QEvent.Type.Hide}
            ) and self._navigation_camera is not None:
                before, self._navigation_camera = self._navigation_camera, None
                # VTK handles the event after this filter. Compare its resulting
                # camera, so clipping-widget interaction does not imply rotation.
                QTimer.singleShot(0, self, lambda: self._camera_gesture_finished(before))
            # Do not consume events: VTK retains its rotate, pan and zoom gestures.
        # QWidgetAction otherwise sends Tab back to the QMenu action list.
        # Keep keyboard focus among the actual, available controls in a popup.
        if event.type() == QEvent.Type.KeyPress and event.key() in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
            for menu, controls in getattr(self, "_popup_controls", {}).items():
                if menu.isVisible() and watched in controls:
                    available = [control for control in controls if control.isVisible() and control.isEnabled()]
                    if watched in available:
                        backward = event.key() == Qt.Key.Key_Backtab or bool(
                            event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                        )
                        index = (available.index(watched) + (-1 if backward else 1)) % len(available)
                        available[index].setFocus(Qt.FocusReason.TabFocusReason)
                        return True
        return super().eventFilter(watched, event)

    def _camera_gesture_finished(self, before):
        if not self._closed and not np.array_equal(before, self.plotter.camera_position):
            for pane in (self, self._comparison_peer):
                if pane is not None and not pane._closed:
                    with QSignalBlocker(pane.view):
                        pane.view.setCurrentText("Custom")

    def _rebuild_capabilities(self):
        arrays = self._stage_arrays if self._live_stage_preview else self._arrays
        model = {**self._preview_geometry, **self._preview_arrays}
        self._source_support_available = resolve_dataset(arrays, "support").available
        self._source_points_available = points_available(arrays)
        self._model_points_available = points_available(self._preview_arrays)
        previews = {key: resolve_dataset(model, key) for key in ("support", "preview_target")}
        self._capabilities = {}
        for original in (False, True):
            catalog = dataset_capabilities(arrays, original=original)
            catalog.update(previews)
            self._capabilities[original] = catalog

    def _capability(self, key, *, original=None, comparison=True):
        original = self.wire_current.currentIndex() == 1 if original is None else original
        capability = self._capabilities.get(original, {}).get(key, DatasetCapability(False, "Unknown dataset."))
        if key in {"support", "preview_target"} and not capability.available and self._preview_error:
            capability = DatasetCapability(False, self._preview_error, capability.required_stage, capability.invalid)
        if comparison and capability.available and self._comparison_peer is not None:
            peer = self._comparison_peer._capability(key, original=original, comparison=False)
            if not peer.available:
                return DatasetCapability(
                    False, "Other comparison run: " + peer.reason, peer.required_stage, peer.invalid
                )
        return capability

    def _refresh_availability(self):
        for index in range(self.dataset.count()):
            key = self.dataset.itemData(index)
            if key is None:
                continue
            capability = self._capability(key)
            item = self.dataset.model().item(index)
            item.setEnabled(capability.available)
            item.setToolTip("Available evidence." if capability.available else capability.explanation)
        source = "SAVED RUN"
        if self._live_stage_preview:
            source = (
                "RUN SNAPSHOT · " + self._stage_execution_status.upper()
                if self._stage_execution_status in {"cancelled", "failed", "completed"}
                else "LIVE STUDY"
            )
        elif self._verification_error:
            source = "CACHED RUN · VERIFICATION FAILED"
        for suffix in ("GEOMETRY", "FIELD MAPS"):
            header = self.dataset.model().item(self._dataset_headers["SAVED RUN · " + suffix])
            header.setText(source + " · " + suffix)
        key = self.dataset.currentData()
        for index, original in enumerate((False, True)):
            if key in WIRE_FIELDS:
                capability = self._capability(key, original=original)
            else:
                options = [self._capability(field, original=original) for field in WIRE_FIELDS]
                capability = next((item for item in options if item.available), options[0])
            item = self.wire_current.model().item(index)
            item.setEnabled(capability.available)
            item.setToolTip("Saved field maps at this current." if capability.available else capability.explanation)
        for task, action in self.inspection_actions.items():
            available = self._inspection_selection(task)[0] is not None
            action.setEnabled(available)
            self.inspection_buttons[task].setEnabled(available)
            if not available and self.inspection_task.currentData() == task:
                self._clear_inspection_task()

    def _wire_current_changed(self, index):
        key = self.dataset.currentData()
        capability = self._capability(key)
        if key not in WIRE_FIELDS:
            options = [self._capability(field, original=index == 1) for field in WIRE_FIELDS]
            capability = next((item for item in options if item.available), options[0])
        if not capability.available:
            with QSignalBlocker(self.wire_current):
                self.wire_current.setCurrentIndex(self._last_wire_current)
            self.error.emit("Current selection unavailable: " + capability.explanation)
            return
        self._last_wire_current = index
        self._render()

    def set_stage_preview_status(self, status):
        """Mark a retained stage snapshot as stopped without promoting it to a run."""
        status = str(status or "running")
        if status == self._stage_execution_status:
            return
        self._stage_execution_status = status
        if self._live_stage_preview:
            self._render()

    def set_verification_error(self, error):
        """Identify cached display evidence after its saved source failed verification."""
        error = str(error or "")
        if error == self._verification_error:
            return
        self._verification_error = error
        self._render()

    @property
    def current_capability(self):
        return self._capability(self.dataset.currentData())

    def _model_dataset(self):
        return "preview_target" if self._capability("preview_target", comparison=False).available else "support"

    def render_project(self, project):
        self.render_config(project.config, project.project_root, _owner=project)

    def render_config(self, config, project_root: Path | None = None, *, _owner=None):
        if self._closed:
            return
        support_identity, target_identity = _preview_identities(config, getattr(_owner, "values", None))
        owner = getattr(_owner, "id", None)
        has_support = str(config.coil_mesh_file).strip().lower() not in {"", "none"}
        complete = has_support and _has_target(config)
        new_model = owner != self._model_owner
        target_changed = target_identity != self._target_identity
        # Applied target choices reveal their spatial result immediately. Select
        # before dispatch: a later explicit plot/run/session choice must win when
        # the asynchronous result arrives.
        if new_model or target_changed or complete and not self._model_complete:
            self._select_model_preview(complete, config)
        if new_model or support_identity != self._support_identity:
            self._support = None
            self._preview_geometry = {}
        self._support_identity, self._target_identity = support_identity, target_identity
        self._model_owner, self._model_complete = owner, complete
        self._preview_generation += 1
        self._preview_error = "Preparing current model preview…"
        self.preflight_ready.emit(
            {
                "available": False,
                "message": self._preview_error,
                "coordinate_frame": "world",
                "target": {"available": False},
            }
        )
        self._preview_arrays = {}
        self._rebuild_capabilities()
        self._refresh_availability()
        root = project_root or Path.cwd()
        if str(config.coil_mesh_file).strip().lower() in {"", "none"}:
            # Incomplete models are normal while choosing inputs. They need no
            # backend worker; any old in-flight result is already invalidated.
            self._pending_preview = None
            _, _, self._preview_error, preflight = _preview_data(config, root)
            self.preflight_ready.emit(deepcopy(preflight))
            if self.dataset.currentData() in {"support", "preview_target"}:
                self._render()
            return
        request = self._preview_generation, deepcopy(config), root, _owner
        if self._synchronous_preview:
            task = _PreviewTask(*request)
            task.signals.finished.connect(self._preview_finished)
            task.run()
        elif self._preview_worker is None:
            self._start_preview(request)
        else:
            # Coalesce rapid edits. Never queue an unbounded sequence of meshes.
            self._pending_preview = request
        if self.dataset.currentData() in {"support", "preview_target"}:
            self._render()

    def _select_model_preview(self, show_target, config):
        self._stage_arrays, self._stage_metadata = {}, {}
        self._live_stage_preview = False
        self._source_notice = ""
        self._clear_inspection_task()
        self._located_issue = None
        self._before_issue_view = None
        with QSignalBlocker(self.dataset), QSignalBlocker(self.support_layer), QSignalBlocker(self.roi_layer):
            self.dataset.setCurrentIndex(self.dataset.findData("preview_target" if show_target else "support"))
            self.support_layer.setChecked(True)
            self.roi_layer.setChecked(show_target)
        if show_target:
            # An old sampled slab/component can otherwise make a newly chosen
            # scalar target appear empty or identically zero.
            component = config.target_field_component
            component = component if component in {"Bx", "By", "Bz"} else "|B|"
            with (
                QSignalBlocker(self.component),
                QSignalBlocker(self.units),
                QSignalBlocker(self.slice_axis),
                QSignalBlocker(self.clip),
            ):
                self.component.setCurrentText(component)
                if self.units.findText(config.field_display_units) >= 0:
                    self.units.setCurrentText(config.field_display_units)
                self.slice_axis.setCurrentText("All points")
                self.clip.setChecked(False)

    def _start_preview(self, request):
        self._preview_worker = _PreviewTask(*request)
        self._preview_worker.signals.finished.connect(self._preview_finished)
        QThreadPool.globalInstance().start(self._preview_worker)

    @Slot(int, object)
    def _preview_finished(self, generation, result):
        if self._closed:
            return
        self._preview_worker = None
        if generation == self._preview_generation:
            mesh, self._preview_arrays, self._preview_error, preflight = result
            self._preview_geometry = {"vertices_m": mesh.vertices, "faces": mesh.faces} if mesh is not None else {}
            geometry = resolve_dataset(self._preview_geometry, "support")
            if mesh is not None and not geometry.available:
                self._preview_error = "Model preview unavailable: " + geometry.explanation
                self._preview_arrays, self._preview_geometry = {}, {}
                preflight = {
                    **preflight,
                    "available": False,
                    "message": self._preview_error,
                    "target": {"available": False},
                }
                mesh = None
            self._support = self._mesh(mesh.vertices, mesh.faces) if mesh is not None else None
            self._rebuild_capabilities()
            self._refresh_availability()
            if self.dataset.currentData() == "preview_target" and not self._capability("preview_target").available:
                with QSignalBlocker(self.dataset):
                    self.dataset.setCurrentIndex(self.dataset.findData("support"))
            self.preflight_ready.emit(deepcopy(preflight))
            if mesh is None or self._preview_error.startswith("Target preview unavailable:"):
                self.error.emit(self._preview_error)
            if self.dataset.currentData() in {"support", "preview_target"}:
                self._render()
        if self._pending_preview is not None:
            request, self._pending_preview = self._pending_preview, None
            self._start_preview(request)

    def display_run(self, arrays, metadata=None):
        self._stage_arrays, self._stage_metadata = {}, {}
        self._live_stage_preview = False
        self._clear_inspection_task()
        self._arrays = {name: np.array(value, copy=True) for name, value in arrays.items()}
        self._metadata = deepcopy(metadata or {})
        self._verification_error = ""
        self._rebuild_capabilities()
        self._located_issue = None
        self._before_issue_view = None
        self._first_view = True
        preferred = self._preferred_run_dataset()
        self._source_notice = ""
        if preferred in {"support", "preview_target"}:
            catalog = dataset_capabilities(self._arrays, original=self.wire_current.currentIndex() == 1)
            reason = next(
                (item.reason for item in catalog.values() if item.invalid),
                "No saved plot data for this stage and current policy.",
            )
            self._source_notice = "Saved run has no available plot data: " + reason
            self.error.emit(self._source_notice)
        with QSignalBlocker(self.dataset):
            self.dataset.setCurrentIndex(self.dataset.findData(preferred))
        self._render()

    def display_stage_preview(self, arrays, metadata):
        """Display verified stage outputs without creating or replacing a run.

        The caller owns job/revision verification. No interpolation, field solve,
        current fitting or acceptance inference occurs in this rendering adapter.
        """
        if self._closed:
            return
        had_scene = bool(self._actors)
        identity_fields = ("request_sha256", "project_id", "project_revision", "project_fingerprint")
        first_snapshot = not self._live_stage_preview or any(
            metadata.get(name) != self._stage_metadata.get(name) for name in identity_fields
        )
        self._clear_inspection_task()
        self._stage_arrays = {name: np.array(value, copy=True) for name, value in arrays.items()}
        self._stage_metadata = {**deepcopy(metadata), "live_stage_preview": True, "acceptance_claim": False}
        self._live_stage_preview = True
        self._source_notice = ""
        if first_snapshot:
            self._stage_execution_status = "running"
        self._rebuild_capabilities()
        self._located_issue = None
        self._before_issue_view = None
        self._first_view = not had_scene
        preferred = self._preferred_run_dataset(self._stage_arrays)
        with QSignalBlocker(self.dataset), QSignalBlocker(self.support_layer), QSignalBlocker(self.roi_layer):
            self.dataset.setCurrentIndex(self.dataset.findData(preferred))
            self.support_layer.setChecked(True)
            self.roi_layer.setChecked(True)
        if first_snapshot and metadata.get("target_field_component") in {"Bx", "By", "Bz"}:
            with QSignalBlocker(self.component):
                self.component.setCurrentText(metadata["target_field_component"])
        self._render()

    @property
    def stage_preview_metadata(self):
        if self._live_stage_preview and self.dataset.currentData() not in {"support", "preview_target"}:
            return deepcopy(self._stage_metadata)
        return None

    def clear_stage_preview(self):
        """Return to current inputs, retaining any separately loaded saved run."""
        self._stage_arrays, self._stage_metadata = {}, {}
        self._live_stage_preview = False
        self._source_notice = ""
        self._clear_inspection_task()
        self._rebuild_capabilities()
        with QSignalBlocker(self.dataset):
            self.dataset.setCurrentIndex(self.dataset.findData(self._model_dataset()))
        self._render()

    def _preferred_run_dataset(self, arrays=None):
        """Show the latest saved geometry without reconstructing missing stages."""
        arrays = self._arrays if arrays is None else arrays
        original = self.wire_current.currentIndex() == 1
        active = self._stage_arrays if self._live_stage_preview else self._arrays
        catalog = self._capabilities[original] if arrays is active else dataset_capabilities(arrays, original=original)
        for key in (
            "review_conductor",
            "swept",
            "fabrication",
            "evaluated_winding",
            "routes",
            "contours",
            "stream_function",
            "target",
            "sheet",
            "validation",
            "routed_field",
            "contour_field",
            "fabricated",
            "review_electrical",
        ):
            if catalog[key].available:
                return key
        return self._model_dataset()

    def clear_run(self):
        """Return to the current model without retaining another project's run."""
        self._clear_inspection_task()
        self._arrays = {}
        self._metadata = {}
        self._stage_arrays, self._stage_metadata = {}, {}
        self._live_stage_preview = False
        self._verification_error = ""
        self._source_notice = ""
        self._located_issue = None
        self._before_issue_view = None
        self._rebuild_capabilities()
        with QSignalBlocker(self.dataset):
            self.dataset.setCurrentIndex(self.dataset.findData(self._model_dataset()))
        self._render()

    def clear_model(self):
        """Clear project evidence and discard any in-flight preview's eventual result."""
        if self._closed:
            return
        self._preview_generation += 1
        self._pending_preview = None
        self._preview_arrays = {}
        self._preview_geometry = {}
        self._support = None
        self._support_identity = self._target_identity = None
        self._model_owner = None
        self._model_complete = False
        self._preview_error = "No model loaded."
        self._colour_limits = None
        self._first_view = True
        self.inspection_menu.hide()
        self.layers_menu.hide()
        self.display_menu.hide()
        if self._comparison is not None:
            self._comparison.close()
        # Retain an active worker until its callback: subsequent model edits can
        # still coalesce behind it without an old callback clearing a new worker.
        self.clear_run()
        self.preflight_ready.emit(
            {
                "available": False,
                "message": self._preview_error,
                "coordinate_frame": "world",
                "target": {"available": False},
            }
        )

    def _sync_inspection_buttons(self, *_args):
        task = self.inspection_task.currentData()
        for key, button in self.inspection_buttons.items():
            button.setChecked(key == task)
            self.inspection_actions[key].setChecked(key == task)

    def _clear_inspection_task(self):
        with QSignalBlocker(self.inspection_task):
            self.inspection_task.setCurrentIndex(0)
        self._sync_inspection_buttons()
        self.inspection_note.hide()

    def _inspection_selection(self, task):
        choices = {
            "target": (("preview_target", "Current model target and ROI; saved runs retain their original target."),),
            "field": (
                (
                    "routed_residual",
                    "Connected-winding residual on its saved design samples; inspect the stored current policy below.",
                ),
                (
                    "contour_residual",
                    "Contour residual on its saved design samples; connected-winding evidence is unavailable.",
                ),
                ("validation", "Independent sheet validation field; final winding evidence is unavailable."),
                (
                    "sheet",
                    "Continuous-sheet field on design samples; independent and winding evidence are unavailable.",
                ),
                (
                    "routed_field",
                    "Connected-winding field; a matching saved target is unavailable, so no residual can be displayed.",
                ),
                (
                    "contour_field",
                    "Contour field; a matching saved target is unavailable, so no residual can be displayed.",
                ),
            ),
            "wire": (
                (
                    "swept",
                    "Saved conductor solids; dimensional acceptance remains tied to the recorded geometry checks.",
                ),
                *(
                    (
                        key,
                        "Conductor solids are unavailable. Showing saved centerlines; these do not establish physical wire clearance.",
                    )
                    for key in ("review_conductor", "fabrication", "routes", "contours")
                ),
            ),
            "connections": (
                ("routes", "Saved routed paths; feed and drive-terminal completeness requires separate evidence."),
                (
                    "evaluated_winding",
                    "Saved evaluated electrical paths; inspect circuit topology and feed/drive completeness in the recorded evidence.",
                ),
                (
                    "fabrication",
                    "Saved conductor centerlines; their circuit topology and drive completeness require the recorded winding evidence.",
                ),
            ),
        }
        for key, note in choices.get(task, ()):
            if self._capability(key).available:
                return key, note
        return None, "No available evidence for this inspection at the selected source and current policy."

    def inspect_task(self, task):
        """Select existing evidence for a question; never generate or refit data."""
        if task not in {"target", "field", "wire", "connections"}:
            return False
        dataset, note = self._inspection_selection(task)
        if dataset is None:
            self.error.emit(note)
            return False
        self.support_layer.setChecked(True)
        self.roi_layer.setChecked(task in {"target", "field"})
        self.dataset.setCurrentIndex(self.dataset.findData(dataset))
        with QSignalBlocker(self.inspection_task):
            self.inspection_task.setCurrentIndex(self.inspection_task.findData(task))
        self._sync_inspection_buttons()
        self.inspection_note.setText(note)
        self.inspection_note.show()
        return True

    def view_state(self) -> dict:
        """Display state only; no configuration or result arrays enter sessions."""
        state = {
            "schema": "coilgen-viewport-v1",
            "dataset": self.dataset.currentData(),
            "source": "model"
            if self.dataset.currentData() in {"support", "preview_target"}
            else "live"
            if self._live_stage_preview
            else "run",
            "slice_position": self.slice_position.value(),
            "camera": [list(map(float, vector)) for vector in self.plotter.camera_position],
        }
        for name in ("component", "units", "wire_current", "slice_axis", "view"):
            state[name] = getattr(self, name).currentText()
        for name in ("orthographic", "support_layer", "roi_layer", "edges", "clip"):
            state[name] = getattr(self, name).isChecked()
        camera = getattr(self.plotter, "camera", None)
        if camera is not None:
            state["parallel_scale"] = float(camera.parallel_scale)
        return state

    def restore_view_state(self, state) -> bool:
        """Atomically reject malformed session values before touching Qt/VTK."""
        try:
            if not isinstance(state, dict) or state.get("schema", "coilgen-viewport-v1") != "coilgen-viewport-v1":
                raise ValueError("Unrecognized viewport state.")
            indices = {}
            if "dataset" in state:
                if not isinstance(state["dataset"], str):
                    raise ValueError("Invalid viewport dataset selection.")
                index = self.dataset.findData(state["dataset"])
                if index < 0:
                    raise ValueError("Unknown viewport dataset.")
                indices["dataset"] = index
            for name in ("component", "units", "wire_current", "slice_axis", "view"):
                if name in state:
                    if not isinstance(state[name], str):
                        raise ValueError(f"Invalid {name} selection.")
                    index = getattr(self, name).findText(state[name])
                    if index < 0:
                        raise ValueError(f"Unknown {name} selection.")
                    indices[name] = index
            flags = {
                name: state[name]
                for name in ("orthographic", "support_layer", "roi_layer", "edges", "clip")
                if name in state
            }
            if any(not isinstance(value, bool) for value in flags.values()):
                raise ValueError("Viewport visibility/projection flags must be booleans.")
            position = state.get("slice_position", self.slice_position.value())
            if isinstance(position, bool) or not isinstance(position, int) or not 0 <= position <= 100:
                raise ValueError("Viewport slice position must be an integer from 0 to 100.")
            dataset = state.get("dataset", self.dataset.currentData())
            expected_source = (
                "model" if dataset in {"support", "preview_target"} else "live" if self._live_stage_preview else "run"
            )
            if "source" in state and state["source"] != expected_source:
                raise ValueError("Viewport model/run source disagrees with its dataset.")
            capability = self._capability(
                dataset, original=indices.get("wire_current", self.wire_current.currentIndex()) == 1
            )
            pending_target = dataset == "preview_target" and self._preview_error == "Preparing current model preview…"
            if not capability.available and dataset != "support" and not pending_target:
                raise ValueError("Dataset unavailable: " + capability.explanation)
            parallel_scale = state.get("parallel_scale")
            if "parallel_scale" in state and (
                not isinstance(parallel_scale, (int, float))
                or isinstance(parallel_scale, bool)
                or not np.isfinite(parallel_scale)
                or parallel_scale <= 0
            ):
                raise ValueError("Viewport orthographic scale must be a finite positive number.")
            camera = None
            if "camera" in state:
                raw_camera = state["camera"]
                if any(
                    not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, (bool, np.bool_))
                    for vector in raw_camera
                    for value in vector
                ):
                    raise ValueError("Viewport camera coordinates must be real numbers.")
                camera = np.asarray(state["camera"], dtype=float)
                if camera.shape != (3, 3) or not np.all(np.isfinite(camera)):
                    raise ValueError("Viewport camera requires three finite XYZ vectors.")
                with np.errstate(over="ignore", invalid="ignore"):
                    direction = camera[0] - camera[1]
                    distance, up_length = np.linalg.norm(direction), np.linalg.norm(camera[2])
                if not np.isfinite(distance) or distance <= 0 or not np.isfinite(up_length) or up_length <= 0:
                    raise ValueError("Viewport camera position/focus/up vectors are degenerate.")
                if np.linalg.norm(np.cross(direction / distance, camera[2] / up_length)) < 1e-10:
                    raise ValueError("Viewport camera up vector is parallel to its viewing direction.")
        except (TypeError, ValueError, OverflowError) as exc:
            self.error.emit(f"Saved viewport state rejected: {exc}")
            return False
        blockers = [QSignalBlocker(getattr(self, name)) for name in {*indices, *flags, "slice_position"}]
        for name, index in indices.items():
            getattr(self, name).setCurrentIndex(index)
        for name, value in flags.items():
            getattr(self, name).setChecked(value)
        self.slice_position.setValue(position)
        self._last_wire_current = self.wire_current.currentIndex()
        del blockers
        self._clear_inspection_task()
        self._located_issue = None
        self._before_issue_view = None
        self._first_view = False
        self._set_projection(self.orthographic.isChecked())
        if camera is not None:
            self.plotter.camera_position = camera.tolist()
        plot_camera = getattr(self.plotter, "camera", None)
        if plot_camera is not None and parallel_scale is not None:
            plot_camera.parallel_scale = float(parallel_scale)
        self._render()
        return True

    def set_colour_limits(self, limits=None):
        if limits is not None:
            values = np.asarray(limits, dtype=float)
            if values.shape != (2,) or not np.all(np.isfinite(values)) or values[0] >= values[1]:
                raise ValueError("Colour limits require two finite increasing values.")
            limits = tuple(map(float, values))
        self._colour_limits = limits
        self._render()

    def locate_issue(self, issue) -> bool:
        points = np.asarray(getattr(issue, "points_m", ()), dtype=float)
        if points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or not np.all(np.isfinite(points)):
            self.error.emit("No valid recorded spatial witness is available for this problem.")
            return False
        segments = getattr(issue, "segments", ())
        if any(
            len(pair) != 2 or any(not isinstance(index, int) or not 0 <= index < len(points) for index in pair)
            for pair in segments
        ):
            self.error.emit("Recorded problem interval indices are invalid.")
            return False
        if self._located_issue is None:
            camera = getattr(self.plotter, "camera", None)
            self._before_issue_view = {
                "dataset": self.dataset.currentData(),
                "camera": deepcopy(self.plotter.camera_position),
                "parallel_scale": camera.parallel_scale if camera is not None else None,
            }
        self._located_issue = issue
        self._clear_inspection_task()
        dataset = getattr(issue, "dataset", "review_conductor")
        index = self.dataset.findData(dataset)
        if index >= 0 and self._capability(dataset).available:
            with QSignalBlocker(self.dataset):
                self.dataset.setCurrentIndex(index)
        self._render()
        # reset_camera updates parallel scale as well as position. Merely moving
        # the camera cannot zoom the default orthographic view.
        lo, hi = points.min(axis=0), points.max(axis=0)
        center = (lo + hi) / 2
        radius = max(float(np.linalg.norm(hi - lo)) * 0.625, 0.0025)
        bounds = tuple(float(value) for axis in range(3) for value in (center[axis] - radius, center[axis] + radius))
        self.plotter.reset_camera(bounds=bounds)
        self.probe.setText(f"Located {getattr(issue, 'body_id', '')}: {getattr(issue, 'title', 'recorded evidence')}")
        self.plotter.render()
        return True

    def _dataset_changed(self, *_args):
        capability = self._capability(self.dataset.currentData())
        if not capability.available:
            previous = self.dataset.findData(self._last_valid_dataset or "support")
            with QSignalBlocker(self.dataset):
                self.dataset.setCurrentIndex(previous)
            self.error.emit("Plot selection unavailable: " + capability.explanation)
            return
        # An explicit dataset choice supersedes the previous inspection task.
        self._clear_inspection_task()
        self._located_issue = None
        self._before_issue_view = None
        self._render()

    def _clear_highlight(self):
        self._clear_inspection_task()
        previous = self._before_issue_view
        self._located_issue = None
        self._before_issue_view = None
        if previous is not None:
            with QSignalBlocker(self.dataset):
                self.dataset.setCurrentIndex(self.dataset.findData(previous["dataset"]))
        self._render()
        if previous is not None:
            self.plotter.camera_position = previous["camera"]
            camera = getattr(self.plotter, "camera", None)
            if camera is not None and previous["parallel_scale"] is not None:
                camera.parallel_scale = previous["parallel_scale"]
            self.plotter.render()

    def display_comparison(self, left, right):
        if self._comparison is not None:
            self._comparison.close()
        self._comparison = ComparisonDialog(left, right, parent=self.window())
        self._comparison.destroyed.connect(lambda *_args, token=id(self._comparison): self._comparison_destroyed(token))
        self._comparison.show()
        return self._comparison

    def _comparison_destroyed(self, token):
        if self._comparison is not None and id(self._comparison) == token:
            self._comparison = None

    def _mesh(self, vertices, faces):
        vertices = np.asarray(vertices, dtype=float)
        faces = np.asarray(faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("Support requires Nx3 vertices and triangular faces.")
        if not np.all(np.isfinite(vertices)) or (faces.size and (faces.min() < 0 or faces.max() >= len(vertices))):
            raise ValueError("Invalid support coordinates or vertex indices.")
        return self.pv.PolyData(vertices, np.column_stack((np.full(len(faces), 3), faces)).ravel())

    def _scalar(self, field):
        field = np.asarray(field, dtype=float)
        if field.ndim != 2 or field.shape[1] != 3 or not np.all(np.isfinite(field)):
            raise ValueError("Field dataset must contain finite Nx3 vectors in tesla.")
        component = self.component.currentText()
        values = (
            np.linalg.norm(field, axis=1) if component == "|B|" else field[:, {"Bx": 0, "By": 1, "Bz": 2}[component]]
        )
        return values * {"T": 1.0, "mT": 1e3, "µT": 1e6, "nT": 1e9}[self.units.currentText()]

    def _add(self, mesh, key, **options):
        actor = self.plotter.add_mesh(mesh, reset_camera=False, **options)
        self._actors[self._actor_key(actor)] = key
        return actor

    @staticmethod
    def _actor_key(actor):
        return actor.GetAddressAsString("") if hasattr(actor, "GetAddressAsString") else id(actor)

    def _picked_actor(self, actor):
        if self._pick_press is not None:
            # PyVista picks on press, before a camera drag can be distinguished.
            self._pending_pick = actor
            return
        actor_key = self._actor_key(actor)
        key = self._actors.get(actor_key)
        if key is None:
            return
        dataset = self.dataset.currentText()
        primary = actor_key == self._dataset_actor_key
        layer = (
            dataset
            if primary
            else {"geometry": "Support layer", "target": "Target points layer"}.get(key, "Winding layer")
        )
        text = f"Selected {layer}"
        point = getattr(self.plotter, "picked_point", None)
        mesh = self._dataset_mesh
        valid_point = point is not None and np.asarray(point).shape == (3,) and np.all(np.isfinite(point))
        if valid_point:
            text += f" at ({point[0]:.6g}, {point[1]:.6g}, {point[2]:.6g}) m"
        if primary and valid_point and mesh is not None and mesh.n_points:
            index = mesh.find_closest_point(point)
            coordinates = mesh.points[index]
            value = ""
            if mesh.active_scalars is not None:
                value = f" · {mesh.active_scalars_name}: {float(mesh.active_scalars[index]):.8g}"
            distance = float(np.linalg.norm(coordinates - point))
            source = "Nearest stored sample before clipping" if self.clip.isChecked() else "Nearest displayed sample"
            text += (
                f"\n{source} in {dataset}: ({coordinates[0]:.6g}, {coordinates[1]:.6g}, {coordinates[2]:.6g}) m"
                f" · distance {distance:.6g} m{value} · no interpolation"
            )
        elif not primary:
            text += f" · viewing {dataset}; no field value sampled at this position"
        # Capture source/sample feedback before navigation can change the model
        # preview or primary plot; the click refers to the scene that was picked.
        self.selected.emit(key)
        self.probe.setText(text)

    def _field_mesh(self, arrays, key):
        point_key = "validation_points_m" if key == "validation" else "target_points_m"
        if key == "fabricated" and "fabricated_points_m" in arrays:
            point_key = "fabricated_points_m"
        if key.startswith("winding_validation"):
            point_key = "winding_validation_points_m"
        field_key = {
            "preview_target": "target_field_T",
            "target": "target_field_T",
            "sheet": "sheet_field_T",
            "validation": "validation_field_T",
            "fabricated": "fabricated_field_T",
            "winding_validation": "winding_validation_field_T",
            "winding_validation_residual": "winding_validation_field_T",
            "contour_field": "contour_field_T",
            "contour_residual": "contour_field_T",
            "routed_field": "routed_field_T",
            "routed_residual": "routed_field_T",
        }[key]
        wire = key in {"contour_field", "contour_residual", "routed_field", "routed_residual"}
        if wire and self.wire_current.currentIndex() == 1:
            field_key = field_key.replace("_field_T", "_field_at_stream_current_T")
        if point_key not in arrays or field_key not in arrays:
            return None
        points = np.asarray(arrays[point_key], dtype=float)
        field = np.asarray(arrays[field_key], dtype=float)
        residual = key.endswith("_residual")
        if residual:
            reference_key = "winding_validation_target_T" if key.startswith("winding_validation") else "target_field_T"
            if reference_key not in arrays:
                return None
            reference = np.asarray(arrays[reference_key], dtype=float)
            if reference.shape != field.shape:
                raise ValueError("Residual target and field have incompatible dimensions.")
            field = field - reference
        values = self._scalar(field)
        if points.shape != (len(values), 3) or not np.all(np.isfinite(points)):
            raise ValueError("Field coordinates and values have incompatible dimensions.")
        self.slice_label.setText("No interpolation")
        if self.slice_axis.currentText() != "All points" and len(points):
            axis = {"x": 0, "y": 1, "z": 2}[self.slice_axis.currentText()]
            lo, hi = np.min(points[:, axis]), np.max(points[:, axis])
            position = lo + (hi - lo) * self.slice_position.value() / 100
            levels = np.unique(points[:, axis])
            position = levels[np.argmin(np.abs(levels - position))]
            distance = np.abs(points[:, axis] - position)
            # Closest sampled slab, without interpolating or refitting physics.
            mask = distance <= distance.min() + max(abs(hi - lo) * 1e-8, 1e-12)
            points, values = points[mask], values[mask]
            self.slice_label.setText(f"{self.slice_axis.currentText()} ≈ {position:.5g} m · {len(points)} samples")
        mesh = self.pv.PolyData(points)
        name = f"Δ{self.component.currentText()}" if residual else self.component.currentText()
        if residual and self.component.currentText() == "|B|":
            name = "|B − target|"
        mesh[f"{name} [{self.units.currentText()}]"] = values
        return mesh

    def _field_caption(self, key):
        if key not in {"contour_field", "contour_residual", "routed_field", "routed_residual"}:
            if self._live_stage_preview and key != "preview_target":
                return {
                    "target": "stage requested target",
                    "sheet": "stage continuous-sheet field on design samples",
                    "validation": "stage independent continuous-sheet grid",
                    "fabricated": "stage field samples; scope unverified",
                }[key]
            return {
                "preview_target": "current model target",
                "target": "immutable run target",
                "sheet": "immutable continuous-sheet field on design samples",
                "validation": "independent continuous-sheet grid; unchanged solved current",
                "fabricated": "fabrication-review thin-wire field; original contour current; design samples",
                "winding_validation": "recorded electrical paths; independent samples; original signed current; no refit",
                "winding_validation_residual": "independent thin-wire residual; original signed current; no refit",
            }[key]
        kind = "contour" if key.startswith("contour") else "routed"
        metadata = self._stage_metadata if self._live_stage_preview else self._metadata
        evidence = metadata.get("wire_fields", {}).get(kind, {})
        original = self.wire_current.currentIndex() == 1
        metric = evidence.get("at_stream_current", {}) if original else evidence
        current = metric.get("current_A")
        current_label = (
            "original contour current"
            if original
            else "stored refitted current"
            if metric.get("current_refitted")
            else "stored run current"
        )
        label = f"immutable {kind} thin-wire dataset; design samples; no independent wire acceptance"
        if current is None:
            label += "; current provenance unavailable"
        else:
            label += f"; {current_label} {current:.6g} A"
        geometry = evidence.get("geometry_sources")
        if geometry:
            label += "; geometry: " + ", ".join(dict.fromkeys(geometry))
        rms = metric.get("rms_error_T")
        if rms is not None:
            unit = self.units.currentText()
            scale = {"T": 1.0, "mT": 1e3, "µT": 1e6, "nT": 1e9}[unit]
            label += f"; global weighted {metric.get('component', 'target')} RMS {rms * scale:.6g} {unit}"
            relative = metric.get("relative_rms_error")
            label += (
                f" ({100 * relative:.4g}% of target RMS)"
                if relative is not None
                else " (relative undefined: zero target RMS)"
            )
        elif metric.get("reason"):
            label += "; " + metric["reason"]
        return label

    def _render(self, *_args):
        if self._closed:
            return
        self._refresh_availability()
        camera = self.plotter.camera_position
        widgets = getattr(self.plotter, "widgets", self.plotter)
        widgets.clear_plane_widgets()
        widgets.plane_clipped_meshes.clear()
        self.plotter.clear()
        self._actors.clear()
        self._dataset_mesh = None
        self._dataset_actor_key = None
        self.probe.setText("Probe: —")
        key = self.dataset.currentData()
        available = self._capability(key, comparison=False).available
        hidden_support = key == "support" and available and not self.support_layer.isChecked()
        self.hidden_geometry.setVisible(hidden_support)
        self._last_valid_dataset = key if available or key in {"support", "preview_target"} else None
        model = key in {"support", "preview_target"}
        live = self._live_stage_preview and not model
        arrays = self._preview_arrays if model else self._stage_arrays if live else self._arrays
        self.plotter.add_axes()
        locating = self._located_issue is not None
        self.clear_highlight.setVisible(locating)
        is_field = key in FIELD_KEYS and available
        has_wire_maps = any(
            self._capability(field, original=original).available for field in WIRE_FIELDS for original in (False, True)
        )
        self.component.setEnabled(is_field)
        self.units.setEnabled(is_field)
        self.field_controls.setVisible(is_field or has_wire_maps)
        self.field_options.setVisible(is_field)
        self.current_controls.setVisible(has_wire_maps)
        self.component.setToolTip(
            "Display a stored Cartesian field component or vector magnitude."
            if is_field
            else "Choose a field map to select its component. Geometry coordinates are in metres."
        )
        self.units.setToolTip(
            "Convert the displayed magnetic field values; stored SI data are unchanged."
            if is_field
            else "Field units apply to field maps. Geometry coordinates are in metres."
        )
        self.wire_current.setEnabled(has_wire_maps)
        self.wire_current.setVisible(has_wire_maps)
        self.slice_axis.setEnabled(is_field)
        self.slice_position.setEnabled(is_field and self.slice_axis.currentText() != "All points")
        self.slice_controls.setVisible(is_field)
        caption = "Coordinates [m]"
        mesh = None
        swept_count = 0
        try:
            support = (
                self._support
                if key in {"support", "preview_target"}
                else self._mesh(arrays["vertices_m"], arrays["faces"])
                if self._source_support_available
                else None
            )
            if support is not None and self.support_layer.isChecked() and key not in {"stream_function", "support"}:
                self._add(
                    support,
                    "geometry",
                    color="#79949d",
                    show_edges=self.edges.isChecked(),
                    opacity=0.04 if locating else 0.24 if key == "preview_target" else 0.15,
                )
            points_available = self._model_points_available if model else self._source_points_available
            if self.roi_layer.isChecked() and points_available and key not in {"target", "preview_target"}:
                if live and self._capability("target", comparison=False).available:
                    points = self._field_mesh(arrays, "target")
                    self._add(
                        points,
                        "target",
                        scalars=points.active_scalars_name,
                        cmap="coolwarm",
                        show_scalar_bar=False,
                        point_size=6,
                        opacity=0.7,
                        render_points_as_spheres=True,
                    )
                else:
                    points = self.pv.PolyData(np.asarray(arrays["target_points_m"], dtype=float))
                    self._add(points, "target", color="#c5965b", point_size=4, opacity=0.45)
            if key == "support":
                mesh = self._support if self.support_layer.isChecked() else None
                caption += " · support hidden" if hidden_support else " · current model support"
                options = {"color": "#79949d", "show_edges": self.edges.isChecked(), "opacity": 0.72}
            elif key == "stream_function" and {"vertices_m", "faces", "stream_function_A"} <= arrays.keys():
                mesh = self._mesh(arrays["vertices_m"], arrays["faces"])
                values = np.asarray(arrays["stream_function_A"], dtype=float).reshape(-1)
                if len(values) != mesh.n_points or not np.all(np.isfinite(values)):
                    raise ValueError("Stream function does not match the run mesh.")
                mesh["Stream function [A]"] = values
                options = {"scalars": "Stream function [A]", "cmap": "coolwarm", "show_edges": self.edges.isChecked()}
                caption += (
                    " · live continuous-sheet solution" if live else " · immutable run / continuous-sheet solution"
                )
            elif key == "surface_current_density" and available:
                mesh = self.pv.PolyData(np.asarray(arrays["sheet_current_density_points_m"], dtype=float))
                mesh["|K| [A/m]"] = np.linalg.norm(arrays["sheet_current_density_A_per_m"], axis=1)
                options = {"scalars": "|K| [A/m]", "cmap": "viridis", "point_size": 6, "render_points_as_spheres": True}
                caption += " · recorded face current density; continuous sheet; not copper volume density"
            elif is_field:
                mesh = self._field_mesh(arrays, key)
                options = {
                    "scalars": mesh.active_scalars_name if mesh is not None else None,
                    "cmap": "coolwarm",
                    "point_size": 8,
                    "render_points_as_spheres": True,
                }
                caption += " · " + self._field_caption(key)
            elif key == "swept":
                for name in sorted(arrays):
                    if not name.startswith("swept_") or not name.endswith("_vertices_m"):
                        continue
                    face_key = name.removesuffix("_vertices_m") + "_faces"
                    if face_key not in arrays:
                        raise ValueError(f"Saved swept conductor is missing triangular faces: {face_key}.")
                    body = self._mesh(arrays[name], arrays[face_key])
                    if not body.n_points or not body.n_cells:
                        continue
                    options = {
                        "color": "#205a75",
                        "show_edges": self.edges.isChecked(),
                        "opacity": 0.18 if locating else 1,
                    }
                    if self.clip.isChecked():
                        actor = widgets.add_mesh_clip_plane(body, reset_camera=False, **options)
                        if actor is not None:
                            self._actors[self._actor_key(actor)] = "winding"
                    else:
                        self._add(body, "winding", **options)
                    swept_count += 1
                caption += (
                    f" · immutable pipeline triangular sweep meshes · {swept_count} {'body' if swept_count == 1 else 'bodies'}"
                    " · inspection only; manufacturing acceptance requires separate checks"
                    if swept_count
                    else " · pipeline swept conductor unavailable in this run; centerlines remain separate datasets"
                )
            elif key in {
                "contours",
                "routes",
                "evaluated_winding",
                "fabrication",
                "review_conductor",
                "review_electrical",
            }:
                prefix = {
                    "contours": "contour_",
                    "routes": "route_",
                    "evaluated_winding": "evaluated_winding_",
                    "fabrication": "fabrication_",
                    "review_conductor": "review_conductor_",
                    "review_electrical": "review_electrical_",
                }[key]
                paths = [
                    value for name, value in arrays.items() if name.startswith(prefix) and name.endswith("_points_m")
                ]
                for path in paths:
                    path = np.asarray(path, dtype=float)
                    if path.ndim == 2 and path.shape[1] == 3 and len(path) > 1 and np.all(np.isfinite(path)):
                        self._add(
                            self.pv.lines_from_points(path),
                            "winding",
                            color="#205a75",
                            line_width=3,
                            opacity=0.18 if locating else 1,
                        )
                scope = (
                    "pipeline conductor"
                    if key == "fabrication"
                    else "fabrication-review conductor"
                    if key == "review_conductor"
                    else "review electrical field paths"
                    if key == "review_electrical"
                    else "evaluated winding field paths"
                    if key == "evaluated_winding"
                    else key
                )
                caption += f" · immutable {scope} centerlines" + (" unavailable at this stage" if not paths else "")
            if mesh is not None and mesh.n_points:
                if options.get("scalars") is not None:
                    if self._colour_limits is not None:
                        options["clim"] = self._colour_limits
                    elif key.endswith("_residual") and self.component.currentText() != "|B|":
                        limits = _signed_residual_limits(mesh.active_scalars)
                        if limits is not None:
                            options["clim"] = limits
                    options["scalar_bar_args"] = {
                        "vertical": True,
                        "position_x": 0.86,
                        "position_y": 0.2,
                        "width": 0.08,
                        "height": 0.6,
                        "title_font_size": 13,
                        "label_font_size": 11,
                    }
                self._dataset_mesh = mesh
                selection = "target" if is_field else "optimization" if key == "stream_function" else "geometry"
                if self.clip.isChecked():
                    actor = widgets.add_mesh_clip_plane(mesh, reset_camera=False, **options)
                else:
                    actor = self._add(mesh, selection, **options)
                if actor is not None:
                    self._dataset_actor_key = self._actor_key(actor)
                    self._actors[self._dataset_actor_key] = selection
            elif key not in {
                "contours",
                "routes",
                "evaluated_winding",
                "fabrication",
                "review_conductor",
                "review_electrical",
                "swept",
                "support",
            }:
                caption += " · dataset unavailable"
            if key in {"support", "preview_target"} and self._preview_error:
                caption += f" · {self._preview_error}"
            if is_field and mesh is not None and mesh.n_points:
                caption += (
                    f" · {mesh.n_points} {'model' if model else 'stage' if live else 'saved'} samples; no interpolation"
                    if self.slice_axis.currentText() == "All points"
                    else f" · displayed slab: {self.slice_label.text()}; no interpolation"
                )
            if self._located_issue is not None:
                issue = self._located_issue
                points = np.asarray(issue.points_m, dtype=float)
                marker = self.pv.PolyData(points)
                if issue.segments:
                    marker.lines = np.asarray([[2, *pair] for pair in issue.segments], dtype=np.int64).ravel()
                self._add(
                    marker, "winding", color="#e87525", line_width=7, point_size=15, render_points_as_spheres=True
                )
                caption += f" · highlighted: {issue.location_description}"
            if self._actors or widgets.plane_clipped_meshes:
                self.plotter.show_grid(
                    color="#666666",
                    xtitle="x [m]",
                    ytitle="y [m]",
                    ztitle="z [m]",
                    fmt="%.3g",
                    n_xlabels=2,
                    n_ylabels=2,
                    n_zlabels=2,
                    font_size=10,
                    bold=False,
                    grid=False,
                )
            if live:
                stage = str(self._stage_metadata.get("stage", self._stage_metadata.get("pipeline_stage", "")))
                stage = stage.replace("_", " ") or "stage output"
                caption = caption.replace("immutable", "stage").replace("saved samples", "stage samples")
                prefix = (
                    f"Run snapshot · {self._stage_execution_status}"
                    if self._stage_execution_status in {"cancelled", "failed", "completed"}
                    else "Live study preview"
                )
                caption = f"{prefix} · {stage} · " + caption + " · not a verified completed run; acceptance pending"
            elif not model and self._verification_error:
                caption = "Cached run · verification failed · " + caption + " · " + self._verification_error
            if self._source_notice:
                caption += " · " + self._source_notice
            self.caption.setText(caption)
            if self._first_view and (mesh is not None or support is not None or swept_count):
                self._fit_view()
                self._first_view = False
            else:
                self.plotter.camera_position = camera
            self.plotter.render()
        except Exception as exc:
            message = f"Display unavailable: {type(exc).__name__}: {exc}"
            self.caption.setText(message)
            self.error.emit(message)
        finally:
            # Scientific selection is sometimes restored under signal blockers
            # (witness location, clear highlight, saved view). Notify observers
            # only after the dataset, sampled slab and caption are finalized.
            self.context_changed.emit()

    def _fit_view(self, *_args):
        """Fit the scene, with useful orthographic fill and room for axis text."""
        self.plotter.reset_camera()
        camera = getattr(self.plotter, "camera", None)
        bounds = getattr(self.plotter, "bounds", None)
        size = getattr(self.plotter, "window_size", None)
        if camera is None or not self.orthographic.isChecked() or bounds is None or size is None:
            return
        bounds, size = np.asarray(bounds, dtype=float), np.asarray(size, dtype=float)
        if bounds.shape != (6,) or size.shape != (2,) or not np.all(np.isfinite(bounds)) or np.any(size <= 0):
            return
        direction, up = np.asarray(camera.direction), np.asarray(camera.up)
        right = np.cross(direction, up)
        length = np.linalg.norm(right)
        if not np.isfinite(length) or length <= 0:
            return
        right /= length
        up = np.cross(right, direction)
        corners = np.asarray([[x, y, z] for x in bounds[:2] for y in bounds[2:4] for z in bounds[4:]])
        scale = 1.25 * max(np.ptp(corners @ up), np.ptp(corners @ right) / (size[0] / size[1])) / 2
        if np.isfinite(scale) and 0 < scale < camera.parallel_scale:
            camera.parallel_scale = float(scale)
            self.plotter.render()

    def _set_view(self, *_args):
        preset = {
            "Isometric": self.plotter.view_isometric,
            "XY": self.plotter.view_xy,
            "XZ": self.plotter.view_xz,
            "YZ": self.plotter.view_yz,
        }.get(self.view.currentText())
        if preset is not None:
            preset()
            self._fit_view()

    def _set_projection(self, enabled):
        self.plotter.enable_parallel_projection() if enabled else self.plotter.disable_parallel_projection()

    def closeEvent(self, event):
        if not self._closed:
            self._closed = True
            self.inspection_menu.hide()
            self.layers_menu.hide()
            self.display_menu.hide()
            if self._comparison is not None:
                self._comparison.close()
            self._pending_preview = None
            if self._preview_worker is not None:
                self._preview_worker.signals.finished.disconnect(self._preview_finished)
                self._preview_worker = None
            self.plotter.close()
        super().closeEvent(event)


class ComparisonDialog(QDialog):
    """Two immutable runs with linked world-frame cameras and real colour limits."""

    def __init__(self, left, right, parent=None, *, viewport_factory=ScientificViewport):
        super().__init__(parent)
        self.setWindowTitle("Compare CoilGen runs")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        screen = self.screen()
        available = screen.availableGeometry() if screen is not None else None
        self.resize(
            min(1450, available.width()) if available else 1450, min(850, available.height()) if available else 850
        )
        self._syncing = False
        self._camera_rendering = False
        self._camera_observer = None
        self._closed = False
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.left = viewport_factory(self)
        self.right = viewport_factory(self)
        self.panes = (self.left, self.right)
        for record, pane in zip((left, right), self.panes):
            column = QVBoxLayout()
            label = QLabel(f"{record.metadata.get('name', 'Run')} · {Path(record.directory).name}")
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setWordWrap(True)
            column.addWidget(label)
            column.addWidget(pane, 1)
            row.addLayout(column, 1)
            arrays = record.select_arrays(copy=False) if hasattr(record, "select_arrays") else record.arrays
            pane.display_run(arrays, metadata=record.metadata)
        layout.addLayout(row, 1)
        footer = QHBoxLayout()
        self.scope = QLabel(
            "Only datasets available in both runs can be selected. Stored grids remain separate; no interpolation or current refitting. Camera, dataset, units, current and sample-slab controls are linked."
        )
        self.scope.setWordWrap(True)
        footer.addWidget(self.scope, 1)
        fit = QPushButton("Fit both")
        fit.clicked.connect(self.fit_both)
        footer.addWidget(fit)
        layout.addLayout(footer)
        self.range_label = QLabel()
        layout.addWidget(self.range_label)
        self._camera = getattr(self.left.plotter, "camera", None) or self.left.pv.Camera()
        self.left.plotter.camera = self._camera
        self.right.plotter.camera = self._camera
        self._camera_observer = self._camera.AddObserver("ModifiedEvent", self._camera_changed)
        for source, target in ((self.left, self.right), (self.right, self.left)):
            for name in ("dataset", "component", "units", "wire_current", "slice_axis", "view"):
                getattr(source, name).currentIndexChanged.connect(
                    lambda index, s=source, t=target, n=name: self._sync(s, t, n, index, "index")
                )
            for name in ("orthographic", "support_layer", "roi_layer", "edges", "clip"):
                getattr(source, name).toggled.connect(
                    lambda checked, s=source, t=target, n=name: self._sync(s, t, n, checked, "checked")
                )
            source.slice_position.valueChanged.connect(
                lambda value, s=source, t=target: self._sync(s, t, "slice_position", value, "value")
            )
        self.left._comparison_peer, self.right._comparison_peer = self.right, self.left
        for pane in self.panes:
            pane._refresh_availability()
        common = [
            self.left.dataset.itemData(index)
            for index in range(self.left.dataset.count())
            if self.left._capability(self.left.dataset.itemData(index)).available
        ]
        if common:
            selected = self.left.dataset.currentData()
            selected = selected if selected in common else common[0]
            with QSignalBlocker(self.left.dataset):
                self.left.dataset.setCurrentIndex(self.left.dataset.findData(selected))
            self.left._render()
            self._sync(self.left, self.right, "dataset", self.left.dataset.currentIndex(), "index")
        else:
            self.scope.setText(
                "No dataset is available in both runs at the selected current policy. Each pane retains its available evidence; no shared field comparison is shown."
            )
        self.fit_both()

    def _sync(self, source, target, name, value, kind):
        if self._syncing or self._closed:
            return
        self._syncing = True
        try:
            widget = getattr(target, name)
            if kind == "index":
                value = getattr(source, name).currentIndex()
                if name == "dataset" and not target._capability(widget.itemData(value)).available:
                    return
                if name == "wire_current" and target.dataset.currentData() in WIRE_FIELDS:
                    if not target._capability(target.dataset.currentData(), original=value == 1).available:
                        return
            with QSignalBlocker(widget):
                if kind == "index":
                    widget.setCurrentIndex(value)
                    if name == "wire_current":
                        target._last_wire_current = value
                elif kind == "checked":
                    widget.setChecked(value)
                else:
                    widget.setValue(value)
            target._render()
            self._shared_range()
        finally:
            self._syncing = False

    def _shared_range(self):
        if self.left.dataset.currentData() != self.right.dataset.currentData():
            self.range_label.setText("No matching dataset is displayed in both panes.")
            return
        meshes = [pane._dataset_mesh for pane in self.panes]
        names = [mesh.active_scalars_name for mesh in meshes if mesh is not None and mesh.active_scalars is not None]
        values = [
            np.asarray(mesh.active_scalars) for mesh in meshes if mesh is not None and mesh.active_scalars is not None
        ]
        if values and len(set(names)) == 1:
            finite = np.concatenate([value[np.isfinite(value)] for value in values])
            if len(finite):
                lo, hi = float(np.min(finite)), float(np.max(finite))
                if self.left.dataset.currentData().endswith("_residual") and self.left.component.currentText() != "|B|":
                    lo, hi = -max(abs(lo), abs(hi)), max(abs(lo), abs(hi))
                if lo == hi:
                    padding = max(abs(lo) * 1e-6, 1e-12)
                    lo, hi = lo - padding, hi + padding
                for pane in self.panes:
                    pane._colour_limits = (lo, hi)
                    pane._render()
                missing = " · one pane has no matching scalar dataset" if len(values) != 2 else ""
                self.range_label.setText(f"Shared {names[0]} colour range: {lo:.6g} to {hi:.6g}{missing}")
                return
        for pane in self.panes:
            pane._colour_limits = None
            pane._render()
        self.range_label.setText("No shared scalar colour range is available for this dataset.")

    def _camera_changed(self, *_args):
        if self._closed or self._camera_rendering:
            return
        self._camera_rendering = True
        try:
            for pane in self.panes:
                pane.plotter.render()
        finally:
            self._camera_rendering = False

    def fit_both(self):
        points = []
        for pane in self.panes:
            for key in ("vertices_m", "target_points_m"):
                if key in pane._arrays:
                    values = np.asarray(pane._arrays[key], dtype=float)
                    if values.ndim == 2 and values.shape[1:] == (3,) and len(values) and np.all(np.isfinite(values)):
                        points.append(values)
        if points:
            points = np.vstack(points)
            lo, hi = points.min(axis=0), points.max(axis=0)
            bounds = tuple(value for axis in range(3) for value in (lo[axis], hi[axis]))
            self.left.plotter.reset_camera(bounds=bounds)
        self._camera_changed()

    def closeEvent(self, event):
        if not self._closed:
            self._closed = True
            if self._camera_observer is not None:
                self._camera.RemoveObserver(self._camera_observer)
                self._camera_observer = None
            for pane in self.panes:
                pane.close()
        super().closeEvent(event)


class UnavailableViewport(QWidget):
    """Keep project editing usable when optional rendering is unavailable."""

    selected = Signal(str)
    preflight_ready = Signal(dict)

    def __init__(self, error, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel(
            "Scientific viewport unavailable.\n\nInstall the optional GUI dependencies:\n"
            'python -m pip install -e ".[gui]"\n\n'
            f"Dependency error: {error}"
        )
        label.setWordWrap(True)
        layout.addWidget(label)

    def render_config(self, *_args, **_kwargs):
        pass

    def display_run(self, *_args, **_kwargs):
        pass

    def clear_run(self):
        pass

    def clear_model(self):
        pass

    def view_state(self):
        return {}

    def restore_view_state(self, _state):
        return False

    def locate_issue(self, _issue):
        return False

    def display_comparison(self, *_args):
        return None


def create_viewport(parent=None):
    try:
        return ScientificViewport(parent)
    except (ImportError, RuntimeError) as exc:
        return UnavailableViewport(exc, parent)
