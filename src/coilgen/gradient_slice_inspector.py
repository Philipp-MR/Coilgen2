"""Interactive display of an already evaluated gradient-slice dataset."""

from __future__ import annotations

from typing import Any

import numpy as np


class GradientSliceInspector:
    """Keep controls and artists alive; changing views never reevaluates fields."""

    def __init__(self, figure: Any, evaluated: Any) -> None:
        from matplotlib.widgets import CheckButtons, RadioButtons, Slider
        from . import matplotlib_visualization as plotting

        self.figure = figure
        self.evaluated = evaluated
        self.components = tuple(evaluated.component_units)
        if not evaluated.slices or not self.components:
            raise ValueError("Gradient slice inspector has no evaluated components or slices.")
        self.component = "Bz" if "Bz" in self.components else self.components[0]
        self.slice_index = len(evaluated.slices) // 2
        self.common_scale = True
        self.engineering_units = True
        self._last_pixel = None
        self._outline_artists: list[Any] = []
        colors = plotting._import_matplotlib_colors()
        self._norms = {
            component: plotting._gradient_slice_component_norm(
                colors, [np.where(item.mask, item.maps[component], np.nan) for item in evaluated.slices], component
            )
            for component in self.components
        }
        self.ax = figure.add_axes([0.09, 0.23, 0.52, 0.61])
        self.colorbar_ax = figure.add_axes([0.65, 0.23, 0.023, 0.61])
        self.image = self.ax.imshow(np.zeros((2, 2)), origin="lower", interpolation="nearest", aspect="equal")
        self.colorbar = figure.colorbar(self.image, cax=self.colorbar_ax)
        self.title = figure.text(0.09, 0.95, "", fontsize=13, fontweight="bold")
        self.metadata = figure.text(0.09, 0.90, "", fontsize=9, color="#475569")
        self.readout = figure.text(0.09, 0.16, "Move over the map to inspect a pixel.", fontsize=9, color="#334155")
        self.slider = Slider(
            figure.add_axes([0.12, 0.085, 0.49, 0.025]),
            "Slice",
            1,
            max(len(evaluated.slices), 2),
            valinit=self.slice_index + 1,
            valstep=1,
            valfmt="%d",
        )
        if len(evaluated.slices) == 1:
            self.slider.set_active(False)
        component_labels = [plotting._gradient_slice_tile_component_label(component) for component in self.components]
        self._component_keys = dict(zip(component_labels, self.components))
        self.component_buttons = RadioButtons(
            figure.add_axes([0.77, 0.37, 0.22, 0.48]),
            component_labels,
            active=self.components.index(self.component),
        )
        self.scale_button = CheckButtons(figure.add_axes([0.77, 0.25, 0.22, 0.08]), ["Common slice scale"], [True])
        self.unit_buttons = RadioButtons(
            figure.add_axes([0.77, 0.105, 0.22, 0.11]), ["Engineering units", "SI units"], active=0
        )
        for widget in (self.component_buttons, self.scale_button, self.unit_buttons):
            for label in widget.labels:
                label.set_fontsize(8.5)
            for spine in widget.ax.spines.values():
                spine.set_visible(False)
        figure.text(
            0.09,
            0.025,
            "All controls use cached fields. Full mosaics remain available as separate pages.",
            fontsize=8,
            color="#64748b",
        )
        self.slider.on_changed(lambda value: self.set_slice(int(value) - 1))
        self.component_buttons.on_clicked(lambda label: self.set_component(self._component_keys[label]))
        self.scale_button.on_clicked(self._set_scale)
        self.unit_buttons.on_clicked(self._set_units)
        self._motion_id = figure.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._close_id = figure.canvas.mpl_connect("close_event", lambda _event: self.disconnect())
        self.update(draw=False)

    def _display_units(self) -> tuple[float, str]:
        unit = self.evaluated.component_units[self.component]
        if self.engineering_units:
            return {"T": (1e6, "µT"), "T/m": (1e3, "mT/m")}.get(unit, (1.0, unit))
        return 1.0, unit

    def set_slice(self, index: int) -> None:
        self.slice_index = min(max(int(index), 0), len(self.evaluated.slices) - 1)
        self.update()

    def set_component(self, component: str) -> None:
        if component not in self.components:
            raise ValueError(f"Unknown gradient-slice component: {component}")
        self.component = component
        self.update()

    def _set_scale(self, _label: str) -> None:
        self.common_scale = self.scale_button.get_status()[0]
        self.update()

    def _set_units(self, label: str) -> None:
        self.engineering_units = label == "Engineering units"
        self.update()

    def update(self, *, draw: bool = True) -> None:
        from matplotlib.colors import Normalize
        from matplotlib.ticker import MaxNLocator
        from . import matplotlib_visualization as plotting

        item = self.evaluated.slices[self.slice_index]
        values = np.where(item.mask, np.asarray(item.maps[self.component], dtype=float), np.nan)
        factor, unit = self._display_units()
        norm = (
            self._norms[self.component]
            if self.common_scale
            else plotting._gradient_slice_component_norm(plotting._import_matplotlib_colors(), [values], self.component)
        )
        self.image.set_data(np.ma.masked_invalid(values * factor))
        extent = plotting._gradient_slice_pixel_extent(item)
        self.image.set_extent(extent)
        self.image.set_norm(Normalize(norm.vmin * factor, norm.vmax * factor))
        self.image.set_cmap(plotting._gradient_slice_component_cmap(plotting._import_pyplot(), self.component))
        self.colorbar.update_normal(self.image)
        plotting._set_scientific_colorbar_ticks(self.colorbar, norm.vmin * factor, norm.vmax * factor)
        self.colorbar.set_label(f"{plotting._gradient_slice_tile_component_label(self.component)} [{unit}]", fontsize=9)
        for artist in self._outline_artists:
            artist.remove()
        before = set(self.ax.lines)
        plotting._draw_gradient_slice_dsv_outline(self.ax, item)
        self._outline_artists = [artist for artist in self.ax.lines if artist not in before]
        self.ax.set_xlim(extent[:2])
        self.ax.set_ylim(extent[2:])
        for axis, dimension in ((self.ax.xaxis, item.plot_axes[0]), (self.ax.yaxis, item.plot_axes[1])):
            axis.set_major_locator(MaxNLocator(5))
            axis.set_label_text(f"{plotting.AXIS_NAMES[dimension]} [m]")
        label = plotting._gradient_slice_tile_component_label(self.component)
        self.title.set_text(f"{label}  ·  {self.evaluated.axis_name} = {item.value:.5g} m")
        model = " · ideal linear-gradient concomitant estimate" if self.component.startswith("concomitant") else ""
        spec = self.evaluated.spec
        self.metadata.set_text(
            f"Source: {spec.source} · current: {spec.current} ({self.evaluated.current_scale_A:.5g} A) · "
            f"domain: {spec.domain}\nSlice {self.slice_index + 1} / {len(self.evaluated.slices)} · "
            f"{'common scale across slices' if self.common_scale else 'individual slice scale'}{model}"
        )
        self._last_pixel = None
        self.readout.set_text("Move over the map to inspect a pixel.")
        self.ax.format_coord = self._format_coord
        if draw:
            self.figure.canvas.draw_idle()

    def _format_coord(self, x: float, y: float) -> str:
        from .matplotlib_visualization import _gradient_slice_pixel_extent

        item = self.evaluated.slices[self.slice_index]
        left, right, bottom, top = _gradient_slice_pixel_extent(item)
        if not (left <= x <= right and bottom <= y <= top):
            return "Outside the evaluated slice grid"
        column = int(np.argmin(np.abs(np.asarray(item.x) - x)))
        row = int(np.argmin(np.abs(np.asarray(item.y) - y)))
        factor, unit = self._display_units()
        value = (
            float(np.asarray(item.maps[self.component])[row, column]) * factor
            if item.mask[row, column]
            else float("nan")
        )
        value_label = f"{value:.6g} {unit}" if np.isfinite(value) else "outside valid domain"
        names = ("x", "y", "z")
        return (
            f"{names[item.plot_axes[0]]}={item.x[column]:.5g} m, "
            f"{names[item.plot_axes[1]]}={item.y[row]:.5g} m, "
            f"{names[item.axis]}={item.value:.5g} m: {value_label}"
        )

    def _on_motion(self, event: Any) -> None:
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        label = self._format_coord(event.xdata, event.ydata)
        if label != self._last_pixel:
            self._last_pixel = label
            self.readout.set_text(label)
            self.figure.canvas.draw_idle()

    def disconnect(self) -> None:
        for widget in (self.slider, self.component_buttons, self.scale_button, self.unit_buttons):
            widget.disconnect_events()
        self.figure.canvas.mpl_disconnect(self._motion_id)
        self.figure.canvas.mpl_disconnect(self._close_id)
