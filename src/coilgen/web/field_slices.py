"""Run-bound browser views of progressively sampled residual slices."""

import asyncio
from base64 import b64encode
from io import BytesIO
from threading import Event
from time import monotonic

import numpy as np

from ..application.field_presentation import field_provenance, paired_color_limits, plot_field_source
from ..field_components import field_component_values
from ..progress import DesignCancelled


PLANES = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}
COMPONENTS = {"Bz": "Bz", "Bx": "Bx", "By": "By", "BdotB0": "B·B₀", "Bmag": "Vector difference |ΔB|"}


def _plot_slice(collection, component, b0, field_limits, residual_limits):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    piece = collection.slices[-1]
    h, v, normal = PLANES[collection.plane]
    xyz = piece.coordinates_m * 1000

    def scalar(field):
        values = field_component_values(field.reshape(-1, 3), component, b0_direction=b0)
        return values.reshape(xyz.shape[:2]) * 1e6

    paired = piece.available_mask & piece.target_valid_mask
    field = np.where(paired, scalar(piece.field_T), np.nan)
    target = np.full_like(field, np.nan)
    residual = np.full_like(field, np.nan)
    if piece.target_T is not None:
        target = np.where(paired, scalar(piece.target_T), np.nan)
        residual = np.where(paired, scalar(piece.field_T - piece.target_T), np.nan)

    figure = Figure(figsize=(8.8, 3.15), dpi=110, layout="constrained")
    axes = figure.subplots(1, 3, sharex=True, sharey=True)
    for axis, title, values, limits in zip(
        axes,
        ("Target", "Field", "Residual"),
        (target, field, residual),
        (field_limits, field_limits, residual_limits),
        strict=True,
    ):
        chart = axis.pcolormesh(
            xyz[:, :, h],
            xyz[:, :, v],
            values,
            shading="nearest",
            cmap="viridis" if component == "Bmag" else "coolwarm",
            vmin=limits[0],
            vmax=limits[1],
        )
        for line in piece.outlines_m:
            axis.plot(line[:, h] * 1000, line[:, v] * 1000, color="#303c44", linewidth=0.55)
        axis.set(title=title, xlabel=f"{'XYZ'[h]} [mm]", aspect="equal")
        if title != "Target":
            color_axes = axes[:2] if title == "Field" else axis
            figure.colorbar(chart, ax=color_axes, label="µT", shrink=0.7)
    axes[0].set_ylabel(f"{'XYZ'[v]} [mm]")
    buffer = BytesIO()
    FigureCanvasAgg(figure).print_png(buffer)
    return {
        "number": len(collection.slices),
        "title": f"{'XYZ'[normal]} = {piece.offset_m * 1000:.5g} mm",
        "image": "data:image/png;base64," + b64encode(buffer.getvalue()).decode("ascii"),
        "scope": f"{np.count_nonzero(paired):,}/{paired.size:,} paired pixels · "
        f"Field: {collection.representation.replace('_', ' ')} · "
        f"Target: {collection.target_representation.replace('_', ' ')}",
    }


def _sample(record, cache, dataset, plane, resolution, component, cancel, report, publish):
    from ..application.field_analysis import owned_roi_bounds
    from ..application.field_sampling import sample_slices

    bounds = owned_roi_bounds(dataset.target_geometry)
    if bounds is None:
        bounds = np.stack((dataset.points_m.min(axis=0), dataset.points_m.max(axis=0)))
    h, v, normal = PLANES[plane]
    centre = bounds.mean(axis=0)
    span = np.ptp(bounds, axis=0)
    size = np.maximum(span[[h, v]], 0.001)
    offsets = centre[normal] + (span[normal] * np.array([-0.25, 0, 0.25]) if span[normal] > 1e-12 else np.array([0]))
    b0 = dataset.provenance.get("b0_direction")
    field = field_component_values(dataset.field_T, component, b0_direction=b0) * 1e6
    desired = field_component_values(dataset.target_T, component, b0_direction=b0) * 1e6
    error = field_component_values(dataset.field_T - dataset.target_T, component, b0_direction=b0) * 1e6
    limits = paired_color_limits(field, desired, signed=component != "Bmag")
    error_limits = paired_color_limits(error, None, signed=component != "Bmag")
    return sample_slices(
        record,
        dataset,
        cache=cache,
        plane=plane,
        offsets_m=offsets,
        centre_m=centre,
        size_m=size,
        resolution=resolution,
        progress=report,
        cancel_check=cancel.is_set,
        on_slice=lambda collection: publish(_plot_slice(collection, component, b0, limits, error_limits), len(offsets)),
    )


class BrowserFieldSlices:
    def __init__(self, workbench):
        self.workbench = workbench
        self.state = state = workbench.server.state
        self.cancel = Event()
        self.catalog = {}
        state.update(
            {
                "field_slices_open": False,
                "slice_sources": [],
                "slice_source": "",
                "slice_plane": "xy",
                "slice_component": "Bz",
                "slice_resolution": 16,
                "slice_panels": [],
                "slice_note": "",
                "slice_busy": False,
                "slice_percent": None,
                "slice_header": "",
                "slice_total": 0,
                "slice_applied": None,
            }
        )
        state.change("field_slices_open")(self._visibility_changed)
        controller = workbench.server.controller
        controller.open_field_slices = self.open
        controller.evaluate_field_slices = self.evaluate
        controller.cancel_field_slices = self.cancel_work

    def _visibility_changed(self, field_slices_open, **_):
        if not field_slices_open:
            self.cancel_work()

    def cancel_work(self):
        self.cancel.set()
        if self.state.slice_busy:
            self.state.slice_busy = False
            self.state.slice_note = "Stopped. Completed slices are retained."

    def clear(self):
        self.cancel_work()
        self.catalog = {}
        self.state.field_slices_open = False
        self.state.slice_panels = []
        self.state.slice_applied = None

    async def open(self):
        record, state = self.workbench.record, self.state
        if record is None or state.selected_run != state.displayed_run:
            return
        self.clear()
        # The dialog's model stays true for this entire preparation/evaluation.
        self.cancel = token = Event()
        state.field_slices_open = True
        state.slice_busy = True
        state.slice_sources = []
        state.slice_header = ""
        state.slice_percent = None
        state.slice_note = "Verify saved field sources…"
        state.flush()
        try:
            sources = await asyncio.to_thread(self.workbench.field_inputs.catalog, record, cancel_check=token.is_set)
        except asyncio.CancelledError:
            token.set()
            raise
        except (DesignCancelled, OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            if not token.is_set():
                state.slice_busy, state.slice_note = False, str(exc)
            return
        if token.is_set() or record is not self.workbench.record:
            return
        self.catalog = {
            source.key: source for source in sources if source.key != "target" and source.target_T is not None
        }
        state.slice_sources = [
            {
                "title": field_provenance(item.key, record.metadata_view, len(item.points_m), evidence=item.provenance),
                "value": item.key,
            }
            for item in self.catalog.values()
        ]
        if not self.catalog:
            state.slice_busy = False
            state.slice_note = "This run has no saved field with a paired target."
            return
        preferred = plot_field_source(state.dataset)
        state.slice_source = (
            preferred
            if preferred in self.catalog
            else ("fabricated" if "fabricated" in self.catalog else next(iter(self.catalog)))
        )
        state.slice_component = state.component if state.component in COMPONENTS else "Bz"
        await self.evaluate()

    async def evaluate(self):
        record, state = self.workbench.record, self.state
        dataset = self.catalog.get(state.slice_source)
        if record is None or dataset is None or not state.field_slices_open:
            return
        plane, component, resolution = state.slice_plane, state.slice_component, state.slice_resolution
        if plane not in PLANES or component not in COMPONENTS or resolution not in {12, 16, 24, 32}:
            state.slice_note = "Choose a supported plane, component and grid size."
            return
        self.cancel.set()
        self.cancel = token = Event()
        state.slice_panels, state.slice_total, state.slice_percent = [], 0, None
        state.slice_applied = {"source": dataset.key, "plane": plane, "component": component, "resolution": resolution}
        state.slice_header = field_provenance(
            dataset.key, record.metadata_view, len(dataset.points_m), evidence=dataset.provenance
        )
        state.slice_header += (
            " · Vector difference includes transverse components"
            if component == "Bmag"
            else f" · Signed {COMPONENTS[component]} residual (field minus target)"
        )
        state.slice_busy, state.slice_note = True, "Prepare slice inputs…"
        state.flush()
        loop = asyncio.get_running_loop()
        last_progress = 0.0

        def current():
            return not token.is_set() and self.cancel is token and record is self.workbench.record

        def show_progress(percent, message):
            if current():
                with state:
                    state.slice_percent, state.slice_note = percent, message

        def report(percent, message):
            nonlocal last_progress
            now = monotonic()
            if now - last_progress >= 0.15:
                last_progress = now
                loop.call_soon_threadsafe(show_progress, percent, message)

        def append(panel, total):
            if current():
                with state:
                    state.slice_panels = [*state.slice_panels, panel]
                    state.slice_total = total

        try:
            result = await asyncio.to_thread(
                _sample,
                record,
                self.workbench.field_inputs,
                dataset,
                plane,
                resolution,
                component,
                token,
                report,
                lambda panel, total: loop.call_soon_threadsafe(append, panel, total),
            )
            if current():
                state.slice_note = f"{len(result.slices)} slices completed."
                state.slice_percent = 100
        except asyncio.CancelledError:
            token.set()
            raise
        except (DesignCancelled, OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            if current():
                state.slice_note = f"Slices unavailable. {exc}"
        finally:
            if current():
                state.slice_busy = False


def build_field_slices_dialog(ctrl):
    from trame.widgets import html, vuetify3 as v3

    with v3.VDialog(
        v_model=("field_slices_open",), max_width=1060, scrollable=True, raw_attrs=['aria-label="Residual slices"']
    ):
        with v3.VCard():
            with html.Div(classes="cg-dialog-head"):
                html.H2("Residual slices")
                v3.VBtn("Close", variant="outlined", size="small", click="field_slices_open = false")
            with v3.VCardText(classes="cg-dialog-body"):
                html.P("{{ slice_header }}", v_if="slice_header", classes="cg-detail-note")
                with html.Div(v_if="slice_sources.length", classes="cg-buttons"):
                    v3.VSelect(
                        v_model=("slice_source",),
                        items=("slice_sources",),
                        label="Field source",
                        density="compact",
                        hide_details=True,
                        disabled=("slice_busy",),
                    )
                    for name, title, choices in (
                        ("slice_plane", "Plane", tuple(PLANES)),
                        (
                            "slice_component",
                            "Component",
                            tuple({"title": v, "value": k} for k, v in COMPONENTS.items()),
                        ),
                        ("slice_resolution", "Grid per axis", (12, 16, 24, 32)),
                    ):
                        v3.VSelect(
                            v_model=(name,),
                            items=(list(choices),),
                            label=title,
                            density="compact",
                            hide_details=True,
                            disabled=("slice_busy",),
                        )
                    v3.VBtn(
                        "Evaluate slices", click=ctrl.evaluate_field_slices, disabled=("slice_busy",), color="primary"
                    )
                html.P(
                    "Target and Field share a scale from the saved samples. The residual uses its own scale. "
                    "Colors saturate beyond the scale. Blank pixels have no paired field and target.",
                    classes="cg-detail-note",
                )
                html.P(
                    "The plots show the last evaluated settings. Choose Evaluate slices to apply these changes.",
                    v_if="slice_applied && (slice_source !== slice_applied.source || slice_plane !== slice_applied.plane "
                    "|| slice_component !== slice_applied.component || slice_resolution !== slice_applied.resolution)",
                    role="status",
                )
                with html.Div(classes="cg-buttons"):
                    html.P("{{ slice_note }}", role="status")
                    v3.VBtn("Stop", v_if="slice_busy", variant="outlined", click=ctrl.cancel_field_slices)
                v3.VProgressLinear(
                    v_if="slice_busy",
                    model_value=("slice_percent ?? 0",),
                    indeterminate=("slice_percent === null",),
                    raw_attrs=['aria-label="Slice progress"'],
                )
                with html.Div(v_for="panel in slice_panels", key="panel.number"):
                    html.H3("Slice {{ panel.number }}/{{ slice_total }} · {{ panel.title }}")
                    html.P("{{ panel.scope }}", classes="cg-detail-note")
                    html.Img(
                        src=("panel.image",),
                        alt=("'Target, field and residual at ' + panel.title",),
                        style="width: 100%; height: auto;",
                    )
