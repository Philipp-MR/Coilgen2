"""Run-bound browser adapter for saved-field solid harmonic analysis."""

import asyncio
import math
from threading import Event

import numpy as np
from trame.widgets import html, vuetify3 as v3

from ..application.field_analysis import analyze_harmonics, field_analysis_available, owned_roi_bounds
from ..application.field_presentation import field_provenance, plot_field_source
from ..field_components import normalized_b0_direction


_COMPONENTS = {"Bx": "Bx · world x", "By": "By · world y", "Bz": "Bz · world z", "Bparallel": "B · B̂₀"}
_FIT_VIEWS = {"residual": "Residual", "field": "Field", "target": "Target"}
_PREFERRED = ("winding_validation", "fabricated", "routed_at_stream_current", "routed", "validation", "sheet")


def harmonic_reference(dataset):
    """Default to the authored world ROI, with an explicit sample-bounds fallback."""
    geometry = dataset.target_geometry
    bounds = owned_roi_bounds(geometry)
    authored = bounds is not None
    if not authored:
        bounds = np.stack((dataset.points_m.min(axis=0), dataset.points_m.max(axis=0)))
    center = (bounds[0] + bounds[1]) / 2
    if authored and geometry["region"]["shape"] == "sphere":
        radius = geometry["region"]["radius_m"]
    else:
        radius = max(float(np.linalg.norm((bounds[1] - bounds[0]) / 2)), 1e-6)
    return center, radius, "Saved ROI" if authored else "Saved sample bounds"


def harmonic_report(dataset, metadata, **controls):
    """Keep signed coefficients and physical per-term RMS in the same fit evidence."""
    result = analyze_harmonics(dataset, **controls)
    fits = result["fits"]
    views = []
    for key, name in _FIT_VIEWS.items():
        if key not in fits:
            continue
        fit = fits[key]
        views.append(
            {
                "value": key,
                "title": name,
                "sample_rms": f"{fit.field_rms * 1e6:.5g}",
                "fit_error_rms": f"{fit.residual_rms * 1e6:.5g}",
                "rows": [
                    {
                        "index": row["index"],
                        "term": row["term_label"],
                        "coefficient": f"{row['coefficient'] * 1e6:+.5g}",
                        "rms": f"{row['contribution_rms'] * 1e6:.5g}",
                    }
                    for row in result["coefficient_rows"][key]
                ],
            }
        )
    condition = result["condition"]
    center = ", ".join(f"{value * 1000:.7g}" for value in result["center_m"])
    return {
        "views": views,
        "source": field_provenance(dataset.key, metadata, len(dataset.points_m), evidence=dataset.provenance),
        "component": _COMPONENTS[result["component"]],
        "degree": result["max_order"],
        "center": f"({center}) mm",
        "radius": f"{result['radius_m'] * 1000:.7g} mm",
        "rank": f"{result['rank']}/{result['term_count']}",
        "condition": f"{condition:.4g}" if math.isfinite(condition) else "Rank deficient",
        "count": f"{result['sample_count']:,}",
        "sampling": result["sampling_convention"],
        "scope": result["sample_scope"],
        "diagnostics": result["diagnostics"],
    }


def _fit_numbers(degree, center_mm, radius_mm):
    try:
        degree_value = float(degree)
        if isinstance(degree, bool) or not degree_value.is_integer() or not 0 <= degree_value <= 12:
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Maximum degree must be a whole number from 0 to 12.") from exc
    try:
        center = tuple(float(value) / 1000 for value in str(center_mm).replace(",", " ").split())
        if len(center) != 3 or not all(math.isfinite(value) for value in center):
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Enter three finite world coordinates in millimetres for the expansion center.") from exc
    try:
        radius = float(radius_mm) / 1000
        if isinstance(radius_mm, bool) or not math.isfinite(radius) or radius <= 0:
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Reference radius must be a positive finite value in millimetres.") from exc
    return int(degree_value), center, radius


class BrowserHarmonics:
    """Hold one saved run's fit and discard late results when the selection changes."""

    def __init__(self, workbench):
        self.workbench = workbench
        self.state = state = workbench.server.state
        self.record = None
        self.sources = {}
        self.selected_source = ""
        self.cancel = Event()
        self.lock = asyncio.Lock()
        state.update(
            {
                "harmonics_available": False,
                "harmonics_open": False,
                "harmonics_status": "",
                "harmonics_error": "",
                "harmonics_sources": [],
                "harmonics_source": "",
                "harmonics_components": [],
                "harmonics_component": "Bz",
                "harmonics_degree": 4,
                "harmonics_center_mm": "",
                "harmonics_radius_mm": None,
                "harmonics_reference": "",
                "harmonics_has_weights": False,
                "harmonics_weighted": False,
                "harmonics_report": None,
                "harmonics_view": "",
            }
        )
        state.change("harmonics_open")(self._dialog_changed)
        state.change("harmonics_source")(self._source_changed)
        state.change(
            "harmonics_component",
            "harmonics_degree",
            "harmonics_center_mm",
            "harmonics_radius_mm",
            "harmonics_weighted",
        )(self._controls_changed)
        workbench.server.controller.open_harmonics = self.open
        workbench.server.controller.fit_harmonics = self.fit
        workbench.server.controller.cancel_harmonics = self.stop

    def set_record(self, record):
        self.cancel.set()
        self.record = record
        self.sources = {}
        self.selected_source = ""
        state = self.state
        state.harmonics_available = field_analysis_available(record)
        state.harmonics_open = False
        state.harmonics_status = state.harmonics_error = ""
        state.harmonics_report = None

    def _dialog_changed(self, harmonics_open, **_):
        if not harmonics_open:
            self.cancel.set()

    def _controls_changed(self, **_):
        self.state.harmonics_report = None

    def _source_changed(self, harmonics_source, **_):
        dataset = self.sources.get(harmonics_source) if isinstance(harmonics_source, str) else None
        if dataset is None or harmonics_source == self.selected_source:
            return
        previous = self.sources.get(self.selected_source)
        self.selected_source = harmonics_source
        state = self.state
        state.harmonics_report = None
        state.harmonics_has_weights = dataset.weights is not None and bool(np.any(dataset.weights > 0))
        if not state.harmonics_has_weights:
            state.harmonics_weighted = False
        if previous is None or dataset.target_geometry is None or previous.target_geometry != dataset.target_geometry:
            center, radius, origin = harmonic_reference(dataset)
            state.harmonics_center_mm = ", ".join(f"{value * 1000:.10g}" for value in center)
            state.harmonics_radius_mm = radius * 1000
            state.harmonics_reference = origin

    def _begin(self, message):
        self.cancel.set()
        cancel = self.cancel = Event()
        self.state.harmonics_status = message
        self.state.harmonics_error = ""
        self.state.flush()
        return cancel

    def _current(self, record, run, cancel):
        state = self.state
        return (
            record is self.record
            and run == state.selected_run == state.displayed_run
            and cancel is self.cancel
            and state.harmonics_open
        )

    def _b0(self, record):
        try:
            return normalized_b0_direction((record.metadata_view.get("config") or {})["b_0_direction"])
        except (KeyError, TypeError, ValueError):
            return None

    async def open(self):
        state, record = self.state, self.record
        run = state.displayed_run
        if not state.harmonics_available or state.selected_run != run or record is None:
            return
        state.harmonics_open = True
        state.harmonics_sources = []
        state.harmonics_report = None
        self.selected_source = ""
        cancel = self._begin("Loading saved field sources…")
        try:
            async with self.lock:
                datasets = await asyncio.to_thread(
                    self.workbench.field_inputs.catalog, record, cancel_check=cancel.is_set
                )
            if not self._current(record, run, cancel) or cancel.is_set():
                return
            self.sources = {dataset.key: dataset for dataset in datasets if dataset.key != "target"}
            if not self.sources:
                raise ValueError("This run has no saved computed field suitable for a harmonic fit.")
            state.harmonics_sources = [{"value": key, "title": dataset.label} for key, dataset in self.sources.items()]
            plotted = plot_field_source(state.dataset)
            source = (
                plotted
                if plotted in self.sources
                else next((key for key in _PREFERRED if key in self.sources), next(iter(self.sources)))
            )
            components = {
                key: label for key, label in _COMPONENTS.items() if key != "Bparallel" or self._b0(record) is not None
            }
            component = (
                state.component
                if plotted in self.sources
                else (record.metadata_view.get("config") or {}).get("target_field_component", state.component)
            )
            selected_component = "Bparallel" if component == "BdotB0" else component
            state.harmonics_components = [{"value": key, "title": label} for key, label in components.items()]
            state.harmonics_component = selected_component if selected_component in components else "Bz"
            state.harmonics_source = source
            self._source_changed(source)
        except (InterruptedError, OSError, RuntimeError, TypeError, ValueError) as exc:
            if self._current(record, run, cancel) and not cancel.is_set():
                state.harmonics_error = str(exc)
        finally:
            if self._current(record, run, cancel):
                state.harmonics_status = ""
                state.flush()

    async def fit(self, source, component, degree, center_mm, radius_mm, weighted):
        state, record = self.state, self.record
        run = state.displayed_run
        if record is None or state.selected_run != run or not state.harmonics_open or state.harmonics_status:
            return
        dataset = self.sources.get(source) if isinstance(source, str) else None
        if (
            dataset is None
            or not isinstance(component, str)
            or component not in _COMPONENTS
            or component == "Bparallel"
            and self._b0(record) is None
        ):
            state.harmonics_error = "Choose a saved field source and a recorded signed component."
            state.flush()
            return
        try:
            order, center, radius = _fit_numbers(degree, center_mm, radius_mm)
        except ValueError as exc:
            state.harmonics_error = str(exc)
            state.flush()
            return
        state.harmonics_report = None
        cancel = self._begin("Fitting solid harmonics to the saved samples…")
        try:
            async with self.lock:
                result = await asyncio.to_thread(
                    harmonic_report,
                    dataset,
                    record.metadata_view,
                    component=component,
                    max_order=order,
                    center_m=center,
                    radius_m=radius,
                    sampling="weighted" if weighted is True else "equal",
                    b0_direction=self._b0(record) if component == "Bparallel" else None,
                    cancel_check=cancel.is_set,
                )
            if self._current(record, run, cancel) and not cancel.is_set():
                state.harmonics_report = result
                state.harmonics_view = result["views"][0]["value"]
        except (InterruptedError, OSError, RuntimeError, TypeError, ValueError) as exc:
            if self._current(record, run, cancel) and not cancel.is_set():
                state.harmonics_error = str(exc)
        finally:
            if self._current(record, run, cancel):
                state.harmonics_status = ""
                state.flush()

    def stop(self):
        if self.state.harmonics_status:
            self.cancel.set()
            self.state.harmonics_status = "Stopping harmonic analysis…"
            self.state.flush()


def build_harmonics_dialog(ctrl):
    with v3.VDialog(
        v_model=("harmonics_open",),
        max_width=820,
        scrollable=True,
        raw_attrs=['aria-labelledby="coilgen-harmonics-title"'],
    ):
        with v3.VCard():
            with html.Div(classes="cg-dialog-head"):
                html.H2("Harmonics", id="coilgen-harmonics-title")
                v3.VBtn("Close", click="harmonics_open = false", variant="outlined", size="small")
            with v3.VCardText(classes="cg-dialog-body"):
                html.P("{{ run_source }} · {{ run_context }}", classes="cg-detail-note")
                html.P(
                    "Fit signed field components on the saved sample grid. Field, target and residual use the same basis."
                )
                with html.Div(v_if="harmonics_sources.length", classes="cg-shielding-controls"):
                    for model, label, items in (
                        ("harmonics_source", "Saved field source", "harmonics_sources"),
                        ("harmonics_component", "Signed field component", "harmonics_components"),
                    ):
                        v3.VSelect(
                            v_model=(model,),
                            label=label,
                            items=(items,),
                            density="compact",
                            hide_details=True,
                            disabled=("!!harmonics_status",),
                        )
                    v3.VTextField(
                        v_model=("harmonics_degree",),
                        label="Maximum degree",
                        type="number",
                        min=0,
                        max=12,
                        step=1,
                        density="compact",
                        hide_details=True,
                        disabled=("!!harmonics_status",),
                    )
                with html.Div(v_if="harmonics_sources.length", classes="cg-shielding-controls mt-2"):
                    v3.VTextField(
                        v_model=("harmonics_center_mm",),
                        label="Expansion center X, Y, Z [mm]",
                        density="compact",
                        hide_details=True,
                        disabled=("!!harmonics_status",),
                    )
                    v3.VTextField(
                        v_model=("harmonics_radius_mm",),
                        label="Reference radius [mm]",
                        type="number",
                        step="any",
                        density="compact",
                        hide_details=True,
                        disabled=("!!harmonics_status",),
                    )
                html.P(
                    "Reference defaults · {{ harmonics_reference }}",
                    v_if="harmonics_reference",
                    classes="cg-detail-note mt-1",
                )
                v3.VCheckbox(
                    v_if="harmonics_has_weights",
                    v_model=("harmonics_weighted",),
                    label="Use saved sample weights",
                    density="compact",
                    hide_details=True,
                    disabled=("!!harmonics_status",),
                )
                with html.Div(classes="cg-buttons mt-2"):
                    v3.VBtn(
                        "Fit harmonics",
                        v_if="harmonics_sources.length",
                        variant="flat",
                        color="primary",
                        disabled=("!!harmonics_status",),
                        click=(
                            ctrl.fit_harmonics,
                            "[harmonics_source, harmonics_component, harmonics_degree, "
                            "harmonics_center_mm, harmonics_radius_mm, harmonics_weighted]",
                        ),
                    )
                    v3.VBtn("Stop", v_if="harmonics_status", variant="outlined", click=ctrl.cancel_harmonics)
                html.P("{{ harmonics_status }}", v_if="harmonics_status", role="status", classes="mt-2")
                v3.VAlert(
                    "{{ harmonics_error }}", v_if="harmonics_error", type="error", density="compact", classes="mt-2"
                )
                _harmonic_fit_evidence()


def _harmonic_fit_evidence():
    with html.Div(v_if="harmonics_report", classes="mt-3"):
        html.P("{{ harmonics_report.source }} · {{ harmonics_report.component }}", classes="cg-detail-note")
        html.P(
            "Degree {{ harmonics_report.degree }} · center {{ harmonics_report.center }} · radius {{ harmonics_report.radius }}"
        )
        html.P(
            "Rank {{ harmonics_report.rank }} · condition {{ harmonics_report.condition }} · {{ harmonics_report.count }} fit samples"
        )
        html.P("{{ harmonics_report.scope }} · {{ harmonics_report.sampling }}", classes="cg-detail-note")
        with html.Ul(v_if="harmonics_report.diagnostics.length", classes="text-warning"):
            html.Li("{{ note }}", v_for="note in harmonics_report.diagnostics", key="note")
        v3.VSelect(
            v_model=("harmonics_view",),
            items=("harmonics_report.views",),
            label="Harmonic terms for",
            density="compact",
            hide_details=True,
            classes="mb-2",
        )
        with html.Div(
            v_for="view in harmonics_report.views.filter(item => item.value === harmonics_view)", key="view.value"
        ):
            html.P("{{ view.title }} sample RMS {{ view.sample_rms }} µT · fit error RMS {{ view.fit_error_rms }} µT")
            with html.Table(classes="cg-stats"):
                html.Caption("Terms ordered by RMS contribution")
                with html.Thead():
                    with html.Tr():
                        for label in ("Real harmonic", "Coefficient [µT]", "Term RMS [µT]"):
                            html.Th(label, scope="col")
                with html.Tbody():
                    with html.Tr(v_for="row in view.rows", key="row.index"):
                        html.Td("{{ row.term }}")
                        html.Td("{{ row.coefficient }}")
                        html.Td("{{ row.rms }}")
        html.P(
            "Coefficients multiply the normalized real basis (r/R)ˡYₗₘ. Term RMS is the RMS of each term on these "
            "samples. Terms can overlap on a finite grid, so their RMS values do not generally add in quadrature.",
            classes="cg-detail-note mt-2",
        )
        html.P(
            "Interpret the fit within a source-free region. Engineering acceptance is assessed separately.",
            classes="cg-detail-note",
        )
