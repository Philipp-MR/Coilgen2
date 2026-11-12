"""Model viewport and saved-run evidence for the browser workbench."""

from trame.widgets import html, vtk as vtk_widgets, vuetify3 as v3

from ..application.cad import PRINT_FORMER_EXPORT_HINT
from ..application.scene_views import VIEW_PRESETS
from .harmonics import build_harmonics_dialog
from .view_panel import (
    build_export_presets,
    build_grouped_export_list,
    build_locate_button,
    build_unwrapped_dialog,
    build_view_tools,
)
from .field_slices import build_field_slices_dialog


MODEL_PLOT = "['support','preview_target'].includes(dataset)"
RESULT_VIEW = f"summary && !{MODEL_PLOT}"


def _run_dialog(section, title, *, max_width=760):
    dialog = v3.VDialog(
        model_value=(f"center_section === '{section}'",),
        update_modelValue="center_section = $event ? center_section : 'results'",
        raw_attrs=[f'aria-labelledby="coilgen-{section}-title"'],
        max_width=max_width,
        scrollable=True,
    )
    with dialog:
        card = v3.VCard()
        with card:
            with html.Div(classes="cg-dialog-head"):
                html.H2(title, id="coilgen-" + section + "-title")
                v3.VBtn("Close", click="center_section = 'results'", variant="outlined", size="small")
            body = v3.VCardText(classes="cg-dialog-body")
    return body


def _checks_dialog(ctrl=None):
    with _run_dialog("checks", "Engineering checks"):
        with html.Div(v_if="summary"):
            html.P("{{ run_source }} · {{ run_context }}", classes="cg-detail-note")
            html.Strong("{{ summary.label }}")
            html.Div("{{ check.name }} · {{ check.value }}", v_for="check in summary.check_rows", key="check.name")
            html.P("{{ summary.construction_released ? 'Construction released' : 'No construction release recorded' }}")
            with html.Details(v_for="group in summary.problems", key="group.title"):
                html.Summary("{{ group.title }} · {{ group.counts }}")
                with html.Ul():
                    with html.Li(v_for="(item, index) in group.items", key="index"):
                        html.Strong("{{ item.title }}", title=("item.explanation",))
                        html.Div("{{ item.details }}")
                        build_locate_button(ctrl)
            with html.Div(v_if="summary.suppression", classes="mt-2"):
                html.Strong("Suppression on {{ summary.suppression.sample_count }} design samples")
                html.Div("{{ summary.suppression.component }} · {{ summary.suppression.weighting }} weighting")
                with html.Div(v_for="kind in ['sheet','fabricated']"):
                    with html.Div(v_if="summary.suppression[kind]"):
                        html.Strong("{{ kind === 'sheet' ? 'Stream function field' : 'Fabricated conductor field' }}")
                        with html.Div(v_for="measure in ['rms', 'peak']"):
                            html.Span(
                                "{{ measure.toUpperCase() }} · {{ summary.suppression[kind][measure+'_T'] == null "
                                "? 'Unavailable' : (summary.suppression[kind][measure+'_T'] * 1e6).toPrecision(5) + ' µT' }}"
                            )
                            html.Span(
                                " (limit {{ (summary.suppression[measure+'_limit_T'] * 1e6).toPrecision(4) }} µT · "
                                "{{ summary.suppression[kind][measure+'_passed'] == null ? 'Unverified' : "
                                "summary.suppression[kind][measure+'_passed'] ? 'Passed' : 'Failed' }})",
                                v_if="summary.suppression[measure+'_limit_T'] != null",
                            )
                        html.Div("{{ summary.suppression[kind].reason }}", v_if="summary.suppression[kind].reason")


def _stats_dialog():
    with _run_dialog("metrics", "Coil stats"):
        with html.Div(v_if="summary"):
            html.P("{{ run_source }} · {{ run_context }}", classes="cg-detail-note")
            with html.Table(classes="cg-stats"):
                with html.Caption():
                    html.Strong("Recorded metrics")
                with html.Tbody():
                    with html.Tr(v_if="summary.reviewed_wire_metric"):
                        html.Th("{{ summary.reviewed_wire_metric.name }}", scope="row")
                        html.Td("{{ summary.reviewed_wire_metric.value }}")
                    with html.Tr(v_for="row in metrics", key="row.name"):
                        html.Th("{{ row.name }}", scope="row")
                        html.Td("{{ row.value }}")
            with html.Details(v_if="summary.overview.body_rows.length"):
                html.Summary("Per-body measurements")
                with html.Table(classes="cg-stats"):
                    with html.Thead():
                        with html.Tr():
                            for label in ("Body", "Measurement", "Value"):
                                html.Th(label, scope="col")
                    with html.Tbody():
                        with html.Tr(v_for="(row, i) in summary.overview.body_rows", key="i"):
                            html.Td("{{ row.body }}")
                            html.Th("{{ row.name }}", scope="row", title=("row.scope",))
                            html.Td("{{ row.value }}")


def _shielding_dialog(ctrl):
    from ..gui.shielding_plot import SHIELDING_COMPONENTS

    with _run_dialog("shielding", "Shielding on the external surface", max_width=1240):
        html.P("{{ run_source }} · {{ run_context }}", classes="cg-detail-note")
        with html.Div(classes="cg-shielding-controls"):
            with html.Div():
                html.Label("Field source", **{"for": "coilgen-shielding-source"})
                with html.Select(
                    id="coilgen-shielding-source",
                    v_model=("shielding_source",),
                    disabled=("shielding_busy",),
                    change=(ctrl.show_shielding, "[$event.target.value, shielding_component, shielding_detail]"),
                ):
                    html.Option("{{ s.title }}", v_for="s in shielding_sources", key="s.value", value=("s.value",))
            with html.Div():
                html.Label("Field component", **{"for": "coilgen-shielding-component"})
                with html.Select(
                    id="coilgen-shielding-component",
                    v_model=("shielding_component",),
                    disabled=("shielding_busy",),
                    change=(ctrl.show_shielding, "[shielding_source, $event.target.value, shielding_detail]"),
                ):
                    for key, label in SHIELDING_COMPONENTS.items():
                        html.Option(label, value=key, disabled=("!shielding_normal",) if key == "normal" else False)
            with html.Div():
                html.Label("Map scale", **{"for": "coilgen-shielding-scale"})
                with html.Select(
                    id="coilgen-shielding-scale",
                    v_model=("shielding_detail",),
                    disabled=("shielding_busy",),
                    change=(
                        ctrl.show_shielding,
                        "[shielding_source, shielding_component, $event.target.value === 'true']",
                    ),
                ):
                    html.Option("Compare fields on a common scale", value=("false",))
                    html.Option("Resolve the net field on its own scale", value=("true",))
        html.P(
            "{{ shielding_source.endsWith('_interior') ? "
            "'Evaluating the saved current distribution inside each observation triangle…' : "
            "'Drawing saved surface fields…' }}",
            v_if="shielding_busy",
            role="status",
        )
        html.Img(
            v_if="shielding_image",
            src=("shielding_image",),
            classes="cg-shielding-image",
            alt="Magnetic field on the external suppression surface",
        )
        html.P("{{ shielding_note }}", classes="cg-shielding-note", role="status")
        html.A(
            "Download full-size plot (PNG)",
            v_if="shielding_image",
            href=("shielding_image",),
            download=("shielding_source + '-' + shielding_component + '-shielding.png'",),
        )
        html.P(
            "Use these magnetostatic fields as excitation data for a magnet interaction study. "
            "Induced currents, heating and forces require a conducting structure and a drive waveform.",
            classes="cg-detail-note",
        )


def _export_dialog(ctrl, *, former=False):
    section, title = ("export_former", "Export print former") if former else ("export", "Export conductor geometry")
    options = "export_options.filter(option => option.former)" if former else "export_options"
    selected = "former_export_selected" if former else "export_selected"
    draft_ack = "former_draft_ack" if former else "draft_ack"
    multiple_files = {"v_if": f"{options}.length > 1"} if former else {}
    with _run_dialog(section, title):
        html.P("{{ run_source }} · {{ run_context }}", classes="cg-detail-note", v_if="summary")
        if former:
            html.P("{{ summary?.print_former_status }}")
            html.P("Former geometry is exported for inspection. Coil acceptance and construction release are separate.")
        html.P("No saved geometry is available for export in this selection.", v_if=f"!{options}.length")
        with html.Div(v_else=True):
            html.P("{{ export_selection_note }}", role="status")
            if former:
                html.P(
                    f"{{{{ {options}[0]?.title }}}}",
                    v_if=f"{options}.length === 1",
                    classes="cg-detail-note",
                )
            if not former:
                build_export_presets()
            with html.Details(key=("selected_run",), **multiple_files):
                html.Summary("Choose files and formats")
                if former:
                    v3.VCheckbox(
                        v_for=f"option in {options}",
                        key="option.value",
                        v_model=(selected,),
                        value=("option.value",),
                        label=("option.title + ' — ' + option.status",),
                        density="compact",
                        hide_details=True,
                    )
                else:
                    build_grouped_export_list(selected)
            v3.VSelect(
                v_model=("export_unit",),
                items=("['mm', 'm']",),
                label="STL unit" if former else "STL / CSV unit",
                density="compact",
                hide_details=True,
            )
            v3.VCheckbox(
                v_model=("include_project",),
                label="Include matching run project",
                density="compact",
                hide_details=True,
            )
            with html.Div(v_if="draft_selected"):
                v3.VAlert(
                    "{{ draft_notice }}", type="error", variant="tonal", density="compact", classes="text-caption"
                )
                v3.VCheckbox(
                    v_model=(draft_ack,),
                    label="Export DRAFT geometry although recorded checks failed",
                    density="compact",
                )
            v3.VBtn(
                f"{{{{ {options}.length === 1 ? 'Export former STL' : 'Export selected' }}}}"
                if former
                else "Export selected",
                color="primary",
                click=ctrl.export_geometry,
                disabled=(f"!{selected}.length || (draft_selected && !{draft_ack})",),
            )
            html.A(
                "{{ download_label }}",
                v_if="download_url",
                href=("download_url",),
                download=True,
                classes="d-block mt-2",
            )


def _metric_cards(key):
    with html.Div(v_if=f"summary.overview.{key}_cards.length", classes=f"cg-metrics cg-metrics-{key}"):
        with html.Div(
            v_for=f"card in summary.overview.{key}_cards",
            key="card.key",
            title=(f"card.tooltip || summary.overview.{key}_tooltip",),
            raw_attrs=[':data-status="card.status || null"'],
        ):
            html.Span("{{ card.name }}")
            html.Strong("{{ card.value }}", classes=("{'cg-unavailable':card.unavailable}",))
    html.Div(
        "{{ summary.overview." + key + "_context }}",
        classes="cg-scope",
        v_if=f"summary.overview.{key}_cards.length",
    )


def build_center_panel(workbench):
    ctrl = workbench.server.controller
    workbench.server.enable_module({"styles": ["coilgen-client/center_panel.css"]})
    with html.Div(classes="cg-center-panel"):
        with html.Div(classes="cg-context"):
            with html.Div(classes="cg-context-heading"):
                html.Div("{{ editing_label || 'Workspace run history' }}")
                html.Div("Viewing · {{ run_source }} · {{ run_context }}", v_if=RESULT_VIEW, classes="cg-viewing")
            with html.Div(classes="cg-buttons"):
                v3.VBtn(
                    "Model",
                    v_if=f"channels.length && {RESULT_VIEW}",
                    click="dataset = 'support'",
                    prepend_icon="mdi-cube-outline",
                    size="small",
                    variant="outlined",
                )
                v3.VBtn(
                    "Coil results",
                    v_if=f"summary && {MODEL_PLOT}",
                    click="dataset = result_plot",
                    prepend_icon="mdi-chart-line",
                    size="small",
                    variant="outlined",
                )
                v3.VBtn(
                    "Run design choices",
                    v_if=RESULT_VIEW,
                    click=ctrl.review_submitted_design,
                    disabled=("selected_run !== displayed_run",),
                    prepend_icon="mdi-tune",
                    size="small",
                    variant="outlined",
                )
        with html.Div(
            v_if="attempt_notice", classes="cg-attempt", role="status", raw_attrs=[':data-tone="attempt_notice.color"']
        ):
            html.Strong("{{ attempt_notice.title }} ")
            html.Span("{{ attempt_notice.action }}")
        with html.Div(v_if=RESULT_VIEW, classes="cg-overview"):
            with html.Div(classes="cg-board-heading"):
                html.Strong("Results")
                v3.VChip("{{ summary.label }}", color=("summary.color",), variant="outlined", size="small")
                v3.VBtn(
                    "{{ results_metrics_expanded ? 'Hide metrics' : 'Show metrics' }}",
                    click="results_metrics_expanded = !results_metrics_expanded",
                    size="small",
                    variant="outlined",
                    raw_attrs=[':aria-expanded="results_metrics_expanded"', 'aria-controls="coilgen-result-metrics"'],
                )
            html.Div(
                "Failed: {{ summary.failed_reasons }}",
                v_if="summary.failed_reasons && summary.color === 'error'",
                classes="cg-failed-reasons",
                role="status",
                title="Open Engineering checks for every failed item and its measured value.",
            )
            with html.Div(id="coilgen-result-metrics", v_show="results_metrics_expanded"):
                _metric_cards("gradient")
                _metric_cards("drive")
                _metric_cards("field")
                _metric_cards("coil")
                html.Div(
                    "{{ summary.overview.partial_note }}", v_if="summary.overview.partial_note", classes="cg-scope"
                )
            with html.Div(classes="cg-buttons"):
                v3.VBtn(
                    "Field map",
                    v_if="summary.overview.field_component && available_plots.some(g => "
                    "g.items.some(p => p.value === summary.overview.field_plot))",
                    click="component = summary.overview.field_component; dataset = summary.overview.field_plot",
                    prepend_icon="mdi-chart-line",
                    size="small",
                    variant="outlined",
                    title="Show the field and component used for the result cards",
                )
                v3.VBtn(
                    "Residual slices",
                    v_if="available_plots.some(group => group.items.some(plot => "
                    "['sheet', 'validation', 'contour_field', 'routed_field', 'fabricated', 'winding_validation'].includes(plot.value)))",
                    click=ctrl.open_field_slices,
                    disabled=("selected_run !== displayed_run",),
                    prepend_icon="mdi-layers-outline",
                    size="small",
                    variant="outlined",
                )
                v3.VBtn(
                    "Harmonics",
                    v_if="harmonics_available",
                    click=ctrl.open_harmonics,
                    disabled=("selected_run !== displayed_run",),
                    prepend_icon="mdi-chart-bar",
                    size="small",
                    variant="outlined",
                )
                v3.VBtn(
                    "Shielding surface",
                    v_if="shielding_sources.length",
                    click=(ctrl.show_shielding, "[shielding_source, shielding_component, shielding_detail]"),
                    disabled=("selected_run !== displayed_run",),
                    prepend_icon="mdi-shield-outline",
                    size="small",
                    variant="outlined",
                )
                v3.VBtn(
                    "Coil stats",
                    click="center_section = 'metrics'",
                    prepend_icon="mdi-chart-box-outline",
                    size="small",
                    variant="outlined",
                )
                v3.VBtn(
                    "Engineering checks",
                    click="center_section = 'checks'",
                    prepend_icon="mdi-check-all",
                    size="small",
                    variant="outlined",
                )
            former_plot = "available_plots.some(group => group.items.some(plot => plot.value === 'print_former'))"
            former_export = "export_options.some(option => option.former)"
            with html.Div(
                v_if=f"summary.print_former_status || {former_plot} || {former_export}", classes="cg-former-row"
            ):
                with html.Div(classes="cg-former-status"):
                    html.Strong("{{ summary.print_former_status || 'Print former' }}")
                    html.Div(PRINT_FORMER_EXPORT_HINT, v_if=f"!{former_export}", classes="cg-detail-note")
                with html.Div(classes="cg-buttons"):
                    v3.VBtn(
                        "Show print former",
                        v_if=former_plot,
                        click="results_metrics_expanded = false; dataset = 'print_former'",
                        disabled=("selected_run !== displayed_run",),
                        prepend_icon="mdi-cube-outline",
                        size="small",
                        variant="outlined",
                    )
                    v3.VBtn(
                        "Export print former…",
                        v_if=former_export,
                        click=ctrl.open_print_former,
                        disabled=("selected_run !== displayed_run",),
                        prepend_icon="mdi-download-outline",
                        size="small",
                        variant="outlined",
                    )
        with html.Div(
            v_if=f"channels.length && {MODEL_PLOT} && (!current_job_active || settings_dirty)",
            classes="cg-model-guide",
        ):
            html.Strong("Model preview")
            html.Div(
                "{{ settings_dirty ? 'Apply or discard pending edits in Model Builder.' : "
                "readiness?.state === 'blocking' ? 'Open Model Builder to fix the required inputs.' : "
                "'Review inputs in Model Builder, then use the blue compute button. Change the scope in Design plan.' }}"
            )
        with html.Div(v_if="!channels.length && !summary"):
            html.P("Open a project to edit it or use Runs to review a saved result.")
        with html.Div(v_show="channels.length || summary", classes="cg-plot"):
            with html.Div(classes="cg-plot-bar", v_if="available_plots.length && dataset !== 'saved_unavailable'"):
                html.Label("Plot", **{"for": "coilgen-plot"})
                with html.Select(id="coilgen-plot", v_model=("dataset",)):
                    with html.Optgroup(v_for="group in available_plots", key="group.label", label=("group.label",)):
                        html.Option(
                            "{{ plot.title }}", v_for="plot in group.items", key="plot.value", value=("plot.value",)
                        )
                with html.Div(v_if="field_plot", classes="cg-component"):
                    html.Label("Component", **{"for": "coilgen-field-component"})
                    with html.Select(id="coilgen-field-component", v_model=("component",)):
                        for component in ("Bz", "Bx", "By", "Bmag", "BdotB0"):
                            html.Option(component, value=component)
                with html.Div(v_if="field_has_pair", classes="cg-buttons"):
                    for label, target in (("Field", "false"), ("Paired target", "true")):
                        v3.VBtn(
                            label,
                            size="small",
                            variant="outlined",
                            click=f"field_show_target = {target}",
                            raw_attrs=[f':aria-pressed="field_show_target === {target}"'],
                        )
            with html.Div(classes="cg-view-tools", v_if="dataset !== 'saved_unavailable'"):
                with v3.VBtn(
                    size="small", variant="outlined", append_icon="mdi-chevron-down", prepend_icon="mdi-axis-arrow"
                ):
                    html.Span("View direction")
                    with v3.VMenu(activator="parent"):
                        with v3.VList(density="compact"):
                            for view in VIEW_PRESETS:
                                v3.VListItem(title=view, click=(workbench.set_camera_view, f"['{view}']"))
                v3.VBtn(
                    "Fit view",
                    click=workbench.server.controller.fit_browser_view,
                    variant="outlined",
                    size="small",
                    prepend_icon="mdi-fit-to-screen-outline",
                )
                build_view_tools(workbench.server.controller)
                with html.Label(v_if="outer_winding_support", classes="cg-support-visibility"):
                    html.Input(type="checkbox", v_model=("see_through_support",))
                    html.Span("See through support")
                html.Span("{{ b0_text }}", classes="cg-b0", v_if="b0_text")
            html.P("{{ field_source_header }}", v_if="field_source_header", classes="cg-detail-note")
            with html.Div(classes="cg-viewport", v_show="dataset !== 'saved_unavailable'"):
                workbench.view = vtk_widgets.VtkLocalView(workbench.plotter.ren_win)
            workbench.server.controller.fit_browser_view = workbench.view.reset_camera
            with html.Div(v_if="field_legend", classes="cg-legend"):
                html.Span("{{ field_legend.label }}")
                html.Span("{{ field_legend.low }}")
                html.Span(classes="cg-ramp", style=("{background:field_legend.gradient}",))
                html.Span("{{ field_legend.high }}")
                html.Span("Shared field and target scale", v_if="field_has_pair")
            html.Div("{{ scene_caption }}", classes="cg-scope")
            html.Div(
                "{{ summary.contour_fit_caption }}",
                v_if="summary && summary.contour_fit_caption && "
                "['contours','routes','evaluated_winding','fabrication','swept','review_conductor','review_electrical'].includes(dataset)",
                classes="cg-scope",
            )
        _checks_dialog(ctrl)
        build_field_slices_dialog(ctrl)
        build_harmonics_dialog(ctrl)
        _stats_dialog()
        _shielding_dialog(ctrl)
        build_unwrapped_dialog(_run_dialog)
        _export_dialog(ctrl)
        _export_dialog(ctrl, former=True)
