"""Browser controls for the view tools: bodies, fit to ROI, located checks and the unwrapped view."""

from trame.widgets import html, vuetify3 as v3


def build_view_tools(ctrl):
    """Controls placed next to "Fit view" above the 3D view."""
    with html.Div(v_if="body_views.length", classes="cg-body-view d-flex align-center ga-1"):
        html.Span("Bodies", id="coilgen-body-view-label", classes="text-caption")
        # A segmented control reads as a choice; a bare native select looked like a label.
        with v3.VBtnToggle(
            id="coilgen-body-view",
            v_model=("body_view",),
            mandatory=True,
            density="compact",
            variant="outlined",
            divided=True,
            raw_attrs=['aria-labelledby="coilgen-body-view-label"'],
        ):
            v3.VBtn(
                "{{ choice.title }}",
                v_for="choice in body_views",
                key="choice.value",
                value=("choice.value",),
                size="small",
            )
        html.Span("{{ body_note }}", v_if="body_note", classes="cg-detail-note")
    v3.VBtn(
        "Fit to ROI",
        v_if="roi_available",
        click=ctrl.toggle_roi_fit,
        variant="outlined",
        size="small",
        prepend_icon="mdi-target",
        raw_attrs=[':aria-pressed="roi_fit"'],
        title="Fit the camera to the imaging region (shielding samples left out).",
    )
    v3.VBtn(
        "Unwrapped view",
        v_if="unwrapped_available",
        click=ctrl.show_unwrapped,
        variant="outlined",
        size="small",
        prepend_icon="mdi-panorama-horizontal-outline",
        title="Flat φ–z map of each cylinder: stream function, contours and openings.",
    )
    with html.Span(v_if="highlight_note", classes="cg-highlight"):
        html.Span("{{ highlight_note }}")
        v3.VBtn("Clear highlight", click=ctrl.clear_highlight, variant="text", size="small")


def build_locate_button(ctrl):
    """The Locate button for a check with a recorded spatial witness (inside the checks list)."""
    if ctrl is None:
        return
    v3.VBtn(
        "Locate",
        v_if="item.locatable",
        click=(ctrl.locate_problem, "[item.points_m, item.segments, item.dataset, item.title]"),
        variant="outlined",
        size="x-small",
        prepend_icon="mdi-crosshairs-gps",
        title=("item.where",),
    )


def build_unwrapped_dialog(run_dialog):
    with run_dialog("unwrapped", "Unwrapped view (φ, z)", max_width=1240):
        html.P("{{ unwrapped_note }}", classes="cg-detail-note", v_if="unwrapped_note")
        html.Img(
            v_if="unwrapped_image",
            src=("unwrapped_image",),
            alt="Primary and shield unwrapped to angle and height",
            style="width: 100%; height: auto",
        )


def build_export_presets():
    """One DRAFT banner for the list, and buttons that select every file of a format."""
    v3.VAlert(
        "{{ export_draft_banner }}",
        v_if="export_draft_banner",
        type="warning",
        variant="tonal",
        density="compact",
        classes="text-caption mb-2",
    )
    with html.Div(v_if="export_presets.length", classes="cg-buttons"):
        html.Span("Select:")
        v3.VBtn(
            "{{ preset.title }}",
            v_for="preset in export_presets",
            key="preset.key",
            click="export_selected = [...preset.values]",
            variant="outlined",
            size="small",
        )


def build_grouped_export_list(selected):
    """Export checkboxes under one heading per body role (Primary, Active shield, Print former…)."""
    with html.Div(v_for="group in export_groups", key="group.title", classes="cg-export-group"):
        html.Strong("{{ group.title }}")
        v3.VCheckbox(
            v_for="option in export_options.filter(o => export_option_group[o.value] === group.title)",
            key="option.value",
            v_model=(selected,),
            value=("option.value",),
            label=("option.title + ' — ' + option.status",),
            density="compact",
            hide_details=True,
        )
