"""Browser presentation of the desktop workbench's project, model, and plan panels."""

from base64 import b64encode

from trame.widgets import html, vuetify3 as v3
from trame_client.widgets.core import HtmlElement

from ..application.preflight import SECTION_ORDER, overall_readiness, section_readiness
from ..application.session import SETTINGS_NODES
from ..application.workbench_presentation import (
    BROWSE_EXAMPLES_LABEL,
    COMPUTE_CAPTIONS,
    DESIGN_PLAN_TITLE,
    MODEL_BUILDER_TITLE,
    OPEN_PROJECT_LABEL,
    RUNS_LABEL,
)
from ..gui.pictograms import model_node_icons, pictogram_svg
from .input_policy import PROJECT_UPLOAD_LIMIT


def _wordmark(size):
    with html.Div(classes=f"cg-wordmark cg-wordmark--{size}"):
        html.Img(src="coilgen-brand/coilgen-logo.png", alt="CoilGen²")


def _tree_symbol(name):
    return "data:image/svg+xml;base64," + b64encode(pictogram_svg(name, "#213746")).decode("ascii")


def preflight_node(node):
    """Browser name of a preflight settings section."""
    return "conductor" if node == "material" else node


def model_tree_nodes(config, report=None):
    """Tree rows with each section's preflight state; "set" is not engineering acceptance."""
    symbols = model_node_icons(config)
    sections = section_readiness(report, nodes=SECTION_ORDER, config=config) if report is not None else {}
    nodes = []
    for key, label in SETTINGS_NODES:
        native = "material" if key == "conductor" else key
        section = sections.get(native)
        nodes.append(
            {
                "title": label,
                "value": key,
                "symbol": _tree_symbol(symbols[native]),
                "state": section.state if section else "unchecked",
                "summary": section.summary if section else "",
                "action": section.next_action if section else "",
                "tooltip": f"{label} · {section.summary}\n{section.details}" if section else label,
            }
        )
    return nodes


def readiness_summary(config, report):
    """Banner above the browser tree: what blocks computing, or that it's ready."""
    if report is None:
        return None
    labels = {("material" if key == "conductor" else key): label for key, label in SETTINGS_NODES}
    overall = overall_readiness(section_readiness(report, nodes=SECTION_ORDER, config=config))
    return {
        "state": overall.state,
        "title": overall.title(),
        "detail": overall.detail(labels),
        "action": overall.action,
        "node": preflight_node(overall.node) if overall.node else "",
    }


def build_toolbar(ctrl):
    with html.Header(classes="cg-toolbar"):
        _wordmark("small")
        v3.VBtn(
            "File",
            variant="outlined",
            size="small",
            append_icon="mdi-menu-down",
            click="file_section = 'examples'; file_dialog = true",
        )
        html.Div("{{ project_name }}", v_if="channels.length", classes="cg-project-name", title=("project_name",))
        with html.Form(action="/logout", method="post", v_if="user !== 'local'", classes="cg-session"):
            html.Span("{{ user }}", classes="cg-user")
            html.Button("Log out", type="submit")
        v3.VBtn(
            "{{ stages.find(s => s.value === stage)?.compute }}",
            v_if="channels.length",
            color="primary",
            prepend_icon="mdi-play-outline",
            classes="cg-compute",
            disabled=("current_job_active || settings_dirty || preflight.some(i => i.blocking)",),
            title=(
                "settings_dirty ? 'Apply or discard pending model edits first' : "
                "current_job_active ? 'This study is already queued or computing' : "
                "readiness?.state === 'blocking' ? readiness.title : "
                "'Compute the editing channel to the boundary selected in Design plan'",
            ),
            click=ctrl.compute,
        )


def build_welcome(ctrl):
    with html.Main(v_if="!channels.length && !history_open", classes="cg-welcome"):
        _wordmark("large")
        html.H1("CoilGen² workbench", classes="cg-sr-only")
        html.P("Choose a gradient, shim coil or coil system in Browse examples.")
        html.P(f"Review the model, then choose {COMPUTE_CAPTIONS['evaluated']} when the inputs are ready.")
        with html.Div(classes="cg-welcome-actions"):
            v3.VBtn(
                BROWSE_EXAMPLES_LABEL,
                color="primary",
                prepend_icon="mdi-folder-multiple-outline",
                click="file_section = 'examples'; file_dialog = true",
            )
            v3.VBtn(
                OPEN_PROJECT_LABEL,
                variant="outlined",
                prepend_icon="mdi-folder-open-outline",
                click="file_section = 'open'; file_dialog = true",
            )
            v3.VBtn(
                "New cylindrical design",
                variant="outlined",
                prepend_icon="mdi-file-outline",
                click=ctrl.new_project,
            )


def build_file_dialog(ctrl):
    with v3.VDialog(
        v_model=("file_dialog",),
        max_width=620,
        scrollable=True,
        __properties=[("aria_label", "aria-label")],
        aria_label="Project files",
    ):
        with v3.VCard():
            v3.VCardTitle("Project files")
            with v3.VCardText():
                with html.Nav(
                    classes="cg-file-sections",
                    __properties=[("aria_label", "aria-label")],
                    aria_label="Project file actions",
                ):
                    for value, label in (
                        ("examples", "Browse examples"),
                        ("open", "Open project"),
                        ("save", "Save / download"),
                    ):
                        v3.VBtn(
                            label,
                            v_if="channels.length" if value == "save" else None,
                            size="small",
                            variant=(f"file_section === '{value}' ? 'tonal' : 'text'",),
                            __properties=[("aria_pressed", "aria-pressed")],
                            aria_pressed=(f"file_section === '{value}'",),
                            click=f"file_section = '{value}'",
                        )
                with html.Div(v_show="file_section === 'examples'"):
                    html.Label("Template", classes="cg-field-label", **{"for": "coilgen-gallery-template"})
                    with html.Select(
                        id="coilgen-gallery-template", v_model=("template",), disabled=("settings_dirty",)
                    ):
                        html.Option("Choose an example", disabled=True, value=("null",))
                        html.Option(
                            "{{ option.title }}",
                            v_for="option in templates",
                            key="option.value",
                            value=("option.value",),
                        )
                    html.P(
                        "{{ templates.find(t => t.value === template)?.description }}",
                        v_if="template",
                        classes="cg-file-description",
                    )
                    with html.Div(classes="cg-file-actions"):
                        v3.VBtn(
                            "Load template",
                            click=(ctrl.load_template, "[template]"),
                            color="primary",
                            disabled=("!template || settings_dirty",),
                        )
                        v3.VBtn(
                            "New cylindrical design",
                            v_if="channels.length || history_open",
                            click=ctrl.new_project,
                            variant="text",
                            disabled=("settings_dirty",),
                        )
                with html.Div(v_show="file_section === 'open'"):
                    HtmlElement(
                        "coilgen-project-upload",
                        __properties=["disabled", "result", "limit"],
                        __events=["uploaded"],
                        disabled=("settings_dirty",),
                        result=("project_open_result",),
                        limit=(str(PROJECT_UPLOAD_LIMIT),),
                        uploaded=(ctrl.open_uploaded_project, "[$event]"),
                    )
                    with html.Div(v_if="saved_projects.length", classes="cg-workspace-files"):
                        html.Label(
                            "Saved in this workspace", classes="cg-field-label", **{"for": "coilgen-saved-project"}
                        )
                        with html.Select(
                            id="coilgen-saved-project", v_model=("workspace_file",), disabled=("settings_dirty",)
                        ):
                            html.Option("Choose a saved project", disabled=True, value=("null",))
                            html.Option(
                                "{{ filename }}",
                                v_for="filename in saved_projects",
                                key="filename",
                                value=("filename",),
                            )
                        v3.VBtn(
                            "Reopen saved project",
                            click=ctrl.open_workspace_project,
                            disabled=("settings_dirty || !workspace_file",),
                            variant="tonal",
                            classes="mt-2",
                        )
                with html.Div(v_show="file_section === 'save' && channels.length"):
                    html.P("Save the editable design and its recorded results to this workspace, then download a copy.")
                    with html.Div(classes="cg-file-actions"):
                        v3.VBtn(
                            "Save project",
                            click=(ctrl.save_project, "['bundle']"),
                            color="primary",
                            disabled=("settings_dirty",),
                        )
                        v3.VBtn(
                            "Save JSON",
                            click=(ctrl.save_project, "['json']"),
                            variant="text",
                            disabled=("settings_dirty",),
                        )
                    html.A(
                        "Download project",
                        v_if="project_download",
                        href=("project_download",),
                        download=True,
                        classes="d-block mt-2",
                    )
                html.P(
                    "Apply or discard the pending edits in Model settings first.", v_if="settings_dirty", role="status"
                )
            with v3.VCardActions():
                v3.VSpacer()
                v3.VBtn("Close", click="file_dialog = false")


def build_model_builder(ctrl):
    html.H2(MODEL_BUILDER_TITLE, classes="cg-pane-heading")
    with html.Div(classes="cg-panel-body"):
        html.P("Open a project to edit the model.", v_if="!channels.length", classes="cg-hint")
        with html.Div(v_if="channels.length"):
            v3.VSelect(
                v_if="channels.length > 1",
                v_model=("active_channel",),
                items=("channels",),
                label="Editing channel",
                disabled=("settings_dirty",),
                density="compact",
                hide_details=True,
                classes="mb-2",
            )
            with html.Div(
                v_if="readiness",
                classes="cg-readiness",
                role="status",
                __properties=[("data_state", "data-state")],
                data_state=("settings_dirty ? 'pending' : current_job_active ? 'computing' : readiness.state",),
            ):
                v3.VIcon(
                    "{{ settings_dirty ? 'mdi-pencil-outline' : current_job_active ? 'mdi-progress-clock' : "
                    "readiness.state === 'blocking' ? 'mdi-close-circle-outline' : 'mdi-check-circle-outline' }}",
                    classes="cg-readiness-icon",
                    __properties=[("aria_hidden", "aria-hidden")],
                    aria_hidden="true",
                )
                with html.Div(classes="cg-readiness-text"):
                    html.Div(
                        "{{ settings_dirty ? 'Edits not applied yet' : current_job_active ? "
                        "'Computing this revision' : readiness.title }}",
                        classes="cg-readiness-title",
                    )
                    html.Div(
                        "{{ settings_dirty ? 'Apply them to check the inputs again.' : current_job_active ? "
                        "'The run keeps the inputs it was submitted with. Edits you apply now start a new revision.' "
                        ": readiness.detail }}",
                        classes="cg-readiness-detail",
                    )
                v3.VBtn(
                    "{{ readiness.action }}",
                    v_if="readiness.action && !settings_dirty",
                    size="small",
                    variant="outlined",
                    click="node = readiness.node; settings_dialog = true",
                )
            for group, keys in (
                ("Model", "['geometry','target','conductor','winding','print']"),
                ("Study", "['optimization','validation']"),
            ):
                html.H3(group, classes="cg-tree-heading")
                with html.Div(v_for=f"item in nodes.filter(n => {keys}.includes(n.value))", key="item.value"):
                    with html.Button(
                        type="button",
                        classes="cg-tree-item",
                        disabled=("settings_dirty && node !== item.value",),
                        click="node = item.value; settings_dialog = true",
                        title=("item.tooltip",),
                        __properties=[("data_state", "data-state"), ("aria_label", "aria-label")],
                        data_state=("item.state",),
                        aria_label=("item.title + (item.summary ? ' · ' + item.summary : '')",),
                    ):
                        html.Img(src=("item.symbol",), alt="", classes="cg-tree-symbol")
                        with html.Span(classes="cg-tree-text"):
                            html.Span("{{ item.title }}")
                            html.Span(
                                "{{ item.summary }}",
                                v_if="['blocking', 'warning', 'set', 'off'].includes(item.state)",
                                classes="cg-tree-status",
                            )
                        v3.VIcon(
                            "{{ {blocking: 'mdi-close-circle-outline', warning: 'mdi-information-outline', "
                            "set: 'mdi-check-circle-outline', off: 'mdi-plus-circle-outline'}[item.state] }}",
                            v_if="['blocking', 'warning', 'set', 'off'].includes(item.state)",
                            size="small",
                            classes="cg-tree-state",
                            __properties=[("aria_hidden", "aria-hidden")],
                            aria_hidden="true",
                        )
            html.P(
                "Ready: nothing missing · Tip: worth a look, never blocks · Needs input: blocks computing",
                classes="cg-hint cg-readiness-legend",
            )
            with html.Div(classes="d-flex ga-1"):
                v3.VBtn("Undo", v_if="can_undo", click=ctrl.undo, disabled=("settings_dirty",), variant="text")
                v3.VBtn("Redo", v_if="can_redo", click=ctrl.redo, disabled=("settings_dirty",), variant="text")
            with html.Details(classes="cg-channels"):
                html.Summary("Manage channels")
                v3.VCheckbox(
                    v_if="channels.length > 1",
                    model_value=("channel_enabled",),
                    disabled=("settings_dirty",),
                    label="Include channel in batch",
                    update_modelValue=(ctrl.toggle_channel, "[active_channel, $event]"),
                    hide_details=True,
                    density="compact",
                )
                v3.VBtn(
                    "Duplicate channel",
                    size="small",
                    click=ctrl.duplicate_channel,
                    variant="text",
                    disabled=("settings_dirty",),
                )
                v3.VBtn(
                    "Remove channel",
                    v_if="channels.length > 1",
                    size="small",
                    click=ctrl.remove_channel,
                    variant="text",
                    disabled=("settings_dirty",),
                )
        with html.Div(v_if="export_options.length"):
            html.H3("Results and provenance", classes="cg-tree-heading")
            with html.Button(
                type="button",
                classes="cg-tree-item",
                click="workspace_panel = 'center'; center_section = 'export'",
            ):
                html.Img(src=_tree_symbol("export-conductor"), alt="", classes="cg-tree-symbol")
                html.Span("Export conductor geometry")


def build_design_plan(ctrl):
    html.H2(DESIGN_PLAN_TITLE, classes="cg-pane-heading")
    with html.Div(classes="cg-panel-body"):
        with html.Div(v_if="channels.length"):
            html.Div("{{ plan_summary }}", classes="cg-plan-summary")
            html.P("Recorded steps describe computation. Review engineering checks in the results.", classes="cg-hint")
            html.Label("Compute to", classes="cg-field-label", **{"for": "coilgen-compute-to"})
            with html.Select(id="coilgen-compute-to", v_model=("stage",), disabled=("settings_dirty",)):
                html.Option("{{ item.title }}", v_for="item in stages", key="item.value", value=("item.value",))
            with html.Div(v_for="channel in plan_channels", key="channel.id", classes="cg-plan-channel"):
                html.H3("{{ channel.name }}", classes="cg-channel-name")
                html.Div("Excluded from batch", v_if="!channel.enabled", classes="cg-hint")
                html.P(
                    "{{ channel.activity }}",
                    v_if="channel.activity",
                    classes=("channel.error ? 'cg-channel-activity text-error' : 'cg-channel-activity'",),
                )
                with html.Ol(classes="cg-plan-steps"):
                    with html.Li(v_for="step in channel.steps", key="step.key", title=("step.help",)):
                        html.Span("{{ step.label }}")
                        html.Span(
                            "{{ step.status }}",
                            classes="cg-step-state",
                            __properties=[("data_state", "data-state")],
                            data_state=("step.code",),
                        )
            html.P(
                "Rows show each channel’s applied study. Compute enabled channels sets every included channel to the selected boundary.",
                v_if="channels.length > 1",
                classes="cg-hint",
            )
            v3.VBtn(
                "Compute enabled channels",
                v_if="channels.length > 1",
                variant="outlined",
                size="small",
                click=ctrl.compute_channels,
                disabled=("settings_dirty || batch_busy_channels.length > 0 || !channels.some(c => c.enabled)",),
            )
            html.P(
                "Already active in this batch: {{ batch_busy_channels.join(', ') }}. Wait for these jobs or compute another channel individually.",
                v_if="batch_busy_channels.length",
                classes="cg-hint",
            )
            with v3.VAlert(
                v_for="issue in preflight.filter(i => i.blocking)",
                type="error",
                density="compact",
                classes="mt-2",
            ):
                html.Div("{{ issue.section }}: {{ issue.title }}", classes="font-weight-medium")
                html.Div("{{ issue.action }}", classes="text-caption")
                v3.VBtn(
                    "Open {{ issue.section }}",
                    size="small",
                    variant="text",
                    disabled=("settings_dirty && node !== issue.node",),
                    click="node = issue.node; settings_dialog = true",
                )
        html.P("Open a project to see its requested steps.", v_if="!channels.length", classes="cg-hint")


def build_activity_bar():
    with html.Div(
        v_if="channels.length || active_jobs.length || jobs.length || runs.length",
        classes="cg-activity",
        raw_attrs=[
            ":data-state=\"active_jobs.length === 1 ? active_jobs[0].status : active_jobs.length ? 'multiple' : 'idle'\""
        ],
    ):
        with html.Div(classes="cg-activity-text", role="status"):
            html.Strong(
                "{{ active_jobs.length > 1 ? active_jobs.length + ' active runs' : "
                "active_jobs[0]?.caption || 'No active runs' }}"
            )
            html.Span(
                "{{ [active_jobs[0].channel, active_jobs[0].stage || "
                "(active_jobs[0].status === 'queued' ? 'Waiting for a worker' : active_jobs[0].boundary)].join(' · ') }}",
                v_if="active_jobs.length === 1",
                classes="cg-activity-detail",
            )
        with html.Div(
            v_if="active_jobs.length === 1 && active_jobs[0].status === 'running'", classes="cg-activity-progress"
        ):
            v3.VProgressLinear(
                model_value=("active_jobs[0].progress ?? 0",),
                indeterminate=("active_jobs[0].progress === null",),
                color="primary",
                rounded=True,
                height=6,
                raw_attrs=['aria-label="Run progress"'],
            )
            html.Span("{{ active_jobs[0].progress_label || 'Working…' }}", classes="cg-progress-label")
            html.Span(
                "{{ active_jobs[0].progress_detail }}",
                v_if="active_jobs[0].progress_detail",
                classes="cg-progress-detail",
            )
        v3.VBtn(RUNS_LABEL, size="small", variant="outlined", prepend_icon="mdi-history", click="runs_dialog = true")


def build_runs_dialog(ctrl):
    with v3.VDialog(
        v_model=("runs_dialog",),
        max_width=560,
        scrollable=True,
        raw_attrs=['aria-labelledby="coilgen-runs-title"'],
    ):
        with v3.VCard(classes="cg-jobs-dialog"):
            with html.Div(classes="cg-dialog-head"):
                html.H2(RUNS_LABEL, id="coilgen-runs-title")
                v3.VBtn("Close", variant="outlined", size="small", click="runs_dialog = false")
            with v3.VCardText(classes="cg-dialog-body"):
                html.H3("Active in this workspace", classes="cg-tree-heading", v_if="active_jobs.length")
                _job_rows("active_jobs", ctrl, active=True)
                v3.VCheckbox(
                    v_if="channels.length",
                    v_model=("workspace_history",),
                    label="Show finished runs from other projects",
                    density="compact",
                    hide_details=True,
                )
                finished = "jobs.filter(j => !['queued', 'running', 'cancelling'].includes(j.status))"
                html.H3("Recent runs", classes="cg-tree-heading", v_if=f"{finished}.length")
                _job_rows(finished, ctrl)
                with html.Details(v_if="older_jobs.length", classes="cg-channels"):
                    html.Summary("Earlier runs ({{ older_jobs.length }})")
                    _job_rows("older_jobs", ctrl)
                html.P("No finished runs in this view.", v_if=f"!{finished}.length", classes="cg-hint")


def _job_rows(expression, ctrl, *, active=False):
    with html.Div(v_for=f"job in {expression}", key="job.id", classes="cg-job"):
        with html.Div(classes="cg-job-heading"):
            html.Strong("{{ job.channel }} · r{{ job.revision }}")
            html.Span(
                "{{ job.caption }}",
                classes="cg-job-state",
                raw_attrs=[':data-state="job.status"'],
            )
        html.Div(
            "{{ job.boundary }} · {{ job.id.slice(0, 8) }}{{ job.embedded ? ' · Archived run' : '' }}",
            classes="cg-hint mb-1",
        )
        if active:
            html.Div("{{ job.stage }}", v_if="job.stage")
            html.Div(
                "{{ job.status === 'queued' ? 'Waiting for a worker' : "
                "job.status === 'cancelling' ? 'Waiting for the worker to stop' : "
                "(job.progress_label || 'Working…') + (job.progress_detail ? ' · ' + job.progress_detail : '') }}",
                classes="cg-hint mb-1",
            )
            v3.VBtn(
                "{{ job.status === 'cancelling' ? 'Stopping…' : 'Cancel run' }}",
                size="small",
                variant="outlined",
                color="error",
                disabled=("job.status === 'cancelling'",),
                title=("'Cancel run ' + job.id + ' for ' + job.channel",),
                click=(ctrl.cancel_job, "[job.id]"),
            )
        else:
            html.P("{{ job.error }}", v_if="job.error", classes="text-error")
            _view_job_result_button(ctrl)


def _view_job_result_button(ctrl):
    v3.VBtn(
        "View result",
        v_if="job.status === 'completed'",
        variant="outlined",
        size="small",
        title=("('View saved result for ' + job.channel + ' · ' + job.id.slice(0, 8))",),
        click=(ctrl.view_job_result, "[job.id]"),
    )
