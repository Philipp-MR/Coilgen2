"""Browser presentation of the desktop workbench's project, model, and plan panels."""

from base64 import b64encode

from trame.widgets import html, vuetify3 as v3
from trame_client.widgets.core import HtmlElement

from ..application.preflight import SECTION_ORDER, section_readiness
from ..application.session import SETTINGS_NODES
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
    sections = section_readiness(report, nodes=SECTION_ORDER) if report is not None else {}
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
            "{{ stage === 'evaluated' ? 'Compute full design' : stages.find(s => s.value === stage)?.title }}",
            v_if="channels.length",
            color="primary",
            prepend_icon="mdi-play-outline",
            classes="cg-compute",
            disabled=("current_job_active || settings_dirty || preflight.some(i => i.blocking)",),
            title="Compute the editing channel to the boundary selected in Design plan",
            click=ctrl.compute,
        )


def build_welcome(ctrl):
    with html.Main(v_if="!channels.length && !history_open", classes="cg-welcome"):
        _wordmark("large")
        html.H1("CoilGen² workbench", classes="cg-sr-only")
        html.P("Choose a gradient, shim coil or coil system in Browse examples.")
        html.P("After loading a design, click Compute full design.")
        with html.Div(classes="cg-welcome-actions"):
            v3.VBtn(
                "Browse examples…",
                color="primary",
                prepend_icon="mdi-folder-multiple-outline",
                click="file_section = 'examples'; file_dialog = true",
            )
            v3.VBtn(
                "Open project…",
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
            v3.VBtn(
                "Review workspace runs",
                v_if="runs.length || jobs.length",
                variant="text",
                click="history_open = true; workspace_panel = 'center'; center_section = 'history'",
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
    html.H2("Model Builder", classes="cg-pane-heading")
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
            for group, keys in (
                ("Model", "['geometry','target','conductor','winding']"),
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
                                v_if="['blocking', 'warning'].includes(item.state)",
                                classes="cg-tree-status",
                            )
                        v3.VIcon(
                            "{{ item.state === 'blocking' ? 'mdi-alert-circle' : 'mdi-alert-outline' }}",
                            v_if="['blocking', 'warning'].includes(item.state)",
                            size="small",
                            classes="cg-tree-state",
                            __properties=[("aria_hidden", "aria-hidden")],
                            aria_hidden="true",
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
        with html.Div(v_if="channels.length || runs.length || jobs.length"):
            html.H3("Results and provenance", classes="cg-tree-heading")
            with html.Button(
                type="button",
                classes="cg-tree-item",
                click="workspace_panel = 'center'; center_section = 'history'",
            ):
                html.Img(src=_tree_symbol("run-history"), alt="", classes="cg-tree-symbol")
                html.Span("Completed runs")
            with html.Button(
                v_if="export_options.length",
                type="button",
                classes="cg-tree-item",
                click="workspace_panel = 'center'; center_section = 'export'",
            ):
                html.Img(src=_tree_symbol("export-conductor"), alt="", classes="cg-tree-symbol")
                html.Span("Export conductor geometry")


def build_design_plan(ctrl):
    html.H2("Design plan", classes="cg-pane-heading")
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
        with html.Div(v_if="jobs.length", classes="cg-job-list"):
            html.H3("Jobs", classes="cg-tree-heading")
            html.P("Active jobs and recent attempts, newest first within each group.", classes="cg-hint")
            with html.Div(v_for="job in jobs", key="job.id", classes="cg-job"):
                html.Strong("{{ job.channel }} · r{{ job.revision }}")
                html.Div("{{ job.boundary }} · {{ job.status }} · {{ job.id.slice(0, 8) }}")
                html.Div("{{ job.stage }}")
                v3.VProgressLinear(
                    v_if="['queued', 'running', 'cancelling'].includes(job.status)",
                    model_value=("job.progress",),
                    color="primary",
                    height=5,
                )
                html.P("{{ job.error }}", v_if="job.error", classes="text-error")
                _view_job_result_button(ctrl)
                v3.VBtn(
                    "{{ job.status === 'cancelling' ? 'Stopping…' : 'Cancel' }}",
                    size="small",
                    variant="outlined",
                    v_if="['queued', 'running', 'cancelling'].includes(job.status)",
                    disabled=("job.status === 'cancelling'",),
                    title=("('Cancel job ' + job.id + ' for ' + job.channel)",),
                    click=(ctrl.cancel_job, "[job.id]"),
                )
            with html.Details(v_if="older_jobs.length", classes="cg-channels"):
                html.Summary("Earlier attempts ({{ older_jobs.length }})")
                with html.Div(v_for="job in older_jobs", key="job.id", classes="cg-job"):
                    html.Strong("{{ job.channel }} · r{{ job.revision }}")
                    html.Div("{{ job.boundary }} · {{ job.status }} · {{ job.id.slice(0, 8) }}")
                    html.P("{{ job.error }}", v_if="job.error", classes="text-error")
                    _view_job_result_button(ctrl)
        html.P("Open a project to see its requested steps.", v_if="!channels.length", classes="cg-hint")


def _view_job_result_button(ctrl):
    v3.VBtn(
        "View result",
        v_if="job.status === 'completed'",
        variant="outlined",
        size="small",
        title=("('View saved result for ' + job.channel + ' · ' + job.id.slice(0, 8))",),
        click=(ctrl.view_job_result, "[job.id]"),
    )
