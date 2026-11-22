"""Browser view of the same retained setup draft used by the desktop."""

from trame.widgets import html, vuetify3 as v3

from ..application.design_setup import DesignSetupDraft, SetupInputError
from ..application import cylinder_pair
from .input_policy import check_browser_inputs


class BrowserDesignSetup:
    def __init__(self, browser):
        self.browser = browser
        self.draft = None
        self.publishing = False
        self.state = browser.server.state
        self.state.update(
            setup_sections=[],
            setup_values={},
            setup_dirty=False,
            setup_error="",
            setup_error_key="",
            setup_open=[],
            setup_focus=None,
            plan_open=False,
        )
        self.state.change("setup_values")(self._changed)
        ctrl = browser.server.controller
        ctrl.preview_setup = self.apply
        ctrl.revert_setup = self.reset
        ctrl.setup_tool = self.open_tool
        ctrl.focus_setup = self.focus

    def synchronize(self):
        project = self.browser.session.project
        if project is None:
            self.draft = None
            self._publish()
        elif self.draft is None or self.draft.project is not project:
            if self.draft is not None and self.draft.project.id != project.id:
                self.state.setup_open = []
            self.draft = DesignSetupDraft(project)
            self.state.setup_error = self.state.setup_error_key = ""
            self._publish()

    def _publish(self):
        self.publishing = True
        try:
            self.state.setup_sections = self.draft.sections() if self.draft else []
            values = dict(self.draft.values) if self.draft else {}
            if self.state.setup_values != values:
                self.state.setup_values = values
            self.state.setup_dirty = bool(self.draft and self.draft.dirty)
        finally:
            self.publishing = False

    def _changed(self, **_):
        if self.publishing or self.draft is None:
            return
        self.draft.update(self.state.setup_values or {})
        self.state.setup_error = self.state.setup_error_key = ""
        self._publish()

    def reset(self):
        self.draft = None
        self.synchronize()

    def apply(self):
        self._changed()
        if self.browser._advanced_has_edits():
            self.state.setup_error = "Apply or cancel the advanced parameter edits first."
            return False
        if self.draft is None or not self.draft.dirty:
            return True
        if self.browser._check_edits():
            self.state.setup_error = "Apply or discard the open specialist editor before applying the setup."
            return False
        draft = self.draft
        if draft.project is not self.browser.session.project:
            self.state.setup_error = "The project changed. Revert the setup edits before continuing."
            return False

        def commit(extra):
            try:
                checked = {
                    key: value
                    for key, value in draft.changes().items()
                    if key not in cylinder_pair.PAIR_KEYS
                    and not (key == "coil_mesh_file" and value == cylinder_pair.SHIELDED_PAIR_SOURCE)
                }
                check_browser_inputs(checked, current_project=draft.project)
                candidate = draft.build(extra_changes=extra)
            except (ValueError, TypeError, KeyError, OSError) as exc:
                self.state.setup_error = str(exc)
                if isinstance(exc, SetupInputError):
                    self.state.setup_error_key = exc.key
                    self.focus(exc.section, exc.key)
                return False
            if candidate.fingerprint != draft.project.fingerprint:
                self.browser.session._replace_project(candidate)
            else:
                self.reset()
            return True

        from ..application.support_switch import support_switch_prompt

        source = draft.values["coil_mesh_file"]
        if support_switch_prompt(draft.project, source) is not None:
            self.browser._confirm_support_switch(source, commit)
            return False
        return commit({})

    def focus(self, section, field=""):
        section = "conductor" if section == "material" else section
        self.state.setup_open = list(dict.fromkeys([*self.state.setup_open, section]))
        self.state.workspace_panel = "model"
        previous = self.state.setup_focus or {}
        self.state.setup_focus = {
            "id": "setup-field-" + field if field else "setup-" + section,
            "sequence": previous.get("sequence", 0) + 1,
        }

    def open_tool(self, key):
        if not self.apply():
            return
        if key == "advanced":
            self.browser.open_advanced()
            self.focus("advanced")
        elif key == "groups":
            self.browser.groups.open()
        elif key in {"geometry", "target", "conductor"}:
            self.state.node = key
            self.browser._refresh_settings()
            self.state.settings_dialog = True


def _field():
    with html.Div(classes="cg-setup-field", id=("'setup-field-' + field.key",)):
        label = ("field.label",)
        update = "setup_values = {...setup_values, [field.key]: $event}; flushState('setup_values')"
        v3.VCheckbox(
            v_if="field.kind === 'bool'",
            model_value=("setup_values[field.key]",),
            label=label,
            update_modelValue=update,
            hide_details=True,
            density="compact",
        )
        v3.VSelect(
            v_else_if="field.kind === 'choice'",
            model_value=("setup_values[field.key]",),
            label=label,
            items=("field.choices",),
            update_modelValue=update,
            variant="outlined",
            density="compact",
            hide_details=True,
        )
        # Text inputs retain incomplete scientific notation until Preview/Compute.
        v3.VTextField(
            v_else=True,
            model_value=("setup_values[field.key]",),
            label=label,
            update_modelValue=update,
            variant="outlined",
            density="compact",
            hide_details=True,
            error=("setup_error_key === field.key",),
        )
        html.P("{{ field.help }}", v_if="field.help && !field.common", classes="cg-setup-help")


def build_design_setup(ctrl):
    html.Div("Design setup", classes="cg-pane-heading")
    with html.Div(classes="cg-setup-scroll", raw_attrs=['v-setup-focus="setup_focus"']):
        with html.Div(v_if="readiness?.state === 'blocking' && !setup_dirty", classes="cg-readiness"):
            html.P("{{ readiness.detail }}", classes="cg-setup-help")
            v3.VBtn(
                "{{ readiness.action }}", size="small", variant="text", click=(ctrl.focus_setup, "[readiness.node]")
            )
        with html.Div(v_if="channels.length > 1", classes="cg-setup-channel"):
            v3.VSelect(
                model_value=("active_channel",),
                update_modelValue="active_channel = $event",
                items=("channels",),
                label="Editing channel",
                density="compact",
                variant="outlined",
                hide_details=True,
            )
        with html.Section(
            v_for="section in setup_sections",
            key="section.key",
            classes="cg-setup-card",
            id=("'setup-' + section.key",),
        ):
            html.H2("{{ section.title }}")
            v3.VBtn(
                "Geometry studio…",
                v_if="section.key === 'geometry'",
                size="small",
                variant="outlined",
                click=ctrl.open_surface_editor,
                disabled=("setup_dirty || settings_dirty",),
            )
            with html.Div(v_for="field in section.fields.filter(f => f.common)", key="field.key"):
                _field()
            v3.VBtn(
                "Details",
                size="small",
                variant="text",
                prepend_icon=("setup_open.includes(section.key) ? 'mdi-chevron-down' : 'mdi-chevron-right'",),
                aria_expanded=("setup_open.includes(section.key)",),
                click="setup_open = setup_open.includes(section.key) ? setup_open.filter(k => k !== section.key) : [...setup_open, section.key]",
            )
            with html.Div(v_show="setup_open.includes(section.key)", classes="cg-setup-details"):
                html.P("{{ section.help }}", classes="cg-setup-help")
                html.P("{{ section.summary }}", v_if="section.summary", classes="cg-setup-help")
                with html.Div(v_for="field in section.fields.filter(f => !f.common)", key="field.key"):
                    _field()
        with html.Details(classes="cg-setup-tools"):
            html.Summary("More tools")
            for key, label in (
                ("geometry", "Import support mesh…"),
                ("target", "Import target map…"),
                ("conductor", "Custom conductor profile…"),
                ("groups", "Groups and circuits…"),
                ("advanced", "Advanced parameters…"),
            ):
                v3.VBtn(label, variant="text", size="small", click=(ctrl.setup_tool, f"['{key}']"))
            with html.Details(classes="cg-channels"):
                html.Summary("Manage channels")
                v3.VCheckbox(
                    v_if="channels.length > 1",
                    model_value=("channel_enabled",),
                    label="Include channel in batch",
                    disabled=("settings_dirty || setup_dirty",),
                    update_modelValue=(ctrl.toggle_channel, "[active_channel, $event]"),
                    hide_details=True,
                    density="compact",
                )
                v3.VBtn(
                    "Duplicate channel",
                    size="small",
                    variant="text",
                    click=ctrl.duplicate_channel,
                    disabled=("settings_dirty",),
                )
                v3.VBtn(
                    "Remove channel",
                    size="small",
                    variant="text",
                    click=ctrl.remove_channel,
                    v_if="channels.length > 1",
                    disabled=("settings_dirty",),
                )
            v3.VBtn(
                "Design progress",
                variant="text",
                size="small",
                click="plan_open = !plan_open; workspace_panel = 'plan'",
            )
            v3.VBtn("Undo", variant="text", size="small", click=ctrl.undo, disabled=("!can_undo || setup_dirty",))
            v3.VBtn("Redo", variant="text", size="small", click=ctrl.redo, disabled=("!can_redo || setup_dirty",))
        build_advanced_settings(ctrl)
    with html.Div(classes="cg-setup-footer"):
        html.P(
            "{{ setup_error || (setup_dirty ? 'Edits pending · Preview or Compute applies the whole setup.' : 'Edit the setup, then preview or compute.') }}",
            role="status",
            classes="cg-setup-help",
        )
        with html.Div(classes="cg-setup-actions"):
            v3.VBtn("Preview", variant="outlined", size="small", click=ctrl.preview_setup)
            v3.VBtn("Revert edits", variant="text", size="small", click=ctrl.revert_setup, disabled=("!setup_dirty",))


def build_advanced_settings(ctrl):
    with html.Section(v_if="advanced_dialog", classes="cg-setup-advanced", id="setup-advanced"):
        with v3.VCard():
            v3.VCardTitle("Advanced settings — every configuration and study field")
            html.Div("{{ advanced_category ? 'Section: ' + advanced_category : '' }}", classes="px-4")
            with v3.VCardText():
                v3.VTextField(
                    v_model="advanced_key_search",
                    label="Search setting or section",
                    density="compact",
                    autofocus=True,
                )
                with html.Div(
                    v_for="entry in advanced_entries.filter(e => !advanced_key_search || (e.key + ' ' + e.category + ' ' + e.aliases).toLowerCase().includes(advanced_key_search.toLowerCase())).slice(0, 40)",
                    key="entry.key",
                ):
                    v3.VTextField(
                        model_value=("advanced_errors[entry.key] ?? entry.value",),
                        label=(
                            "entry.key + ' · ' + entry.category + (entry.key === 'normal_shift_length' ? ' · Return-path lift [m]' : '')",
                        ),
                        error_messages=("entry.key in advanced_errors ? ['Enter a valid JSON value.'] : []",),
                        disabled=("entry.inactive",),
                        hint=("entry.inactive ? 'The active Groups & circuits plan supplies these choices.' : ''",),
                        density="compact",
                        update_modelValue="try { const v = JSON.parse($event); const c = JSON.parse(advanced_config); c[entry.key] = v; advanced_config = JSON.stringify(c,null,2); const errors = {...advanced_errors}; delete errors[entry.key]; advanced_errors = errors; } catch(e) { advanced_errors = {...advanced_errors, [entry.key]: $event ?? ''}; }",
                    )
                v3.VTextarea(
                    v_model=("advanced_config",),
                    update_modelValue="advanced_errors = {}",
                    label="Configuration (JSON)",
                    rows=16,
                    classes="text-caption",
                )
                v3.VTextarea(v_model=("advanced_study",), label="Study (JSON)", rows=6, classes="text-caption")
            with v3.VCardActions():
                v3.VSpacer()
                v3.VBtn("Cancel", click="advanced_dialog = false")
                v3.VBtn(
                    "Apply",
                    color="primary",
                    disabled=("Object.keys(advanced_errors).length > 0",),
                    click=(ctrl.apply_advanced, "[advanced_config, advanced_study, advanced_errors]"),
                )
