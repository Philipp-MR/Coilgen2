"""Browser adapter for the same revision-bound group draft as the Qt editor."""

import asyncio
from copy import deepcopy

from ..application.group_editor import group_map_image, group_source, load_checkpoint_groups, load_group_editor


class BrowserGroups:
    def __init__(self, browser):
        self.browser = browser
        self.editor = None
        self.invalid_fields = {}
        self.fingerprint = ""
        self.job_id = None
        self.positions = []
        self.rejected = set()
        self.state.update(
            {
                "groups_open": False,
                "groups_busy": False,
                "groups_rows": [],
                "groups_image": "",
                "groups_view": "3D",
                "groups_views": ["3D", "XY", "XZ", "YZ"],
                "groups_dirty": False,
                "groups_invalid": False,
                "groups_stale": False,
                "groups_note": "Prepare groups to inspect the original contour topology.",
                "groups_source": "",
                "groups_eligible": False,
                "groups_waiting": False,
                "groups_counts": "",
                "groups_spirals": [
                    {"title": "Auto", "value": "auto"},
                    {"title": "Outside → inside", "value": "spiral_in"},
                    {"title": "Inside → outside", "value": "spiral_out"},
                ],
                "groups_cuts": ["auto", "high", "low"],
            }
        )
        self.state.change("groups_view")(lambda **_: self.refresh())

    @property
    def state(self):
        return self.browser.server.state

    def open(self):
        self.synchronize(self.browser.session.project)
        self.source()
        self.state.groups_open = True
        self.state.workspace_panel = "center"

    def source(self):
        session = self.browser.session
        state, note = group_source(
            session.project,
            session.states.values(),
            self.state.selected_run or self.browser.follow_job_id,
            self.rejected,
        )
        waiting = bool(
            state
            and state.get("status") in {"queued", "running", "cancelling"}
            and "contours" not in state.get("completed_stages", ())
        )
        self.state.groups_source, self.state.groups_waiting = note, waiting
        self.state.groups_eligible = bool(state and not waiting)
        return state

    async def load(self, record=None):
        browser = self.browser
        if self.editor is not None and (self.editor.dirty or self.invalid_fields):
            self.state.groups_note = "Apply or revert pending group edits before loading another snapshot."
            return
        source = self.source()
        project = browser.session.project
        if project is None or (record is None and (not source or not self.state.groups_eligible)):
            self.state.groups_note = "Prepare groups, or select a completed run containing contours."
            return
        if self.state.groups_busy:
            return
        self.state.groups_busy = True
        self.state.groups_note = "Verifying the contour checkpoint and grouping its topology…"
        self.state.flush()
        try:
            if record is None and source["status"] == "completed":
                record = await asyncio.to_thread(browser.session.load_run, source["id"])
            if record is not None:
                editor = await asyncio.to_thread(load_group_editor, project, record)
            else:
                threads = getattr(browser.session.jobs.policy, "threads_per_worker", 1)
                editor = await asyncio.to_thread(
                    load_checkpoint_groups, project, source, browser.session.jobs_directory, threads
                )
            current = browser.session.project
            if current is None or (current.id, current.project_root, current.fingerprint) != (
                project.id,
                project.project_root,
                project.fingerprint,
            ):
                self.state.groups_note = "The model changed while preparing groups. Load the current snapshot again."
                return
            if self.editor is not None and (self.editor.dirty or self.invalid_fields):
                self.state.groups_note = "Group edits changed while loading. Apply or revert the retained draft first."
                return
            self.editor, self.fingerprint = editor, project.fingerprint
            self.editor.bind(project)
            self.invalid_fields.clear()
            self.state.groups_note = (
                "Source groups changed. Previous assignments were not reapplied; review and apply the new plan."
                if editor.stale_plan
                else "Spiral order preserves current circulation. All circuits use the common evaluation current."
            )
            self.refresh()
        except (OSError, ValueError, TypeError) as exc:
            if source:
                self.rejected.add((project.id, project.fingerprint, source["id"]))
                self.source()
            self.state.groups_note = str(exc)
        finally:
            # Worker polling has no browser event to flush the async result.
            with self.state:
                self.state.groups_busy = False

    def synchronize(self, project):
        if self.editor is None:
            return
        if (
            project is not None
            and project.fingerprint == self.fingerprint
            and (self.editor.binding is None or self.editor.matches_inputs(project))
        ):
            if self.editor.binding is None:
                self.editor.bind(project)
            if self.state.groups_stale:
                self.state.groups_note = "Draft retained for this model revision. Apply or revert the group edits."
            self.state.groups_stale = False
            return
        if project is not None and not self.invalid_fields and self.editor.synchronize(project):
            self.fingerprint = project.fingerprint
            self.state.groups_stale = False
            self.state.groups_note = "Group choices follow the current project revision."
            self.refresh()
        elif self.editor.dirty or self.invalid_fields:
            self.state.groups_stale = True
            self.state.groups_note = "The model changed. Revert this draft, then prepare its current groups."
        else:
            self.editor, self.fingerprint, self.positions = None, "", []
            self.state.groups_rows, self.state.groups_image = [], ""
            self.state.groups_dirty = self.state.groups_invalid = self.state.groups_stale = False
            self.state.groups_note = "The model changed. Prepare its current contour groups."

    def prepare(self):
        if self.editor is not None and (self.editor.dirty or self.invalid_fields):
            self.state.groups_note = "Apply or revert pending group edits first."
            return
        source = self.source()
        if source:
            if self.state.groups_eligible:
                return asyncio.ensure_future(self.load())
            self.state.groups_note = self.state.groups_source
            return
        if not self.browser._ready_to_submit():
            return
        state = self.browser._guard(self.browser.session.compute, "contours", temporary_stage=True)
        if state:
            self.job_id = state["id"]
            self.browser.follow_job_id = self.job_id
            self.state.groups_note = "Preparing groups in the numerical worker…"

    async def poll(self):
        if self.state.groups_open:
            self.source()
        if self.job_id is None:
            return
        job = self.browser.session.states.get(self.job_id, {})
        if job.get("status") == "completed":
            job_id, self.job_id = self.job_id, None
            project = self.browser.session.project
            if project is None or job.get("project_id") != project.id:
                self.source()
                return
            try:
                record = await asyncio.to_thread(self.browser.session.load_run, job_id)
                await self.load(record)
            except (OSError, ValueError, TypeError) as exc:
                with self.state:
                    self.state.groups_note = str(exc)
        elif job.get("status") in {"failed", "cancelled", "stopped", "interrupted"}:
            self.job_id = None
            with self.state:
                self.state.groups_note = "Group preparation did not complete. Inspect the run details."

    def refresh(self):
        if self.editor is None:
            return
        rows = self.editor.rows()
        for row in rows:
            for (key, field), value in self.invalid_fields.items():
                if row["key"] == key:
                    row[field] = value
        self.state.groups_rows = rows
        self.state.groups_counts = self.editor.summary(self.browser.session.project.config.levels)
        self.state.groups_dirty = self.editor.dirty or bool(self.invalid_fields)
        self.state.groups_invalid = bool(self.invalid_fields)
        self.state.groups_image, self.positions = group_map_image(self.editor, self.state.groups_view)

    def select(self, key, checked):
        if self.editor is None:
            return
        if key not in {row["key"] for row in self.editor.rows()}:
            raise ValueError("Unknown group.")
        if checked:
            self.editor.selected.add(key)
        else:
            self.editor.selected.discard(key)
        self.refresh()

    def focus(self, key):
        if self.editor is not None and key in {row["key"] for row in self.editor.rows()}:
            self.editor.focused = key
            self.refresh()

    def pick(self, x, y, width, height):
        if not self.positions or width <= 0 or height <= 0:
            return
        x, y = float(x) * 800 / float(width), float(y) * 370 / float(height)
        key, px, py = min(self.positions, key=lambda item: (item[1] - x) ** 2 + (item[2] - y) ** 2)
        if (px - x) ** 2 + (py - y) ** 2 <= 30**2:
            self.focus(key)

    def change(self, key, field, value):
        if self.editor is None:
            return
        try:
            self.editor.update(key, **{field: value})
            self.invalid_fields.pop((key, field), None)
            self.refresh()
        except ValueError as exc:
            self.invalid_fields[(key, field)] = value
            self.state.groups_dirty = True
            self.state.groups_invalid = True
            self.state.groups_note = str(exc)

    def assign(self, mode):
        if self.editor is not None:
            try:
                self.editor.assign(mode)
                self.refresh()
            except ValueError as exc:
                self.state.groups_note = str(exc)

    def apply(self):
        project = self.browser.session.project
        if self.editor is None or project is None:
            return
        if self.invalid_fields:
            self.state.groups_note = "Correct the invalid group choices before applying the plan."
            return
        if self.state.groups_stale:
            self.state.groups_note = "The model changed. Revert this draft, then prepare its current groups."
            return
        if project.fingerprint != self.fingerprint:
            self.state.groups_note = "The model changed. Revert this draft and prepare its current groups."
            return
        try:
            changes = self.editor.apply_changes(project)
            if changes:
                self.browser.session.apply_config(changes)
            self.fingerprint = self.browser.session.project.fingerprint
            self.editor.bind(self.browser.session.project)
            self.state.groups_stale = False
            self.editor.original = deepcopy(self.editor.plan)
            self.editor.initial_plan = deepcopy(self.editor.plan)
            self.state.groups_note = (
                "Plan applied. Recompute routing or the full design, then review its fields and manufacturing checks."
            )
            self.refresh()
        except (ValueError, TypeError) as exc:
            self.state.groups_note = str(exc)

    def revert(self):
        if self.editor is not None:
            self.editor.discard()
            self.invalid_fields.clear()
            self.synchronize(self.browser.session.project)
            self.refresh()


def build_groups_editor(workbench):
    from trame.widgets import html, vuetify3 as v3

    groups = workbench.groups
    with html.Section(
        v_if="groups_open",
        classes="cg-groups-editor",
        role="region",
        aria_label="Groups and circuits editor",
        __properties=[("aria_label", "aria-label")],
    ):
        with html.Div(classes="cg-buttons"):
            html.Strong("Groups & circuits")
            v3.VBtn(
                "Prepare groups",
                click=groups.prepare,
                disabled=("groups_busy || groups_waiting || groups_eligible",),
                size="small",
            )
            v3.VBtn(
                "Load contour groups", click=groups.load, disabled=("groups_busy || !groups_eligible",), size="small"
            )
            v3.VBtn("Advanced: Routing", click=(workbench.open_advanced, "['Routing']"), size="small")
            v3.VBtn("Back to coil view", click="groups_open = false", size="small")
        html.P("{{ groups_source }}", role="status")
        html.P("{{ groups_note }}", role="status")
        html.Strong("{{ groups_counts }}", v_if="groups_rows.length")
        with html.Div(classes="cg-buttons"):
            v3.VSelect(
                v_model="groups_view",
                items=("groups_views",),
                label="Group plot view",
                density="compact",
                hide_details=True,
            )
            for label, mode in (
                ("Join selected", "join"),
                ("Separate selected", "separate"),
                ("Connect all", "all"),
                ("Separate per group", "per_group"),
            ):
                v3.VBtn(
                    label,
                    click=(groups.assign, f"['{mode}']"),
                    disabled=("!groups_rows.length || groups_busy",),
                    size="small",
                )
        html.Img(
            src=("groups_image",),
            v_if="groups_image",
            alt="Original contour groups with group labels and current arrows",
            style="display:block; width:auto; max-width:100%; max-height:38vh; align-self:center",
            click=(
                groups.pick,
                "[$event.offsetX, $event.offsetY, $event.target.clientWidth, $event.target.clientHeight]",
            ),
        )
        with html.Div(classes="cg-groups-table"):
            with html.Table():
                with html.Thead():
                    with html.Tr():
                        for label in ("Select", "Group / hierarchy", "Contours", "Circuit", "Spiral order", "Cut side"):
                            html.Th(label)
                with html.Tbody():
                    with html.Tr(
                        v_for="row in groups_rows",
                        key="row.key",
                        classes=("row.focused ? 'cg-group-focused' : ''",),
                        raw_attrs=['v-group-focus="row.focused"'],
                    ):
                        with html.Td():
                            html.Input(
                                type="checkbox",
                                checked=("row.selected",),
                                aria_label=("'Select ' + row.label",),
                                __properties=[("aria_label", "aria-label")],
                                change=(groups.select, "[row.key, $event.target.checked]"),
                            )
                        with html.Td():
                            html.Button(
                                "{{ row.label }}", style=("'color:' + row.color",), click=(groups.focus, "[row.key]")
                            )
                            html.Div("{{ row.hierarchy }}", classes="text-caption")
                        html.Td("{{ row.contour_count }}")
                        with html.Td():
                            html.Input(
                                value=("row.circuit",),
                                maxlength=40,
                                aria_label=("row.label + ' circuit'",),
                                __properties=[("aria_label", "aria-label")],
                                change=(groups.change, "[row.key, 'circuit', $event.target.value]"),
                            )
                        with html.Td():
                            v3.VSelect(
                                model_value=("row.spiral",),
                                items=("groups_spirals",),
                                aria_label=("row.label + ' spiral order'",),
                                __properties=[("aria_label", "aria-label")],
                                density="compact",
                                hide_details=True,
                                update_modelValue=(groups.change, "[row.key, 'spiral', $event]"),
                            )
                        with html.Td():
                            v3.VSelect(
                                model_value=("row.cut",),
                                items=("groups_cuts",),
                                aria_label=("row.label + ' cut side'",),
                                __properties=[("aria_label", "aria-label")],
                                density="compact",
                                hide_details=True,
                                update_modelValue=(groups.change, "[row.key, 'cut', $event]"),
                            )
        with html.Div(classes="cg-buttons"):
            v3.VBtn(
                "Apply routing plan",
                click=groups.apply,
                disabled=("!groups_dirty || groups_busy || groups_invalid || groups_stale",),
                color="primary",
                size="small",
            )
            v3.VBtn("Revert group edits", click=groups.revert, disabled=("!groups_dirty",), size="small")
            html.Span("Common evaluation current · external leads require separate review")
