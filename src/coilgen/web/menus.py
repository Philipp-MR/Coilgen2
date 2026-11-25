"""Top-level commands; unavailable inputs lead to an actionable explanation."""

from inspect import isawaitable

from trame.widgets import html, vuetify3 as v3


MENUS = {
    "File": (
        ("new", "New cylindrical design"),
        ("examples", "Browse examples…"),
        ("open", "Open project…"),
        ("save", "Save / download…"),
        ("export", "Export geometry…"),
        ("pcb", "PCB layout / export…"),
    ),
    "Edit": (("undo", "Undo"), ("redo", "Redo"), ("revert", "Revert setup edits")),
    "Model": (
        ("surface", "Geometry studio…"),
        ("geometry", "Import support mesh…"),
        ("target", "Import target map…"),
        ("conductor", "Custom conductor profile…"),
        ("groups", "Groups and circuits…"),
        ("advanced", "Advanced settings…"),
        ("duplicate", "Duplicate active channel"),
        ("remove", "Remove active channel"),
    ),
    "Run": (
        ("compute", "Compute selected stage"),
        ("batch", "Compute enabled channels independently"),
        ("runs", "Runs and cancellation…"),
    ),
    "Results": (
        ("metrics", "Coil results"),
        ("checks", "Engineering checks"),
        ("slices", "Residual slices…"),
        ("harmonics", "Harmonics…"),
        ("shielding", "Shielding surface…"),
        ("winding", "Edit interconnections…"),
        ("pcb", "PCB layout…"),
        ("submitted", "Submitted design choices…"),
    ),
    "View": (
        ("setup", "Design setup"),
        ("plan", "Design progress"),
        ("plot", "Plot and results"),
        ("runs", "Runs…"),
        ("fit", "Fit view"),
    ),
    "Help": (("help", "Workbench guide…"),),
}


def build_menus(ctrl):
    with html.Nav(classes="cg-top-menus", aria_label="Workbench menus", __properties=[("aria_label", "aria-label")]):
        for name, entries in MENUS.items():
            with v3.VBtn(variant="text", size="small", append_icon="mdi-menu-down"):
                html.Span(name)
                with v3.VMenu(activator="parent"):
                    with v3.VList(density="compact"):
                        for key, label in entries:
                            v3.VListItem(title=label, click=(ctrl.menu_action, f"['{key}']"))


async def menu_action(browser, key):
    state = browser.server.state
    if key in {"examples", "open"}:
        state.file_section, state.file_dialog = key, True
        return
    if key == "runs":
        state.runs_dialog = True
        return
    if key == "help":
        state.menu_help = True
        return
    if key == "new":
        browser.new_project()
        return
    if browser.session.project is None and key in {
        "save",
        "surface",
        "advanced",
        "geometry",
        "target",
        "conductor",
        "groups",
        "duplicate",
        "remove",
        "compute",
        "batch",
        "undo",
        "redo",
        "revert",
        "setup",
        "plan",
    }:
        browser._notify("Create or open a project from File first.", "info")
        state.file_section, state.file_dialog = "examples", True
        return
    if key in {"metrics", "checks", "slices", "harmonics", "shielding", "winding", "submitted", "export", "pcb"}:
        if browser.record is None or state.selected_run != state.displayed_run:
            browser._notify("Choose View results for a completed run first.", "info")
            state.runs_dialog = True
            return
    if key in {"save", "setup", "plan", "plot", "revert", "export"}:
        if key == "save":
            state.file_section, state.file_dialog = "save", True
        elif key == "revert":
            browser.setup.reset()
        elif key == "export":
            state.workspace_panel, state.center_section = "center", "export"
        else:
            state.workspace_panel = {"setup": "model", "plan": "plan", "plot": "center"}[key]
            if key == "plan":
                state.plan_open = True
        return
    if key in {"metrics", "checks"}:
        state.workspace_panel, state.center_section = "center", key
        return
    if key in {"geometry", "target", "conductor", "groups"}:
        browser.setup.open_tool(key)
        return
    if key == "remove" and len(browser.session.assembly.channels) <= 1:
        browser._notify("Keep at least one channel in the project. Use File to start another design.", "info")
        return
    actions = {
        "surface": browser.surface.open,
        "advanced": browser.open_advanced,
        "undo": browser.undo,
        "redo": browser.redo,
        "duplicate": browser.duplicate_channel,
        "remove": browser.remove_channel,
        "compute": browser.compute,
        "batch": browser.compute_channels,
        "slices": browser.field_slices.open,
        "harmonics": browser.harmonics.open,
        "shielding": lambda: browser.show_shielding(
            state.shielding_source, state.shielding_component, state.shielding_detail
        ),
        "winding": browser.winding.open,
        "pcb": lambda: browser.pcb.open(),
        "submitted": browser.review_submitted_design,
        "fit": browser.server.controller.fit_browser_view,
    }
    action = actions.get(key)
    if action is not None:
        result = action()
        if isawaitable(result):
            await result


def build_help():
    with v3.VDialog(v_model=("menu_help", False), max_width=580):
        with v3.VCard():
            v3.VCardTitle("Workbench guide")
            with v3.VCardText():
                html.P(
                    "File manages projects. Model edits the support, target, conductor and channels. Run computes designs. Results reviews a saved run. View opens panels."
                )
                html.P(
                    "An editor with pending changes stays available so you can apply or discard its draft. Saved results keep their original model and current."
                )
                html.A(
                    "Full browser guide",
                    href="https://github.com/Philipp-MR/Coilgen2/blob/main/docs/BROWSER_WORKBENCH.md",
                    target="_blank",
                    rel="noopener",
                )
            with v3.VCardActions():
                v3.VSpacer()
                v3.VBtn("Close", click="menu_help = false")
