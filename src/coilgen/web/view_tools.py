"""Browser viewport tools: body filter, fit to ROI, located check evidence, unwrapped view.

The logic lives in :mod:`coilgen.application.scene_views`; this module only keeps
the browser state and camera in step with it.
"""

from __future__ import annotations

from functools import partial

from ..application.scene_spec import highlight_layers
from ..application.scene_views import (
    BOTH,
    body_view_choices,
    filter_body_view,
    roi_bounds,
    support_roles,
    witness_bounds,
)

VIEW_STATE = {
    "body_view": BOTH,
    "body_views": [],
    # Off by default: the support and conductor solids surround the ROI, so a
    # camera fitted to it sits inside them. The toggle zooms in on request.
    "roi_fit": False,
    "roi_available": False,
    "highlight_note": "",
    "body_note": "",
    "unwrapped_available": False,
    "unwrapped_image": "",
    "unwrapped_note": "",
    "export_groups": [],
    "export_option_group": {},
    "export_presets": [],
    "export_draft_banner": "",
}


def install(workbench):
    """Add the view-tool state, handlers and controller actions to a browser workbench."""
    state, ctrl = workbench.server.state, workbench.server.controller
    workbench.highlight = None
    workbench.fit_bounds = None
    workbench.preview_roles = None
    state.update(dict(VIEW_STATE))
    state.change("body_view")(lambda **_: workbench._render())
    ctrl.toggle_roi_fit = partial(toggle_roi_fit, workbench)
    ctrl.locate_problem = partial(locate_problem, workbench)
    ctrl.clear_highlight = partial(clear_highlight, workbench)
    ctrl.show_unwrapped = partial(show_unwrapped, workbench)


def body_arrays(workbench, arrays, *, current_model):
    """Arrays cut to the chosen bodies; offers the choice only for a primary + shield support."""
    from ..application.evidence import resolve_dataset

    state = workbench.server.state
    state.body_note = ""
    if not arrays:
        state.body_views, state.unwrapped_available = [], False
        return arrays
    from ..application.unwrapped_view import is_cylindrical

    state.unwrapped_available = is_cylindrical(arrays)
    metadata = workbench.record.metadata_view if workbench.record is not None and not current_model else None
    roles = support_roles(arrays, workbench.preview_roles if current_model else None, metadata)
    state.body_views = body_view_choices(roles)
    filtered = filter_body_view(arrays, state.body_view if state.body_views else BOTH, roles)
    key = state.dataset
    if (
        filtered is not arrays
        and resolve_dataset(arrays, key).available
        and not resolve_dataset(filtered, key).available
    ):
        state.body_note = "Nothing of this plot lies on the chosen body; showing both."
        return arrays
    return filtered


def extra_layers(workbench, arrays, key):
    """Highlight layers to draw after the dataset; also chooses what the camera fits."""
    state = workbench.server.state
    bounds = roi_bounds(arrays, key)
    state.roi_available = bounds is not None
    workbench.fit_bounds = bounds if bounds is not None and state.roi_fit else None
    if workbench.highlight is None:
        state.highlight_note = ""
        return []
    points, segments, title = workbench.highlight
    layers = highlight_layers(points, segments)
    if layers:
        workbench.fit_bounds = witness_bounds(layers[0].points)
        state.highlight_note = f"Located: {title}"
    return layers


def toggle_roi_fit(workbench):
    state = workbench.server.state
    state.roi_fit = not state.roi_fit
    workbench._render()


def locate_problem(workbench, points, segments=(), dataset="review_conductor", title=""):
    """Show a failed check's recorded witness in the 3D view, like the desktop's Locate evidence."""
    state = workbench.server.state
    layers = highlight_layers(points, segments)
    if not layers:
        workbench._notify("No spatial witness was recorded for this check.", "warning")
        return
    workbench.highlight = (points, segments or (), str(title or "recorded evidence"))
    state.center_section = "results"
    # The witness lies on or in the support; an opaque support would hide it.
    state.see_through_support = True
    available = {item["value"] for item in state.datasets if not item["props"]["disabled"]}
    if dataset in available:
        state.dataset = dataset
    workbench._render()


def clear_highlight(workbench):
    workbench.highlight = None
    workbench._render()


def show_unwrapped(workbench):
    """Render the unwrapped (φ, z) view of the displayed run or model support."""
    from ..application.unwrapped_view import unwrapped_image

    state = workbench.server.state
    record = workbench.record
    current_model = state.dataset in {"support", "preview_target"}
    if record is not None and not current_model:
        arrays, metadata, roles = record.select_arrays(copy=False), record.metadata_view, None
        title = record.metadata_view.get("name", "Saved run")
    else:
        project = workbench.session.project
        if project is None:
            return
        from ..application.scene_spec import model_preview

        mesh, *_ = model_preview(project.config, project.project_root)
        if mesh is None:
            state.unwrapped_image, state.unwrapped_note = "", "Choose a support surface first."
            state.center_section = "unwrapped"
            return
        arrays = {"vertices_m": mesh.vertices, "faces": mesh.faces}
        metadata, roles, title = None, (getattr(mesh, "metadata", None) or {}).get("part_roles"), project.name
    try:
        state.unwrapped_image, state.unwrapped_note = unwrapped_image(
            arrays, roles=roles, metadata=metadata, title=title
        )
    except (ValueError, KeyError, TypeError) as exc:
        state.unwrapped_image, state.unwrapped_note = "", f"Unwrapped view unavailable · {exc}"
    state.center_section = "unwrapped"


def export_view_state(options):
    """Export list grouped by body role, the format presets and one DRAFT banner."""
    from ..application.cad import (
        EXPORT_PRESETS,
        export_draft_banner,
        export_group,
        export_option_groups,
        export_preset_selection,
    )

    return {
        "export_groups": [{"title": group["title"]} for group in export_option_groups(options)],
        "export_option_group": {option.name: export_group(option) for option in options},
        "export_presets": [
            {"key": key, "title": title, "values": list(values)}
            for key, title, _formats in EXPORT_PRESETS
            if (values := export_preset_selection(options, key))
        ],
        "export_draft_banner": export_draft_banner(options),
    }
