"""Toolkit-independent labels and colors for the desktop and browser workbenches."""

COMPUTE_CAPTIONS = {
    "prepared": "Prepare model",
    "sheet": "Optimize stream function",
    "contours": "Generate contours",
    "routed": "Connect / sweep",
    "evaluated": "Compute full design",
}

MODEL_BUILDER_TITLE = "Model Builder"
DESIGN_PLAN_TITLE = "Design plan"
BROWSE_EXAMPLES_LABEL = "Browse examples…"
OPEN_PROJECT_LABEL = "Open project…"
RUNS_LABEL = "Runs"
FINALIZING_FABRICATION = "Finalizing fabrication"


def is_finalizing_fabrication(state) -> bool:
    """Identify strict review after the worker has recorded all design milestones."""
    recorded = state.get("completed_stages")
    return (
        state.get("status") == "running"
        and state.get("pipeline_stage") == "evaluated"
        and state.get("fabricate") is True
        and state.get("progress_operation") == "Strict fabrication review"
        and isinstance(recorded, (tuple, list))
        and all(key in recorded for key in COMPUTE_CAPTIONS)
    )


ACTION_COLORS = {
    "primary": "#075aaa",
    "action-hover": "#064b8e",
    "action-pressed": "#04386d",
    "action-focus": "#002d5c",
    "action-disabled": "#dce3eb",
    "action-disabled-text": "#596777",
    "action-disabled-border": "#b9c5d2",
}

# Semantic foregrounds for light and dark palettes. Keep a text label alongside each color.
STATUS_COLORS = {
    "info": (ACTION_COLORS["primary"], "#87bcff"),
    "success": ("#08766c", "#76d1bb"),
    "warning": ("#8b5900", "#f3ca77"),
    "error": ("#b32e28", "#ffaaa4"),
}
