"""Physical winding choices mapped to the existing numerical configuration."""

from __future__ import annotations

WINDING_INTENTS = {
    "Connected circuit": "single",
    "Separate circuits per support": "per_part",
    "Independent closed turns": "independent",
    "Contours only": "contours",
    "Groups & circuits plan": "groups",
}


def winding_intent(config):
    if config.contour_only_sweep_mode:
        return "Independent closed turns"
    if config.skip_postprocessing:
        return "Contours only"
    if config.routing_group_plan is not None and config.routing_group_plan["mode"] != "existing":
        return "Groups & circuits plan"
    return "Separate circuits per support" if config.routing_circuit_scope == "per_part" else "Connected circuit"


def winding_overrides(config, label):
    """Change topology explicitly, preserving preferences unused by this mode."""
    mode = WINDING_INTENTS[label]
    if mode == "groups":
        if config.routing_group_plan is None:
            raise ValueError("Prepare and apply a Groups & circuits plan first.")
        return {
            "contour_only_sweep_mode": False,
            "skip_postprocessing": False,
            "routing_group_plan": config.routing_group_plan,
        }
    clear = {"routing_group_plan": None} if config.routing_group_plan is not None else {}
    if mode == "independent":
        return {"contour_only_sweep_mode": True, **clear}
    if mode == "contours":
        return {"contour_only_sweep_mode": False, "skip_postprocessing": True, "skip_sweep": True, **clear}
    if config.routing_group_plan is not None:
        clear = {"routing_group_plan": {**config.routing_group_plan, "mode": "existing"}}
    return {"contour_only_sweep_mode": False, "skip_postprocessing": False, "routing_circuit_scope": mode, **clear}
