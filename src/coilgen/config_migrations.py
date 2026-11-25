"""Explicit historical configuration migrations; never inherit today's defaults."""

from __future__ import annotations

from dataclasses import fields
from typing import Any
from types import MappingProxyType

from .config import CoilGenConfig

CONFIG_VERSION = 1
# These opt-in fields had no effect before they were recorded in project inputs.
PROJECT_OPT_IN_DEFAULTS = MappingProxyType(
    {
        "target_region_geometry": None,
        "contour_smoothing_local_fallback": False,
        "routing_editor_plan": None,
    }
)


def migrate_project_config(values, *, version=0):
    if type(version) is not int or not 0 <= version <= CONFIG_VERSION:
        raise ValueError("Unsupported project configuration version.")
    migrated = dict(values)
    # Version zero is the unversioned, fully resolved project-v1/v2 format.
    if version == 0:
        for name, historical_default in PROJECT_OPT_IN_DEFAULTS.items():
            migrated.setdefault(name, historical_default)
    return migrated


def project_identity_values(values):
    """Inactive additions retain the identity of pre-extension project evidence."""
    identity = dict(values)
    for name, default in PROJECT_OPT_IN_DEFAULTS.items():
        if identity.get(name) is default:
            identity.pop(name, None)
    return identity


def migrate_checkpoint_payload(value: Any, schema: str) -> Any:
    """Add only the known missing config fields to authenticated old evidence.

    Version 2 used physical selection; version 3 recorded its own policy. Keep
    those policies and disable new port searches on historical evidence.
    Unknown fields and all other missing fields still fail the strict decoder.
    Original stage/source fingerprints are retained, so migration grants no
    permission to resume an old numerical result under a different engine.
    """
    if isinstance(value, list):
        return [migrate_checkpoint_payload(item, schema) for item in value]
    if not isinstance(value, dict):
        return value
    migrated = {key: migrate_checkpoint_payload(item, schema) for key, item in value.items()}
    if value.get("$class") == "PipelineState" and "shielding_arrays" not in migrated["fields"]:
        migrated["fields"]["shielding_arrays"] = {"$dict": []}
    if value.get("$class") == "ConnectedGroup":
        migrated["fields"].setdefault("routing_spiral_order", "auto")
    if value.get("$class") in {"PipelineState", "CoilGenResult"}:
        migrated["fields"].setdefault("source_parts", {"$list": []})
    if value.get("$class") == "CoilGenConfig":
        migrated["fields"].setdefault("routing_group_plan", None)
        migrated["fields"].setdefault("routing_editor_plan", None)
    if value.get("$class") == "CoilPart":
        migrated["fields"].setdefault("routing_editor_baseline", None)
    if value.get("$class") == "CoilGenConfig" and schema not in {
        "coilgen.pipeline-checkpoint/5",
        "coilgen.pipeline-checkpoint/6",
        "coilgen.pipeline-checkpoint/7",
        "coilgen.pipeline-checkpoint/8",
        "coilgen.pipeline-checkpoint/9",
    }:
        additions = (
            {}
            if schema == "coilgen.pipeline-checkpoint/4"
            else {
                # Loading old evidence must not retroactively enable new port searches.
                "joint_group_routing_port_search_count": 0,
                "joint_group_routing_cut_span_max_factor": 1.0,
                "joint_group_routing_seam_search_extent_rel": 0.0,
            }
        )
        if schema == "coilgen.pipeline-checkpoint/2":
            additions.update(
                joint_group_routing_selection_policy="physical", joint_group_routing_regular_field_tolerance_rel=0.001
            )
        attributes = migrated.get("fields")
        if isinstance(attributes, dict) and "contour_smoothing_local_fallback" in attributes:
            raise ValueError("Checkpoint data-class fields do not match the historical schema.")
        additions["contour_smoothing_local_fallback"] = False
        expected_legacy = {item.name for item in fields(CoilGenConfig)} - additions.keys()
        if isinstance(attributes, dict) and set(attributes) == expected_legacy:
            migrated["fields"] = {**attributes, **additions}
    return migrated
