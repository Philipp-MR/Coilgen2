"""Display quantities and check outcomes from a verified run, without recomputing them."""

import numpy as np

from ..application.diagnostics import recorded_stage, run_problems
from ..gui.coil_statistics import build_coil_statistics
from ..gui.result_summary import coil_metric_cards, field_metric_cards, summarize_field_result


_METRICS = (
    ("relative_field_error", "Stream function relative field error (design samples)", 100, "%"),
    ("validated_relative_field_error", "Stream function relative field error (independent samples)", 100, "%"),
    ("power_w", "Sheet Joule power", 1, "W"),
    ("energy_j", "Sheet magnetic energy", 1, "J"),
    ("max_current_density_a_per_m", "Peak surface current density", 1, "A/m"),
    ("current_a", "Fabricated circuit current", 1, "A"),
    ("resistance_ohm", "Fabricated circuit resistance", 1e3, "mΩ"),
    ("inductance_h", "Fabricated circuit inductance", 1e6, "µH"),
    ("fabricated_power_w", "Fabricated conductor Joule power", 1, "W"),
    ("wire_length_m", "Total fabricated wire length", 1, "m"),
    ("minimum_spacing_m", "Minimum unconnected contour spacing (surface distance)", 1e3, "mm"),
    ("sheet_thickness_m", "Continuous-sheet thickness", 1e3, "mm"),
    ("conductor_resistivity_ohm_m", "Conductor resistivity", 1e9, "nΩ·m"),
    ("b0_t", "Mechanical model background field (x, y, z)", 1, "T"),
    ("torque_origin_m", "Torque origin (x, y, z)", 1e3, "mm"),
    ("force_n", "Sheet force (x, y, z)", 1, "N"),
    ("torque_nm", "Sheet torque (x, y, z)", 1, "N·m"),
    ("fabricated_force_n", "Fabricated circuit force (x, y, z)", 1, "N"),
    ("fabricated_torque_nm", "Fabricated circuit torque (x, y, z)", 1, "N·m"),
    ("vertices", "Support vertices", 1, ""),
    ("triangles", "Support triangles", 1, ""),
    ("contour_components", "Unconnected contour components", 1, ""),
    ("elapsed_seconds", "Computation time", 1, "s"),
)


def _quantity(value, scale, unit):
    if value is None:
        return None
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or array.shape not in {(), (3,)} or not np.all(np.isfinite(array)):
        return None
    numbers = [f"{number * scale:.6g}" for number in array.reshape(-1)]
    text = numbers[0] if len(numbers) == 1 else "(" + ", ".join(numbers) + ")"
    return text + (" " + unit if unit else "")


def result_metrics(metadata):
    summary = metadata.get("summary") or {}
    rows = []
    for key, label, scale, unit in _METRICS:
        value = _quantity(summary.get(key), scale, unit)
        if value is None and key in {"relative_field_error", "validated_relative_field_error"}:
            value = "Not run" if recorded_stage(metadata) == "prepared" else "Not recorded"
        if value is not None:
            rows.append({"name": label, "value": value})
    return rows


def reviewed_wire_metric(metadata):
    """Use the field measured on the reviewed conductor, at its recorded current policy."""
    field = (metadata.get("fabrication") or {}).get("field_metrics") or {}
    error = field.get("dsv_relative_residual_percent")
    if not isinstance(error, (int, float)) or isinstance(error, bool) or not np.isfinite(error) or error < 0:
        return None
    scope = (
        "design DSV, original current"
        if field.get("current_source") == "original_contour_step_no_refit"
        else "design DSV"
    )
    return {"name": f"Reviewed wire relative RMS field error ({scope})", "value": f"{error:.4g} %"}


def result_overview(record):
    """Use the desktop's immutable readouts, including per-body and current scope."""
    field = summarize_field_result(record)
    coil = build_coil_statistics(record)
    cards, scope, scope_tooltip = coil_metric_cards(coil)
    field_cards = field_metric_cards(field) if field.visible else []
    coil_cards = cards if coil.has_values else []
    stage = recorded_stage(record.metadata_view)
    partial_note = ""
    boundaries = {
        "prepared": "Prepare model stops before stream function optimization.",
        "sheet": "Optimize stream function stops before contour generation.",
        "contours": "Generate contours stops before conductor routing.",
        "routed": "Connect / sweep conductors stops before full design evaluation.",
    }
    if stage in boundaries:
        if not any(not card["unavailable"] or card["value"] == "Per body" for card in field_cards):
            field_cards = []
        if not any(not card["unavailable"] or card["value"] == "Per body" for card in coil_cards):
            coil_cards = []
        if not field_cards and not coil_cards:
            partial_note = boundaries[stage]
    return {
        "field_cards": field_cards,
        # These datasets show the exact current policy used for the scalar board.
        # Other wire datasets may display a fitted current in older records.
        "field_plot": field.source_key
        if field.source_key in {"fabricated", "winding_validation", "validation", "sheet"}
        else "",
        "field_component": field.component if field.component in {"Bx", "By", "Bz", "BdotB0", "Bmag"} else "",
        "field_context": field.context if field.visible else "",
        "field_tooltip": field.tooltip,
        "coil_cards": coil_cards,
        "partial_note": partial_note,
        "coil_context": scope,
        "coil_tooltip": scope_tooltip,
        "body_rows": [
            {
                "body": row.body_id,
                "name": row.quantity,
                "value": _quantity(row.value, 1, row.unit) if row.unit else str(row.value),
                "scope": row.scope,
            }
            for row in coil.rows
            if row.body_id and row.value is not None
        ],
    }


def problem_groups(record):
    groups = {}
    for issue in run_problems(record):
        if issue.key.startswith("constraint:"):
            scope = "Numerical and engineering constraints"
        elif issue.scope == "pipeline_sweep_geometry":
            scope = "Pipeline conductor sweep"
        elif issue.scope == "fabrication_review":
            scope = "Strict fabrication review"
        else:
            scope = "Recorded warnings"
        status = (
            "Failed"
            if issue.severity == "error"
            else "Unverified"
            if issue.key.startswith("constraint:")
            else "Warning"
        )
        unit, scale = ("mm", 1e3) if issue.unit == "m" else (issue.unit, 1)
        measured = _quantity(issue.value, scale, unit)
        limit = _quantity(issue.limit, scale, unit)
        details = [status, issue.body_id]
        if measured is not None:
            details.append("Measured " + measured)
        if limit is not None:
            details.append("Limit " + limit)
        groups.setdefault(scope, []).append(
            {
                "title": issue.title,
                "status": status,
                "details": " · ".join(x for x in details if x),
                "explanation": issue.explanation,
            }
        )
    result = []
    for name, items in groups.items():
        counts = []
        for status in ("Failed", "Unverified", "Warning"):
            count = sum(item["status"] == status for item in items)
            if count:
                label = "warnings" if status == "Warning" and count != 1 else status.lower()
                counts.append(f"{count} {label}")
        result.append({"title": name, "counts": " · ".join(counts), "items": items})
    return result
