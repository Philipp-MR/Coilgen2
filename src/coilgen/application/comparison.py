"""Side-by-side evidence with explicit sampling, current and geometry scopes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from ..field_components import normalize_field_component, normalized_b0_direction
from ..pipeline_stages import config_field_stage
from .service import DesignService, RunRecord
from .evidence import read_verified_artifact


@dataclass(frozen=True, slots=True)
class ParameterDelta:
    field: str
    left: object
    right: object


@dataclass(frozen=True, slots=True)
class ComparisonMetric:
    name: str
    unit: str
    left: float | None
    right: float | None
    left_scope: str
    right_scope: str
    comparable: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class RunComparison:
    left: RunRecord
    right: RunRecord
    parameter_deltas: tuple[ParameterDelta, ...]
    metrics: tuple[ComparisonMetric, ...]
    notes: tuple[str, ...]


def _request(record):
    if not any(item.get("name") == "request.json" and item.get("sha256") for item in record.metadata["artifacts"]):
        raise ValueError("Comparison requires an integrity-checked original request.json artifact.")
    request = json.loads(read_verified_artifact(record, "request.json"))
    if not isinstance(request, dict) or not isinstance(request.get("config"), dict):
        raise ValueError("Original run request has no configuration mapping.")
    return request


def _flatten(values, prefix=""):
    rows = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and value:
            rows.update(_flatten(value, name))
        else:
            rows[name] = value
    return rows


def _finite(value, factor=1):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
        converted = float(value) * factor
        return converted if np.isfinite(converted) else None
    return None


def _same_data(left, right, keys):
    return all(key in left and key in right and np.array_equal(left[key], right[key]) for key in keys)


def compare_runs(left: RunRecord | str | Path, right: RunRecord | str | Path) -> RunComparison:
    """Reverify evidence and report differences without recalculation or refitting.

    Different sampling/current/geometry contracts are shown side by side with
    reasons; they are not labeled as improvement or independent acceptance.
    Missing R/L remains unavailable rather than inferred from wire length.
    """
    try:
        left = DesignService.load_run(left.directory if isinstance(left, RunRecord) else left)
        right = DesignService.load_run(right.directory if isinstance(right, RunRecord) else right)
    except (AttributeError, EOFError, BadZipFile) as exc:
        raise ValueError(f"Cannot read comparison run evidence: {exc}") from exc
    requests = (_request(left), _request(right))
    configs = tuple(request["config"] for request in requests)
    resolved = tuple(
        json.loads((record.directory / "manifest.json").read_text(encoding="utf-8")) for record in (left, right)
    )
    notes = ["Design-grid wire fields are descriptive thin-wire evidence; they do not claim independent acceptance."]
    values = []
    for request in requests:
        config = {key: value for key, value in request["config"].items() if config_field_stage(key) is not None}
        values.append(
            _flatten(config, "config")
            | _flatten(
                {
                    key: request.get(key)
                    for key in (
                        "pipeline_stage",
                        "validation_resolution",
                        "design_margin",
                        "fabricate",
                        "kernel_backend",
                        "resume_checkpoint",
                        "engineering",
                        "requirements",
                        "scene",
                        "dynamic_analysis",
                        "array_study",
                    )
                },
                "study",
            )
        )
    deltas = []
    for key in sorted(values[0].keys() | values[1].keys()):
        first, second = values[0].get(key), values[1].get(key)
        if first == second:
            continue
        config_key = key.removeprefix("config.")
        inputs = [manifest.get("input_files", {}).get(config_key) for manifest in resolved]
        if all(inputs) and inputs[0].get("sha256") == inputs[1].get("sha256"):
            notes.append(f"{config_key}: source locations differ but input content is identical.")
            continue
        deltas.append(ParameterDelta(key, first, second))
    inputs = [manifest.get("input_files", {}) for manifest in resolved]
    for key in sorted(inputs[0].keys() | inputs[1].keys()):
        hashes = tuple(value.get(key, {}).get("sha256") for value in inputs)
        if hashes[0] != hashes[1]:
            deltas.append(ParameterDelta(f"inputs.{key}.sha256", *hashes))
    metadata = (left.metadata, right.metadata)
    arrays = (left.select_arrays(copy=False), right.select_arrays(copy=False))
    summaries = tuple(value.get("summary", {}) for value in metadata)
    validations = tuple(value.get("validation", {}) for value in metadata)
    component = tuple(normalize_field_component(config.get("target_field_component", "Bz")) for config in configs)
    same_component = component[0] == component[1]
    if same_component and component[0] == "BdotB0":
        same_component = np.array_equal(
            *[normalized_b0_direction(config.get("b_0_direction", [0, 0, 1])) for config in configs]
        )
    same_design = same_component and _same_data(
        *arrays, ("target_points_m", "target_field_T", "target_weights", "target_suppressed")
    )
    same_independent = same_component and _same_data(*arrays, ("validation_points_m", "validation_target_T"))
    metrics = []

    def metric(name, unit, first, second, scopes, comparable, reason="", factor=1):
        first, second = _finite(first, factor), _finite(second, factor)
        if first is None or second is None:
            comparable, reason = False, "A value is unavailable or unverified."
        elif comparable:
            reason = ""
        metrics.append(ComparisonMetric(name, unit, first, second, *scopes, comparable, reason))

    design_scope = tuple(f"continuous sheet; design grid; component={value}" for value in component)
    metric(
        "Sheet design relative error",
        "%",
        *(value.get("relative_field_error") for value in summaries),
        design_scope,
        same_design,
        "Target values, samples, weights or selected components differ.",
        factor=100,
    )
    independent_scope = tuple(
        f"continuous sheet; independent grid; component={component[index]}; n={value.get('point_count', 'unknown')}; "
        f"current unchanged; quadrature check={value.get('independent_quadrature_check', False)}"
        for index, value in enumerate(validations)
    )
    for name, key, unit, factor in (
        ("Sheet independent RMS error", "rms_error_T", "µT", 1e6),
        ("Sheet independent relative error", "relative_field_error", "%", 100),
        ("Sheet independent peak error", "peak_error_T", "µT", 1e6),
    ):
        metric(
            name,
            unit,
            *(value.get(key) for value in validations),
            independent_scope,
            same_independent,
            "Independent target samples or selected components differ.",
            factor=factor,
        )
    for kind, title in (("contour", "Contour"), ("routed", "Routed")):
        evidence = tuple(value.get("wire_fields", {}).get(kind, {}) for value in metadata)
        scopes = tuple(
            f"design-grid wire field; component={value.get('component', component[index])}; "
            f"n={value.get('sample_count', 'unknown')}; current={value.get('current_source', 'unknown')}; "
            f"I={value.get('current_A', 'unknown')} A; "
            f"refitted={value.get('current_refitted', 'unknown')}; "
            f"geometry={value.get('geometry_sources', value.get('geometry_source', 'unknown'))}; "
            "independent acceptance=False"
            for index, value in enumerate(evidence)
        )
        policy_keys = (
            "current_source",
            "current_refitted",
            "geometry_sources",
            "geometry_source",
            "field_model",
            "rms_convention",
        )
        same_policy = all(evidence[0].get(key) == evidence[1].get(key) for key in policy_keys)
        same = same_design and same_policy and all(value.get("available") is True for value in evidence)
        reason = "Target grid/component or current/refitting/geometry policy differs."
        for suffix, key, unit, factor in (
            ("current", "current_A", "A", 1),
            ("RMS error", "rms_error_T", "µT", 1e6),
            ("relative error", "relative_rms_error", "%", 100),
            ("peak error", "peak_error_T", "µT", 1e6),
        ):
            metric(
                f"{title} {suffix}", unit, *(value.get(key) for value in evidence), scopes, same, reason, factor=factor
            )
        original = tuple(value.get("at_stream_current", {}) for value in evidence)
        original_scope = tuple(
            f"design-grid wire field; component={evidence[index].get('component', component[index])}; "
            f"n={value.get('sample_count', 'unknown')}; current={value.get('current_source', 'unknown')}; "
            f"I={value.get('current_A', 'unknown')} A; "
            f"refitted={value.get('current_refitted', 'unknown')}; "
            f"geometry={evidence[index].get('geometry_sources', evidence[index].get('geometry_source', 'unknown'))}; "
            "independent acceptance=False"
            for index, value in enumerate(original)
        )
        original_same = (
            same_design
            and all(value.get("available") is True for value in original)
            and all(
                evidence[0].get(key) == evidence[1].get(key)
                for key in ("geometry_sources", "geometry_source", "field_model")
            )
        )
        metric(
            f"{title} at original current RMS error",
            "µT",
            *(value.get("rms_error_T") for value in original),
            original_scope,
            original_same,
            "Target grid/component or evaluated geometry policy differs.",
            factor=1e6,
        )
    for name, key, unit, factor in (
        ("Circuit current", "current_a", "A", 1),
        ("Circuit resistance", "resistance_ohm", "Ω", 1),
        ("Circuit inductance", "inductance_h", "µH", 1e6),
    ):
        scope = tuple(
            f"{summary.get('scope', 'unknown')}; single complete modeled circuit only" for summary in summaries
        )
        same = all(summary.get("scope") == "fabrication_review" for summary in summaries)
        metric(
            name,
            unit,
            *(summary.get(key) for summary in summaries),
            scope,
            same,
            "Complete circuit scope differs.",
            factor=factor,
        )
    metric(
        "Sheet power",
        "W",
        *(summary.get("power_w") for summary in summaries),
        ("continuous sheet before contouring/routing",) * 2,
        True,
    )
    winding = tuple(value.get("winding_validation") or {} for value in metadata)
    same_winding_grid = same_component and _same_data(
        *arrays, ("winding_validation_points_m", "winding_validation_target_T")
    )
    same_winding_policy = all(
        winding[0].get(key) == winding[1].get(key)
        for key in ("geometry_source", "current_source", "current_refitted", "field_model", "sampling_weights")
    )
    winding_scope = tuple(
        f"{value.get('geometry_source', 'unavailable')}; independent samples; original current; component={component[index]}"
        for index, value in enumerate(winding)
    )
    for title, key, unit, factor in (
        ("Winding independent RMS error", "rms_error_T", "µT", 1e6),
        ("Winding independent peak error", "peak_error_T", "µT", 1e6),
        ("Winding independent relative error", "relative_field_error", "%", 100),
    ):
        metric(
            title,
            unit,
            *(value.get(key) for value in winding),
            winding_scope,
            same_winding_grid and same_winding_policy and all(value.get("available") is True for value in winding),
            "Independent winding target grid/component or geometry/current policy differs.",
            factor=factor,
        )
    from .gradient_review import gradient_review

    observations = tuple({row.key: row for row in gradient_review(value)} for value in metadata)
    for key in sorted(observations[0].keys() | observations[1].keys()):
        first, second = observations[0].get(key), observations[1].get(key)
        row = first or second
        comparable = first is not None and second is not None and first.scope == second.scope
        if key.startswith("linearity_") or key.endswith(":efficiency"):
            comparable = comparable and same_winding_grid and same_winding_policy
        if "margin" in key or "rms_current" in key or "average_joule" in key:
            comparable = comparable and requests[0].get("engineering", {}).get("drive") == requests[1].get(
                "engineering", {}
            ).get("drive")
        metric(
            row.title + (f" [{row.body_id}]" if row.body_id else ""),
            row.unit,
            first.value if first else None,
            second.value if second else None,
            (first.scope if first else "Unavailable", second.scope if second else "Unavailable"),
            comparable,
            "Engineering representation, sampling or prescribed drive scope differs.",
        )
    if not same_design:
        notes.append(
            "Design target samples, values, weights or selected field components differ; wire error percentages are not a common-grid improvement measure."
        )
    regional = tuple(value.get("requirements_review") or {} for value in metadata)
    regional_rows = tuple(
        {(r["region"], r["scenario"], r["sampling"], r["scope"]): r for r in value.get("rows", ())}
        for value in regional
    )
    same_regional_contract = bool(regional[0].get("evaluation_contract_sha256")) and regional[0].get(
        "evaluation_contract_sha256"
    ) == regional[1].get("evaluation_contract_sha256")
    for key in sorted(regional_rows[0].keys() | regional_rows[1].keys()):
        first, second = (rows.get(key) or {} for rows in regional_rows)
        same = same_regional_contract and all(row.get("available") is True for row in (first, second))
        for suffix in ("_points_m", "_target_T"):
            names = tuple(row.get("array_prefix", "") + suffix for row in (first, second))
            same = (
                same
                and all(name in source for source, name in zip(arrays, names, strict=True))
                and np.array_equal(arrays[0].get(names[0]), arrays[1].get(names[1]))
            )
        scopes = tuple(
            f"{key[3]}; {key[2]}; pose={key[1]}; unchanged recorded design current; feeds included={value.get('external_feeds_included', False)}"
            for value in regional
        )
        same = same and regional[0].get("external_feeds_included") == regional[1].get("external_feeds_included")
        for title, value in (("RMS", "rms_error_T"), ("Peak", "peak_error_T")):
            metric(
                f"Region {key[0]} · {key[1]} · {key[2]} · {key[3]} {title}",
                "µT",
                first.get(value),
                second.get(value),
                scopes,
                same,
                "Named target, pose, field component, bounds, saved sampling or winding current policy differs.",
                factor=1e6,
            )
    if any(regional):
        notes.append(
            "Regional comparisons retain every named region and finite pose. Nominal/robust training membership may differ while the evaluation contract stays identical; each winding uses its own unchanged design current."
        )
    if not same_independent:
        notes.append(
            "Independent validation grids/targets differ or are unavailable; values retain their separate scopes."
        )
    for value in metadata:
        if "metadata_sha256" not in value:
            notes.append("A legacy result has artifact checks but lacks acceptance-metadata checksum protection.")
    return RunComparison(left, right, tuple(deltas), tuple(metrics), tuple(dict.fromkeys(notes)))
