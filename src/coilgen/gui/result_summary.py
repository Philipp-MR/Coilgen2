"""Run-bound field and coil readouts shared by both workbenches, independent of Qt."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real

from ..application.diagnostics import run_verdicts
from ..field_components import field_component_label, normalize_field_component
from .run_metadata import read_run_metadata


@dataclass(frozen=True, slots=True)
class FieldResultSummary:
    visible: bool = False
    title: str = ""
    badge: str = ""
    tone: str = "neutral"
    source_key: str = ""
    source_label: str = ""
    component: str = ""
    rms_uT: float | None = None
    peak_uT: float | None = None
    relative_percent: float | None = None
    current_A: float | None = None
    current_label: str = "Original current"
    current_display: str = ""
    context: str = ""
    tooltip: str = ""
    analysis_available: bool = False
    # Fit of the shown field to the target with one free current scale s:
    # drive current = s × design current; shape error after that fit; field
    # amplitude at the design current relative to the target.
    drive_current_A: float | None = None
    shape_percent: float | None = None
    gain_percent: float | None = None


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _text(value):
    return " ".join(value.split()) if isinstance(value, str) else ""


def _number(value, scale=1.0, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        result = float(value) * scale
    except (ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and (not nonnegative or result >= 0) else None


def _count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _component(evidence, parent, metadata):
    value = (
        evidence.get("component")
        or parent.get("component")
        or _mapping(metadata.get("config")).get("target_field_component")
        or metadata.get("target_field_component")
    )
    if not value:
        return ""
    try:
        return normalize_field_component(value, allow_bmag=True)
    except ValueError:
        return ""


def _candidate(record, metadata):
    names = set(getattr(record, "array_names", ()))
    wires = _mapping(metadata.get("wire_fields"))

    def wire(kind):
        parent = _mapping(wires.get(kind))
        original = _mapping(parent.get("at_stream_current"))
        label = "Routed winding" if kind == "routed" else "Unconnected contours"
        if original.get("available") is True and {"target_points_m", kind + "_field_at_stream_current_T"} <= names:
            return kind + "_at_stream_current", label, original, parent, "original contour current", "design"
        if (
            parent.get("available") is True
            and parent.get("current_refitted") is False
            and {"target_points_m", kind + "_field_T"} <= names
        ):
            return kind, label, parent, parent, "original recorded current", "design"
        return None

    physics = _mapping(metadata.get("fabricated_physics"))
    winding = _mapping(metadata.get("winding_validation"))
    if (
        winding.get("available") is True
        and winding.get("current_refitted") is False
        and winding.get("current_source") == "original_contour_current_no_refit"
        and {"winding_validation_points_m", "winding_validation_field_T"} <= names
    ):
        return "winding_validation", "Reviewed winding", winding, winding, "original contour current", "independent"
    if (
        "fabricated_field_T" in names
        and {"fabricated_points_m", "target_points_m"} & names
        and physics.get("current_source") == "original_contour_current_no_refit"
    ):
        rows = physics.get("checks", ())
        for row in rows if isinstance(rows, (list, tuple)) else ():
            if (
                isinstance(row, Mapping)
                and row.get("name") == "Fabricated path field error (original current)"
                and row.get("verified") is True
                and _number(row.get("rms_error_T"), nonnegative=True) is not None
            ):
                return "fabricated", "Reviewed conductor", row, physics, "original contour current", "design"
    if candidate := wire("routed"):
        return candidate
    if candidate := wire("contour"):
        return candidate
    validation = _mapping(metadata.get("validation"))
    if validation.get("available") is True and {"validation_points_m", "validation_field_T"} <= names:
        return "validation", "Stream function model", validation, validation, "stream function solution", "independent"
    if {"target_points_m", "sheet_field_T"} <= names:
        return "sheet", "Stream function model", {}, {}, "stream function solution", "design"
    # Legacy results may retain only a refitted wire field. Identify that fact
    # rather than presenting it as the physical original-current result.
    for kind in ("routed", "contour"):
        parent = _mapping(wires.get(kind))
        if parent.get("available") is True and {"target_points_m", kind + "_field_T"} <= names:
            return (
                kind,
                "Routed winding" if kind == "routed" else "Unconnected contours",
                parent,
                parent,
                (
                    "refitted recorded current"
                    if parent.get("current_refitted") is True
                    else "current policy unrecorded"
                ),
                "design",
            )
    return None


def _identity(metadata):
    name = _text(metadata.get("name")) or "Saved run"
    revision = metadata.get("project_revision", _mapping(metadata.get("job")).get("project_revision"))
    return f"Results · {name}" + (
        f" · revision {revision}" if isinstance(revision, int) and not isinstance(revision, bool) else ""
    )


def _badge(metadata, source_error, stale):
    verdicts = run_verdicts(metadata)
    summary = _mapping(metadata.get("summary"))
    independent_failed = any(
        _mapping(metadata.get(key)).get("available") is True and _mapping(metadata.get(key)).get("passed") is False
        for key in ("validation", "winding_validation")
    )
    checks = (
        "Checks failed"
        if "Failed" in verdicts.values() or summary.get("accepted") is False or independent_failed
        else (
            "Saved checks passed"
            if all(value == "Passed" for value in verdicts.values())
            else "Stream function checks passed"
            if verdicts["sheet"] == "Passed"
            else "Checks unverified"
        )
    )
    badge, tone = (
        ("Verification failed", "error")
        if source_error
        else ("Earlier model", "warning")
        if stale
        else (
            checks,
            "error"
            if checks == "Checks failed"
            else "success"
            if checks == "Saved checks passed"
            else "info"
            if checks == "Stream function checks passed"
            else "warning",
        )
    )
    return checks, badge, tone


def summarize_field_result(record, *, source_error="", stale=False):
    """Read saved error metrics without mistaking generic field RMS for error."""
    if record is None:
        return FieldResultSummary()
    metadata = read_run_metadata(record)
    title = _identity(metadata)
    checks, badge, tone = _badge(metadata, source_error, stale)
    candidate = _candidate(record, metadata)
    if candidate is None:
        context = "Field not calculated · saved coil properties" + (" · " + checks if stale else "")
        if source_error:
            context = "Retained snapshot · source verification failed · " + context
        return FieldResultSummary(
            title=title, badge=badge, tone=tone, context=context, tooltip=str(source_error) if source_error else context
        )
    key, label, evidence, parent, policy, sampling = candidate
    component = _component(evidence, parent, metadata)
    component_label = field_component_label(component) if component else "component unrecorded"
    count = _count(evidence.get("sample_count")) or _count(evidence.get("point_count"))
    rms = _number(evidence.get("rms_error_T"), 1e6, nonnegative=True)
    peak = _number(evidence.get("peak_error_T"), 1e6, nonnegative=True)
    target_rms = _number(evidence.get("target_rms_T"), nonnegative=True)
    relative = (
        _number(evidence.get("relative_rms_error", evidence.get("relative_field_error")), 100, nonnegative=True)
        if target_rms is not None and target_rms > 0
        else None
    )
    current = _number(evidence.get("current_A"))
    current_label = (
        "Refitted current"
        if policy == "refitted recorded current"
        else ("Recorded current" if policy == "current policy unrecorded" else "Original current")
    )
    current_display = ""
    if key == "winding_validation" and _mapping(evidence.get("currents_A")):
        current, current_label, current_display = None, "Original body currents", "Per body"
    summary = _mapping(metadata.get("summary"))
    if key == "fabricated" and summary.get("scope") == "fabrication_review":
        current = _number(summary.get("current_a"))
        if current is None:
            currents = _mapping(parent.get("currents_A"))
            if any(_number(value) is not None for value in currents.values()):
                current_label, current_display = "Original body currents", "Per body"
            else:
                electrical = _mapping(_mapping(metadata.get("fabrication")).get("electrical_metrics"))
                bodies = electrical.get("bodies", ())
                if isinstance(bodies, (list, tuple)) and any(
                    _number(_mapping(body).get("current_A")) is not None for body in bodies
                ):
                    current_label, current_display = "Original body currents", "Per body"
    if key in {"sheet", "validation"}:
        current, current_label, current_display = None, "Wire current", "Not applicable"
    sampled = "Independent samples" if sampling == "independent" else "Design samples"
    sample_text = sampled + (f" · n={count:,}" if count is not None else " · sample count unrecorded")
    detail = (
        "Independent quadrature unverified"
        if key == "validation" and evidence.get("independent_quadrature_check") is not True
        else "Independent thin-wire winding check"
        if key == "winding_validation"
        else (
            "Field diagnostic; wire acceptance is separate"
            if key not in {"sheet", "validation"}
            else "Stream function field evidence; wire acceptance is separate"
        )
    )
    context = f"{label} · {policy} · {component_label} error\n{sample_text} · {detail}"
    if source_error:
        context = "Retained snapshot · source verification failed · " + context
    elif stale:
        context = "Earlier applied model · " + context + f" · {checks}"
    convention = _text(evidence.get("rms_convention")) or "unrecorded"
    tooltip = (
        f"{title}\n{context}\nResidual: calculated minus target. Recorded RMS convention: {convention}. "
        "Relative RMS uses the recorded global target RMS, never pointwise division at target zeros. "
        "These scalar summaries are retained from the saved run; no field is recalculated for this panel."
    )
    if key.startswith(("routed", "contour")) and parent.get("scope") == "design_grid_wire_field":
        tooltip += " Wire RMS uses active unsuppressed positive-weight design samples; thin-wire Biot–Savart model."
    if current_display == "Per body":
        tooltip += " Signed original body currents are recorded separately; no aggregate circuit current is inferred."
    if key in {"sheet", "validation"}:
        tooltip += " A stream function solution does not define a single discrete wire current."
    if source_error:
        tooltip += "\n" + str(source_error)
    # Saved by the evaluation (metrics.current_scale_fit); never recomputed here.
    scale = _number(evidence.get("best_fit_current_scale"), nonnegative=True)
    shape = _number(evidence.get("best_fit_relative_rms_error"), 100, nonnegative=True)
    # For the weighted least-squares fit, target-mode amplitude is (1-r²)/s.
    # The shape residual r removes field components orthogonal to the target.
    gain = 100.0 * ((1.0 - (shape / 100.0) ** 2) / scale - 1.0) if scale and shape is not None else None
    drive_current = scale * current if scale and current is not None else None
    return FieldResultSummary(
        visible=True,
        title=title,
        badge=badge,
        tone=tone,
        source_key=key,
        source_label=label,
        component=component,
        rms_uT=rms,
        peak_uT=peak,
        relative_percent=relative,
        current_A=current,
        current_label=current_label,
        current_display=current_display,
        context=context,
        tooltip=tooltip,
        analysis_available=not bool(source_error),
        drive_current_A=drive_current,
        shape_percent=shape,
        gain_percent=gain,
    )


DRIVE_TOOLTIP = (
    "One free current scale is fitted so the shown field best matches the target. "
    "Drive current: the current that produces the requested field. "
    "Shape error: RMS residual relative to the target RMS after that fit — what a current change cannot fix. "
    "Field at the design current: the weighted projection onto the target, shown as a percentage deviation "
    "from the requested amplitude. Orthogonal field components contribute to shape error."
)


def drive_metric_cards(summary):
    """Headline readouts at the current that produces the requested field."""
    current = summary.current_A
    cards = []
    for key, name, number, text in (
        (
            "drive",
            "Drive current",
            summary.drive_current_A,
            None if summary.drive_current_A is None else f"{summary.drive_current_A:.4g} A",
        ),
        (
            "shape",
            "Shape error",
            summary.shape_percent,
            None if summary.shape_percent is None else f"{summary.shape_percent:.3g} %",
        ),
        (
            "gain",
            f"Field at {current:.4g} A" if current is not None else "Field at design current",
            summary.gain_percent,
            None if summary.gain_percent is None else f"{summary.gain_percent:+.1f} %",
        ),
    ):
        display = text or (
            "Not applicable" if key == "drive" and summary.source_key in {"sheet", "validation"} else "Not recorded"
        )
        cards.append(
            {"key": key, "name": name, "value": display, "unavailable": number is None, "tooltip": DRIVE_TOOLTIP}
        )
    return cards


def field_metric_cards(summary):
    cards = []
    for key, name, number, unit in (
        ("rms", "RMS error", summary.rms_uT, "µT"),
        ("peak", "Peak error", summary.peak_uT, "µT"),
        ("relative", "Relative RMS error", summary.relative_percent, "%"),
        ("current", summary.current_label, summary.current_A, "A"),
    ):
        display = (summary.current_display if key == "current" else "") or "Not recorded"
        if number is not None:
            display = (f"{number:+.4g}" if key == "current" else f"{number:.4g}") + " " + unit
        cards.append({"key": key, "name": name, "value": display, "unavailable": number is None})
    return cards


def coil_metric_cards(report):
    cards, scopes = [], []
    volume_prefix = "reviewed:" if any(row.key.startswith("reviewed:") for row in report.rows) else "pipeline:"
    for key, name, scale, unit, prefix, suffix in (
        ("resistance", "Resistance", 1.0, "Ω", "body:", ":resistance_ohm"),
        ("inductance", "Self inductance", 1e6, "µH", "body:", ":inductance_H"),
        ("mass", "Conductor mass", 1e3, "g", "", ""),
        ("volume", "Conductor volume", 1e6, "cm³", volume_prefix, ":mesh_volume"),
    ):
        row = report.row("coil:" + key)
        number = _number(row.value, nonnegative=True) if row is not None else None
        body_count = sum(
            1
            for item in report.rows
            if suffix
            and item.key.startswith(prefix)
            and item.key.endswith(suffix)
            and _number(item.value, nonnegative=True) is not None
        )
        display = "Needs density" if key == "mass" else "Not recorded"
        if number is None and body_count:
            display = "Per body"
        if number is not None:
            converted = _number(number, scale, nonnegative=True)
            display = f"{converted:.4g} {unit}" if converted is not None else "Not recorded"
        scope = row.scope if row is not None else "No saved quantity or scope."
        tooltip = (
            f"{row.quantity}: {row.status}.\n{scope}\nSaved source: {row.source_key}" if row is not None else scope
        )
        if number is None and body_count:
            body_word = "body" if body_count == 1 else "bodies"
            tooltip += f"\nSaved values for {body_count} {body_word}; open Coil stats for values and individual scopes."
        if key == "mass" and number is None:
            tooltip += "\n" + report.mass_unavailable_reason
        cards.append({"key": key, "name": name, "value": display, "unavailable": number is None, "tooltip": tooltip})
        if key in {"resistance", "volume"}:
            scopes.append(("R/L" if key == "resistance" else "Volume") + ": " + scope)
    has_mass = _number(report.row("coil:mass").value, nonnegative=True) is not None
    scopes.append(
        "Mass: recorded conductor only" if has_mass else "Mass: density required; insulation, feed and former excluded"
    )
    mass_scope = "recorded conductor mass" if has_mass else "mass needs density"
    return cards, report.overview_scope + " · " + mass_scope, "\n".join(scopes) + "\n" + report.message
