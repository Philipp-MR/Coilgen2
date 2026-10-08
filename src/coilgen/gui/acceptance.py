"""Decision-facing summaries of immutable evidence; no solve or new acceptance.

Saved constraint limits are authoritative. Descriptive wire-field metrics never
borrow a sheet budget, and unknown evidence never becomes a passed check.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from ..application.diagnostics import run_verdicts
from ..application.gradient_review import gradient_review


@dataclass(frozen=True, slots=True)
class AcceptanceCheck:
    key: str
    stage: str
    requirement: str
    status: str
    observed: float | None = None
    limit: float | None = None
    effective_limit: float | None = None
    margin: float | None = None
    unit: str = ""
    scope: str = ""
    settings_node: str = "results"
    issue_key: str = ""
    body_id: str = ""
    comparison: str = "upper"

    @property
    def tone(self):
        return {
            "Passed": "success",
            "Failed": "error",
            "Evaluated": "info",
            "Unverified": "warning",
            "Not designed": "warning",
        }.get(self.status, "neutral")


@dataclass(frozen=True, slots=True)
class AcceptanceStage:
    key: str
    title: str
    status: str
    detail: str


@dataclass(frozen=True, slots=True)
class AcceptanceReport:
    identity: str
    identity_tone: str
    stages: tuple[AcceptanceStage, ...]
    checks: tuple[AcceptanceCheck, ...]
    focus_key: str
    message: str


_STAGES = (
    ("sheet", "Continuous sheet"),
    ("contours", "Original contours"),
    ("routed", "Connected winding"),
    ("physical", "Physical conductor"),
)
_ORDER = {"prepared": 0, "sheet": 1, "contours": 2, "routed": 3, "evaluated": 4}
_BODY_CHECKS = (
    (
        "bend",
        "Minimum bend radius",
        "sweep_final_min_bend_radius_m",
        "sweep_required_bend_radius_m",
        "sweep_final_min_bend_radius_violation",
        "m",
        "lower",
    ),
    (
        "clearance",
        "Conductor envelope separation",
        "sweep_capsule_min_centerline_distance_m",
        "sweep_required_centerline_clearance_m",
        "sweep_capsule_clearance_violation",
        "m",
        "lower",
    ),
    (
        "penetration",
        "Support penetration",
        "sweep_support_max_penetration_m",
        "sweep_support_conformity_tolerance_m",
        "sweep_support_penetration_violation",
        "m",
        "upper",
    ),
    (
        "frame",
        "Frame rotation between samples",
        "sweep_final_max_frame_rotation_deg",
        "sweep_max_frame_rotation_deg",
        "sweep_final_frame_rotation_violation",
        "deg",
        "upper",
    ),
)


def _number(value):
    if not isinstance(value, Real) or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (OverflowError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _status(value):
    return "Passed" if value is True else "Failed" if value is False else "Unverified"


def _identity_status(metadata, project, pending):
    if project is None:
        return "Saved run evidence · current model identity not checked", "neutral"
    job = metadata.get("job") or {}
    owner = metadata.get("project_id") or job.get("project_id")
    fingerprint = metadata.get("project_fingerprint") or job.get("project_fingerprint")
    if owner and owner != project.id:
        return "Another channel · limits and evidence belong to the saved run", "warning"
    if not owner or not fingerprint:
        return "Run identity unverified · limits and evidence belong to the saved run", "warning"
    if fingerprint != project.fingerprint:
        return "Earlier model/study revision · limits and evidence belong to the saved run", "warning"
    if pending:
        return "Pending edits · evidence belongs to the last applied revision", "warning"
    return "Current applied model/study · immutable saved evidence", "info"


def _identity(metadata, project, pending):
    status, tone = _identity_status(metadata, project, pending)
    name = str(metadata.get("name") or "Saved run")
    revision = metadata.get("project_revision", (metadata.get("job") or {}).get("project_revision"))
    label = (
        f"revision {revision}"
        if isinstance(revision, int) and not isinstance(revision, bool) and revision >= 0
        else "revision unrecorded"
    )
    return f"{name} · {label}\n{status}", tone


def _constraint(index, row, metadata):
    name = str(row.get("name") or "Recorded constraint")
    title = "Design field residual [" + name.partition(":")[2] + "]" if name.startswith("standard:") else name
    scope = str(row.get("scope") or "")
    lower_name = name.lower()
    physical = (
        scope.startswith("fabricat")
        or scope == "independent_realized_winding"
        or "fabricat" in lower_name
        or row in (metadata.get("fabricated_physics") or {}).get("checks", ())
    )
    stage = (
        "physical"
        if physical
        else "sheet"
        if scope == "continuous_sheet"
        else "physical"
        if (metadata.get("summary") or {}).get("scope") == "fabrication_review"
        else "sheet"
    )
    limit = _number(row.get("limit"))
    effective = _number(row.get("effective_limit"))
    if effective is None:
        effective = limit
    value = _number(row.get("value"))
    validation = metadata.get("validation") or {}
    passed = (
        row.get("passed")
        if row.get("verified") is not False
        and not (
            "independent" in lower_name
            and row.get("scope") != "independent_realized_winding"
            and validation.get("verified") is False
        )
        else None
    )
    status = _status(passed)
    # Contract rows retained by both solver paths are upper-bound residuals,
    # including residual norms for equality constraints.
    margin = (
        effective - value if status in {"Passed", "Failed"} and value is not None and effective is not None else None
    )
    if margin is not None and not math.isfinite(margin):
        margin = None
    sampling = "independent grid" if "independent" in lower_name else "design grid"
    if stage == "physical" and "field" in lower_name:
        scope = (
            "Recorded electrical path · independent grid · original signed current; no refit"
            if row.get("scope") == "independent_realized_winding"
            else "Strict reviewed electrical path · design grid · original current; separate independent winding check"
        )
    elif stage == "sheet" and ("field" in lower_name or lower_name.startswith("standard:")):
        count = (metadata.get("validation") or {}).get("point_count") if sampling == "independent grid" else None
        scope = f"Continuous sheet · {sampling}" + (
            f" · {count} samples" if isinstance(count, int) and count > 0 else ""
        )
        if name.startswith("standard:"):
            scope += " · solver residual contract, not an RMS field metric"
    elif stage == "sheet":
        scope = (
            "Continuous sheet · solver contract"
            if row.get("scope") == "continuous_sheet"
            else "Saved constraint scope unverified"
        )
    else:
        scope = scope.replace("_", " ") or "Strict fabrication review"
    return AcceptanceCheck(
        f"constraint:{index}",
        stage,
        title,
        status,
        value,
        limit,
        effective,
        margin,
        str(row.get("unit") or ""),
        scope,
        "winding"
        if stage == "physical"
        else "target"
        if "field" in lower_name or lower_name.startswith("standard:")
        else "optimization",
        f"constraint:{index}",
        body_id=str(row.get("body_id") or ""),
    )


def _wire_scope(evidence):
    count = evidence.get("sample_count")
    current = _number(evidence.get("current_A"))
    refit = (
        "current refitted"
        if evidence.get("current_refitted") is True
        else "current not refitted"
        if evidence.get("current_refitted") is False
        else "refit policy unverified"
    )
    source = evidence.get("geometry_sources", evidence.get("geometry_source", "geometry unverified"))
    if isinstance(source, (tuple, list)):
        source = ", ".join(str(item) for item in source)
    return (
        f"Design grid · {count if isinstance(count, int) and count > 0 else '?'} active samples · {evidence.get('component', 'component unverified')} · I={current:g} A · {refit} · {source}; descriptive thin-wire evidence"
        if current is not None
        else f"Design grid · {count if isinstance(count, int) and count > 0 else '?'} active samples · current unverified · {refit} · {source}; descriptive thin-wire evidence"
    )


def _body_rows(metadata):
    rows = []
    for scope, bodies in (
        ("Pipeline sweep", (metadata.get("pipeline_sweeps") or {}).get("bodies", ())),
        ("Strict review", (metadata.get("fabrication") or {}).get("paths", ())),
    ):
        for index, body in enumerate(bodies):
            body_id = str(body.get("body_id") or f"body {index + 1}")
            diagnostics = body.get("diagnostics") or {}
            for key, title, value_key, limit_key, flag, unit, comparison in _BODY_CHECKS:
                value, limit = (
                    _number(diagnostics.get(value_key)),
                    _number(diagnostics.get(limit_key, (metadata.get("config") or {}).get(limit_key))),
                )
                if value is None and flag not in diagnostics:
                    continue
                violation = diagnostics.get(flag)
                status = (
                    "Failed"
                    if violation is True
                    else "Passed"
                    if violation is False and value is not None and limit is not None
                    else "Unverified"
                )
                skipped = (
                    key == "clearance"
                    and (
                        diagnostics.get("sweep_self_intersection_skipped") is True
                        or diagnostics.get("sweep_self_intersection_skipped_due_route_precheck_failure") is True
                    )
                    or key == "penetration"
                    and diagnostics.get("sweep_support_conformity_skipped_due_route_precheck_failure") is True
                )
                if skipped:
                    status = "Not run"
                margin = (
                    (value - limit if comparison == "lower" else limit - value)
                    if value is not None and limit is not None and status in {"Passed", "Failed"}
                    else None
                )
                margin = _number(margin)
                rows.append(
                    AcceptanceCheck(
                        f"body:{scope}:{body_id}:{key}",
                        "physical",
                        title,
                        status,
                        value,
                        limit,
                        limit,
                        margin,
                        unit,
                        f"{scope} · {body_id} · saved geometry only",
                        "winding",
                        body_id=body_id,
                        comparison=comparison,
                    )
                )
    return rows


def acceptance_report(metadata=None, *, project=None, pending=False) -> AcceptanceReport:
    """Interpret caller-verified metadata, never current editable acceptance limits."""
    if metadata is None:
        stages = tuple(AcceptanceStage(key, title, "Not run", "No saved run selected.") for key, title in _STAGES)
        return AcceptanceReport(
            "No saved run selected",
            "neutral",
            stages,
            (),
            "",
            "Compute or select a saved run to inspect its acceptance evidence.",
        )
    summary, config = metadata.get("summary") or {}, metadata.get("config") or {}
    verdicts = run_verdicts(metadata)
    boundary = _ORDER.get(metadata.get("pipeline_stage", summary.get("pipeline_stage")))
    rows = [_constraint(index, row, metadata) for index, row in enumerate(metadata.get("constraints") or ())]
    sheet_rows = [row for row in rows if row.stage == "sheet"]
    sheet_status = (
        "Failed"
        if any(row.status == "Failed" for row in sheet_rows)
        else "Unverified"
        if any(row.status == "Unverified" for row in sheet_rows)
        else verdicts["sheet"]
    )
    if boundary == 0:
        sheet_status = "Not run"
    stages = [
        AcceptanceStage(
            "sheet",
            "Continuous sheet",
            sheet_status,
            "Saved solver contract and independent-grid checks; current unchanged.",
        )
    ]
    rows.insert(
        0,
        AcceptanceCheck(
            "sheet:acceptance",
            "sheet",
            "Sheet acceptance",
            sheet_status,
            scope="Continuous sheet before contouring/routing",
            settings_node="optimization",
        ),
    )
    wire_fields = metadata.get("wire_fields") or {}
    for key, kind, title, minimum in (
        ("contours", "contour", "Original contours", 2),
        ("routed", "routed", "Connected winding", 3),
    ):
        evidence = wire_fields.get(kind) or {}
        skipped = key == "routed" and (
            config.get("skip_postprocessing") is True or config.get("contour_only_sweep_mode") is True
        )
        status = (
            "Evaluated"
            if evidence.get("available") is True
            else "Not run"
            if skipped or boundary is not None and boundary < minimum
            else "Unverified"
        )
        scope = (
            _wire_scope(evidence)
            if evidence.get("available") is True
            else str(
                evidence.get("reason")
                or "Field evidence unavailable; stage completion does not establish field acceptance."
            )
        )
        stages.append(AcceptanceStage(key, title, status, scope))
        for suffix, label, field, factor, unit in (
            ("relative", "Relative RMS field error", "relative_rms_error", 100, "%"),
            ("rms", "RMS field error", "rms_error_T", 1e6, "µT"),
        ):
            value = _number(evidence.get(field)) if evidence.get("available") is True else None
            value = _number(value * factor) if value is not None else None
            rows.append(
                AcceptanceCheck(
                    f"{key}:{suffix}",
                    key,
                    label,
                    "Evaluated" if value is not None else status if status != "Evaluated" else "Unverified",
                    value,
                    unit=unit,
                    scope=scope,
                    settings_node="winding",
                )
            )
    rows.extend(_body_rows(metadata))
    if metadata.get("engineering_review") or metadata.get("winding_validation"):
        for observation in gradient_review(metadata):
            rows.append(
                AcceptanceCheck(
                    "engineering:" + observation.key,
                    "sheet" if observation.key == "sheet_K" else "physical",
                    observation.title,
                    "Evaluated" if observation.value is not None else "Unverified",
                    observation.value,
                    unit=observation.unit,
                    scope=observation.scope,
                    body_id=observation.body_id,
                )
            )
    physical_status = (
        "Failed"
        if "Failed" in (verdicts["sweeps"], verdicts["fabrication"])
        or any(row.stage == "physical" and row.status == "Failed" for row in rows)
        else "Unverified"
        if any(row.stage == "physical" and row.status == "Unverified" for row in rows)
        else "Passed in scope"
        if verdicts["sweeps"] == "Passed" and verdicts["fabrication"] == "Passed"
        else "Not run"
        if boundary is not None and boundary < 4 and not (metadata.get("pipeline_sweeps") or {}).get("body_count")
        else "Unverified"
    )
    stages.append(
        AcceptanceStage(
            "physical",
            "Physical conductor",
            physical_status,
            f"Pipeline sweep: {verdicts['sweeps']} · strict review: {verdicts['fabrication']}. Geometry checks cover their saved realization only.",
        )
    )
    rows.extend(
        (
            AcceptanceCheck(
                "physical:sweep",
                "physical",
                "Pipeline conductor geometry",
                verdicts["sweeps"],
                scope="Pipeline sweep geometry; separate from strict review",
                settings_node="winding",
            ),
            AcceptanceCheck(
                "physical:review",
                "physical",
                "Strict reviewed geometry",
                verdicts["fabrication"],
                scope="Strict reviewed conductor and assembly",
                settings_node="winding",
            ),
        )
    )
    plans = metadata.get("winding_steps") or ()
    if plans:
        feed_status = (
            "Designed in record"
            if all(plan.get("drive_feed_designed") is True for plan in plans)
            else "Not designed"
            if any(plan.get("drive_feed_designed") is False for plan in plans)
            else "Unverified"
        )
        rows.append(
            AcceptanceCheck(
                "physical:feeds",
                "physical",
                "Drive feeds and terminal breaks",
                feed_status,
                scope="Saved winding-plan declarations; geometric closed circuits do not establish drive connections",
                settings_node="winding",
            )
        )
    rows.sort(key=lambda row: dict((key, index) for index, (key, _) in enumerate(_STAGES))[row.stage])
    focus = next((row for row in rows if row.status == "Failed"), None) or next(
        (row for row in rows if row.status in {"Unverified", "Not designed"}), None
    )
    message = (
        f"First failed requirement: {focus.requirement}. Inspect its saved evidence before changing a requirement."
        if focus is not None and focus.status == "Failed"
        else f"Evidence gap: {focus.requirement}."
        if focus is not None
        else "Available checks retain their recorded scope."
    )
    message += " Construction release is not established by this view."
    identity, tone = _identity(metadata, project, pending)
    return AcceptanceReport(identity, tone, tuple(stages), tuple(rows), focus.key if focus is not None else "", message)
