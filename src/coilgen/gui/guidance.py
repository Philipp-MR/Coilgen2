"""Pure next-action guidance over editable inputs and immutable saved evidence.

This module neither runs a solver nor verifies files. Callers supply checked
readiness reports and verified RunRecords; the export catalog remains the final
authority for a CAD operation. Sparse metadata cannot establish a passed check.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import math
from pathlib import PurePosixPath

import numpy as np

from ..application.diagnostics import run_verdicts
from ..application.targets import recognize_solid_harmonic_target
from ..config import CoilGenConfig
from ..field_units import PROTON_GYROMAGNETIC_RATIO_HZ_PER_T
from ..target import _axis_from_expression, _target_gradient_normalization_method


CONTINUATION_STAGES = {
    "prepared": ("sheet", "Optimize sheet"),
    "sheet": ("contours", "Generate contours"),
    "contours": ("routed", "Connect / sweep"),
    "routed": ("evaluated", "Full design evaluation"),
}


@dataclass(frozen=True, slots=True)
class GuidanceStage:
    key: str
    label: str
    status: str
    detail: str
    tone: str = "neutral"


@dataclass(frozen=True, slots=True)
class GuidanceState:
    title: str
    detail: str
    action_label: str
    action_key: str
    tone: str
    stages: tuple[GuidanceStage, ...] = ()


def _config(project):
    # Canonical values retain owned asset IDs and avoid writing temporary files.
    return CoilGenConfig(**project.values)


def _metadata(record):
    return record.metadata if record is not None else {}


def _identity(metadata, key):
    for source in (metadata, metadata.get("job") or {}):
        value = source.get(key)
        if value is not None and not (isinstance(value, str) and not value.strip()):
            return value
    return None


def _solid_artifacts(metadata):
    """Identify saved solid interchange entries, never centerline STEP files."""
    candidates = []
    for artifact in metadata.get("artifacts") or ():
        name = artifact.get("name", "")
        digest = artifact.get("sha256")
        if (
            not isinstance(name, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
        ):
            continue
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
            continue
        if path.suffix.lower() == ".stl" or path.name.lower().endswith("_faceted.step"):
            candidates.append(name)
    return tuple(candidates)


def _blocking(readiness):
    if readiness is None:
        return ()
    if hasattr(readiness, "blocking"):
        return tuple(readiness.blocking)
    return tuple(getattr(readiness, "errors", ()))


def _input_node(issue):
    node = getattr(issue, "node", "")
    field = getattr(issue, "key", getattr(issue, "field", ""))
    if node:
        return {"geometry": "support", "material": "conductor", "optimization": "solve", "validation": "check"}.get(
            node, node
        )
    if "mesh" in field and not field.startswith("target"):
        return "support"
    if field.startswith(("target_error", "sf_opt", "research_physics")):
        return "solve"
    if field.startswith(("target", "field_shape")):
        return "target"
    if field.startswith(("conductor", "specific_conductivity", "cross_section")):
        return "conductor"
    return "solve"


def _explicit_failure(metadata, issues):
    errors = tuple(issue for issue in issues if getattr(issue, "severity", "error") == "error")
    verdicts = run_verdicts(metadata)
    failures = any(value == "Failed" for value in verdicts.values())
    failures |= any(row.get("passed") is False for row in metadata.get("constraints") or ())
    summary = metadata.get("summary") or {}
    failures |= summary.get("accepted") is False and summary.get("scope") in {"continuous_sheet", "fabrication_review"}
    return failures or bool(errors), errors


def _winding_detail(metadata):
    summary = metadata.get("summary") or {}
    parts = []
    count = summary.get("contour_components")
    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
        parts.append(f"Actual contour components: {count}; these are distinct from configured contour levels.")
    else:
        parts.append("Actual contour component count is not recorded; configured levels are not a derived loop count.")
    for key, label in (("contour", "Contour field"), ("routed", "Connected-wire field")):
        evidence = (metadata.get("wire_fields") or {}).get(key) or {}
        if evidence.get("available") is False:
            continue
        current = evidence.get("current_A")
        if not isinstance(current, (int, float)) or isinstance(current, bool) or not math.isfinite(current):
            continue
        policy = (
            "refitted"
            if evidence.get("current_refitted") is True
            else "not refitted"
            if evidence.get("current_refitted") is False
            else "refitting policy unverified"
        )
        count = evidence.get("sample_count")
        samples = (
            f"{count} design-grid samples"
            if isinstance(count, int) and not isinstance(count, bool) and count > 0
            else "sampling count unverified"
        )
        parts.append(
            f"{label}: saved I={current:g} A; {policy}; {samples}; this is not independent winding validation."
        )
    length = summary.get("wire_length_m")
    if isinstance(length, (int, float)) and not isinstance(length, bool) and math.isfinite(length) and length >= 0:
        parts.append(f"Saved electrical-path length: {length:g} m; its circuit/feed scope remains recorded separately.")
    return " ".join(parts)


def _stages(metadata, blockers, issues, *, pending, stale, running):
    stage = metadata.get("pipeline_stage", (metadata.get("summary") or {}).get("pipeline_stage"))
    order = ("prepared", "sheet", "contours", "routed", "evaluated")
    index = order.index(stage) if stage in order else -1
    verdicts = run_verdicts(metadata)
    solids = _solid_artifacts(metadata)
    blocked_nodes = {_input_node(issue) for issue in blockers}
    output = []
    for key, label in (("target", "Target"), ("support", "Support"), ("conductor", "Conductor")):
        status = "needs_input" if key in blocked_nodes else "pending" if pending else "defined"
        output.append(
            GuidanceStage(
                key,
                label,
                status,
                "Correct the highlighted input."
                if key in blocked_nodes
                else "Current model inputs; engineering checks are evaluated after computation.",
                "error" if key in blocked_nodes else "warning" if pending else "neutral",
            )
        )
    for key, label, boundary in (("solve", "Solve", 1), ("winding", "Winding", 2)):
        status = (
            "needs_input"
            if key in blocked_nodes
            else "pending"
            if pending
            else "running"
            if running
            else "stale"
            if stale
            else "computed"
            if index >= boundary
            else "not_computed"
        )
        detail = "Saved current-sheet solution." if key == "solve" else _winding_detail(metadata)
        output.append(
            GuidanceStage(
                key,
                label,
                status,
                detail if index >= boundary else "Computed evidence is not yet available for this step.",
                "error"
                if key in blocked_nodes
                else "warning"
                if status in {"pending", "stale"}
                else "info"
                if status in {"computed", "running"}
                else "neutral",
            )
        )
    checks_failed = _explicit_failure(metadata, issues)[0]
    checked = (
        (metadata.get("summary") or {}).get("accepted") is True
        and verdicts["sheet"] == "Passed"
        and verdicts["fabrication"] == "Passed"
        and verdicts["sweeps"] in {"Passed", "Not run"}
    )
    status = (
        "needs_input"
        if "check" in blocked_nodes
        else "pending"
        if pending
        else "running"
        if running
        else "stale"
        if stale
        else "failed"
        if checks_failed
        else "passed"
        if checked
        else "unverified"
        if metadata
        else "not_computed"
    )
    output.append(
        GuidanceStage(
            "check",
            "Check",
            status,
            f"Sheet: {verdicts['sheet']}; pipeline solids: {verdicts['sweeps']}; fabrication review: {verdicts['fabrication']}. These scopes do not establish construction release.",
            "error"
            if status in {"failed", "needs_input"}
            else "success"
            if status == "passed"
            else "warning"
            if status in {"pending", "stale", "unverified"}
            else "neutral",
        )
    )
    status = "pending" if pending else "stale" if stale else "available" if solids else "not_run"
    output.append(
        GuidanceStage(
            "export",
            "Export",
            status,
            f"{len(solids)} saved solid CAD files. Export uses the selected run's revision and units; availability does not establish field acceptance."
            if solids
            else "No saved solid STL or faceted STEP artifact; centerlines and arrays do not establish export availability.",
            "warning"
            if status in {"pending", "stale"} or (solids and checks_failed)
            else "success"
            if solids
            else "neutral",
        )
    )
    return tuple(output)


def assess_guidance(project, record=None, *, issues=(), active_job=None, readiness=None, pending=False):
    """Choose a next action without changing inputs, study settings or evidence.

    Geometry and electromagnetic failures are computed outcomes, not missing
    inputs. A different channel owner cannot share a current result merely
    because its numerical fingerprint matches.
    """
    issues = tuple(issues)
    config, metadata = _config(project), _metadata(record)
    if readiness is None:
        from ..config_validation import validate_config

        # Owned IDs have no extension; use their recorded filenames for the
        # lightweight format checks. Actual content readiness belongs to the
        # caller's report and the numerical submission service.
        owned = {
            key: project._assets[value[6:]].filename
            for key, value in project.values.items()
            if isinstance(value, str) and value.startswith("asset:")
        }
        readiness = validate_config(replace(config, **owned), project_root=project.project_root)
    blockers = _blocking(readiness)
    missing = []
    if str(config.coil_mesh_file).strip().lower() in {"", "none"}:
        missing.append(("support", "Define a support surface or import an owned mesh."))
    if str(config.target_field_definition_file).lower() == "none" and str(
        config.field_shape_function
    ).strip().lower() in {"", "none"}:
        missing.append(("target", "Define a target expression, spherical harmonic or imported field map."))
    if missing:
        from types import SimpleNamespace

        blocked_nodes = {_input_node(issue) for issue in blockers}
        blockers += tuple(
            SimpleNamespace(node=key, message=message) for key, message in missing if key not in blocked_nodes
        )
    owner = _identity(metadata, "project_id")
    fingerprint = _identity(metadata, "project_fingerprint")
    model_fingerprint = _identity(metadata, "project_model_fingerprint")
    other_owner = owner is not None and owner != project.id
    stale = bool(metadata) and (other_owner or fingerprint != project.fingerprint)
    job = active_job or {}
    running = job.get("status") in {"queued", "running", "cancelling"} and _identity(job, "project_id") in {
        None,
        project.id,
    }
    stages = _stages(metadata, blockers, issues, pending=pending, stale=stale, running=running)

    def state(title, detail, label, action, tone):
        return GuidanceState(title, detail, label, action, tone, stages)

    if pending:
        return state(
            "Pending model edits",
            "Apply or correct the draft before computing the edited model. Saved results and CAD still belong to their recorded revision.",
            "Check and compute edits",
            "compute_full",
            "warning",
        )
    if blockers:
        first = blockers[0]
        return state(
            "Correct model inputs",
            getattr(first, "message", "Correct the highlighted settings before computation."),
            "Open required settings",
            "correct_input",
            "error",
        )
    if running:
        current = job.get("project_fingerprint") == project.fingerprint
        return state(
            "Computing current study" if current else "An earlier revision is computing",
            "The worker owns its submitted inputs. Model edits do not alter that job; inspect its activity before starting another study.",
            "View computation",
            "show_job",
            "info",
        )
    if stale:
        title = (
            "Result belongs to another channel"
            if other_owner
            else "Saved result identity is unverified"
            if fingerprint is None
            else "Study settings changed"
            if model_fingerprint == project.model_fingerprint
            else "Saved result differs from this model"
        )
        detail = (
            "Select this channel's run or compute its owned model; channel fields and circuits remain independent."
            if other_owner
            else "Recompute with the current model and study. Verified compatible stages can be reused; the selected saved run remains inspectable and exportable as its own revision."
        )
        return state(title, detail, "Compute current design", "recompute", "warning")
    if not metadata:
        return state(
            "Model inputs defined",
            "Compute the full design to optimize the sheet, create the requested winding, evaluate fields and check conductor geometry. Numerical feasibility is determined by the study.",
            "Compute full design",
            "compute_full",
            "info",
        )
    failed, errors = _explicit_failure(metadata, issues)
    solids = _solid_artifacts(metadata)
    if failed:
        detail = "Inspect the recorded field and conductor failures before changing model settings. A passed sheet cannot override failed conductor checks."
        if solids:
            detail += f" {len(solids)} solid CAD files remain available for the saved geometry; that availability does not resolve the failed checks."
        return state(
            "Review computed engineering failures",
            detail,
            f"Review {len(errors)} problems" if errors else "Review failed checks",
            "inspect_problems",
            "error",
        )
    stage = metadata.get("pipeline_stage", (metadata.get("summary") or {}).get("pipeline_stage"))
    if stage in {"prepared", "sheet", "contours", "routed"}:
        next_stage, next_label = CONTINUATION_STAGES[stage]
        return state(
            f"Selected {stage} study complete",
            f"The requested {stage} boundary completed. Continue explicitly to {next_stage} for {next_label.lower()}; downstream winding checks remain separate evidence.",
            f"Continue to {next_label.lower()}",
            "continue_full",
            "info",
        )
    verdicts = run_verdicts(metadata)
    if solids:
        return state(
            "Saved conductor geometry is available",
            f"{len(solids)} solid STL/STEP files are recorded for this run. Export their recorded geometry and units. Review field/current scope and feed assumptions; this is not construction release.",
            "Export conductor geometry",
            "export_geometry",
            "success" if verdicts["fabrication"] == "Passed" else "warning",
        )
    if stage == "evaluated" and verdicts["fabrication"] in {"Not run", "Unverified"}:
        return state(
            "Solid geometry needs fabrication review",
            "No reviewed solid CAD artifact is recorded. Enable strict fabrication review and compute full design evaluation; review can report geometry failures instead of producing STL.",
            "Run fabrication review",
            "review_fabrication",
            "warning",
        )
    return state(
        "Saved evidence needs inspection",
        "The saved evidence does not establish both check completion and available solid CAD. Inspect its scopes and artifacts; missing evidence remains unverified.",
        "Inspect saved results",
        "inspect_problems",
        "warning",
    )


def _asset_label(project, value):
    if str(value).startswith("asset:"):
        asset = getattr(project, "_assets", {}).get(str(value)[6:])
        source = getattr(asset, "source", "")
        return source.replace("\\", "/").rsplit("/", 1)[-1] or f"owned {getattr(asset, 'kind', 'input')}"
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


def _model_summary(project):
    config = _config(project)
    units = "Hz" if str(config.target_field_units).lower() == "hz" else "T"
    imported = str(config.target_field_definition_file).lower() != "none"
    method = _target_gradient_normalization_method(config, source="expression")
    recipe = recognize_solid_harmonic_target(config.field_shape_function)
    expression = str(config.field_shape_function).strip()
    if imported:
        target = f"Owned target map: {_asset_label(project, config.target_field_definition_file)}; component {config.target_field_component}. Map values and world coordinates are authoritative."
    elif recipe is not None and method == "none":
        coefficient, label = (recipe.coefficient_T * 1e6, "µT") if units == "T" else (recipe.coefficient_T, "Hz")
        center = ", ".join(f"{value * 1e3:g}" for value in recipe.center_m)
        target = f"Bz: real Y({recipe.degree},{recipe.order}) coefficient {coefficient:g} {label}; R={recipe.reference_radius_m * 1e3:g} mm; harmonic world center ({center}) mm. Coefficient is not the pole amplitude."
    else:
        axis = _axis_from_expression(expression)
        chosen = str(config.target_gradient_axis).lower()
        if method == "linear_fit" and axis is not None and chosen in {axis, "auto"}:
            gradient = config.target_gradient_strength * (1000 if units == "T" else 1)
            target = f"G{axis} = dBz/d{axis} = {gradient:g} {'mT/m' if units == 'T' else 'Hz/m'}; Bz varies with world {axis}."
        elif method == "none":
            try:
                constant = ast.literal_eval(expression)
            except (ValueError, SyntaxError, RecursionError):
                constant = None
            if isinstance(constant, (int, float)) and not isinstance(constant, bool) and math.isfinite(constant):
                field = constant if units == "T" else constant / PROTON_GYROMAGNETIC_RATIO_HZ_PER_T
                target = f"Uniform Bz = {field * 1e6:g} µT (absolute{' Hz input' if units == 'Hz' else ''})."
            else:
                target = f"Absolute Bz = {expression[:160]} [{units}]; world x, y, z in metres."
        else:
            scaling = (
                f"dimensionless multiplier {config.target_gradient_strength:g}"
                if method == "scale"
                else f"{method} normalization; inspect resolved preview"
            )
            target = f"Bz expression {expression[:160]} [{units}]; {scaling}; world coordinates in metres."
    if imported:
        roi = "Sampling coordinates and regions come from the imported map; generated spherical ROI controls do not define those samples."
    elif config.target_region_geometry is not None:
        from ..target_regions import target_region_summary

        roi = target_region_summary(config) + f"; fitting resolution {config.target_region_resolution}."
        if config.target_region_geometry.get("sampling", "volume") == "surface":
            roi += " Surface constraints do not certify the interior."
    elif str(config.target_mesh_file).lower() != "none":
        policy = "vertices only" if config.use_only_target_mesh_verts else "vertices plus bounding-box grid"
        roi = f"Owned region mesh: {_asset_label(project, config.target_mesh_file)}; legacy {policy}, with recorded world coordinates."
    else:
        center = (
            "support vertex centroid, determined from the generated/imported mesh"
            if config.set_roi_into_mesh_center
            else "world origin"
        )
        roi = f"Spherical ROI: {2 * config.target_region_radius * 1e3:g} mm diameter; center at {center}; fitting resolution {config.target_region_resolution}."
    if config.coil_mesh_file == "create cylinder mesh":
        length, radius, _angular, divisions, *_rotation = config.cylinder_mesh_parameter_list
        span = length * max(1, int(divisions)) / (max(1, int(divisions)) + 1)
        support = f"Cylinder: physical span {span * 1e3:g} mm, radius {radius * 1e3:g} mm; local axis z, configured rotation about world origin."
    elif config.coil_mesh_file == "create bi-planary mesh":
        height, width, _nx, _ny, *rest = config.biplanar_mesh_parameter_list
        support = f"Biplanar: {width * 1e3:g} × {height * 1e3:g} mm; full separation {rest[6] * 1e3:g} mm along the configured plane normal."
    elif config.coil_mesh_file == "create conical frustum mesh":
        length, bottom, top, *_rest = config.conical_frustum_mesh_parameter_list
        support = f"Conical frustum: {length * 1e3:g} mm long; local bottom/top radii {bottom * 1e3:g}/{top * 1e3:g} mm; world center and rotation are configured separately."
    else:
        from .editors import SUPPORT_EDITORS

        editor = next((item for item in SUPPORT_EDITORS if item.source == config.coil_mesh_file), None)
        if editor:
            parameters = getattr(config, editor.parameter_field)
            dimensions = "; ".join(
                f"{field.label.lower()} {editor.display_value(field, parameters):g}{field.suffix}"
                for field in editor.fields
                if field.suffix and not field.key.startswith(("center_", "rotation_"))
            )
            support = f"{editor.label}: {dimensions}; inspect world bounds and orientation in the preview."
        else:
            support = f"Support: {_asset_label(project, config.coil_mesh_file)}; inspect its world bounds in the model preview."
    profile = np.asarray(config.cross_sectional_points)
    if profile.shape == (1, 2) and np.all(profile == 0):
        width, height = config.conductor_cross_section_width * 1e3, config.conductor_cross_section_height * 1e3
        conductor = f"Conducting profile: {width:g} × {height:g} mm ({width * height:g} mm²); insulation {config.sweep_insulation_thickness * 1e3:g} mm per side."
    else:
        conductor = "Custom conducting cross-section; nominal dimensions retain their rectangle and legacy routing/report roles."
    conductor += f" Resistivity {config.specific_conductivity_conductor * 1e9:g} nΩ·m; effective optimization sheet thickness {config.conductor_thickness * 1e3:g} mm."
    study = project.study
    ratio = study["design_margin"]
    policy = (
        "independent closed turns"
        if config.contour_only_sweep_mode
        else "connected windings"
        if not config.skip_postprocessing
        else "unconnected contours"
    )
    study_text = f"{config.levels} contour levels; {policy}. Design relative budget {config.target_error_tolerance_percent:g}%; validation limits ÷ {ratio:g}; strict fabrication review {'requested' if study['fabricate'] else 'not requested'}."
    return {"target": target, "roi": roi, "support": support, "conductor": conductor, "study": study_text}


def resolved_model_summary(project):
    """Compact physical definitions; does not infer fields, currents or centers.

    An editable project can contain malformed imported settings. Its overview
    must remain usable while readiness links the user to the offending input.
    """
    try:
        return _model_summary(project)
    except (ValueError, TypeError, IndexError, OverflowError) as exc:
        return {
            "target": "Inspect the target definition and its units.",
            "roi": "Inspect the sampling region and world-coordinate policy.",
            "support": "Inspect the support dimensions and mesh parameters.",
            "conductor": "Inspect conducting dimensions and material properties.",
            "study": f"Physical summary unavailable until invalid settings are corrected: {exc}",
        }
