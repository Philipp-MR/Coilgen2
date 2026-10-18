"""Interpret saved engineering evidence without solving or inventing witnesses."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class ProblemIssue:
    key: str
    title: str
    explanation: str
    severity: str = "error"
    body_id: str = ""
    value: float | None = None
    limit: float | None = None
    unit: str = ""
    settings_node: str = "winding"
    points_m: tuple[tuple[float, float, float], ...] = ()
    segments: tuple[tuple[int, int], ...] = ()
    dataset: str = "review_conductor"
    location_description: str = "No spatial witness was recorded."
    run_directory: str = ""
    scope: str = ""

    @property
    def locatable(self):
        return bool(self.points_m)


def run_verdicts(metadata) -> dict[str, str]:
    """Label distinct saved checks without broadening their engineering scope."""

    def verdict(value):
        return "Passed" if value is True else "Failed" if value is False else "Unverified"

    summary = metadata.get("summary") or {}
    sweeps = metadata.get("pipeline_sweeps") or {}
    fabrication = metadata.get("fabrication") or {}
    sheet = summary.get("sheet_accepted")
    if sheet is None and summary.get("scope") == "continuous_sheet":
        sheet = summary.get("accepted")
    review = verdict(fabrication.get("ready"))
    if not fabrication and (
        summary.get("fabrication_review_performed") is False
        or ("fabrication" in metadata and metadata["fabrication"] is None)
    ):
        review = "Not run"
    sweep_status = verdict(sweeps.get("checks_passed"))
    if sweeps.get("body_count") == 0 and (metadata.get("config") or {}).get("skip_sweep") is True:
        sweep_status = "Not run"
    return {"sheet": verdict(sheet), "sweeps": sweep_status, "fabrication": review}


def preview_diagnostics(config, mesh, target=None, *, message="") -> dict:
    """Summarize the actual prepared target in SI/world coordinates, not an operator."""
    output = {"available": mesh is not None, "message": message, "coordinate_frame": "world"}
    if mesh is None:
        output.update(geometry={}, target={"available": False})
        return output
    vertices = np.asarray(mesh.vertices, dtype=float)
    output["geometry"] = {
        "vertex_count": len(vertices),
        "triangle_count": len(mesh.faces),
        "bounds_min_m": np.min(vertices, axis=0).tolist(),
        "bounds_max_m": np.max(vertices, axis=0).tolist(),
        "type": mesh.metadata.get("geometry_type", config.coil_mesh_file),
    }
    if target is None:
        output["target"] = {"available": False, "reason": message or "Target preview is unavailable."}
        return output
    points, field = np.asarray(target.coords), np.asarray(target.b)
    weights = np.asarray(target.weights, dtype=float)
    active = ~np.asarray(target.suppressed, bool) & np.isfinite(weights) & (weights > 0)
    selected = field[active, 2]
    if len(selected):
        weights = weights[active] / np.max(weights[active])
        weights /= np.sum(weights)
        scale = float(np.max(np.abs(selected)))
        rms = scale * float(np.sqrt(weights @ (selected / scale) ** 2)) if scale else 0.0
    else:
        rms = None
    diagnostics = target.target_diagnostics
    imported = str(config.target_field_definition_file).strip().lower() not in {"", "none"}
    output["target"] = {
        "available": True,
        "sample_count": len(points),
        "active_sample_count": int(np.count_nonzero(active)),
        "suppressed_count": int(np.count_nonzero(target.suppressed)),
        "Bz_min_T": float(np.min(selected)) if len(selected) else None,
        "Bz_max_T": float(np.max(selected)) if len(selected) else None,
        "Bz_rms_T": rms,
        "bounds_min_m": np.min(points, axis=0).tolist() if len(points) else None,
        "bounds_max_m": np.max(points, axis=0).tolist() if len(points) else None,
        "field_units": "T",
        "normalization_requested": config.target_gradient_normalization,
        "normalization_method": diagnostics.get("target_gradient_normalization_method"),
        "normalization_scale": diagnostics.get("target_gradient_normalization_scale"),
        "roi_center_policy": "imported world coordinates"
        if imported
        else "support vertex centroid"
        if config.set_roi_into_mesh_center
        else "world origin / explicit target mesh",
        "expression": None if imported else config.field_shape_function,
        "requested_roi_radius_m": config.target_region_radius,
        "statistics_scope": "positive-weight unsuppressed saved target samples; Bz only",
    }
    return output


def _finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _saved_frames(record, body_id):
    """Read only a hash-verified frame artifact belonging to this run/body."""
    from ..research.storage import file_digest

    root = Path(record.directory).resolve()
    for artifact in record.metadata.get("artifacts", []):
        name = artifact.get("name", "")
        if Path(name).name != f"{body_id}_fabrication_frames.npz":
            continue
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or not artifact.get("sha256"):
            return None
        if file_digest(path) != artifact["sha256"]:
            return None
        try:
            with np.load(path, allow_pickle=False) as loaded:
                required = ("points_m", "normals", "transverse", "cross_section_m", "closed")
                frames = {key: np.array(loaded[key], copy=True) for key in required}
            points = frames["points_m"]
            if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2:
                return None
            if any(frames[key].shape != points.shape for key in ("normals", "transverse")):
                return None
            profile = frames["cross_section_m"]
            if profile.ndim != 2 or profile.shape[1:] != (2,) or len(profile) < 3:
                return None
            if not all(np.all(np.isfinite(frames[key])) for key in required[:-1]):
                return None
            if frames["closed"].size != 1 or frames["closed"].dtype != np.dtype(bool):
                return None
            frames["closed"] = bool(frames["closed"].item())
            return frames
        except (KeyError, OSError, ValueError):
            return None
    return None


def _frame_witness(frames, diagnostics, kind):
    if frames is None or diagnostics.get("sweep_ring_count") != len(frames["points_m"]):
        return (), (), "No matching verified sweep-frame coordinates were saved."
    from ..manufacturing import _polygon_centroid

    points = frames["points_m"]
    count = len(points)
    closed = frames["closed"]
    centroid = _polygon_centroid(frames["cross_section_m"])

    def interval(index):
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= count - (not closed):
            return None
        return index, (index + 1) % count

    if kind == "capsule":
        if diagnostics.get("sweep_capsule_centerline_reference") != "mounted_profile_origin":
            return (), (), "The capsule coordinate reference is not recorded or unsupported."
        first = interval(diagnostics.get("sweep_capsule_min_segment_a"))
        second = interval(diagnostics.get("sweep_capsule_min_segment_b"))
        if first is None or second is None:
            return (), (), "The capsule interval indices do not match the saved path."
        # Stored frame origins are profile centroids. Undo only their exact
        # saved profile offset to recover the capsule's mounted profile origin.
        origin = points - centroid[0] * frames["transverse"] - centroid[1] * frames["normals"]
        selected = origin[[*first, *second]]
        segments = ((0, 1), (2, 3))
        description = (
            "Saved minimum-clearance interval pair; mounted profile origins recovered from saved centroid frames."
        )
    elif kind == "bend":
        index = diagnostics.get("sweep_final_min_bend_point_index")
        if (
            diagnostics.get("sweep_final_bend_point_count") != count
            or isinstance(index, bool)
            or not isinstance(index, int)
            or not (0 <= index < count if closed else 1 <= index < count - 1)
        ):
            return (), (), "The minimum-bend sample does not match the verified final centroid frames."
        indices = ((index - 1) % count, index, (index + 1) % count)
        selected = points[list(indices)]
        segments = ((0, 1), (1, 2))
        description = f"Saved minimum-bend triplet at final centroid sample {index}; no location is inferred."
    elif kind == "support":
        profile = frames["cross_section_m"]
        section_count = len(profile)
        lateral_count = count * section_count
        vertex_count = lateral_count + (0 if closed else 2)
        index = diagnostics.get("sweep_support_penetration_vertex_index")
        if (
            diagnostics.get("sweep_support_vertex_order") != "ring_major_profile_then_open_cap_centers"
            or diagnostics.get("sweep_support_penetration_vertex_count") != vertex_count
            or isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < vertex_count
        ):
            return (), (), "No verified sweep-vertex ordering/count matches the saved support-penetration index."
        if index < lateral_count:
            ring, section = divmod(index, section_count)
            offset = profile[section] - centroid
        else:
            ring = 0 if index == lateral_count else count - 1
            # Sweep end-cap centers use the mean of the profile vertices, not
            # the area centroid; retain asymmetric profile geometry exactly.
            offset = np.mean(profile, axis=0) - centroid
        selected = np.array(
            [points[ring] + offset[0] * frames["transverse"][ring] + offset[1] * frames["normals"][ring]]
        )
        segments = ()
        description = f"Recorded most penetrating sweep vertex {index}, reconstructed from verified final frames and declared ordering."
    elif kind == "fold":
        rings = interval(diagnostics.get("sweep_worst_quad_ring"))
        section = diagnostics.get("sweep_worst_quad_section")
        profile = frames["cross_section_m"]
        if rings is None or not isinstance(section, int) or not 0 <= section < len(profile):
            return (), (), "The folded-patch indices do not match the saved sweep frames."
        selected = []
        for ring, cross in (
            (rings[0], section),
            (rings[1], section),
            (rings[1], (section + 1) % len(profile)),
            (rings[0], (section + 1) % len(profile)),
        ):
            offset = profile[cross] - centroid
            selected.append(points[ring] + offset[0] * frames["transverse"][ring] + offset[1] * frames["normals"][ring])
        selected = np.asarray(selected)
        segments = ((0, 1), (1, 2), (2, 3), (3, 0))
        description = "Recorded worst folded sweep patch, reconstructed from its saved ring/section frames."
    else:
        rings = interval(diagnostics.get("sweep_final_frame_rotation_worst_interval"))
        if rings is None:
            return (), (), "The worst frame interval does not match the saved path."
        selected = points[list(rings)]
        segments = ((0, 1),)
        description = "Saved worst frame-rotation interval on the mounted centroid path."
    if not np.all(np.isfinite(selected)):
        return (), (), "The saved frames cannot provide a finite spatial witness."
    return tuple(tuple(float(v) for v in point) for point in selected), segments, description


_BODY_REASONS = {
    "minimum_bend_radius_violation": (
        "Conductor bend radius is too small",
        "The saved final centroid path violates its configured minimum bend radius.",
        "sweep_final_min_bend_radius_m",
        "sweep_required_bend_radius_m",
        "m",
        "bend",
    ),
    "route_topology_failure": (
        "Route continuity or topology failed",
        "The recorded route topology precheck failed. Continuous-sheet acceptance does not certify a connected winding.",
        None,
        None,
        "",
        None,
    ),
    "route_clearance_violation": (
        "Route clearance is insufficient",
        "The routed path fails the stored conductor-clearance margin. Its precise route witness was not saved.",
        "sweep_route_clearance_margin_m",
        0.0,
        "m",
        None,
    ),
    "sweep_clearance_violation": (
        "Conductor envelopes are too close",
        "The conservative capsule check found insufficient separation between non-neighbouring winding intervals.",
        "sweep_capsule_min_centerline_distance_m",
        "sweep_required_centerline_clearance_m",
        "m",
        "capsule",
    ),
    "sweep_self_intersection": (
        "Conservative conductor envelopes overlap",
        "The saved capsule envelope screen fails. This is not automatically an exact swept-surface intersection certificate.",
        "sweep_capsule_min_centerline_distance_m",
        "sweep_conductor_envelope_diameter_m",
        "m",
        "capsule",
    ),
    "folded_sweep_quad": (
        "A conductor surface patch folds",
        "A recorded sweep quadrilateral has inconsistent surface orientation; inspect the saved worst patch before manufacturing.",
        "sweep_worst_quad_normal_dot",
        0.0,
        "",
        "fold",
    ),
    "unresolved_frame_sampling": (
        "Conductor frame rotation is unresolved",
        "The stored final sweep frame rotates beyond the configured bound between samples.",
        "sweep_final_max_frame_rotation_deg",
        "sweep_max_frame_rotation_deg",
        "deg",
        "frame",
    ),
    "support_surface_penetration": (
        "Conductor penetrates its support",
        "The stored conductor envelope penetrates the intended mounting side; inspect its recorded sweep vertex when available.",
        "sweep_support_max_penetration_m",
        "sweep_support_conformity_tolerance_m",
        "m",
        "support",
    ),
    "poor_sweep_triangle_quality": (
        "Conductor surface triangles are poorly shaped",
        "The saved swept-surface triangle aspect-ratio check fails. No worst triangle coordinates were retained.",
        "sweep_triangle_altitude_aspect_ratio_max",
        "sweep_triangle_altitude_aspect_ratio_limit",
        "",
        None,
    ),
}


def _field_witness(record, name, scope=""):
    from ..field_components import field_component_values

    # A sample residual locates field-error evidence only. A fitted gradient,
    # circuit load or integration-consistency bound has no such witness.
    if "field error" not in name.lower():
        return (), "target", "No spatial witness was recorded for this scalar or unavailable constraint."
    arrays = record.arrays
    config = record.metadata.get("config", {})
    if scope == "independent_realized_winding" or "independent winding" in name.lower():
        field_key, point_key, target_key, dataset = (
            "winding_validation_field_T",
            "winding_validation_points_m",
            "winding_validation_target_T",
            "winding_validation_residual",
        )
    elif "fabricated" in name.lower():
        field_key, point_key, target_key, dataset = (
            "fabricated_field_T",
            "fabricated_points_m",
            "target_field_T",
            "fabricated",
        )
    elif "independent" in name.lower():
        field_key, point_key, target_key, dataset = (
            "validation_field_T",
            "validation_points_m",
            "validation_target_T",
            "validation",
        )
    else:
        return (), "target", "No spatial witness was recorded for this constraint."
    if not all(key in arrays for key in (field_key, point_key, target_key)):
        return (), dataset, "The saved field samples needed for a spatial witness are unavailable."
    points, field, target = (np.asarray(arrays[key]) for key in (point_key, field_key, target_key))
    if field.shape != target.shape or field.shape != points.shape or points.ndim != 2 or points.shape[1] != 3:
        return (), dataset, "The saved field and coordinate grids do not match."
    active = np.ones(len(points), bool)
    if dataset == "fabricated":
        active &= ~np.asarray(arrays.get("target_suppressed", np.zeros(len(points), bool)), bool)
        active &= np.asarray(arrays.get("target_weights", np.ones(len(points)))) > 0
    active &= (
        np.all(np.isfinite(points), axis=1) & np.all(np.isfinite(field), axis=1) & np.all(np.isfinite(target), axis=1)
    )
    if not np.any(active):
        return (), dataset, "No finite active sample can provide a spatial witness."
    component = config.get("target_field_component", "Bz")
    direction = config.get("b_0_direction", (0, 0, 1))
    values = field_component_values(field[active] - target[active], component, b0_direction=direction)
    errors = np.hypot.reduce(np.asarray(values).reshape(np.count_nonzero(active), -1), axis=1)
    point = points[active][int(np.argmax(errors))]
    return (
        (tuple(float(v) for v in point),),
        dataset,
        "Largest recorded active sample residual; the field acceptance limit uses the complete sample set.",
    )


def run_problems(record) -> tuple[ProblemIssue, ...]:
    """Only recorded failures and verified spatial witnesses become problems."""
    metadata = record.metadata
    fabrication = metadata.get("fabrication") or {}
    issues = []
    reasons_seen = set()
    bodies = [("fabrication_review", body) for body in fabrication.get("paths", [])]
    bodies.extend(
        ("pipeline_sweep_geometry", body) for body in (metadata.get("pipeline_sweeps") or {}).get("bodies", [])
    )
    for scope, body in bodies:
        body_id = body.get("body_id", "")
        diagnostics = body.get("diagnostics") or {}
        frames = _saved_frames(record, body_id) if scope == "fabrication_review" else None
        reasons = [reason for reason in str(diagnostics.get("manufacturing_failure_reasons", "")).split(";") if reason]
        if (
            not reasons
            and scope == "pipeline_sweep_geometry"
            and (body.get("manufacturing_ok") is False or body.get("mesh_solid_ok") is False)
        ):
            reasons = ["pipeline_sweep_checks_failed"]
        for reason in reasons:
            title, explanation, value_key, limit_key, unit, witness = _BODY_REASONS.get(
                reason,
                (
                    reason.replace("_", " ").capitalize(),
                    f"The backend recorded {reason!r}; no specific spatial witness is retained for this diagnostic.",
                    None,
                    None,
                    "",
                    None,
                ),
            )
            value = _finite_number(diagnostics.get(value_key)) if value_key else None
            limit = (
                _finite_number(diagnostics.get(limit_key, metadata.get("config", {}).get(limit_key)))
                if isinstance(limit_key, str)
                else limit_key
            )
            points, segments, description = (
                _frame_witness(frames, diagnostics, witness)
                if witness
                else ((), (), "No spatial witness was recorded.")
            )
            if reason == "route_topology_failure":
                details = diagnostics.get("manufacturing_route_topology_failure_reasons")
                if isinstance(details, str) and details.strip():
                    explanation += " Recorded topology reasons: " + details.replace(";", ", ") + "."
                for key in ("group_id", "route_group_id", "connector_id", "route_connector_id"):
                    identity = body.get(key, diagnostics.get(key))
                    if isinstance(identity, (str, int)) and not isinstance(identity, bool):
                        explanation += f" Recorded {key}: {identity}."
            if scope == "pipeline_sweep_geometry":
                explanation += " Scope: saved pipeline sweep geometry; strict fabrication review is a separate check."
                description = "No spatial failure witness was saved for this pipeline body. Inspect its recorded geometry and settings."
            issues.append(
                ProblemIssue(
                    f"{body_id}:{reason}",
                    title,
                    explanation,
                    body_id=body_id,
                    value=value,
                    limit=limit,
                    unit=unit,
                    points_m=points,
                    segments=segments,
                    location_description=description,
                    dataset="swept" if scope == "pipeline_sweep_geometry" else "review_conductor",
                    scope=scope,
                )
            )
            reasons_seen.add(f"{body_id}:{reason}")
    for index, row in enumerate(fabrication.get("assembly_clearance", {}).get("pairs", [])):
        if not row.get("confirmed_clearance_violation") and not row.get("confirmed_conductor_intersection"):
            continue
        issues.append(
            ProblemIssue(
                f"assembly:{index}",
                "Assembly conductor clearance fails",
                "The saved faceted assembly clearance check fails between these bodies; no contact-point coordinate was recorded.",
                body_id=f"{row.get('first_body', '?')} / {row.get('second_body', '?')}",
                value=_finite_number(row.get("minimum_surface_distance_m")),
                limit=_finite_number(row.get("required_surface_gap_m")),
                unit="m",
                scope="fabrication_review",
            )
        )
    body_ids = {body.get("body_id") for body in fabrication.get("paths", [])}
    for text in fabrication.get("failure_reasons", []):
        body_id, separator, reasons = text.partition(":")
        if not separator or body_id not in body_ids:
            if text == "assembly_clearance_violation" and any(issue.key.startswith("assembly:") for issue in issues):
                continue
            body_id, reasons = "", text
        for reason in reasons.split(";"):
            key = f"{body_id}:{reason}"
            if key in reasons_seen:
                continue
            issues.append(
                ProblemIssue(
                    key,
                    reason.replace("_", " ").capitalize(),
                    "The fabrication review recorded this failure; inspect the referenced body and its saved evidence.",
                    body_id=body_id,
                    scope="fabrication_review",
                )
            )
            reasons_seen.add(key)
    for index, row in enumerate(metadata.get("constraints", [])):
        if row.get("passed") is True or (
            row.get("name") == "Fabrication geometry and complete assembly"
            and any(issue.scope == "fabrication_review" for issue in issues)
        ):
            continue
        name = row.get("name", "Constraint")
        points, dataset, description = _field_witness(record, name, row.get("scope", ""))
        failed = row.get("passed") is False
        issues.append(
            ProblemIssue(
                f"constraint:{index}",
                name,
                "The recorded constraint exceeds its acceptance budget."
                if failed
                else "The recorded constraint remains unverified; its acceptance cannot be inferred.",
                severity="error" if failed else "warning",
                value=_finite_number(row.get("value")),
                limit=_finite_number(row.get("effective_limit", row.get("limit"))),
                unit=row.get("unit", ""),
                settings_node="winding" if "fabricat" in name.lower() else "optimization",
                points_m=points,
                dataset=dataset,
                location_description=description,
                scope=row.get("scope", (metadata.get("summary") or {}).get("scope", "")),
            )
        )
    for index, text in enumerate(metadata.get("warnings", [])):
        if any(key.split(":")[0] and key.split(":")[0] + ":" in text for key in reasons_seen):
            continue
        issues.append(
            ProblemIssue(
                f"warning:{index}",
                text,
                "Saved run warning; inspect the stated modeling or workflow limitation.",
                severity="warning",
                settings_node="winding"
                if any(term in text.lower() for term in ("feed", "return", "winding", "fabricat"))
                else "optimization",
            )
        )
    return tuple(replace(issue, run_directory=str(Path(record.directory).resolve())) for issue in issues)
