"""Display-only winding review. Advisory geometry never grants acceptance."""

import numpy as np


def sampled_bend(points):
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return None
    a, b = np.diff(points, axis=0)[:-1], np.diff(points, axis=0)[1:]
    area2 = np.linalg.norm(np.cross(a, b), axis=1)
    numerator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(a + b, axis=1)
    radii = np.divide(numerator, 2 * area2, out=np.full(len(a), np.inf), where=area2 > 1e-18)
    i = int(np.argmin(radii))
    return None if not np.isfinite(radii[i]) else (float(radii[i]), points[i + 1].tolist())


def quick_review(editor, minimum_radius):
    """Local centerline estimates; conductor realization may differ substantially."""
    warnings = []
    for section in editor.sections:
        if section.disconnected:
            warnings.append(
                dict(
                    key=section.key,
                    point=section.points[-1].tolist(),
                    category="continuity",
                    label="Open connection",
                    detail="Reconnect this endpoint.",
                )
            )
        bend = sampled_bend(section.points)
        if section.editable and not section.disconnected and bend and bend[0] < minimum_radius:
            warnings.append(
                dict(
                    key=section.key,
                    point=bend[1],
                    category="advisory",
                    label=f"Sampled bend {bend[0] * 1000:.2f} / {minimum_radius * 1000:g} mm",
                    detail="Centerline estimate; run Check winding for the realized conductor.",
                )
            )
    return warnings


def winding_sequence(editor):
    """Follow directed adjacency, never the incidental section array order."""
    rows = {s.key: s for s in editor.sections}
    if not editor.connectivity["complete"]:
        return []
    targeted = {s.target for s in rows.values()}
    key = next((s.key for s in rows.values() if s.key not in targeted), next(iter(rows)))
    result, seen = [], set()
    while key in rows and key not in seen:
        seen.add(key)
        s = rows[key]
        result.append(
            dict(
                key=key,
                label=s.label,
                bridge=None
                if not s.bridge
                else dict(other=s.bridge["other"], relation=s.bridge.get("relation", "over")),
                length_m=float(np.linalg.norm(np.diff(s.points, axis=0), axis=1).sum()),
            )
        )
        key = s.target
    order = {step["key"]: i for i, step in enumerate(result)}
    for step in result:
        if step["bridge"]:
            other = step["bridge"]["other"]
            step["bridge"]["other_label"] = rows[other].label
            step["bridge"]["order_note"] = (
                "Lower crossing already placed"
                if order.get(other, len(result)) < order[step["key"]]
                else "Lower crossing comes later; keep the bridge accessible"
            )
    return result


def review_issues(editor, report, circuit, minimum_radius):
    issues = quick_review(editor, minimum_radius)
    if report is None:
        return issues
    # A missing certificate is a capability limitation, not a fabricated collision.
    for reason in report.reasons:
        if reason.startswith("Circuit ") and not reason.startswith(f"Circuit {circuit + 1}:"):
            continue
        unavailable = any(word in reason for word in ("unsupported", "incomplete", "unresolved", "budget_exhausted"))
        issue = dict(
            key=None,
            point=None,
            points=[],
            category="unverified" if unavailable else "failed",
            label=reason.replace("_", " "),
            detail="Not certified by the current validator."
            if unavailable
            else "Full engineering check failed. No spatial witness was recorded for this check.",
        )
        kind = (
            "bend"
            if "minimum_bend_radius" in reason
            else "support"
            if "support_surface_penetration" in reason
            else "copper"
            if "copper_gap" in reason or "self_intersection" in reason
            else None
        )
        if kind and circuit < len(report.parts) and report.parts[circuit].fabrication_path is not None:
            from .manufacturing_witness import _frame_witness

            path = report.parts[circuit].fabrication_path
            frames = {
                name: getattr(path, name) for name in ("points_m", "normals", "transverse", "cross_section_m", "closed")
            }
            metrics = report.diagnostics["circuits"][circuit]["manufacturing"]
            points, segments, detail = _frame_witness(frames, metrics, kind)
            if points:
                issue.update(
                    points=list(points),
                    segments=list(segments),
                    point=list(points[1] if kind == "bend" else points[0]),
                    detail=detail,
                )
                issue["key"] = min(
                    editor.sections, key=lambda s: np.linalg.norm(s.points - issue["point"], axis=1).min()
                ).key
                if kind == "bend":
                    issue["label"] = (
                        f"Bend {metrics['sweep_final_min_bend_radius_m'] * 1000:.2f} / required {minimum_radius * 1000:g} mm"
                    )
        if "pointwise_field_delta_budget" in reason:
            stage = reason.split(":")[0]
            witness = report.diagnostics.get(stage, {}).get("maximum_delta_witness")
            if witness:
                issue.update(
                    point=witness["point_m"],
                    label=f"{stage.capitalize()} ΔB {witness['delta_T'] * 1e6:.3g} / limit {witness['budget_T'] * 1e6:.3g} µT",
                    detail="Target sample with maximum fixed-current vector field change; this is not a wire collision.",
                )
        issues.append(issue)
    return issues


def local_frame(editor, key):
    """Tangent frame for a visibly labelled local projection, not an unrolling."""
    row = editor.section(key)
    point = editor.cut_point(key) if row.cut is not None else row.points[len(row.points) // 2]
    mesh = editor.support
    triangles = mesh.vertices[mesh.faces]
    i = int(np.argmin(np.linalg.norm(triangles.mean(axis=1) - point, axis=1)))
    a, b, c = triangles[i]
    normal = np.cross(b - a, c - a)
    normal /= np.linalg.norm(normal)
    return dict(center=point.tolist(), normal=normal.tolist())
