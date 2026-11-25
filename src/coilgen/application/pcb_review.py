"""Shared PCB review wording and read-only connection previews for both clients."""

import numpy as np


# Order actionable circuit defects before comparisons and CAD completion notes.
_GROUPS = {
    "seam": (
        0,
        "Seam cuts",
        "Move the opening away from copper. Every cut remains a real open end.",
        "seam",
        "Find better seam",
    ),
    "circuit": (
        1,
        "Circuit connections",
        "Pick two free endpoints, connect the groups, then assign the two supply terminals.",
        "connect",
        "Edit connections",
    ),
    "self_crossing": (
        2,
        "Crossing paths",
        "Inspect the crossing. Use two-layer group spirals or reroute the affected copper in PCB CAD.",
        "connect",
        "Inspect connections",
    ),
    "spacing": (
        3,
        "Copper spacing",
        "Inspect each location; reduce width where appropriate, change a layer, or reroute in PCB CAD.",
        "copper",
        "Edit copper",
    ),
    "outside": (
        4,
        "Copper outside board",
        "Inspect support openings and copper extent. Board extensions and local rerouting are completed in PCB CAD.",
        "board",
        "Inspect board",
    ),
    "edge": (
        5,
        "Board-edge clearance",
        "Compare the required edge margin with copper width. Extend the board or reroute in PCB CAD when needed.",
        "copper",
        "Edit copper",
    ),
    "off_surface": (
        6,
        "Saved-path displacement",
        "Compare the saved path with the PCB path. Evaluate the completed circuit to quantify the field change; local rerouting is completed in PCB CAD.",
        "compare",
        "Compare saved path",
    ),
    "via_spec": (
        8,
        "Finish in PCB CAD",
        "Define drill, plating, pad and stackup dimensions in PCB CAD before fabrication.",
        "export",
        "Review handoff",
    ),
}


def issue_groups(issues):
    groups = {}
    for index, issue in enumerate(issues):
        code = issue["code"]
        if code in {"branch", "terminals", "terminal_location", "disconnected", "layer_stack"}:
            code = "circuit"
        elif code in {"clearance", "turn_clearance", "via_clearance"}:
            code = "spacing"
        order, title, hint, action, label = _GROUPS.get(
            code, (7, "Checks need attention", "Inspect the finding before using this artwork.", "", "")
        )
        group = groups.setdefault(
            code,
            dict(
                key=code,
                order=order,
                title=title,
                hint=hint,
                action=action,
                action_label=label,
                indices=[],
                locations=[],
                traces=[],
            ),
        )
        group["indices"].append(index)
        if issue.get("uv"):
            group["locations"].append(index)
        if issue.get("trace") and issue["trace"] not in group["traces"]:
            group["traces"].append(issue["trace"])
    for group in groups.values():
        group["count"] = len(group["indices"])
        group["label"] = f"{group['title']} · {group['count']}"
        if len(group["traces"]) > 1:
            group["label"] += f" findings on {len(group['traces'])} traces"
    return sorted(groups.values(), key=lambda g: (g["order"], g["key"]))


def review_summary(design, evaluation, problem=""):
    errors = len(design.errors) if design else 0
    notes = sum(i.severity == "warning" for i in design.issues) if design else 0
    complete = design is not None and design.path is not None
    budget = (evaluation or {}).get("field_budget", {})
    field = (
        "Field not evaluated"
        if evaluation is None
        else "Field budget passed"
        if budget.get("passed") is True
        else "Field budget failed"
        if budget.get("passed") is False
        else "Field budget unverified"
    )
    inspection = bool(problem or errors or not complete or budget.get("passed") is not True)
    summary = f"{errors} issues" + (f" · {notes} notes" if notes else "")
    summary += f" · {'Circuit complete' if complete else 'Circuit incomplete'} · {field}"
    if problem:
        summary = "Layout unavailable · " + problem
    return dict(
        errors=errors,
        notes=notes,
        circuit_complete=complete,
        field=field,
        summary=summary,
        inspection=inspection,
        export_label="Export inspection artwork" if inspection else "Export PCB CAD package",
    )


def connection_preview(design, a, b, layer, connections=()):
    """Check only the proposed link and landings; full layout checks run on commit."""
    from shapely.geometry import LineString, Point
    from shapely.ops import unary_union

    if design is None or a not in design.endpoints or b not in design.endpoints or a == b:
        raise ValueError("Choose two different source endpoints.")
    if layer not in {"top", "bottom"}:
        raise ValueError("Choose top or bottom copper.")
    ends = [design.endpoints[k] for k in (a, b)]
    uv = np.array([e["uv"] for e in ends])
    frame, settings = design.frame, design.settings
    points = frame.path_points(uv, layer, settings.layer_gap_m)
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    transitions = [e for e in ends if e["layer"] != layer]
    duplicate = any({c["a"], c["b"]} == {a, b} for c in connections)
    warnings = ["These endpoints are already connected."] if duplicate else []
    line = LineString(uv) if np.linalg.norm(uv[1] - uv[0]) > 1e-10 else Point(uv[0])
    copper = line.buffer(settings.width_m / 2, quad_segs=12)
    if not design.board.buffer(1e-9).covers(copper):
        warnings.append("The connection leaves the board or crosses a support opening.")
    elif not design.board.buffer(-settings.edge_m).buffer(1e-9).covers(copper):
        warnings.append("The connection does not meet the board-edge clearance.")
    threshold = settings.width_m + settings.clearance_m
    for trace in design.traces:
        existing = unary_union([LineString(p) for p in design.flat_lines[trace.key]])
        if trace.layer == layer:
            candidate, other = line, existing
            for endpoint in ends:
                if endpoint["trace"] == trace.key:
                    local = Point(endpoint["uv"]).buffer(threshold * 1.5)
                    candidate, other = candidate.difference(local), other.difference(local)
            if not candidate.is_empty and not other.is_empty and candidate.distance(other) < threshold - 1e-10:
                warnings.append(f"Copper crossing or insufficient gap to {trace.label}.")
        for endpoint in transitions:
            other = existing
            if endpoint["trace"] == trace.key or any(
                np.linalg.norm(endpoint["uv"] - p) < 1e-8 for p in (trace.uv[0], trace.uv[-1])
            ):
                other = other.difference(Point(endpoint["uv"]).buffer(threshold * 1.5))
            if not other.is_empty and Point(endpoint["uv"]).distance(other) < threshold - 1e-10:
                warnings.append(f"Layer transition is too close to {trace.label}.")
    for via in design.vias:
        if min(np.linalg.norm(via - p) for p in uv) > 1e-8 and Point(via).distance(line) < threshold - 1e-10:
            warnings.append("The connection is too close to an existing layer transition.")
    if transitions and settings.layer_gap_m <= 0:
        warnings.append("Layer transitions require positive layer separation.")
    warnings = list(dict.fromkeys(warnings))
    return dict(
        a=a,
        b=b,
        layer=layer,
        flat=uv.tolist(),
        wrapped=points.tolist(),
        length_mm=length * 1000,
        transitions=len(transitions),
        warnings=warnings,
        duplicate=duplicate,
        summary=f"{length * 1000:.2f} mm trace · {len(transitions)} layer transitions · {len(warnings)} concerns",
        scope="Proposed connection only; full PCB checks run when added.",
        pins=[
            dict(
                label=label,
                key=key,
                name=e["label"],
                uv=e["uv"].tolist(),
                xyz=frame.wrap([e["uv"]], e["layer"], settings.layer_gap_m)[0].tolist(),
            )
            for label, key, e in zip(("A", "B"), (a, b), ends)
        ],
        via_lines=[
            np.vstack(
                (
                    frame.wrap([e["uv"]], e["layer"], settings.layer_gap_m),
                    frame.wrap([e["uv"]], layer, settings.layer_gap_m),
                )
            ).tolist()
            for e in transitions
        ],
    )
