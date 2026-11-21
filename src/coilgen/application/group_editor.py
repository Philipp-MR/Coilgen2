"""Shared draft editor over a verified, contour-stage group snapshot."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from io import BytesIO
import base64
import json

import numpy as np

from ..group_circuits import default_group_plan, group_snapshot, validate_group_plan
from ..layout import calculate_group_centers, topological_loop_grouping
from ..pipeline_stages import PipelineCheckpoint, PipelineStage
from ..models import CoilPart

GROUP_COLORS = ("#24659a", "#a45e16", "#8250a0", "#278168", "#b24f60", "#687831")


@dataclass
class GroupEditor:
    snapshot: dict
    parts: list
    plan: dict
    original: dict
    selected: set[str] = field(default_factory=set)
    focused: str = ""
    stale_plan: bool = False
    initial_plan: dict = field(default_factory=dict)
    binding: tuple | None = field(default=None, repr=False)

    @classmethod
    def from_parts(cls, parts, config):
        geometry = []
        for part in parts:
            mesh, contours, groups = deepcopy((part.mesh, part.contour_lines, part.groups))
            geometry.append(
                CoilPart(
                    mesh=mesh,
                    contour_step=part.contour_step,
                    contour_lines=contours,
                    groups=groups,
                    loop_groups=deepcopy(part.loop_groups),
                    group_levels=deepcopy(part.group_levels),
                    level_positions=deepcopy(part.level_positions),
                    loop_in_loop_matrix=deepcopy(part.loop_in_loop_matrix),
                )
            )
        parts = geometry
        if not all(part.groups for part in parts):
            topological_loop_grouping(parts, config)
        calculate_group_centers(parts)
        snapshot = group_snapshot(parts)
        plan = deepcopy(config.routing_group_plan or default_group_plan(snapshot, config))
        validate_group_plan(plan)
        original = deepcopy(plan)
        stale = plan["fingerprint"] != snapshot["fingerprint"] or {row["key"] for row in plan["groups"]} != {
            row["key"] for row in snapshot["groups"]
        }
        if stale:
            plan = default_group_plan(snapshot, config)
        validate_group_plan(plan, snapshot)
        return cls(
            snapshot,
            parts,
            plan,
            original,
            focused=snapshot["groups"][0]["key"] if snapshot["groups"] else "",
            stale_plan=stale,
            initial_plan=deepcopy(plan),
        )

    @property
    def dirty(self):
        return self.plan != self.original

    @staticmethod
    def _project_binding(project):
        values = deepcopy(project.values)
        values.pop("routing_group_plan", None)
        inputs = json.dumps(
            {"values": values, "study": project.study}, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return project.id, str(project.project_root), inputs

    def bind(self, project):
        """Bind a verified snapshot to its owner and every input except its routing plan."""
        self.binding = self._project_binding(project)

    def matches_inputs(self, project):
        return self.binding == self._project_binding(project)

    def synchronize(self, project):
        """Follow plan-only undo/redo; preserve drafts and reject changed source inputs."""
        if not self.matches_inputs(project):
            return False
        plan = deepcopy(project.config.routing_group_plan or default_group_plan(self.snapshot, project.config))
        try:
            validate_group_plan(plan, self.snapshot)
        except ValueError:
            return False
        if self.dirty and self.plan != plan:
            return False
        self.plan = plan
        self.original = deepcopy(plan)
        self.initial_plan = deepcopy(plan)
        self.stale_plan = False
        return True

    def rows(self):
        choices = {row["key"]: row for row in self.plan["groups"]}
        result = []
        for index, group in enumerate(self.snapshot["groups"]):
            parents = group["parent_groups"]
            parent_ids = sorted({index for chain in parents for index in chain})
            hierarchy = "Within " + ", ".join(f"G{index + 1}" for index in parent_ids) if parent_ids else "Top level"
            result.append(
                {
                    **group,
                    **choices[group["key"]],
                    "color": GROUP_COLORS[index % len(GROUP_COLORS)],
                    "hierarchy": hierarchy,
                    "selected": group["key"] in self.selected,
                    "focused": group["key"] == self.focused,
                }
            )
            if self.plan["mode"] == "per_group":
                result[-1]["circuit"] = group["key"].replace(":", "_")
        return result

    def summary(self, levels):
        rows = self.rows()
        paths = sum(len(part.contour_lines) for part in self.parts)
        circuits = len({row["circuit"] for row in rows})
        return f"{levels} levels · {paths} contour paths · {len(rows)} groups · {circuits} requested circuit{'s' if circuits != 1 else ''} · common evaluation current"

    def discard(self):
        self.plan = deepcopy(self.initial_plan)
        self.original = deepcopy(self.plan)

    def update(self, key, **changes):
        if set(changes) - {"circuit", "spiral", "cut"}:
            raise ValueError("Unknown group choice.")
        candidate = deepcopy(self.plan)
        if "circuit" in changes and candidate["mode"] == "per_group":
            candidate["mode"] = "custom"
            for entry in candidate["groups"]:
                entry["circuit"] = entry["key"].replace(":", "_")
        row = next((row for row in candidate["groups"] if row["key"] == key), None)
        if row is None:
            raise ValueError("The group is outside this snapshot.")
        row.update(changes)
        if "circuit" in changes:
            candidate["mode"] = "custom"
        validate_group_plan(candidate, self.snapshot)
        self.plan = candidate

    def assign(self, mode):
        if mode not in {"join", "separate", "all", "per_group"}:
            raise ValueError("Unknown group assignment operation.")
        keys = {row["key"] for row in self.plan["groups"]} if mode in {"all", "per_group"} else self.selected
        if not keys or mode == "join" and len(keys) < 2:
            raise ValueError("Select at least two groups to join, or one group to separate.")
        candidate = deepcopy(self.plan)
        if candidate["mode"] == "per_group":
            for entry in candidate["groups"]:
                entry["circuit"] = entry["key"].replace(":", "_")
        used = {row["circuit"] for row in candidate["groups"] if row["key"] not in keys}
        next_id = 1
        assigned = None
        for row in candidate["groups"]:
            if row["key"] not in keys:
                continue
            if assigned is None or mode in {"separate", "per_group"}:
                while f"C{next_id}" in used:
                    next_id += 1
                assigned = f"C{next_id}"
                used.add(assigned)
            row["circuit"] = assigned
        candidate["mode"] = "custom"
        validate_group_plan(candidate, self.snapshot)
        self.plan = candidate

    def apply_changes(self, project):
        """No-op acceptance preserves None and every existing numerical default."""
        if not self.dirty and project.config.routing_group_plan is None:
            return {}
        validate_group_plan(self.plan, self.snapshot)
        return {"routing_group_plan": deepcopy(self.plan)}


def load_group_editor(project, record):
    """Use the owned contour checkpoint, never infer topology from plotted lines."""
    from .evidence import verify_record

    record = verify_record(record, array_names=())
    if record.metadata_view.get("project_id") != project.id:
        raise ValueError("These contours belong to a different project. Load contours for the editing project.")
    owned = {artifact.get("name") for artifact in record.metadata.get("artifacts", ())}
    paths = [
        record.directory / name
        for name in ("checkpoints/contours.cgstage", "pipeline_checkpoint.cgstage")
        if name in owned
    ]
    checkpoint = next((PipelineCheckpoint.load(path) for path in paths if path.is_file()), None)
    return _inspect_contours(project, checkpoint)


def group_source(project, states, selected_id=None, rejected=()):
    """Cheap eligibility check. Numerical identity and artifact hashes are verified on Load."""
    if project is None:
        return None, "Open a model to inspect its contour groups."
    prefix = f"{project.name} · r{project.revision}"
    model = project.model_fingerprint
    matching = [
        s
        for s in states
        if s.get("project_id") == project.id and (project.id, project.fingerprint, s.get("id")) not in rejected
    ]
    matching.sort(
        key=lambda s: (
            s.get("project_model_fingerprint") == model,
            s.get("id") == selected_id,
            s.get("created_at") or "",
        ),
        reverse=True,
    )
    for state in matching:
        if "contours" in state.get("completed_stages", ()) or (
            state.get("status") == "completed" and state.get("pipeline_stage") in {"contours", "routed", "evaluated"}
        ):
            return (
                state,
                f"{prefix} · Contours recorded in run {state['id'][:8]} (r{state.get('project_revision')}, {state.get('status')}). Load verifies this checkpoint.",
            )
    active = next(
        (
            s
            for s in matching
            if s.get("status") in {"queued", "running", "cancelling"}
            and s.get("project_model_fingerprint") == model
            and s.get("pipeline_stage") in {"contours", "routed", "evaluated"}
        ),
        None,
    )
    if active:
        return (
            active,
            f"{prefix} · Run {active['id'][:8]} is preparing contours. Groups become available after that boundary.",
        )
    return (
        None,
        f"{prefix} · Prepare groups computes through contours. Inspecting groups preserves the current winding.",
    )


def load_checkpoint_groups(project, state, jobs_directory, threads):
    from .reuse import plan_stopped_reuse

    plan = plan_stopped_reuse(project, [state], jobs_directory, threads=threads, inspect_active=True)
    source = next((row for row in plan.available if row.stage == PipelineStage.CONTOURS), None)
    if source is None:
        raise ValueError(
            next(
                iter(plan.diagnostics), "This contour checkpoint is unavailable or incompatible with the applied model."
            )
        )
    return _inspect_contours(project, PipelineCheckpoint.load(source.path))


def _inspect_contours(project, checkpoint):
    if checkpoint is None or checkpoint.stage != PipelineStage.CONTOURS:
        raise ValueError("Prepare contours, then load groups from that completed run.")
    # The worker owns these execution controls. Align them for inspection of
    # its verified snapshot; this does not grant permission to resume a solve.
    recorded = checkpoint.config
    inspection_config = replace(
        project.config,
        parallel_worker_count=recorded.parallel_worker_count,
        constrained_optimization_threads=recorded.constrained_optimization_threads,
    )
    checkpoint.assert_compatible(inspection_config, project.project_root)
    return GroupEditor.from_parts(checkpoint._state.parts, project.config)


def draw_group_map(figure, editor, view="3D", camera=None):
    """Draw exact original contours with recorded group membership and direction."""
    if view not in {"3D", "XY", "XZ", "YZ"}:
        raise ValueError("Unknown group-map view.")
    figure.clear()
    axis = figure.add_subplot(111, projection="3d" if view == "3D" else None)
    coordinates = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}
    for index, row in enumerate(editor.snapshot["groups"]):
        group = editor.parts[row["support_index"]].groups[row["group_index"]]
        color = GROUP_COLORS[index % len(GROUP_COLORS)]
        active = row["key"] == editor.focused
        for loop in group.loops:
            points = loop.v * 1e3
            if view == "3D":
                axis.plot(*points.T, color=color, linewidth=1.8 if active else 0.9, alpha=1 if active else 0.65)
            else:
                first, second = coordinates[view]
                axis.plot(
                    points[:, first],
                    points[:, second],
                    color=color,
                    linewidth=1.8 if active else 0.9,
                    alpha=1 if active else 0.65,
                )
        points = group.loops[0].v * 1e3
        segment = max(0, (len(points) - 2) // 3)
        sign = -1 if (editor.parts[row["support_index"]].contour_step or 1) < 0 else 1
        start, end = (points[segment], points[segment + 1]) if sign > 0 else (points[segment + 1], points[segment])
        center = np.asarray(row["center_m"]) * 1e3
        if view == "3D":
            axis.text(*center, row["label"] if len(editor.parts) > 1 else f"G{row['group_index'] + 1}", color=color)
            axis.quiver(*start, *(end - start), color=color, arrow_length_ratio=0.5)
        else:
            first, second = coordinates[view]
            axis.annotate(row["label"], center[[first, second]], xytext=(4, 5), textcoords="offset points", color=color)
            axis.annotate(
                "", end[[first, second]], start[[first, second]], arrowprops={"arrowstyle": "->", "color": color}
            )
    if view == "3D":
        axis.set_xlabel("X [mm]")
        axis.set_ylabel("Y [mm]")
        axis.set_zlabel("Z [mm]")
        all_points = np.concatenate([loop.v for part in editor.parts for loop in part.contour_lines])
        axis.set_box_aspect(np.maximum(np.ptp(all_points, axis=0), 1e-6))
    else:
        first, second = coordinates[view]
        axis.set_xlabel("XYZ"[first] + " [mm]")
        axis.set_ylabel("XYZ"[second] + " [mm]")
        axis.set_aspect("equal", adjustable="datalim")
    if camera is not None:
        if view == "3D":
            axis.view_init(elev=camera["elev"], azim=camera["azim"])
            axis.set_zlim(camera["zlim"])
        axis.set_xlim(camera["xlim"])
        axis.set_ylim(camera["ylim"])
    if not figure.get_constrained_layout():
        figure.tight_layout()
    figure.canvas.draw()
    return group_hit_positions(axis, editor, view)


def group_hit_positions(axis, editor, view):
    """Pick the projected contour geometry, including nested groups with coincident centers."""
    from mpl_toolkits.mplot3d import proj3d

    positions = []
    coordinates = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}
    for row in editor.snapshot["groups"]:
        group = editor.parts[row["support_index"]].groups[row["group_index"]]
        for loop in group.loops:
            points = np.vstack((loop.v, (loop.v[1:] + loop.v[:-1]) / 2)) * 1e3
            points = points[:: max(1, len(points) // 512)]
            projected = (
                np.column_stack(proj3d.proj_transform(*points.T, axis.get_proj())[:2])
                if view == "3D"
                else points[:, coordinates[view]]
            )
            pixels = axis.transData.transform(projected)
            positions.extend(
                (row["key"], float(pixel[0]), float(axis.figure.bbox.height - pixel[1])) for pixel in pixels
            )
    return positions


def group_map_image(editor, view="3D"):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    figure = Figure(figsize=(8, 3.7), dpi=100)
    FigureCanvasAgg(figure)
    positions = draw_group_map(figure, editor, view)
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=100)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"), positions
