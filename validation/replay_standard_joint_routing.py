"""Replay standard-gradient routing from saved oriented contours, without a solve.

The STEP export preserves directed XYZ samples but omits scalar potentials and
surface provenance. This script recovers triangle provenance on the unchanged
generated support, preserves exported contour order for potential ordering, and
uses the saved contour current. No stream function is estimated or optimized.
Reconstructed cases are routing regression inputs, not exact solver checkpoints.

Run with the repository virtualenv, for example::

    .venv/Scripts/python.exe validation/replay_standard_joint_routing.py --axis GY
    .venv/Scripts/python.exe validation/replay_standard_joint_routing.py --axis GZ --joint

Intermediate checkpoints allow independent routing experiments without repeating
topology, lift, or sweep construction. Their pickle files are local trusted data.
"""

# ruff: noqa: E402 -- Direct execution must expose repository imports first.

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import re
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from coilgen.geometry import load_mesh_from_config, refine_mesh, split_disconnected_mesh
from coilgen.joint_group_adoption import improve_group_routes
from coilgen.manufacturing import (
    _SurfaceProjector,
    create_swept_coil_meshes,
    save_manufacturing_diagnostics,
    save_swept_bodies_as_stl,
    save_wire_trace_as_csv,
)
from coilgen.models import CoilPart, ContourLine
from coilgen.routing import (
    calculate_group_centers,
    cleanup_wire_paths,
    interconnect_among_groups,
    interconnect_within_groups,
    shift_return_paths,
    topological_loop_grouping,
)
from coilgen.routing_jumps import fabricated_jump_metrics
from coilgen.target import define_target_field
from projects.standard_cylindrical_3axis_gradient.run_standard_cylindrical_3axis_gradient import PROJECT

if __name__ == "__main__":
    sys.modules["validation.replay_standard_joint_routing"] = sys.modules[__name__]


@dataclass
class ReplayCase:
    axis: str
    part: CoilPart
    config: object
    target: object
    source_dir: Path
    source_metadata: dict
    original_contours: tuple[np.ndarray, ...]

    def assert_source_unchanged(self) -> None:
        if self.part.contour_step != self.source_metadata["fixed_current_a"]:
            raise AssertionError("Replay changed the contour current.")
        if len(self.part.contour_lines) != len(self.original_contours):
            raise AssertionError("Replay changed the original contour count.")
        for contour, original in zip(self.part.contour_lines, self.original_contours):
            if not np.array_equal(contour.v, original):
                raise AssertionError("Replay changed the original oriented contour samples.")


ReplayCase.__module__ = "validation.replay_standard_joint_routing"


def latest_source_dir(axis: str) -> Path:
    base = ROOT / "outputs" / "standard_cylindrical_3axis_gradient"
    for run in sorted(base.iterdir(), reverse=True):
        candidate = run / "standard_cylindrical_3axis_gradient" / axis
        if candidate.is_dir() and list(candidate.glob("*_contour_traces.step")):
            return candidate
    raise FileNotFoundError(f"No saved {axis} contours below {base}.")


def read_step_contours(path: Path) -> list[np.ndarray]:
    """Read CoilGen's directed linear GEOMETRIC_CURVE_SET export, in file order."""
    entities = dict(re.findall(r"#(\d+)\s*=\s*(.*?);", path.read_text(encoding="ascii"), re.S))
    points, segments = {}, {}
    for identifier, entity in entities.items():
        point = re.fullmatch(r"CARTESIAN_POINT\('[^']*',\(([^)]+)\)\)", entity)
        if point:
            points[identifier] = np.array([float(x) for x in point.group(1).split(",")])
        if entity.startswith("TRIMMED_CURVE("):
            refs = re.findall(r"#(\d+)", entity)
            if len(refs) != 3:
                raise ValueError("Unsupported trimmed curve representation.")
            segments[identifier] = (refs[1], refs[2])
    traces = []
    for entity in entities.values():
        if not entity.startswith("GEOMETRIC_CURVE_SET("):
            continue
        refs = re.findall(r"#(\d+)", entity)
        vertices = []
        for reference in refs:
            first, last = segments[reference]
            if vertices and not np.array_equal(vertices[-1], points[first]):
                raise ValueError("Disconnected STEP contour export.")
            if not vertices:
                vertices.append(points[first])
            vertices.append(points[last])
        values = np.asarray(vertices)
        if not np.array_equal(values[0], values[-1]):
            raise ValueError("Open STEP contour export.")
        traces.append(values)
    if not traces:
        raise ValueError("No contour curves in STEP file.")
    return traces


def restore_surface_provenance(mesh, points, potential, projector):
    projected, _normals, _face_normals, point_faces, point_barycentric = projector.project_with_details(points)
    mids = 0.5 * (points[:-1] + points[1:])
    _projected_mid, _normal_mid, _face_normal_mid, faces, _bary_mid = projector.project_with_details(mids)
    triangles = mesh.vertices[mesh.faces[faces]]
    origin, u, v = triangles[:, 0], triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    uu, uv, vv = np.einsum("ij,ij->i", u, u), np.einsum("ij,ij->i", u, v), np.einsum("ij,ij->i", v, v)
    determinants = uu * vv - uv * uv
    bary = np.empty((len(faces), 2, 3))
    residual = 0.0
    for endpoint, samples in enumerate((points[:-1], points[1:])):
        delta = samples - origin
        du, dv = np.einsum("ij,ij->i", delta, u), np.einsum("ij,ij->i", delta, v)
        beta, gamma = (vv * du - uv * dv) / determinants, (uu * dv - uv * du) / determinants
        bary[:, endpoint] = np.column_stack((1.0 - beta - gamma, beta, gamma))
        reconstructed = np.einsum("ij,ijk->ik", bary[:, endpoint], triangles)
        residual = max(residual, float(np.max(np.linalg.norm(reconstructed - samples, axis=1))))
        # STEP decimal rounding makes independent adjacent-face coordinates
        # disagree by a few picometres. Use one shared edge parameter to avoid
        # spurious overlap intervals leaking through the virtual-cut graph.
        for segment, sample in enumerate(samples):
            row = bary[segment, endpoint]
            other = int(np.argmin(np.abs(row)))
            if abs(row[other]) >= 1e-8:
                continue
            local = [index for index in range(3) if index != other]
            indices = mesh.faces[faces[segment]]
            low, high = sorted(int(indices[index]) for index in local)
            a, b = mesh.vertices[[low, high]]
            fraction = float(np.dot(sample - a, b - a) / np.dot(b - a, b - a))
            row[:] = 0.0
            row[int(np.where(indices == low)[0][0])] = 1.0 - fraction
            row[int(np.where(indices == high)[0][0])] = fraction
    if residual > 1e-10 or np.min(bary) < -1e-8 or np.max(bary) > 1.0 + 1e-8:
        raise ValueError(f"Contour provenance does not fit generated mesh: residual {residual:g} m.")
    if np.max(np.linalg.norm(projected - points, axis=1)) > 1e-10:
        raise ValueError("Saved contour does not lie on regenerated support.")
    return ContourLine(
        points.copy(), potential=potential, current_orientation=1.0,
        surface_faces=point_faces, surface_barycentric=point_barycentric,
        segment_faces=faces, segment_barycentric=bary,
    )


def load_saved_case(axis="GZ", source_dir=None) -> ReplayCase:
    source_dir = latest_source_dir(axis) if source_dir is None else Path(source_dir).resolve()
    prefix = f"standard_cylindrical_3axis_gradient_{axis.lower()}"
    step_path = source_dir / f"{prefix}_contour_traces.step"
    stored_joint = json.loads((source_dir / "stl" / f"{prefix}_swept_body_part001_joint_routing.json").read_text())
    current = float(stored_joint["diagnostics"]["fixed_current_a"])
    config = PROJECT.resolved_designs()[axis].to_config()
    config.progress_flag = False
    config.parallel_worker_count = 1
    meshes = split_disconnected_mesh(load_mesh_from_config(config, ROOT))
    if len(meshes) != 1:
        raise ValueError("Standard routing replay expects one cylindrical support.")
    mesh = refine_mesh(meshes[0], config.iteration_num_mesh_refinement)
    original = tuple(read_step_contours(step_path))
    projector = _SurfaceProjector(mesh)
    contours = [restore_surface_provenance(mesh, points, float(index), projector) for index, points in enumerate(original)]
    part = CoilPart(mesh=mesh, contour_step=current, contour_lines=contours)
    source_metadata = {
        "source_dir": str(source_dir), "contour_step_sha256": hashlib.sha256(step_path.read_bytes()).hexdigest(),
        "fixed_current_a": current, "original_contour_count": len(contours),
        "stored_joint_diagnostics": stored_joint["diagnostics"],
        "potential_metadata": "Ordinal labels preserve STEP contour ordering; scalar potential values were not exported.",
        "stream_function": "Not reconstructed or optimized; routing uses saved directed contours and saved current.",
    }
    target = define_target_field([mesh], None, None, config)
    return ReplayCase(axis, part, config, target, source_dir, source_metadata, original)


def reconstruct_groups(case: ReplayCase) -> ReplayCase:
    parts = topological_loop_grouping([case.part], config=case.config)
    calculate_group_centers(parts)
    interconnect_within_groups(parts, case.config)
    case.assert_source_unchanged()
    return case


def reconstruct_baseline(case: ReplayCase) -> ReplayCase:
    reconstruct_groups(case)
    parts = interconnect_among_groups([case.part], case.config)
    shift_return_paths(parts, case.config)
    cleanup_wire_paths(parts, config=case.config)
    create_swept_coil_meshes(parts, case.config)
    case.assert_source_unchanged()
    return case


def summarize(case):
    case.assert_source_unchanged()
    part = case.part
    raw = part.wire_path.v if part.wire_path is not None else np.empty((0, 3))
    values = {
        "axis": case.axis, "source_dir": str(case.source_dir), "fixed_current_a": part.contour_step,
        "contour_count": len(part.contour_lines), "group_count": len(part.groups),
        "loops_per_group": [len(group.loops) for group in part.groups],
        "raw_route_length_m": float(np.linalg.norm(np.diff(raw, axis=0), axis=1).sum()),
        "route_diagnostics": part.route_diagnostics,
        "manufacturing_diagnostics": part.manufacturing_diagnostics,
        "joint_route_diagnostics": part.joint_route_diagnostics,
        "original_contour_samples_unchanged": True,
    }
    if part.fabrication_path is not None:
        values["fabrication_length_m"] = part.fabrication_path.length_m
        values["jump_metrics"] = fabricated_jump_metrics(part, case.config)
    stored_trace = next(case.source_dir.glob("*_wire_trace.csv"))
    with stored_trace.open(newline="") as stream:
        original_route = np.array([[float(row[key]) for key in ("x_m", "y_m", "z_m")] for row in csv.DictReader(stream)])
    values["stored_raw_route_point_count"] = len(original_route)
    values["replayed_raw_route_point_count"] = len(raw)
    values["stored_raw_route_max_coordinate_delta_m"] = (
        float(np.max(np.linalg.norm(raw - original_route, axis=1))) if raw.shape == original_route.shape else None
    )
    if raw.shape == original_route.shape and len(raw) > 1:
        closed_raw, closed_stored = np.linalg.norm(raw[0] - raw[-1]) < 1e-9, np.linalg.norm(original_route[0] - original_route[-1]) < 1e-9
        if closed_raw and closed_stored:
            phase = int(np.argmin(np.linalg.norm(raw[:-1] - original_route[0], axis=1)))
            aligned = np.roll(raw[:-1], -phase, axis=0)
            values["stored_raw_route_cyclic_coordinate_delta_m"] = float(np.max(np.linalg.norm(aligned - original_route[:-1], axis=1)))
            values["stored_raw_route_cyclic_phase_samples"] = phase
    return values


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def export_case(case, output, *, stl=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = SimpleNamespace(coil_parts=[case.part], input_data=case.config)
    save_wire_trace_as_csv(result, output, prefix=case.axis.lower())
    save_manufacturing_diagnostics(result, output, prefix=case.axis.lower())
    if stl:
        save_swept_bodies_as_stl(result, output, prefix=case.axis.lower())
    (output / "summary.json").write_text(json.dumps(summarize(case), indent=2, default=json_value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=("GY", "GZ"), default="GZ")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path, help="Trusted local baseline checkpoint")
    parser.add_argument("--groups-only", action="store_true")
    parser.add_argument("--joint", action="store_true")
    parser.add_argument("--export-stl", action="store_true")
    parser.add_argument("--config-overrides", type=Path, help="JSON configuration changes for an explicit routing trial")
    args = parser.parse_args()
    output = args.output or ROOT / "tmp" / "replay_standard_joint_routing" / args.axis
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if args.resume:
        with args.resume.open("rb") as stream:
            case = pickle.load(stream)
    else:
        case = load_saved_case(args.axis, args.source_dir)
        if args.config_overrides:
            for name, value in json.loads(args.config_overrides.read_text()).items():
                if not hasattr(case.config, name):
                    raise ValueError(f"Unknown config override: {name}.")
                setattr(case.config, name, value)
        reconstruct_groups(case) if args.groups_only else reconstruct_baseline(case)
        with (output / ("groups.pkl" if args.groups_only else "baseline.pkl")).open("wb") as stream:
            pickle.dump(case, stream, protocol=5)
    print(f"{case.axis}: reconstructed {len(case.part.groups)} groups in {time.perf_counter() - start:.3f}s", flush=True)
    if args.joint:
        case.part = improve_group_routes([case.part], case.config, case.target)[0]
        with (output / "selected.pkl").open("wb") as stream:
            pickle.dump(case, stream, protocol=5)
    if args.groups_only:
        (output / "source_metadata.json").write_text(json.dumps(case.source_metadata, indent=2, default=json_value))
    else:
        export_case(case, output, stl=args.export_stl)
    print(json.dumps({"axis": case.axis, "seconds": time.perf_counter() - start, "joint_status": case.part.joint_route_diagnostics.get("status")}), flush=True)


if __name__ == "__main__":
    main()
