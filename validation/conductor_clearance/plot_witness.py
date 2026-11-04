"""Plot measured copper facets and a true section through a selected GUI STL clearance."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from audit_exports import load_bodies
from surface_gap import closest_features


COLORS = ("#bc6429", "#157583")
INK = "#203342"


def unit(v):
    return v / np.linalg.norm(v)


def crossing_faces(body, normal, ids, roles):
    """Select intervals that cross when projected along the first recorded section normal."""
    p = body.points
    v = np.diff(p, axis=0)
    count = 2 * len(body.profile)
    copper = body.owners.astype(str)
    coils = (copper == "inter_group_connector") | (copper == "within_group_return")
    # Prefer the two opposite legs of one cut. Otherwise illustrate a
    # connector or return passing over an independent winding turn.
    for same_cut in (True, False):
        pairs = []
        for a in np.flatnonzero(coils):
            if same_cut:
                allowed = (ids == ids[a]) & (roles != roles[a]) & (copper == "inter_group_connector")
                if copper[a] != "inter_group_connector" or roles[a] != "primary":
                    continue
            else:
                allowed = np.isin(copper, ("contour_loop", "contour_group"))
            b = np.flatnonzero(allowed)
            b = b[body.separation(a * count, b * count) > 4 * body.diameter]
            if not len(b):
                continue
            n = normal[a]
            det = np.cross(v[a], v[b]) @ n
            transverse = abs(det) > 1e-5 * np.linalg.norm(v[a]) * np.linalg.norm(v[b], axis=1)
            b, det = b[transverse], det[transverse]
            offset = p[b] - p[a]
            s = (np.cross(offset, v[b]) @ n) / det
            t = (np.cross(offset, v[a]) @ n) / det
            z = (offset + t[:, None] * v[b] - s[:, None] * v[a]) @ n
            hit = (s >= -1e-8) & (s <= 1 + 1e-8) & (t >= -1e-8) & (t <= 1 + 1e-8) & (abs(z) < 8 * body.diameter)
            pairs.extend((a, k) for k in b[hit])
        if not pairs:
            continue
        ia, ib = np.indices((count, count)).reshape(2, -1)
        best = None
        for a, b in pairs:
            i, j = count * a + ia, count * b + ib
            d, _, _ = closest_features(body.triangles[i], body.triangles[j])
            k = np.argmin(d)
            candidate = d[k], int(i[k]), int(j[k])
            if best is None or candidate < best:
                best = candidate
        return best[1:], "opposing cut legs" if same_cut else "connection and winding turn"
    raise ValueError("No separated visits with a projected crossing were found. Select a pair using --faces.")


def owner_label(owner):
    kind = owner["kind"].replace("_", " ")
    if owner["kind"] == "inter_group_connector":
        return f"Cut {owner['connector_index']} · {owner['connector_role']} connector"
    detail = [f"group {owner['group_index']}"] if owner["group_index"] >= 0 else []
    if owner["loop_index"] >= 0:
        detail.append(f"loop {owner['loop_index']}")
    return " · ".join([kind.capitalize(), *detail])


def section_lines(triangles, perpendicular, horizontal, vertical):
    lines = []
    for tri in triangles:
        h = tri @ perpendicular
        if h.min() > 1e-9 or h.max() < -1e-9:
            continue
        points = list(tri[abs(h) < 1e-9])
        for i in range(3):
            j = (i + 1) % 3
            if h[i] * h[j] < 0:
                points.append(tri[i] + (h[i] / (h[i] - h[j])) * (tri[j] - tri[i]))
        if len(points) >= 2:
            points = np.asarray(points)
            k, j = np.unravel_index(
                np.argmax(np.linalg.norm(points[:, None] - points, axis=2)), (len(points), len(points))
            )
            if np.linalg.norm(points[k] - points[j]) > 1e-9:
                lines.append(points[[k, j]] @ np.column_stack((horizontal, vertical)))
    return lines


def draw(body, frames, faces, title, output, method):
    normal = frames["normals"]
    i, j = faces
    d, x, y = closest_features(body.triangles[[i]], body.triangles[[j]])
    d = float(d[0])
    x, y = x[0], y[0]
    origin = (x + y) / 2
    stations = body.station[[i, j]]
    labels = []
    features = []
    for face, station, point in zip(faces, stations, (x, y)):
        attrs = {
            k: (str(v[station]) if k in {"kind", "connector_role"} else int(v[station]))
            for k, v in frames.items()
            if k != "normals"
        }
        labels.append(owner_label(attrs))
        features.append(
            {**body.feature(face, point), "owner_detail": attrs, "triangle_vertices_m": body.triangles[face].tolist()}
        )
    required = 2 * body.insulation + body.tolerance + body.margin
    digest = body.stl_source["sha256"]
    result = {
        "body": body.name,
        "selection": method,
        "index_base": 0,
        "scope": "minimum between the two selected STL station intervals"
        if method != "specified facets"
        else "minimum between the two selected STL facets",
        "required_gap_m": required,
        "distance_m": d,
        "stl": body.stl_source,
        "first": features[0],
        "second": features[1],
    }
    a, b = stations
    ta, tb = unit(body.points[a + 1] - body.points[a]), unit(body.points[b + 1] - body.points[b])
    n = unit(normal[a] + np.sign(normal[a] @ normal[b]) * normal[b])
    along = unit(ta - n * (ta @ n))
    across = unit(np.cross(n, along))
    basis = np.column_stack((along, across, n))
    gap = unit(y - x) if d > 1e-12 else n
    tangent = ta + (1 if ta @ tb >= 0 else -1) * tb
    perp = tangent - gap * (tangent @ gap)
    if np.linalg.norm(perp) < 1e-8:
        perp = across - gap * (across @ gap)
    perp = unit(perp)
    up = n - perp * (perp @ n)
    if np.linalg.norm(up) < 1e-8:
        up = gap
    up = unit(up)
    horiz = unit(np.cross(up, perp))
    section = np.column_stack((horiz, up))
    result["section_plane"] = {
        "origin_m": origin.tolist(),
        "normal": perp.tolist(),
        "transverse_axis": horiz.tolist(),
        "height_axis": up.tolist(),
    }
    points = np.array([x - origin, y - origin]) * 1000
    witness3d = points @ basis
    witness2d = points @ section
    patches = []
    arc = (body.arc[:-1] + body.arc[1:]) / 2
    for k in stations:
        distance = abs(arc - arc[k])
        distance = np.minimum(distance, body.arc[-1] - distance) if body.closed else distance
        distance = np.maximum(distance - np.diff(body.arc) / 2, 0.0)
        patches.append((body.triangles[distance[body.station] < 2.6 * body.diameter] - origin) * 1000)

    fig = plt.figure(figsize=(11.4, 6.9), facecolor="white")
    ax = fig.add_axes([0.025, 0.22, 0.54, 0.61], projection="3d")
    sec = fig.add_axes([0.626, 0.292, 0.315, 0.40])
    fig.text(0.045, 0.938, title, fontsize=21, weight="semibold", color=INK)
    fig.text(0.046, 0.884, "Exported copper surfaces   /   " + method.capitalize(), fontsize=11, color="#49616c")
    fig.text(0.046, 0.811, "Oblique view", fontsize=12, weight="semibold", color=INK)
    fig.text(0.626, 0.775, "Section through the closest points", fontsize=12, weight="semibold", color=INK)
    allmesh = []
    alllines = []
    for patch, color in zip(patches, COLORS):
        local = patch @ basis
        allmesh.append(local.reshape(-1, 3))
        mesh = Poly3DCollection(local, facecolor=color, edgecolor="#ffffff55", linewidth=0.25, alpha=0.90)
        ax.add_collection3d(mesh)
        lines = section_lines(patch, perp, horiz, up)
        alllines.extend(lines)
        sec.add_collection(LineCollection(lines, colors=color, linewidths=2.2))
    ax.plot(*witness3d.T, color=INK, linewidth=2.2)
    ax.scatter(*witness3d.T, c=INK, s=20, depthshade=False)
    ax.view_init(elev=21, azim=-61)
    ax.set_proj_type("ortho")
    verts = np.vstack(allmesh)
    lo, hi = verts.min(0), verts.max(0)
    span = np.maximum(hi - lo, 0.7 * body.diameter * 1000)
    center = (lo + hi) / 2
    for k, set_limit in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        set_limit(center[k] - span[k] * 0.53, center[k] + span[k] * 0.53)
    ax.set_box_aspect(span)
    ax.set_axis_off()
    sec.plot(*witness2d.T, "o-", color=INK, markersize=4, linewidth=1.6, zorder=3)
    if alllines:
        bounds = np.vstack(alllines)
        lo, hi = bounds.min(0), bounds.max(0)
        pad = max(0.2, max(hi - lo) * 0.10)
        sec.set_xlim(lo[0] - pad, hi[0] + pad)
        sec.set_ylim(lo[1] - pad, hi[1] + pad)
    sec.set_aspect("equal", adjustable="box")
    sec.set_xlabel("Transverse position (mm)")
    sec.set_ylabel("Height (mm)")
    sec.tick_params(labelsize=9, colors=INK)
    sec.grid(color="#dae3e5", linewidth=0.55)
    sec.set_axisbelow(True)
    for edge in ("top", "right"):
        sec.spines[edge].set_visible(False)
    sec.spines["bottom"].set_color("#a8b8ba")
    sec.spines["left"].set_color("#a8b8ba")
    fig.text(0.626, 0.727, f"{1000 * d:.4f} mm", color=INK, fontsize=17, weight="bold")
    fig.text(0.79, 0.732, f"Required  {1000 * required:.3f} mm", color="#526b76", fontsize=10)
    for idx, (label, point, face, station, color) in enumerate(zip(labels, (x, y), faces, stations, COLORS)):
        h = 0.178 - 0.059 * idx
        fig.text(0.047, h, "●", color=color, fontsize=13)
        fig.text(0.069, h, label, fontsize=10.7, color=INK, weight="semibold")
        fig.text(0.405, h, f"STL face {face}   ·   station {station}", fontsize=9.6, color=INK)
        fig.text(0.653, h, "XYZ (mm)   " + "   ".join(f"{q:.4f}" for q in point * 1000), fontsize=9.2, color=INK)
    fig.text(
        0.046,
        0.046,
        f"{body.name}    ·    STL SHA-256  {digest[:16]}…    ·    Zero-based indices    ·    XYZ in the GUI world frame",
        fontsize=8.5,
        color="#596b73",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True, type=Path)
    p.add_argument("--case", required=True)
    p.add_argument("--body", type=int, default=0, help="Zero-based reviewed conductor index.")
    p.add_argument("--faces", nargs=2, type=int, help="Two zero-based STL face ids to reproduce a measured witness.")
    p.add_argument("--title")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    directory = Path(json.loads(args.runs.read_bytes())[args.case])
    body = load_bodies(args.case, directory)[args.body]
    record = json.loads((directory / "result.json").read_bytes())
    body.stl_source.update(
        job=directory.name,
        request_sha256=record["request_sha256"],
        project_model_fingerprint=record["project_model_fingerprint"],
    )
    prefix = f"review_conductor_{args.body}"
    with np.load(directory / "scientific_result.npz", allow_pickle=False) as source:
        frames = {
            k: source[f"{prefix}_owner_{k}"]
            for k in ("kind", "group_index", "loop_index", "connector_index", "connector_role")
        }
        frames["normals"] = source[f"{prefix}_frame_normals"]
    faces, method = (
        (args.faces, "specified facets")
        if args.faces
        else crossing_faces(body, frames["normals"], frames["connector_index"], frames["connector_role"])
    )
    r = draw(
        body,
        frames,
        faces,
        args.title or f"Copper crossing in {args.case.split('__')[0].replace('_', ' ')}",
        args.output,
        method,
    )
    print(f"{args.output}   {1000 * r['distance_m']:.6f} mm")


if __name__ == "__main__":
    main()
