"""Reproducible, standalone contour topology atlas. Geometry is in millimetres.

No CoilGen imports or production changes. Candidate open series paths are
illustrations, not collision-certified or field-optimized manufactured coils.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent
TAU = 2 * np.pi
COLORS = {"contour": "#087e8b", "join": "#dd7925", "transfer": "#9654bb"}


@dataclass
class Piece:
    xyz: np.ndarray
    kind: str


@dataclass
class Case:
    key: str
    title: str
    topology: str
    strategy: str
    caveat: str
    grids: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    pieces: list[Piece] = field(default_factory=list)
    ports: list = field(default_factory=list)

    def add_arc(self, points, source):
        self.sources.append(source)
        self.pieces.append(Piece(points, "contour"))

    def connect(self, points, lifted=False):
        self.pieces.append(Piece(points, "transfer" if lifted else "join"))


def plane(uv):
    uv = np.asarray(uv)
    return np.column_stack((uv, np.zeros(len(uv))))


def dome(uv):
    uv = np.asarray(uv)
    x, y = uv.T
    return np.column_stack((x, y, 15 - .004 * (x*x + y*y)))


def saddle(uv):
    uv = np.asarray(uv)
    x, y = uv.T
    return np.column_stack((x, y, .0025 * (x*x-y*y)))


def cylinder(uv):
    u, z = np.asarray(uv).T
    return np.column_stack((32*np.cos(u), 32*np.sin(u), z))


def torus(uv):
    u, v = np.asarray(uv).T
    return np.column_stack(((37+14*np.cos(v))*np.cos(u),
                            (37+14*np.cos(v))*np.sin(u), 14*np.sin(v)))


def grid(case, mapping, xr, yr, n=17):
    for x in np.linspace(*xr, n):
        case.grids.append(mapping(np.column_stack((np.full(45, x), np.linspace(*yr, 45)))))
    for y in np.linspace(*yr, n):
        case.grids.append(mapping(np.column_stack((np.linspace(*xr, 45), np.full(45, y)))))


def bezier(a, b, ta, tb, mapping, lift=0, handle=None):
    a, b, ta, tb = (np.asarray(v, dtype=float) for v in (a, b, ta, tb))
    h = .32 * np.linalg.norm(b-a) if handle is None else handle
    c = a + h*ta/np.linalg.norm(ta)
    d = b - h*tb/np.linalg.norm(tb)
    t = np.linspace(0, 1, 65)[:, None]
    uv = (1-t)**3*a + 3*(1-t)**2*t*c + 3*(1-t)*t*t*d + t**3*b
    xyz = mapping(uv)
    # This explicit vertical service-layer excursion is a schematic candidate.
    xyz[:, 2] += lift * np.sin(np.pi*t[:, 0])**2
    return xyz


def disk_group(case, mapping, radii, center=(0, 0), direction=1, seam=0, aspect=1):
    """Append a directed chain; return its parameter-space entry/exit state."""
    center = np.asarray(center)
    states = []
    a = .16
    for r in radii:
        angles = np.linspace(seam+direction*a, seam+direction*(TAU-a), 240)
        xy = center + np.column_stack((r*np.cos(angles), aspect*r*np.sin(angles)))
        full = direction * np.linspace(0, TAU, 260)
        source = mapping(center + np.column_stack((r*np.cos(full), aspect*r*np.sin(full))))
        entry = xy[0]
        end = xy[-1]
        te = direction*np.array([-np.sin(angles[0]), aspect*np.cos(angles[0])])
        tx = direction*np.array([-np.sin(angles[-1]), aspect*np.cos(angles[-1])])
        if states:
            last = states[-1]
            case.connect(bezier(last[1], entry, last[3], te, mapping))
        case.add_arc(mapping(xy), source)
        states.append((entry, end, te, tx))
    return states[0][0], states[-1][1], states[0][2], states[-1][3]


def append_transfer(case, previous, following, mapping, lift):
    # Insert before the following group's pieces to retain actual winding order.
    return Piece(bezier(previous[1], following[0], previous[3], following[2], mapping, lift), "transfer")


def periodic_group(case, mapping, levels, directions):
    previous = None
    for level, direction in zip(levels, directions):
        a = .16
        u = direction * np.linspace(a, TAU-a, 240)
        pts = mapping(np.column_stack((u, np.full(len(u), level))))
        source_u = direction * np.linspace(0, TAU, 260)
        source = mapping(np.column_stack((source_u, np.full(len(source_u), level))))
        entry, end = np.array([direction*a, level]), np.array([-direction*a, level])
        tangent = np.array([direction, 0])
        if previous is not None:
            # Keep all seam coordinates close to zero, never across a UV wrap.
            case.connect(bezier(previous[0], entry, previous[1], tangent, mapping, handle=.15))
        case.add_arc(pts, source)
        previous = (end, tangent)


def make_cases():
    cases = []
    c = Case("01_nested_flat", "01  Nested disk loops", "Contractible / one nesting chain",
             "One coherent seam; monotone outside-to-inside spiral.",
             "Open path: inner terminal still needs a designed return.")
    grid(c, plane, (-55, 55), (-55, 55))
    disk_group(c, plane, [48, 42, 36, 30, 24, 18, 12])
    cases.append(c)

    c = Case("02_nested_curved", "02  Same nesting, curved support", "Same intrinsic topology as case 01",
             "Transport the seam over the surface; check physical spacing.",
             "Curvature changes length, bend radius and winding access.")
    grid(c, dome, (-55, 55), (-55, 55))
    disk_group(c, dome, [48, 42, 36, 30, 24, 18, 12])
    cases.append(c)

    for opposite in (False, True):
        num = 4 if opposite else 3
        c = Case(f"0{num}_lobes", f"0{num}  Two lobes: {'opposite' if opposite else 'same'} circulation",
                 "Two separate contractible contour families",
                 "Spiral in, transfer, spiral out; choose both port states together.",
                 "Purple transfer is lifted; closed-circuit splice parity differs.")
        grid(c, saddle, (-65, 65), (-34, 34))
        left = disk_group(c, saddle, [26, 19, 12], (-33, 0), 1, 0)
        split = len(c.pieces)
        right = disk_group(c, saddle, [12, 19, 26], (33, 0), -1 if opposite else 1, np.pi)
        c.pieces.insert(split, append_transfer(c, left, right, saddle, 18))
        cases.append(c)

    c = Case("05_branching", "05  Parent family with two children", "Nesting tree / saddle split between levels",
             "Local spirals; coordinate sibling visits and transfer corridors.",
             "The contour tree branches; the electrical wire must remain a chain.")
    grid(c, dome, (-69, 69), (-45, 45))
    parent = disk_group(c, dome, [62, 56], aspect=.61, seam=-np.pi/2)
    split = len(c.pieces)
    left = disk_group(c, dome, [10, 17], (-26, 0), seam=-np.pi/2)
    c.pieces.insert(split, append_transfer(c, parent, left, dome, 20))
    split = len(c.pieces)
    right = disk_group(c, dome, [17, 10], (26, 0), seam=-np.pi/2)
    c.pieces.insert(split, append_transfer(c, left, right, dome, 20))
    cases.append(c)

    c = Case("06_hole", "06  Punctured patch / annulus", "Loops encircle a real boundary hole",
             "Cut across an annular band; exploit inner-rim terminal access.",
             "The hole is not a permissible surface shortcut.")
    for r in np.linspace(8, 53, 12):
        t = np.linspace(0, TAU, 100)
        c.grids.append(dome(np.column_stack((r*np.cos(t), r*np.sin(t)))))
    for t in np.linspace(0, TAU, 18, endpoint=False):
        r = np.linspace(8, 53, 40)
        c.grids.append(dome(np.column_stack((r*np.cos(t), r*np.sin(t)))))
    disk_group(c, dome, [48, 42, 36, 30, 24, 18, 12])
    cases.append(c)

    c = Case("07_cylinder", "07  Cylindrical circumferential bands", "Essential loops on a cylinder",
             "Use one axial seam, producing a localized-pitch helix.",
             "A developed rectangle must retain periodic seam identification.")
    grid(c, cylinder, (0, TAU), (-49, 49), 13)
    periodic_group(c, cylinder, [-42, -28, -14, 0, 14, 28, 42], [1]*7)
    cases.append(c)

    c = Case("08_torus", "08  Torus: paired essential families", "Real stream function psi = sin(v)",
             "Track homology and circulation; connect through a common seam.",
             "Full regular levels contain opposite homology contributions.")
    grid(c, torus, (0, TAU), (0, TAU), 15)
    levels = [-np.pi/6, 0, np.pi/6, 5*np.pi/6, np.pi, 7*np.pi/6]
    periodic_group(c, torus, levels, [-1, -1, -1, 1, 1, 1])
    cases.append(c)

    c = Case("09_boundary_arcs", "09  Boundary-to-boundary arcs", "Nonconstant boundary stream function",
             "Preserve each arc direction; budget a return for every pass.",
             "An ordinary back-and-forth meander reverses alternate arcs.")
    grid(c, dome, (-48, 48), (-40, 40))
    for i, x in enumerate([-36, -18, 0, 18, 36]):
        uv = np.column_stack((np.full(120, x), np.linspace(-40, 40, 120)))
        if i:
            c.connect(bezier([x-18, 40], [x, -40], [0, 1], [0, 1], dome, lift=28), True)
        c.add_arc(dome(uv), dome(uv))
    cases.append(c)

    c = Case("10_disconnected", "10  Disconnected support panels", "Different connected components",
             "Wind each module locally; connect via an explicit service cable.",
             "An off-surface transfer is unavoidable for one series path.")
    grid(c, plane, (-65, -8), (-32, 32))
    grid(c, plane, (8, 65), (-32, 32))
    left = disk_group(c, plane, [25, 18, 11], (-36, 0))
    split = len(c.pieces)
    right = disk_group(c, plane, [11, 18, 25], (36, 0), seam=np.pi)
    c.pieces.insert(split, append_transfer(c, left, right, plane, 22))
    cases.append(c)
    return cases


def decorate_3d(ax, case, title=True):
    for p in case.grids:
        ax.plot(*p.T, color="#b2bdc5", lw=.45, alpha=.40)
    for p in case.sources:
        ax.plot(*p.T, color="#84949e", lw=.65, ls="--", alpha=.45)
    for p in case.pieces:
        ax.plot(*p.xyz.T, color=COLORS[p.kind], lw=2.0 if p.kind == "contour" else 2.5)
        if p.kind == "contour":
            k = len(p.xyz)//3
            delta = p.xyz[k+3] - p.xyz[k]
            ax.quiver(*p.xyz[k], *delta, color=COLORS[p.kind], length=5.5,
                      normalize=True, arrow_length_ratio=.6, linewidth=1.2)
    start, end = case.pieces[0].xyz[0], case.pieces[-1].xyz[-1]
    ax.scatter(*start, color="#152b3c", s=25, marker="o", depthshade=False)
    ax.scatter(*end, color="#152b3c", s=30, marker="s", depthshade=False)
    ax.text(*start, "  A", fontsize=8, color="#152b3c")
    ax.text(*end, "  B", fontsize=8, color="#152b3c")
    xyz = np.concatenate([p for p in case.grids] + [p.xyz for p in case.pieces])
    lo, hi = xyz.min(0), xyz.max(0)
    center = (lo+hi)/2
    span = max(hi-lo)*1.05
    for setter, mid in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(mid-span/2, mid+span/2)
    ax.set_box_aspect((1, 1, 1), zoom=1.15)
    ax.view_init(35, -65)
    ax.set_axis_off()
    if title:
        ax.set_title(case.title, fontsize=12, fontweight="normal", loc="left", pad=-8)


def validate(cases):
    report = []
    for c in cases:
        gaps = [float(np.linalg.norm(p.xyz[-1] - q.xyz[0])) for p, q in zip(c.pieces, c.pieces[1:])]
        all_points = np.concatenate([p.xyz for p in c.pieces])
        assert np.isfinite(all_points).all()
        assert max(gaps, default=0) < 1e-9, (c.key, max(gaps))
        orientation_checks = 0
        for source, retained in zip(c.sources, (p.xyz for p in c.pieces if p.kind == "contour")):
            if np.linalg.norm(source[0]-source[-1]) < 1e-9:
                def signed_area(points):
                    xy = points[:, :2]
                    return np.sum(xy[:, 0]*np.roll(xy[:, 1], -1)-xy[:, 1]*np.roll(xy[:, 0], -1))/2
                assert signed_area(source)*signed_area(retained) > 0, c.key
                orientation_checks += 1
        report.append({"case": c.key, "retained_arcs": sum(p.kind == "contour" for p in c.pieces),
                       "connections": sum(p.kind != "contour" for p in c.pieces),
                       "source_arc_orientation_checks": orientation_checks,
                       "max_endpoint_gap_mm": max(gaps, default=0),
                       "status": "finite coordinates, source/arc circulation and ordered C0 continuity verified; fabrication and fields not certified"})
    return report


def export(cases):
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.facecolor": "white"})
    for page, subset in enumerate((cases[:5], cases[5:]), 1):
        fig = plt.figure(figsize=(16, 11.5))
        for i, c in enumerate(subset):
            ax = fig.add_subplot(2, 3, i+1, projection="3d")
            decorate_3d(ax, c)
            ax.text2D(.02, -.015, c.topology + "\n" + c.strategy, transform=ax.transAxes,
                      fontsize=8.5, color="#354957", wrap=True)
        ax = fig.add_subplot(2, 3, 6)
        ax.axis("off")
        ax.text(.06, .83, "READING THE ATLAS", fontsize=16, color="#173749")
        ax.text(.06, .68, "Teal + arrows: prescribed contour direction\nOrange: joins within the surface\nPurple: lifted or off-surface transfers\nCircle A / square B: open-path terminals\nGray dashes: original uncut contours",
                fontsize=12, linespacing=1.8, va="top", color="#354957")
        ax.text(.06, .22, "Constructive routing motifs, not certified coils.\nMillimetres; equal 3D axis scale.\nA return/feed path must still be designed.",
                fontsize=11, linespacing=1.6, color="#354957")
        fig.suptitle(f"Contour topology and interconnection  |  {page}/2", x=.04, ha="left", fontsize=22, color="#173749")
        fig.subplots_adjust(left=.02, right=.98, top=.92, bottom=.06, wspace=.08, hspace=.20)
        fig.savefig(out / f"atlas_{page}.png", dpi=160)
        fig.savefig(out / f"atlas_{page}.svg")
        plt.close(fig)
    for c in cases:
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(projection="3d")
        decorate_3d(ax, c)
        fig.text(.07, .075, c.strategy+"\n"+c.caveat, fontsize=11, color="#354957")
        fig.savefig(out / f"{c.key}.png", dpi=150)
        plt.close(fig)
    data = []
    for c in cases:
        rounded = lambda x: np.round(x, 3).tolist()
        data.append({"id": c.key, "title": c.title, "topology": c.topology,
                     "strategy": c.strategy, "caveat": c.caveat,
                     "grid": [rounded(p[::2]) for p in c.grids],
                     "sources": [rounded(p[::3]) for p in c.sources],
                     "pieces": [{"kind": p.kind, "p": rounded(p.xyz[::2] if len(p.xyz)%2 else np.vstack((p.xyz[::2], p.xyz[-1])))} for p in c.pieces]})
    (ROOT / "atlas_data.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    (ROOT / "validation.json").write_text(json.dumps(validate(cases), indent=2), encoding="utf-8")
    print(f"Created {len(cases)} cases; figures, geometry and continuity checks in {ROOT}")


if __name__ == "__main__":
    export(make_cases())
