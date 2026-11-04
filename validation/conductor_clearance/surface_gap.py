"""Independent Euclidean closest points on the actual exported conductor facets (SI)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.spatial import cKDTree


CONTACT_EPS = 1e-12
ACCEPTANCE_EPS = 1e-10


def dot(a, b):
    return np.einsum("ij,ij->i", a, b)


def closest_features(a, b):
    """Return distances and a pair of points. The complete feature set includes piercing edges."""
    n = len(a)
    best = np.full(n, np.inf)
    pa = np.zeros((n, 3))
    pb = pa.copy()

    def update(x, y, eligible):
        sq = dot(x - y, x - y)
        take = (sq < best) & eligible
        best[take], pa[take], pb[take] = sq[take], x[take], y[take]

    for swapped, (source, target) in enumerate(((a, b), (b, a))):
        normal = np.cross(target[:, 1] - target[:, 0], target[:, 2] - target[:, 0])
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        heights = np.einsum("ijk,ik->ij", source - target[:, :1], normal)
        edges = np.roll(target, -1, axis=1) - target

        def within(point):
            side = np.einsum("ijk,ik->ij", np.cross(edges, point[:, None] - target), normal)
            return np.all(side >= 0, axis=1)

        for i in range(3):
            x = source[:, i]
            y = x - heights[:, i, None] * normal
            update(y, x, within(y)) if swapped else update(x, y, within(y))
            j = (i + 1) % 3
            denominator = heights[:, i] - heights[:, j]
            t = np.divide(heights[:, i], denominator, out=np.zeros(n), where=denominator != 0)
            y = x + t[:, None] * (source[:, j] - x)
            update(y, y, (denominator != 0) & (t >= 0) & (t <= 1) & within(y))

    for i in range(3):
        p = a[:, i]
        u = a[:, (i + 1) % 3] - p
        uu = dot(u, u)
        for j in range(3):
            q = b[:, j]
            v = b[:, (j + 1) % 3] - q
            w = q - p
            vv = dot(v, v)
            normal = np.cross(u, v)
            den = dot(normal, normal)
            s = np.divide(dot(np.cross(w, v), normal), den, out=np.zeros(n), where=den != 0)
            t = np.divide(dot(np.cross(w, u), normal), den, out=np.zeros(n), where=den != 0)
            update(p + s[:, None] * u, q + t[:, None] * v, (den != 0) & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1))
            # Endpoint projections onto these directed edges. Across the nine directed
            # edge pairs this covers every vertex-to-edge and vertex-to-vertex minimum.
            t = np.clip(dot(-w, v) / vv, 0, 1)
            update(p, q + t[:, None] * v, True)
            s = np.clip(dot(w, u) / uu, 0, 1)
            update(p + s[:, None] * u, q, True)
    return np.sqrt(best), pa, pb


def stl_triangles(path):
    raw = Path(path).read_bytes()
    n = int.from_bytes(raw[80:84], "little")
    if len(raw) == 84 + 50 * n:
        record = np.dtype([("normal", "<f4", 3), ("points", "<f4", (3, 3)), ("attribute", "<u2")])
        tri = np.frombuffer(raw, record, count=n, offset=84)["points"].astype(float)
    else:
        rows = re.findall(rb"(?:^|\n)\s*vertex\s+([^\r\n]+)", raw)
        tri = np.fromstring(b" ".join(rows), sep=" ").reshape(-1, 3, 3)
    if not len(tri) or not np.isfinite(tri).all():
        raise ValueError(f"Invalid exported triangles in {path}")
    return tri


@dataclass
class Body:
    name: str
    triangles: np.ndarray
    profile: np.ndarray
    points: np.ndarray
    owners: np.ndarray
    closed: bool
    insulation: float
    tolerance: float
    margin: float

    def __post_init__(self):
        self.triangles = np.asarray(self.triangles, dtype=float)
        t = self.triangles
        if (
            not len(t)
            or not np.all(np.isfinite(t))
            or np.any(np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1) == 0)
        ):
            raise ValueError(f"{self.name} has empty, invalid or degenerate triangles")
        self.arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))]
        tangent = np.diff(self.points, axis=0) / np.diff(self.arc)[:, None]
        turns = np.arccos(np.clip(dot(tangent[:-1], tangent[1:]), -1.0, 1.0))
        self.rotation = np.r_[0.0, np.cumsum(turns)]
        self.total_rotation = self.rotation[-1] + (
            np.arccos(np.clip(tangent[-1] @ tangent[0], -1.0, 1.0)) if self.closed else 0.0
        )
        self.diameter = float(np.max(np.linalg.norm(self.profile[:, None] - self.profile[None, :], axis=2)))
        self.station = np.arange(len(t)) // (2 * len(self.profile))
        caps = self.station >= len(self.owners)
        self.station[caps] = np.where(np.flatnonzero(caps) % 2, len(self.owners) - 1, 0)
        if len(self.owners) != len(self.points) - 1 or len(self.triangles) < 2 * len(self.owners) * len(self.profile):
            raise ValueError(f"{self.name} STL and fabrication ring counts disagree")
        self.vertices = t.reshape(-1, 3)
        self.vertex_tree = None
        self.levels = [(t.min(axis=1), t.max(axis=1))]
        while len(self.levels[-1][0]) > 1:
            low, high = self.levels[-1]
            next_low, next_high = low[::2].copy(), high[::2].copy()
            half = len(low) // 2
            next_low[:half] = np.minimum(next_low[:half], low[1::2])
            next_high[:half] = np.maximum(next_high[:half], high[1::2])
            self.levels.append((next_low, next_high))

    def separation(self, ia, ib):
        """Length of copper path strictly between the two owning station intervals."""
        i, j = self.station[ia], self.station[ib]
        left, right = np.minimum(i, j), np.maximum(i, j)
        gap = np.maximum(0, self.arc[right] - self.arc[left + 1])
        if self.closed:
            gap = np.minimum(gap, self.arc[-1] - self.arc[right + 1] + self.arc[left])
        return gap

    def feature(self, i, point):
        return dict(
            body=self.name,
            face=int(i),
            station=int(self.station[i]),
            owner=str(self.owners[self.station[i]]),
            world_xyz_m=point.tolist(),
        )

    def short_folds(self, ia, ib):
        """Nonadjacent station pairs on short returning arcs. Sharp local knees are reported separately."""
        i, j = np.minimum(self.station[ia], self.station[ib]), np.maximum(self.station[ia], self.station[ib])
        arc = self.arc[j] - self.arc[i + 1]
        theta = self.rotation[j] - self.rotation[i]
        apart = (j - i) > 1
        if self.closed:
            wrap = self.arc[-1] - self.arc[j + 1] + self.arc[i]
            # In an arclength tie, either intervening arc may fold.
            theta = np.where(
                wrap < arc,
                self.total_rotation - theta,
                np.where(wrap == arc, np.maximum(theta, self.total_rotation - theta), theta),
            )
            arc = np.minimum(arc, wrap)
            apart &= (j - i) < len(self.owners) - 1
        return apart & (arc <= 4 * self.diameter) & (theta >= np.pi / 2 - 1e-10)

    def vertex_upper(self, other):
        if self.vertex_tree is None:
            self.vertex_tree = cKDTree(self.vertices)
        # A subsample suffices for a conservative upper bound. The hierarchy
        # still searches every possibly closer face, including unsampled facets.
        return float(self.vertex_tree.query(other.vertices[:: max(len(other.vertices) // 8000, 1)])[0].min())


def leaves(first, second, limit, scope="remote"):
    """Ordered hierarchical AABB pruning. No fixed nearest-neighbor budget or sampling."""
    same = first is second

    def walk(la, lb, ia, ib):
        a, b = first.levels[la], second.levels[lb]
        delta = np.maximum(0, np.maximum(a[0][ia] - b[1][ib], b[0][ib] - a[1][ia]))
        ok = dot(delta, delta) <= limit**2
        if same and la == lb:
            ok &= ia <= ib
        ia, ib = ia[ok], ib[ok]
        if la == lb == 0:
            if same:
                ok = ia < ib
                if scope == "remote":
                    ok &= first.separation(ia, ib) > 4 * first.diameter
                elif scope == "short_fold":
                    ok &= first.short_folds(ia, ib)
                elif scope != "all":
                    raise ValueError(f"Unknown station scope {scope}")
                ia, ib = ia[ok], ib[ok]
            if len(ia):
                yield ia, ib
            return
        if len(ia) == 0:
            return
        na = max(0, la - 1)
        nb = max(0, lb - 1)
        for start in range(0, len(ia), 4096):
            x, y = ia[start : start + 4096], ib[start : start + 4096]
            xs = np.column_stack((2 * x, 2 * x + 1)) if la else x[:, None]
            ys = np.column_stack((2 * y, 2 * y + 1)) if lb else y[:, None]
            xa = np.repeat(xs, ys.shape[1], axis=1).ravel()
            yb = np.tile(ys, (1, xs.shape[1])).ravel()
            good = (xa < len(first.levels[na][0])) & (yb < len(second.levels[nb][0]))
            yield from walk(na, nb, xa[good], yb[good])

    yield from walk(len(first.levels) - 1, len(second.levels) - 1, np.array([0]), np.array([0]))


def contains_point(body, point):
    """Solid angle of a closed faceted copper skin, independent of coordinate axes."""
    vectors = body.triangles - point
    lengths = np.linalg.norm(vectors, axis=2)
    numerator = dot(vectors[:, 0], np.cross(vectors[:, 1], vectors[:, 2]))
    denominator = np.prod(lengths, axis=1)
    for i in range(3):
        denominator += dot(vectors[:, i], vectors[:, (i + 1) % 3]) * lengths[:, (i + 2) % 3]
    return bool(abs(np.arctan2(numerator, denominator).sum() / (2 * np.pi)) > 0.5)


def compare(first, second, scope="remote"):
    """Minimum copper gap, complete near-pair counts, and witnesses for this pair of bodies."""
    same = first is second
    required = (
        first.insulation + second.insulation + max(first.tolerance, second.tolerance) + max(first.margin, second.margin)
    )
    radius = max(required, 1e-6) if same else max(required, first.vertex_upper(second) + ACCEPTANCE_EPS)
    while True:
        witnesses = {}
        counts = dict(contact=0, positive_subminimum=0)
        near_equality = 0
        count = 0
        for ia, ib in leaves(first, second, radius + CONTACT_EPS, scope):
            d, pa, pb = closest_features(first.triangles[ia], second.triangles[ib])
            valid = d <= radius + CONTACT_EPS
            contact = valid & (d <= CONTACT_EPS)
            below = valid & (d > CONTACT_EPS) & (d < required - ACCEPTANCE_EPS)
            near_equality += int((valid & (d > CONTACT_EPS) & (abs(d - required) <= ACCEPTANCE_EPS)).sum())
            counts["contact"] += int(contact.sum())
            counts["positive_subminimum"] += int(below.sum())
            count += len(ia)
            for label, mask in (("minimum", valid), ("positive_violation", below)):
                indices = np.flatnonzero(mask)
                if not len(indices):
                    continue
                k = indices[np.argmin(d[indices])]
                if label in witnesses and witnesses[label]["distance_m"] <= d[k]:
                    continue
                witnesses[label] = dict(
                    distance_m=float(d[k]), first=first.feature(ia[k], pa[k]), second=second.feature(ib[k], pb[k])
                )
                if same:
                    witnesses[label]["intervening_arclength_m"] = float(first.separation(ia[k], ib[k]))
        if "minimum" in witnesses:
            break
        if scope == "short_fold":
            return dict(
                first=first.name,
                second=second.name,
                status="no_short_fold_gap_within_threshold",
                required_m=required,
                scope=scope,
                failed=False,
                facet_candidates=count,
                facet_pair_counts=counts,
                witnesses={},
            )
        if not same:
            raise RuntimeError("Vertex upper bound failed to enclose a candidate")
        radius *= 2
        if radius > 2 * np.linalg.norm(first.levels[-1][1] - first.levels[-1][0]):
            return dict(first=first.name, second=second.name, status="no_nonlocal_intervals", required_m=required)
    containment = False
    if not same and not counts["contact"]:
        lo1, hi1 = (x[0] for x in first.levels[-1])
        lo2, hi2 = (x[0] for x in second.levels[-1])
        if np.all(hi1 >= lo2) and np.all(hi2 >= lo1):
            containment = contains_point(first, second.vertices[0]) or contains_point(second, first.vertices[0])
    return dict(
        first=first.name,
        second=second.name,
        same_body=same,
        required_m=required,
        scope=scope,
        nonlocal_rule_m=(4 * first.diameter if same else None),
        distance_acceptance_epsilon_m=ACCEPTANCE_EPS,
        contact_tolerance_m=CONTACT_EPS,
        failed=bool(containment or counts["contact"] or counts["positive_subminimum"]),
        contained=containment,
        facet_candidates=count,
        threshold_equality_pairs=near_equality,
        facet_pair_counts=counts,
        witnesses=witnesses,
    )
