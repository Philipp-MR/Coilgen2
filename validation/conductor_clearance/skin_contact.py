"""All-station skin collision audit, preserving only legitimate incident vertex/edge joins."""

from __future__ import annotations

import numpy as np

from surface_gap import CONTACT_EPS, closest_features, dot, leaves


def _cuts(t, h, direction, plane_angle_sine):
    points = np.zeros((len(t), 6, 3))
    active = np.zeros((len(t), 6), dtype=bool)
    for i in range(3):
        j = (i + 1) % 3
        den = h[:, i] - h[:, j]
        s = np.divide(h[:, i], den, out=np.zeros(len(t)), where=den != 0)
        active[:, i] = (den != 0) & (s >= 0) & (s <= 1)
        points[:, i] = t[:, i] + s[:, None] * (t[:, j] - t[:, i])
        # A point displaced in the facet plane is only h/sin(theta) away
        # from the plane-intersection line. Using h alone admits points far
        # from that line when two nearby loft facets are almost parallel.
        active[:, i + 3] = np.abs(h[:, i]) <= CONTACT_EPS * plane_angle_sine
        points[:, i + 3] = t[:, i]
    pos = np.einsum("ijk,ik->ij", points, direction)
    low = np.where(active, pos, np.inf).min(axis=1)
    high = np.where(active, pos, -np.inf).max(axis=1)
    anchor = points[np.arange(len(t)), np.argmax(active, axis=1)]
    return low, high, anchor


def classify(a, b):
    """Closed-triangle contact and relative-interior penetration, in float64 metres."""
    origin = a[:, 0].copy()
    a, b = a - origin[:, None], b - origin[:, None]
    ea = np.roll(a, -1, axis=1) - a
    eb = np.roll(b, -1, axis=1) - b
    na = np.cross(ea[:, 0], ea[:, 1])
    nb = np.cross(eb[:, 0], eb[:, 1])
    na /= np.linalg.norm(na, axis=1)[:, None]
    nb /= np.linalg.norm(nb, axis=1)[:, None]
    ha = np.einsum("ijk,ik->ij", a - b[:, :1], nb)
    hb = np.einsum("ijk,ik->ij", b - a[:, :1], na)
    matches = np.all(a[:, :, None, :] == b[:, None, :, :], axis=3)
    ha[matches.any(axis=2)] = 0.0
    hb[matches.any(axis=1)] = 0.0
    candidate = (
        (ha.min(axis=1) <= CONTACT_EPS)
        & (ha.max(axis=1) >= -CONTACT_EPS)
        & (hb.min(axis=1) <= CONTACT_EPS)
        & (hb.max(axis=1) >= -CONTACT_EPS)
    )
    coplanar = candidate & (np.abs(ha).max(axis=1) <= CONTACT_EPS) & (np.abs(hb).max(axis=1) <= CONTACT_EPS)
    hit = np.zeros(len(a), dtype=bool)
    penetrates = hit.copy()
    patch = hit.copy()
    extended = hit.copy()
    point = np.full((len(a), 3), np.nan)
    line = np.cross(na, nb)
    norm = np.linalg.norm(line, axis=1)
    ix = np.flatnonzero(candidate & ~coplanar & (norm > 32 * np.finfo(float).eps))
    if len(ix):
        direction = line[ix] / norm[ix, None]
        x0, x1, anchor = _cuts(a[ix], ha[ix], direction, norm[ix])
        y0, y1, _ = _cuts(b[ix], hb[ix], direction, norm[ix])
        lo = np.maximum(x0, y0)
        hi = np.minimum(x1, y1)
        valid = np.isfinite(lo) & np.isfinite(hi)
        hit[ix] = valid & (hi >= lo - CONTACT_EPS)
        inner = (
            (ha[ix].min(axis=1) < -CONTACT_EPS)
            & (ha[ix].max(axis=1) > CONTACT_EPS)
            & (hb[ix].min(axis=1) < -CONTACT_EPS)
            & (hb[ix].max(axis=1) > CONTACT_EPS)
        )
        penetrates[ix] = valid & inner & (hi > lo + CONTACT_EPS)
        extended[ix] = valid & (hi > lo + CONTACT_EPS)
        mask = hit[ix]
        pos = (lo[mask] + hi[mask]) / 2
        point[ix[mask]] = (
            origin[ix[mask]] + anchor[mask] + (pos - dot(anchor[mask], direction[mask]))[:, None] * direction[mask]
        )
    # For nearly parallel planes the intersection line is ill-conditioned.
    # Measure the actual finite triangles before assigning a boundary contact.
    ix = np.flatnonzero(candidate & ~coplanar & (norm < 1e-5) & ~matches.any(axis=(1, 2)))
    if len(ix):
        d, pa, pb = closest_features(a[ix], b[ix])
        hit[ix] = d <= CONTACT_EPS
        penetrates[ix] &= hit[ix]
        extended[ix] &= hit[ix]
        point[ix] = np.where(hit[ix, None], (pa + pb) / 2 + origin[ix], np.nan)
    ix = np.flatnonzero(coplanar)
    if len(ix):
        overlap = np.full(len(ix), np.inf)
        # An incidental single shared corner may not share a longer collinear
        # edge. For a coplanar line overlap, at least one unshared endpoint
        # of the shorter limiting edge lies on the other triangle.
        inside_a = ~matches[ix].any(axis=2)
        inside_b = ~matches[ix].any(axis=1)
        for edges, normal in ((ea, na), (eb, nb)):
            for i in range(3):
                axis = np.cross(normal[ix], edges[ix, i])
                axis /= np.linalg.norm(axis, axis=1)[:, None]
                av = np.einsum("ijk,ik->ij", a[ix], axis)
                bv = np.einsum("ijk,ik->ij", b[ix], axis)
                if edges is ea:
                    inside_b &= bv >= av[:, i, None] - CONTACT_EPS
                else:
                    inside_a &= av >= bv[:, i, None] - CONTACT_EPS
                overlap = np.minimum(
                    overlap, np.minimum(av.max(axis=1), bv.max(axis=1)) - np.maximum(av.min(axis=1), bv.min(axis=1))
                )
        hit[ix] = overlap >= -CONTACT_EPS
        patch[ix] = overlap > CONTACT_EPS
        extended[ix] = hit[ix] & (inside_a.any(axis=1) | inside_b.any(axis=1) | patch[ix])
    return hit, penetrates, patch, point, extended


def inspect_skin(body):
    result = dict(
        body=body.name,
        all_nonincident_contact_pairs=0,
        strict_piercing_pairs=0,
        incident_strict_piercing_pairs=0,
        coplanar_area_overlap_pairs=0,
        incident_coplanar_area_overlap_pairs=0,
        incident_excess_line_contact_pairs=0,
        examples={},
    )
    for ia, ib in leaves(body, body, CONTACT_EPS, scope="all"):
        a, b = body.triangles[ia], body.triangles[ib]
        hits, pierces, patch, point, extended = classify(a, b)
        if not hits.any():
            continue
        delta = np.abs(body.station[ia] - body.station[ib])
        neighbor = (delta <= 1) | (body.closed & (delta == len(body.owners) - 1))
        common = np.sum(np.all(a[:, :, None, :] == b[:, None, :, :], axis=3), axis=(1, 2))
        incident = neighbor & (common > 0)
        for key, mask in (
            ("all_nonincident_contact_pairs", hits & ~incident),
            ("strict_piercing_pairs", pierces),
            ("incident_strict_piercing_pairs", pierces & incident),
            ("coplanar_area_overlap_pairs", patch),
            ("incident_coplanar_area_overlap_pairs", patch & incident),
            ("incident_excess_line_contact_pairs", extended & neighbor & (common == 1) & ~(pierces | patch)),
        ):
            result[key] += int(mask.sum())
            if mask.any() and key not in result["examples"]:
                k = np.flatnonzero(mask)[0]
                row = dict(
                    faces=[int(ia[k]), int(ib[k])],
                    stations=body.station[[ia[k], ib[k]]].tolist(),
                    owners=body.owners[body.station[[ia[k], ib[k]]]].tolist(),
                )
                if np.all(np.isfinite(point[k])):
                    row["world_xyz_m"] = point[k].tolist()
                else:
                    # The vertex and edge witnesses of a coplanar overlap can lie on
                    # its boundary. Retain both full facets for exact reproducibility.
                    row["triangles_m"] = [a[k].tolist(), b[k].tolist()]
                result["examples"][key] = row
    result["failed"] = bool(
        result["all_nonincident_contact_pairs"]
        or result["strict_piercing_pairs"]
        or result["coplanar_area_overlap_pairs"]
        or result["incident_excess_line_contact_pairs"]
    )
    return result
