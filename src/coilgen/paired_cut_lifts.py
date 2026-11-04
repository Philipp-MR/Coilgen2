"""Give opposing legs of a physically crossed cut consistent clearance over the turns."""

from dataclasses import replace

import numpy as np
from scipy.spatial import cKDTree

from .models import WirePath, WirePathGeometry, WireSegmentOwner


def _crossed_cuts(points, owners, span):
    """Require an interior crossing between the two source legs, away from their attachments."""
    from .geometry_kernels import segment_closest_parameters

    cuts = {}
    for i, owner in enumerate(owners):
        if owner.kind == "inter_group_connector" and owner.connector_role in {"primary", "closing"}:
            cuts.setdefault(owner.connector_index, {}).setdefault(owner.connector_role, []).append(i)

    def interior(indices):
        # Trim the attachments before testing the intersection. For two
        # overlapping collinear legs, the closest point is not unique and a
        # segment-distance solver is otherwise free to return an attachment.
        idx = np.asarray(indices)
        v = points[idx + 1] - points[idx]
        length = np.linalg.norm(v, axis=1)
        arc = np.r_[0.0, length.cumsum()]
        left = np.maximum(arc[:-1], span)
        right = np.minimum(arc[1:], arc[-1] - span)
        keep = right > left
        step = v[keep] / length[keep, None]
        return points[idx[keep]] + (left[keep] - arc[:-1][keep])[:, None] * step, (right[keep] - left[keep])[
            :, None
        ] * step

    for cut, roles in cuts.items():
        if len(roles) != 2:
            continue
        (ab, va), (bb, vb) = (interior(roles[r]) for r in ("primary", "closing"))
        for i in range(len(ab)):
            # Only compare segments with intersecting source boxes. Long,
            # already-sampled cuts need not materialize a dense pair matrix.
            lo, hi = np.minimum(ab[i], ab[i] + va[i]), np.maximum(ab[i], ab[i] + va[i])
            tol = 1e-8
            j = np.flatnonzero(
                np.all(np.minimum(bb, bb + vb) <= hi + tol, axis=1)
                & np.all(np.maximum(bb, bb + vb) >= lo - tol, axis=1)
            )
            if not len(j):
                continue
            s, t = segment_closest_parameters(
                np.broadcast_to(ab[i], (len(j), 3)), np.broadcast_to(va[i], (len(j), 3)), bb[j], vb[j]
            )
            gap = np.linalg.norm(ab[i] + s[:, None] * va[i] - bb[j] - t[:, None] * vb[j], axis=1)
            if np.any(gap <= tol):
                yield cut
                break


def paired_cut_proposals(part, config):
    """Yield detached source routes. Each must still pass cleanup, resweep and export checks."""
    from .crossing_lifts import _active_intervals, _cosine_plateaus, crossing_lift_candidate, apply_lift_proposal
    from .crossing_relaxation import _CutMotion, _GapPreview
    from .group_route_validation import _isolated_part
    from .layout import cleanup_wire_paths
    from .manufacturing import create_swept_coil_meshes
    from .swept_clearance import required_copper_gap

    cap = abs(config.normal_shift_length)
    profile = max(config.conductor_cross_section_width, config.conductor_cross_section_height)
    gap = required_copper_gap(config)
    if part.wire_path is None or profile <= 0 or cap < 2 * (profile + gap):
        return
    prov = part.normal_shift_provenance
    owners = (
        [WireSegmentOwner(**row) for row in prov["segment_owners"]]
        if prov.get("segment_owners")
        else part.wire_path.segment_owners
    )
    cuts = list(_crossed_cuts(prov.get("points_before_m", part.wire_path.v), owners, 2 * profile))
    if not cuts:
        return
    seed_cfg = replace(config, normal_shift_length=np.copysign(min(cap, 2 * profile), config.normal_shift_length))
    seed = crossing_lift_candidate(part, seed_cfg, ramp_scale=0.7, clearance_factor=1.5, separate_joins=True)
    if seed is None:
        return
    trial = _isolated_part(part)
    if not apply_lift_proposal(trial, config, seed):
        return
    cleanup_wire_paths([trial], config=config)
    create_swept_coil_meshes([trial], replace(config, sweep_surface_realization="mesh"))
    if len(trial.swept_bodies) != 1:
        return
    m = _CutMotion(trial, config)
    body = trial.swept_bodies[0]
    n = len(body.fabrication_path.cross_section_m)
    if not (3 <= n <= 8):
        return
    preview = _GapPreview(body, m, m.points)
    raw, source = seed
    profile = float(np.ptp(body.fabrication_path.cross_section_m, axis=0).max())
    if cap < 2 * (profile + gap):
        return
    sign = np.sign(config.normal_shift_length)
    height = source["lift_distance_m"]
    lateral = source["lateral_displacement_m"]
    ordinary = np.r_[
        [o.kind in {"contour_loop", "contour_group"} for o in m.owners],
        m.owners[-1].kind in {"contour_loop", "contour_group"},
    ]
    peaks = np.unique(np.minimum(cap, 2 * (profile + gap) + profile * np.array([0.15, 0.5, 1.0])))
    for cut in cuts:
        pair = [
            np.array(
                [
                    o.kind == "inter_group_connector" and o.connector_index == cut and o.connector_role == role
                    for o in m.owners
                ]
            )
            for role in ("primary", "closing")
        ]
        mask = pair[0] | pair[1]
        ends = np.r_[mask, False] | np.r_[False, mask]
        dist = cKDTree(m.origin[ends]).query(m.origin)[0]
        locality = 0.5 + 0.5 * np.cos(np.pi * np.clip((dist - 1.5 * profile) / (6 * profile), 0.0, 1.0))
        best = []
        for ramp in (3 * profile, 5 * profile):
            envelope = [
                _cosine_plateaus(m.arc, *_active_intervals(active, m.arc, ramp, m.closed), ramp) for active in pair
            ]
            for high in (0, 1):
                for peak in peaks:
                    # Remove incidental turn lifts near the crossed pair. Try
                    # restoring only that cut region and the full source footprint.
                    h = height * (1 - ordinary * locality)
                    for which, value in ((high, peak), (1 - high, (peak - 0.4 * gap) / 2)):
                        h = h * (1 - envelope[which]) + sign * value * envelope[which]
                    for footprint, restore in enumerate((locality, 1.0)):
                        removed = lateral * np.asarray(restore).reshape(-1, 1)
                        change = (h - height)[:, None] * m.normals - removed
                        if not m.closed:
                            change[[0, -1]] = 0.0
                        rank = preview.merit(change)
                        best.append((rank, peak, high, ramp, footprint, change))
        # Re-tessellation can change which near-threshold variant clears.
        # Preserve both the local and complete source-footprint restorations.
        seen = set()
        for _, peak, high, ramp, footprint, change in sorted(best, key=lambda item: item[0]):
            if (peak, footprint) in seen:
                continue
            seen.add((peak, footprint))
            coords = m.radial_limit(m.points + change, np.arange(len(m.points)))
            mounted = m.mount(coords)
            if mounted is None:
                continue
            coords, h, lat, normals = mounted
            positions = source["source_positions"]
            mid = np.floor(0.5 * (positions[:-1] + positions[1:])).astype(int)
            geometry = [
                WirePathGeometry.OFFSET_SURFACE
                if source["source_geometry"][i] != WirePathGeometry.FREE_3D and max(abs(h[k : k + 2])) > 1e-12
                else source["source_geometry"][i]
                for k, i in enumerate(mid)
            ]
            from .geometry_kernels import segment_closest_parameters

            contacts = np.asarray(source["contacts_before"])
            aa, bb = contacts[:, :2].astype(int).T
            foot = source["routing_footprint_m"]
            vec = np.diff(foot, axis=0)
            s, t = segment_closest_parameters(foot[aa], vec[aa], foot[bb], vec[bb])
            delta = sign * (np.interp(aa + s, positions, h) - np.interp(bb + t, positions, h))
            selected = [
                (int(a), int(b), float(d)) if dz > 0 else (int(b), int(a), float(d))
                for a, b, d, dz in zip(aa, bb, contacts[:, 2], delta)
                if abs(dz) > 1e-10
            ]
            proposal = (
                WirePath(coords, list(raw.segment_owners), geometry),
                {
                    **source,
                    "lift_distance_m": h,
                    "output_normals": normals,
                    "lateral_displacement_m": lat,
                    "shoulder_length_m": ramp,
                    "selected_contacts": selected,
                },
            )
            candidate = _isolated_part(part)
            if apply_lift_proposal(candidate, config, proposal):
                candidate.normal_shift_provenance["paired_cut_levels"] = {
                    "connector_index": cut,
                    "upper_role": ("primary", "closing")[high],
                    "peak_m": float(sign * peak),
                    "lower_m": float(sign * (peak - 0.4 * gap) / 2),
                }
                yield candidate
