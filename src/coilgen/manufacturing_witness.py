"""Reconstruct physical witnesses from verified saved conductor frames."""

import numpy as np


def _finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _frame_witness(frames, diagnostics, kind):
    if frames is None or diagnostics.get("sweep_ring_count") != len(frames["points_m"]):
        return (), (), "No matching verified sweep-frame coordinates were saved."
    from .manufacturing import _polygon_centroid

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
    elif kind == "copper":
        pairs = diagnostics.get("copper_clearance_witness_intervals", ())
        candidates = [
            p
            for p in pairs
            if isinstance(p, (tuple, list))
            and len(p) == 3
            and interval(p[0]) is not None
            and interval(p[1]) is not None
            and _finite_number(p[2]) is not None
        ]
        if not candidates:
            return (), (), "No matching facet intervals were saved."
        i, j, _ = min(candidates, key=lambda p: p[2])
        selected = points[[*interval(i), *interval(j)]]
        segments = ((0, 1), (2, 3))
        description = "Mounted winding intervals containing the measured copper-facet witness."
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
