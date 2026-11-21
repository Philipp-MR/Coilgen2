"""Shared winding display colors and directed, arc-length-spaced markers."""

import numpy as np

GROUP_COLORS = ("#2478c5", "#d97724", "#7b55b5", "#159884", "#cc5278", "#80972c", "#ba643f", "#527e93")


def group_color(index):
    return GROUP_COLORS[index % len(GROUP_COLORS)] if index >= 0 else "#66798c"


def direction_markers(points, spacing):
    """Return points and unit tangents in the stored electrical path direction."""
    points = np.asarray(points)
    delta = np.diff(points, axis=0)
    length = np.linalg.norm(delta, axis=1)
    station = np.r_[0.0, np.cumsum(length)]
    if station[-1] < spacing * 0.35:
        return []
    count = min(12, max(1, int(station[-1] / spacing)))
    result = []
    for s in (np.arange(count) + 0.5) * station[-1] / count:
        i = min(int(np.searchsorted(station, s, side="right") - 1), len(delta) - 1)
        if length[i] > 1e-15:
            result.append((points[i] + (s - station[i]) * delta[i] / length[i], delta[i] / length[i]))
    return result
