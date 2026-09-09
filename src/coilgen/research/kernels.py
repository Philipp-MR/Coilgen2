"""Compatibility facade for the shared core geometry kernels."""

from ..geometry_kernels import (
    NUMBA_AVAILABLE,
    DistanceResult,
    active_geometry_backend,
    geometry_acceleration,
    segment_pair_distances,
    triangle_pair_distances,
)

__all__ = [
    "NUMBA_AVAILABLE",
    "DistanceResult",
    "active_geometry_backend",
    "geometry_acceleration",
    "segment_pair_distances",
    "triangle_pair_distances",
]
