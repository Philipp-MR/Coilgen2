"""Canonical saved-field source names shared by plot adapters."""


def shim_source_key(source: str) -> str:
    key = str(source).strip().lower().replace("-", "_")
    if key in {
        "before",
        "pre",
        "pre_shim",
        "pre_shimming",
        "unshimmed",
        "fieldmap",
        "inhomogeneity",
        "before_inhomogeneity",
    }:
        return "before_inhomogeneity"
    if key in {
        "after",
        "post",
        "post_shim",
        "post_shimming",
        "remaining",
        "remaining_inhomogeneity",
        "layout",
        "connected",
        "connected_wire",
        "wire",
        "wire_layout",
        "coil",
        "coilfield",
        "layout_remaining",
        "layout_remaining_inhomogeneity",
    }:
        return "layout_remaining_inhomogeneity"
    if key in {"after_stream", "stream_remaining", "stream_remaining_inhomogeneity"}:
        return "stream_remaining_inhomogeneity"
    if key in {"layout_residual", "target_minus_layout"}:
        return "layout_residual"
    if key in {"stream", "stream_function", "sf"}:
        return "stream_remaining_inhomogeneity"
    if key in {"stream_residual", "target_minus_stream"}:
        return "stream_residual"
    if key in {
        "improvement",
        "delta",
        "delta_improvement",
        "before_minus_after",
        "layout_improvement",
        "wire_improvement",
        "connected_improvement",
        "layout_improvement_before_minus_after",
    }:
        return "layout_improvement_before_minus_after"
    if key in {"stream_improvement", "before_minus_after_stream", "stream_improvement_before_minus_after"}:
        return "stream_improvement_before_minus_after"
    raise ValueError("source must be 'before', 'after', 'layout'/'coil', 'stream', or 'improvement'.")
