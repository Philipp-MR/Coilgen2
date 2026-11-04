"""Matplotlib view shared by the desktop and browser workbenches (no Qt dependency)."""

import numpy as np


SHIELDING_COMPONENTS = {"magnitude": "|B|", "Bx": "Bx", "By": "By", "Bz": "Bz", "normal": "B · n"}
SHIELDING_SOURCES = {
    "sheet": "Stream function model",
    "contour": "Unconnected contours",
    "routed": "Routed winding",
    "fabricated": "Complete electrical paths",
    "sheet_interior": "Stream function field · triangle-interior check",
    "routed_interior": "Routed winding · triangle-interior check",
}


def draw_shielding_surface(figure, fields, component="magnitude", *, title=None, net_detail=False):
    """Compare the actual vector contributions on the same physical samples and scale."""
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MaxNLocator, NullFormatter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    component_label = SHIELDING_COMPONENTS[component]
    has_parts = fields.primary_T is not None and fields.shield_T is not None
    vectors = [("Combined field", fields.total_T)]
    if has_parts:
        vectors[:0] = [("Primary alone", fields.primary_T), ("Shield alone", fields.shield_T)]
    nonzero_target = fields.target_T is not None and np.any(fields.target_T != 0)
    if nonzero_target:
        vectors.append(("Combined − requested", fields.total_T - fields.target_T))
    values = []
    for _, field in vectors:
        if component == "magnitude":
            scalar = np.linalg.norm(field, axis=1)
        elif component == "normal":
            if fields.normals is None:
                raise ValueError("Surface normals are unavailable for this run.")
            scalar = np.einsum("ij,ij->i", field, fields.normals)
        else:
            scalar = field[:, ("Bx", "By", "Bz").index(component)]
        values.append(scalar * 1e6)

    peaks = [float(np.max(np.abs(scalar))) for scalar in values]
    rms = [float(np.sqrt(np.average(scalar**2, weights=fields.area_weights_m2))) for scalar in values]
    total = 2 if has_parts else 0
    net_readout = f"Net RMS {rms[total]:.3g} µT · peak {peaks[total]:.3g} µT"
    weighting = "area-weighted" if fields.area_weights_m2 is not None else "equal-sample"
    interior = fields.sampling == "triangle_interior"
    sample_label = "triangle-interior samples" if interior else "suppression samples"
    summary = f"{len(fields.points_m):,} {sample_label} · {weighting} RMS"
    if has_parts:
        ratio = f"{100 * rms[2] / rms[0]:.3g}%" if rms[0] > 0 else "undefined (zero primary RMS)"
        summary += f" · Combined / primary RMS = {ratio}"
        if rms[0] > 0:
            db = f"{20 * (np.log10(rms[0]) - np.log10(rms[2])):.3g}" if rms[2] > 0 else "∞"
            summary += f" · Attenuation {db} dB"
    else:
        summary += " · Separate primary and shield contributions unavailable"
    if net_detail:
        vectors, values, peaks, rms = [vectors[total]], [values[total]], [peaks[total]], [rms[total]]
        vectors[0] = ("Combined field · detail scale", fields.total_T)

    points = fields.points_m * 1e3
    if net_detail:
        peak_point = points[int(np.argmax(np.abs(values[0])))]
        coordinates = ", ".join(f"{value + 0.0:.3g}" for value in np.round(peak_point, 4))
        label = "Peak vertex" if fields.faces is not None else "Peak sample"
        summary += f" · {label} at ({coordinates}) mm (star)"
    # Bound rendering only. The common scale, peak and RMS use every saved sample.
    triangulated = fields.faces is not None and len(fields.faces) <= 20_000
    selection = np.arange(len(points))
    if not triangulated and len(points) > 10_000:
        selection = np.unique(
            np.r_[np.linspace(0, len(points) - 1, 10_000, dtype=int), [np.argmax(np.abs(v)) for v in values]]
        )
        summary += f" · {len(selection):,} display points"
    span = np.ptp(points, axis=0)
    box = np.maximum(span, max(float(span.max()), 1.0) * 0.05)
    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    maximum = max(peaks) or 1.0
    signed = component != "magnitude"
    norm = Normalize(-maximum if signed else 0, maximum)
    cmap = colormaps["coolwarm" if signed else "viridis"]
    figure.clear()
    excitation = fields.current_source.replace("_", " ")
    if fields.current_A is not None:
        excitation = f"{fields.current_A:.5g} A · {excitation}"
    location = (
        "on the external suppression surface"
        if fields.faces is not None or interior
        else "at saved suppression samples"
    )
    figure.suptitle(
        (f"{title}\n" if title else "") + f"{component_label} {location} · {excitation}",
        fontsize=12,
    )
    # Opposing viewpoints make the back of a closed surface visible in static exports.
    rows, columns = (1, 2) if len(values) == 1 else (2, len(values))
    axes = []
    for side, (elevation, azimuth) in enumerate(((22, -56), (-22, 124))):
        for index, ((label, _), scalar) in enumerate(zip(vectors, values, strict=True)):
            axis = figure.add_subplot(rows, columns, side * len(values) + index + 1, projection="3d")
            axes.append(axis)
            axis.set_proj_type("ortho")
            axis.view_init(elev=elevation, azim=azimuth)
            if triangulated:
                collection = Poly3DCollection(points[fields.faces], linewidths=0, cmap=cmap, norm=norm)
                collection.set_array(np.mean(scalar[fields.faces], axis=1))
                axis.add_collection3d(collection)
            else:
                axis.scatter(
                    *points[selection].T,
                    c=scalar[selection],
                    s=11,
                    cmap=cmap,
                    norm=norm,
                    depthshade=False,
                    linewidths=0,
                )
            if net_detail:
                # Show the projection of the same peak vertex from either side.
                axis.computed_zorder = False
                axis.scatter(
                    *peak_point,
                    marker="*",
                    s=110,
                    facecolor="white",
                    edgecolor="black",
                    linewidths=1.1,
                    depthshade=False,
                    zorder=10,
                )
            axis.set(
                xlim=center[0] + box[0] * np.array([-0.52, 0.52]),
                ylim=center[1] + box[1] * np.array([-0.52, 0.52]),
                zlim=center[2] + box[2] * np.array([-0.52, 0.52]),
            )
            axis.set_box_aspect(box.copy())
            for dimension, name in zip((axis.xaxis, axis.yaxis, axis.zaxis), "xyz", strict=True):
                dimension.set_major_locator(MaxNLocator(3))
                if index == 0 and side == 0:
                    dimension.set_label_text(name, fontsize=8)
                    dimension.labelpad = -4
                if side or index:
                    dimension.set_major_formatter(NullFormatter())
            axis.tick_params(labelsize=7, pad=0)
            if side == 0 or len(values) == 1:
                axis.set_title(f"{label}\nRMS {rms[index]:.3g} µT   ·   Peak {peaks[index]:.3g} µT", fontsize=9)
            if side == 1 and index == 0:
                axis.text2D(0.03, 0.03, "Opposite side", transform=axis.transAxes, fontsize=7, color="#444444")
    legend = figure.colorbar(
        ScalarMappable(norm, cmap), ax=axes, orientation="horizontal", shrink=0.68, fraction=0.05, pad=0.06
    )
    legend.set_label(f"{component_label} [µT]   ·   world coordinates [mm]\n{summary}", fontsize=9)
    if max(peaks) == 0:
        legend.set_ticks([0])
    legend.ax.tick_params(labelsize=8)
    if triangulated:
        summary += " · Facet colors average the vertex samples"
    elif interior:
        summary += " · Direct field at three points per triangle"
    if fields.target_T is not None and not nonzero_target:
        summary += " · Requested field is zero"
    return f"{summary} · {net_readout} · {excitation}"
