from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from copy import deepcopy
from dataclasses import MISSING, asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


SECTION_DESCRIPTIONS: dict[str, str] = {
    "run": "Execution switches. These values decide which cases run, whether plots are shown, and how failures are handled.",
    "study": "Scientific comparison settings shared by a study or parameter sweep.",
    "design": "Single coil-design settings: target field, optimization, contour extraction, conductor, and postprocessing.",
    "geometry": "Physical support-body geometry. Lengths are SI units in metres unless stated otherwise.",
    "shielding": "Active-shielding and magnet-interface suppression-target settings.",
    "numerics": "Low-level numerical settings. Change these when you need accuracy, speed, or solver behavior control.",
    "target": "Target-field or DSV sampling settings.",
    "optimization": "Stream-function optimization and regularization choices.",
    "layout": "Contour extraction, routing, postprocessing, and electrical layout choices.",
    "conductor": "Physical conductor dimensions and material parameters.",
    "plots": "Interactive and saved plotting settings.",
    "output": "Output directory, file names, and report switches.",
    "paper_figures": "Manuscript figure switches and figure-resolution settings.",
    "regularization_search": "Optional high-resolution L-curve or regularization pre-scan settings.",
    "mesh_resolution": "Mesh sampling densities for generated primitive support surfaces.",
    "benchmark": "Benchmark or validation repetition/case settings.",
    "template": "Code-generation settings for creating a new geometry API template.",
    "csv_target": "CSV target-field generation settings.",
    "coilgen": "Direct CoilGen pipeline configuration. Names match CoilGenConfig where possible.",
    "symmetry": "Mirror-symmetry checking and symmetry-reduction settings.",
}


KEY_DESCRIPTIONS: dict[str, str] = {
    "name": "Human-readable design or output name. Used in reports and output folders.",
    "gradient_axis": "Requested linear gradient direction for Bz.",
    "field_shape": "Target Bz expression or predefined target-field shape.",
    "field_shape_function": "Target Bz expression used by the low-level CoilGen pipeline.",
    "target_bz_expression": "Target Bz expression used by a scientist-facing design class.",
    "target_scale": "Scale factor for the requested target field.",
    "target_gradient_strength_T_per_m": "Requested gradient strength used for current scaling.",
    "target_gradient_strength": "Requested target-field scale in low-level CoilGen configuration.",
    "nifti_target_recenter_mode": "NIfTI target coordinate placement: pointcloud_centroid_to_origin centers the final sampled cloud at (0,0,0); none preserves NIfTI world coordinates; legacy volume/selected-center modes can align to the origin or current-surface bounding-box center.",
    "field_evaluation_current_source": "Current used for final layout field evaluation: target_fit for gradient coils, target_projection for RMS target scaling, target_std_projection for mean-free shim homogeneity, or stream_function for fixed-current layouts.",
    "target_dsv_m": "Diameter of the spherical design spherical volume used for internal field metrics.",
    "target_region_radius_m": "Radius of the DSV target point cloud.",
    "target_region_radius": "Radius of the DSV target point cloud.",
    "target_region_resolution": "Number of samples per coordinate axis before spherical DSV masking.",
    "contour_levels": "Number of stream-function contour levels converted into discrete wire turns.",
    "levels": "Number of stream-function contour levels converted into discrete wire turns.",
    "contour_discretization_method": "Contour discretization mode: level_count uses a requested level count; engineering derives contour current from max current and minimum wire distance.",
    "max_current_A": "Maximum allowed current per contour wire.",
    "max_wire_current_A": "Maximum allowed current per contour wire.",
    "min_wire_distance_m": "Minimum requested distance between neighboring contour wires.",
    "surface_current_density_limit_A_per_m": "Optional explicit upper bound for stream-function surface current density.",
    "engineering_contour_step_safety_factor": "Multiplier in (0, 1] applied to max current when engineering contouring chooses a contour current.",
    "engineering_contour_max_levels": "Optional maximum contour level count in engineering contouring. Zero disables the cap.",
    "level_set_method": "How contour levels are shared across coil surfaces: primary, combined, or independent.",
    "zero_crossing_level_strategy": "How zero-crossing stream functions are contoured: paired for symmetric gradient layouts or full_range for asymmetric shim maps.",
    "optimization_method": "Stream-function solver family.",
    "method": "Stream-function solver family.",
    "sf_opt_method": "Low-level stream-function solver family.",
    "stream_function_cost_function": "Mathematical objective used for the stream-function solve.",
    "regularization_method": "How the regularization factor is selected.",
    "regularization_selection_method": "How the regularization factor is selected.",
    "regularization_factor": "Dimensionless Tikhonov factor when fixed regularization is used.",
    "tikonov_reg_factor": "Dimensionless Tikhonov factor used by the low-level CoilGen pipeline.",
    "l_curve_candidate_count": "Number of regularization candidates in an L-curve scan.",
    "l_curve_factor_count": "Number of regularization candidates in an L-curve scan.",
    "l_curve_smoothing_window": "Odd moving-window size used to smooth L-curve curvature.",
    "l_curve_curvature_smoothing_window": "Odd moving-window size used to smooth L-curve curvature.",
    "target_error_tolerance_percent": "Relative target-field residual budget used by fixed-error optimization.",
    "target_error_tolerance_absolute_T": "Absolute residual floor added to fixed-error optimization.",
    "target_error_constraint_metric": "Residual metric constrained by fixed-error optimization: norm, RMS, or mean-free standard deviation.",
    "target_error_std_limit_T": "Absolute DSV residual standard-deviation limit for fixed-error optimization.",
    "target_error_rms_limit_T": "Absolute DSV residual RMS limit for fixed-error optimization.",
    "layout_target_error_std_limit_T": "Absolute standard-deviation limit applied to the realized contour and connected-wire layout field.",
    "layout_target_error_rms_limit_T": "Absolute RMS limit applied to the realized contour and connected-wire layout field.",
    "layout_target_error_acceptance_absolute_T": "Numerical acceptance margin added only when validating the realized layout-field limit.",
    "target_error_constraint_candidate_count": "Candidate count for the dependency-free fixed-error Tikhonov scan.",
    "target_error_constraint_factor_count": "Candidate count for the dependency-free fixed-error Tikhonov scan.",
    "target_error_auto_retry_enabled": "If enabled, rerun fixed-error stream-function solves with relaxed search restrictions when the requested residual specification is not met.",
    "target_error_auto_retry_max_reruns": "Maximum number of automatic fixed-error solver reruns after the initial attempt.",
    "target_error_auto_retry_factor_shrink": "Multiplier applied to the lower Tikhonov scan factor on each automatic rerun.",
    "target_error_auto_retry_candidate_growth": "Candidate-count growth factor applied on each automatic fixed-error solver rerun.",
    "target_error_auto_retry_max_candidate_count": "Upper bound for the fixed-error scan candidate count during automatic reruns.",
    "constrained_quadratic_fallback_policy": "Fallback behavior for constrained_quadratic: target_error_scan allows the Tikhonov scan, strict fails instead.",
    "constrained_quadratic_objective_mode": "Constrained-solver objective: balanced minimizes resistance plus penalties; smoothness_priority uses resistance only as a small tie-breaker.",
    "constrained_quadratic_resistance_tiebreak_factor": "Relative resistance weight retained in smoothness_priority mode to regularize null directions.",
    "potential_offset_factor": "Offset fraction used when choosing contour potential levels.",
    "pot_offset_factor": "Offset fraction used when choosing contour potential levels.",
    "contour_part_phase_offset_deltas": "Per-part contour-grid phase deltas added modulo one to pot_offset_factor; use staggered values for interleaved multilayer quantization.",
    "min_contour_points": "Minimum number of points required for a contour to be retained.",
    "min_point_loop_number": "Minimum number of points required for a contour to be retained.",
    "min_turn_spacing_m": "Requested minimum spacing between turns. Zero disables the constraint.",
    "min_turn_distance": "Requested minimum spacing between turns. Zero disables the constraint.",
    "turn_spacing_regularization_factor": "Optional quadratic penalty weight used to reshape the stream function toward larger turn spacing.",
    "turn_spacing_max_iterations": "Maximum turn-spacing penalty escalation attempts during stream-function optimization.",
    "turn_spacing_penalty_growth": "Multiplicative growth factor for turn-spacing penalty retries.",
    "contour_field_constrained_smoothing_enabled": "Search on-surface contour smoothing strengths and retain the smoothest candidate that still passes field and geometry limits.",
    "contour_field_smoothing_candidates": "Candidate (iteration count, smoothing factor) pairs for field-constrained contour smoothing.",
    "contour_field_smoothing_min_improvement_percent": "Minimum relative contour-curvature reduction required before accepting a smoothed candidate.",
    "contour_level_range_percentile": "Percentile of stream-function values used to place contour levels; lower values ignore isolated extrema when choosing the contour step.",
    "skip_calculation_min_winding_distance": "If False, measure the final minimum contour spacing and report whether min_turn_distance was met.",
    "conductor_width_m": "Rectangular conductor width.",
    "conductor_cross_section_width": "Rectangular conductor width.",
    "width_m": "Rectangular conductor width.",
    "conductor_height_m": "Rectangular conductor height.",
    "conductor_cross_section_height": "Rectangular conductor height.",
    "height_m": "Rectangular conductor height.",
    "conductor_resistivity_ohm_m": "Electrical resistivity of the conductor material.",
    "specific_conductivity_conductor": "Conductor resistivity used by the current code path, despite the historical name.",
    "postprocess_layout": "If True, build connected wire paths from contour groups.",
    "skip_postprocessing": "If True, skip contour routing and evaluate unconnected contours.",
    "contour_only_sweep_mode": "If True, sweep every closed contour into an independent volumetric mesh without routing or interconnection.",
    "skip_sweep": "If True, skip swept conductor-body generation.",
    "skip_inductance": "If True, skip inductance and resistance evaluation.",
    "skip_inductance_calculation": "If True, skip inductance and resistance evaluation.",
    "save_wire_trace_csv": "If True, save unswept contour STEP wires plus routed-path CSV/STEP geometry when routing is enabled.",
    "wire_trace_csv_format": "Wire-trace CSV schema: detailed includes diagnostics and a header; xyz_no_header writes ordered X,Y,Z points only and one file per routed part.",
    "wire_trace_csv_length_unit": "Coordinate unit for xyz_no_header wire-path CSV files: m, cm, or mm.",
    "wire_trace_csv_decimal_places": "Fixed number of digits after the decimal point in xyz_no_header wire-path CSV files.",
    "inductance_source": "Source geometry for inductance: auto, connected wire path, or unconnected contours.",
    "cpu_workers": "Parallel CPU worker count. 0 uses all logical cores but can use much more RAM; use 1-4 for long high-resolution sweeps.",
    "parallel_worker_count": "Parallel CPU worker count. 0 uses all logical cores but can use much more RAM; use 1-4 for long high-resolution sweeps.",
    "constrained_optimization_solver": "Conic solver preference for constrained_quadratic: auto, clarabel, ecos, or scs.",
    "constrained_optimization_tolerance": "Numerical tolerance for constrained stream-function optimization.",
    "constrained_optimization_max_iterations": "Maximum iterations for constrained stream-function optimization.",
    "constrained_optimization_threads": "Optional CVXPY conic-solver thread count. 0 keeps the solver default; currently applied to CLARABEL max_threads.",
    "progress": "If True, print progress messages during long calculations.",
    "progress_flag": "If True, print progress messages during long calculations.",
    "enabled": "Enables this section or feature.",
    "secondary_target_weight": "Relative optimization weight for zero-field suppression target points.",
    "unshielded_reference_weight": "Suppression target weight for the unshielded reference design.",
    "suppression_separation_m": "Distance from the active shield to the magnet-interface suppression surface.",
    "suppression_edge_length_m": "Target edge length used to sample the suppression surface.",
    "suppression_shell_thickness_m": "Thickness of an auxiliary suppression shell where that generator uses one.",
    "suppression_interface_mode": "Which shield faces define the suppression interface for split-bore supports.",
    "field_component": "Field component used for metrics.",
    "primary_outer_radius_m": "Outer radius of the primary circular biplanar support.",
    "primary_edge_length_m": "Target mesh edge length on the primary support.",
    "outer_radius_m": "Outer support radius.",
    "inner_radius_m": "Inner aperture radius.",
    "gap_m": "Physical access gap.",
    "plane_distance_m": "Distance between the two primary biplanar surfaces.",
    "shield_separation_m": "Normal or radial distance from primary support to active-shield support.",
    "edge_length_m": "Target mesh edge length on generated support surfaces.",
    "half_length_m": "Length of each split-bore half measured away from the central gap.",
    "normal": "Surface normal direction.",
    "center_m": "Geometry center in scanner coordinates.",
    "b0_direction": "Direction of the main B0 field in scanner coordinates.",
    "mesh_rotation_axis": "Axis used to rotate a generated mesh into scanner coordinates.",
    "mesh_rotation_angle_rad": "Rotation angle used to orient a generated mesh.",
    "selected_cases": "Primitive or example case names to run.",
    "show_plots": "Open interactive Matplotlib/plot-book windows.",
    "show_active": "Open interactive Matplotlib/plot-book windows.",
    "defer_open_to_main": "Build the plot book without showing it inside the workflow, leaving one explicit open command in the calling script.",
    "failure_fallback_enabled": "If a calculation fails after input preparation, preserve the setup context so the calling script can build a fallback plot book.",
    "preload_pages": "Plot-book page loading policy: none renders only selected pages, idle loads remaining pages gradually, and all loads them immediately.",
    "plot_pointcloud_max_points": "Maximum number of target samples drawn in each point-cloud plot; zero draws every point and does not affect optimization or metrics.",
    "plot_pointcloud_random_seed": "Seed for the reproducible random subset used when a point-cloud plot exceeds its display limit.",
    "close_previous_plots": "Close old Matplotlib windows before opening new plots.",
    "close_previous": "Close old Matplotlib windows before opening new plots.",
    "save_plots": "Save figures to disk.",
    "save_plot_formats": "Figure file formats to save.",
    "save_reports": "Save CSV/Markdown reports to disk.",
    "include": "Plot-book page names to include.",
    "mode": "Plot display mode.",
    "run_design": "If True, execute CoilGen. If False, only validate generated configurations.",
    "continue_on_failure": "If True, keep parameter studies running when a single case fails.",
    "continue_on_case_failure": "If True, keep parameter studies running when a single case fails.",
    "write_case_cache": "Save each completed parameter-study row immediately so interrupted sweeps can be resumed.",
    "resume_from_case_cache": "Reuse cached parameter-study rows instead of rerunning already completed cases.",
    "retained_result_policy": "How many heavyweight CoilGenResult objects are kept after a parameter sweep case.",
    "rerun_cached_representative_layouts": "Recompute only central representative layouts when scalar rows are loaded from cache.",
    "preset": "Named study preset that selects a predefined quick, validation, or paper-quality parameter set.",
    "scenario": "Scenario identifier used to group related geometry, target, and reporting settings.",
    "scenario_key": "Parameter-section key for the scenario-specific geometry and solver settings.",
    "scenario_title": "Publication or plot label describing the physical scanner scenario.",
    "study_type": "Study category used for reporting, plotting, and metadata grouping.",
    "design": "Configured CoilGen design object for this case.",
    "active_design": "Active-shielded CoilGen design object for this case.",
    "unshielded_design": "Unshielded reference CoilGen design object for this case.",
    "shielded_design": "Active-shielded CoilGen design object produced by the single-case workflow.",
    "geometry_parameters": "Resolved physical geometry parameters used to generate support meshes.",
    "geometry": "Resolved physical geometry parameters or geometry object for a study case.",
    "output_dir": "Directory where study outputs, metadata, and figures are written.",
    "rows": "List of result dictionaries; one row per evaluated case or parameter point.",
    "last_results": "Representative CoilGen results retained for plotting after a parameter study.",
    "figures": "Matplotlib figure objects returned for interactive inspection.",
    "include_layout_snapshots": "If True, save one lightweight isometric layout PNG for each successful active-shielded sweep case.",
    "include_layout_snapshot_gallery": "If True, build paged gallery figures from saved per-case layout snapshot PNGs.",
    "layout_snapshot_dpi": "Raster resolution for per-case layout snapshot PNGs.",
    "layout_snapshot_figsize": "Figure size in inches for per-case layout snapshot PNGs.",
    "layout_snapshot_max_mesh_edges": "Maximum number of support-mesh edges drawn per coil part in lightweight layout snapshots.",
    "layout_snapshot_elev": "Elevation view angle in degrees for isometric-style layout snapshots.",
    "layout_snapshot_azim": "Azimuth view angle in degrees for isometric-style layout snapshots.",
    "layout_snapshot_gallery_columns": "Number of thumbnail columns in each layout snapshot gallery page.",
    "layout_snapshot_gallery_max_per_page": "Maximum number of layout snapshot thumbnails per gallery figure page.",
    "saved_files": "Paths to files written by the workflow.",
    "files": "Paths to files written by the workflow.",
    "run_label": "Short label used in console output and saved reports for a single workflow run.",
    "l_curve_label": "Short label used for L-curve diagnostic plots and files.",
    "fringe_figure_title": "Figure title for active-shield fringe-field comparison plots.",
    "fringe_metric_title": "Metric label for active-shield fringe-field reports.",
    "build_design": "Callable that converts parameters into the active-shielded CoilGen design.",
    "build_unshielded_design": "Callable that converts parameters into an unshielded reference design.",
    "output_dir_factory": "Callable that maps an output name to a project-specific output directory.",
    "high_res_l_curve_result": "Optional high-resolution L-curve diagnostic result.",
    "unshielded_result": "CoilGen result for the unshielded reference design.",
    "shielded_result": "CoilGen result for the active-shielded design.",
    "suppression_surface": "Mesh or point set representing the magnet-interface suppression target surface.",
    "shielding_effect_metrics": "Dictionary of active-shielding performance metrics.",
}


VALUE_EXPECTATIONS: dict[str, str] = {
    "gradient_axis": "'x', 'y', 'z', or a supported CoilGen target expression such as 'xy'.",
    "field_shape": "'x', 'y', 'z', 'none', or a supported symbolic target-field expression.",
    "field_shape_function": "'x', 'y', 'z', 'none', or a supported symbolic target-field expression.",
    "field_component": "'Bx', 'By', 'Bz', or 'Bmag'.",
    "optimization_method": "'tikhonov', 'target_error_constrained_efficiency', or 'constrained_quadratic'.",
    "method": "'tikhonov', 'target_error_constrained_efficiency', or 'constrained_quadratic'.",
    "sf_opt_method": "'tikhonov', 'target_error_constrained_efficiency', or 'constrained_quadratic'.",
    "regularization_method": "'fixed', 'l_curve', or 'auto'.",
    "regularization_selection_method": "'fixed', 'l_curve', or 'auto'.",
    "stream_function_cost_function": "Usually 'field_error_plus_resistance'. Additional schemes must be implemented in coilgen.optimization.",
    "inductance_source": "'auto', 'wire_path', or 'unconnected_contours'.",
    "mode": "'book' for the interactive plot book, or a supported plotting backend mode.",
    "retained_result_policy": "'none', 'representative', or 'all'. Use 'none' for lowest memory; 'representative' keeps only central layouts for figures.",
    "inter_group_connection_path_method": "'direct_3d', 'geodesic', or strict 'surface_geodesic'.",
    "suppression_interface_mode": "'radial_split', 'radial', 'split', or 'full'.",
    "sensitivity_calculation_method": "'vectorized' or another implemented sensitivity backend.",
}


def _format_value(value: Any, *, limit: int = 84) -> str:
    if isinstance(value, np.ndarray):
        text = f"ndarray shape={value.shape} dtype={value.dtype}"
    else:
        text = repr(value)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _section_name(path: str) -> str:
    return path.split(".")[-1]


def _unit_hint(key: str) -> str:
    if key.endswith("_m") or key in {"target_region_radius", "min_turn_distance", "min_wire_distance_m"}:
        return "SI length in metres."
    if key.endswith("_A") or key.endswith("_current_A"):
        return "Electrical current in amperes."
    if key.endswith("_A_per_m"):
        return "Surface current density in amperes per metre."
    if key.endswith("_T") or key.endswith("_T_per_A"):
        return "Magnetic field in tesla."
    if key.endswith("_T_per_m") or key.endswith("_mT_per_m_per_A"):
        return "Gradient quantity in SI-derived units from the name."
    if key.endswith("_rad"):
        return "Angle in radians."
    if key.endswith("_percent"):
        return "Percent value."
    if key.endswith("_count") or key.endswith("_levels") or key.endswith("_resolution"):
        return "Positive integer count."
    if key.endswith("_weight") or key.endswith("_factor"):
        return "Dimensionless scalar."
    if key.endswith("_file") or key.endswith("_filename"):
        return "File path or file name."
    if key.startswith(("show_", "save_", "skip_", "run_", "include_", "close_", "compare_", "evaluate_")):
        return "Boolean switch."
    return ""


def parameter_field_description(key: str, value: Any = None) -> str:
    """Return a concise human-facing expectation string for a parameter key."""

    parts: list[str] = []
    description = KEY_DESCRIPTIONS.get(key)
    if description:
        parts.append(description)
    expectation = VALUE_EXPECTATIONS.get(key)
    if expectation:
        parts.append(f"Expected: {expectation}")
    unit = _unit_hint(key)
    if unit:
        parts.append(unit)
    if not parts:
        value_type = type(value).__name__ if value is not None else "value"
        parts.append(f"Project parameter ({value_type}). See the surrounding script section for physical context.")
    return " ".join(parts)


def namespace_doc(path: str, values: Mapping[str, Any]) -> str:
    """Build IDE-visible documentation for a parameter namespace."""

    section = _section_name(path)
    lines = [path]
    description = SECTION_DESCRIPTIONS.get(section)
    if description:
        lines.extend(("", description))
    lines.extend(("", "Fields:"))
    if not values:
        lines.append("- <empty>: this object currently contains only user overrides.")
    for key, value in values.items():
        if isinstance(value, ParameterNamespace):
            summary = SECTION_DESCRIPTIONS.get(key, "Nested parameter section.")
            current = f"{len(value)} fields"
        else:
            summary = parameter_field_description(key, value)
            current = _format_value(value)
        lines.append(f"- {key}: {summary} Current: {current}.")
    lines.extend(
        (
            "",
            "Use obj.help() or obj.describe() in Spyder/IPython for this text.",
            "Use obj.to_dict() to pass the parameters into workflow functions.",
        )
    )
    return "\n".join(lines)


def dataclass_parameter_doc(cls: type[Any]) -> str:
    """Build a class docstring extension from dataclass fields."""

    if not is_dataclass(cls):
        return ""
    lines = ["", "Fields and expected values:"]
    for item in fields(cls):
        if item.name.startswith("_"):
            continue
        default = ""
        if item.default is not MISSING:
            default = f" Default: {_format_value(item.default)}."
        summary = item.metadata.get("doc") if item.metadata else ""
        if not summary:
            summary = parameter_field_description(item.name)
        lines.append(f"- {item.name} ({_type_name(item.type)}): {summary}{default}")
    return "\n".join(lines)


def append_dataclass_field_docs(*classes: type[Any]) -> None:
    """Append generated field documentation to dataclass class docstrings.

    This is intentionally runtime-based so existing dataclass declarations stay
    compact. IDEs and IPython see the updated ``__doc__`` after import.
    """

    marker = "Fields and expected values:"
    for cls in classes:
        if not is_dataclass(cls):
            continue
        base = (cls.__doc__ or "").strip()
        if marker in base:
            continue
        cls.__doc__ = f"{base}\n{dataclass_parameter_doc(cls)}".strip()


def _type_name(value: Any) -> str:
    text = getattr(value, "__name__", None)
    if text:
        return str(text)
    return str(value).replace("typing.", "")


class DocumentedDataclassMixin:
    """Mixin that gives dataclass parameter objects IDE-readable help methods."""

    __slots__ = ()

    def describe(self, field_name: str | None = None) -> str:
        if field_name is not None:
            return self.field_help(field_name)
        if not is_dataclass(self):
            return inspect_doc(type(self))
        lines = [inspect_doc(type(self))]
        lines.append("")
        lines.append("Current values:")
        for item in fields(self):
            if item.name.startswith("_"):
                continue
            value = getattr(self, item.name)
            lines.append(f"- {item.name}: {_format_value(value)}")
        return "\n".join(line for line in lines if line is not None)

    def help(self, field_name: str | None = None) -> str:
        return self.describe(field_name)

    def field_help(self, field_name: str) -> str:
        if not is_dataclass(self):
            raise TypeError("field_help is only available on dataclass instances.")
        for item in fields(self):
            if item.name == field_name:
                value = getattr(self, field_name)
                return (
                    f"{type(self).__name__}.{field_name}\n"
                    f"Type: {_type_name(item.type)}\n"
                    f"{parameter_field_description(field_name, value)}\n"
                    f"Current: {_format_value(value)}"
                )
        raise AttributeError(field_name)


def inspect_doc(cls: type[Any]) -> str:
    return (getattr(cls, "__doc__", "") or getattr(cls, "__name__", str(cls))).strip()


class ParameterNamespace(MutableMapping[str, Any]):
    """Recursive object-and-dict parameter container for scientist scripts.

    The class supports both attribute access and mapping access, so existing
    script internals can keep using ``parameters["section"]["value"]`` while
    user-facing scripts can expose ``PARAMETERS.section.value``.

    IDE usage:
        ``PARAMETERS.help()`` shows all fields in the current section.
        ``PARAMETERS.study.help("gradient_axis")`` shows one field.
        ``PARAMETERS.to_dict()`` returns plain Python containers for workflows.
    """

    def __init__(self, values: Mapping[str, Any] | None = None, *, _path: str = "PARAMETERS", **updates: Any) -> None:
        object.__setattr__(self, "_values", {})
        object.__setattr__(self, "_path", str(_path))
        if values:
            self.update(values)
        if updates:
            self.update(updates)

    def __getattribute__(self, name: str) -> Any:
        if name == "__doc__":
            return object.__getattribute__(self, "describe")()
        return object.__getattribute__(self, name)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __setitem__(self, key: str, value: Any) -> None:
        key_text = str(key)
        self._values[key_text] = _to_namespace_value(value, path=f"{self._path}.{key_text}")

    def __delitem__(self, key: str) -> None:
        del self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self[name] = value

    def __repr__(self) -> str:
        keys = ", ".join(self._values)
        return f"{type(self).__name__}({self._path}: {keys})"

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(self._values))

    def copy(self) -> "ParameterNamespace":
        return type(self)(self.to_dict(), _path=self._path)

    def to_dict(self) -> dict[str, Any]:
        return {key: parameters_to_dict(value) for key, value in self._values.items()}

    def update_from(self, values: Mapping[str, Any] | "ParameterNamespace") -> "ParameterNamespace":
        self.update(parameters_to_dict(values))
        return self

    def describe(self, key: str | None = None) -> str:
        if key is None:
            return namespace_doc(self._path, self._values)
        if key not in self._values:
            raise KeyError(key)
        value = self._values[key]
        if isinstance(value, ParameterNamespace):
            return value.describe()
        return f"{self._path}.{key}\n{parameter_field_description(key, value)}\nCurrent: {_format_value(value)}"

    def help(self, key: str | None = None) -> str:
        return self.describe(key)


def object_parameters(
    values: Mapping[str, Any] | Any, *, root_name: str = "PARAMETERS", **updates: Any
) -> ParameterNamespace:
    """Create recursive object-style parameters from a mapping or dataclass."""

    namespace = ParameterNamespace(parameters_to_dict(values), _path=root_name)
    if updates:
        namespace.update(updates)
    return namespace


def parameters_to_dict(value: Any) -> Any:
    """Convert supported parameter objects to plain Python containers."""

    if isinstance(value, ParameterNamespace):
        return value.to_dict()
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return parameters_to_dict(value.to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return parameters_to_dict(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): parameters_to_dict(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(parameters_to_dict(item) for item in value)
    if isinstance(value, list):
        return [parameters_to_dict(item) for item in value]
    if isinstance(value, Path):
        return value
    if isinstance(value, np.ndarray):
        return np.asarray(value).copy()
    if isinstance(value, np.generic):
        return value.item()
    return deepcopy(value)


def _to_namespace_value(value: Any, *, path: str = "PARAMETERS") -> Any:
    if isinstance(value, ParameterNamespace):
        object.__setattr__(value, "_path", path)
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return object_parameters(asdict(value), root_name=path)
    if isinstance(value, Mapping):
        return object_parameters(value, root_name=path)
    return value
