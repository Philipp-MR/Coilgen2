from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

from projects._shared.object_parameters import ParameterNamespace, parameters_to_dict
from projects._shared.paths import REPOSITORY_ROOT, project_output_dir, project_output_root


PROJECT_ROOT = REPOSITORY_ROOT
GENERAL_EXAMPLES_OUTPUT_ROOT = project_output_root("general_examples")


def interactive_plots_enabled(requested: bool) -> bool:
    """Honor plot choices and explicit headless command/environment settings."""
    static_backends = {"agg", "pdf", "pgf", "ps", "svg", "template"}
    return (
        bool(requested)
        and "--headless" not in sys.argv[1:]
        and os.environ.get("MPLBACKEND", "").lower() not in static_backends
    )


@contextmanager
def native_thread_budget(threads: int = 1):
    """Bound inner BLAS pools when threadpoolctl is installed; restore on exit."""
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("Native thread count must be a positive integer.")
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError:
        yield
    else:
        with threadpool_limits(limits=threads):
            yield


def example_output_dir(name: str) -> Path:
    return project_output_dir("general_examples", name)


def parse_example_parameters(
    user_parameters: Mapping[str, Any] | None,
    default_parameters: Mapping[str, Any],
    *,
    example_name: str,
) -> dict[str, Any]:
    """Merge Spyder-editable object/dict parameters with defaults and report fallbacks."""

    if user_parameters is None:
        user_parameters = {}
    merged, missing, unknown = _merge_parameter_dicts(
        parameters_to_dict(user_parameters),
        parameters_to_dict(default_parameters),
    )
    if isinstance(user_parameters, ParameterNamespace):
        _attach_missing_defaults_for_help(user_parameters, default_parameters)
    if missing:
        print(f"{example_name}: using default values for missing parameter(s):")
        for path, value in missing:
            print(f"  {path} = {value!r}")
    if unknown:
        print(f"{example_name}: ignoring unknown parameter(s):")
        for path in unknown:
            print(f"  {path}")
    return merged


def _attach_missing_defaults_for_help(
    namespace: ParameterNamespace,
    default_parameters: Mapping[str, Any],
) -> None:
    """Populate missing namespace keys with defaults so IDE help is complete.

    This does not change the parser semantics: the same defaults would be used
    during merging anyway. It only makes ``PARAMETERS.help()`` useful in Spyder
    for scripts whose public ``PARAMETERS`` object starts as a small override.
    """

    for key, value in default_parameters.items():
        if key not in namespace:
            namespace[key] = deepcopy(value)
            continue
        current = namespace[key]
        if isinstance(current, ParameterNamespace) and isinstance(value, Mapping):
            _attach_missing_defaults_for_help(current, value)


def _merge_parameter_dicts(
    user_parameters: Mapping[str, Any],
    default_parameters: Mapping[str, Any],
    *,
    prefix: str = "",
) -> tuple[dict[str, Any], list[tuple[str, Any]], list[str]]:
    merged: dict[str, Any] = {}
    missing: list[tuple[str, Any]] = []
    unknown: list[str] = []
    for key, default_value in default_parameters.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in user_parameters:
            merged[key] = deepcopy(default_value)
            missing.append((path, default_value))
            continue
        user_value = user_parameters[key]
        if isinstance(default_value, Mapping) and isinstance(user_value, Mapping):
            nested, nested_missing, nested_unknown = _merge_parameter_dicts(
                user_value,
                default_value,
                prefix=path,
            )
            merged[key] = nested
            missing.extend(nested_missing)
            unknown.extend(nested_unknown)
        else:
            merged[key] = deepcopy(user_value)
    for key in user_parameters:
        if key not in default_parameters:
            path = f"{prefix}.{key}" if prefix else str(key)
            unknown.append(path)
    return merged, missing, unknown


def close_previous_matplotlib_plots(*, except_figures: Sequence[Any] | None = None) -> None:
    """Close matplotlib figures from previous Spyder runs if matplotlib exists."""

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return
    try:
        from coilgen import close_matplotlib_plot_books

        close_matplotlib_plot_books()
    except Exception:
        pass
    keep_ids = {id(fig) for fig in (except_figures or ()) if fig is not None}
    if not keep_ids:
        plt.close("all")
        return
    for fig_num in list(plt.get_fignums()):
        try:
            fig = plt.figure(fig_num)
        except Exception:
            continue
        if id(fig) not in keep_ids:
            plt.close(fig)


def plot_figures_book_if_requested(
    figures: Sequence[Any],
    *,
    enabled: bool,
    close_previous: bool = True,
    title: str = "CoilGen Study Plot Book",
    names: Sequence[str] | None = None,
    categories: Sequence[str] | None = None,
) -> list:
    """Open non-result study figures in one active plot-book window."""

    if not interactive_plots_enabled(enabled) or not figures:
        if close_previous:
            close_previous_matplotlib_plots(except_figures=figures)
        return []
    if close_previous:
        close_previous_matplotlib_plots(except_figures=figures)
    try:
        from coilgen import activate_matplotlib_interactive_backend, plot_matplotlib_figures_book

        backend = activate_matplotlib_interactive_backend()
        start = time.perf_counter()
        book = plot_matplotlib_figures_book(
            figures,
            title=title,
            names=names,
            categories=categories,
            show=True,
            block=False,
        )
        elapsed = time.perf_counter() - start
    except ImportError as exc:
        print(f"Active matplotlib plot book skipped: {exc}")
        return []
    print(f"Matplotlib backend: {backend}")
    print(f"Opened 1 active matplotlib plot book with {len(figures)} page(s).")
    print(f"Active matplotlib plotting time: {elapsed:.3f} s")
    return [book]


def open_project_plot_book_after_run(
    project_run: Any,
    *,
    figures: Sequence[Any] | None = None,
    include: Sequence[str] | None = None,
    close_previous: bool = False,
    max_parts: int | None = None,
    enabled: bool = True,
) -> Any | None:
    """Open or reopen a plot book from already-computed script variables.

    This is intended for Spyder-style design scripts. Keep the returned
    ``plot_book`` variable in the workspace; rerun the same line or call
    ``plot_book.open()`` to show the book again without rerunning CoilGen.
    """

    if not interactive_plots_enabled(enabled):
        return None
    for figure in figures or ():
        if hasattr(figure, "open"):
            figure.open(block=False)
            return figure
    if project_run is None or not hasattr(project_run, "show_plot_book"):
        return None
    return project_run.show_plot_book(
        include=include,
        max_parts=max_parts,
        close_previous=close_previous,
        show=True,
        block=False,
    )


def print_result_summary(result, *, title: str = "CoilGen run") -> None:
    """Print the compact summary most examples need."""

    from coilgen import (
        regularization_selection_report,
        result_performance_table_rows,
        result_quality_status,
        result_summary_dict,
        symmetry_summary_dict,
    )

    quality = result_quality_status(result)
    summary = result_summary_dict(result)
    symmetry = symmetry_summary_dict(result)
    print(title)
    print(f"execution status: {summary['execution_status']}")
    print(f"engineering acceptance: {quality.status} (reported checks only)")
    for message in quality.failures:
        print(f"  FAIL: {message}")
    for message in quality.warnings:
        print(f"  WARN: {message}")
    print(f"parts: {len(result.coil_parts)}")
    print(f"target points: {len(result.target_field.coords)}")
    print(f"potential step: {result.potential_step:.6g}")
    print(f"needed current layout: {result.needed_current_layout:.6g} A")
    if getattr(result, "regularization_diagnostics", None):
        diagnostics = result.regularization_diagnostics
        print(f"regularization method: {diagnostics.get('selection_method', 'unknown')}")
        print(f"regularization factor: {result.regularization_factor:.6g}")
        print(f"regularization lambda: {result.regularization_lambda:.6g}")
        candidates = diagnostics.get("candidate_factors", ())
        if diagnostics.get("selection_method") == "l_curve" and candidates:
            print(f"L-curve candidates: {len(candidates)}")
            print(f"L-curve scan solver: {diagnostics.get('candidate_solver', 'unknown')}")
            print(f"L-curve smoothing window: {diagnostics.get('l_curve_curvature_smoothing_window', 'unknown')}")
            if diagnostics.get("fallback"):
                print(f"L-curve fallback: {diagnostics.get('fallback_reason', 'no reliable knee found')}")
            try:
                regularization = regularization_selection_report(result)
                print(f"L-curve knee type: {regularization.knee_type}")
                print(
                    "L-curve selected positions: "
                    f"residual={regularization.selected_residual_position:.3g}, "
                    f"resistance={regularization.selected_regularization_position:.3g}"
                )
                for warning in regularization.warnings:
                    print(f"  L-curve WARN: {warning}")
            except ValueError:
                pass
    if symmetry.get("symmetry_diagnostics_available"):
        active = symmetry.get("symmetry_reduction_active")
        factor = symmetry.get("symmetry_reduction_factor")
        reason = symmetry.get("symmetry_reduction_reason")
        print(f"symmetry reduction: active={active}, factor={factor:.3g} ({reason})")
        failed_pairs = symmetry.get("contour_symmetry_failed_pair_count", 0)
        pair_count = symmetry.get("contour_symmetry_pair_count", 0)
        warnings = symmetry.get("symmetry_warning_count", 0)
        print(f"contour symmetry pairs: {pair_count}, failed={failed_pairs}, warnings={warnings}")
    if getattr(result, "timing_summary", None):
        print(f"execution time: {result.timing_summary[-1]['elapsed_seconds']:.3f} s")
    if result.coil_lengths is not None and len(result.coil_lengths):
        print(f"coil length part 1: {result.coil_lengths[0]:.6g} m")
        if len(result.coil_lengths) > 1:
            print(f"coil length total: {float(sum(result.coil_lengths)):.6g} m")
    electrical_calculation_disabled = bool(getattr(result.input_data, "skip_inductance_calculation", False))
    if electrical_calculation_disabled:
        print("coil resistance and inductance: unavailable (calculation disabled)")
    elif result.resistance is not None and len(result.resistance):
        print(f"coil resistance part 1: {result.resistance[0]:.6g} ohm")
        if len(result.resistance) > 1:
            print(f"coil resistance total: {float(sum(result.resistance)):.6g} ohm")
    if not electrical_calculation_disabled and result.inductance_matrix is not None and result.inductance_matrix.size:
        inductance = result.inductance_matrix
        same_current = float(inductance.sum())
        print(f"coil inductance part 1 self: {inductance[0, 0]:.6g} H")
        if inductance.shape[0] > 1:
            print(f"coil inductance same-current total: {same_current:.6g} H")
            print("coil inductance matrix [H]:")
            for row in inductance:
                print("  " + " ".join(f"{float(value): .6e}" for value in row))
    print(f"contours: {sum(len(part.contour_lines) for part in result.coil_parts)}")
    print("DSV errors:")
    for key, value in result.error_vals.items():
        print(f"  {key}: {value:.3f}%")
    print("Plot-book summary table:")
    _print_grouped_result_table_rows(result_performance_table_rows(result))


def _print_grouped_result_table_rows(rows: Sequence[Sequence[str]]) -> None:
    current_group = None
    quantity_width = max((len(str(row[1])) for row in rows if len(row) >= 4), default=0)
    quantity_width = min(max(quantity_width, 24), 48)
    for row in rows:
        if len(row) < 4:
            continue
        group, quantity, value, unit = (str(row[0]), str(row[1]), str(row[2]), str(row[3]))
        if group != current_group:
            print(f"  {group}:")
            current_group = group
        suffix = "" if not unit or unit == "-" else f" {unit}"
        print(f"    {quantity:<{quantity_width}} {value}{suffix}")


def print_saved_files(files: Sequence[Path], *, title: str) -> None:
    """Print saved file paths in a consistent way."""

    print(title)
    for file in files:
        print(f"  {file}")


def save_standard_reports(result, output_dir: Path, *, prefix: str, design=None) -> list[Path]:
    from coilgen import write_design_report, write_design_summary_csv

    report = write_design_report(
        result,
        output_dir / f"{prefix}_design_report.md",
        title=f"{prefix} CoilGen Design Report",
        design=design,
    )
    summary = write_design_summary_csv(result, output_dir / f"{prefix}_design_summary.csv")
    return [report, summary]
