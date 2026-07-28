from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .models import CoilGenResult
from .manufacturing import save_wire_trace_as_csv
from .reporting import ResultQuality, result_quality_status, write_design_report, write_design_summary_csv
from .study import DesignStudyResult, study_result_from_coilgen_result


@dataclass(frozen=True, slots=True)
class CoilDesignRunOptions:
    """High-level options for user-facing coil-design scripts."""

    project_root: Path | str = Path.cwd()
    output_dir: Path | str | None = None
    output_prefix: str = "coilgen_design"
    save_reports: bool = True
    open_plot_book: bool = False
    close_previous_plots: bool = True
    plot_include: Sequence[str] | None = None
    max_plot_parts: int | None = None


@dataclass(frozen=True, slots=True)
class CoilDesignRunArtifacts:
    """Outputs returned by ``run_coil_design``."""

    result: CoilGenResult
    quality: ResultQuality
    study_result: DesignStudyResult
    files: tuple[Path, ...] = field(default_factory=tuple)
    figures: tuple[Any, ...] = field(default_factory=tuple)


def run_coil_design(design: Any, options: CoilDesignRunOptions | None = None) -> CoilDesignRunArtifacts:
    """Run a CoilGen design with reports and optional active plot book.

    The ``design`` object is expected to provide ``run(project_root=...)`` and
    may optionally provide ``as_user_parameters()``. This keeps examples compact
    while preserving direct access to the full ``CoilGenResult``.
    """

    options = options or CoilDesignRunOptions()
    project_root = Path(options.project_root)
    result = design.run(project_root=project_root)
    output_dir = (
        Path(options.output_dir) if options.output_dir is not None else project_root / "outputs" / options.output_prefix
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    if options.save_reports:
        files.append(
            write_design_report(
                result,
                output_dir / f"{options.output_prefix}_design_report.md",
                title=f"{options.output_prefix} CoilGen Design Report",
                design=design,
            )
        )
        files.append(write_design_summary_csv(result, output_dir / f"{options.output_prefix}_design_summary.csv"))
    if _save_wire_trace_csv_enabled(result):
        files.extend(save_wire_trace_as_csv(result, output_dir, prefix=options.output_prefix))

    figures: list[Any] = []
    if options.open_plot_book:
        figures.extend(
            _open_result_plot_book(
                result,
                include=options.plot_include,
                max_parts=options.max_plot_parts,
                close_previous=options.close_previous_plots,
            )
        )

    quality = result_quality_status(result)
    parameters = design.as_user_parameters() if hasattr(design, "as_user_parameters") else {}
    study_result = study_result_from_coilgen_result(options.output_prefix, parameters, result)
    return CoilDesignRunArtifacts(
        result=result,
        quality=quality,
        study_result=study_result,
        files=tuple(files),
        figures=tuple(figures),
    )


def run_user_design(design: Any, options: CoilDesignRunOptions | None = None) -> CoilDesignRunArtifacts:
    """Alias for ``run_coil_design`` used by user-facing scripts."""

    return run_coil_design(design, options)


def _save_wire_trace_csv_enabled(result: CoilGenResult) -> bool:
    return bool(getattr(getattr(result, "input_data", None), "save_wire_trace_csv", False))


def _open_result_plot_book(
    result: CoilGenResult,
    *,
    include: Sequence[str] | None,
    max_parts: int | None,
    close_previous: bool,
) -> list[Any]:
    from .matplotlib_visualization import (
        activate_matplotlib_interactive_backend,
        close_matplotlib_plot_books,
        plot_result_matplotlib_book,
    )

    if close_previous:
        close_matplotlib_plot_books()
        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except Exception:
            pass
    activate_matplotlib_interactive_backend()
    book = plot_result_matplotlib_book(
        result,
        include=include,
        max_parts=max_parts,
        show=True,
        block=False,
    )
    return [book]
