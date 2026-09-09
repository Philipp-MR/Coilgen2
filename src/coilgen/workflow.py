from __future__ import annotations

import csv
import hashlib
import json
import time
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .designs import CoilDesign
from .manufacturing import save_wire_trace_artifacts
from .models import CoilGenResult
from .persistence import atomic_text, atomic_text_writer, file_digest, json_value, source_digest
from .reporting import result_quality_status, result_summary_dict, write_design_report, write_design_summary_csv
from .study import ParameterAxis
from .visualization import save_plot_gallery


MetricFunction = Callable[["CoilCaseResult"], Mapping[str, Any]]
RUN_RECORD_SCHEMA_VERSION = "coilgen_run_record_v1"
RUN_RECORD_RETENTION_COUNT = 20
RUN_RECORD_MAX_SUMMARY_BYTES = 1_000_000
RUN_RECORD_MAX_EVENTS = 200
RUN_RECORD_MAX_MARKDOWN_BYTES = 250_000
RUN_RECORD_MAX_CSV_COLUMNS = 240
RUN_RECORD_MAX_CELL_CHARS = 8_192
RUN_RECORD_MAX_STRING_CHARS = 8_192
RUN_RECORD_MAX_MAPPING_ITEMS = 300
RUN_RECORD_MAX_SEQUENCE_ITEMS = 200
RUN_RECORD_MAX_TIMING_ROWS = 120
RUN_RECORD_MAX_ARTIFACTS = 80
RUN_RECORD_MAX_MESSAGES = 80
RUN_RECORD_MAX_TRACEBACK_LINES = 80
RUN_RECORD_FILE_SUFFIXES = (
    "_run_summary.json",
    "_run_metrics.csv",
    "_run_events.jsonl",
    "_run_log.md",
)


@dataclass(slots=True)
class CoilVariantSpec:
    """One design variant inside a generic CoilGen case.

    A variant can be an active-shielded design, an unshielded reference, a
    manufacturing alternative, or any other project-defined comparison. The
    workflow treats all variants uniformly and does not know project-specific
    scenario names.
    """

    name: str
    design: CoilDesign
    role: str = "main"
    save_reports: bool = True
    save_gallery: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CoilCaseSpec:
    """Generic executable CoilGen case.

    Project scripts should construct this object from readable parameter
    dictionaries. The core workflow then handles validation, execution, reports,
    optional plot books, failure capture, and metric table rows.
    """

    name: str
    variants: Sequence[CoilVariantSpec]
    output_dir: Path
    parameters: Mapping[str, Any] = field(default_factory=dict)
    project_root: Path | None = None
    show_plots: bool = False
    plot_mode: str = "book"
    plot_include: Sequence[str] | None = None
    plot_max_parts: int | None = None
    close_previous_plots: bool = True
    continue_on_variant_failure: bool = False
    metric_functions: Sequence[MetricFunction] = field(default_factory=tuple)
    write_case_table: bool = True


@dataclass(slots=True)
class CoilCaseResult:
    """Result of running one generic CoilGen case."""

    spec: CoilCaseSpec
    variant_results: dict[str, CoilGenResult] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    files: list[Path] = field(default_factory=list)
    figures: list[Any] = field(default_factory=list)
    failures: dict[str, Exception] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def main_result(self) -> CoilGenResult | None:
        for variant in self.spec.variants:
            if variant.role == "main" and variant.name in self.variant_results:
                return self.variant_results[variant.name]
        return next(iter(self.variant_results.values()), None)

    def row(self) -> dict[str, Any]:
        row = {
            "case": self.spec.name,
            "status": "success" if self.ok else "failed",
            "failure_count": len(self.failures),
        }
        row.update(_flatten_prefixed("parameter", self.spec.parameters))
        row.update(self.metrics)
        if self.failures:
            row["failure_variants"] = "; ".join(self.failures)
            row["failure_messages"] = "; ".join(f"{name}: {exc}" for name, exc in self.failures.items())
        return row


class CoilCaseWorkflow:
    """Run a generic CoilGen case composed of one or more design variants."""

    def run(self, spec: CoilCaseSpec) -> CoilCaseResult:
        if not spec.name:
            raise ValueError("CoilCaseSpec.name must be non-empty.")
        if not spec.variants:
            raise ValueError("CoilCaseSpec requires at least one CoilVariantSpec.")
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        result = CoilCaseResult(spec=spec)

        for variant in spec.variants:
            started_at_utc = _utc_timestamp()
            started = time.perf_counter()
            try:
                variant_result = variant.design.run(project_root=spec.project_root)
                result.variant_results[variant.name] = variant_result
                if variant.save_reports:
                    result.files.extend(
                        _save_variant_reports(
                            variant_result,
                            variant,
                            spec,
                            spec.output_dir,
                            started_at_utc=started_at_utc,
                            duration_seconds=time.perf_counter() - started,
                        )
                    )
                if variant.save_gallery:
                    result.files.extend(
                        save_plot_gallery(
                            variant_result,
                            spec.output_dir,
                            prefix=_safe_file_stem(variant.name),
                        )
                    )
            except Exception as exc:
                result.failures[variant.name] = exc
                if variant.save_reports:
                    try:
                        result.files.extend(
                            write_variant_failure_record(
                                exc,
                                variant,
                                spec,
                                spec.output_dir,
                                started_at_utc=started_at_utc,
                                duration_seconds=time.perf_counter() - started,
                            )
                        )
                    except Exception as report_exc:
                        print(f"Failed to write run failure record for {variant.name}: {report_exc}")
                if not spec.continue_on_variant_failure:
                    raise

        result.metrics.update(default_case_metrics(result))
        for metric_function in spec.metric_functions:
            result.metrics.update(dict(metric_function(result)))
        if spec.write_case_table:
            result.files.extend(
                write_case_result_tables(result, spec.output_dir / f"{_safe_file_stem(spec.name)}_case_result")
            )
        if spec.show_plots and result.main_result is not None:
            result.figures.extend(
                plot_case_result(
                    result,
                    mode=spec.plot_mode,
                    include=spec.plot_include,
                    max_parts=spec.plot_max_parts,
                    close_previous=spec.close_previous_plots,
                )
            )
        return result


@dataclass(slots=True)
class ParameterSweepSpec:
    """Generic Cartesian parameter sweep using a project-provided case factory."""

    name: str
    axes: Sequence[ParameterAxis]
    case_factory: Callable[[Mapping[str, Any]], CoilCaseSpec]
    fixed_parameters: Mapping[str, Any] = field(default_factory=dict)
    output_dir: Path | None = None
    continue_on_failure: bool = True
    write_tables: bool = True
    resume_from_case_cache: bool = True
    case_cache_dir: Path | None = None
    write_incremental_jsonl: bool = True
    retain_case_results: bool = False
    cache_revision: str = ""

    def parameter_points(self) -> list[dict[str, Any]]:
        if not self.name:
            raise ValueError("ParameterSweepSpec.name must be non-empty.")
        if not self.axes:
            raise ValueError("ParameterSweepSpec requires at least one ParameterAxis.")
        names = [axis.name for axis in self.axes]
        if len(set(names)) != len(names):
            raise ValueError("ParameterSweepSpec axes must have unique names.")
        points: list[dict[str, Any]] = []
        for values in product(*[axis.values for axis in self.axes]):
            point = dict(self.fixed_parameters)
            point.update(dict(zip(names, values)))
            points.append(point)
        return points


@dataclass(slots=True)
class ParameterSweepResult:
    """Result of a generic CoilGen parameter sweep."""

    spec: ParameterSweepSpec
    case_results: list[CoilCaseResult] = field(default_factory=list)
    cached_rows: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)

    _ordered_rows: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def rows(self) -> list[dict[str, Any]]:
        return (
            list(self._ordered_rows)
            if self._ordered_rows
            else [*self.cached_rows, *[case_result.row() for case_result in self.case_results]]
        )


class ParameterSweepWorkflow:
    """Run a generic Cartesian sweep of CoilCaseSpec objects."""

    def __init__(self, case_workflow: CoilCaseWorkflow | None = None) -> None:
        self.case_workflow = case_workflow or CoilCaseWorkflow()

    def run(self, spec: ParameterSweepSpec) -> ParameterSweepResult:
        result = ParameterSweepResult(spec=spec)
        points = spec.parameter_points()
        output_dir = Path(spec.output_dir) if spec.output_dir is not None else None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = Path(spec.case_cache_dir) if spec.case_cache_dir is not None else None
        if cache_dir is None and output_dir is not None:
            cache_dir = output_dir / "case_cache"
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / f"{_safe_file_stem(spec.name)}_incremental.jsonl" if output_dir is not None else None
        code_digest = _sweep_code_digest() if cache_dir is not None else ""

        for index, point in enumerate(points, start=1):
            cache_path = None
            try:
                case_spec = spec.case_factory(point)
                if cache_dir is not None:
                    context = _sweep_cache_context(spec, case_spec, code_digest)
                    cache_path = _case_cache_path(cache_dir, index, {"point": point, "context": context})
                if spec.resume_from_case_cache and cache_path is not None:
                    row = _successful_cached_row(cache_path)
                    if row is not None:
                        result.cached_rows.append(row)
                        result._ordered_rows.append(row)
                        continue
                case_result = self.case_workflow.run(case_spec)
                row = case_result.row()
                if not case_result.ok and not spec.continue_on_failure:
                    raise RuntimeError(
                        f"Case {case_spec.name!r} has failed variants: {row.get('failure_messages', '')}"
                    )
                if cache_path is not None:
                    _write_json(row, cache_path)
                if spec.write_incremental_jsonl and jsonl_path is not None:
                    _append_jsonl(row, jsonl_path)
                result._ordered_rows.append(row)
                if spec.retain_case_results:
                    result.case_results.append(case_result)
                else:
                    result.cached_rows.append(row)
                if not case_result.ok:
                    result.failures.append(
                        {
                            "case_index": index,
                            "parameters": _json_ready(point),
                            "failure_type": "VariantFailure",
                            "failure_message": row.get("failure_messages", ""),
                        }
                    )
            except Exception as exc:
                failure = {
                    "case_index": index,
                    "parameters": _json_ready(point),
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                }
                result.failures.append(failure)
                failure_row = {
                    "case": f"{spec.name}_{index:04d}",
                    "status": "failed",
                    **failure,
                    **_flatten_prefixed("parameter", point),
                }
                result._ordered_rows.append(failure_row)
                if cache_path is not None:
                    _write_json(failure_row, cache_path)
                if spec.write_incremental_jsonl and jsonl_path is not None:
                    _append_jsonl(failure_row, jsonl_path)
                if not spec.continue_on_failure:
                    raise

        if spec.write_tables and output_dir is not None:
            result.files.extend(write_sweep_result_tables(result, output_dir / _safe_file_stem(spec.name)))
        return result


def default_case_metrics(case_result: CoilCaseResult) -> dict[str, Any]:
    """Return generic metrics for every completed variant and basic comparisons."""

    metrics: dict[str, Any] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for variant in case_result.spec.variants:
        result = case_result.variant_results.get(variant.name)
        if result is None:
            continue
        prefix = _metric_prefix(variant.name)
        summary = result_summary_dict(result)
        summaries[variant.name] = summary
        quality = result_quality_status(result)
        metrics[f"{prefix}_role"] = variant.role
        metrics[f"{prefix}_quality_status"] = quality.status
        for key, value in summary.items():
            if _is_scalar_report_value(value):
                metrics[f"{prefix}_{key}"] = value

    main_name = _first_variant_name(case_result, role="main")
    reference_name = _first_variant_name(case_result, role="reference")
    if main_name and reference_name and main_name in summaries and reference_name in summaries:
        metrics.update(_comparison_metrics(summaries[main_name], summaries[reference_name], main_name, reference_name))
    return metrics


def write_case_result_tables(case_result: CoilCaseResult, path_stem: str | Path) -> list[Path]:
    path_stem = Path(path_stem)
    row = case_result.row()
    csv_path = path_stem.with_suffix(".csv")
    json_path = path_stem.with_suffix(".json")
    _write_rows_csv([row], csv_path)
    _write_json(row, json_path)
    return [csv_path, json_path]


def write_sweep_result_tables(sweep_result: ParameterSweepResult, path_stem: str | Path) -> list[Path]:
    path_stem = Path(path_stem)
    rows = sweep_result.rows()
    csv_path = path_stem.with_suffix(".csv")
    json_path = path_stem.with_suffix(".json")
    _write_rows_csv(rows, csv_path)
    _write_json(
        {
            "sweep": sweep_result.spec.name,
            "rows": rows,
            "failures": sweep_result.failures,
        },
        json_path,
    )
    return [csv_path, json_path]


def plot_case_result(
    case_result: CoilCaseResult,
    *,
    mode: str = "book",
    include: Sequence[str] | None = None,
    max_parts: int | None = None,
    close_previous: bool = True,
) -> list[Any]:
    """Open generic active Matplotlib plots for the main variant if available."""

    result = case_result.main_result
    if result is None:
        return []
    try:
        from .matplotlib_visualization import (
            activate_matplotlib_interactive_backend,
            close_matplotlib_plot_books,
            plot_result_matplotlib,
            plot_result_matplotlib_book,
        )
    except ImportError as exc:
        print(f"Active matplotlib plots skipped: {exc}")
        return []

    if close_previous:
        close_matplotlib_plot_books()
    activate_matplotlib_interactive_backend()
    plot_mode = str(mode).strip().lower()
    if plot_mode in {"book", "plot_book", "pages", "paged"}:
        return [
            plot_result_matplotlib_book(
                result,
                include=include,
                max_parts=max_parts,
                show=True,
                block=False,
            )
        ]
    if plot_mode in {"windows", "window", "figures", "separate"}:
        return plot_result_matplotlib(
            result,
            include=include,
            max_parts=max_parts,
            show=True,
            block=False,
        )
    raise ValueError("plot_mode must be 'book' or 'windows'.")


def _save_variant_reports(
    result: CoilGenResult,
    variant: CoilVariantSpec,
    spec: CoilCaseSpec,
    output_dir: Path,
    *,
    started_at_utc: str | None = None,
    duration_seconds: float | None = None,
) -> list[Path]:
    prefix = _safe_file_stem(variant.name)
    report_path = output_dir / f"{prefix}_design_report.md"
    summary_path = output_dir / f"{prefix}_design_summary.csv"
    files = [
        write_design_report(result, report_path, title=f"{variant.name} Design Report", design=variant.design),
        write_design_summary_csv(result, summary_path),
    ]
    if _save_wire_trace_enabled(result):
        files.extend(save_wire_trace_artifacts(result, output_dir, prefix=prefix))
    files.extend(
        write_variant_run_record(
            result,
            variant,
            spec,
            output_dir,
            artifact_paths=files,
            started_at_utc=started_at_utc,
            duration_seconds=duration_seconds,
        )
    )
    return files


def _save_wire_trace_enabled(result: CoilGenResult) -> bool:
    return bool(getattr(getattr(result, "input_data", None), "save_wire_trace_csv", False))


def write_variant_run_record(
    result: CoilGenResult,
    variant: CoilVariantSpec,
    case_spec: CoilCaseSpec,
    output_dir: str | Path,
    *,
    artifact_paths: Sequence[str | Path] = (),
    started_at_utc: str | None = None,
    completed_at_utc: str | None = None,
    duration_seconds: float | None = None,
) -> list[Path]:
    """Write compact JSON/CSV/JSONL/Markdown run-record artifacts for one variant."""

    output = Path(output_dir)
    prefix = _safe_file_stem(variant.name)
    record = build_variant_run_record(
        result,
        variant,
        case_spec,
        output,
        artifact_paths=artifact_paths,
        started_at_utc=started_at_utc,
        completed_at_utc=completed_at_utc,
        duration_seconds=duration_seconds,
    )
    summary_path = output / f"{prefix}_run_summary.json"
    metrics_path = output / f"{prefix}_run_metrics.csv"
    events_path = output / f"{prefix}_run_events.jsonl"
    log_path = output / f"{prefix}_run_log.md"
    _write_run_record_json(record, summary_path)
    _write_rows_csv([_bounded_run_metrics_row(record)], metrics_path)
    _write_run_record_events(record, events_path)
    _write_run_log_markdown(record, log_path)
    _prune_run_record_artifacts(output)
    return [summary_path, metrics_path, events_path, log_path]


def build_variant_run_record(
    result: CoilGenResult,
    variant: CoilVariantSpec,
    case_spec: CoilCaseSpec,
    output_dir: str | Path,
    *,
    artifact_paths: Sequence[str | Path] = (),
    started_at_utc: str | None = None,
    completed_at_utc: str | None = None,
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    """Return the compact machine-readable run record for one successful variant."""

    completed = completed_at_utc or _utc_timestamp()
    quality = result_quality_status(result)
    summary = result_summary_dict(result)
    config = _config_record(getattr(result, "input_data", None))
    design = _design_record(variant.design)
    return {
        "schema_version": RUN_RECORD_SCHEMA_VERSION,
        "run": {
            "case": case_spec.name,
            "variant": variant.name,
            "role": variant.role,
            "status": "success",
            "quality_status": quality.status,
            "started_at_utc": started_at_utc or "",
            "completed_at_utc": completed,
            "duration_seconds": _none_or_float(duration_seconds),
            "project_root": str(case_spec.project_root) if case_spec.project_root is not None else "",
            "output_dir": str(Path(output_dir)),
        },
        "case_parameters": dict(case_spec.parameters),
        "variant_metadata": dict(variant.metadata),
        "design": design,
        "config": config,
        "digests": {
            "case_parameters_sha1": _stable_digest(case_spec.parameters),
            "design_sha1": _stable_digest(design),
            "config_sha1": _stable_digest(config),
            "result_summary_sha1": _stable_digest(summary),
        },
        "quality": {
            "status": quality.status,
            "failure_count": len(quality.failures),
            "warning_count": len(quality.warnings),
            "failures": _bounded_messages(quality.failures),
            "warnings": _bounded_messages(quality.warnings),
        },
        "metrics": summary,
        "timing": _bounded_sequence(getattr(result, "timing_summary", ()) or (), RUN_RECORD_MAX_TIMING_ROWS),
        "artifacts": _bounded_sequence(
            _artifact_records(artifact_paths, output_dir=Path(output_dir)),
            RUN_RECORD_MAX_ARTIFACTS,
        ),
    }


def write_variant_failure_record(
    exc: Exception,
    variant: CoilVariantSpec,
    case_spec: CoilCaseSpec,
    output_dir: str | Path,
    *,
    started_at_utc: str | None = None,
    completed_at_utc: str | None = None,
    duration_seconds: float | None = None,
) -> list[Path]:
    """Write compact failure artifacts when a variant fails before producing a result."""

    output = Path(output_dir)
    prefix = _safe_file_stem(variant.name)
    completed = completed_at_utc or _utc_timestamp()
    design = _design_record(variant.design)
    record = {
        "schema_version": RUN_RECORD_SCHEMA_VERSION,
        "run": {
            "case": case_spec.name,
            "variant": variant.name,
            "role": variant.role,
            "status": "failed",
            "quality_status": "FAIL",
            "started_at_utc": started_at_utc or "",
            "completed_at_utc": completed,
            "duration_seconds": _none_or_float(duration_seconds),
            "project_root": str(case_spec.project_root) if case_spec.project_root is not None else "",
            "output_dir": str(output),
        },
        "case_parameters": dict(case_spec.parameters),
        "variant_metadata": dict(variant.metadata),
        "design": design,
        "digests": {
            "case_parameters_sha1": _stable_digest(case_spec.parameters),
            "design_sha1": _stable_digest(design),
        },
        "quality": {
            "status": "FAIL",
            "failure_count": 1,
            "warning_count": 0,
            "failures": [f"{type(exc).__name__}: {exc}"],
            "warnings": [],
        },
        "exception": {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": _bounded_sequence(
                traceback.format_exception(type(exc), exc, exc.__traceback__),
                RUN_RECORD_MAX_TRACEBACK_LINES,
            ),
        },
        "metrics": {},
        "timing": [],
        "artifacts": [],
    }
    summary_path = output / f"{prefix}_run_summary.json"
    metrics_path = output / f"{prefix}_run_metrics.csv"
    events_path = output / f"{prefix}_run_events.jsonl"
    log_path = output / f"{prefix}_run_log.md"
    _write_run_record_json(record, summary_path)
    _write_rows_csv([_bounded_run_metrics_row(record)], metrics_path)
    _write_run_record_events(record, events_path)
    _write_run_log_markdown(record, log_path)
    _prune_run_record_artifacts(output)
    return [summary_path, metrics_path, events_path, log_path]


def _run_metrics_row(record: Mapping[str, Any]) -> dict[str, Any]:
    run = record.get("run", {})
    quality = record.get("quality", {})
    row: dict[str, Any] = {
        "case": run.get("case", ""),
        "variant": run.get("variant", ""),
        "role": run.get("role", ""),
        "status": run.get("status", ""),
        "quality_status": run.get("quality_status", ""),
        "started_at_utc": run.get("started_at_utc", ""),
        "completed_at_utc": run.get("completed_at_utc", ""),
        "duration_seconds": run.get("duration_seconds", ""),
        "failure_count": quality.get("failure_count", 0),
        "warning_count": quality.get("warning_count", 0),
    }
    # Reserve columns for engineering decisions before optional parameters and
    # the growing diagnostic summary consume the fixed CSV budget.
    row.update(_compact_log_metrics(record.get("metrics", {})))
    row.update(_flatten_prefixed("parameter", record.get("case_parameters", {})))
    for key, value in record.get("metrics", {}).items():
        if _is_scalar_report_value(value):
            row[str(key)] = value
    return row


def _bounded_run_metrics_row(record: Mapping[str, Any]) -> dict[str, Any]:
    row = _run_metrics_row(record)
    bounded: dict[str, Any] = {}
    omitted = 0
    for index, (key, value) in enumerate(row.items()):
        if index >= RUN_RECORD_MAX_CSV_COLUMNS:
            omitted += 1
            continue
        bounded[key] = _truncate_text(_csv_ready(value), RUN_RECORD_MAX_CELL_CHARS)
    if omitted:
        bounded["run_record_omitted_metric_columns"] = omitted
    return bounded


def _run_record_events(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    run = record.get("run", {})
    timestamp = str(run.get("completed_at_utc", "")) or _utc_timestamp()
    base = {
        "timestamp_utc": timestamp,
        "case": run.get("case", ""),
        "variant": run.get("variant", ""),
    }
    events = [
        {
            **base,
            "event": "variant_finished",
            "status": run.get("status", ""),
            "quality_status": run.get("quality_status", ""),
            "duration_seconds": run.get("duration_seconds", ""),
        }
    ]
    for message in record.get("quality", {}).get("failures", ()) or ():
        events.append({**base, "event": "failure", "message": str(message)})
    for message in record.get("quality", {}).get("warnings", ()) or ():
        events.append({**base, "event": "warning", "message": str(message)})
    exception = record.get("exception", {})
    if exception:
        events.append(
            {
                **base,
                "event": "exception",
                "exception_type": exception.get("type", ""),
                "message": exception.get("message", ""),
            }
        )
    for row in record.get("timing", ()) or ():
        if isinstance(row, Mapping):
            events.append({**base, "event": "timing", **dict(row)})
    for artifact in record.get("artifacts", ()) or ():
        if isinstance(artifact, Mapping):
            events.append({**base, "event": "artifact", **dict(artifact)})
    return events


def _bounded_run_record_events(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    events = _run_record_events(record)
    if len(events) <= RUN_RECORD_MAX_EVENTS:
        return [_bounded_json_ready(event) for event in events]
    keep_tail = max(RUN_RECORD_MAX_EVENTS - 2, 0)
    run = record.get("run", {})
    marker = {
        "timestamp_utc": str(run.get("completed_at_utc", "")) or _utc_timestamp(),
        "case": run.get("case", ""),
        "variant": run.get("variant", ""),
        "event": "events_truncated",
        "omitted_event_count": len(events) - 1 - keep_tail,
    }
    selected = [events[0], marker, *events[-keep_tail:]] if keep_tail else [events[0], marker]
    return [_bounded_json_ready(event) for event in selected]


def _write_run_record_events(record: Mapping[str, Any], path: Path) -> None:
    _write_jsonl(_bounded_run_record_events(record), path)


def _write_run_log_markdown(record: Mapping[str, Any], path: Path) -> None:
    run = record.get("run", {})
    quality = record.get("quality", {})
    metrics = record.get("metrics", {})
    lines = [
        f"# CoilGen Run Log: {run.get('variant', '')}",
        "",
        f"- Case: `{run.get('case', '')}`",
        f"- Variant: `{run.get('variant', '')}` ({run.get('role', '')})",
        f"- Status: `{run.get('status', '')}` / `{run.get('quality_status', '')}`",
        f"- Started UTC: `{run.get('started_at_utc', '')}`",
        f"- Completed UTC: `{run.get('completed_at_utc', '')}`",
        f"- Duration: `{run.get('duration_seconds', '')}` s",
        f"- Output directory: `{run.get('output_dir', '')}`",
        "",
        "## Key Metrics",
        "",
    ]
    lines.extend(_markdown_key_value_table(_compact_log_metrics(metrics)))
    lines.append("")
    lines.extend(["## Warnings And Failures", ""])
    failures = list(quality.get("failures", ()) or ())
    warnings = list(quality.get("warnings", ()) or ())
    if not failures and not warnings:
        lines.append("No failures or warnings recorded.")
    else:
        lines.extend(f"- FAIL: {_truncate_text(message, RUN_RECORD_MAX_STRING_CHARS)}" for message in failures)
        lines.extend(f"- WARN: {_truncate_text(message, RUN_RECORD_MAX_STRING_CHARS)}" for message in warnings)
    if record.get("exception"):
        exception = record["exception"]
        lines.extend(["", "## Exception", ""])
        lines.append(f"- Type: `{exception.get('type', '')}`")
        lines.append(f"- Message: `{_truncate_text(exception.get('message', ''), RUN_RECORD_MAX_STRING_CHARS)}`")
    lines.extend(["", "## Digests", ""])
    lines.extend(_markdown_key_value_table(record.get("digests", {})))
    artifacts = list(record.get("artifacts", ()) or ())
    lines.extend(["", "## Artifacts", ""])
    if artifacts:
        for artifact in artifacts:
            lines.append(f"- `{artifact.get('path', '')}`")
    else:
        lines.append("No artifacts recorded.")
    atomic_text(path, _bounded_text("\n".join(lines) + "\n", RUN_RECORD_MAX_MARKDOWN_BYTES))


def _compact_log_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Select engineering metrics retained first in every compact run format."""
    keys = (
        "report_schema_version",
        "design_context",
        "part_count",
        "target_point_count",
        "dsv_target_point_count",
        "suppression_point_count",
        "regularization_method",
        "regularization_factor",
        "regularization_lambda",
        "needed_current_A",
        "layout_needed_current_A",
        "layout_gradient_efficiency_mT_per_m_per_A",
        "layout_mean_rel_error_percent",
        "layout_max_rel_error_percent",
        "stream_function_mean_rel_error_percent",
        "stream_function_max_rel_error_percent",
        "suppression_field_rms_T",
        "suppression_field_peak_T",
        "target_error_constraint_met",
        "layout_target_checked",
        "layout_target_met",
        "layout_target_metric_T",
        "layout_target_effective_limit_T",
        "constrained_solver_backend",
        "constrained_solver_name",
        "constrained_solver_status",
        "constrained_solver_contract_met",
        "constrained_solver_physical_constraints_met",
        "constrained_solver_diagnostic_result",
        "constrained_solver_diagnostic_reason",
        "constrained_solver_failed_constraints",
        "coil_length_total_m",
        "coil_resistance_total_ohm",
        "coil_inductance_same_current_H",
        "resistive_power_target_gradient_W",
        "current_density_A_per_mm2",
        "surface_current_density_max_A_per_m",
        "surface_current_density_limit_met",
        "min_contour_distance_m",
        "unconnected_contours_min_wire_distance_m",
        "min_wire_wire_distance_m",
        "route_statuses",
        "execution_time_s",
    )
    display_unit = str(metrics.get("field_display_units", "T"))
    keys += (
        f"shim_layout_residual_rms_{display_unit}",
        f"shim_layout_residual_std_{display_unit}",
        f"shim_layout_residual_max_abs_{display_unit}",
    )
    return {key: metrics[key] for key in keys if key in metrics and _is_scalar_report_value(metrics[key])}


def _markdown_key_value_table(values: Mapping[str, Any]) -> list[str]:
    lines = ["| key | value |", "| --- | --- |"]
    for key, value in values.items():
        lines.append(f"| `{key}` | `{_markdown_cell(value)}` |")
    return lines


def _markdown_cell(value: Any) -> str:
    text = json.dumps(_json_ready(value), sort_keys=True) if isinstance(value, (Mapping, list, tuple)) else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _artifact_records(paths: Sequence[str | Path], *, output_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        artifact_path = Path(path)
        try:
            relative = artifact_path.relative_to(output_dir)
        except ValueError:
            relative = artifact_path
        records.append(
            {
                "path": str(artifact_path),
                "relative_path": str(relative),
                "suffix": artifact_path.suffix.lower(),
                "exists": artifact_path.exists(),
                "size_bytes": artifact_path.stat().st_size if artifact_path.exists() else None,
            }
        )
    return records


def _config_record(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if is_dataclass(config):
        return asdict(config)
    if hasattr(config, "as_dict"):
        return dict(config.as_dict())
    if hasattr(config, "__dict__"):
        return {key: value for key, value in vars(config).items() if not key.startswith("_")}
    return {"value": str(config)}


def _design_record(design: Any) -> dict[str, Any]:
    if hasattr(design, "as_user_parameters"):
        return dict(design.as_user_parameters())
    if is_dataclass(design):
        return asdict(design)
    if hasattr(design, "__dict__"):
        return {key: value for key, value in vars(design).items() if not key.startswith("_")}
    return {"value": str(design)}


def _stable_digest(value: Any) -> str:
    # Legacy short digests also identify result summaries with missing metrics.
    # Keep their NaN tokens distinct from None; actual cache inputs are strict.
    payload = json.dumps(_json_ready(value), sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _none_or_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _comparison_metrics(
    main: Mapping[str, Any],
    reference: Mapping[str, Any],
    main_name: str,
    reference_name: str,
) -> dict[str, Any]:
    main_prefix = _metric_prefix(main_name)
    reference_prefix = _metric_prefix(reference_name)
    metrics = {
        f"{main_prefix}_to_{reference_prefix}_layout_efficiency_ratio": _safe_ratio(
            main.get("layout_gradient_efficiency_mT_per_m_per_A"),
            reference.get("layout_gradient_efficiency_mT_per_m_per_A"),
        ),
        f"{main_prefix}_to_{reference_prefix}_needed_current_ratio": _safe_ratio(
            main.get("layout_needed_current_A"),
            reference.get("layout_needed_current_A"),
        ),
        f"{main_prefix}_to_{reference_prefix}_same_current_inductance_ratio": _safe_ratio(
            main.get("coil_inductance_same_current_H"),
            reference.get("coil_inductance_same_current_H"),
        ),
    }
    active_rms = _safe_float(main.get("layout_suppressed_rms_Bz_T_at_fitted_current"))
    reference_rms = _safe_float(reference.get("layout_suppressed_rms_Bz_T_at_fitted_current"))
    if np.isfinite(active_rms) and np.isfinite(reference_rms) and reference_rms > 0.0:
        metrics[f"{main_prefix}_to_{reference_prefix}_suppressed_rms_Bz_reduction_percent"] = 100.0 * (
            1.0 - active_rms / reference_rms
        )
    return metrics


def _first_variant_name(case_result: CoilCaseResult, *, role: str) -> str | None:
    for variant in case_result.spec.variants:
        if variant.role == role:
            return variant.name
    return None


def _metric_prefix(name: str) -> str:
    return _safe_file_stem(name).lower()


def _safe_file_stem(value: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(value).strip())
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or "coil"


def _safe_ratio(numerator: Any, denominator: Any) -> float:
    numerator_value = _safe_float(numerator)
    denominator_value = _safe_float(denominator)
    if not np.isfinite(numerator_value) or not np.isfinite(denominator_value) or denominator_value == 0.0:
        return float("nan")
    return float(numerator_value / denominator_value)


def _safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _is_scalar_report_value(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)) or value is None


def _flatten_prefixed(prefix: str, values: Mapping[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in values.items():
        path = f"{prefix}.{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten_prefixed(path, value))
        elif _is_scalar_report_value(value):
            flat[path] = value
        else:
            flat[path] = _json_ready(value)
    return flat


def _write_rows_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with atomic_text_writer(path, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_ready(row.get(key, "")) for key in keys})


def _write_json(payload: Any, path: Path) -> None:
    atomic_text(path, json.dumps(json_value(_json_ready(payload)), indent=2, sort_keys=True, allow_nan=False))


def _write_run_record_json(record: Mapping[str, Any], path: Path) -> None:
    metrics = record.get("metrics", {})
    prioritized_record = {**record, "metrics": {**_compact_log_metrics(metrics), **metrics}}
    compact = _bounded_json_ready(prioritized_record)
    text = json.dumps(json_value(compact), indent=2, sort_keys=True, allow_nan=False)
    if len(text.encode("utf-8")) > RUN_RECORD_MAX_SUMMARY_BYTES:
        compact = _minimal_run_record(record, reason="summary exceeded run-record byte budget")
        text = json.dumps(json_value(compact), indent=2, sort_keys=True, allow_nan=False)
    if len(text.encode("utf-8")) > RUN_RECORD_MAX_SUMMARY_BYTES:
        compact = _minimal_run_record(record, reason="minimal summary exceeded run-record byte budget")
        compact["metrics"] = {}
        compact["timing"] = []
        compact["artifacts"] = []
        text = json.dumps(json_value(compact), indent=2, sort_keys=True, allow_nan=False)
    atomic_text(path, text)


def _minimal_run_record(record: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    exception = record.get("exception", {})
    minimal: dict[str, Any] = {
        "schema_version": record.get("schema_version", RUN_RECORD_SCHEMA_VERSION),
        "run": _bounded_json_ready(record.get("run", {})),
        "quality": _bounded_json_ready(record.get("quality", {})),
        "digests": _bounded_json_ready(record.get("digests", {})),
        "metrics": _bounded_json_ready(_compact_log_metrics(record.get("metrics", {}))),
        "timing": _bounded_json_ready(_bounded_sequence(record.get("timing", ()) or (), 20)),
        "artifacts": _bounded_json_ready(_bounded_sequence(record.get("artifacts", ()) or (), 20)),
        "run_record_truncation": {
            "truncated": True,
            "reason": reason,
            "summary_byte_budget": RUN_RECORD_MAX_SUMMARY_BYTES,
        },
    }
    if exception:
        minimal["exception"] = {
            "type": exception.get("type", ""),
            "message": _truncate_text(exception.get("message", ""), RUN_RECORD_MAX_STRING_CHARS),
        }
    return minimal


def _bounded_json_ready(value: Any, *, depth: int = 0) -> Any:
    if depth > 12:
        return _truncate_text(value, RUN_RECORD_MAX_STRING_CHARS)
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        omitted = 0
        for index, (key, item) in enumerate(value.items()):
            if index >= RUN_RECORD_MAX_MAPPING_ITEMS:
                omitted += 1
                continue
            output[str(key)] = _bounded_json_ready(item, depth=depth + 1)
        if omitted:
            output["__omitted_items__"] = omitted
        return output
    if isinstance(value, (list, tuple)):
        sequence = _bounded_sequence(value, RUN_RECORD_MAX_SEQUENCE_ITEMS)
        return [_bounded_json_ready(item, depth=depth + 1) for item in sequence]
    if isinstance(value, np.ndarray):
        if value.size > RUN_RECORD_MAX_SEQUENCE_ITEMS:
            return {
                "__ndarray__": True,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "omitted_values": int(value.size),
            }
        return _bounded_json_ready(value.tolist(), depth=depth + 1)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, str):
        return _truncate_text(value, RUN_RECORD_MAX_STRING_CHARS)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _truncate_text(value, RUN_RECORD_MAX_STRING_CHARS)


def _bounded_sequence(values: Sequence[Any], limit: int) -> list[Any]:
    items = list(values)
    if len(items) <= int(limit):
        return items
    if limit <= 1:
        return [{"__omitted_items__": len(items)}]
    tail_count = max(int(limit) - 2, 0)
    marker = {"__omitted_items__": len(items) - 1 - tail_count}
    return [items[0], marker, *items[-tail_count:]] if tail_count else [items[0], marker]


def _bounded_messages(messages: Sequence[Any]) -> list[str]:
    return [
        _truncate_text(message, RUN_RECORD_MAX_STRING_CHARS)
        for message in _bounded_sequence(tuple(messages), RUN_RECORD_MAX_MESSAGES)
    ]


def _truncate_text(value: Any, max_chars: int) -> str:
    text = str(value)
    if len(text) <= int(max_chars):
        return text
    omitted = len(text) - int(max_chars)
    suffix = f"... [truncated {omitted} chars]"
    keep = max(int(max_chars) - len(suffix), 0)
    return text[:keep] + suffix


def _bounded_text(text: str, max_bytes: int) -> str:
    if len(text.encode("utf-8")) <= int(max_bytes):
        return text
    suffix = f"\n\n[run record truncated to {max_bytes} bytes]\n"
    payload_budget = max(int(max_bytes) - len(suffix.encode("utf-8")), 0)
    encoded = text.encode("utf-8")[:payload_budget]
    return encoded.decode("utf-8", errors="ignore").rstrip() + suffix


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_jsonl(payload: Any, path: Path) -> None:
    # Serialize before opening: an invalid row cannot leave a partial JSON value.
    # Incremental events remain a streaming log; resume uses atomic case files.
    text = json.dumps(json_value(_json_ready(payload)), sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def _write_jsonl(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    with atomic_text_writer(path) as handle:
        for row in rows:
            handle.write(json.dumps(json_value(_json_ready(row)), sort_keys=True, allow_nan=False) + "\n")


def _prune_run_record_artifacts(output_dir: Path, *, keep: int = RUN_RECORD_RETENTION_COUNT) -> None:
    if int(keep) < 1 or not output_dir.exists():
        return
    groups: dict[str, list[Path]] = {}
    for suffix in RUN_RECORD_FILE_SUFFIXES:
        for path in output_dir.glob(f"*{suffix}"):
            prefix = path.name[: -len(suffix)]
            groups.setdefault(prefix, []).append(path)
    if len(groups) <= int(keep):
        return

    def group_mtime(paths: Sequence[Path]) -> float:
        mtimes: list[float] = []
        for path in paths:
            try:
                mtimes.append(path.stat().st_mtime)
            except OSError:
                continue
        return max(mtimes) if mtimes else 0.0

    def group_sort_key(item: tuple[str, list[Path]]) -> tuple[str, float]:
        prefix, paths = item
        summary_path = output_dir / f"{prefix}_run_summary.json"
        completed = ""
        try:
            payload = _read_json(summary_path)
            completed = str(payload.get("run", {}).get("completed_at_utc", ""))
        except (OSError, json.JSONDecodeError, AttributeError):
            completed = ""
        return completed, group_mtime(paths)

    ordered = sorted(groups.items(), key=group_sort_key, reverse=True)
    for prefix, paths in ordered[int(keep) :]:
        candidates = {path for path in paths}
        candidates.update(output_dir / f"{prefix}{suffix}" for suffix in RUN_RECORD_FILE_SUFFIXES)
        for path in candidates:
            try:
                if path.is_file():
                    path.unlink()
            except OSError:
                pass


def _successful_cached_row(path: Path) -> dict[str, Any] | None:
    """Incomplete writes and failed attempts must be retried on resume."""

    try:
        row = _read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(row, dict) or row.get("status") != "success" or row.get("failure_count") != 0:
        return None
    return row


def _sweep_code_digest() -> str:
    return source_digest(Path(__file__).resolve().parent)


def _sweep_cache_context(spec: ParameterSweepSpec, case: CoilCaseSpec, code_digest: str) -> dict[str, Any]:
    """Fingerprint resolved designs and standard inputs before reusing metrics.

    Custom factory/metric dependencies and external solver changes must be
    identified by ``cache_revision``. The factory is always called on resume.
    """

    root = Path.cwd() if case.project_root is None else Path(case.project_root)
    variants = []
    for variant in case.variants:
        config = variant.design.to_config() if hasattr(variant.design, "to_config") else None
        variants.append(
            {
                "name": variant.name,
                "role": variant.role,
                "design_type": f"{type(variant.design).__module__}.{type(variant.design).__qualname__}",
                "design": _design_record(variant.design),
                "config": _config_record(config),
                "inputs": _sweep_input_digests(config, root) if config is not None else {},
                "metadata": variant.metadata,
            }
        )
    return {
        "schema": "coilgen_case_cache_v2",
        "sweep": spec.name,
        "revision": spec.cache_revision,
        "code": code_digest,
        "case": case.name,
        "parameters": case.parameters,
        "project_root": root.resolve(),
        "variants": variants,
        "metric_functions": [
            f"{getattr(function, '__module__', type(function).__module__)}."
            f"{getattr(function, '__qualname__', type(function).__qualname__)}"
            for function in case.metric_functions
        ],
    }


def _sweep_input_digests(config: Any, root: Path) -> dict[str, Any]:
    from .object_registry import (
        get_in_memory_mesh,
        get_in_memory_target_field,
        is_in_memory_mesh_key,
        is_in_memory_target_key,
    )
    from .preoptimization import _resolve_preoptimized_path, preoptimized_stream_function_source

    inputs: dict[str, Any] = {}
    sources = {
        key: str(getattr(config, key, "none"))
        for key in ("coil_mesh_file", "target_mesh_file", "secondary_target_mesh_file", "target_field_definition_file")
    }
    sources["preoptimized_stream_function_file"] = preoptimized_stream_function_source(config)
    for key, source in sources.items():
        if source.lower() == "none" or not source or source.lower().startswith("create "):
            continue
        if is_in_memory_mesh_key(source):
            mesh = get_in_memory_mesh(source)
            inputs[key] = _stable_digest({"vertices": mesh.vertices, "faces": mesh.faces})
            continue
        if is_in_memory_target_key(source):
            inputs[key] = _stable_digest(asdict(get_in_memory_target_field(source)))
            continue
        path = Path(source)
        if key.endswith("mesh_file") and hasattr(config, "resolved_geometry_source_path"):
            path = config.resolved_geometry_source_path(root) / path
        elif key == "preoptimized_stream_function_file":
            path = _resolve_preoptimized_path(source, root)
        if not path.is_file():
            inputs[key] = {"path": str(path.resolve()), "missing": True}
            continue
        inputs[key] = {"path": str(path.resolve()), "sha256": file_digest(path)}
    return inputs


def _case_cache_path(cache_dir: Path | None, index: int, point: Mapping[str, Any]) -> Path | None:
    if cache_dir is None:
        return None
    payload = json.dumps(_json_ready(point), sort_keys=True, allow_nan=False)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"case_{index:04d}_{digest}.json"


def _csv_ready(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(_json_ready(value), sort_keys=True)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
