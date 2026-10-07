from __future__ import annotations

import copy
import csv
import json
import math
import time
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .design_context import CoilDesignContext, EvaluationFeature, design_context_key, normalize_design_context
from .engineering_metrics import drive_metrics_from_inductance
from .gradient_linearity import GradientLinearitySpec, gradient_linearity_spec_from_value
from .gradient_slice_maps import GradientSliceMapSpec, gradient_slice_map_spec_from_value
from .parameters import CoilProjectParameters
from .scientist_api import CoilDesignRunArtifacts, CoilDesignRunOptions, run_coil_design


RunDesignFunction = Callable[[Any, CoilDesignRunOptions], CoilDesignRunArtifacts]

PROJECT_SUMMARY_KEYS = (
    "design_context",
    "field_display_units",
    "dsv_target_point_count",
    "contour_discretization_method",
    "max_wire_current_A",
    "min_wire_distance_m",
    "target_gradient_requested_T_per_m",
    "target_gradient_hardware_effective_T_per_m",
    "gradient_hardware_feasible",
    "gradient_hardware_scale_factor",
    "linearity_status",
    "linearity_rms_percent",
    "linearity_rms_tolerance_percent",
    "linearity_peak_percent",
    "linearity_peak_tolerance_percent",
    "linearity_cross_gradient_percent",
    "linearity_cross_gradient_tolerance_percent",
    "linearity_gradient_strength_error_percent",
    "linearity_gradient_strength_tolerance_percent",
    "gradient_slice_maps_status",
    "gradient_slice_map_linearity_rms_percent",
    "gradient_slice_map_linearity_peak_percent",
    "gradient_slice_map_concomitant_peak_abs_T",
    "surface_current_density_limit_met",
    "surface_current_density_limit_A_per_m",
    "surface_current_density_max_A_per_m",
    "contour_count",
    "needed_current_A",
    "stream_function_needed_current_A",
    "stream_function_gradient_efficiency_mT_per_m_per_A",
    "stream_function_mean_rel_error_percent",
    "stream_function_max_rel_error_percent",
    "unconnected_contours_needed_current_A",
    "unconnected_contours_gradient_efficiency_mT_per_m_per_A",
    "unconnected_contours_mean_rel_error_percent",
    "unconnected_contours_max_rel_error_percent",
    "layout_needed_current_A",
    "layout_gradient_efficiency_mT_per_m_per_A",
    "layout_mean_rel_error_percent",
    "layout_max_rel_error_percent",
    "shim_layout_residual_rms_T",
    "shim_layout_residual_max_abs_T",
    "shim_layout_residual_peak_to_peak_T",
    "shim_layout_residual_rms_Hz",
    "shim_layout_residual_max_abs_Hz",
    "shim_layout_residual_peak_to_peak_Hz",
    "coil_length_total_m",
    "coil_resistance_total_ohm",
    "coil_inductance_same_current_H",
    "min_contour_distance_m",
    "unconnected_contours_min_wire_distance_m",
    "target_error_tolerance_percent",
    "target_error_tolerance_absolute_T",
    "min_wire_wire_distance_m",
    "route_statuses",
)

PROJECT_COMPARISON_METRICS = (
    ("layout_needed_current_A", "Needed current", "A"),
    ("layout_gradient_efficiency_mT_per_m_per_A", "Gradient efficiency", "mT/m/A"),
    ("layout_mean_rel_error_percent", "Mean rel. error", "%"),
    ("layout_max_rel_error_percent", "Max rel. error", "%"),
    ("linearity_rms_percent", "Linearity RMS", "%"),
    ("linearity_peak_percent", "Linearity peak", "%"),
    ("linearity_cross_gradient_percent", "Cross gradient", "%"),
    ("coil_resistance_total_ohm", "Resistance", "Ohm"),
    ("coil_inductance_same_current_H", "Inductance", "H"),
)

PROJECT_GENERIC_COMPARISON_METRICS = (
    ("layout_needed_current_A", "Needed current", "A"),
    ("layout_mean_rel_error_percent", "Mean rel. error", "%"),
    ("layout_max_rel_error_percent", "Max rel. error", "%"),
    ("coil_resistance_total_ohm", "Resistance", "Ohm"),
    ("coil_inductance_same_current_H", "Inductance", "H"),
)


@dataclass(frozen=True, slots=True)
class CoilChannelSpec:
    """One named design channel inside a multi-channel CoilGen project.

    A channel resolves to an ordinary user-facing design object before running.
    For a 3-axis gradient project, the channels are typically ``Gx``, ``Gy``,
    and ``Gz`` with only ``target_bz_expression`` differing between them.
    """

    name: str
    target_bz_expression: str | None = None
    target_scale: float | None = None
    design: Any | None = None
    design_overrides: Mapping[str, Any] = field(default_factory=dict)
    label: str = ""
    role: str = "channel"
    color: str = ""
    enabled: bool = True
    save_reports: bool | None = None
    save_gallery: bool | None = None
    design_context: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("CoilChannelSpec.name must be non-empty.")
        object.__setattr__(self, "design_overrides", dict(self.design_overrides))
        if self.design_context is not None:
            object.__setattr__(self, "design_context", design_context_key(self.design_context))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def display_label(self) -> str:
        return str(self.label or self.name)

    @property
    def safe_name(self) -> str:
        return _safe_file_stem(self.name)

    def resolve_design(
        self,
        template: Any | None,
        *,
        project_name: str,
        project_overrides: Mapping[str, Any] | None = None,
        project_parameters: CoilProjectParameters | None = None,
        project_design_context: Any | None = None,
    ) -> Any:
        """Return this channel as an executable design object."""

        source = self.design if self.design is not None else template
        if source is None:
            raise ValueError(f"Channel {self.name!r} has no explicit design and the project has no template design.")

        overrides = {}
        if project_parameters is not None:
            overrides.update(project_parameters.to_design_overrides(source))
        if project_design_context is not None and _design_supports_override(source, "design_context"):
            overrides["design_context"] = design_context_key(project_design_context)
        overrides.update(project_overrides or {})
        if self.design_context is not None and _design_supports_override(source, "design_context"):
            overrides["design_context"] = design_context_key(self.design_context)
        overrides.update(self.design_overrides)
        if self.target_bz_expression is not None:
            overrides["target_bz_expression"] = str(self.target_bz_expression)
        if self.target_scale is not None:
            overrides["target_scale"] = float(self.target_scale)
        if self.design is None and "name" not in overrides and hasattr(source, "name"):
            overrides["name"] = f"{_safe_file_stem(project_name)}_{self.safe_name}"
        return _copy_design_with_overrides(source, overrides, channel_name=self.name)

    def row(self, design: Any | None = None) -> dict[str, Any]:
        """Return a compact setup row for this channel."""

        row = {
            "channel": self.name,
            "label": self.display_label,
            "role": self.role,
            "enabled": bool(self.enabled),
            "color": self.color,
        }
        if self.design_context is not None:
            row["design_context"] = self.design_context
        if self.save_reports is not None:
            row["save_reports"] = bool(self.save_reports)
        if self.save_gallery is not None:
            row["save_gallery"] = bool(self.save_gallery)
        if self.target_bz_expression is not None:
            row["target_bz_expression"] = self.target_bz_expression
        if self.target_scale is not None:
            row["target_scale"] = self.target_scale
        if design is not None:
            row.update(_design_setup_row(design))
        row.update(_flatten_prefixed("metadata", self.metadata))
        return row


CoilChannel = CoilChannelSpec


@dataclass(slots=True)
class CoilChannelRun:
    """Run artifacts for one project channel."""

    channel: CoilChannelSpec
    design: Any
    artifacts: CoilDesignRunArtifacts | None = None
    files: tuple[Path, ...] = field(default_factory=tuple)
    figures: tuple[Any, ...] = field(default_factory=tuple)
    exception: Exception | None = None

    @property
    def ok(self) -> bool:
        return self.exception is None and self.artifacts is not None

    @property
    def result(self) -> Any | None:
        return None if self.artifacts is None else getattr(self.artifacts, "result", None)

    @property
    def quality(self) -> Any | None:
        return None if self.artifacts is None else getattr(self.artifacts, "quality", None)

    def summary_row(self) -> dict[str, Any]:
        """Return one machine-readable row for project summary tables."""

        row = {
            "channel": self.channel.name,
            "label": self.channel.display_label,
            "role": self.channel.role,
            "status": "failed" if self.exception is not None else "success",
            "design_name": getattr(self.design, "name", ""),
            "target_bz_expression": getattr(
                self.design, "target_bz_expression", self.channel.target_bz_expression or ""
            ),
            "target_scale": getattr(
                self.design, "target_scale", self.channel.target_scale if self.channel.target_scale is not None else ""
            ),
        }
        row.update(_design_setup_row(self.design))
        if self.exception is not None:
            row["quality_status"] = "failed"
            row["failure_type"] = type(self.exception).__name__
            row["failure_message"] = str(self.exception)
            return row

        result = self.result
        if result is None:
            row["quality_status"] = getattr(self.quality, "status", "")
            return row

        from .reporting import result_quality_status, result_summary_dict

        quality = self.quality or result_quality_status(result)
        summary = result_summary_dict(result)
        row["quality_status"] = quality.status
        for key in PROJECT_SUMMARY_KEYS:
            row[key] = summary.get(key, "")
        row.update(_project_drive_metrics_from_summary(row))
        return row


@dataclass(slots=True)
class CoilProjectRun:
    """Result of running a multi-channel CoilGen project."""

    project: "CoilProject"
    output_dir: Path
    channel_runs: tuple[CoilChannelRun, ...]
    summary_rows: tuple[dict[str, Any], ...]
    files: tuple[Path, ...] = field(default_factory=tuple)
    figures: tuple[Any, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return all(channel_run.ok for channel_run in self.channel_runs)

    @property
    def failures(self) -> dict[str, Exception]:
        return {
            channel_run.channel.name: channel_run.exception
            for channel_run in self.channel_runs
            if channel_run.exception is not None
        }

    @property
    def results(self) -> dict[str, Any]:
        return {
            channel_run.channel.name: channel_run.result
            for channel_run in self.channel_runs
            if channel_run.result is not None
        }

    def set_channel_result(self, channel: str, result: Any) -> CoilChannelRun:
        """Replace one channel result after deliberate post-processing.

        This keeps project plotting and summaries tied to the current result
        when a script applies a domain-specific transformation after solving.
        """

        key = str(channel).strip().lower()
        for channel_run in self.channel_runs:
            if channel_run.channel.name.strip().lower() != key:
                continue
            if channel_run.artifacts is None:
                raise ValueError(f"Channel {channel!r} has no successful artifacts to update.")
            channel_run.artifacts = replace(channel_run.artifacts, result=result)
            self.summary_rows = tuple(run.summary_row() for run in self.channel_runs)
            return channel_run
        raise ValueError(f"Unknown project channel: {channel!r}.")

    def write_summary(self, output_dir: Path | str | None = None, *, prefix: str | None = None) -> tuple[Path, ...]:
        """Write or rewrite this run's project summary files."""

        files = write_project_summary(self, output_dir=output_dir, prefix=prefix)
        existing = list(self.files)
        for file in files:
            if file not in existing:
                existing.append(file)
        self.files = tuple(existing)
        return files

    def show_plot_book(
        self,
        *,
        include: Sequence[str] | None = None,
        max_parts: int | None = None,
        close_previous: bool | None = None,
        show: bool = True,
        block: bool = False,
        preload_pages: str | bool = "idle",
    ) -> Any | None:
        """Open an interactive project plot book for this run."""

        return plot_project_run_book(
            self,
            include=include,
            max_parts=max_parts,
            close_previous=self.project.close_previous_plots if close_previous is None else close_previous,
            show=show,
            block=block,
            preload_pages=preload_pages,
        )

    def plot(
        self,
        *,
        include: Sequence[str] | None = None,
        max_parts: int | None = None,
        close_previous: bool | None = None,
        mode: str = "book",
        show: bool = True,
        block: bool = False,
        channels: Sequence[str] | str | None = None,
        extra_figures: Sequence[Any] | None = None,
        extra_names: Sequence[str] | None = None,
        extra_categories: Sequence[str] | None = None,
        extra_specs: Sequence[Any] | None = None,
        preload_pages: str | bool = "idle",
    ) -> list[Any]:
        """Plot this project run and return the opened figures or plot books."""

        return self.project.plot(
            self,
            include=include,
            max_parts=max_parts,
            close_previous=close_previous,
            mode=mode,
            show=show,
            block=block,
            channels=channels,
            extra_figures=extra_figures,
            extra_names=extra_names,
            extra_categories=extra_categories,
            extra_specs=extra_specs,
            preload_pages=preload_pages,
        )


@dataclass(slots=True)
class CoilProject:
    """User-facing container for a multi-channel CoilGen design project.

    The project owns shared defaults, channel-specific overrides, output
    structure, summary tables, and combined plot books. Each channel still runs
    through the existing single-design CoilGen pipeline.
    """

    name: str
    template: Any | None = None
    design_overrides: Mapping[str, Any] = field(default_factory=dict)
    channels: list[CoilChannelSpec] = field(default_factory=list)
    project_root: Path | str | None = None
    output_dir: Path | str | None = None
    save_reports: bool = True
    save_gallery: bool = False
    write_summary: bool = True
    open_plot_book: bool = False
    close_previous_plots: bool = True
    plot_include: Sequence[str] | None = None
    max_plot_parts: int | None = None
    continue_on_channel_failure: bool = False
    parameters: CoilProjectParameters | Mapping[str, Any] = field(default_factory=CoilProjectParameters)
    design_context: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    last_run: CoilProjectRun | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("CoilProject.name must be non-empty.")
        if self.design_context is not None:
            self.design_context = design_context_key(self.design_context)
        self.design_overrides = dict(self.design_overrides)
        self.channels = list(self.channels)
        if not isinstance(self.parameters, CoilProjectParameters):
            self.parameters = CoilProjectParameters.from_mapping(self.parameters)
        self.metadata = dict(self.metadata)
        self._validate_channel_names()

    @property
    def safe_name(self) -> str:
        return _safe_file_stem(self.name)

    def add_channel(
        self,
        name: str,
        *,
        target_bz_expression: str | None = None,
        target_scale: float | None = None,
        design: Any | None = None,
        design_overrides: Mapping[str, Any] | None = None,
        label: str = "",
        role: str = "channel",
        color: str = "",
        design_context: Any | None = None,
        enabled: bool = True,
        save_reports: bool | None = None,
        save_gallery: bool | None = None,
        metadata: Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> CoilChannelSpec:
        """Append one channel and return its spec.

        Keyword arguments not consumed by the channel itself are treated as
        design overrides, so scripts can write
        ``project.channel("Gx", target_bz_expression="x", contour_levels=10)``.
        """

        merged_overrides = dict(design_overrides or {})
        merged_overrides.update(overrides)
        channel = CoilChannelSpec(
            name=name,
            target_bz_expression=target_bz_expression,
            target_scale=target_scale,
            design=design,
            design_overrides=merged_overrides,
            label=label,
            role=role,
            color=color,
            design_context=design_context,
            enabled=enabled,
            save_reports=save_reports,
            save_gallery=save_gallery,
            metadata={} if metadata is None else dict(metadata),
        )
        self.channels.append(channel)
        self._validate_channel_names()
        return channel

    def channel(
        self,
        name: str,
        *,
        target_bz_expression: str | None = None,
        target_scale: float | None = None,
        design: Any | None = None,
        design_overrides: Mapping[str, Any] | None = None,
        label: str = "",
        role: str = "channel",
        color: str = "",
        design_context: Any | None = None,
        enabled: bool = True,
        save_reports: bool | None = None,
        save_gallery: bool | None = None,
        metadata: Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> CoilChannelSpec:
        """Create one project channel.

        This is the preferred user-facing spelling in design scripts. It is
        equivalent to ``add_channel`` and returns the channel object for IDE
        inspection or later mutation by replacing it in ``project.channels``.
        """

        return self.add_channel(
            name,
            target_bz_expression=target_bz_expression,
            target_scale=target_scale,
            design=design,
            design_overrides=design_overrides,
            label=label,
            role=role,
            color=color,
            design_context=design_context,
            enabled=enabled,
            save_reports=save_reports,
            save_gallery=save_gallery,
            metadata=metadata,
            **overrides,
        )

    def gradient_channel(
        self,
        name: str,
        *,
        axis: str,
        gradient_strength_T_per_m: float,
        gradient_strength_tolerance_percent: float | None = None,
        linearity: GradientLinearitySpec | Mapping[str, Any] | None = None,
        slice_maps: GradientSliceMapSpec | Mapping[str, Any] | None = None,
        max_wire_current_A: float | None = None,
        min_wire_distance_m: float | None = None,
        surface_current_density_limit_A_per_m: float | None = None,
        engineering_contour_step_safety_factor: float = 1.0,
        engineering_contour_max_levels: int | None = None,
        contour_levels: int | None = None,
        gradient_hardware_infeasible_policy: str | None = None,
        design: Any | None = None,
        design_overrides: Mapping[str, Any] | None = None,
        label: str = "",
        role: str = "gradient",
        color: str = "",
        enabled: bool = True,
        save_reports: bool | None = None,
        save_gallery: bool | None = None,
        metadata: Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> CoilChannelSpec:
        """Create an MR gradient channel with physical target and hardware settings."""

        gradient_axis = _normalize_gradient_axis(axis)
        merged_overrides = dict(design_overrides or {})
        merged_overrides.update(overrides)
        merged_overrides.setdefault("target_gradient_normalization", "linear_fit")
        merged_overrides.setdefault("target_gradient_axis", gradient_axis)
        merged_overrides.setdefault(
            "engineering_contour_step_safety_factor", float(engineering_contour_step_safety_factor)
        )
        if linearity is not None:
            spec = gradient_linearity_spec_from_value(linearity)
            merged_overrides.update(spec.to_config_overrides())
        if slice_maps is not None:
            merged_overrides["slice_maps"] = gradient_slice_map_spec_from_value(slice_maps)
        if gradient_strength_tolerance_percent is not None:
            merged_overrides["gradient_strength_tolerance_percent"] = float(gradient_strength_tolerance_percent)
        if max_wire_current_A is not None:
            merged_overrides["max_current_A"] = float(max_wire_current_A)
            merged_overrides["max_wire_current_A"] = float(max_wire_current_A)
        if min_wire_distance_m is not None:
            merged_overrides["min_wire_distance_m"] = float(min_wire_distance_m)
        if surface_current_density_limit_A_per_m is not None:
            merged_overrides["surface_current_density_limit_A_per_m"] = float(surface_current_density_limit_A_per_m)
        if contour_levels is not None:
            merged_overrides["contour_levels"] = int(contour_levels)
        if engineering_contour_max_levels is not None:
            merged_overrides["engineering_contour_max_levels"] = int(engineering_contour_max_levels)
        elif contour_levels is not None:
            merged_overrides.setdefault("engineering_contour_max_levels", int(contour_levels))
        if gradient_hardware_infeasible_policy is not None:
            merged_overrides["gradient_hardware_infeasible_policy"] = str(gradient_hardware_infeasible_policy)
        if max_wire_current_A is not None and min_wire_distance_m is not None:
            merged_overrides.setdefault("contour_discretization_method", "engineering")
            merged_overrides.setdefault("field_evaluation_current_source", "stream_function")
        return self.channel(
            name,
            target_bz_expression=gradient_axis,
            target_scale=float(gradient_strength_T_per_m),
            design=design,
            design_overrides=merged_overrides,
            label=label,
            role=role,
            color=color,
            design_context="mr_gradient",
            enabled=enabled,
            save_reports=save_reports,
            save_gallery=save_gallery,
            metadata=metadata,
        )

    def enabled_channels(self) -> tuple[CoilChannelSpec, ...]:
        return tuple(channel for channel in self.channels if channel.enabled)

    def resolved_designs(self, *, include_disabled: bool = False) -> dict[str, Any]:
        """Return channel names mapped to resolved design objects."""

        self._validate_channel_names()
        selected = self.channels if include_disabled else self.enabled_channels()
        return {
            channel.name: channel.resolve_design(
                self.template,
                project_name=self.name,
                project_overrides=self.design_overrides,
                project_parameters=self.parameters,
                project_design_context=self.design_context,
            )
            for channel in selected
        }

    def channel_rows(self, *, include_disabled: bool = True) -> list[dict[str, Any]]:
        """Return setup rows that are useful in Spyder before running."""

        selected = self.channels if include_disabled else self.enabled_channels()
        rows: list[dict[str, Any]] = []
        for channel in selected:
            design = None
            if channel.enabled or include_disabled:
                try:
                    design = channel.resolve_design(
                        self.template,
                        project_name=self.name,
                        project_overrides=self.design_overrides,
                        project_parameters=self.parameters,
                        project_design_context=self.design_context,
                    )
                except Exception:
                    design = None
            rows.append(channel.row(design))
        return rows

    def parameter_table(self) -> list[dict[str, Any]]:
        """Alias for ``channel_rows`` for user-facing scripts."""

        return self.channel_rows(include_disabled=True)

    def describe(self) -> str:
        """Return a compact text summary of the project setup."""

        rows = self.channel_rows(include_disabled=True)
        lines = [
            f"CoilProject: {self.name}",
            f"Channels: {len(self.enabled_channels())} enabled / {len(self.channels)} total",
            f"Template: {type(self.template).__name__ if self.template is not None else '<per-channel designs>'}",
            f"Design context: {self.design_context if self.design_context is not None else '<per-design>'}",
            "",
            "Channel setup:",
        ]
        if not rows:
            lines.append("- <no channels>")
            return "\n".join(lines)
        for row in rows:
            parts = [
                f"{row.get('channel', '')}",
                f"target={row.get('target_bz_expression', '')}",
                f"scale={row.get('target_scale', '')}",
                f"design={row.get('design_class', '')}",
            ]
            if not row.get("enabled", True):
                parts.append("disabled")
            lines.append("- " + ", ".join(str(part) for part in parts if str(part)))
        return "\n".join(lines)

    def validate(self, *, project_root: Path | str | None = None) -> dict[str, Any]:
        """Validate every enabled channel and return reports by channel name."""

        root = Path(project_root) if project_root is not None else self._project_root()
        reports: dict[str, Any] = {}
        for name, design in self.resolved_designs().items():
            if not hasattr(design, "validate"):
                continue
            reports[name] = design.validate(project_root=root)
        return reports

    def run(
        self,
        *,
        project_root: Path | str | None = None,
        output_dir: Path | str | None = None,
        save_reports: bool | None = None,
        save_gallery: bool | None = None,
        write_summary: bool | None = None,
        open_plot_book: bool | None = None,
        close_previous_plots: bool | None = None,
        plot_include: Sequence[str] | None = None,
        max_plot_parts: int | None = None,
        continue_on_channel_failure: bool | None = None,
        runner: RunDesignFunction | None = None,
    ) -> CoilProjectRun:
        """Run every enabled project channel."""

        root = Path(project_root) if project_root is not None else self._project_root()
        output = Path(output_dir) if output_dir is not None else self._output_dir(root)
        output.mkdir(parents=True, exist_ok=True)
        do_save_reports = self.save_reports if save_reports is None else bool(save_reports)
        do_save_gallery = self.save_gallery if save_gallery is None else bool(save_gallery)
        do_write_summary = self.write_summary if write_summary is None else bool(write_summary)
        do_open_plot_book = self.open_plot_book if open_plot_book is None else bool(open_plot_book)
        do_close_previous = self.close_previous_plots if close_previous_plots is None else bool(close_previous_plots)
        include = self.plot_include if plot_include is None else plot_include
        max_parts = self.max_plot_parts if max_plot_parts is None else max_plot_parts
        keep_going = (
            self.continue_on_channel_failure
            if continue_on_channel_failure is None
            else bool(continue_on_channel_failure)
        )
        run_design = runner or run_coil_design

        channel_runs: list[CoilChannelRun] = []
        saved_files: list[Path] = []
        for channel in self.enabled_channels():
            channel_save_reports = do_save_reports if channel.save_reports is None else bool(channel.save_reports)
            channel_save_gallery = do_save_gallery if channel.save_gallery is None else bool(channel.save_gallery)
            design = channel.resolve_design(
                self.template,
                project_name=self.name,
                project_overrides=self.design_overrides,
                project_parameters=self.parameters,
                project_design_context=self.design_context,
            )
            channel_dir = output / channel.safe_name
            output_prefix = _safe_file_stem(getattr(design, "name", f"{self.safe_name}_{channel.safe_name}"))
            try:
                artifacts = run_design(
                    design,
                    CoilDesignRunOptions(
                        project_root=root,
                        output_dir=channel_dir,
                        output_prefix=output_prefix,
                        save_reports=channel_save_reports,
                        open_plot_book=False,
                        close_previous_plots=False,
                        plot_include=include,
                        max_plot_parts=max_parts,
                    ),
                )
                files = list(getattr(artifacts, "files", ()) or ())
                result = getattr(artifacts, "result", None)
                if channel_save_gallery and result is not None:
                    from .visualization import save_plot_gallery

                    files.extend(save_plot_gallery(result, channel_dir, prefix=output_prefix))
                channel_run = CoilChannelRun(channel=channel, design=design, artifacts=artifacts, files=tuple(files))
                saved_files.extend(files)
            except Exception as exc:
                channel_run = CoilChannelRun(channel=channel, design=design, exception=exc)
                if not keep_going:
                    raise
            channel_runs.append(channel_run)

        summary_rows = tuple(channel_run.summary_row() for channel_run in channel_runs)
        project_run = CoilProjectRun(
            project=self,
            output_dir=output,
            channel_runs=tuple(channel_runs),
            summary_rows=summary_rows,
            files=tuple(saved_files),
        )
        if do_write_summary:
            summary_files = write_project_summary(project_run)
            saved_files.extend(summary_files)
            project_run.files = tuple(saved_files)
        if do_open_plot_book:
            book = plot_project_run_book(
                project_run,
                include=include,
                max_parts=max_parts,
                close_previous=do_close_previous,
                show=True,
                block=False,
            )
            if book is not None:
                project_run.figures = (book,)
        self.last_run = project_run
        return project_run

    def solve(
        self,
        *,
        project_root: Path | str | None = None,
        output_dir: Path | str | None = None,
        save_reports: bool | None = None,
        save_gallery: bool | None = None,
        write_summary: bool | None = None,
        open_plot_book: bool | None = None,
        close_previous_plots: bool | None = None,
        plot_include: Sequence[str] | None = None,
        max_plot_parts: int | None = None,
        continue_on_channel_failure: bool | None = None,
        runner: RunDesignFunction | None = None,
    ) -> CoilProjectRun:
        """Solve all enabled project channels and store the result on the project."""

        return self.run(
            project_root=project_root,
            output_dir=output_dir,
            save_reports=save_reports,
            save_gallery=save_gallery,
            write_summary=write_summary,
            open_plot_book=open_plot_book,
            close_previous_plots=close_previous_plots,
            plot_include=plot_include,
            max_plot_parts=max_plot_parts,
            continue_on_channel_failure=continue_on_channel_failure,
            runner=runner,
        )

    def show_plot_book(
        self,
        run: CoilProjectRun,
        *,
        include: Sequence[str] | None = None,
        max_parts: int | None = None,
        close_previous: bool | None = None,
        show: bool = True,
        block: bool = False,
        preload_pages: str | bool = "idle",
    ) -> Any | None:
        """Open a project-level plot book for an existing run."""

        return run.show_plot_book(
            include=include,
            max_parts=max_parts,
            close_previous=close_previous,
            show=show,
            block=block,
            preload_pages=preload_pages,
        )

    def plot(
        self,
        run: CoilProjectRun | None = None,
        *,
        include: Sequence[str] | None = None,
        max_parts: int | None = None,
        close_previous: bool | None = None,
        mode: str = "book",
        show: bool = True,
        block: bool = False,
        channels: Sequence[str] | str | None = None,
        extra_figures: Sequence[Any] | None = None,
        extra_names: Sequence[str] | None = None,
        extra_categories: Sequence[str] | None = None,
        extra_specs: Sequence[Any] | None = None,
        preload_pages: str | bool = "idle",
    ) -> list[Any]:
        """Open plots for a solved project.

        ``mode="book"`` opens one project plot book with project summary pages
        and channel-prefixed result pages. ``mode="windows"`` opens the normal
        result figures for each selected channel.
        """

        project_run = run if run is not None else self.last_run
        if project_run is None:
            raise RuntimeError("Solve the project before plotting: project.solve().")
        plot_mode = str(mode or "book").strip().lower()
        selected = _selected_channel_runs(project_run, channels)
        if plot_mode in {"book", "plot_book", "pages", "paged"}:
            selected_run = (
                project_run
                if len(selected) == len(project_run.channel_runs)
                else replace(
                    project_run,
                    channel_runs=tuple(selected),
                    summary_rows=tuple(
                        row
                        for row in project_run.summary_rows
                        if str(row.get("channel", "")).strip().lower()
                        in {channel_run.channel.name.strip().lower() for channel_run in selected}
                    ),
                )
            )
            book = plot_project_run_book(
                selected_run,
                include=self.plot_include if include is None else include,
                max_parts=self.max_plot_parts if max_parts is None else max_parts,
                close_previous=self.close_previous_plots if close_previous is None else bool(close_previous),
                show=show,
                block=block,
                extra_figures=extra_figures,
                extra_names=extra_names,
                extra_categories=extra_categories,
                extra_specs=extra_specs,
                preload_pages=preload_pages,
            )
            return [] if book is None else [book]
        if plot_mode in {"windows", "window", "figures", "separate"}:
            return _plot_project_channel_windows(
                selected,
                include=self.plot_include if include is None else include,
                max_parts=self.max_plot_parts if max_parts is None else max_parts,
                close_previous=self.close_previous_plots if close_previous is None else bool(close_previous),
                show=show,
                block=block,
            )
        raise ValueError("mode must be 'book' or 'windows'.")

    @property
    def results(self) -> dict[str, Any]:
        """Successful channel results from the latest solve."""

        return {} if self.last_run is None else self.last_run.results

    @property
    def failures(self) -> dict[str, Exception]:
        """Channel failures from the latest solve."""

        return {} if self.last_run is None else self.last_run.failures

    @property
    def files(self) -> tuple[Path, ...]:
        """Files written by the latest solve."""

        return () if self.last_run is None else self.last_run.files

    @property
    def summary_rows(self) -> tuple[dict[str, Any], ...]:
        """Summary rows from the latest solve."""

        return () if self.last_run is None else self.last_run.summary_rows

    def _project_root(self) -> Path:
        return Path(self.project_root) if self.project_root is not None else Path.cwd()

    def _output_dir(self, project_root: Path) -> Path:
        return Path(self.output_dir) if self.output_dir is not None else project_root / "outputs" / self.safe_name

    def _validate_channel_names(self) -> None:
        seen: dict[str, str] = {}
        for channel in self.channels:
            key = channel.name.strip().lower()
            if key in seen:
                raise ValueError(f"Duplicate CoilProject channel name: {channel.name!r} conflicts with {seen[key]!r}.")
            seen[key] = channel.name


def write_project_summary(
    project_run: CoilProjectRun,
    output_dir: Path | str | None = None,
    *,
    prefix: str | None = None,
) -> tuple[Path, ...]:
    """Write project-level CSV, JSON, and Markdown summary files."""

    output = Path(output_dir) if output_dir is not None else project_run.output_dir
    output.mkdir(parents=True, exist_ok=True)
    stem = _safe_file_stem(prefix or project_run.project.name)
    csv_path = output / f"{stem}_project_summary.csv"
    json_path = output / f"{stem}_project_summary.json"
    md_path = output / f"{stem}_project_summary.md"
    _write_rows_csv(project_run.summary_rows, csv_path)
    json_path.write_text(
        json.dumps(
            {
                "project": project_run.project.name,
                "ok": project_run.ok,
                "rows": _json_ready(project_run.summary_rows),
                "failures": {name: str(exc) for name, exc in project_run.failures.items()},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    md_path.write_text(_summary_markdown(project_run.project.name, project_run.summary_rows), encoding="utf-8")
    return (csv_path, json_path, md_path)


def plot_project_run_book(
    project_run: CoilProjectRun,
    *,
    include: Sequence[str] | None = None,
    max_parts: int | None = None,
    close_previous: bool = True,
    show: bool = True,
    block: bool = False,
    title: str | None = None,
    extra_figures: Sequence[Any] | None = None,
    extra_names: Sequence[str] | None = None,
    extra_categories: Sequence[str] | None = None,
    extra_specs: Sequence[Any] | None = None,
    preload_pages: str | bool = "idle",
) -> Any | None:
    """Open one active plot book for a multi-channel project run."""

    if not project_run.channel_runs:
        return None
    try:
        from .matplotlib_visualization import (
            MatplotlibPlotBookBuilder,
            MatplotlibPlotSpec,
            activate_matplotlib_interactive_backend,
            build_result_plot_specs,
            close_matplotlib_plot_books,
        )
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"Project plot book skipped: {exc}")
        return None

    if close_previous:
        close_matplotlib_plot_books()
        try:
            plt.close("all")
        except Exception:
            pass
    backend = activate_matplotlib_interactive_backend() if show else str(plt.get_backend())
    timing_report: list[dict[str, float | int | str]] = []
    start = time.perf_counter()
    builder = MatplotlibPlotBookBuilder(
        title=title or f"{project_run.project.name} Plot Book",
        plt=plt,
        timing_report=timing_report,
        preload_pages=preload_pages,
    )
    builder.add_spec(
        MatplotlibPlotSpec(
            key="parameter_table",
            title="Project Channel Summary",
            category="Project",
            factory=lambda: _plot_project_channel_table(project_run),
        ),
        short_name_factory=_project_plot_book_short_name,
    )
    if _has_numeric_project_metrics(project_run.summary_rows):
        builder.add_spec(
            MatplotlibPlotSpec(
                key="metric_bars",
                title="Project Channel Metric Comparison",
                category="Project",
                factory=lambda: _plot_project_metric_bars(project_run),
            ),
            short_name_factory=_project_plot_book_short_name,
        )
    if sum(1 for channel_run in project_run.channel_runs if channel_run.result is not None) > 1:
        builder.add_spec(
            MatplotlibPlotSpec(
                key="channel_spherical_harmonics",
                title="Project Channel Spherical Harmonic Coefficients",
                category="Project",
                factory=lambda: _plot_project_channel_spherical_harmonics(project_run),
            ),
            short_name_factory=_project_plot_book_short_name,
        )
    for spec in extra_specs or ():
        builder.add_spec(spec, short_name_factory=_project_plot_book_short_name)
    for index, figure in enumerate(extra_figures or (), start=1):
        name = _sequence_item(extra_names, index - 1, f"Extra Figure {index}")
        category = _sequence_item(extra_categories, index - 1, "Project")
        builder.add_spec(
            MatplotlibPlotSpec(
                key=f"extra_{index}",
                title=name,
                category=category,
                factory=lambda fig=figure: fig,
            ),
            short_name_factory=_project_plot_book_short_name,
        )
    requested = project_run.project.plot_include if include is None else include
    for channel_run in project_run.channel_runs:
        if channel_run.result is None:
            continue
        specs = build_result_plot_specs(channel_run.result, include=requested, max_parts=max_parts)
        for spec in specs:
            builder.add_spec(
                _channel_prefixed_plot_spec(MatplotlibPlotSpec, channel_run, spec),
                short_name_factory=_project_plot_book_short_name,
            )
    if not builder.entries:
        return None
    book = builder.build()
    book.export_metadata = {
        "project": project_run.project.name,
        "output_directory": str(getattr(project_run, "output_dir", "")),
        "channels": [
            {
                "name": channel_run.channel.name,
                "display_label": channel_run.channel.display_label,
                "design_name": str(getattr(getattr(channel_run, "design", None), "name", "")),
                "result_available": channel_run.result is not None,
            }
            for channel_run in project_run.channel_runs
        ],
    }
    if show:
        show_start = time.perf_counter()
        book.show(block=block)
        timing_report.append({"plot": "show", "seconds": float(time.perf_counter() - show_start), "figures": 1})
    elapsed = time.perf_counter() - start
    print(f"Matplotlib backend: {backend}")
    print(f"{'Opened' if show else 'Built'} 1 project plot book for {len(project_run.results)} successful channel(s).")
    print(f"Project plot-book time: {elapsed:.3f} s")
    return book


def _selected_channel_runs(
    project_run: CoilProjectRun, channels: Sequence[str] | str | None
) -> tuple[CoilChannelRun, ...]:
    if channels is None:
        return project_run.channel_runs
    requested = (
        {str(channels).strip().lower()}
        if isinstance(channels, str)
        else {str(channel).strip().lower() for channel in channels}
    )
    selected = tuple(
        channel_run for channel_run in project_run.channel_runs if channel_run.channel.name.strip().lower() in requested
    )
    missing = sorted(requested - {channel_run.channel.name.strip().lower() for channel_run in selected})
    if missing:
        raise ValueError(f"Unknown project channel(s): {', '.join(missing)}.")
    return selected


def _plot_project_channel_windows(
    channel_runs: Sequence[CoilChannelRun],
    *,
    include: Sequence[str] | None,
    max_parts: int | None,
    close_previous: bool,
    show: bool,
    block: bool,
) -> list[Any]:
    try:
        from .matplotlib_visualization import (
            activate_matplotlib_interactive_backend,
            close_matplotlib_plot_books,
            plot_result_matplotlib,
        )
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"Project channel plots skipped: {exc}")
        return []
    if close_previous:
        close_matplotlib_plot_books()
        try:
            plt.close("all")
        except Exception:
            pass
    if show:
        activate_matplotlib_interactive_backend()
    figures: list[Any] = []
    for channel_run in channel_runs:
        if channel_run.result is None:
            continue
        channel_figures = plot_result_matplotlib(
            channel_run.result,
            include=include,
            max_parts=max_parts,
            show=show,
            block=block,
        )
        for figure in channel_figures:
            _prefix_figure_label(figure, channel_run.channel.display_label, "CoilGen result")
        figures.extend(channel_figures)
    return figures


def _copy_design_with_overrides(source: Any, overrides: Mapping[str, Any], *, channel_name: str) -> Any:
    base = copy.deepcopy(source)
    if not overrides:
        return base
    if is_dataclass(source) and not isinstance(source, type):
        valid_fields = {item.name for item in fields(source)}
        unknown = sorted(key for key in overrides if key not in valid_fields)
        if unknown:
            joined = ", ".join(unknown)
            raise ValueError(
                f"Channel {channel_name!r} has override(s) not present on {type(source).__name__}: {joined}."
            )
        return replace(base, **dict(overrides))
    clone = base
    for key, value in overrides.items():
        if not hasattr(clone, key):
            raise ValueError(f"Channel {channel_name!r} cannot set unknown design attribute {key!r}.")
        setattr(clone, key, value)
    return clone


def _design_supports_override(source: Any, key: str) -> bool:
    if is_dataclass(source) and not isinstance(source, type):
        return key in {item.name for item in fields(source)}
    return hasattr(source, key)


def _normalize_gradient_axis(axis: object) -> str:
    text = str(axis).strip().lower()
    if text.startswith("g") and len(text) == 2:
        text = text[1]
    if text not in {"x", "y", "z"}:
        raise ValueError("Gradient axis must be 'x', 'y', 'z', 'Gx', 'Gy', or 'Gz'.")
    return text


def _design_setup_row(design: Any) -> dict[str, Any]:
    row = {"design_class": type(design).__name__}
    for key in (
        "name",
        "design_context",
        "target_bz_expression",
        "target_scale",
        "target_gradient_strength_T_per_m",
        "target_slew_rate_T_per_m_per_s",
        "target_gradient_normalization",
        "target_gradient_axis",
        "field_evaluation_current_source",
        "target_region_radius_m",
        "target_region_resolution",
        "contour_discretization_method",
        "contour_levels",
        "min_turn_spacing_m",
        "max_current_A",
        "max_wire_current_A",
        "min_wire_distance_m",
        "surface_current_density_limit_A_per_m",
        "target_error_tolerance_percent",
        "target_error_tolerance_absolute_T",
        "gradient_strength_tolerance_percent",
        "linearity_rms_tolerance_percent",
        "linearity_peak_tolerance_percent",
        "linearity_cross_gradient_tolerance_percent",
        "suppression_field_rms_limit_T",
        "suppression_field_peak_limit_T",
        "gradient_hardware_infeasible_policy",
        "conductor_width_m",
        "conductor_height_m",
        "radius_m",
        "length_m",
        "width_m",
        "height_m",
        "plane_distance_m",
    ):
        if hasattr(design, key):
            row[key] = getattr(design, key)
    return row


def _channel_prefixed_plot_spec(plot_spec_class: type[Any], channel_run: CoilChannelRun, spec: Any) -> Any:
    channel_label = channel_run.channel.display_label
    title = f"{channel_label}: {spec.title}"
    category = str(spec.category)

    def factory() -> Any:
        figure_or_figures = spec.factory()
        figures = figure_or_figures if isinstance(figure_or_figures, (list, tuple)) else [figure_or_figures]
        for figure in figures:
            if figure is not None:
                _prefix_figure_label(figure, channel_label, spec.title)
        return figure_or_figures

    return plot_spec_class(
        key=f"{channel_run.channel.safe_name}_{spec.key}",
        title=title,
        category=category,
        factory=factory,
        channel=channel_label,
    )


def _prefix_figure_label(figure: Any, channel_label: str, fallback_title: str) -> None:
    try:
        label = str(figure.get_label()).strip()
    except Exception:
        label = ""
    title = label or str(fallback_title)
    prefixed = title if title.startswith(f"{channel_label}:") else f"{channel_label}: {title}"
    try:
        figure.set_label(prefixed)
    except Exception:
        pass
    try:
        figure.canvas.manager.set_window_title(prefixed)
    except Exception:
        pass


def _plot_project_channel_table(project_run: CoilProjectRun) -> Any:
    import matplotlib.pyplot as plt

    rows = project_run.summary_rows or tuple(channel_run.summary_row() for channel_run in project_run.channel_runs)
    sections = _project_channel_summary_sections(project_run)
    row_count = max(len(rows), 1)
    height = max(4.2, 0.9 + len(sections) * (0.56 + 0.32 * (row_count + 1)))
    fig = plt.figure(figsize=(_project_channel_table_width(sections, rows), height))
    fig.set_label("Project Channel Summary")
    grid = fig.add_gridspec(
        len(sections),
        1,
        left=0.035,
        right=0.985,
        bottom=0.035,
        top=0.93,
        hspace=0.36,
    )
    for section_index, (section_title, columns) in enumerate(sections):
        ax = fig.add_subplot(grid[section_index, 0])
        _draw_project_channel_summary_section(ax, section_title, columns, rows)
    fig.suptitle(f"{project_run.project.name} channel summary", fontsize=13, y=0.985)
    return fig


def _draw_project_channel_summary_section(
    ax: Any,
    section_title: str,
    columns: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    ax.axis("off")
    ax.text(
        0.0,
        1.0,
        section_title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.4,
        fontweight="bold",
        color="#111827",
    )
    cell_text = _project_summary_section_cell_text(columns, rows)
    table = ax.table(
        cellText=cell_text,
        colLabels=[_project_section_column_label(section_title, column) for column in columns],
        cellLoc="left",
        colLoc="left",
        colWidths=_project_summary_section_col_widths(columns, rows),
        bbox=(0.0, 0.0, 1.0, 0.82),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    for (row_index, _column_index), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_text_props(weight="bold", color="#111827")
            cell.set_facecolor("#e5e7eb")
        else:
            cell.set_facecolor("#f8fafc" if row_index % 2 else "#ffffff")
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(0.55)


def _project_channel_table_width(
    sections: Sequence[tuple[str, Sequence[str]]],
    rows: Sequence[Mapping[str, Any]],
) -> float:
    if not sections:
        return 12.6
    widest_weight = max(sum(_project_summary_section_column_weights(columns, rows)) for _title, columns in sections)
    return min(max(12.8, 1.48 * widest_weight), 24.0)


def _project_summary_section_col_widths(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> list[float]:
    weights = _project_summary_section_column_weights(columns, rows)
    total = sum(weights) or 1.0
    widths = [weight / total for weight in weights]
    return _cap_project_summary_channel_width(columns, widths)


def _project_summary_section_cell_text(
    columns: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> list[list[str]]:
    channel_prefix = _project_summary_common_channel_prefix(rows)
    return [
        [
            _project_summary_table_display_value(
                column,
                row.get(column, ""),
                row=row,
                channel_prefix=channel_prefix,
            )
            for column in columns
        ]
        for row in rows
    ]


def _project_summary_section_column_weights(
    columns: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> list[float]:
    count = max(1, len(columns))
    if count == 1:
        return [1.0]
    weights: list[float] = []
    channel_prefix = _project_summary_common_channel_prefix(rows)
    for column in columns:
        label_length = len(_project_section_column_label("", column))
        value_length = max(
            [
                len(
                    _project_summary_table_display_value(
                        column,
                        row.get(column, ""),
                        row=row,
                        channel_prefix=channel_prefix,
                    )
                )
                for row in rows
            ]
            or [0]
        )
        length = max(label_length, value_length)
        if column == "channel":
            weights.append(min(max(2.2, 0.10 * length), 5.4))
        else:
            weights.append(min(max(1.15, 0.105 * length), 2.3))
    return weights


def _project_summary_table_display_value(
    column: str,
    value: Any,
    *,
    row: Mapping[str, Any] | None = None,
    channel_prefix: str = "",
) -> str:
    text = _project_summary_table_value(column, value)
    if str(column) == "channel" and channel_prefix and text.startswith(channel_prefix):
        compact = text[len(channel_prefix) :]
        if compact:
            return compact
    if _project_summary_metric_reference_required(column):
        if not text:
            text = "unspecified"
        return f"{text} ({_project_summary_metric_reference(column, row or {})})"
    return text


def _project_summary_metric_reference_required(column: str) -> bool:
    return column not in {
        "channel",
        "label",
        "role",
        "status",
        "design_context",
        "target_bz_expression",
        "target_scale",
        "quality_status",
        "field_display_units",
    }


def _project_summary_metric_reference(column: str, row: Mapping[str, Any]) -> str:
    current_columns = {
        "needed_current_A",
        "stream_function_needed_current_A",
        "unconnected_contours_needed_current_A",
        "layout_needed_current_A",
    }
    error_columns = {
        "stream_function_mean_rel_error_percent",
        "stream_function_max_rel_error_percent",
        "unconnected_contours_mean_rel_error_percent",
        "unconnected_contours_max_rel_error_percent",
        "layout_mean_rel_error_percent",
        "layout_max_rel_error_percent",
    }
    spacing_columns = {
        "unconnected_contours_min_wire_distance_m",
        "min_wire_wire_distance_m",
        "min_contour_distance_m",
    }
    if column in current_columns:
        return _project_summary_threshold_reference(row, ("max_wire_current_A", "max_current_A"), "<=")
    if column in error_columns:
        return _project_summary_threshold_reference(row, ("target_error_tolerance_percent",), "<=")
    if column in spacing_columns:
        return _project_summary_threshold_reference(row, ("min_wire_distance_m", "min_turn_spacing_m"), ">=")
    if column in {"linearity_rms_percent", "gradient_slice_map_linearity_rms_percent"}:
        return _project_summary_threshold_reference(row, ("linearity_rms_tolerance_percent",), "<=")
    if column in {"linearity_peak_percent", "gradient_slice_map_linearity_peak_percent"}:
        return _project_summary_threshold_reference(row, ("linearity_peak_tolerance_percent",), "<=")
    if column == "linearity_cross_gradient_percent":
        return _project_summary_threshold_reference(row, ("linearity_cross_gradient_tolerance_percent",), "<=")
    if column == "linearity_gradient_strength_error_percent":
        return _project_summary_threshold_reference(
            row,
            ("linearity_gradient_strength_tolerance_percent", "gradient_strength_tolerance_percent"),
            "<=",
        )
    if column == "surface_current_density_max_A_per_m":
        return _project_summary_threshold_reference(row, ("surface_current_density_limit_A_per_m",), "<=")
    if column == "target_gradient_hardware_effective_T_per_m":
        return _project_summary_target_reference(row, ("target_gradient_requested_T_per_m", "target_scale"))
    if column == "drive_voltage_target_slew_V":
        return _project_summary_slew_reference(row)
    return "threshold: unspecified"


def _project_summary_threshold_reference(
    row: Mapping[str, Any],
    keys: Sequence[str],
    comparator: str,
) -> str:
    value = _project_summary_first_positive_value(row, keys)
    if value is None:
        return "threshold: unspecified"
    return f"threshold: {comparator} {_short_value(value)}"


def _project_summary_target_reference(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    value = _project_summary_first_finite_value(row, keys)
    if value is None:
        return "target: unspecified"
    return f"target: {_short_value(value)}"


def _project_summary_slew_reference(row: Mapping[str, Any]) -> str:
    value = _project_summary_first_positive_value(row, ("target_slew_rate_T_per_m_per_s",))
    if value is None:
        return "target slew: unspecified"
    return f"target slew: {_short_value(value)} T/m/s"


def _project_drive_metrics_from_summary(row: Mapping[str, Any]) -> dict[str, float]:
    target_gradient = _project_summary_first_positive_value(
        row,
        ("target_gradient_strength_T_per_m", "target_gradient_requested_T_per_m", "target_scale"),
    )
    target_slew = _project_summary_first_positive_value(row, ("target_slew_rate_T_per_m_per_s",))
    current = _project_summary_first_finite_value(row, ("layout_needed_current_A", "needed_current_A"))
    resistance = _project_summary_first_finite_value(row, ("coil_resistance_total_ohm",))
    inductance = _project_summary_first_finite_value(row, ("coil_inductance_same_current_H",))
    if None in (target_gradient, target_slew, current, resistance, inductance):
        return {}
    metrics = drive_metrics_from_inductance(
        current_A=float(current),
        resistance_ohm=float(resistance),
        inductance_H=float(inductance),
        target_gradient_T_per_m=float(target_gradient),
        target_slew_rate_T_per_m_per_s=float(target_slew),
    )
    return metrics.as_dict()


def _project_summary_first_positive_value(row: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = _project_summary_first_finite_value(row, (key,))
        if value is not None and value > 0.0:
            return value
    return None


def _project_summary_first_finite_value(row: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        try:
            value = float(row.get(key, np.nan))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return float(value)
    return None


def _project_summary_common_channel_prefix(rows: Sequence[Mapping[str, Any]]) -> str:
    channel_names = [
        _project_summary_table_value("channel", row.get("channel", ""))
        for row in rows
        if str(row.get("channel", "")).strip()
    ]
    if len(channel_names) < 2:
        return ""
    prefix = str(channel_names[0])
    for name in channel_names[1:]:
        while prefix and not str(name).startswith(prefix):
            prefix = prefix[:-1]
    split_at = prefix.rfind("_")
    if split_at < 6:
        return ""
    common_prefix = prefix[: split_at + 1]
    return common_prefix if all(len(name) > len(common_prefix) for name in channel_names) else ""


def _cap_project_summary_channel_width(columns: Sequence[str], widths: Sequence[float]) -> list[float]:
    adjusted = [float(width) for width in widths]
    try:
        channel_index = tuple(columns).index("channel")
    except ValueError:
        return adjusted
    if len(adjusted) <= 1:
        return adjusted
    max_fraction = _project_summary_channel_max_fraction(len(adjusted))
    if adjusted[channel_index] <= max_fraction:
        return adjusted
    excess = adjusted[channel_index] - max_fraction
    adjusted[channel_index] = max_fraction
    other_total = sum(width for index, width in enumerate(adjusted) if index != channel_index)
    if other_total <= 0.0:
        return adjusted
    for index, width in enumerate(tuple(adjusted)):
        if index != channel_index:
            adjusted[index] = width + excess * (width / other_total)
    return adjusted


def _project_summary_channel_max_fraction(column_count: int) -> float:
    if column_count <= 2:
        return 0.70
    if column_count == 3:
        return 0.58
    if column_count == 4:
        return 0.50
    return 0.44


def _plot_project_metric_bars(project_run: CoilProjectRun) -> Any:
    import matplotlib.pyplot as plt

    rows = list(project_run.summary_rows)
    requested_metrics = _project_comparison_metrics(project_run)
    metrics = tuple(metric for metric in requested_metrics if _project_metric_series_available(rows, metric[0]))
    omitted_metrics = tuple(metric for metric in requested_metrics if metric not in metrics)
    column_count = 2 if len(metrics) <= 4 else 3
    row_count = max(1, math.ceil(len(metrics) / column_count))
    figure_width = 13.6 if column_count == 2 else 15.2
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(figure_width, 2.35 + 3.15 * row_count),
        squeeze=False,
    )
    fig.set_label("Project Channel Metric Comparison")
    colors = [_channel_color(project_run, str(row.get("channel", ""))) for row in rows]
    channel_labels = _project_metric_bar_label_map(rows)
    single_channel = len(rows) == 1
    legend_handles = [] if single_channel else _project_metric_bar_legend_handles(rows, colors, channel_labels)
    axes_list = list(axes.flat)
    for ax, (key, title, unit) in zip(axes_list, metrics):
        labels, values, metric_colors = _metric_values(rows, key, colors, channel_labels)
        if values:
            bars = ax.bar(
                labels,
                values,
                width=0.54 if single_channel else 0.72,
                color=metric_colors,
                edgecolor="#1f2937",
                linewidth=0.45,
            )
            _annotate_project_metric_bars(ax, bars, values)
        ax.set_title(title, fontsize=11.5, pad=8)
        ax.set_ylabel(unit)
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", alpha=0.22)
        ax.tick_params(axis="x", labelrotation=0, labelsize=8, pad=2)
        if single_channel:
            ax.set_xticks([])
        else:
            for tick_label in ax.get_xticklabels():
                tick_label.set_multialignment("center")
    for ax in axes_list[len(metrics) :]:
        ax.axis("off")

    descriptor = _project_metric_page_descriptor(project_run, rows)
    fig.suptitle(descriptor, y=0.975, fontsize=14.5, fontweight="bold")
    fig.text(
        0.5,
        0.94,
        "Single-channel performance summary" if single_channel else "Channel metric comparison",
        ha="center",
        va="top",
        fontsize=10.0,
        color="#475569",
    )
    footer_note = _project_metric_omission_note(project_run, omitted_metrics)
    if footer_note:
        fig.text(0.5, 0.018, footer_note, ha="center", va="bottom", fontsize=8.5, color="#475569")

    bottom_margin = 0.07 if legend_handles else 0.06 if footer_note else 0.035
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.018 if not footer_note else 0.034),
            ncol=min(max(len(legend_handles), 1), 3),
            fontsize=8,
            frameon=False,
        )
        bottom_margin = 0.09 if footer_note else 0.07
    fig.tight_layout(rect=(0.025, bottom_margin, 0.985, 0.90), h_pad=2.0, w_pad=2.2)
    return fig


_POSITIVE_HARDWARE_COMPARISON_METRICS = {
    "coil_resistance_total_ohm",
    "coil_inductance_same_current_H",
}


def _project_metric_series_available(rows: Sequence[Mapping[str, Any]], key: str) -> bool:
    values = [_safe_float(row.get(key)) for row in rows]
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return False
    if key in _POSITIVE_HARDWARE_COMPARISON_METRICS:
        return any(value > 0.0 for value in finite)
    return True


def _annotate_project_metric_bars(ax: Any, bars: Any, values: Sequence[float]) -> None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return
    lower = min([0.0, *finite])
    upper = max([0.0, *finite])
    span = max(upper - lower, max(abs(value) for value in finite), 1.0e-12)
    ax.set_ylim(lower - (0.12 * span if lower < 0.0 else 0.0), upper + 0.18 * span)
    for bar, value in zip(bars, values, strict=True):
        value = float(value)
        if not math.isfinite(value):
            continue
        ax.annotate(
            f"{value:.3g}",
            xy=(bar.get_x() + 0.5 * bar.get_width(), value),
            xytext=(0, 4 if value >= 0.0 else -4),
            textcoords="offset points",
            ha="center",
            va="bottom" if value >= 0.0 else "top",
            fontsize=9.0,
            fontweight="bold",
            color="#1e293b",
        )


def _project_metric_page_descriptor(project_run: CoilProjectRun, rows: Sequence[Mapping[str, Any]]) -> str:
    if len(rows) == 1:
        descriptor = _project_metric_descriptive_label(rows[0])
    else:
        descriptor = str(project_run.project.name)
    words = [word for word in str(descriptor).replace("-", "_").split("_") if word]
    replacements = {
        "b0": "B0",
        "dbsm": "DBSM",
        "mri": "MRI",
        "rf": "RF",
    }
    text = " ".join(replacements.get(word.lower(), word) for word in words).strip()
    return text[:1].upper() + text[1:] if text else "Coil channel performance"


def _project_metric_omission_note(
    project_run: CoilProjectRun,
    omitted_metrics: Sequence[tuple[str, str, str]],
) -> str:
    hardware_titles = [title for key, title, _unit in omitted_metrics if key in _POSITIVE_HARDWARE_COMPARISON_METRICS]
    if not hardware_titles:
        return ""
    skipped = []
    for channel_run in project_run.channel_runs:
        result = getattr(channel_run, "result", None)
        config = getattr(result, "input_data", None)
        if config is not None:
            skipped.append(bool(getattr(config, "skip_inductance_calculation", False)))
    reason = "calculation disabled for this run" if skipped and all(skipped) else "values unavailable"
    return f"{' and '.join(hardware_titles)} omitted: {reason}."


def _project_channel_sh_fits(
    project_run: CoilProjectRun, *, max_order: int = 4
) -> tuple[list[tuple[CoilChannelRun, Any]], list[str]]:
    """Use one physical SH basis and display unit across project channels.

    Each channel retains its own DSV samples and weights. Coefficients share
    the first available channel's center/radius; RMS contributions describe
    each channel's own evaluation domain, which can differ between channels.
    """
    from .field_maps import coil_spherical_harmonic_fits
    from .field_units import field_values_from_tesla, field_values_to_tesla

    channel_fits: list[tuple[CoilChannelRun, Any]] = []
    unavailable: list[str] = []
    reference = None
    for channel_run in project_run.channel_runs:
        if channel_run.result is None:
            continue
        try:
            fit = coil_spherical_harmonic_fits(
                channel_run.result,
                domain="dsv",
                max_order=max_order,
                sources=("layout",),
                center=None if reference is None else reference.center,
                radius=None if reference is None else reference.radius,
            )["layout"]
        except ValueError as exc:
            unavailable.append(f"{channel_run.channel.display_label}: {exc}")
            continue
        if reference is None:
            reference = fit
        elif fit.unit != reference.unit:
            factor = float(field_values_from_tesla(field_values_to_tesla(1.0, fit.unit), reference.unit))
            fit = replace(
                fit,
                unit=reference.unit,
                coefficients=fit.coefficients * factor,
                fitted_values=fit.fitted_values * factor,
                residual_values=fit.residual_values * factor,
                field_rms=fit.field_rms * abs(factor),
                residual_rms=fit.residual_rms * abs(factor),
            )
        channel_fits.append((channel_run, fit))
    return channel_fits, unavailable


def _plot_project_channel_spherical_harmonics(project_run: CoilProjectRun) -> Any:
    import matplotlib.pyplot as plt

    from .matplotlib_visualization import (
        _draw_sh_order_strip,
        _draw_sh_term_label_column,
        _engineer_sh_plot_rows,
        _format_performance_table_value,
        _mr_sh_term_name,
        _prepare_sh_data_axis,
        _set_axis_title,
        _shade_sh_degree_groups,
    )

    max_order = 4
    channel_fits, unavailable = _project_channel_sh_fits(project_run, max_order=max_order)
    if not channel_fits:
        raise ValueError("No project channel SH fits were available.")

    reference = channel_fits[0][1]
    rows: list[dict[str, Any]] = []
    for index, term in enumerate(reference.terms):
        row: dict[str, Any] = {
            "term": term,
            "term_label": term.display_label,
            "degree": int(term.degree),
            "order": int(term.order),
            "index": int(index),
            "unit": reference.unit,
        }
        max_contribution = 0.0
        max_fraction = 0.0
        for channel_run, fit in channel_fits:
            coefficient = float(np.asarray(fit.coefficients, dtype=float).reshape(-1)[index])
            basis_rms = (
                1.0 if fit.basis_rms is None else float(np.asarray(fit.basis_rms, dtype=float).reshape(-1)[index])
            )
            contribution = abs(coefficient) * basis_rms
            fraction = contribution / max(float(fit.field_rms), 1.0e-30)
            key = channel_run.channel.safe_name
            row[f"{key}_coefficient"] = coefficient
            row[f"{key}_contribution_rms"] = float(contribution)
            row[f"{key}_contribution_fraction"] = float(fraction)
            max_contribution = max(max_contribution, float(contribution))
            max_fraction = max(max_fraction, float(fraction))
        row["max_contribution_rms"] = float(max_contribution)
        row["max_contribution_fraction"] = float(max_fraction)
        rows.append(row)
    rows = _engineer_sh_plot_rows(rows, max_order=max_order, min_contribution_fraction=0.01, max_terms=24)
    if not rows:
        raise ValueError("No project channel SH coefficient rows were available.")

    y = np.arange(len(rows), dtype=float)
    term_names = [_mr_sh_term_name(row) for row in rows]
    term_indices = [f"l={int(row['degree'])}, m={int(row['order']):+d}" for row in rows]
    unit = str(reference.unit or "")
    unit_suffix = f" [{unit}]" if unit else ""
    bar_height = min(0.70 / max(len(channel_fits), 1), 0.22)
    offsets = (
        np.linspace(-0.35 + bar_height / 2.0, 0.35 - bar_height / 2.0, len(channel_fits))
        if len(channel_fits) > 1
        else np.asarray([0.0])
    )

    fig = plt.figure(
        num="Project Channel Spherical Harmonic Coefficients", figsize=(16.4, max(7.0, 0.44 * len(rows) + 3.1))
    )
    fig.set_label("Project Channel Spherical Harmonic Coefficients")
    fig.clf()
    grid = fig.add_gridspec(
        2,
        4,
        height_ratios=[0.25, 1.0],
        width_ratios=[0.62, 2.85, 2.75, 2.55],
        left=0.035,
        right=0.985,
        top=0.96,
        bottom=0.10,
        wspace=0.13,
        hspace=0.15,
    )
    title_ax = fig.add_subplot(grid[0, :])
    title_ax.set_axis_off()
    title_ax.text(
        0.0,
        0.70,
        f"{project_run.project.name}: final-layout SH coefficients by channel",
        ha="left",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    title_ax.text(
        0.0,
        0.38,
        (
            f"Shared SH basis, order <= {max_order}; "
            f"center=({reference.center[0]:.4g}, {reference.center[1]:.4g}, {reference.center[2]:.4g}) m, "
            f"radius={reference.radius:.4g} m; "
            + "; ".join(
                f"{channel_run.channel.display_label}: n={fit.sample_count}, signal RMS "
                f"{_format_performance_table_value(fit.field_rms)}"
                for channel_run, fit in channel_fits
            )
            + unit_suffix
        ),
        ha="left",
        va="center",
        fontsize=9,
        color="#475569",
    )
    title_ax.text(
        0.0,
        0.10,
        "Final-current Bz; common center, radius and units. RMS contributions use each channel's own DSV samples.",
        ha="left",
        va="center",
        fontsize=9,
        color="#334155",
        clip_on=False,
    )

    order_ax = fig.add_subplot(grid[1, 0])
    label_ax = fig.add_subplot(grid[1, 1], sharey=order_ax)
    coefficient_ax = fig.add_subplot(grid[1, 2], sharey=order_ax)
    contribution_ax = fig.add_subplot(grid[1, 3], sharey=order_ax)
    data_axes = (label_ax, coefficient_ax, contribution_ax)
    _shade_sh_degree_groups(data_axes, rows)
    _draw_sh_order_strip(order_ax, rows)
    _draw_sh_term_label_column(label_ax, term_names, term_indices, rows)

    for offset, (channel_run, _fit) in zip(offsets, channel_fits):
        key = channel_run.channel.safe_name
        color = channel_run.channel.color or _channel_color(project_run, channel_run.channel.name)
        label = channel_run.channel.display_label
        coefficients = np.asarray([float(row.get(f"{key}_coefficient", np.nan)) for row in rows], dtype=float)
        contributions = np.asarray([float(row.get(f"{key}_contribution_rms", np.nan)) for row in rows], dtype=float)
        coefficient_ax.barh(y + offset, coefficients, height=bar_height, color=color, alpha=0.80, label=label)
        contribution_ax.barh(y + offset, contributions, height=bar_height, color=color, alpha=0.80, label=label)

    coefficient_ax.axvline(0.0, color="#111827", linewidth=1.0, alpha=0.65)
    _prepare_sh_data_axis(coefficient_ax, rows)
    coefficient_ax.set_xlabel(f"signed normalized SH coefficient{unit_suffix}")
    _set_axis_title(coefficient_ax, "Signed coefficient\nfield-pattern polarity", width=30)
    coefficient_ax.grid(True, axis="x", alpha=0.22)

    _prepare_sh_data_axis(contribution_ax, rows)
    contribution_ax.set_xlabel(f"RMS contribution on each channel's DSV{unit_suffix}")
    _set_axis_title(contribution_ax, "RMS contribution\non each channel's DSV", width=30)
    contribution_ax.grid(True, axis="x", alpha=0.22)
    contribution_ax.legend(loc="best", fontsize=8.4, frameon=True)
    if unavailable:
        fig.text(0.035, 0.02, "Unavailable: " + "; ".join(unavailable), fontsize=8, wrap=True)
    return fig


def _has_numeric_project_metrics(rows: Sequence[Mapping[str, Any]]) -> bool:
    metrics = PROJECT_COMPARISON_METRICS
    if rows:
        context = normalize_design_context(rows[0].get("design_context", "mr_gradient"))
        metrics = _comparison_metrics_for_context(context, unit=_project_field_display_unit(rows))
    for key, _title, _unit in metrics:
        if _project_metric_series_available(rows, key):
            return True
    return False


def _project_channel_summary_columns(project_run: CoilProjectRun) -> tuple[str, ...]:
    context = _project_run_design_context(project_run)
    columns = [
        "channel",
        "design_context",
        "target_bz_expression",
        "target_scale",
        "quality_status",
    ]
    if context.enabled(EvaluationFeature.TARGET_MODE_FIT):
        columns.extend(
            [
                "stream_function_needed_current_A",
                "stream_function_gradient_efficiency_mT_per_m_per_A",
                "stream_function_mean_rel_error_percent",
                "stream_function_max_rel_error_percent",
                "unconnected_contours_needed_current_A",
                "unconnected_contours_gradient_efficiency_mT_per_m_per_A",
                "unconnected_contours_mean_rel_error_percent",
                "unconnected_contours_max_rel_error_percent",
                "unconnected_contours_min_wire_distance_m",
                "layout_needed_current_A",
                "layout_gradient_efficiency_mT_per_m_per_A",
                "layout_mean_rel_error_percent",
                "layout_max_rel_error_percent",
            ]
        )
    else:
        columns.extend(
            [
                "layout_needed_current_A",
                "layout_mean_rel_error_percent",
                "layout_max_rel_error_percent",
            ]
        )
    if context.enabled(EvaluationFeature.SHIM_HOMOGENEITY):
        unit = _project_field_display_unit(project_run.summary_rows)
        columns.extend((f"shim_layout_residual_rms_{unit}", f"shim_layout_residual_max_abs_{unit}"))
    columns.extend(("coil_length_total_m", "coil_resistance_total_ohm", "coil_inductance_same_current_H"))
    if any(math.isfinite(_safe_float(row.get("drive_voltage_target_slew_V"))) for row in project_run.summary_rows):
        columns.append("drive_voltage_target_slew_V")
    return tuple(columns)


def _project_channel_summary_sections(project_run: CoilProjectRun) -> tuple[tuple[str, tuple[str, ...]], ...]:
    context = _project_run_design_context(project_run)
    available = set(_project_channel_summary_columns(project_run))
    sections: list[tuple[str, tuple[str, ...]]] = []

    def add(title: str, columns: Sequence[str]) -> None:
        selected = tuple(column for column in columns if column in available)
        if selected:
            sections.append((title, ("channel", *selected)))

    add(
        "Channel / Target",
        (
            "design_context",
            "target_bz_expression",
            "target_scale",
            "quality_status",
        ),
    )
    if context.enabled(EvaluationFeature.TARGET_MODE_FIT):
        add(
            "Stream Function",
            (
                "stream_function_needed_current_A",
                "stream_function_gradient_efficiency_mT_per_m_per_A",
                "stream_function_mean_rel_error_percent",
                "stream_function_max_rel_error_percent",
            ),
        )
        add(
            "Unconnected Contours",
            (
                "unconnected_contours_needed_current_A",
                "unconnected_contours_gradient_efficiency_mT_per_m_per_A",
                "unconnected_contours_mean_rel_error_percent",
                "unconnected_contours_max_rel_error_percent",
                "unconnected_contours_min_wire_distance_m",
            ),
        )
    add(
        "Connected Layout",
        (
            "layout_needed_current_A",
            "layout_gradient_efficiency_mT_per_m_per_A",
            "layout_mean_rel_error_percent",
            "layout_max_rel_error_percent",
        ),
    )
    if context.enabled(EvaluationFeature.SHIM_HOMOGENEITY):
        unit = _project_field_display_unit(project_run.summary_rows)
        add(
            "Shim Homogeneity",
            (
                f"shim_layout_residual_rms_{unit}",
                f"shim_layout_residual_max_abs_{unit}",
            ),
        )
    add(
        "Electrical / Hardware",
        (
            "coil_length_total_m",
            "coil_resistance_total_ohm",
            "coil_inductance_same_current_H",
            "drive_voltage_target_slew_V",
        ),
    )
    return tuple(sections)


def _project_comparison_metrics(project_run: CoilProjectRun) -> tuple[tuple[str, str, str], ...]:
    context = _project_run_design_context(project_run)
    return _comparison_metrics_for_context(context, unit=_project_field_display_unit(project_run.summary_rows))


def _comparison_metrics_for_context(context: Any, *, unit: str = "T") -> tuple[tuple[str, str, str], ...]:
    resolved = normalize_design_context(context)
    if resolved.enabled(EvaluationFeature.TARGET_MODE_FIT):
        return PROJECT_COMPARISON_METRICS
    if resolved.enabled(EvaluationFeature.SHIM_HOMOGENEITY):
        return (
            ("layout_needed_current_A", "Needed current", "A"),
            (f"shim_layout_residual_rms_{unit}", "Layout residual RMS", unit),
            (f"shim_layout_residual_max_abs_{unit}", "Layout residual max |Bz|", unit),
            (f"shim_layout_residual_peak_to_peak_{unit}", "Layout residual peak-to-peak", unit),
            ("coil_resistance_total_ohm", "Resistance", "Ohm"),
            ("coil_inductance_same_current_H", "Inductance", "H"),
        )
    return PROJECT_GENERIC_COMPARISON_METRICS


def _project_field_display_unit(rows: Sequence[Mapping[str, Any]]) -> str:
    for row in rows:
        unit = str(row.get("field_display_units", "")).strip()
        if unit in {"T", "Hz"}:
            return unit
    return "T"


def _project_run_design_context(project_run: CoilProjectRun) -> CoilDesignContext:
    if project_run.project.design_context is not None:
        return normalize_design_context(project_run.project.design_context)
    for row in project_run.summary_rows:
        value = str(row.get("design_context", "")).strip()
        if value:
            return normalize_design_context(value)
    return normalize_design_context("mr_gradient")


def _metric_values(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    colors: Sequence[str],
    label_map: Mapping[str, str] | None = None,
) -> tuple[list[str], list[float], list[str]]:
    labels: list[str] = []
    values: list[float] = []
    metric_colors: list[str] = []
    for index, row in enumerate(rows):
        value = _safe_float(row.get(key))
        if not math.isfinite(value):
            continue
        channel = str(row.get("channel", ""))
        labels.append(str(label_map.get(channel, channel)) if label_map is not None else channel)
        values.append(value)
        metric_colors.append(colors[index] if index < len(colors) else "#3b82f6")
    return labels, values, metric_colors


def _project_metric_bar_label_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    raw_labels = [_project_metric_raw_label(row) for row in rows]
    compact_labels = [_project_metric_compact_label(row, index) for index, row in enumerate(rows, start=1)]
    if len(set(compact_labels)) < len(compact_labels):
        compact_labels = [f"C{index}" for index in range(1, len(rows) + 1)]
    return {
        str(row.get("channel", "")): compact
        for row, raw, compact in zip(rows, raw_labels, compact_labels)
        if str(row.get("channel", "")) and raw
    }


def _project_metric_raw_label(row: Mapping[str, Any]) -> str:
    for key in ("label", "channel", "design_name"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def _project_metric_compact_label(row: Mapping[str, Any], index: int) -> str:
    axis = _project_metric_axis_label(row)
    role = _project_metric_role_label(row)
    if axis and role:
        return f"{axis}\n{role}"
    if axis:
        return axis
    raw = _project_metric_raw_label(row)
    if not raw:
        return f"C{index}"
    words = [part for part in raw.replace("-", "_").split("_") if part]
    if len(raw) <= 14:
        return raw
    if len(words) >= 2:
        tail = words[-1]
        head = "".join(part[:1].upper() for part in words[:-1] if part)[:4]
        compact = f"{head}\n{tail[:8]}"
        if len(compact.replace("\n", "")) <= 14:
            return compact
    return f"C{index}"


def _project_metric_axis_label(row: Mapping[str, Any]) -> str:
    for key in ("gradient_axis", "target_bz_expression"):
        value = str(row.get(key, "")).strip().lower()
        if value in {"x", "y", "z"}:
            return f"G{value}"
    raw = _project_metric_raw_label(row).lower().replace("-", "_")
    for token in ("gx", "gy", "gz"):
        if f"_{token}_" in f"_{raw}_" or raw.endswith(f"_{token}") or raw.startswith(f"{token}_"):
            return token.capitalize()
    return ""


def _project_metric_role_label(row: Mapping[str, Any]) -> str:
    texts = " ".join(
        str(row.get(key, "")).strip().lower().replace("-", "_") for key in ("role", "label", "channel", "design_name")
    )
    if "unshielded" in texts or "reference" in texts or texts.strip() == "ref":
        return "ref"
    if "active_shielded" in texts or "active shield" in texts:
        return "active"
    role = str(row.get("role", "")).strip().lower()
    if role == "main":
        return "active"
    if role:
        return role[:8]
    return ""


def _project_metric_bar_legend_handles(
    rows: Sequence[Mapping[str, Any]],
    colors: Sequence[str],
    label_map: Mapping[str, str],
) -> list[Any]:
    from matplotlib.patches import Patch

    handles: list[Any] = []
    for index, row in enumerate(rows):
        channel = str(row.get("channel", ""))
        compact = label_map.get(channel, "")
        raw = _project_metric_raw_label(row)
        if not compact or not raw or compact.replace("\n", " ") == raw:
            continue
        color = colors[index] if index < len(colors) else "#3b82f6"
        handles.append(
            Patch(
                facecolor=color,
                edgecolor="#1f2937",
                label=f"{compact.replace(chr(10), ' ')}: {_project_metric_descriptive_label(row)}",
            )
        )
    return handles


def _project_metric_descriptive_label(row: Mapping[str, Any]) -> str:
    label = str(row.get("label", "")).strip()
    if label:
        return label
    axis = _project_metric_axis_label(row)
    role = _project_metric_role_label(row)
    if axis and role == "active":
        return f"{axis} active shielded"
    if axis and role == "ref":
        return f"{axis} unshielded reference"
    if axis and role:
        return f"{axis} {role}"
    raw = _project_metric_raw_label(row).replace("_", " ")
    words = [word for word in raw.split() if word]
    return " ".join(words[-4:]) if len(words) > 4 else raw


def _channel_color(project_run: CoilProjectRun, channel_name: str) -> str:
    colors = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9", "#000000")
    for index, channel_run in enumerate(project_run.channel_runs):
        if channel_run.channel.name == channel_name:
            return channel_run.channel.color or colors[index % len(colors)]
    return colors[0]


def _project_plot_book_short_name(spec: Any, title: str, index: int, total: int) -> str:
    text = str(title or getattr(spec, "title", "") or getattr(spec, "key", "")).strip()
    channel = str(getattr(spec, "channel", "") or "").strip()
    if channel and text.startswith(f"{channel}:"):
        text = text[len(channel) + 1 :].strip()
    if total > 1 and not text.endswith(f"({index}/{total})"):
        text = f"{text} {index}/{total}"
    return " ".join(text.split()) or "plot"


def _summary_markdown(project_name: str, rows: Sequence[Mapping[str, Any]]) -> str:
    columns = (
        "channel",
        "quality_status",
        "target_bz_expression",
        "target_scale",
        "layout_needed_current_A",
        "layout_gradient_efficiency_mT_per_m_per_A",
        "layout_mean_rel_error_percent",
        "layout_max_rel_error_percent",
        "coil_inductance_same_current_H",
    )
    lines = [
        f"# {project_name} Project Summary",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_value(row.get(column, "")) for column in columns) + " |")
    lines.append("")
    return "\n".join(lines)


def _write_rows_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_ready(row.get(key, "")) for key in keys})


def _flatten_prefixed(prefix: str, values: Mapping[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in values.items():
        path = f"{prefix}.{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten_prefixed(path, value))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flat[path] = value
        else:
            flat[path] = _json_ready(value)
    return flat


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _csv_ready(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(_json_ready(value), sort_keys=True)


def _markdown_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    text = str(value)
    return text.replace("|", "\\|")


def _short_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    text = str(value)
    return text if len(text) <= 42 else text[:39] + "..."


def _project_summary_table_value(column: str, value: Any) -> str:
    if str(column) == "channel":
        return "" if value is None else str(value)
    return _short_value(value)


def _sequence_item(values: Sequence[Any] | None, index: int, default: Any) -> Any:
    if values is None:
        return default
    return values[index] if index < len(values) else default


def _column_label(value: str) -> str:
    labels = {
        "design_context": "context",
        "target_bz_expression": "target",
        "target_scale": "scale",
        "quality_status": "quality",
        "stream_function_needed_current_A": "sf current A",
        "stream_function_gradient_efficiency_mT_per_m_per_A": "sf eff mT/m/A",
        "stream_function_mean_rel_error_percent": "sf mean %",
        "stream_function_max_rel_error_percent": "sf max %",
        "unconnected_contours_needed_current_A": "loop current A",
        "unconnected_contours_gradient_efficiency_mT_per_m_per_A": "loop eff mT/m/A",
        "unconnected_contours_mean_rel_error_percent": "loop mean %",
        "unconnected_contours_max_rel_error_percent": "loop max %",
        "unconnected_contours_min_wire_distance_m": "loop min wire m",
        "layout_needed_current_A": "layout current A",
        "layout_gradient_efficiency_mT_per_m_per_A": "layout eff mT/m/A",
        "layout_mean_rel_error_percent": "layout mean %",
        "layout_max_rel_error_percent": "layout max %",
        "shim_layout_residual_rms_T": "shim RMS T",
        "shim_layout_residual_max_abs_T": "shim max T",
        "shim_layout_residual_peak_to_peak_T": "shim p2p T",
        "shim_layout_residual_rms_Hz": "shim RMS Hz",
        "shim_layout_residual_max_abs_Hz": "shim max Hz",
        "shim_layout_residual_peak_to_peak_Hz": "shim p2p Hz",
        "coil_length_total_m": "length m",
        "coil_resistance_total_ohm": "R ohm",
        "coil_inductance_same_current_H": "L H",
        "drive_voltage_target_slew_V": "V @ slew",
    }
    return labels.get(value, value)


def _project_section_column_label(section_title: str, column: str) -> str:
    labels = {
        "channel": "channel",
        "design_context": "context",
        "target_bz_expression": "target",
        "target_scale": "scale",
        "quality_status": "quality",
        "stream_function_needed_current_A": "current A",
        "stream_function_gradient_efficiency_mT_per_m_per_A": "eff mT/m/A",
        "stream_function_mean_rel_error_percent": "mean %",
        "stream_function_max_rel_error_percent": "max %",
        "unconnected_contours_needed_current_A": "current A",
        "unconnected_contours_gradient_efficiency_mT_per_m_per_A": "eff mT/m/A",
        "unconnected_contours_mean_rel_error_percent": "mean %",
        "unconnected_contours_max_rel_error_percent": "max %",
        "unconnected_contours_min_wire_distance_m": "min wire m",
        "layout_needed_current_A": "current A",
        "layout_gradient_efficiency_mT_per_m_per_A": "eff mT/m/A",
        "layout_mean_rel_error_percent": "mean %",
        "layout_max_rel_error_percent": "max %",
        "shim_layout_residual_rms_T": "RMS T",
        "shim_layout_residual_max_abs_T": "max |Bz| T",
        "shim_layout_residual_peak_to_peak_T": "p2p T",
        "shim_layout_residual_rms_Hz": "RMS Hz",
        "shim_layout_residual_max_abs_Hz": "max |Bz| Hz",
        "shim_layout_residual_peak_to_peak_Hz": "p2p Hz",
        "coil_length_total_m": "length m",
        "coil_resistance_total_ohm": "R ohm",
        "coil_inductance_same_current_H": "L H",
        "drive_voltage_target_slew_V": "V @ slew",
    }
    return labels.get(column, _column_label(column))


def _safe_file_stem(value: Any) -> str:
    text = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value).strip())
    text = "_".join(part for part in text.split("_") if part)
    return text[:80] or "coil_project"


def _safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")
