from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


REPORT_SCHEMA_VERSION = "coilgen-report-v1"

REQUIRED_SUMMARY_KEYS: tuple[str, ...] = (
    "report_schema_version",
    "design_context",
    "part_count",
    "target_point_count",
    "dsv_target_point_count",
    "contour_count",
    "needed_current_A",
    "coil_length_total_m",
    "coil_resistance_total_ohm",
    "coil_inductance_same_current_H",
    "layout_gradient_efficiency_mT_per_m_per_A",
    "layout_needed_current_A",
    "layout_mean_rel_error_percent",
    "layout_max_rel_error_percent",
    "concomitant_model",
    "concomitant_sample_count",
    "concomitant_rms_T",
    "concomitant_max_abs_T",
    "suppression_clearance_status",
    "suppression_clearance_min_m",
    "suppression_clearance_violation_count",
    "current_density_A_per_mm2",
    "current_density_class",
    "thermal_screening_status",
    "engineering_readiness_status",
    "finished_engineering_design_ready",
    "execution_time_s",
)

REQUIRED_MARKDOWN_SECTIONS: tuple[str, ...] = (
    "## Magnetostatic Feasibility",
    "## Layout Feasibility",
    "## Engineering Checks Not Included",
    "## Suppression-Clearance Report",
    "## Concomitant Field Report",
    "## Current-Density Report",
)


@dataclass(frozen=True, slots=True)
class ReportSchemaValidation:
    """Machine-checkable report schema validation result."""

    schema_version: str
    missing_summary_keys: tuple[str, ...] = ()
    missing_markdown_sections: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.missing_summary_keys and not self.missing_markdown_sections


def validate_summary_schema(summary: Mapping[str, Any]) -> ReportSchemaValidation:
    """Validate stable summary keys used by CSV and Markdown reports."""

    missing = tuple(key for key in REQUIRED_SUMMARY_KEYS if key not in summary)
    return ReportSchemaValidation(
        schema_version=str(summary.get("report_schema_version", "")),
        missing_summary_keys=missing,
    )


def validate_markdown_sections(
    markdown_text: str, sections: Sequence[str] = REQUIRED_MARKDOWN_SECTIONS
) -> ReportSchemaValidation:
    """Validate that a Markdown design report still contains required sections."""

    missing = tuple(section for section in sections if section not in markdown_text)
    return ReportSchemaValidation(
        schema_version=REPORT_SCHEMA_VERSION,
        missing_markdown_sections=missing,
    )
