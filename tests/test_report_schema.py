from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from coilgen import (
    CoilGenConfig,
    REPORT_SCHEMA_VERSION,
    result_summary_dict,
    run_coilgen,
    validate_markdown_sections,
    validate_summary_schema,
    write_design_report,
    write_design_summary_csv,
)
from coilgen.inductance import calculate_inductance_by_coil_layout
from coilgen.models import ContourLine


def _small_report_config() -> CoilGenConfig:
    return CoilGenConfig(
        field_shape_function="y",
        coil_mesh_file="create cylinder mesh",
        cylinder_mesh_parameter_list=(0.2, 0.12, 8, 4, 1, 0, 0, 0),
        target_region_radius=0.035,
        target_region_resolution=3,
        levels=4,
        pot_offset_factor=0.25,
        min_point_loop_number=3,
        gauss_order=1,
        regularization_selection_method="fixed",
        tikonov_reg_factor=20.0,
        plot_flag=False,
        save_stl_flag=False,
        skip_postprocessing=True,
        skip_sweep=True,
        skip_inductance_calculation=True,
        low_memory_mode=True,
        retain_heavy_fields=False,
    )


def test_design_report_schema_has_stable_required_keys_and_sections(tmp_path: Path) -> None:
    result = run_coilgen(_small_report_config())
    summary = result_summary_dict(result)

    assert summary["report_schema_version"] == REPORT_SCHEMA_VERSION
    assert summary["software_name"] == "CoilGen2"
    assert summary["software_creator"] == "Philipp Amrein"
    assert summary["software_creator_role"] == "MRI hardware engineer"
    summary_validation = validate_summary_schema(summary)
    assert summary_validation.ok, summary_validation.missing_summary_keys

    report_path = write_design_report(result, tmp_path / "design_report.md")
    csv_path = write_design_summary_csv(result, tmp_path / "design_summary.csv")

    report_text = report_path.read_text(encoding="utf-8")
    assert "Generated with **CoilGen2**. Created and maintained by Philipp Amrein" in report_text
    assert "Amrein P, Jia F, Zaitsev M, Littin S." in report_text
    assert "https://doi.org/10.1002/mrm.29294" in report_text
    markdown_validation = validate_markdown_sections(report_text)
    assert markdown_validation.ok, markdown_validation.missing_markdown_sections
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "report_schema_version" in csv_text
    assert "current_density_A_per_mm2" in csv_text
    assert "engineering_readiness_status" in csv_text
    assert "software_creator,Philipp Amrein\n" in csv_text


def test_free_design_context_keeps_schema_but_skips_gradient_specific_report(tmp_path: Path) -> None:
    config = _small_report_config()
    config.design_context = "free"
    result = run_coilgen(config)
    summary = result_summary_dict(result)

    assert result.layout_gradient is None
    assert summary["design_context"] == "free"
    assert math.isnan(summary["layout_gradient_efficiency_mT_per_m_per_A"])
    assert summary["concomitant_gradient_source"] == "not_applicable_for_design_context"
    assert validate_summary_schema(summary).ok

    report_path = write_design_report(result, tmp_path / "free_report.md")
    report_text = report_path.read_text(encoding="utf-8")
    assert "Not applicable for design_context=free" in report_text
    assert validate_markdown_sections(report_text).ok


@pytest.mark.parametrize("skipped", (True, False))
def test_skipped_electrical_arrays_are_not_exported_as_zero_hardware(tmp_path: Path, skipped: bool) -> None:
    config = _small_report_config()
    config.skip_inductance_calculation = skipped
    result = run_coilgen(config)
    # A known closed contour keeps this reporting regression independent of
    # contour pruning by the deliberately tiny sheet fixture.
    angle = np.linspace(0, 2 * np.pi, 17)
    result.coil_parts[0].contour_lines = [
        ContourLine(np.column_stack((0.05 * np.cos(angle), 0.05 * np.sin(angle), np.zeros_like(angle))), 1.0)
    ]
    _, result.inductance_matrix, result.resistance, result.coil_lengths, _ = calculate_inductance_by_coil_layout(
        result.coil_parts, config
    )
    resistance_before = result.resistance.copy()
    inductance_before = result.inductance_matrix.copy()
    summary = result_summary_dict(result)
    assert summary["coil_electrical_calculation_status"] == ("skipped" if skipped else "enabled")
    assert summary["coil_length_total_m"] > 0
    keys = ("coil_resistance_ohm", "coil_resistance_total_ohm", "coil_inductance_H", "coil_inductance_same_current_H")
    if skipped:
        assert all(math.isnan(summary[key]) for key in keys)
        assert "coil_inductance_self_part1_H" not in summary
        assert not any(key.startswith("coil_mutual_inductance_") for key in summary)
        assert not np.any(resistance_before) and not np.any(inductance_before)
    else:
        assert all(summary[key] > 0 for key in keys)
        assert summary["coil_inductance_self_part1_H"] == inductance_before[0, 0]
    text = write_design_report(result, tmp_path / "electrical.md").read_text(encoding="utf-8")
    assert ("## Inductance Matrix" in text) is (not skipped)
    csv_text = write_design_summary_csv(result, tmp_path / "electrical.csv").read_text(encoding="utf-8")
    assert f"coil_electrical_calculation_status,{'skipped' if skipped else 'enabled'}" in csv_text
    np.testing.assert_array_equal(result.resistance, resistance_before)
    np.testing.assert_array_equal(result.inductance_matrix, inductance_before)
