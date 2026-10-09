from __future__ import annotations

import math
from pathlib import Path

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
