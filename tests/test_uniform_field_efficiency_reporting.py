"""Uniform Bz efficiency has field-per-current units, without a length factor."""

import csv
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen import CoilGenConfig, run_coilgen
from coilgen.designs import CoilGenConfigDesign
from coilgen.field_units import PROTON_GYROMAGNETIC_RATIO_HZ_PER_T
from coilgen.metric_definitions import DEFAULT_METRIC_REGISTRY
from coilgen.metrics import target_mode_fit_performance
from coilgen.project import (
    CoilChannelRun,
    CoilChannelSpec,
    CoilProject,
    CoilProjectRun,
    _project_channel_summary_columns,
    _project_channel_summary_sections,
    _project_section_column_label,
    write_project_summary,
)
from coilgen.reporting import (
    result_performance_table_rows,
    result_summary_dict,
    target_mode_not_applicable_summary_dict,
    target_mode_summary_dict,
    write_design_report,
    write_design_summary_csv,
)
from coilgen.study import study_result_from_coilgen_result
from coilgen.target import define_target_field
from coilgen.workflow import CoilCaseSpec, CoilVariantSpec, write_variant_run_record


@pytest.mark.parametrize("target_scale", [1e-4, -1e-4, 7.5, 4257.747892])
@pytest.mark.parametrize("polarity", [1, -1])
def test_uniform_bz_reports_physical_field_efficiency_independent_of_target_scale(target_scale, polarity):
    # This numerical T/A dataset has a known mean and deliberate nonuniformity.
    field = polarity * np.array([60, 64, 66], dtype=float) * 1e-6
    target = np.full(3, 1e-4)
    expected_current = target[0] / np.mean(field)
    expected_residual = 100 * np.abs(expected_current * field - target) / target[0]

    fit = target_mode_fit_performance(field, target, target_scale)

    assert fit["field_efficiency_mT_per_A"] == pytest.approx(0.06333333333333334)
    assert np.isnan(fit["gradient_efficiency_mT_per_m_per_A"])
    assert fit["needed_current_A"] == pytest.approx(expected_current)
    assert fit["mean_rel_error_percent"] == pytest.approx(np.mean(expected_residual))
    assert fit["max_rel_error_percent"] == pytest.approx(np.max(expected_residual))
    assert fit["sample_count"] == 3


def test_uniform_detection_uses_finite_selected_samples_and_excludes_suppression_zeros():
    target = np.array([1e-4, 1e-4, 1e-4, 0, 0, np.nan])
    field = np.zeros((6, 3))
    field[:, 2] = [60e-6, 64e-6, 66e-6, 40, -30, 1e6]
    mask = np.array([True, True, True, False, False, True])

    fit = target_mode_fit_performance(field, target, 1e-4, sample_mask=mask)

    assert fit["sample_count"] == 3
    assert fit["field_efficiency_mT_per_A"] == pytest.approx(0.06333333333333334)
    assert fit["needed_current_A"] == pytest.approx(1e-4 / np.mean(field[:3, 2]))
    assert np.isnan(fit["gradient_efficiency_mT_per_m_per_A"])


@pytest.mark.parametrize(
    "field,target,scale,mask",
    [
        (None, np.ones(3), 1, None),
        (np.ones(2), np.ones(3), 1, None),
        (np.ones(3), np.ones(3), 1, np.ones(2, bool)),
        (np.ones(3), np.ones(3), 1, np.zeros(3, bool)),
        (np.full(3, np.nan), np.ones(3), 1, None),
        (np.ones(3), np.full(3, np.nan), 1, None),
        (np.zeros(3), np.ones(3), 1, None),
        (np.ones(3), np.zeros(3), 1, None),
        (np.ones(3), np.full(3, 1e-40), 1, None),
        (np.ones(3), np.ones(3), 0, None),
        (np.ones(3), np.ones(3), 1e-40, None),
        (np.ones(3), np.ones(3), np.nan, None),
    ],
)
def test_unavailable_or_zero_fit_does_not_publish_an_efficiency(field, target, scale, mask):
    fit = target_mode_fit_performance(field, target, scale, sample_mask=mask)
    for key in ("needed_current_A", "field_efficiency_mT_per_A", "gradient_efficiency_mT_per_m_per_A"):
        assert np.isnan(fit[key])
    assert fit["sample_count"] == 0


def test_linear_gradient_preserves_its_efficiency_current_and_errors():
    target_mode = np.array([-1, 0, 1, 0], dtype=float)
    target = 2 * target_mode
    field = 0.5 * target_mode + np.array([0, 40, 0, -40])

    fit = target_mode_fit_performance(field, target, 2)

    assert fit["gradient_efficiency_mT_per_m_per_A"] == 500
    assert np.isnan(fit["field_efficiency_mT_per_A"])
    assert fit["needed_current_A"] == 4
    assert fit["mean_rel_error_percent"] == 4000
    assert fit["max_rel_error_percent"] == 8000


def test_uniform_summary_uses_only_dsv_samples_and_publishes_field_units():
    field = np.zeros((4, 3))
    field[:, 2] = [60e-6, 66e-6, 40, -30]
    target = field.copy()
    target[:, 2] = [1e-4, 1e-4, 0, 0]
    result = SimpleNamespace(
        target_field=SimpleNamespace(
            coords=np.zeros((4, 3)), b=target, suppressed=np.array([False, False, True, True])
        ),
        input_data=CoilGenConfig(target_gradient_strength=7.5),
        field_layout_per1Amp=field,
        field_loops_per1Amp=field,
        b_field_opt_sf_1A=field,
    )

    summary = target_mode_summary_dict(result)

    for source in ("layout", "unconnected_contours", "stream_function"):
        assert summary[f"{source}_field_efficiency_mT_per_A"] == pytest.approx(0.063)
        assert np.isnan(summary[f"{source}_gradient_efficiency_mT_per_m_per_A"])
        assert summary[f"{source}_needed_current_A"] == pytest.approx(1e-4 / 63e-6)
        assert summary[f"{source}_fit_sample_count"] == 2


@pytest.fixture(scope="module")
def uniform_result():
    return run_coilgen(
        CoilGenConfig(
            coil_mesh_file="create cylinder mesh",
            cylinder_mesh_parameter_list=(0.30, 0.08, 12, 6, 1, 0, 0, 0),
            surface_is_cylinder_flag=True,
            field_shape_function="1",
            target_gradient_strength=1e-4,
            target_region_radius=0.025,
            target_region_resolution=7,
            regularization_selection_method="fixed",
            tikonov_reg_factor=20,
            levels=8,
            min_point_loop_number=3,
            gauss_order=3,
            skip_postprocessing=True,
            skip_sweep=True,
            skip_inductance_calculation=True,
            plot_flag=False,
            save_stl_flag=False,
            progress_flag=False,
        )
    )


def test_computed_constant_bz_summary_and_display_rows_use_field_efficiency(uniform_result):
    result = uniform_result
    current_before = result.needed_current_layout
    errors_before = dict(result.error_vals)
    summary = result_summary_dict(result)
    rows = result_performance_table_rows(result)

    for source, field in (
        ("layout", result.field_layout_per1Amp),
        ("unconnected_contours", result.field_loops_per1Amp),
        ("stream_function", result.b_field_opt_sf_1A),
    ):
        assert summary[f"{source}_field_efficiency_mT_per_A"] == pytest.approx(1000 * abs(np.mean(field[:, 2])))
        assert np.isnan(summary[f"{source}_gradient_efficiency_mT_per_m_per_A"])
    efficiency_rows = [row for row in rows if "efficiency" in row[1]]
    assert {row[1] for row in efficiency_rows} == {
        "connected-wire field efficiency",
        "unconnected-contour field efficiency",
        "stream-function field efficiency",
    }
    assert all(row[3] == "mT A^-1" for row in efficiency_rows)
    assert result.needed_current_layout == current_before
    assert result.error_vals == errors_before


def test_not_applicable_context_exposes_unavailable_field_keys(uniform_result):
    result = replace(uniform_result, input_data=replace(uniform_result.input_data, design_context="free"))
    summary = result_summary_dict(result)
    unavailable = target_mode_not_applicable_summary_dict(result)
    for source in ("layout", "unconnected_contours", "stream_function"):
        key = f"{source}_field_efficiency_mT_per_A"
        assert np.isnan(summary[key])
        assert np.isnan(unavailable[key])


def test_raw_csv_and_markdown_exports_retain_physical_field_efficiency(uniform_result, tmp_path):
    summary = result_summary_dict(uniform_result)
    csv_path = write_design_summary_csv(uniform_result, tmp_path / "uniform.csv")
    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_values = dict(list(csv.reader(handle))[1:])
    for source in ("layout", "unconnected_contours", "stream_function"):
        key = f"{source}_field_efficiency_mT_per_A"
        assert float(csv_values[key]) == summary[key]
        assert np.isnan(float(csv_values[f"{source}_gradient_efficiency_mT_per_m_per_A"]))
    markdown = write_design_report(uniform_result, tmp_path / "uniform.md").read_text(encoding="utf-8")
    feasibility = markdown.split("## Magnetostatic Feasibility\n", 1)[1].split("\n## ", 1)[0]
    assert "layout_field_efficiency_mT_per_A" in feasibility
    assert "gradient_efficiency" not in feasibility


def test_project_exports_and_plot_table_register_field_efficiency(uniform_result, tmp_path):
    design = CoilGenConfigDesign(config=uniform_result.input_data, target_bz_expression="1", target_scale=1e-4)
    channel = CoilChannelRun(
        CoilChannelSpec("Z0"), design, artifacts=SimpleNamespace(result=uniform_result, quality=None)
    )
    row = channel.summary_row()
    project = CoilProject("uniform", template=design)
    run = CoilProjectRun(project, tmp_path, (channel,), (row,))
    csv_path, json_path, markdown_path = write_project_summary(run)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_row = next(csv.DictReader(handle))
    json_row = json.loads(json_path.read_text(encoding="utf-8"))["rows"][0]
    columns = _project_channel_summary_columns(run)
    sections = dict(_project_channel_summary_sections(run))
    for source, section in (
        ("layout", "Connected Layout"),
        ("unconnected_contours", "Unconnected Contours"),
        ("stream_function", "Stream Function"),
    ):
        key = f"{source}_field_efficiency_mT_per_A"
        assert float(csv_row[key]) == json_row[key] == row[key]
        assert key in columns and key in sections[section]
        assert _project_section_column_label(section, key) == "eff mT/A"
    assert "layout_field_efficiency_mT_per_A" in markdown_path.read_text(encoding="utf-8")


def test_run_record_json_and_csv_retain_all_field_efficiency_sources(uniform_result, tmp_path):
    design = CoilGenConfigDesign(config=uniform_result.input_data)
    variant = CoilVariantSpec("uniform", design)
    case = CoilCaseSpec("uniform", (variant,), tmp_path)
    json_path, csv_path, _, markdown_path = write_variant_run_record(uniform_result, variant, case, tmp_path)
    json_values = json.loads(json_path.read_text(encoding="utf-8"))["metrics"]
    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_values = next(csv.DictReader(handle))
    markdown = markdown_path.read_text(encoding="utf-8")
    for source in ("layout", "unconnected_contours", "stream_function"):
        key = f"{source}_field_efficiency_mT_per_A"
        assert float(csv_values[key]) == json_values[key]
        assert key in markdown


def test_typed_study_metric_and_registry_declare_field_efficiency_units(uniform_result):
    study = study_result_from_coilgen_result("uniform", {}, uniform_result)
    key = "layout_field_efficiency_mT_per_A"
    metric = study.metrics[key]
    definition = DEFAULT_METRIC_REGISTRY.get(key)
    assert metric.value == result_summary_dict(uniform_result)[key]
    assert metric.unit == definition.unit == "mT A^-1"
    assert np.isnan(study.metrics["layout_gradient_efficiency_mT_per_m_per_A"].value)


@pytest.mark.parametrize(
    "expression,normalization,units,scale,axis,expected",
    [
        ("x**2", "auto", "T", 0.01, "auto", None),
        ("y+0.1*y**3", "auto", "T", 0.01, "auto", 5.0),
        ("y", "auto", "Hz", 20.0, "auto", 1000 * 10 / PROTON_GYROMAGNETIC_RATIO_HZ_PER_T),
        ("y", "none", "T", 7.5, "auto", 500.0),
        ("3*y", "scale", "T", 0.02, "auto", 30.0),
        ("x+2*y", "auto", "T", 0.01, "x", 5.0),
        ("x+2*y", "auto", "T", 0.01, "magnitude", 5.0),
    ],
)
def test_summary_gradient_efficiency_uses_physical_affine_target_strength(
    expression, normalization, units, scale, axis, expected
):
    config = CoilGenConfig(
        field_shape_function=expression,
        target_gradient_strength=scale,
        target_gradient_normalization=normalization,
        target_gradient_axis=axis,
        target_field_units=units,
        target_region_radius=0.05,
        target_region_resolution=15,
    )
    target = define_target_field([], None, None, config)
    # An exact two-amp target-mode dataset isolates reporting from the solver.
    field = target.b / 2
    unchanged_fit = target_mode_fit_performance(field, target.b[:, 2], scale)
    result = SimpleNamespace(
        target_field=target,
        input_data=config,
        field_layout_per1Amp=field,
        field_loops_per1Amp=field,
        b_field_opt_sf_1A=field,
    )

    summary = target_mode_summary_dict(result)

    for source in ("layout", "unconnected_contours", "stream_function"):
        efficiency = summary[f"{source}_gradient_efficiency_mT_per_m_per_A"]
        if expected is None:
            assert target.target_diagnostics["target_gradient_normalization_fallback"]
            assert np.isnan(efficiency)
        else:
            assert efficiency == pytest.approx(expected)
        assert np.isnan(summary[f"{source}_field_efficiency_mT_per_A"])
        assert summary[f"{source}_needed_current_A"] == unchanged_fit["needed_current_A"]
        assert summary[f"{source}_mean_rel_error_percent"] == unchanged_fit["mean_rel_error_percent"]
        assert summary[f"{source}_max_rel_error_percent"] == unchanged_fit["max_rel_error_percent"]


def test_imported_unscaled_gradient_uses_si_target_and_excludes_suppression_regions():
    config = CoilGenConfig(
        target_gradient_strength=7.5,
        target_gradient_normalization="none",
        target_field_definition_file="imported-target.npz",
    )
    coords = np.array([[0, -0.02, 0], [0, 0.02, 0], [0.01, 0.01, 0.01], [0, 0, -0.01], [0, 40, 0]])
    target = np.zeros_like(coords)
    target[:4, 2] = 0.03 * coords[:4, 1]
    field = target / 2
    field[-1, 2] = 100
    target_field = SimpleNamespace(
        coords=coords, b=target, suppressed=np.array([False, False, False, False, True]), target_diagnostics={}
    )
    result = SimpleNamespace(
        target_field=target_field,
        input_data=config,
        field_layout_per1Amp=field,
        field_loops_per1Amp=field,
        b_field_opt_sf_1A=field,
    )
    unchanged_fit = target_mode_fit_performance(field, target[:, 2], 7.5, sample_mask=~target_field.suppressed)

    summary = target_mode_summary_dict(result)

    for source in ("layout", "unconnected_contours", "stream_function"):
        assert summary[f"{source}_gradient_efficiency_mT_per_m_per_A"] == pytest.approx(15.0)
        assert summary[f"{source}_needed_current_A"] == unchanged_fit["needed_current_A"]
        assert summary[f"{source}_fit_sample_count"] == 4


def test_nonlinear_target_without_affine_gradient_omits_efficiency_display_rows(uniform_result):
    config = replace(uniform_result.input_data, field_shape_function="x**2", target_gradient_strength=0.01)
    target = define_target_field([], None, None, config)
    field = target.b / 2
    result = replace(
        uniform_result,
        input_data=config,
        target_field=target,
        target_field_1A=target,
        field_by_layout=target.b,
        field_by_unconnected_loops=target.b,
        field_layout_per1Amp=field,
        field_loops_per1Amp=field,
        b_field_opt_sf=target.b,
        b_field_opt_sf_1A=field,
        needed_current_layout=2,
    )

    rows = result_performance_table_rows(result)

    assert not any("efficiency" in row[1] for row in rows)


@pytest.mark.parametrize("axis", ["x", "y", "z", "magnitude"])
@pytest.mark.parametrize("offset", [np.zeros(3), np.array([0.2, -0.3, 0.7]), np.full(3, 1e6)])
def test_planar_dsv_reports_only_identifiable_gradient_components(axis, offset):
    xy = np.array(np.meshgrid([-0.02, 0, 0.02], [-0.03, 0, 0.03])).reshape(2, -1).T
    points = np.column_stack((xy, np.zeros(len(xy)))) + offset
    target = np.zeros_like(points)
    target[:, 2] = 0.03 * (points[:, 0] - offset[0]) + 0.04 * (points[:, 1] - offset[1]) + 0.001
    config = CoilGenConfig(
        target_field_definition_file="planar-target.npz", target_gradient_strength=7.5, target_gradient_axis=axis
    )
    target_field = SimpleNamespace(coords=points, b=target, suppressed=np.zeros(len(points), bool))
    field = target / 2
    fit_before = target_mode_fit_performance(field, target[:, 2], config.target_gradient_strength)
    result = SimpleNamespace(
        target_field=target_field,
        input_data=config,
        field_layout_per1Amp=field,
        field_loops_per1Amp=field,
        b_field_opt_sf_1A=field,
    )

    summary = target_mode_summary_dict(result)

    for source in ("layout", "unconnected_contours", "stream_function"):
        efficiency = summary[f"{source}_gradient_efficiency_mT_per_m_per_A"]
        if axis in {"x", "y"}:
            assert efficiency == pytest.approx({"x": 15, "y": 20}[axis])
        else:
            assert np.isnan(efficiency)
        assert summary[f"{source}_needed_current_A"] == fit_before["needed_current_A"]
        assert summary[f"{source}_mean_rel_error_percent"] == fit_before["mean_rel_error_percent"]
        assert summary[f"{source}_max_rel_error_percent"] == fit_before["max_rel_error_percent"]


@pytest.mark.parametrize("length_scale", [1e-6, 1.0, 1e6])
def test_diagonal_line_dsv_cannot_identify_cartesian_gradient_components(length_scale):
    axis = length_scale * np.linspace(-0.02, 0.02, 9)
    points = np.column_stack((axis, axis, np.zeros_like(axis)))
    target = np.zeros_like(points)
    target[:, 2] = 0.03 * points[:, 0]
    target_field = SimpleNamespace(coords=points, b=target, suppressed=np.zeros(len(points), bool))
    field = target / 2
    config = CoilGenConfig(target_field_definition_file="line-target.npz", target_gradient_axis="y")
    fit_before = target_mode_fit_performance(field, target[:, 2], config.target_gradient_strength)
    result = SimpleNamespace(
        target_field=target_field,
        input_data=config,
        field_layout_per1Amp=field,
        field_loops_per1Amp=field,
        b_field_opt_sf_1A=field,
    )

    summary = target_mode_summary_dict(result)

    for source in ("layout", "unconnected_contours", "stream_function"):
        assert np.isnan(summary[f"{source}_gradient_efficiency_mT_per_m_per_A"])
        assert summary[f"{source}_needed_current_A"] == fit_before["needed_current_A"]
        assert summary[f"{source}_mean_rel_error_percent"] == fit_before["mean_rel_error_percent"]
        assert summary[f"{source}_max_rel_error_percent"] == fit_before["max_rel_error_percent"]


def test_large_translation_does_not_mix_intercept_into_reported_gradient():
    offsets = np.full(3, 1e6)
    points = np.array(np.meshgrid(*([[-0.02, 0, 0.02]] * 3))).reshape(3, -1).T + offsets
    target = np.zeros_like(points)
    target[:, 2] = 0.03 * (points[:, 1] - offsets[1])
    target_field = SimpleNamespace(coords=points, b=target, suppressed=np.zeros(len(points), bool))
    field = target / 2
    result = SimpleNamespace(
        target_field=target_field,
        input_data=CoilGenConfig(target_field_definition_file="translated-target.npz", target_gradient_axis="y"),
        field_layout_per1Amp=field,
        field_loops_per1Amp=field,
        b_field_opt_sf_1A=field,
    )

    summary = target_mode_summary_dict(result)

    assert summary["layout_gradient_efficiency_mT_per_m_per_A"] == pytest.approx(15)
    assert summary["layout_needed_current_A"] == pytest.approx(2)
