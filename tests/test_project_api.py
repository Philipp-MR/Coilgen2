from __future__ import annotations

import unittest
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from coilgen import (
    CoilApplication,
    CoilChannel,
    CoilChannelSpec,
    CoilDesignContext,
    CoilGenConfig,
    CoilGenConfigDesign,
    CoilProject,
    CoilProjectParameters,
    CoilProjectRun,
    CylinderCoilDesign,
    EvaluationFeature,
    GradientLinearitySpec,
    GradientSliceMapSpec,
    SystemParameters,
    normalize_design_context,
)
from projects.standard_cylindrical_3axis_gradient import (
    run_standard_cylindrical_3axis_gradient as standard_gradient_script,
)
from projects.standard_cylindrical_3axis_gradient.gradient_project import build_gradient_project
from coilgen.project import (
    _project_channel_summary_columns,
    _project_channel_summary_sections,
    _project_drive_metrics_from_summary,
    _project_summary_section_cell_text,
    _channel_prefixed_plot_spec,
)
from projects._shared.example_utils import open_project_plot_book_after_run


class FakeQuality:
    status = "ok"


class FakeArtifacts:
    result = None
    quality = FakeQuality()
    study_result = None
    files: tuple[Path, ...] = ()
    figures: tuple[Any, ...] = ()


class ProjectApiTest(unittest.TestCase):
    def test_standard_gradient_plotbook_includes_current_flow_pages(self) -> None:
        self.assertIn("wire_sequence", standard_gradient_script.PLOT_INCLUDE)
        self.assertIn("contour_current_orientation", standard_gradient_script.PLOT_INCLUDE)
        self.assertIn("wire_sequence", standard_gradient_script.PROJECT.plot_include)
        self.assertIn("contour_current_orientation", standard_gradient_script.PROJECT.plot_include)

    def test_open_project_plot_book_after_run_reuses_or_builds_book(self) -> None:
        class FakeBook:
            def __init__(self) -> None:
                self.opened = False

            def open(self, *, block: bool = False) -> None:
                self.opened = not block

        class FakeRun:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []
                self.book = FakeBook()

            def show_plot_book(self, **kwargs: Any) -> FakeBook:
                self.calls.append(kwargs)
                return self.book

        existing = FakeBook()
        run = FakeRun()

        # Simulate an interactive session even when the suite uses Agg.
        with patch.dict(os.environ, {"MPLBACKEND": "qtagg"}):
            reused = open_project_plot_book_after_run(run, figures=[existing], include=("wire",))
            built = open_project_plot_book_after_run(run, figures=[], include=("wire",), close_previous=True)

        self.assertIs(reused, existing)
        self.assertTrue(existing.opened)
        self.assertIs(built, run.book)
        self.assertEqual(run.calls[0]["include"], ("wire",))
        self.assertTrue(run.calls[0]["close_previous"])

    def test_channel_resolution_keeps_template_unchanged(self) -> None:
        template = CylinderCoilDesign(
            name="base",
            target_bz_expression="y",
            target_scale=0.01,
            contour_levels=8,
            radius_m=0.25,
        )
        project = CoilProject("three_axis", template=template)
        project.channel(
            "Gx",
            target_bz_expression="x",
            target_scale=0.02,
            contour_levels=12,
            color="tab:blue",
        )

        designs = project.resolved_designs()

        self.assertEqual(designs["Gx"].name, "three_axis_Gx")
        self.assertEqual(designs["Gx"].target_bz_expression, "x")
        self.assertEqual(designs["Gx"].target_scale, 0.02)
        self.assertEqual(designs["Gx"].contour_levels, 12)
        self.assertEqual(designs["Gx"].radius_m, 0.25)
        self.assertEqual(template.name, "base")
        self.assertEqual(template.target_bz_expression, "y")
        self.assertEqual(template.contour_levels, 8)

    def test_project_overrides_are_shared_and_channel_overrides_are_specific(self) -> None:
        template = CylinderCoilDesign(
            name="base",
            target_bz_expression="y",
            target_scale=0.01,
            contour_levels=8,
            min_turn_spacing_m=0.0,
            advanced_parameters={"shared_solver_flag": True},
        )
        project = CoilProject(
            "override_precedence",
            template=template,
            design_overrides={
                "contour_levels": 10,
                "min_turn_spacing_m": 0.002,
                "advanced_parameters": {"project_solver_flag": True},
            },
        )
        project.channel("Gx", target_bz_expression="x")
        project.channel("Gy", target_bz_expression="y", contour_levels=14)

        designs = project.resolved_designs()

        self.assertEqual(designs["Gx"].contour_levels, 10)
        self.assertEqual(designs["Gy"].contour_levels, 14)
        self.assertEqual(designs["Gx"].min_turn_spacing_m, 0.002)
        self.assertEqual(designs["Gy"].min_turn_spacing_m, 0.002)
        self.assertEqual(designs["Gx"].target_bz_expression, "x")
        self.assertEqual(designs["Gy"].target_bz_expression, "y")
        self.assertEqual(designs["Gx"].advanced_parameters, {"project_solver_flag": True})
        self.assertEqual(designs["Gy"].advanced_parameters, {"project_solver_flag": True})
        self.assertEqual(template.contour_levels, 8)
        self.assertEqual(template.min_turn_spacing_m, 0.0)
        self.assertEqual(template.advanced_parameters, {"shared_solver_flag": True})

    def test_project_parameters_are_shared_and_channel_overrides_still_win(self) -> None:
        template = CylinderCoilDesign(
            name="base",
            target_bz_expression="y",
            target_scale=0.01,
            contour_levels=8,
            radius_m=0.25,
            angular_divisions=6,
            length_divisions=6,
        )
        project = CoilProject("nested_parameters", template=template)
        self.assertIsInstance(project.parameters, CoilProjectParameters)
        self.assertIsInstance(project.parameters.system, SystemParameters)
        project.parameters.mesh.mesh_size = 11
        project.parameters.mesh.radius_m = 0.31
        project.parameters.system.main_field_strength_T = 3.0
        project.parameters.system.b0_direction = (1.0, 0.0, 0.0)
        project.parameters.optimization.contour_levels = 10
        project.parameters.optimization.contour_discretization_method = "engineering"
        project.parameters.conductor.max_current_A = 2.5
        project.parameters.conductor.min_wire_distance_m = 0.003
        project.channel("Gx", target_bz_expression="x")
        project.channel(
            "Gy",
            target_bz_expression="y",
            length_divisions=14,
            contour_levels=12,
            max_current_A=3.5,
            main_field_strength_T=7.0,
        )

        designs = project.resolved_designs()

        self.assertEqual(designs["Gx"].angular_divisions, 11)
        self.assertEqual(designs["Gx"].length_divisions, 11)
        self.assertEqual(designs["Gy"].angular_divisions, 11)
        self.assertEqual(designs["Gy"].length_divisions, 14)
        self.assertEqual(designs["Gx"].radius_m, 0.31)
        self.assertEqual(designs["Gy"].radius_m, 0.31)
        self.assertEqual(designs["Gx"].contour_levels, 10)
        self.assertEqual(designs["Gy"].contour_levels, 12)
        self.assertEqual(designs["Gx"].contour_discretization_method, "engineering")
        self.assertEqual(designs["Gy"].contour_discretization_method, "engineering")
        self.assertEqual(designs["Gx"].max_current_A, 2.5)
        self.assertEqual(designs["Gy"].max_current_A, 3.5)
        self.assertEqual(designs["Gx"].min_wire_distance_m, 0.003)
        self.assertEqual(designs["Gy"].min_wire_distance_m, 0.003)
        self.assertEqual(designs["Gx"].main_field_strength_T, 3.0)
        self.assertEqual(designs["Gy"].main_field_strength_T, 7.0)
        self.assertEqual(designs["Gx"].b0_direction, (1.0, 0.0, 0.0))
        self.assertEqual(designs["Gy"].b0_direction, (1.0, 0.0, 0.0))
        self.assertEqual(template.radius_m, 0.25)
        self.assertEqual(template.length_divisions, 6)

    def test_project_parameters_can_be_created_from_nested_mapping(self) -> None:
        template = CylinderCoilDesign(angular_divisions=6, length_divisions=6)
        project = CoilProject(
            "mapped_parameters",
            template=template,
            parameters={"mesh": {"mesh_size": 9}, "target": {"scale": 0.02}},
        )
        project.channel("Gx", target_bz_expression="x")

        design = project.resolved_designs()["Gx"]

        self.assertEqual(design.angular_divisions, 9)
        self.assertEqual(design.length_divisions, 9)
        self.assertEqual(design.target_scale, 0.02)

    def test_design_context_normalization_exposes_evaluation_features(self) -> None:
        context = normalize_design_context("b0_shim")

        self.assertEqual(context.application, CoilApplication.MR_SHIM)
        self.assertTrue(context.is_mr)
        self.assertTrue(context.enabled(EvaluationFeature.SHIM_HOMOGENEITY))
        self.assertTrue(context.enabled("shim_spherical_harmonics"))
        self.assertFalse(context.enabled(EvaluationFeature.GRADIENT_FIT))

    def test_project_design_context_flows_to_design_config_and_channel_override(self) -> None:
        project = CoilProject(
            "context_project",
            template=CylinderCoilDesign(angular_divisions=6, length_divisions=6),
            design_context=CoilDesignContext.mr_shim(),
        )
        project.channel("B0", target_bz_expression="1")
        project.channel("Gx", target_bz_expression="x", design_context="mr_gradient")

        designs = project.resolved_designs()

        self.assertEqual(project.design_context, "mr_shim")
        self.assertEqual(designs["B0"].design_context, "mr_shim")
        self.assertEqual(designs["B0"].to_config().design_context, "mr_shim")
        self.assertEqual(designs["Gx"].design_context, "mr_gradient")
        self.assertEqual(designs["Gx"].to_config().design_context, "mr_gradient")

    def test_resolved_channel_designs_do_not_share_mutable_template_state(self) -> None:
        template = CylinderCoilDesign(advanced_parameters={"cache": {"enabled": True}})
        project = CoilProject("mutable_isolation", template=template)
        project.channel("Gx", target_bz_expression="x")
        project.channel("Gy", target_bz_expression="y")

        designs = project.resolved_designs()
        designs["Gx"].advanced_parameters["cache"]["enabled"] = False

        self.assertEqual(designs["Gy"].advanced_parameters["cache"]["enabled"], True)
        self.assertEqual(template.advanced_parameters["cache"]["enabled"], True)

    def test_duplicate_channel_names_are_rejected_case_insensitively(self) -> None:
        project = CoilProject("duplicate_check", template=CylinderCoilDesign())
        project.channel("Gx", target_bz_expression="x")

        with self.assertRaisesRegex(ValueError, "Duplicate CoilProject channel name"):
            project.channel("gx", target_bz_expression="x")

    def test_unknown_dataclass_override_is_reported_with_channel_name(self) -> None:
        self.assertIs(CoilChannel, CoilChannelSpec)
        channel = CoilChannel("Gx", design_overrides={"not_a_field": 1})

        with self.assertRaisesRegex(ValueError, "Gx.*not_a_field"):
            channel.resolve_design(CylinderCoilDesign(), project_name="project")

    def test_project_run_can_continue_after_channel_failure_and_write_summary(self) -> None:
        def runner(design: Any, _options: Any) -> FakeArtifacts:
            if getattr(design, "target_bz_expression", "") == "y":
                raise RuntimeError("intentional failure")
            return FakeArtifacts()

        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            project = CoilProject(
                "run_failure_case",
                template=CylinderCoilDesign(),
                project_root=tmp,
                output_dir=output_dir,
                continue_on_channel_failure=True,
            )
            project.channel("Gx", target_bz_expression="x")
            project.channel("Gy", target_bz_expression="y")

            run = project.solve(runner=runner)

            self.assertIsInstance(run, CoilProjectRun)
            self.assertIs(project.last_run, run)
            self.assertEqual(set(run.results), set())
            self.assertEqual(project.results, {})
            self.assertEqual(set(run.failures), {"Gy"})
            self.assertEqual(set(project.failures), {"Gy"})
            self.assertEqual(len(run.summary_rows), 2)
            self.assertEqual(project.summary_rows, run.summary_rows)
            self.assertEqual(run.summary_rows[0]["status"], "success")
            self.assertEqual(run.summary_rows[1]["status"], "failed")
            self.assertTrue((output_dir / "run_failure_case_project_summary.csv").exists())
            self.assertTrue((output_dir / "run_failure_case_project_summary.json").exists())
            self.assertTrue((output_dir / "run_failure_case_project_summary.md").exists())
            self.assertEqual(set(project.files), set(run.files))

    def test_project_describe_includes_resolved_channel_setup(self) -> None:
        project = CoilProject("describe_case", template=CylinderCoilDesign(target_scale=0.01))
        project.channel("Gz", target_bz_expression="z", label="Z gradient")

        text = project.describe()

        self.assertIn("CoilProject: describe_case", text)
        self.assertIn("Gz", text)
        self.assertIn("target=z", text)
        self.assertIn("scale=0.01", text)

    def test_project_gradient_channel_maps_physical_hardware_contract(self) -> None:
        project = CoilProject("gradient_contract", template=CylinderCoilDesign())
        project.gradient_channel(
            "Gz",
            axis="Gz",
            gradient_strength_T_per_m=0.02,
            gradient_strength_tolerance_percent=1.0,
            linearity=GradientLinearitySpec(
                region="dsv",
                reference_radius_m=0.08,
                rms_tolerance_percent=1.0,
                peak_tolerance_percent=5.0,
                cross_gradient_tolerance_percent=1.0,
            ),
            slice_maps=GradientSliceMapSpec(enabled=True, slice_count=5, grid_size=32),
            max_wire_current_A=8.0,
            min_wire_distance_m=1.5e-3,
            contour_levels=12,
        )

        design = project.resolved_designs()["Gz"]
        config = design.to_config()

        self.assertEqual(design.design_context, "mr_gradient")
        self.assertEqual(design.target_bz_expression, "z")
        self.assertEqual(design.target_gradient_axis, "z")
        self.assertEqual(design.target_gradient_normalization, "linear_fit")
        self.assertEqual(design.contour_discretization_method, "engineering")
        self.assertAlmostEqual(config.target_gradient_strength, 0.02)
        self.assertAlmostEqual(config.max_wire_current_A, 8.0)
        self.assertAlmostEqual(config.min_wire_distance_m, 1.5e-3)
        self.assertEqual(config.engineering_contour_max_levels, 12)
        self.assertAlmostEqual(config.gradient_strength_tolerance_percent, 1.0)
        self.assertAlmostEqual(config.linearity_reference_radius_m, 0.08)
        self.assertAlmostEqual(config.linearity_rms_tolerance_percent, 1.0)
        self.assertAlmostEqual(config.linearity_peak_tolerance_percent, 5.0)
        self.assertAlmostEqual(config.linearity_cross_gradient_tolerance_percent, 1.0)
        self.assertTrue(config.gradient_slice_maps_enabled)
        self.assertEqual(config.gradient_slice_map_slice_count, 5)
        self.assertEqual(config.gradient_slice_map_grid_size, (32, 32))

    def test_design_gradient_tolerance_field_wins_over_linearity_default(self) -> None:
        design = CylinderCoilDesign(
            gradient_strength_tolerance_percent=1.0,
            linearity=GradientLinearitySpec(reference_radius_m=0.08, rms_tolerance_percent=1.0),
        )

        config = design.to_config()

        self.assertAlmostEqual(config.gradient_strength_tolerance_percent, 1.0)
        self.assertAlmostEqual(config.linearity_reference_radius_m, 0.08)
        self.assertAlmostEqual(config.linearity_rms_tolerance_percent, 1.0)

    def test_config_design_adapter_supports_project_channel_overrides(self) -> None:
        config = CoilGenConfig(field_shape_function="y", target_gradient_strength=1.0, levels=8)
        template = CoilGenConfigDesign(name="config_base", config=config)
        project = CoilProject("config_project", template=template, design_overrides={"target_scale": 2.0})
        project.channel("Gx", target_bz_expression="x")
        project.channel("Gz", target_bz_expression="z", target_scale=3.0)

        designs = project.resolved_designs()

        self.assertEqual(designs["Gx"].to_config().field_shape_function, "x")
        self.assertEqual(designs["Gx"].to_config().target_gradient_strength, 2.0)
        self.assertEqual(designs["Gz"].to_config().field_shape_function, "z")
        self.assertEqual(designs["Gz"].to_config().target_gradient_strength, 3.0)
        self.assertEqual(config.field_shape_function, "y")
        self.assertEqual(config.target_gradient_strength, 1.0)

    def test_config_design_adapter_exports_user_parameters(self) -> None:
        config = CoilGenConfig(
            field_shape_function="none",
            target_gradient_strength=1.0,
            design_context="mr_shim",
            target_field_units="Hz",
            target_error_constraint_metric="std",
            target_error_std_limit_T=1.5e-6,
        )
        design = CoilGenConfigDesign(name="explicit_shim_config", config=config)

        values = design.as_user_parameters()

        self.assertEqual(values["name"], "explicit_shim_config")
        self.assertEqual(values["design_context"], "mr_shim")
        self.assertEqual(values["target_field_units"], "Hz")
        self.assertEqual(values["target_error_constraint_metric"], "std")
        self.assertAlmostEqual(values["target_error_std_limit_T"], 1.5e-6)

    def test_project_summary_plot_columns_normalize_string_context(self) -> None:
        project = CoilProject(
            "plot_context",
            template=CylinderCoilDesign(),
            design_context="mr_gradient",
        )
        project.channel("Gx", target_bz_expression="x")
        run = CoilProjectRun(
            project=project,
            output_dir=Path("."),
            channel_runs=(),
            summary_rows=(
                {
                    "channel": "Gx",
                    "design_context": "mr_gradient",
                    "layout_gradient_efficiency_mT_per_m_per_A": 1.0,
                    "coil_length_total_m": 12.5,
                },
            ),
        )

        columns = _project_channel_summary_columns(run)

        self.assertIn("stream_function_gradient_efficiency_mT_per_m_per_A", columns)
        self.assertIn("unconnected_contours_gradient_efficiency_mT_per_m_per_A", columns)
        self.assertIn("unconnected_contours_min_wire_distance_m", columns)
        self.assertIn("layout_gradient_efficiency_mT_per_m_per_A", columns)

    def test_project_summary_plot_sections_cover_existing_columns(self) -> None:
        project = CoilProject(
            "plot_context",
            template=CylinderCoilDesign(),
            design_context="mr_gradient",
        )
        project.channel("Gx", target_bz_expression="x")
        run = CoilProjectRun(
            project=project,
            output_dir=Path("."),
            channel_runs=(),
            summary_rows=(
                {
                    "channel": "Gx",
                    "design_context": "mr_gradient",
                    "layout_gradient_efficiency_mT_per_m_per_A": 1.0,
                },
            ),
        )

        columns = _project_channel_summary_columns(run)
        sections = _project_channel_summary_sections(run)
        section_titles = [title for title, _section_columns in sections]
        unique_section_columns = []
        for _title, section_columns in sections:
            self.assertEqual(section_columns[0], "channel")
            for column in section_columns:
                if column not in unique_section_columns:
                    unique_section_columns.append(column)

        self.assertEqual(
            section_titles,
            [
                "Channel / Target",
                "Stream Function",
                "Unconnected Contours",
                "Connected Layout",
                "Electrical / Hardware",
            ],
        )
        self.assertEqual(tuple(unique_section_columns), columns)

    def test_project_summary_table_appends_user_thresholds_to_metric_values(self) -> None:
        rows = (
            {
                "channel": "Gx",
                "layout_needed_current_A": 200.0,
                "max_wire_current_A": 300.0,
                "layout_mean_rel_error_percent": 1.25,
                "target_error_tolerance_percent": 5.0,
                "coil_length_total_m": 12.5,
            },
        )
        columns = (
            "channel",
            "layout_needed_current_A",
            "layout_mean_rel_error_percent",
            "coil_length_total_m",
        )

        cell_text = _project_summary_section_cell_text(columns, rows)

        self.assertEqual(cell_text[0][1], "200 (threshold: <= 300)")
        self.assertEqual(cell_text[0][2], "1.25 (threshold: <= 5)")
        self.assertEqual(cell_text[0][3], "12.5 (threshold: unspecified)")

    def test_project_summary_table_includes_voltage_at_target_slew_when_available(self) -> None:
        project = CoilProject(
            "plot_context",
            template=CylinderCoilDesign(),
            design_context="mr_gradient",
        )
        row = {
            "channel": "Gx",
            "design_context": "mr_gradient",
            "drive_voltage_target_slew_V": 1020.0,
            "target_slew_rate_T_per_m_per_s": 200.0,
        }
        run = CoilProjectRun(project=project, output_dir=Path("."), channel_runs=(), summary_rows=(row,))

        columns = _project_channel_summary_columns(run)
        sections = dict(_project_channel_summary_sections(run))
        cell_text = _project_summary_section_cell_text(("channel", "drive_voltage_target_slew_V"), (row,))

        self.assertIn("drive_voltage_target_slew_V", columns)
        self.assertIn("drive_voltage_target_slew_V", sections["Electrical / Hardware"])
        self.assertEqual(cell_text[0][1], "1020 (target slew: 200 T/m/s)")

    def test_project_summary_drive_metrics_use_target_slew_rate(self) -> None:
        metrics = _project_drive_metrics_from_summary(
            {
                "target_gradient_strength_T_per_m": 0.02,
                "target_slew_rate_T_per_m_per_s": 200.0,
                "layout_needed_current_A": 10.0,
                "coil_resistance_total_ohm": 2.0,
                "coil_inductance_same_current_H": 0.01,
            }
        )

        self.assertAlmostEqual(metrics["drive_ramp_time_s"], 1.0e-4)
        self.assertAlmostEqual(metrics["drive_di_dt_A_per_s"], 1.0e5)
        self.assertAlmostEqual(metrics["drive_voltage_target_slew_V"], 1020.0)

    def test_channel_prefixed_plot_spec_exposes_channel_metadata(self) -> None:
        from coilgen.matplotlib_visualization import MatplotlibPlotSpec

        channel_run = SimpleNamespace(channel=SimpleNamespace(display_label="Gx active", safe_name="gx_active"))
        source = MatplotlibPlotSpec(
            key="layout_field_overview",
            title="Layout and Field",
            category="field",
            factory=lambda: None,
        )

        spec = _channel_prefixed_plot_spec(MatplotlibPlotSpec, channel_run, source)

        self.assertEqual(spec.channel, "Gx active")
        self.assertEqual(spec.category, "field")
        self.assertEqual(spec.title, "Gx active: Layout and Field")

    def test_standard_cylindrical_3axis_builder_uses_project_channels(self) -> None:
        parameters = {
            "run": {"axes": ("x", "y", "z"), "progress": False},
            "system": {"main_field_strength_T": 1.5, "b0_direction": (0.0, 0.0, 1.0)},
            "geometry": {
                "radius_m": 0.26,
                "length_m": 0.55,
                "angular_divisions": 12,
                "length_divisions": 8,
                "rotation_axis": (1.0, 0.0, 0.0),
                "rotation_angle_rad": 0.0,
            },
            "target": {
                "gradient_strength_T_per_m": 0.010,
                "dsv_radius_m": 0.08,
                "dsv_resolution": 5,
                "field_evaluation_current_source": "target_fit",
            },
            "optimization": {
                "method": "tikhonov",
                "regularization_method": "l_curve",
                "regularization_factor": 50.0,
                "l_curve_candidate_count": 11,
                "l_curve_smoothing_window": 5,
                "target_error_tolerance_percent": 5.0,
                "target_error_tolerance_absolute_T": 0.0,
                "target_error_constraint_candidate_count": 31,
            },
            "layout": {
                "contour_levels": 6,
                "potential_offset_factor": 0.25,
                "min_contour_points": 4,
                "min_turn_spacing_m": 0.0,
                "skip_postprocessing": True,
                "skip_sweep": True,
                "skip_inductance": True,
                "inductance_source": "auto",
                "contour_symmetry_validation": "auto",
                "zero_crossing_level_strategy": "paired",
            },
            "conductor": {"width_m": 0.001, "height_m": 0.001, "resistivity_ohm_m": 0.018e-6},
            "numerics": {
                "cpu_workers": 1,
                "max_chunk_memory_mb": 64.0,
                "parallel_min_work_size": 50000,
                "sensitivity_chunks_per_worker": 4,
                "progress_min_interval_seconds": 1.0,
                "progress_percent_step": 5.0,
            },
            "output": {"name": "standard_cylindrical_3axis_gradient", "save_reports": False, "save_svg_gallery": False},
            "plots": {
                "show_plot_book": False,
                "plot_book_scope": "system",
                "plot_book_axes": ("x", "y", "z"),
                "close_previous": False,
                "mode": "book",
                "max_parts": None,
                "include": ("wire", "performance_table"),
            },
        }

        project = build_gradient_project(parameters)
        designs = project.resolved_designs()

        self.assertIsInstance(project, CoilProject)
        self.assertEqual([channel.name for channel in project.channels], ["GX", "GY", "GZ"])
        self.assertEqual([designs[name].target_bz_expression for name in ("GX", "GY", "GZ")], ["x", "y", "z"])
        self.assertEqual(
            [designs[name].name for name in ("GX", "GY", "GZ")],
            [
                "standard_cylindrical_3axis_gradient_gx",
                "standard_cylindrical_3axis_gradient_gy",
                "standard_cylindrical_3axis_gradient_gz",
            ],
        )

    def test_standard_cylindrical_3axis_script_is_object_based(self) -> None:
        project = standard_gradient_script.PROJECT
        designs = project.resolved_designs()
        source = Path(standard_gradient_script.__file__).read_text(encoding="utf-8")

        self.assertNotIn("object_parameters", source)
        self.assertNotIn("run_gradient_system", source)
        self.assertNotIn("build_project", source)
        self.assertNotIn("project.parameters", source)
        self.assertIsInstance(project, CoilProject)
        self.assertIsInstance(standard_gradient_script.GRADIENT_TEMPLATE, CylinderCoilDesign)
        channel_names = [channel.name for channel in project.channels]
        self.assertTrue(channel_names)
        self.assertTrue(set(channel_names).issubset({"GX", "GY", "GZ"}))
        for name in channel_names:
            self.assertEqual(designs[name].target_bz_expression, name[1].lower())
            self.assertEqual(designs[name].design_context, "mr_gradient")


if __name__ == "__main__":
    unittest.main()
