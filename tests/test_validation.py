from __future__ import annotations

import json
import runpy
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from coilgen import (
    CoilGenConfig,
    ConcomitantFieldReport,
    CylinderCoilDesign,
    ParameterAxis,
    apply_ordered_solenoid_wire_path,
    coil_geometry_api_template,
    result_quality_status,
    run_coilgen,
    validate_active_shield_suppression,
    validate_biplanar_gradient_scaling,
    validate_circular_loop_biot_savart,
    validate_solenoid_against_analytical_formulas,
    validate_config,
    write_design_report,
    write_coil_geometry_api_template,
)
from coilgen.workflow import (
    RUN_RECORD_MAX_SUMMARY_BYTES,
    RUN_RECORD_RETENTION_COUNT,
    CoilCaseSpec,
    CoilCaseWorkflow,
    CoilVariantSpec,
    ParameterSweepSpec,
    ParameterSweepWorkflow,
    write_variant_failure_record,
)
from projects._shared.object_parameters import ParameterNamespace


class ValidationExamplesTest(unittest.TestCase):
    def test_config_grouped_settings_are_available(self) -> None:
        config = CoilGenConfig(field_shape_function="1", levels=6, parallel_worker_count=1)

        self.assertEqual(config.target_settings.field_shape_function, "1")
        self.assertEqual(config.target_settings.target_field_units, "T")
        self.assertEqual(config.target_settings.field_display_units, "T")
        self.assertEqual(config.target_settings.target_gradient_normalization, "auto")
        self.assertEqual(config.target_settings.target_gradient_axis, "auto")
        self.assertEqual(config.optimization_settings.sf_opt_method, "tikhonov")
        self.assertEqual(config.optimization_settings.gradient_hardware_infeasible_policy, "scale_to_limit")
        self.assertEqual(config.performance_settings.parallel_worker_count, 1)
        self.assertEqual(config.routing_settings.topological_grouping_method, "auto")
        self.assertEqual(config.routing_settings.inter_group_connection_path_method, "direct_3d")
        self.assertEqual(config.levels, 6)

    def test_config_validation_accepts_hz_field_units(self) -> None:
        report = validate_config(
            CoilGenConfig(
                coil_mesh_file="create cylinder mesh",
                target_field_units="Hz",
                field_display_units="Hz",
            )
        )

        self.assertTrue(report.ok, report.format())

    def test_config_validation_accepts_strict_constrained_quadratic_fallback_policy(self) -> None:
        report = validate_config(
            CoilGenConfig(
                coil_mesh_file="create cylinder mesh",
                constrained_quadratic_fallback_policy="strict",
            )
        )

        self.assertTrue(report.ok, report.format())

    def test_config_validation_rejects_unknown_constrained_quadratic_fallback_policy(self) -> None:
        report = validate_config(CoilGenConfig(constrained_quadratic_fallback_policy="silent_relaxation"))

        self.assertFalse(report.ok)
        self.assertIn("constrained_quadratic_fallback_policy", report.format())

    def test_config_validation_checks_mesh_coupled_contour_smoothing_controls(self) -> None:
        valid = validate_config(
            CoilGenConfig(
                coil_mesh_file="create cylinder mesh",
                contour_smoothing_scale_mode="mesh_coupled",
                contour_smoothing_mesh_gain=0.1,
                contour_smoothing_work_samples_per_mesh_length=6,
                contour_smoothing_output_samples_per_mesh_length=8,
                contour_smoothing_max_displacement=0.0,
            )
        )
        invalid = validate_config(
            CoilGenConfig(
                coil_mesh_file="create cylinder mesh",
                contour_smoothing_scale_mode="unknown",
                contour_smoothing_mesh_gain=0.0,
                contour_smoothing_work_samples_per_mesh_length=3,
                contour_smoothing_output_samples_per_mesh_length=1,
            )
        )

        self.assertTrue(valid.ok, valid.format())
        self.assertFalse(invalid.ok)
        self.assertIn("contour_smoothing_scale_mode", invalid.format())
        self.assertIn("contour_smoothing_mesh_gain", invalid.format())
        self.assertIn("contour_smoothing_work_samples_per_mesh_length", invalid.format())
        self.assertIn("contour_smoothing_output_samples_per_mesh_length", invalid.format())

    def test_config_validation_rejects_unknown_field_units(self) -> None:
        report = validate_config(CoilGenConfig(target_field_units="ppm"))

        self.assertFalse(report.ok)
        self.assertIn("target_field_units", report.format())

    def test_config_validation_rejects_unknown_target_component(self) -> None:
        report = validate_config(CoilGenConfig(target_field_component="Bq"))

        self.assertFalse(report.ok)
        self.assertIn("target_field_component", report.format())

    def test_config_validation_rejects_unknown_design_context(self) -> None:
        report = validate_config(CoilGenConfig(coil_mesh_file="create cylinder mesh", design_context="not_a_context"))

        self.assertFalse(report.ok)
        self.assertIn("design_context", report.format())

    def test_config_validation_warns_for_mr_rf_context(self) -> None:
        report = validate_config(CoilGenConfig(coil_mesh_file="create cylinder mesh", design_context="mr_rf"))

        self.assertTrue(report.ok, report.format(levels=("error",)))
        self.assertIn("magnetostatic", report.format(levels=("warning",)))

    def test_config_validation_warns_for_generalized_target_component(self) -> None:
        report = validate_config(CoilGenConfig(coil_mesh_file="create cylinder mesh", target_field_component="BdotB0"))

        self.assertTrue(report.ok, report.format(levels=("error",)))
        self.assertIn("target_field_component", report.format(levels=("warning",)))

    def test_config_validation_warns_for_missing_layout_terminal_strategy(self) -> None:
        report = validate_config(
            CoilGenConfig(
                coil_mesh_file="create cylinder mesh",
                skip_postprocessing=False,
                skip_sweep=False,
            )
        )

        self.assertTrue(report.ok, report.format(levels=("error",)))
        self.assertIn("layout_terminal_strategy", report.format(levels=("warning",)))

    def test_config_validation_accepts_explicit_relaxed_surface_geodesic_routing(self) -> None:
        report = validate_config(
            CoilGenConfig(
                coil_mesh_file="create cylinder mesh",
                inter_group_connection_path_method="surface_geodesic_or_direct",
            )
        )

        self.assertTrue(report.ok, report.format(levels=("error",)))

    def test_config_validation_rejects_unknown_topological_grouping_method(self) -> None:
        report = validate_config(CoilGenConfig(topological_grouping_method="nearest_opening_magic"))

        self.assertFalse(report.ok)
        self.assertIn("topological_grouping_method", report.format())

    def test_config_validation_rejects_unknown_gradient_hardware_settings(self) -> None:
        report = validate_config(
            CoilGenConfig(
                target_gradient_normalization="magic_fit",
                target_gradient_axis="theta",
                gradient_hardware_infeasible_policy="silently_ignore",
            )
        )

        self.assertFalse(report.ok)
        text = report.format()
        self.assertIn("target_gradient_normalization", text)
        self.assertIn("target_gradient_axis", text)
        self.assertIn("gradient_hardware_infeasible_policy", text)

    def test_constrained_quadratic_optimizer_config_is_valid(self) -> None:
        config = CoilGenConfig(
            coil_mesh_file="create cylinder mesh",
            sf_opt_method="constrained_quadratic",
            constrained_optimization_solver="auto",
            constrained_optimization_tolerance=1e-7,
            constrained_optimization_max_iterations=20000,
        )
        report = validate_config(config)

        self.assertTrue(report.ok, report.format())
        self.assertEqual(config.optimization_settings.constrained_optimization_solver, "auto")

    def test_concomitant_field_report_formula(self) -> None:
        coords = [[0.0, 0.0, -0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.1]]
        gradient = [2.0, 0.0, 0.0]

        values = ConcomitantFieldReport.field_values(coords, gradient, main_field_T=1.0)
        report = ConcomitantFieldReport.from_linear_gradient(coords, gradient, main_field_T=1.0)

        self.assertAlmostEqual(float(values[0]), 0.02)
        self.assertAlmostEqual(float(values[1]), 0.0)
        self.assertAlmostEqual(float(values[2]), 0.02)
        self.assertAlmostEqual(report.max_abs_concomitant_T, 0.02)
        self.assertAlmostEqual(report.rms_concomitant_T, ((0.02**2 + 0.02**2) / 3.0) ** 0.5)
        self.assertAlmostEqual(report.max_abs_concomitant_ppm_of_B0, 20000.0)

    def test_scientist_cylinder_design_builds_valid_config(self) -> None:
        design = CylinderCoilDesign(
            target_bz_expression="y",
            target_scale=1.0,
            radius_m=0.2,
            length_m=0.4,
            angular_divisions=16,
            length_divisions=8,
            contour_levels=6,
            progress=False,
            skip_sweep=True,
            skip_inductance=True,
        )
        config = design.to_config()
        report = validate_config(config)

        self.assertTrue(report.ok, report.format())
        self.assertEqual(config.coil_mesh_file, "create cylinder mesh")
        self.assertEqual(config.cylinder_mesh_parameter_list[:4], (0.45, 0.2, 16, 8))
        self.assertEqual(config.field_shape_function, "y")
        self.assertEqual(config.levels, 6)

    def test_generic_case_workflow_runs_design_and_writes_tables(self) -> None:
        design = CylinderCoilDesign(
            name="workflow_tiny_cylinder",
            target_bz_expression="y",
            radius_m=0.08,
            length_m=0.14,
            angular_divisions=8,
            length_divisions=4,
            target_region_resolution=3,
            contour_levels=3,
            progress=False,
            skip_postprocessing=True,
            skip_sweep=True,
            skip_inductance=True,
        )
        with TemporaryDirectory() as tmp:
            case = CoilCaseSpec(
                name="workflow_case",
                variants=(CoilVariantSpec("main", design, save_reports=False),),
                output_dir=Path(tmp),
                parameters={"radius_m": design.radius_m, "length_m": design.length_m},
            )
            result = CoilCaseWorkflow().run(case)

            self.assertTrue(result.ok)
            self.assertIn("main_layout_gradient_efficiency_mT_per_m_per_A", result.metrics)
            self.assertTrue((Path(tmp) / "workflow_case_case_result.csv").exists())
            self.assertTrue((Path(tmp) / "workflow_case_case_result.json").exists())

    def test_generic_case_workflow_writes_compact_run_logs(self) -> None:
        design = CylinderCoilDesign(
            name="workflow_tiny_cylinder_logs",
            target_bz_expression="y",
            radius_m=0.08,
            length_m=0.14,
            angular_divisions=8,
            length_divisions=4,
            target_region_resolution=3,
            contour_levels=3,
            progress=False,
            skip_postprocessing=True,
            skip_sweep=True,
            skip_inductance=True,
        )
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            case = CoilCaseSpec(
                name="workflow_case_logs",
                variants=(CoilVariantSpec("main", design, save_reports=True, metadata={"purpose": "test"}),),
                output_dir=output_dir,
                parameters={"project": "validation", "radius_m": design.radius_m, "length_m": design.length_m},
            )
            result = CoilCaseWorkflow().run(case)

            self.assertTrue(result.ok)
            summary_path = output_dir / "main_run_summary.json"
            metrics_path = output_dir / "main_run_metrics.csv"
            events_path = output_dir / "main_run_events.jsonl"
            log_path = output_dir / "main_run_log.md"
            for path in (summary_path, metrics_path, events_path, log_path):
                self.assertTrue(path.exists(), path)
                self.assertIn(path, result.files)

            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "coilgen_run_record_v1")
            self.assertEqual(payload["run"]["case"], "workflow_case_logs")
            self.assertEqual(payload["run"]["variant"], "main")
            self.assertEqual(payload["run"]["status"], "success")
            self.assertEqual(payload["variant_metadata"]["purpose"], "test")
            self.assertIn("result_summary_sha1", payload["digests"])
            self.assertIn("layout_gradient_efficiency_mT_per_m_per_A", payload["metrics"])
            self.assertIn("variant_finished", events_path.read_text(encoding="utf-8"))
            self.assertIn("quality_status", metrics_path.read_text(encoding="utf-8"))
            self.assertIn("Key Metrics", log_path.read_text(encoding="utf-8"))

    def test_generic_case_workflow_writes_failure_run_logs(self) -> None:
        design = CylinderCoilDesign(
            name="workflow_invalid_cylinder_logs",
            target_bz_expression="y",
            radius_m=-0.08,
            length_m=0.14,
            angular_divisions=8,
            length_divisions=4,
            target_region_resolution=3,
            contour_levels=3,
            progress=False,
            skip_postprocessing=True,
            skip_sweep=True,
            skip_inductance=True,
        )
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            case = CoilCaseSpec(
                name="workflow_case_failed_logs",
                variants=(CoilVariantSpec("failed", design, save_reports=True),),
                output_dir=output_dir,
                continue_on_variant_failure=True,
            )
            result = CoilCaseWorkflow().run(case)

            self.assertFalse(result.ok)
            summary_path = output_dir / "failed_run_summary.json"
            events_path = output_dir / "failed_run_events.jsonl"
            self.assertTrue(summary_path.exists())
            self.assertTrue(events_path.exists())
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run"]["status"], "failed")
            self.assertEqual(payload["quality"]["status"], "FAIL")
            self.assertIn("exception", payload)
            self.assertIn("exception", events_path.read_text(encoding="utf-8"))

    def test_generic_run_logs_keep_only_recent_variant_groups(self) -> None:
        design = CylinderCoilDesign(
            name="workflow_retention_design",
            target_bz_expression="y",
            radius_m=0.08,
            length_m=0.14,
            angular_divisions=8,
            length_divisions=4,
            target_region_resolution=3,
            contour_levels=3,
            progress=False,
            skip_postprocessing=True,
            skip_sweep=True,
            skip_inductance=True,
        )
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            case = CoilCaseSpec(name="workflow_retention_case", variants=(), output_dir=output_dir)
            for index in range(RUN_RECORD_RETENTION_COUNT + 5):
                write_variant_failure_record(
                    ValueError(f"failure {index}"),
                    CoilVariantSpec(f"variant_{index:02d}", design, save_reports=True),
                    case,
                    output_dir,
                    started_at_utc=f"2026-01-01T00:{index:02d}:00Z",
                    completed_at_utc=f"2026-01-01T00:{index:02d}:01Z",
                )

            summaries = sorted(output_dir.glob("*_run_summary.json"))
            events = sorted(output_dir.glob("*_run_events.jsonl"))
            logs = sorted(output_dir.glob("*_run_log.md"))

            self.assertEqual(len(summaries), RUN_RECORD_RETENTION_COUNT)
            self.assertEqual(len(events), RUN_RECORD_RETENTION_COUNT)
            self.assertEqual(len(logs), RUN_RECORD_RETENTION_COUNT)
            self.assertFalse((output_dir / "variant_00_run_summary.json").exists())
            self.assertTrue((output_dir / f"variant_{RUN_RECORD_RETENTION_COUNT + 4:02d}_run_summary.json").exists())

    def test_generic_failure_run_logs_bound_large_payloads(self) -> None:
        design = CylinderCoilDesign(
            name="workflow_large_failure_logs",
            target_bz_expression="y",
            radius_m=0.08,
            length_m=0.14,
            angular_divisions=8,
            length_divisions=4,
            target_region_resolution=3,
            contour_levels=3,
            progress=False,
            skip_postprocessing=True,
            skip_sweep=True,
            skip_inductance=True,
        )
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            case = CoilCaseSpec(name="workflow_large_failure_case", variants=(), output_dir=output_dir)
            huge_message = "oversized diagnostic " + ("x" * (RUN_RECORD_MAX_SUMMARY_BYTES * 2))

            write_variant_failure_record(
                RuntimeError(huge_message),
                CoilVariantSpec("large_failure", design, save_reports=True),
                case,
                output_dir,
            )

            summary_path = output_dir / "large_failure_run_summary.json"
            events_path = output_dir / "large_failure_run_events.jsonl"
            log_path = output_dir / "large_failure_run_log.md"
            payload = json.loads(summary_path.read_text(encoding="utf-8"))

            self.assertLess(summary_path.stat().st_size, RUN_RECORD_MAX_SUMMARY_BYTES)
            self.assertLess(log_path.stat().st_size, 300_000)
            self.assertIn("truncated", payload["exception"]["message"])
            self.assertIn("truncated", events_path.read_text(encoding="utf-8"))
            self.assertIn("truncated", log_path.read_text(encoding="utf-8"))

    def test_generic_parameter_sweep_continues_after_failed_case(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            def case_factory(point: dict) -> CoilCaseSpec:
                radius = float(point["radius_m"])
                if radius <= 0.0:
                    raise ValueError("radius_m must be positive")
                design = CylinderCoilDesign(
                    name=f"workflow_sweep_radius_{radius:.3f}",
                    target_bz_expression="y",
                    radius_m=radius,
                    length_m=0.14,
                    angular_divisions=8,
                    length_divisions=4,
                    target_region_resolution=3,
                    contour_levels=3,
                    progress=False,
                    skip_postprocessing=True,
                    skip_sweep=True,
                    skip_inductance=True,
                )
                return CoilCaseSpec(
                    name=f"case_radius_{radius:.3f}",
                    variants=(CoilVariantSpec("main", design, save_reports=False),),
                    output_dir=tmp_path / f"case_radius_{radius:.3f}",
                    parameters=point,
                )

            sweep = ParameterSweepSpec(
                name="workflow_sweep",
                axes=(ParameterAxis.from_values("radius_m", (0.08, -0.1), unit="m"),),
                case_factory=case_factory,
                output_dir=tmp_path / "sweep",
                continue_on_failure=True,
                retain_case_results=True,
            )
            result = ParameterSweepWorkflow().run(sweep)

            self.assertEqual(len(result.case_results), 1)
            self.assertEqual(len(result.failures), 1)
            self.assertTrue((tmp_path / "sweep" / "workflow_sweep.csv").exists())
            self.assertTrue((tmp_path / "sweep" / "workflow_sweep.json").exists())

    def test_config_validation_reports_actionable_errors(self) -> None:
        config = CoilGenConfig(
            coil_mesh_file="create cylinder mesh",
            cylinder_mesh_parameter_list=(0.5, -0.25, 3, 1, 1, 0, 0, 0),
        )
        report = validate_config(config)

        self.assertFalse(report.ok)
        self.assertIn("cylinder radius", report.format())

    def test_coil_geometry_template_can_be_generated(self) -> None:
        text = coil_geometry_api_template("custom saddle surface", class_name="CustomSaddleSurfaceCoilDesign")
        self.assertIn("class CustomSaddleSurfaceCoilDesign", text)
        self.assertIn("geometry_config", text)
        with TemporaryDirectory() as tmp:
            path = write_coil_geometry_api_template("custom saddle surface", Path(tmp) / "template.py")
            self.assertTrue(path.exists())
            self.assertIn("CoilDesign", path.read_text(encoding="utf-8"))

    def test_general_example_scripts_expose_object_parameters(self) -> None:
        scripts_dir = Path(__file__).resolve().parents[1] / "projects" / "general_examples" / "scripts"
        script_names = tuple(
            path.name
            for path in sorted(scripts_dir.glob("*.py"))
            if not path.name.startswith("_") and path.name != "__init__.py"
        )

        self.assertGreater(len(script_names), 0)
        for script_name in script_names:
            sys.path.insert(0, str(scripts_dir))
            try:
                namespace = runpy.run_path(str(scripts_dir / script_name))
            finally:
                try:
                    sys.path.remove(str(scripts_dir))
                except ValueError:
                    pass
            if "PARAMETERS" not in namespace:
                continue
            self.assertIsInstance(namespace["PARAMETERS"], ParameterNamespace, script_name)
            self.assertIsInstance(namespace["PARAMETERS"].to_dict(), dict, script_name)
            self.assertTrue(hasattr(namespace["PARAMETERS"], "help"), script_name)

    def test_primitive_support_project_configs_are_valid(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1]
            / "projects"
            / "primitive_support_structures"
            / "scripts"
            / "run_primitive_support_designs.py"
        )
        namespace = runpy.run_path(str(script_path))
        self.assertIn("PARAMETERS", namespace)
        self.assertIn("DEFAULT_PARAMETERS", namespace)
        self.assertIsInstance(namespace["PARAMETERS"], ParameterNamespace)
        self.assertIn("PARAMETERS", namespace["PARAMETERS"].help())
        selected_cases = (
            "circular_disc",
            "annular_disc",
            "shielded_annular_disc",
            "partial_cylinder",
            "elliptical_cylinder",
            "superellipse_cylinder",
            "spherical_cap",
            "ellipsoidal_cap",
            "conical_frustum",
            "toroidal_segment",
            "ruled_surface",
            "surface_of_revolution",
            "parametric_surface",
        )
        configs = namespace["build_configs"]({"run": {"selected_cases": selected_cases}})

        self.assertEqual(set(configs), set(selected_cases))
        for config in configs.values():
            report = validate_config(config)
            self.assertTrue(report.ok, report.format())

    def test_analytical_benchmark_helpers_are_sane(self) -> None:
        loop = validate_circular_loop_biot_savart(radius_m=0.10, z_offset_m=0.03, segment_count=720)
        self.assertLess(loop.relative_error_percent, 0.1)

        biplanar = validate_biplanar_gradient_scaling(
            radius_m=0.30,
            gap_values_m=(0.15, 0.24, 0.30, 0.36, 0.45),
        )
        self.assertAlmostEqual(biplanar.maximum_gap_m, 0.30, places=12)
        self.assertLess(biplanar.maximum_gap_error_percent, 1e-9)

        shield = validate_active_shield_suppression(
            shielded_field=[[0.0, 0.0, 0.2], [0.0, 0.0, -0.4]],
            unshielded_field=[[0.0, 0.0, 1.0], [0.0, 0.0, -2.0]],
            component="Bz",
        )
        self.assertTrue(shield.rms_reduced)
        self.assertGreater(shield.rms_reduction_percent, 70.0)

    def test_shielding_benchmark_rejects_unmatched_samples_instead_of_claiming_reduction(self) -> None:
        for shielded, unshielded in (([0.0], [1.0, 2.0]), ([0.0, 0.0], [1.0]), ([], [1.0])):
            with self.subTest(shielded=shielded, unshielded=unshielded):
                with self.assertRaisesRegex(ValueError, "same sample count"):
                    validate_active_shield_suppression(shielded, unshielded)

        empty = validate_active_shield_suppression([], [])
        self.assertEqual(empty.sample_count, 0)
        self.assertFalse(empty.rms_reduced)
        self.assertFalse(empty.peak_reduced)

    def test_constant_bz_solenoid_matches_analytical_center_field_and_inductance(self) -> None:
        config = CoilGenConfig(
            field_shape_function="1",
            target_gradient_strength=1.0e-4,
            coil_mesh_file="create cylinder mesh",
            cylinder_mesh_parameter_list=(0.8, 0.08, 32, 32, 1, 0, 0, 0),
            target_region_radius=0.03,
            target_region_resolution=5,
            levels=10,
            pot_offset_factor=0.2,
            min_point_loop_number=8,
            sf_opt_method="tikhonov",
            stream_function_cost_function="field_error_plus_resistance",
            tikonov_reg_factor=20.0,
            regularization_selection_method="fixed",
            skip_postprocessing=True,
            skip_sweep=True,
            skip_inductance_calculation=False,
            progress_flag=False,
            conductor_cross_section_width=0.001,
            conductor_cross_section_height=0.001,
            plot_flag=False,
            save_stl_flag=False,
        )

        result = run_coilgen(config)
        result = apply_ordered_solenoid_wire_path(result, config)
        validation = validate_solenoid_against_analytical_formulas(result, config)

        self.assertGreaterEqual(validation.turn_count, 8)
        self.assertLess(validation.center_vs_circular_loop_stack_error_percent, 1.0)
        self.assertLess(validation.center_vs_requested_error_percent, 1.0)
        self.assertLess(validation.inductance_vs_circular_loop_stack_error_percent, 15.0)
        self.assertIn(result_quality_status(result).status, {"PASS", "WARNING", "FAIL"})
        with TemporaryDirectory() as tmp:
            report_path = write_design_report(result, Path(tmp) / "design_report.md")
            self.assertTrue(report_path.exists())
            self.assertIn("CoilGen Design Report", report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
