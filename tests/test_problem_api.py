from __future__ import annotations

import unittest

import numpy as np

from coilgen import (
    CoilGenConfig,
    CoilDesignProblem,
    CoilSupportSpec,
    ConductorSpec,
    DesignMetadata,
    FieldComponentSpec,
    LayoutSpec,
    Mesh,
    ParameterAxis,
    ParameterStudySpec,
    PipelineGeometrySpec,
    PipelineCoilDesignSolver,
    PROBLEM_SCHEMA_VERSION,
    SupportMesh,
    SuppressionRegion,
    TargetRegion,
    TargetFieldSpec,
    CoilPart,
    TargetField,
    component_sample_values,
    field_component_values,
    load_target_field_definition,
    recommended_settings,
    sensitivity_component_matrix,
    validate_config,
)
from coilgen.geometry import load_mesh_from_config
from coilgen.optimization import StreamFunctionOptimizationProblem


class ProblemApiTest(unittest.TestCase):
    def test_target_field_spec_preserves_dsv_and_suppression_labels(self) -> None:
        dsv_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.02, 0.0]])
        dsv_bz = np.array([0.0, 2e-5])
        suppression = SuppressionRegion(
            "magnet_interface",
            coords=np.array([[0.2, 0.0, 0.0], [0.2, 0.1, 0.0]]),
            weight=4.0,
            group_index=3,
        )

        spec = TargetFieldSpec.from_arrays(dsv_coords, dsv_bz, name="y_gradient").with_suppression_region(suppression)
        target = spec.to_target_field()

        self.assertEqual(spec.dsv_point_count, 2)
        self.assertEqual(spec.suppression_point_count, 2)
        np.testing.assert_array_equal(target.suppressed, np.array([False, False, True, True]))
        np.testing.assert_array_equal(target.target_field_group_inds, np.array([1, 1, 3, 3]))
        np.testing.assert_allclose(target.weights, np.array([1.0, 1.0, 4.0, 4.0]))
        np.testing.assert_allclose(target.b[:, 2], np.array([0.0, 2e-5, 0.0, 0.0]))

    def test_named_target_regions_separate_optimization_and_diagnostics(self) -> None:
        regions = (
            TargetRegion(
                "dsv",
                "useful_field",
                coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.02, 0.0]]),
                target_b=np.array([[1e-5, 0.0, 0.0], [2e-5, 0.0, 0.0]]),
                component=FieldComponentSpec("BdotB0", b0_direction=(1.0, 0.0, 0.0)),
                weight=2.0,
            ),
            TargetRegion(
                "shield",
                "suppression",
                coords=np.array([[0.2, 0.0, 0.0]]),
                weight=5.0,
            ),
            TargetRegion(
                "probe_only",
                "diagnostic_only",
                coords=np.array([[0.0, 0.0, 0.1]]),
                target_b=np.array([[0.0, 0.0, 3e-6]]),
            ),
            TargetRegion(
                "connector_limit",
                "engineering_limit",
                coords=np.array([[0.25, 0.0, 0.0]]),
                hard_limit=1e-6,
            ),
        )

        spec = TargetFieldSpec.from_regions(
            regions,
            name="semantic_target",
            component=FieldComponentSpec("BdotB0", b0_direction=(1.0, 0.0, 0.0)),
        )
        target = spec.to_target_field()

        self.assertEqual(spec.dsv_point_count, 2)
        self.assertEqual(spec.suppression_point_count, 2)
        self.assertEqual(spec.diagnostic_point_count, 1)
        self.assertEqual(len(spec.optimization_regions()), 3)
        self.assertEqual(spec.regions_by_role("diagnostic_only")[0].name, "probe_only")
        self.assertEqual(len(target.coords), 4)
        np.testing.assert_array_equal(target.suppressed, np.array([False, False, True, True]))
        np.testing.assert_allclose(target.weights, np.array([2.0, 2.0, 5.0, 1.0]))
        self.assertEqual(spec.region_summaries()[2]["role"], "diagnostic_only")

    def test_field_component_helpers_preserve_vector_row_order(self) -> None:
        field = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        sensitivity = np.arange(6.0).reshape(3, 2, 1)

        np.testing.assert_allclose(field_component_values(field, "vector"), np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        np.testing.assert_allclose(
            sensitivity_component_matrix(sensitivity, "vector")[:, 0], np.array([0.0, 2.0, 4.0, 1.0, 3.0, 5.0])
        )
        np.testing.assert_allclose(
            component_sample_values(np.array([7.0, 8.0]), "vector"), np.array([7.0, 7.0, 7.0, 8.0, 8.0, 8.0])
        )

    def test_optimization_problem_selects_generalized_target_component_rows(self) -> None:
        mesh = Mesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            faces=np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]),
        )
        part = CoilPart(mesh=mesh)
        part.sensitivity_matrix = np.arange(24.0).reshape(3, 2, 4)
        part.resistance_matrix = np.eye(4)
        target = TargetField(
            coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.01, 0.0]]),
            b=np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
            weights=np.array([1.0, 4.0]),
            suppressed=np.array([False, True]),
            target_field_group_inds=np.array([1, 2]),
        )

        bx_problem = StreamFunctionOptimizationProblem.from_parts(
            [part],
            target,
            CoilGenConfig(target_field_component="Bx"),
        )
        b0_problem = StreamFunctionOptimizationProblem.from_parts(
            [part],
            target,
            CoilGenConfig(target_field_component="BdotB0", b_0_direction=(0.0, 1.0, 0.0)),
        )
        vector_problem = StreamFunctionOptimizationProblem.from_parts(
            [part],
            target,
            CoilGenConfig(target_field_component="vector"),
        )

        np.testing.assert_allclose(bx_problem.unweighted_target_z, np.array([10.0, 40.0]))
        np.testing.assert_allclose(
            bx_problem.unweighted_reduced_sensitivity,
            part.sensitivity_matrix[0] @ bx_problem.reducer.expansion_matrix,
        )
        np.testing.assert_allclose(b0_problem.unweighted_target_z, np.array([20.0, 50.0]))
        np.testing.assert_allclose(
            b0_problem.unweighted_reduced_sensitivity,
            part.sensitivity_matrix[1] @ b0_problem.reducer.expansion_matrix,
        )
        np.testing.assert_allclose(vector_problem.unweighted_target_z, np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]))
        np.testing.assert_allclose(
            vector_problem.unweighted_reduced_sensitivity,
            np.vstack(
                [
                    part.sensitivity_matrix[0, 0],
                    part.sensitivity_matrix[1, 0],
                    part.sensitivity_matrix[2, 0],
                    part.sensitivity_matrix[0, 1],
                    part.sensitivity_matrix[1, 1],
                    part.sensitivity_matrix[2, 1],
                ]
            )
            @ vector_problem.reducer.expansion_matrix,
        )
        np.testing.assert_allclose(vector_problem.target_weights, np.array([1.0, 1.0, 1.0, 4.0, 4.0, 4.0]))
        np.testing.assert_array_equal(
            vector_problem.target_suppressed,
            np.array([False, False, False, True, True, True]),
        )

    def test_coil_design_problem_validates_support_and_conductor(self) -> None:
        mesh = Mesh(
            vertices=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]),
            faces=np.array([[0, 1, 2]]),
        )
        target = TargetFieldSpec.from_arrays(
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 1e-5]]),
        )
        problem = CoilDesignProblem(
            target_field=target,
            support=CoilSupportSpec("single_mesh", (SupportMesh(mesh, role="primary", name="primary"),)),
            conductor=ConductorSpec(width_m=0.003, height_m=0.002),
            metadata=DesignMetadata(name="unit_problem"),
        )

        problem.validate()
        self.assertAlmostEqual(problem.conductor.cross_section_area_m2, 6e-6)
        self.assertEqual(problem.summary()["primary_mesh_count"], 1)
        solver_result = PipelineCoilDesignSolver(dry_run=True).solve(problem)
        self.assertEqual(solver_result.status, "ok")
        self.assertEqual(solver_result.solution.coil_mesh_file.startswith("memory:mesh:"), True)
        self.assertEqual(solver_result.solution.target_field_definition_file.startswith("memory:target:"), True)
        self.assertEqual(validate_config(solver_result.solution).ok, True)
        loaded_mesh = load_mesh_from_config(solver_result.solution, project_root=".")
        loaded_target = load_target_field_definition(
            solver_result.solution.target_field_definition_file,
            None,
            solver_result.solution,
        )
        self.assertEqual(loaded_mesh.n_faces, 1)
        self.assertEqual(len(loaded_target.coords), 1)

    def test_problem_to_config_carries_component_schema_and_layout_intent(self) -> None:
        mesh = Mesh(
            vertices=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]),
            faces=np.array([[0, 1, 2]]),
        )
        target = TargetFieldSpec.from_arrays(
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[1e-5, 0.0, 0.0]]),
            component=FieldComponentSpec("BdotB0", b0_direction=(1.0, 0.0, 0.0)),
        )
        layout = LayoutSpec(
            terminal_strategy="pads",
            terminal_pad_count=2,
            connector_clearance_m=0.004,
            min_bend_radius_m=0.01,
            manufacturing_intent="manufacturing_review",
        )
        problem = CoilDesignProblem(
            target_field=target,
            support=CoilSupportSpec("single_mesh", (SupportMesh(mesh, role="primary"),)),
            layout=layout,
            metadata=DesignMetadata(name="component_bridge"),
        )

        config = problem.to_config()

        self.assertEqual(problem.summary()["problem_schema_version"], PROBLEM_SCHEMA_VERSION)
        self.assertEqual(problem.summary()["layout_readiness_level"], "manufacturing_review")
        self.assertEqual(config.problem_schema_version, PROBLEM_SCHEMA_VERSION)
        self.assertEqual(config.target_field_component, "BdotB0")
        self.assertEqual(tuple(config.b_0_direction), (1.0, 0.0, 0.0))
        self.assertEqual(config.layout_terminal_strategy, "pads")
        self.assertEqual(config.layout_terminal_pad_count, 2)
        self.assertAlmostEqual(config.min_inter_group_connector_clearance, 0.004)
        self.assertAlmostEqual(config.sweep_min_bend_radius, 0.01)

    def test_recommended_presets_cover_cross_project_run_modes(self) -> None:
        for name in (
            "quick",
            "validation",
            "paper_quality",
            "manufacturing_review",
            "local_b0_shim",
            "active_shielded_gradient",
            "quick_debug",
            "balanced",
            "layout_review",
        ):
            preset = recommended_settings(name)
            self.assertIn("levels", preset.values)
            config = preset.apply_to_config(CoilGenConfig(coil_mesh_file="create cylinder mesh"))
            self.assertTrue(validate_config(config).ok)

    def test_parameter_study_spec_generates_cases(self) -> None:
        study = ParameterStudySpec(
            "gap_radius",
            axes=(
                ParameterAxis.from_values("gap_m", [0.6, 0.8], unit="m"),
                ParameterAxis.from_values("inner_radius_m", [0.3, 0.4], unit="m"),
            ),
            fixed_parameters={"target_gradient_strength": 1.0},
        )

        cases = study.cases()

        self.assertEqual(study.dimension, 2)
        self.assertEqual(study.point_count, 4)
        self.assertEqual(len(cases), 4)
        self.assertEqual(cases[0].parameters["target_gradient_strength"], 1.0)
        self.assertEqual(cases[-1].parameters["gap_m"], 0.8)
        self.assertEqual(cases[-1].parameters["inner_radius_m"], 0.4)

    def test_pipeline_geometry_spec_preserves_supported_geometry_config(self) -> None:
        target = TargetFieldSpec.from_arrays(
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 1e-5]]),
        )
        support = CoilSupportSpec(
            "generated_cylinder",
            pipeline_geometry=PipelineGeometrySpec.cylinder(
                length_m=0.3,
                radius_m=0.2,
                angular_divisions=8,
                length_divisions=4,
            ),
        )
        problem = CoilDesignProblem(target_field=target, support=support)

        config = problem.to_config()

        self.assertEqual(config.coil_mesh_file, "create cylinder mesh")
        self.assertEqual(config.surface_is_cylinder_flag, True)
        self.assertEqual(tuple(config.cylinder_mesh_parameter_list[:4]), (0.3, 0.2, 8, 4))
        self.assertEqual(config.target_field_definition_file.startswith("memory:target:"), True)
        self.assertEqual(validate_config(config).ok, True)

    def test_pipeline_geometry_spec_factories_cover_native_geometry_keys(self) -> None:
        specs = (
            PipelineGeometrySpec.cylinder(length_m=0.3, radius_m=0.2, angular_divisions=8, length_divisions=4),
            PipelineGeometrySpec.planar(width_m=0.2, height_m=0.3, lateral_divisions=3, longitudinal_divisions=4),
            PipelineGeometrySpec.circular(radius_m=0.25, radial_divisions=8),
            PipelineGeometrySpec.annular_disc(
                inner_radius_m=0.08,
                outer_radius_m=0.25,
                radial_divisions=3,
                circular_divisions=12,
            ),
            PipelineGeometrySpec.partial_cylinder(
                length_m=0.3,
                radius_m=0.2,
                start_angle_rad=-1.0,
                end_angle_rad=1.0,
                angular_divisions=6,
                length_divisions=4,
            ),
            PipelineGeometrySpec.elliptical_cylinder(
                length_m=0.3,
                radius_x_m=0.24,
                radius_y_m=0.18,
                angular_divisions=8,
                length_divisions=4,
            ),
            PipelineGeometrySpec.superellipse_cylinder(
                length_m=0.3,
                radius_x_m=0.24,
                radius_y_m=0.18,
                exponent=4.0,
                angular_divisions=12,
                length_divisions=4,
            ),
            PipelineGeometrySpec.spherical_cap(
                radius_m=0.2,
                cap_polar_angle_rad=1.2,
                azimuthal_divisions=12,
                polar_divisions=4,
            ),
            PipelineGeometrySpec.ellipsoidal_cap(
                radius_x_m=0.22,
                radius_y_m=0.18,
                radius_z_m=0.16,
                cap_polar_angle_rad=1.2,
                azimuthal_divisions=12,
                polar_divisions=4,
            ),
            PipelineGeometrySpec.conical_frustum(
                length_m=0.3,
                radius_bottom_m=0.2,
                radius_top_m=0.12,
                angular_divisions=12,
                length_divisions=4,
            ),
            PipelineGeometrySpec.toroidal_segment(
                major_radius_m=0.3,
                minor_radius_m=0.06,
                start_angle_rad=-1.0,
                end_angle_rad=1.0,
                tube_start_angle_rad=0.0,
                tube_end_angle_rad=2.0 * np.pi,
                major_divisions=8,
                minor_divisions=6,
            ),
            PipelineGeometrySpec.ruled_surface(
                curve_a_m=((-0.2, -0.1, 0.0), (0.2, -0.1, 0.0)),
                curve_b_m=((-0.2, 0.1, 0.0), (0.2, 0.1, 0.0)),
                span_divisions=3,
            ),
            PipelineGeometrySpec.surface_of_revolution(
                circular_divisions=8,
                z_radius_pairs_m=((-0.2, 0.14), (0.0, 0.2), (0.2, 0.14)),
            ),
            PipelineGeometrySpec.biplanar(
                width_m=0.2,
                height_m=0.3,
                lateral_divisions=3,
                longitudinal_divisions=4,
                plane_distance_m=0.4,
            ),
            PipelineGeometrySpec.circular_biplanar(
                outer_radius_m=0.3,
                edge_length_m=0.05,
                plane_distance_m=0.4,
            ),
            PipelineGeometrySpec.shielded_circular_biplanar(
                primary_outer_radius_m=0.3,
                primary_edge_length_m=0.05,
                shield_outer_radius_m=0.35,
                shield_edge_length_m=0.05,
                plane_distance_m=0.4,
                shield_gap_m=0.04,
            ),
            PipelineGeometrySpec.flanged_split_bore(
                edge_length_m=0.05,
                inner_radius_m=0.2,
                outer_radius_m=0.3,
                gap_m=0.25,
                half_length_m=0.4,
            ),
            PipelineGeometrySpec.shielded_flanged_split_bore(
                edge_length_m=0.05,
                inner_radius_m=0.2,
                outer_radius_m=0.3,
                gap_m=0.25,
                half_length_m=0.4,
                shield_edge_length_m=0.05,
                shield_inner_radius_m=0.35,
                shield_outer_radius_m=0.45,
                shield_gap_m=0.25,
                shield_half_length_m=0.4,
            ),
            PipelineGeometrySpec.imported_mesh(mesh_file="coil_surface.stl", surface_is_cylinder=False),
        )

        keys = [spec.to_config_overrides()["coil_mesh_file"] for spec in specs]

        self.assertEqual(
            keys,
            [
                "create cylinder mesh",
                "create planary mesh",
                "create circular mesh",
                "create annular disc mesh",
                "create partial cylinder mesh",
                "create elliptical cylinder mesh",
                "create superellipse cylinder mesh",
                "create spherical cap mesh",
                "create ellipsoidal cap mesh",
                "create conical frustum mesh",
                "create toroidal segment mesh",
                "create ruled surface mesh",
                "create surface of revolution mesh",
                "create bi-planary mesh",
                "create circular bi-planary mesh",
                "create shielded bi-planary mesh",
                "create flanged split bore mesh",
                "create shielded flanged split bore mesh",
                "coil_surface.stl",
            ],
        )
        self.assertIn("annular_disc", PipelineGeometrySpec.supported_geometry_names())
        self.assertIn("superellipse_cylinder", PipelineGeometrySpec.supported_geometry_names())
        self.assertIn("ruled_surface", PipelineGeometrySpec.supported_geometry_names())
        self.assertIn("shielded_flanged_split_bore", PipelineGeometrySpec.supported_geometry_names())


if __name__ == "__main__":
    unittest.main()
