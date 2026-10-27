from __future__ import annotations

import io
import math
import unittest
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

import numpy as np

from coilgen import (
    build_result_plot_specs,
    coil_spherical_harmonic_coefficient_rows,
    coil_spherical_harmonic_fits,
    CoilGenConfig,
    DesignStudyTable,
    GeometryFactory,
    InterGroupCut,
    MatplotlibPlotSpec,
    StudyMetric,
    available_geometry_factories,
    available_stream_function_optimization_schemes,
    central_slice_selection,
    contour_stability_warnings,
    contour_current_alignment,
    contour_orientation_diagnostics,
    contour_orientation_diagnostics_rows,
    dsv_target_mask,
    field_error_scalar_maps,
    field_error_diagnostics_rows,
    field_relative_error_scalar_maps,
    field_scalar_maps,
    gradient_component_scalar_maps,
    gradient_relative_error_scalar_maps,
    geometry_factory,
    inductance_matrix_summary_dict,
    load_target_field_definition,
    mirror_symmetric_mesh_from_half,
    mesh_surface_diagnostics,
    plot_contour_current_orientation_3d,
    plot_performance_table_matplotlib,
    plot_wire_path_3d,
    plot_wire_path_sequence_3d,
    run_coilgen,
    manufacturing_diagnostics_rows,
    save_contour_group_isometric,
    save_contour_group_projection,
    save_contour_orientation_diagnostics,
    save_field_error_diagnostics,
    save_field_error_map,
    save_field_relative_error_map,
    save_shim_homogeneity_mosaic,
    save_shim_pointcloud_before_after,
    save_gradient_map_isometric,
    save_gradient_relative_error_map_isometric,
    save_interconnect_debug_isometric,
    save_interconnect_debug_projection,
    save_manufacturing_diagnostics,
    save_result_visualizations,
    save_swept_bodies_as_step,
    save_swept_bodies_as_stl,
    SurfaceGeodesicSolver,
    inter_group_diagnostics_rows,
    route_diagnostics_rows,
    save_inter_group_diagnostics,
    save_route_diagnostics,
    save_plot_gallery,
    save_topology_diagnostics,
    save_within_group_cut_diagnostics,
    suppression_field_scalar_maps,
    shim_field_scalar_maps,
    shim_homogeneity_stats,
    shim_spherical_harmonic_comparison,
    shim_spherical_harmonic_fits,
    slice_mosaic_selections,
    fit_spherical_harmonics,
    solid_spherical_harmonic_basis,
    spherical_harmonic_coefficient_rows,
    suppression_target_mask,
    PROTON_GYROMAGNETIC_RATIO_HZ_PER_T,
    result_summary_dict,
    regularization_candidate_rows,
    regularization_selection_report,
    register_geometry_factory,
    stream_function_optimization_scheme,
    study_result_from_coilgen_result,
    target_mode_fit_performance,
    reverse_contour_line,
    topology_diagnostics_rows,
    validate_config,
    validate_and_correct_contour_currents,
    validate_intrinsic_topology_inputs,
    within_group_cut_diagnostics_rows,
)
from coilgen.evaluation import _field_error_values, _field_evaluation_current, layout_target_residual_diagnostics
from coilgen.basis import calculate_basis_functions, calculate_one_ring_by_mesh
from coilgen.contours import (
    _contour_pair_min_distance,
    _enforce_symmetric_extracted_contour_counts,
    _spacing_limited_level_count_and_diagnostics,
    calc_potential_levels,
    clean_contour_line,
    extract_contours,
    regularize_contour_paths,
    regularize_contour_levels,
)
from coilgen.contourability import (
    _contourability_score,
    _contour_roughness_report,
    _current_contour_candidate,
    _repair_candidates_for_report,
    _repair_contour_candidates,
    _target_contour_field_report,
    validate_and_repair_contourability,
)
from coilgen.geometry import (
    build_circular_biplanar_mesh,
    build_cylinder_mesh,
    build_mesh_topology_suppression_surface,
    build_planar_mesh,
    build_shielded_biplanar_mesh,
    build_split_bore_flange_mesh,
    load_mesh_from_config,
    load_secondary_target_mesh,
    normalize_mesh,
    refine_mesh,
    split_disconnected_mesh,
)
from coilgen.inductance import calculate_inductance_by_coil_layout
from coilgen.layout import (
    calculate_unconnected_contour_wire_distances,
    calculate_group_centers,
    cleanup_wire_paths,
    find_wire_segment_contacts_3d,
    find_min_wire_segment_distance_3d,
    find_minimal_contour_distance,
    find_min_mutual_loop_distance,
    _combined_return_group_fusion_candidate,
    _combined_return_is_shorter,
    _crossing_event_segment_selections,
    _crossing_shift_profile,
    _extend_normal_shift_routes_by_arclength,
    _fuse_opened_wire_paths_with_connector,
    _fuse_opened_wire_paths_with_connector_pair_diagnostics,
    _fillet_path_join,
    _mark_reference_wire_path,
    _NormalShiftRoute,
    _outer_group_connection_search_path,
    _repair_shifted_wire_reversal_spurs,
    _repair_surface_parallel_axial_outliers,
    _select_crossing_overpass,
    _strict_geodesic_inter_group_paths,
    _select_group_connection,
    _select_preferred_within_group_rail_candidate,
    _segment_segment_distance_3d,
    _smooth_shift_levels_by_arclength,
    _smooth_wire_path_normals,
    _select_within_group_cut_rail,
    _use_geodesic_inter_group_paths,
    LoopCut,
    LoopCutShape,
    WithinGroupRailCandidate,
    GroupFusionResult,
    interconnect_among_groups,
    interconnect_within_groups,
    open_loop_with_3d_sphere,
    shift_return_paths,
    topological_loop_grouping,
)
from coilgen.matplotlib_visualization import (
    _colorbar_tick_values,
    _format_colorbar_tick,
    _performance_table_rows,
    _regularization_scan_arrays,
    _wrap_plot_title,
)
from coilgen.optimization import (
    BoundaryReducer,
    ResidualConstraintBlock,
    StreamFunctionOptimizationProblem,
    enforce_stream_function_current_density_limit,
    minimum_residual_optimality_certificate,
    _constrained_quadratic_resistance_weight,
    _normalise_constrained_quadratic_objective_mode,
    _residual_constraint_diagnostics,
    _reduced_surface_current_density_metric,
    _select_l_curve_knee,
    _surface_biharmonic_metric,
    _target_error_constraint_blocks,
    solve_stream_function_problem,
    solve_tikhonov_stream_function,
)
from coilgen.plot_book import (
    MatplotlibPlotBook,
    _native_figure_canvas_size,
    _safe_file_stem,
)
from coilgen.gradient_linearity import gradient_linearity_summary
from coilgen.reporting import _gradient_linearity_field_for_result, _suppression_field_summary
from projects.general_examples.scripts.run_standard_examples import (
    dense_generated_example_configs,
    save_generated_example_outputs,
    standard_example_configs,
    validate_generated_example_result,
)
from projects._shared.example_utils import parse_example_parameters
from projects.general_examples.parameters import DEFAULTS_BY_EXAMPLE, default_parameters_for
from coilgen.manufacturing import (
    _capsule_clearance_diagnostics,
    _parallel_transport_path_frames,
    _resolve_sweep_centerline_bends,
    _rotate_between_vectors,
    _sweep_corner_candidates,
    conductor_cross_section_points,
    create_swept_coil_bodies,
    mesh_manufacturing_diagnostics,
)
from coilgen.models import (
    CoilGenResult,
    CoilPart,
    ConnectedGroup,
    ContourLine,
    LoopGroup,
    Mesh,
    TargetField,
    WirePath,
    WirePathGeometry,
    WireSegmentOwner,
)
from coilgen.physics import biot_savart_wire_path, calculate_resistance_matrix, calculate_sensitivity_matrix
from coilgen.pipeline import _prepare_half_mesh_symmetry_if_requested, _restore_refined_single_mesh_symmetry
from coilgen.spectral_stream_function import cotangent_stiffness_and_lumped_mass
from coilgen.symmetry import nearest_indices as symmetry_nearest_indices, reflect_points as symmetry_reflect_points
from coilgen.target import define_target_field, evaluate_field_expression


class FirstPipelineTest(unittest.TestCase):
    def test_inter_group_connection_path_defaults_to_direct_3d(self) -> None:
        self.assertEqual(CoilGenConfig().inter_group_connection_path_method, "direct_3d")

    def test_strict_contourability_rejects_missing_contours(self) -> None:
        mesh = Mesh(
            vertices=np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]], dtype=float),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
        )
        part = CoilPart(mesh=mesh)
        target = TargetField(
            coords=np.array([[0.0, 0.0, 0.01]], dtype=float),
            b=np.array([[0.0, 0.0, 1.0e-6]], dtype=float),
            weights=np.ones(1),
            suppressed=np.zeros(1, dtype=bool),
            target_field_group_inds=np.ones(1, dtype=np.int64),
        )
        config = CoilGenConfig(contourability_validation="strict", contourability_repair_enabled=False)
        with self.assertRaisesRegex(ValueError, "no contours extracted"):
            validate_and_repair_contourability([part], mesh, target, target.b, config)

    def test_contourability_score_prefers_field_reproduction_over_tiny_loop_count(self) -> None:
        base = {
            "contourability_contour_count": 100,
            "contourability_open_loop_count": 0,
            "contourability_bad_orientation_count": 0,
            "contourability_max_closure_gap_m": 0.0,
            "contourability_closure_tolerance_m": 1.0e-8,
            "max_rms_error_percent": 15.0,
            "max_mean_error_percent": 10.0,
            "max_peak_error_percent": 75.0,
            "max_tiny_loop_fraction": 0.5,
        }
        low_level_bad_field = {
            **base,
            "contourability_field_rms_error_percent": 50.0,
            "contourability_field_mean_abs_error_percent": 39.0,
            "contourability_field_peak_error_percent": 487.0,
            "contourability_tiny_loop_fraction": 0.0,
        }
        high_level_better_field = {
            **base,
            "contourability_field_rms_error_percent": 10.0,
            "contourability_field_mean_abs_error_percent": 8.0,
            "contourability_field_peak_error_percent": 70.0,
            "contourability_tiny_loop_fraction": 1.0,
        }

        self.assertLess(_contourability_score(high_level_better_field), _contourability_score(low_level_bad_field))

    def test_contourability_score_treats_absolute_layout_target_as_hard_gate(self) -> None:
        base = {
            "contourability_contour_count": 80,
            "contourability_open_loop_count": 0,
            "contourability_bad_orientation_count": 0,
            "contourability_max_closure_gap_m": 0.0,
            "contourability_closure_tolerance_m": 1.0e-8,
            "max_rms_error_percent": 15.0,
            "max_mean_error_percent": 10.0,
            "max_peak_error_percent": 75.0,
            "max_tiny_loop_fraction": 0.5,
            "contourability_tiny_loop_fraction": 0.0,
            "contourability_target_limit_enforced": True,
        }
        misses_layout_target = {
            **base,
            "contourability_target_metric_ratio": 1.05,
            "contourability_field_rms_error_percent": 1.0,
            "contourability_field_mean_abs_error_percent": 1.0,
            "contourability_field_peak_error_percent": 1.0,
        }
        meets_layout_target = {
            **base,
            "contourability_target_metric_ratio": 0.99,
            "contourability_field_rms_error_percent": 14.0,
            "contourability_field_mean_abs_error_percent": 9.0,
            "contourability_field_peak_error_percent": 70.0,
        }

        self.assertLess(_contourability_score(meets_layout_target), _contourability_score(misses_layout_target))

    def test_contour_target_report_uses_layout_limit_instead_of_stream_limit(self) -> None:
        mesh = Mesh(
            vertices=np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]], dtype=float),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
        )
        part = CoilPart(mesh=mesh, contour_step=1.0)
        target = TargetField(
            coords=np.array([[0.0, 0.0, 0.01], [0.0, 0.0, 0.02]], dtype=float),
            b=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0e-6]], dtype=float),
            weights=np.ones(2),
            suppressed=np.zeros(2, dtype=bool),
            target_field_group_inds=np.ones(2, dtype=np.int64),
        )
        config = CoilGenConfig(
            target_error_constraint_metric="std",
            target_error_std_limit_T=2.0e-6,
            layout_target_error_std_limit_T=0.5e-6,
            field_evaluation_current_source="target_std_projection",
        )

        report = _target_contour_field_report(
            [part],
            np.zeros((2, 3), dtype=float),
            target,
            config,
            component="Bz",
            b0_direction=(0.0, 0.0, 1.0),
            mask=np.ones(2, dtype=bool),
        )

        self.assertTrue(report["contourability_target_limit_enforced"])
        self.assertEqual(report["contourability_target_limit_source"], "layout")
        self.assertAlmostEqual(report["contourability_target_residual_std_T"], 1.0e-6)
        self.assertAlmostEqual(report["contourability_target_allowed_metric_T"], 0.5e-6)
        self.assertAlmostEqual(report["contourability_target_metric_ratio"], 2.0)

    def test_realized_layout_target_diagnostic_checks_connected_wire_field(self) -> None:
        target = TargetField(
            coords=np.array([[0.0, 0.0, 0.01], [0.0, 0.0, 0.02]], dtype=float),
            b=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0e-6]], dtype=float),
            weights=np.ones(2),
            suppressed=np.zeros(2, dtype=bool),
            target_field_group_inds=np.ones(2, dtype=np.int64),
        )
        config = CoilGenConfig(
            target_error_constraint_metric="std",
            layout_target_error_std_limit_T=0.5e-6,
            layout_target_error_acceptance_absolute_T=0.1e-6,
            skip_postprocessing=False,
        )

        diagnostics = layout_target_residual_diagnostics(
            np.zeros((2, 3), dtype=float),
            target,
            config,
        )

        self.assertTrue(diagnostics["layout_target_checked"])
        self.assertFalse(diagnostics["layout_target_met"])
        self.assertEqual(diagnostics["layout_target_source"], "connected_wire")
        self.assertAlmostEqual(diagnostics["layout_target_metric_T"], 1.0e-6)
        self.assertAlmostEqual(diagnostics["layout_target_effective_limit_T"], 0.6e-6)

    def test_smoothness_priority_uses_configured_resistance_tiebreak(self) -> None:
        config = CoilGenConfig(
            constrained_quadratic_objective_mode="smoothness_priority",
            constrained_quadratic_resistance_tiebreak_factor=2.5e-3,
        )
        mode = _normalise_constrained_quadratic_objective_mode(config)

        self.assertEqual(mode, "smoothness_priority")
        self.assertAlmostEqual(_constrained_quadratic_resistance_weight(config, mode), 2.5e-3)
        self.assertAlmostEqual(_constrained_quadratic_resistance_weight(CoilGenConfig(), "balanced"), 1.0)

    def test_contour_roughness_metric_penalizes_zigzag_path(self) -> None:
        mesh = Mesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
        )
        angles = np.linspace(0.0, 2.0 * np.pi, 17)
        smooth_points = np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))
        radii = np.where(np.arange(len(angles)) % 2 == 0, 1.0, 0.65)
        zigzag_points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles), np.zeros_like(angles)))
        smooth_part = CoilPart(mesh=mesh, contour_lines=[ContourLine(v=smooth_points, potential=1.0)])
        zigzag_part = CoilPart(mesh=mesh, contour_lines=[ContourLine(v=zigzag_points, potential=1.0)])

        smooth = _contour_roughness_report([smooth_part])
        zigzag = _contour_roughness_report([zigzag_part])

        self.assertGreater(
            zigzag["contourability_curvature_rms_per_m"],
            smooth["contourability_curvature_rms_per_m"],
        )

    def test_contourability_repair_candidates_include_phase_offsets(self) -> None:
        config = CoilGenConfig(
            levels=220,
            pot_offset_factor=0.25,
            zero_crossing_level_strategy="full_range",
            contourability_repair_level_factors=(1.0,),
            contourability_repair_max_levels=220,
            contourability_repair_phase_offsets=(0.05, 0.25, 1.25),
            contourability_repair_range_percentiles=(100.0,),
            contourability_repair_zero_crossing_strategies=("full_range", "centered"),
        )

        requested = _current_contour_candidate(config, config.levels)
        candidates = _repair_contour_candidates(requested, config)
        keys = {
            (
                candidate.levels,
                candidate.phase_offset,
                candidate.range_percentile,
                candidate.zero_crossing_strategy,
            )
            for candidate in candidates
        }

        self.assertIn((220, 0.25, 100.0, "full_range"), keys)
        self.assertIn((220, 0.05, 100.0, "full_range"), keys)
        self.assertNotIn((220, 1.25, 100.0, "full_range"), keys)
        self.assertEqual(2, len(keys))

    def test_contourability_repair_prunes_denser_levels_after_loop_cap_failure(self) -> None:
        config = CoilGenConfig(
            levels=32,
            contourability_max_contour_count=100,
            contourability_repair_level_factors=(0.75, 1.0, 1.25, 1.5),
            contourability_repair_phase_offsets=(0.25, 0.75),
        )
        requested = _current_contour_candidate(config, config.levels)
        candidates, pruned = _repair_candidates_for_report(
            requested,
            {"max_contour_count": 100, "contourability_contour_count": 104},
            config,
        )

        self.assertGreater(pruned, 0)
        self.assertTrue(candidates)
        self.assertLessEqual(max(candidate.levels for candidate in candidates), config.levels)

    def test_primary_level_method_preserves_equal_range_count_for_same_current_step(self) -> None:
        parts = [_stream_function_range_part(1.0), _stream_function_range_part(1.0)]

        parts, primary_ind = calc_potential_levels(
            parts,
            Mesh(vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]], dtype=np.int64)),
            CoilGenConfig(levels=5, pot_offset_factor=0.5, level_set_method="primary"),
        )

        self.assertEqual(primary_ind, 0)
        self.assertEqual([len(part.potential_level_list) for part in parts], [4, 4])
        self.assertAlmostEqual(parts[0].contour_step, parts[1].contour_step)
        self.assertAlmostEqual(float(np.diff(parts[1].potential_level_list).mean()), parts[0].contour_step)
        self.assertTrue(all(part.contour_level_diagnostics["contour_levels_gauge_invariant"] for part in parts))

    def test_multilayer_phase_deltas_interleave_fixed_step_contours(self) -> None:
        parts = [_stream_function_range_part(1.0), _stream_function_range_part(1.0)]
        config = CoilGenConfig(
            levels=4,
            pot_offset_factor=0.25,
            contour_part_phase_offset_deltas=(0.0, 0.5),
            level_set_method="primary",
            contour_level_grid_mode="fixed_step_phase",
            zero_crossing_level_strategy="full_range",
        )

        parts, _primary_ind = calc_potential_levels(
            parts,
            Mesh(vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]], dtype=np.int64)),
            config,
        )

        self.assertEqual([len(part.potential_level_list) for part in parts], [4, 4])
        step = float(parts[0].contour_step)
        np.testing.assert_allclose(parts[1].potential_level_list - parts[0].potential_level_list, 0.5 * step)
        self.assertAlmostEqual(parts[0].contour_level_diagnostics["contour_phase_effective"], 0.25)
        self.assertAlmostEqual(parts[1].contour_level_diagnostics["contour_phase_effective"], 0.75)

    def test_generated_shielded_parts_lock_symmetric_level_counts_with_shared_current_step(self) -> None:
        parts = [
            _stream_function_range_part(1.0),
            _stream_function_range_part(0.99),
            _stream_function_range_part(0.72),
            _stream_function_range_part(0.69),
        ]

        parts, primary_ind = calc_potential_levels(
            parts,
            Mesh(vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]], dtype=np.int64)),
            CoilGenConfig(
                coil_mesh_file="create shielded bi-planary mesh",
                levels=7,
                pot_offset_factor=0.5,
                level_set_method="primary",
            ),
        )

        self.assertEqual(primary_ind, 0)
        self.assertEqual([len(part.potential_level_list) for part in parts], [6, 6, 4, 4])
        self.assertTrue(all(np.isclose(part.contour_step, parts[0].contour_step) for part in parts))
        self.assertEqual(parts[0].contour_level_diagnostics["symmetry_pair"], "1,2")
        self.assertEqual(parts[2].contour_level_diagnostics["symmetry_pair"], "3,4")
        self.assertTrue(parts[0].contour_level_diagnostics["symmetry_pair_locked"])
        self.assertTrue(parts[2].contour_level_diagnostics["symmetry_pair_locked"])
        self.assertTrue(parts[0].contour_level_diagnostics["symmetry_pair_reduced_level_count"])
        self.assertFalse(parts[1].contour_level_diagnostics["symmetry_pair_reduced_level_count"])
        self.assertTrue(parts[2].contour_level_diagnostics["symmetry_pair_reduced_level_count"])
        self.assertFalse(parts[3].contour_level_diagnostics["symmetry_pair_reduced_level_count"])
        for part in parts:
            levels = part.potential_level_list
            sf_min = float(np.min(part.stream_function))
            sf_max = float(np.max(part.stream_function))
            self.assertAlmostEqual(float(levels[0] - sf_min), float(sf_max - levels[-1]))

    def test_zero_crossing_stream_function_uses_even_paired_levels(self) -> None:
        part = _stream_function_range_part(
            1.0,
            stream_function=np.array([-1.0, -0.2, 0.2, 1.0], dtype=float),
        )

        parts, _primary_ind = calc_potential_levels(
            [part],
            Mesh(vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]], dtype=np.int64)),
            CoilGenConfig(levels=5, pot_offset_factor=0.5, level_set_method="primary"),
        )

        levels = parts[0].potential_level_list
        self.assertEqual(len(levels) % 2, 0)
        self.assertFalse(np.any(np.isclose(levels, 0.0)))
        np.testing.assert_allclose(levels, -levels[::-1])
        self.assertTrue(parts[0].contour_level_diagnostics["zero_crossing_paired_levels"])
        self.assertTrue(parts[0].contour_level_diagnostics["zero_crossing_even_level_count_enforced"])

    def test_paired_contour_levels_are_gauge_invariant(self) -> None:
        stream_function = np.array([-1.0, -0.2, 0.2, 1.0], dtype=float)
        shifted_stream_function = stream_function + 5.0
        config = CoilGenConfig(levels=5, pot_offset_factor=0.5, level_set_method="primary")
        mesh = Mesh(vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]], dtype=np.int64))

        reference_parts, _primary_ind = calc_potential_levels(
            [_stream_function_range_part(1.0, stream_function=stream_function)],
            mesh,
            config,
        )
        shifted_parts, _primary_ind = calc_potential_levels(
            [_stream_function_range_part(1.0, stream_function=shifted_stream_function)],
            mesh,
            config,
        )

        reference_levels = reference_parts[0].potential_level_list
        shifted_levels = shifted_parts[0].potential_level_list
        self.assertEqual(len(reference_levels), len(shifted_levels))
        self.assertEqual(len(reference_levels) % 2, 0)
        np.testing.assert_allclose(shifted_levels - 5.0, reference_levels)
        self.assertAlmostEqual(reference_parts[0].contour_level_diagnostics["contour_level_origin_A"], 0.0)
        self.assertAlmostEqual(shifted_parts[0].contour_level_diagnostics["contour_level_origin_A"], 5.0)
        self.assertTrue(shifted_parts[0].contour_level_diagnostics["contour_levels_gauge_invariant"])

    def test_full_range_zero_crossing_levels_do_not_clip_asymmetric_shim_stream_function(self) -> None:
        part = _stream_function_range_part(
            1.0,
            stream_function=np.array([-1.0, -0.2, 0.2, 3.0], dtype=float),
        )

        parts, _primary_ind = calc_potential_levels(
            [part],
            Mesh(vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]], dtype=np.int64)),
            CoilGenConfig(
                levels=5,
                pot_offset_factor=0.5,
                level_set_method="primary",
                zero_crossing_level_strategy="full_range",
            ),
        )

        levels = parts[0].potential_level_list
        self.assertEqual(len(levels), 5)
        self.assertGreater(float(levels[-1]), 2.0)
        self.assertFalse(np.allclose(levels, -levels[::-1]))
        self.assertFalse(parts[0].contour_level_diagnostics["zero_crossing_paired_levels"])
        self.assertFalse(parts[0].contour_level_diagnostics["zero_crossing_even_level_count_enforced"])

    def test_symmetric_extracted_contour_pruning_removes_unpaired_shield_loops(self) -> None:
        parts = [_stream_function_range_part(1.0) for _ in range(4)]
        for part in parts:
            part.potential_level_list = np.array([0.0, 1.0, 2.0], dtype=float)

        small = _square_loop_at(0.0, 0.0, 0.2, 0.0)
        small.potential = 1.0
        large_a = _square_loop_at(0.0, 0.0, 1.0, 0.0)
        large_a.potential = 1.0
        large_b = _square_loop_at(0.0, 0.0, 1.0, 0.0)
        large_b.potential = 1.0
        parts[2].contour_lines = [small, large_a]
        parts[3].contour_lines = [large_b]
        for part in parts[2:]:
            part.contour_level_diagnostics["symmetry_stream_function_sign"] = 1

        _enforce_symmetric_extracted_contour_counts(
            parts,
            CoilGenConfig(coil_mesh_file="create shielded bi-planary mesh"),
        )

        self.assertEqual([len(parts[2].contour_lines), len(parts[3].contour_lines)], [1, 1])
        self.assertAlmostEqual(parts[2].contour_lines[0].length, large_a.length)
        self.assertEqual(
            parts[2].contour_level_diagnostics["symmetry_pair_extracted_contours_pruned"],
            [1, 0],
        )

    def test_strict_contour_symmetry_accepts_mirrored_generated_pair(self) -> None:
        parts = [_stream_function_range_part(1.0, z=0.5), _stream_function_range_part(1.0, z=-0.5)]

        parts, _primary_ind = calc_potential_levels(
            parts,
            Mesh(vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]], dtype=np.int64)),
            CoilGenConfig(
                coil_mesh_file="create bi-planary mesh",
                contour_symmetry_validation="strict",
                levels=5,
                pot_offset_factor=0.5,
                level_set_method="primary",
            ),
        )

        self.assertTrue(parts[0].contour_level_diagnostics["symmetry_geometry_ok"])
        self.assertTrue(parts[0].contour_level_diagnostics["symmetry_stream_function_ok"])
        self.assertTrue(parts[0].contour_level_diagnostics["symmetry_level_ok"])

    def test_strict_contour_symmetry_rejects_nonmirrored_generated_pair(self) -> None:
        parts = [_stream_function_range_part(1.0), _stream_function_range_part(1.0)]

        with self.assertRaisesRegex(ValueError, "geometric mirror check failed"):
            calc_potential_levels(
                parts,
                Mesh(vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]], dtype=np.int64)),
                CoilGenConfig(
                    coil_mesh_file="create bi-planary mesh",
                    contour_symmetry_validation="strict",
                    levels=5,
                    pot_offset_factor=0.5,
                    level_set_method="primary",
                ),
            )

    def test_strict_contour_symmetry_rejects_nonmirrored_stream_function(self) -> None:
        parts = [
            _stream_function_range_part(1.0, z=0.5, stream_function=np.array([0.0, 0.2, 0.7, 1.0])),
            _stream_function_range_part(1.0, z=-0.5, stream_function=np.array([0.0, 0.1, 0.9, 1.0])),
        ]

        with self.assertRaisesRegex(ValueError, "stream-function mirror check failed"):
            calc_potential_levels(
                parts,
                Mesh(vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]], dtype=np.int64)),
                CoilGenConfig(
                    coil_mesh_file="create bi-planary mesh",
                    contour_symmetry_validation="strict",
                    levels=5,
                    pot_offset_factor=0.5,
                    level_set_method="primary",
                ),
            )

    def test_symmetry_reduction_activates_for_mirrored_generated_problem(self) -> None:
        parts = [_stream_function_range_part(1.0, z=0.5), _stream_function_range_part(1.0, z=-0.5)]
        _assign_symmetric_optimization_matrices(parts)
        target = _two_point_symmetric_target(np.array([1.0, 1.0]))

        problem = StreamFunctionOptimizationProblem.from_parts(
            parts,
            target,
            CoilGenConfig(
                coil_mesh_file="create bi-planary mesh",
                symmetry_reduction="auto",
                contour_symmetry_geometry_tolerance=1e-9,
                symmetry_reduction_target_tolerance=1e-9,
            ),
        )

        self.assertTrue(problem.symmetry_diagnostics["symmetry_reduction_active"])
        self.assertEqual(problem.symmetry_diagnostics["symmetry_reduction_unknown_count_before"], 8)
        self.assertEqual(problem.symmetry_diagnostics["symmetry_reduction_unknown_count_after"], 4)
        self.assertEqual(problem.symmetry_diagnostics["stream_function_gauge_mode_count"], 1)
        self.assertEqual(problem.reduced_sensitivity.shape[1], 3)

    def test_symmetry_reduction_auto_falls_back_for_asymmetric_target(self) -> None:
        parts = [_stream_function_range_part(1.0, z=0.5), _stream_function_range_part(1.0, z=-0.5)]
        _assign_symmetric_optimization_matrices(parts)
        target = _two_point_symmetric_target(np.array([1.0, 2.0]))

        problem = StreamFunctionOptimizationProblem.from_parts(
            parts,
            target,
            CoilGenConfig(coil_mesh_file="create bi-planary mesh", symmetry_reduction="auto"),
        )

        self.assertFalse(problem.symmetry_diagnostics["symmetry_reduction_active"])
        self.assertEqual(problem.symmetry_diagnostics["stream_function_gauge_mode_count"], 2)
        self.assertEqual(problem.reduced_sensitivity.shape[1], 6)
        self.assertIn("target", problem.symmetry_diagnostics["symmetry_reduction_reason"])

    def test_symmetry_reduction_strict_rejects_asymmetric_target(self) -> None:
        parts = [_stream_function_range_part(1.0, z=0.5), _stream_function_range_part(1.0, z=-0.5)]
        _assign_symmetric_optimization_matrices(parts)
        target = _two_point_symmetric_target(np.array([1.0, 2.0]))

        with self.assertRaisesRegex(ValueError, "Symmetry reduction validation failed"):
            StreamFunctionOptimizationProblem.from_parts(
                parts,
                target,
                CoilGenConfig(coil_mesh_file="create bi-planary mesh", symmetry_reduction="strict"),
            )

    def test_symmetry_reduction_diagnostic_policy_keeps_symmetric_basis_for_asymmetric_target(self) -> None:
        parts = [_stream_function_range_part(1.0, z=0.5), _stream_function_range_part(1.0, z=-0.5)]
        _assign_symmetric_optimization_matrices(parts)
        target = _two_point_symmetric_target(np.array([1.0, 2.0]))

        problem = StreamFunctionOptimizationProblem.from_parts(
            parts,
            target,
            CoilGenConfig(
                coil_mesh_file="create bi-planary mesh",
                symmetry_reduction="strict",
                symmetry_reduction_target_parity="even",
                symmetry_reduction_target_parity_policy="diagnostic",
            ),
        )

        diagnostics = problem.symmetry_diagnostics or {}
        self.assertTrue(diagnostics["symmetry_reduction_active"])
        pair = diagnostics["symmetry_reduction_pairs"][0]
        self.assertFalse(
            pair["symmetry_reduction_target_parity_normalized_rms"]
            <= pair["symmetry_reduction_target_parity_tolerance"]
        )
        self.assertFalse(pair["symmetry_reduction_target_parity_required"])

    def test_minimum_residual_certificate_reports_exact_mean_free_projection_floor(self) -> None:
        mesh = Mesh(
            vertices=np.array([[0.0, 0.0, 0.0]], dtype=float),
            faces=np.empty((0, 3), dtype=np.int64),
        )
        reducer = BoundaryReducer(
            full_node_count=1,
            boundary_nodes=[],
            non_boundary_nodes=np.array([0], dtype=np.int64),
            expansion_matrix=np.ones((1, 1), dtype=float),
        )
        matrix = np.ones((2, 1), dtype=float)
        target = np.array([1.0, -1.0], dtype=float)
        problem = StreamFunctionOptimizationProblem(
            sensitivity_matrix=np.zeros((3, 2, 1), dtype=float),
            resistance_matrix=np.ones((1, 1), dtype=float),
            combined_mesh=mesh,
            reducer=reducer,
            target_z=target,
            reduced_sensitivity=matrix,
            reduced_resistance=np.ones((1, 1), dtype=float),
            unweighted_target_z=target,
            unweighted_reduced_sensitivity=matrix,
            target_weights=np.ones(2, dtype=float),
            target_suppressed=np.zeros(2, dtype=bool),
        )
        config = CoilGenConfig(
            target_error_constraint_mode="separate",
            target_error_constraint_metric="std",
            target_error_std_limit_T=2.0,
        )

        certificate = minimum_residual_optimality_certificate(
            problem,
            config,
            solve_engineering_constrained_floor=True,
        )

        self.assertAlmostEqual(certificate["algebraic_projection"]["field_metric_T"], 1.0)
        self.assertAlmostEqual(certificate["stable_projection"]["field_metric_T"], 1.0)
        self.assertEqual(certificate["stable_projection"]["rank"], 0)
        self.assertTrue(certificate["engineering_constrained"]["available"])
        self.assertAlmostEqual(certificate["engineering_constrained"]["field_metric_T"], 1.0)

    def test_single_mesh_symmetry_reduction_activates_for_imported_mirror_mesh(self) -> None:
        part = _single_mesh_x_mirror_part()
        target = _two_point_x_symmetric_target(np.array([1.0, 1.0]))

        problem = StreamFunctionOptimizationProblem.from_parts(
            [part],
            target,
            CoilGenConfig(
                coil_mesh_file="imported_mask.stl",
                symmetry_reduction="strict",
                symmetry_reduction_source="single_mesh",
                symmetry_reduction_mirror_plane_point=(0.0, 0.0, 0.0),
                symmetry_reduction_mirror_plane_normal=(1.0, 0.0, 0.0),
                symmetry_reduction_target_parity="even",
                contour_symmetry_geometry_tolerance=1e-9,
                symmetry_reduction_target_tolerance=1e-9,
            ),
        )

        self.assertTrue(problem.symmetry_diagnostics["symmetry_reduction_active"])
        self.assertEqual(problem.symmetry_diagnostics["symmetry_reduction_unknown_count_before"], 4)
        self.assertEqual(problem.symmetry_diagnostics["symmetry_reduction_unknown_count_after"], 2)
        self.assertTrue(problem.symmetry_diagnostics["stream_function_gauge_removed"])
        self.assertEqual(problem.reduced_sensitivity.shape[1], 1)
        self.assertEqual(
            problem.symmetry_diagnostics["symmetry_reduction_pairs"][0]["symmetry_reduction_source"], "single_mesh"
        )

    def test_half_mesh_reconstruction_clips_and_mirrors_exact_single_mesh(self) -> None:
        mesh = _plane_crossing_square_mesh()

        symmetric, summary = mirror_symmetric_mesh_from_half(
            mesh,
            plane_point=(0.0, 0.0, 0.0),
            plane_normal=(1.0, 0.0, 0.0),
            half_space="positive",
            clip_tolerance=1e-12,
        )

        self.assertGreater(summary["clipped_original_face_count"], 0)
        self.assertGreater(summary["mirror_pair_count"], 0)
        self.assertGreater(summary["mirror_plane_vertex_count"], 0)
        self.assertAlmostEqual(float(np.min(symmetric.vertices[:, 0])), -1.0)
        self.assertAlmostEqual(float(np.max(symmetric.vertices[:, 0])), 1.0)
        self.assertEqual(len(split_disconnected_mesh(symmetric)), 1)
        matches = _assert_exact_x_mirror_vertex_pairs(symmetric)
        self.assertTrue(np.all(matches[matches] == np.arange(symmetric.n_vertices)))
        for face in symmetric.faces:
            mirrored = np.sort(matches[np.asarray(face, dtype=np.int64)])
            face_sets = {tuple(np.sort(candidate)) for candidate in symmetric.faces}
            self.assertIn(tuple(mirrored), face_sets)

    def test_half_mesh_reconstruction_collapses_only_short_seam_edges(self) -> None:
        epsilon = 1.0e-6
        mesh = Mesh(
            vertices=np.array(
                [
                    [-1.0, -1.0, 0.0],
                    [epsilon, -1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [-1.0, 1.0, 0.0],
                ],
                dtype=float,
            ),
            faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        )

        symmetric, summary = mirror_symmetric_mesh_from_half(
            mesh,
            half_space="positive",
            clip_tolerance=1.0e-12,
            plane_snap_tolerance=1.0e-4,
        )

        self.assertGreater(summary["plane_short_edge_collapse_count"], 0)
        self.assertEqual(summary["result_nonmanifold_edge_count"], 0)
        self.assertEqual(summary["result_mirror_seam_boundary_edge_count"], 0)
        self.assertEqual(len(split_disconnected_mesh(symmetric)), 1)

    def test_half_mesh_reconstruction_strict_optimizer_enforces_stream_function_symmetry(self) -> None:
        mesh, matches = _half_reconstructed_square_with_pairs()
        part = CoilPart(mesh=mesh)
        _assign_x_mirror_even_sensitivity(part, matches)
        target = _two_point_x_symmetric_target(np.array([1.0, 1.0]))

        problem = StreamFunctionOptimizationProblem.from_parts(
            [part],
            target,
            CoilGenConfig(
                coil_mesh_file="prepared_symmetric_mask.stl",
                symmetry_reduction="strict",
                symmetry_reduction_source="single_mesh",
                symmetry_reduction_mirror_plane_point=(0.0, 0.0, 0.0),
                symmetry_reduction_mirror_plane_normal=(1.0, 0.0, 0.0),
                symmetry_reduction_target_parity="even",
                contour_symmetry_geometry_tolerance=1e-12,
                symmetry_reduction_target_tolerance=1e-12,
            ),
        )

        self.assertTrue(problem.symmetry_diagnostics["symmetry_reduction_active"])
        self.assertLess(problem.reduced_sensitivity.shape[1], mesh.n_vertices)
        reduced = np.linspace(0.2, 1.0, problem.reduced_sensitivity.shape[1])
        psi = problem.expand(reduced)
        np.testing.assert_allclose(psi[matches], psi, atol=1e-12)

    def test_half_mesh_reconstruction_physical_bz_response_has_target_parity(self) -> None:
        mesh, matches = _half_reconstructed_square_with_pairs()
        part = CoilPart(mesh=mesh)
        _assign_x_mirror_even_sensitivity(part, matches)
        target = _two_point_x_symmetric_target(np.array([1.0, 1.0]))
        config = CoilGenConfig(
            symmetry_reduction="strict",
            symmetry_reduction_source="single_mesh",
            symmetry_reduction_mirror_plane_point=(0.0, 0.0, 0.0),
            symmetry_reduction_mirror_plane_normal=(1.0, 0.0, 0.0),
            symmetry_reduction_target_parity="even",
            contour_symmetry_geometry_tolerance=1e-12,
            symmetry_reduction_target_tolerance=1e-12,
        )
        problem = StreamFunctionOptimizationProblem.from_parts([part], target, config)

        psi = problem.expand(np.linspace(0.1, 0.9, problem.reduced_sensitivity.shape[1]))
        b_field = problem.evaluate_full_b_field(psi)

        np.testing.assert_allclose(b_field[1, 2], b_field[0, 2], rtol=1e-12, atol=1e-12)
        self.assertGreater(abs(float(b_field[0, 2])), 0.0)

    def test_quantized_nearest_indices_handles_large_exact_mirror_mesh_without_scipy(self) -> None:
        y = np.linspace(-0.09, 0.09, 40)
        z = np.linspace(-0.06, 0.14, 25)
        half = np.array([(0.04 + 0.001 * row, yy, zz) for row, yy in enumerate(y) for zz in z], dtype=float)
        points = np.vstack((half, half * np.array([-1.0, 1.0, 1.0])))
        reflected = points * np.array([-1.0, 1.0, 1.0])
        reflected += 2.0e-11

        matches = symmetry_nearest_indices(reflected, points, max_bruteforce_work=1)

        self.assertIsNotNone(matches)
        matches = np.asarray(matches, dtype=np.int64)
        distances = np.linalg.norm(reflected - points[matches], axis=1)
        self.assertLess(float(np.max(distances)), 1.0e-9)

    def test_half_mesh_source_is_validated_and_reduced_as_single_mesh_after_preparation(self) -> None:
        report = validate_config(
            CoilGenConfig(
                symmetry_reduction_source="half_mesh",
                symmetry_reduction_half_space="positive",
                symmetry_reduction_clip_tolerance=1e-9,
                plot_flag=False,
                save_stl_flag=False,
            )
        )

        source_errors = [message for message in report.errors if message.field == "symmetry_reduction_source"]
        self.assertEqual(source_errors, [])

    def test_pipeline_half_mesh_preparation_runs_before_part_splitting(self) -> None:
        prepared, diagnostics = _prepare_half_mesh_symmetry_if_requested(
            _plane_crossing_square_mesh(),
            CoilGenConfig(
                symmetry_reduction="strict",
                symmetry_reduction_source="half_mesh",
                symmetry_reduction_mirror_plane_point=(0.0, 0.0, 0.0),
                symmetry_reduction_mirror_plane_normal=(1.0, 0.0, 0.0),
                symmetry_reduction_half_space="positive",
                symmetry_reduction_clip_tolerance=1e-12,
            ),
        )

        self.assertTrue(diagnostics["symmetry_half_mesh_active"])
        self.assertEqual(diagnostics["symmetry_half_mesh_half_space"], "positive")
        self.assertEqual(len(split_disconnected_mesh(prepared)), 1)
        _assert_exact_x_mirror_vertex_pairs(prepared)

    def test_mesh_refinement_subdivision(self) -> None:
        mesh = normalize_mesh(build_cylinder_mesh(0.3, 0.1, 8, 4, 1, 0, 0, 0))

        refined = refine_mesh(mesh, 1)

        self.assertEqual(refined.n_faces, mesh.n_faces * 4)
        self.assertGreater(refined.n_vertices, mesh.n_vertices)
        self.assertFalse(hasattr(refined, "uv"))

    def test_refined_single_mesh_reconstruction_preserves_bz_sensitivity_parity(self) -> None:
        mesh, _ = mirror_symmetric_mesh_from_half(
            _centered_plane_grid_mesh(7, 5),
            plane_normal=(1.0, 0.0, 0.0),
            half_space="positive",
            clip_tolerance=1.0e-12,
        )
        config = CoilGenConfig(
            iteration_num_mesh_refinement=1,
            symmetry_reduction="strict",
            symmetry_reduction_source="single_mesh",
            symmetry_reduction_mirror_plane_point=(0.0, 0.0, 0.0),
            symmetry_reduction_mirror_plane_normal=(1.0, 0.0, 0.0),
            gauss_order=2,
            progress_flag=False,
        )
        refined = refine_mesh(mesh, 1)
        rebuilt, diagnostics = _restore_refined_single_mesh_symmetry([refined], config)
        part = CoilPart(mesh=rebuilt[0])
        calculate_basis_functions(calculate_one_ring_by_mesh([part]))
        coords = np.array(
            [[-0.3, -0.2, 0.2], [0.3, -0.2, 0.2], [-0.2, 0.25, 0.3], [0.2, 0.25, 0.3]],
            dtype=float,
        )
        target = TargetField(
            coords=coords,
            b=np.zeros((len(coords), 3), dtype=float),
            weights=np.ones(len(coords), dtype=float),
            suppressed=np.zeros(len(coords), dtype=bool),
            target_field_group_inds=np.ones(len(coords), dtype=np.int64),
        )
        calculate_sensitivity_matrix([part], target, config)
        vertex_pairs = _assert_exact_x_mirror_vertex_pairs(part.mesh)
        target_pairs = np.array([1, 0, 3, 2], dtype=np.int64)
        paired_columns = part.sensitivity_matrix[2] + part.sensitivity_matrix[2][:, vertex_pairs]

        self.assertTrue(diagnostics["refinement_symmetry_reconstruction_enabled"])
        np.testing.assert_allclose(paired_columns[target_pairs], paired_columns, rtol=1.0e-10, atol=1.0e-16)

    def test_reduced_current_density_metric_matches_dense_assembly(self) -> None:
        mesh = normalize_mesh(build_cylinder_mesh(0.3, 0.1, 8, 4, 1, 0, 0, 0))
        part = CoilPart(mesh=mesh)
        calculate_basis_functions(calculate_one_ring_by_mesh([part]))
        reducer = BoundaryReducer.from_mesh(mesh)

        # Keep the dense reference in the regression suite; production assembles
        # directly in the reduced basis and should not allocate this nodal Gram matrix.
        triangles = mesh.vertices[mesh.faces]
        face_areas = 0.5 * np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
        )
        density = part.current_density_mat
        dense_reduced = reducer.reduce_square(np.einsum("ifa,jfa,f->ij", density, density, face_areas, optimize=True))
        element_reduced = _reduced_surface_current_density_metric([part], reducer)

        np.testing.assert_allclose(element_reduced, dense_reduced, rtol=1.0e-12, atol=1.0e-12)

    def test_cylinder_sensitivity_annihilates_constant_stream_function(self) -> None:
        mesh = normalize_mesh(build_cylinder_mesh(0.3, 0.1, 8, 4, 1, 0, 0, 0))
        part = CoilPart(mesh=mesh)
        calculate_one_ring_by_mesh([part])
        calculate_basis_functions([part])
        target = TargetField(
            coords=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.02, -0.01, 0.03],
                    [-0.03, 0.01, -0.04],
                ],
                dtype=float,
            ),
            b=np.zeros((3, 3), dtype=float),
            weights=np.ones(3, dtype=float),
            suppressed=np.zeros(3, dtype=bool),
            target_field_group_inds=np.ones(3, dtype=np.int64),
        )

        calculate_sensitivity_matrix([part], target, CoilGenConfig(gauss_order=2, progress_flag=False))

        constant_stream = np.ones(mesh.n_vertices, dtype=float)
        constant_field = np.column_stack(
            [
                part.sensitivity_matrix[0] @ constant_stream,
                part.sensitivity_matrix[1] @ constant_stream,
                part.sensitivity_matrix[2] @ constant_stream,
            ]
        )
        np.testing.assert_allclose(constant_field, 0.0, atol=1e-16)

    def test_shielded_biplanar_normals_match_circular_biplanar_convention(self) -> None:
        circular = build_circular_biplanar_mesh(0.25, 0.8, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.2)
        shielded = build_shielded_biplanar_mesh(0.8, 0.25, 0.88, 0.25, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.2, 0.08)

        self.assertGreater(_mean_face_normal_z_at(circular, -0.6), 0.9)
        self.assertLess(_mean_face_normal_z_at(circular, 0.6), -0.9)
        for z in (-0.68, -0.6):
            self.assertGreater(_mean_face_normal_z_at(shielded, z), 0.9)
        for z in (0.6, 0.68):
            self.assertLess(_mean_face_normal_z_at(shielded, z), -0.9)

    def test_shielded_biplanar_shield_is_primary_offset_copy(self) -> None:
        separation = 0.07
        mesh = build_shielded_biplanar_mesh(
            0.5,
            0.1,
            0.9,
            0.2,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            separation,
        )
        n_vertices = mesh.n_vertices // 4
        n_faces = mesh.n_faces // 4

        primary_plus = mesh.vertices[:n_vertices]
        primary_minus = mesh.vertices[n_vertices : 2 * n_vertices]
        shield_plus = mesh.vertices[2 * n_vertices : 3 * n_vertices]
        shield_minus = mesh.vertices[3 * n_vertices : 4 * n_vertices]

        np.testing.assert_allclose(
            shield_plus - primary_plus,
            np.tile(np.array([0.0, 0.0, separation]), (n_vertices, 1)),
        )
        np.testing.assert_allclose(
            shield_minus - primary_minus,
            np.tile(np.array([0.0, 0.0, -separation]), (n_vertices, 1)),
        )
        np.testing.assert_array_equal(mesh.faces[:n_faces] + 2 * n_vertices, mesh.faces[2 * n_faces : 3 * n_faces])
        np.testing.assert_array_equal(
            mesh.faces[n_faces : 2 * n_faces] + 2 * n_vertices,
            mesh.faces[3 * n_faces : 4 * n_faces],
        )

    def test_shielded_split_bore_shield_is_primary_offset_copy(self) -> None:
        edge_len = 0.14
        r_inner = 0.45
        r_outer = 0.85
        gap = 1.1
        half_length = 0.55
        separation = 0.05
        primary = build_split_bore_flange_mesh(edge_len, r_inner, r_outer, gap, half_length)
        shielded = build_split_bore_flange_mesh(
            edge_len,
            r_inner,
            r_outer,
            gap,
            half_length,
            shield=True,
            shield_r_inner=r_outer + separation,
        )

        n_vertices = primary.n_vertices
        n_faces = primary.n_faces
        self.assertEqual(shielded.n_vertices, 2 * n_vertices)
        self.assertEqual(shielded.n_faces, 2 * n_faces)
        np.testing.assert_array_equal(shielded.faces[n_faces:], primary.faces + n_vertices)

        shield_vertices = shielded.vertices[n_vertices:]
        offsets = shield_vertices - primary.vertices
        np.testing.assert_allclose(np.linalg.norm(offsets, axis=1), separation)

        radial = np.linalg.norm(primary.vertices[:, :2], axis=1)
        shield_radial = np.linalg.norm(shield_vertices[:, :2], axis=1)
        z = primary.vertices[:, 2]
        tol = 1e-10
        cylinder = np.isclose(radial, r_inner, rtol=0.0, atol=tol) & ~np.isclose(
            np.abs(z), gap / 2.0, rtol=0.0, atol=tol
        )
        flange = np.isclose(np.abs(z), gap / 2.0, rtol=0.0, atol=tol) & ~np.isclose(radial, r_inner, rtol=0.0, atol=tol)
        corner = np.isclose(radial, r_inner, rtol=0.0, atol=tol) & np.isclose(np.abs(z), gap / 2.0, rtol=0.0, atol=tol)

        np.testing.assert_allclose(shield_radial[cylinder] - radial[cylinder], separation)
        np.testing.assert_allclose(offsets[cylinder, 2], 0.0)
        np.testing.assert_allclose(shield_radial[flange] - radial[flange], 0.0)
        np.testing.assert_allclose(offsets[flange, 2], separation * np.sign(z[flange]))
        np.testing.assert_allclose(shield_radial[corner] - radial[corner], separation / math.sqrt(2.0))
        np.testing.assert_allclose(offsets[corner, 2], separation / math.sqrt(2.0) * np.sign(z[corner]))

    def test_topology_suppression_surface_uses_shield_vertices(self) -> None:
        shield_separation = 0.08
        suppression_separation = 0.03
        coil = normalize_mesh(
            build_shielded_biplanar_mesh(
                0.5,
                0.1,
                0.5,
                0.1,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                shield_separation,
            )
        )
        parts = [refine_mesh(part, 1) for part in split_disconnected_mesh(coil)]
        suppression = normalize_mesh(
            build_mesh_topology_suppression_surface(
                coil_meshes=parts,
                separation=suppression_separation,
                source="shield",
            )
        )
        suppression_parts = split_disconnected_mesh(suppression)
        primary_parts = parts[:2]
        shield_parts = parts[2:]

        self.assertEqual(len(suppression_parts), len(shield_parts))
        for primary, shield, surface in zip(primary_parts, shield_parts, suppression_parts):
            self.assertEqual(surface.n_vertices, shield.n_vertices)
            self.assertEqual(surface.n_faces, shield.n_faces)
            directions = shield.vertices - primary.vertices
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            np.testing.assert_allclose(surface.vertices, shield.vertices + suppression_separation * directions)
            np.testing.assert_allclose(
                np.linalg.norm(surface.vertices - shield.vertices, axis=1),
                suppression_separation,
            )

    def test_topology_suppression_surface_all_mode_offsets_cylinder_outward(self) -> None:
        separation = 0.04
        cylinder = normalize_mesh(build_cylinder_mesh(0.6, 0.3, 20, 6, 1, 0, 0, 0))
        suppression = normalize_mesh(
            build_mesh_topology_suppression_surface(coil_mesh=cylinder, separation=separation, source="all")
        )

        self.assertEqual(suppression.n_vertices, cylinder.n_vertices)
        self.assertEqual(suppression.n_faces, cylinder.n_faces)
        original_radius = np.linalg.norm(cylinder.vertices[:, :2], axis=1)
        suppression_radius = np.linalg.norm(suppression.vertices[:, :2], axis=1)
        np.testing.assert_allclose(suppression_radius - original_radius, separation, atol=2e-3)

    def test_intrinsic_grouping_and_3d_interconnect_from_extracted_contours(self) -> None:
        part = _radial_contour_part(levels=(0.16, 0.49))

        parts = topological_loop_grouping([part])
        self.assertEqual(len(parts[0].groups), 1)
        self.assertEqual(len(parts[0].groups[0].loops), 2)
        self.assertTrue(parts[0].loop_in_loop_matrix[1, 0])
        self.assertEqual(parts[0].groups[0].outer_loop_indices, (1,))

        parts = calculate_group_centers(parts)
        parts = interconnect_within_groups(parts, CoilGenConfig(interconnection_cut_width=0.12))
        group = parts[0].groups[0]
        self.assertEqual(len(parts[0].connected_groups), 1)
        self.assertEqual([cut.shape for cut in group.opening_cuts_v], [(2, 3), (2, 3)])

        parts = interconnect_among_groups(parts)
        parts = shift_return_paths(parts, CoilGenConfig(normal_shift_length=0.01))
        parts = cleanup_wire_paths(parts)
        wire = parts[0].wire_path
        self.assertIsNotNone(wire)
        self.assertEqual(wire.v.shape[1], 3)
        self.assertTrue(np.linalg.norm(wire.v[0] - wire.v[-1]) <= 1e-8)
        self.assertTrue(parts[0].route_diagnostics["closed_v"])
        self.assertNotIn("closed_uv", parts[0].route_diagnostics)
        loop_owners = [owner for owner in wire.segment_owners if owner.kind == "contour_loop"]
        self.assertTrue(loop_owners)
        self.assertEqual({owner.loop_index for owner in loop_owners}, {0, 1})

    def test_within_group_cut_selection_prefers_straight_return_rail(self) -> None:
        group, cut_infos = _rail_selection_group()

        selected = _select_within_group_cut_rail(
            group,
            cut_infos,
            CoilGenConfig(interconnection_cut_width=0.04),
            group_ind=0,
        )

        self.assertEqual(selected.side, "high")
        self.assertLess(
            selected.score,
            _select_within_group_cut_rail(
                group,
                cut_infos,
                CoilGenConfig(interconnection_cut_width=0.04, force_cut_selection=("low",)),
                group_ind=0,
            ).score,
        )
        self.assertGreater(len(selected.connected_group.return_path_v), 1)

    def test_within_group_cut_selection_breaks_numerical_ties_to_high_side(self) -> None:
        empty = np.zeros((0, 3), dtype=float)
        high = WithinGroupRailCandidate(
            side="high",
            opened_loops=[],
            opening_cuts=[],
            connected_group=ConnectedGroup(v=empty, return_path_v=empty, spiral_in_v=empty, spiral_out_v=empty),
            score=1.0 + 5.0e-7,
        )
        low = WithinGroupRailCandidate(
            side="low",
            opened_loops=[],
            opening_cuts=[],
            connected_group=ConnectedGroup(v=empty, return_path_v=empty, spiral_in_v=empty, spiral_out_v=empty),
            score=1.0,
        )

        selected = _select_preferred_within_group_rail_candidate([high, low])

        self.assertEqual(selected.side, "high")

    def test_within_group_cut_selection_honors_forced_side_without_jumping(self) -> None:
        group, cut_infos = _rail_selection_group()

        selected = _select_within_group_cut_rail(
            group,
            cut_infos,
            CoilGenConfig(interconnection_cut_width=0.04, force_cut_selection=("low",)),
            group_ind=0,
        )

        self.assertEqual(selected.side, "low")
        self.assertEqual(len(selected.opening_cuts), len(group.loops))

    def test_within_group_cut_selection_searches_alternative_pair_positions(self) -> None:
        group, cut_infos = _multi_pair_rail_selection_group()

        selected = _select_within_group_cut_rail(
            group,
            cut_infos,
            CoilGenConfig(interconnection_cut_width=0.04, within_group_cut_candidate_count=4),
            group_ind=0,
        )

        self.assertEqual(selected.side, "high")
        self.assertEqual(selected.pair_indices, (0, 1, 0))
        self.assertGreater(selected.candidate_count, 1)

    def test_3d_sphere_opening_returns_3d_cut_points(self) -> None:
        loop = _circle_loop(radius=1.0, samples=65)

        opened, cut_points = open_loop_with_3d_sphere(loop, np.array([1.0, 0.0, 0.0]), 0.35)

        self.assertLess(opened.length, loop.length)
        self.assertEqual(cut_points.shape, (2, 3))
        self.assertGreater(np.linalg.norm(opened.v[0] - opened.v[-1]), 1e-6)
        self.assertAlmostEqual(np.linalg.norm(cut_points[0] - cut_points[1]), 0.35, places=5)

    def test_within_group_join_fillet_removes_right_angle_corner(self) -> None:
        before = np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
        after = np.asarray([[0.01, 0.0, 0.0], [0.01, 0.01, 0.0]])

        rounded = _fillet_path_join(before, after, radius=0.003, angle_step_deg=8.0)

        directions = np.diff(rounded, axis=0)
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        turns = np.degrees(np.arccos(np.clip(np.sum(directions[:-1] * directions[1:], axis=1), -1.0, 1.0)))
        self.assertGreater(len(rounded), len(before) + len(after) - 1)
        self.assertLess(float(np.max(turns)), 15.0)

    def test_normal_shift_ramp_uses_physical_arclength(self) -> None:
        points = np.asarray([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
        levels = np.asarray([1.0, 0.0, 0.0, 0.0])

        ramp = _smooth_shift_levels_by_arclength(levels, points, transition_length=0.05)

        self.assertEqual(ramp[0], 1.0)
        self.assertGreater(ramp[1], 0.99)
        self.assertEqual(ramp[2], 0.0)

    def test_normal_shift_marks_inserted_points_inside_return_path(self) -> None:
        wire = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [3.0, 3.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        reference = wire[[1, 3]]
        levels = np.zeros(len(wire), dtype=float)

        _mark_reference_wire_path(wire, reference, levels)

        np.testing.assert_array_equal(levels[1:4], np.ones(3))
        self.assertEqual(float(np.sum(levels)), 3.0)

    def test_normal_shift_smooths_mirrored_mesh_seam_normals(self) -> None:
        normals = np.asarray(
            [
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        normals /= np.linalg.norm(normals, axis=1)[:, None]

        smoothed = _smooth_wire_path_normals(normals, 3)

        self.assertTrue(np.allclose(np.linalg.norm(smoothed, axis=1), 1.0))
        self.assertLess(float(np.ptp(smoothed[:, 0])), 0.1)

    def test_connected_wire_plot_deduplicates_shifted_section_legend(self) -> None:
        mesh = Mesh(
            vertices=np.asarray([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0]]),
            faces=np.asarray([[0, 1, 2]]),
        )
        part = CoilPart(mesh=mesh)
        part.wire_path = WirePath(
            np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [0.005, 0.0, 0.0],
                    [0.01, 0.0, 0.0],
                    [0.01, 0.005, 0.0],
                    [0.005, 0.005, 0.0],
                    [0.0, 0.005, 0.0],
                ]
            )
        )
        part.shift_array = np.asarray([0.0, 1.0, 1.0, 0.0, 1.0, 1.0])

        figure = plot_wire_path_3d(part, show=False)
        labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]

        self.assertEqual(labels.count("normal-offset return path section"), 1)

    def test_contour_current_orientation_plot_shows_arrows_and_signed_chirality(self) -> None:
        part = _radial_contour_part(levels=(0.16, 0.49))
        part.contour_lines[1] = reverse_contour_line(part.contour_lines[1])

        figure = plot_contour_current_orientation_3d(part, show=False)
        axis = figure.axes[0]
        legend_labels = [text.get_text() for text in axis.get_legend().get_texts()]
        annotations = [text.get_text() for text in axis.texts]

        self.assertIn("positive chirality (chi=+1)", legend_labels)
        self.assertIn("negative chirality (chi=-1)", legend_labels)
        self.assertIn("arrowheads: actual current direction", legend_labels)
        self.assertTrue(any("L0" in text for text in annotations))
        self.assertTrue(any("L1" in text for text in annotations))
        self.assertTrue(any("chi+" in text for text in annotations))
        self.assertTrue(any("chi-" in text for text in annotations))

    def test_wire_path_sequence_plot_shows_owner_order_and_connector_geometry(self) -> None:
        mesh = Mesh(
            vertices=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [0.04, 0.0, 0.0],
                    [0.04, 0.04, 0.0],
                    [0.0, 0.04, 0.0],
                ]
            ),
            faces=np.asarray([[0, 1, 2], [0, 2, 3]]),
        )
        part = CoilPart(mesh=mesh)
        part.wire_path = WirePath(
            np.asarray(
                [
                    [0.005, 0.005, 0.0],
                    [0.030, 0.005, 0.0],
                    [0.030, 0.020, 0.0],
                    [0.020, 0.030, 0.0],
                    [0.005, 0.030, 0.006],
                ]
            ),
            segment_owners=[
                WireSegmentOwner(kind="contour_loop", group_index=0, loop_index=3),
                WireSegmentOwner(kind="contour_loop", group_index=0, loop_index=3),
                WireSegmentOwner(kind="contour_group", group_index=0),
                WireSegmentOwner(
                    kind="inter_group_connector",
                    connector_index=0,
                    group1_index=0,
                    group2_index=1,
                    connector_role="primary",
                ),
            ],
            segment_geometry=[
                WirePathGeometry.SURFACE,
                WirePathGeometry.SURFACE,
                WirePathGeometry.SURFACE,
                WirePathGeometry.FREE_3D,
            ],
        )

        figure = plot_wire_path_sequence_3d(part, show=False)
        axis = figure.axes[0]
        legend_labels = [text.get_text() for text in axis.get_legend().get_texts()]
        annotations = [text.get_text() for text in axis.texts]

        self.assertIn("winding segment", legend_labels)
        self.assertIn("surface connection", legend_labels)
        self.assertIn("free-3D connector", legend_labels)
        self.assertIn("IN terminal", legend_labels)
        self.assertIn("OUT terminal", legend_labels)
        self.assertTrue(any("1:L3" in text for text in annotations))
        self.assertTrue(any("C0" in text for text in annotations))
        self.assertEqual(figure.axes[1].get_ylabel(), "normalized traversal distance, s / L")

    def test_3d_sphere_opening_preserves_cyclic_loop_order(self) -> None:
        radius = 0.25
        angle = np.linspace(0.0, 2.0 * np.pi, 129, endpoint=False)
        loop = ContourLine(
            v=np.column_stack((radius * np.cos(angle), radius * np.sin(angle), np.zeros_like(angle))),
            potential=0.0,
        )
        cut_angle = 1.37
        cut_center = np.array([radius * np.cos(cut_angle), radius * np.sin(cut_angle), 0.0])

        opened, _ = open_loop_with_3d_sphere(loop, cut_center, 0.08)

        segment_lengths = np.linalg.norm(np.diff(opened.v, axis=0), axis=1)
        midpoints = 0.5 * (opened.v[:-1] + opened.v[1:])
        midpoint_radius = np.linalg.norm(midpoints[:, :2], axis=1)
        self.assertLess(float(np.max(segment_lengths)), 0.03)
        self.assertLess(float(np.max(radius - midpoint_radius)), 0.002)

    def test_3d_sphere_opening_tiny_cut_uses_nondegenerate_fallback(self) -> None:
        loop = _circle_loop(radius=1.0, samples=33)

        opened, cut_points = open_loop_with_3d_sphere(loop, np.array([1.0, 0.0, 0.0]), 1e-14)

        self.assertGreater(len(opened.v), 2)
        self.assertEqual(cut_points.shape, (2, 3))
        self.assertGreater(np.linalg.norm(cut_points[0] - cut_points[1]), 1e-10)

    def test_loop_distance_metrics_are_3d(self) -> None:
        loop_a = _square_loop_at(0.0, 0.0, 1.0, 0.0)
        loop_b = _square_loop_at(2.0, 0.0, 1.0, 0.0)

        mutual = find_min_mutual_loop_distance(loop_a, loop_b)

        self.assertAlmostEqual(mutual.distance, 1.0)
        self.assertEqual(mutual.near_point_a.v.shape, (3,))
        self.assertEqual(mutual.near_point_b.v.shape, (3,))

    def test_unconnected_contour_wire_distance_records_min_segment_clearance(self) -> None:
        loop_a = _square_loop_at(0.0, 0.0, 1.0, 0.0)
        loop_b = _square_loop_at(1.2, 0.0, 1.0, 0.0)
        part = CoilPart(
            mesh=Mesh(
                vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                faces=np.array([[0, 1, 2]], dtype=np.int64),
            ),
            contour_lines=[loop_a, loop_b],
        )

        part = calculate_unconnected_contour_wire_distances([part], CoilGenConfig())[0]

        self.assertAlmostEqual(part.unconnected_contour_min_wire_distance, 0.2)
        self.assertEqual(
            part.unconnected_contour_wire_distance_diagnostics["unconnected_contours_wire_distance_pair_count"], 1
        )
        self.assertEqual(part.unconnected_contour_wire_distance_diagnostics["unconnected_contours_min_wire_loop_a"], 0)
        self.assertEqual(part.unconnected_contour_wire_distance_diagnostics["unconnected_contours_min_wire_loop_b"], 1)

    def test_surface_geodesic_solver_point_and_contour_queries(self) -> None:
        mesh = normalize_mesh(build_cylinder_mesh(0.5, 1.0, 64, 4, 1, 0, 0, 0))
        solver = SurfaceGeodesicSolver(mesh)
        point_a = np.array([0.0, 1.0, -0.05])
        point_b = np.array([0.0, -1.0, -0.05])
        contour_b = ContourLine(v=point_b.reshape(1, 3), potential=0.0)

        point_result = solver.distance_between_points(point_a, point_b)
        point_contour_result = solver.distance_point_to_contour(point_a, contour_b)
        contour_result = solver.distance_between_contours(
            ContourLine(v=point_a.reshape(1, 3), potential=0.0), contour_b
        )

        self.assertGreater(point_result.distance, 2.5)
        self.assertTrue(np.isfinite(point_contour_result.distance))
        self.assertTrue(np.isfinite(contour_result.distance))
        self.assertGreater(len(point_result.path_v), 2)

    def test_surface_geodesic_point_sets_use_local_triangle_support(self) -> None:
        mesh = build_planar_mesh(0.25, 0.25, 18, 18, 1, 0, 0, 0, 0, 0, 0)
        solver = SurfaceGeodesicSolver(mesh)
        central_a = np.array([-0.00823111, 0.0, 0.0])
        central_b = np.array([0.00823004, -0.00054962, 0.0])
        decoy_a = np.array([-0.01295618, -0.05555556, 0.0])
        decoy_b = np.array([0.01299591, -0.05555556, 0.0])

        result = solver.distance_between_point_sets(
            np.vstack((central_a, decoy_a)),
            np.vstack((central_b, decoy_b)),
        )

        self.assertEqual((result.index_a, result.index_b), (0, 0))
        self.assertLess(result.distance, np.linalg.norm(decoy_a - decoy_b))

    def test_surface_geodesic_dijkstra_stops_after_target_settled(self) -> None:
        mesh = _grid_mesh(30, 30)
        solver = SurfaceGeodesicSolver(mesh)

        result = solver.distance_between_points(
            np.array([1.1, 1.1, 0.0]),
            np.array([4.1, 1.1, 0.0]),
        )

        self.assertTrue(np.isfinite(result.distance))
        self.assertLess(solver.last_dijkstra_settled_count, mesh.n_vertices // 2)

    def test_surface_geodesic_reuses_sample_cache(self) -> None:
        part = _radial_contour_part(levels=(0.16, 0.49))
        solver = SurfaceGeodesicSolver(part.mesh)

        first = solver.distance_between_contours(part.contour_lines[0], part.contour_lines[1])
        second = solver.distance_between_contours(part.contour_lines[0], part.contour_lines[1])

        self.assertAlmostEqual(first.distance, second.distance)
        self.assertEqual(solver.sample_cache_misses, 2)
        self.assertEqual(solver.sample_cache_hits, 2)

        solver.clear_sample_cache()
        self.assertEqual(solver.sample_cache_misses, 0)
        self.assertEqual(solver.sample_cache_hits, 0)

    def test_geodesic_inter_group_connector_inserts_surface_path(self) -> None:
        mesh = _grid_mesh(3, 3)
        solver = SurfaceGeodesicSolver(mesh)
        path_a = WirePath(v=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float))
        path_b = WirePath(v=np.array([[2.0, 1.0, 0.0], [2.0, 0.0, 0.0]], dtype=float))

        direct, direct_connector = _fuse_opened_wire_paths_with_connector(path_a, path_b)
        geodesic, geodesic_connector = _fuse_opened_wire_paths_with_connector(path_a, path_b, solver)

        self.assertEqual(len(direct_connector), 2)
        self.assertGreater(len(geodesic_connector), 2)
        self.assertGreater(len(geodesic.v), len(direct.v))
        self.assertTrue(np.linalg.norm(direct.v[0] - direct.v[-1]) <= 1e-8)
        self.assertTrue(np.linalg.norm(geodesic.v[0] - geodesic.v[-1]) <= 1e-8)
        self.assertTrue(np.allclose(geodesic_connector[0], path_a.v[-1]))
        self.assertTrue(np.allclose(geodesic_connector[-1], path_b.v[0]))

    def test_inter_group_fusion_retains_both_bridges_between_open_paths(self) -> None:
        path_a = WirePath(v=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float))
        path_b = WirePath(v=np.array([[2.0, 1.0, 0.0], [2.0, 0.0, 0.0]], dtype=float))

        fused, connector, closing_connector, used_fallback = _fuse_opened_wire_paths_with_connector_pair_diagnostics(
            path_a, path_b
        )

        self.assertFalse(used_fallback)
        np.testing.assert_allclose(connector[[0, -1]], np.vstack((path_a.v[-1], path_b.v[0])))
        np.testing.assert_allclose(closing_connector[[0, -1]], np.vstack((path_b.v[-1], path_a.v[0])))
        self.assertTrue(np.allclose(fused.v[0], fused.v[-1]))
        for bridge in (connector, closing_connector):
            self.assertTrue(all(np.any(np.all(np.isclose(fused.v, point, atol=1e-12), axis=1)) for point in bridge))

    def test_strict_surface_geodesic_connector_never_falls_back_to_direct_chord(self) -> None:
        mesh = Mesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [11.0, 0.0, 0.0],
                    [10.0, 1.0, 0.0],
                ],
                dtype=float,
            ),
            faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
        )
        solver = SurfaceGeodesicSolver(mesh)
        path_a = WirePath(v=np.array([[0.0, 0.0, 0.0], [0.2, 0.2, 0.0]], dtype=float))
        path_b = WirePath(v=np.array([[10.2, 0.2, 0.0], [10.0, 0.0, 0.0]], dtype=float))

        relaxed, relaxed_connector = _fuse_opened_wire_paths_with_connector(path_a, path_b, solver)
        strict, strict_connector = _fuse_opened_wire_paths_with_connector(
            path_a,
            path_b,
            solver,
            allow_direct_fallback=False,
        )

        self.assertEqual(len(relaxed_connector), 2)
        self.assertGreater(len(relaxed.v), 0)
        self.assertEqual(len(strict_connector), 0)
        self.assertEqual(len(strict.v), 0)

    def test_surface_geodesic_or_direct_is_relaxed_geodesic_mode(self) -> None:
        relaxed = CoilGenConfig(inter_group_connection_path_method="surface_geodesic_or_direct")
        strict = CoilGenConfig(inter_group_connection_path_method="surface_geodesic")

        self.assertTrue(_use_geodesic_inter_group_paths(relaxed))
        self.assertFalse(_strict_geodesic_inter_group_paths(relaxed))
        self.assertTrue(_use_geodesic_inter_group_paths(strict))
        self.assertTrue(_strict_geodesic_inter_group_paths(strict))

    def test_minimal_contour_distance_uses_finite_surface_geodesics(self) -> None:
        part = _radial_contour_part(levels=(0.16, 0.49))

        part = find_minimal_contour_distance(
            [part],
            CoilGenConfig(skip_calculation_min_winding_distance=False),
        )[0]

        self.assertEqual(part.contour_distance_matrix.shape, (2, 2))
        self.assertTrue(np.isfinite(part.contour_distance_matrix[0, 1]))
        self.assertGreater(part.minimal_contour_distance, 0.0)
        self.assertAlmostEqual(part.pcb_track_width, part.minimal_contour_distance)

    def test_topology_preflight_rejects_missing_contour_provenance(self) -> None:
        mesh = Mesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
        )
        part = CoilPart(mesh=mesh, contour_lines=[_square_loop_at(0.0, 0.0, 0.5, 0.0)])

        report = validate_intrinsic_topology_inputs(part)

        self.assertFalse(report.ok)
        self.assertTrue(any("provenance" in error for error in report.errors))
        with self.assertRaisesRegex(ValueError, "preflight"):
            topological_loop_grouping([part])

    def test_topology_preflight_rejects_non_orientable_mesh(self) -> None:
        part = CoilPart(mesh=_mobius_strip_mesh())

        report = validate_intrinsic_topology_inputs(part)

        self.assertFalse(report.ok)
        self.assertTrue(any("orientable" in error for error in report.errors))

    def test_extracted_cylinder_contour_passes_topology_preflight(self) -> None:
        mesh = normalize_mesh(build_cylinder_mesh(0.4, 0.2, 18, 8, 1, 0, 0, 0))
        part = CoilPart(mesh=mesh)
        z_values = mesh.vertices[:, 2]
        part.stream_function = z_values
        part.potential_level_list = np.array([float(np.min(z_values) + 0.37 * np.ptp(z_values))])
        part.contour_step = 1.0
        part = extract_contours([part], CoilGenConfig(min_point_loop_number=8))[0]

        report = validate_intrinsic_topology_inputs(part, check_separating=True)

        self.assertGreaterEqual(len(part.contour_lines), 1)
        self.assertTrue(report.ok, report.errors)

    def test_auto_grouping_merges_boundary_associated_parallel_loops(self) -> None:
        part = _cylinder_z_ring_part(level_count=3)

        grouped = topological_loop_grouping(
            [part],
            config=CoilGenConfig(topological_grouping_method="auto"),
        )[0]

        self.assertEqual(grouped.topology_diagnostics["topological_grouping_method"], "parallel_surface")
        self.assertEqual(grouped.topology_diagnostics["boundary_associated_loop_count"], 3)
        self.assertEqual(len(grouped.groups), 1)
        self.assertEqual(len(grouped.groups[0].loops), 3)
        self.assertTrue(
            all(row["boundary_associated"] for row in grouped.topology_diagnostics["loop_topology_signatures"])
        )

    def test_parallel_surface_grouping_splits_boundary_loops_by_orientation(self) -> None:
        part = _cylinder_z_ring_part(level_count=3)
        part.contour_lines[1] = reverse_contour_line(part.contour_lines[1])

        grouped = topological_loop_grouping(
            [part],
            config=CoilGenConfig(topological_grouping_method="parallel_surface"),
        )[0]

        self.assertEqual(grouped.topology_diagnostics["topological_grouping_method"], "parallel_surface")
        self.assertEqual(sorted(len(group.loops) for group in grouped.groups), [1, 2])

    def test_parallel_surface_grouping_repairs_small_axial_side_outliers(self) -> None:
        part = _cylinder_z_ring_part(level_count=10)
        z_centers = [float(np.mean(contour.v[:, 2])) for contour in part.contour_lines]
        negative = [loop_ind for loop_ind, z_center in enumerate(z_centers) if z_center < 0.0]
        positive = [loop_ind for loop_ind, z_center in enumerate(z_centers) if z_center > 0.0]
        if len(negative) < 4 or len(positive) < 4:
            self.fail("Expected enough negative and positive cylinder rings for axial outlier repair test.")
        swapped_groups = [
            [*negative[:-1], positive[0]],
            [negative[-1], *positive[1:]],
        ]

        repaired, repairs = _repair_surface_parallel_axial_outliers(
            part,
            swapped_groups,
            set(negative + positive),
        )

        self.assertTrue(repairs)
        self.assertEqual(sorted(repaired), sorted([negative, positive]))

    def test_parallel_surface_grouping_routes_boundary_associated_loops(self) -> None:
        part = _cylinder_z_ring_part(level_count=3)
        config = CoilGenConfig(
            topological_grouping_method="parallel_surface",
            interconnection_cut_width=0.03,
            normal_shift_length=0.01,
        )

        parts = topological_loop_grouping([part], config=config)
        parts = calculate_group_centers(parts)
        parts = interconnect_within_groups(parts, config)
        parts = interconnect_among_groups(parts, config)
        parts = shift_return_paths(parts, config)
        parts = cleanup_wire_paths(parts)

        routed = parts[0]
        self.assertEqual(len(routed.groups), 1)
        self.assertEqual(len(routed.connected_groups), 1)
        self.assertIsNotNone(routed.wire_path)
        self.assertEqual(routed.wire_path.v.shape[1], 3)
        self.assertTrue(routed.route_diagnostics["closed_v"])

    def test_contour_level_regularization_avoids_mesh_vertices(self) -> None:
        mesh = _grid_mesh(3, 3)
        stream_function = mesh.vertices[:, 0] + 0.5 * mesh.vertices[:, 1]
        exact_vertex_level = float(stream_function[4])

        adjusted = regularize_contour_levels(mesh, stream_function, np.array([exact_vertex_level]))

        self.assertNotEqual(float(adjusted[0]), exact_vertex_level)
        self.assertGreater(float(adjusted[0]), float(np.min(stream_function)))
        self.assertLess(float(adjusted[0]), float(np.max(stream_function)))

    def test_fixed_step_phase_grid_shifts_levels_without_changing_contour_current(self) -> None:
        mesh = _grid_mesh(5, 4)
        stream_function = mesh.vertices[:, 0] + 0.2 * mesh.vertices[:, 1]
        part_a = CoilPart(mesh=mesh.copy(), stream_function=stream_function.copy())
        part_b = CoilPart(mesh=mesh.copy(), stream_function=stream_function.copy())
        config_a = CoilGenConfig(
            levels=8,
            pot_offset_factor=0.25,
            level_set_method="primary",
            zero_crossing_level_strategy="full_range",
            contour_level_grid_mode="fixed_step_phase",
            min_turn_distance=0.0,
        )
        config_b = CoilGenConfig(
            levels=8,
            pot_offset_factor=0.75,
            level_set_method="primary",
            zero_crossing_level_strategy="full_range",
            contour_level_grid_mode="fixed_step_phase",
            min_turn_distance=0.0,
        )

        parts_a, _ = calc_potential_levels([part_a], part_a.mesh, config_a)
        parts_b, _ = calc_potential_levels([part_b], part_b.mesh, config_b)

        expected_step = float(np.ptp(stream_function) / 8.0)
        self.assertAlmostEqual(parts_a[0].contour_step, expected_step)
        self.assertAlmostEqual(parts_b[0].contour_step, expected_step)
        np.testing.assert_allclose(
            parts_b[0].potential_level_list - parts_a[0].potential_level_list,
            0.5 * expected_step,
        )

    def test_surface_contour_cleanup_removes_degenerate_segment(self) -> None:
        mesh = normalize_mesh(
            Mesh(
                vertices=np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.5, 0.5, 0.0],
                    ],
                    dtype=float,
                ),
                faces=np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=np.int64),
            )
        )
        points = np.array(
            [
                [0.5, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 0.5, 0.0],
                [0.5, 1.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.5, 0.0, 0.0],
            ],
            dtype=float,
        )
        segment_faces = np.array([0, 1, 1, 2, 3], dtype=np.int64)
        segment_barycentric = np.array(
            [
                [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]],
                [[0.0, 0.5, 0.5], [0.0, 0.5, 0.5]],
                [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5]],
                [[0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
                [[0.0, 0.5, 0.5], [0.5, 0.5, 0.0]],
            ],
            dtype=float,
        )
        contour = ContourLine(
            v=points,
            potential=0.0,
            surface_faces=np.array([0, 1, 1, 2, 3, 0], dtype=np.int64),
            surface_barycentric=np.array(
                [
                    [0.5, 0.5, 0.0],
                    [0.0, 0.5, 0.5],
                    [0.0, 0.5, 0.5],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                    [0.5, 0.5, 0.0],
                ],
                dtype=float,
            ),
            segment_faces=segment_faces,
            segment_barycentric=segment_barycentric,
        )
        part = CoilPart(mesh=mesh, contour_lines=[contour])

        cleaned = clean_contour_line(contour, CoilGenConfig(min_point_loop_number=4))
        self.assertIsNotNone(cleaned)
        part.contour_lines = [cleaned]
        report = validate_intrinsic_topology_inputs(part)

        self.assertFalse(np.any(np.linalg.norm(np.diff(cleaned.v, axis=0), axis=1) <= 1e-12))
        self.assertTrue(report.ok, report.errors)

    def test_3d_self_contact_without_return_path_is_diagnostic_only(self) -> None:
        vertices = np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        mesh = normalize_mesh(Mesh(vertices=vertices, faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)))
        wire = np.array(
            [
                [-0.8, -0.8, 0.0],
                [0.8, 0.8, 0.0],
                [-0.8, 0.8, 0.0],
                [0.8, -0.8, 0.0],
            ],
            dtype=float,
        )
        part = CoilPart(mesh=mesh)
        part.wire_path = WirePath(v=wire)

        contacts = find_wire_segment_contacts_3d(wire, threshold=1e-8)
        min_seg_a, min_seg_b, min_distance = find_min_wire_segment_distance_3d(wire)
        shifted = shift_return_paths(
            [part],
            CoilGenConfig(normal_shift_length=0.01, conductor_cross_section_width=0.001),
        )[0]
        shifted = cleanup_wire_paths([shifted])[0]

        self.assertEqual([(a, b) for a, b, _ in contacts], [(0, 2)])
        self.assertEqual((min_seg_a, min_seg_b), (0, 2))
        self.assertAlmostEqual(min_distance, 0.0)
        self.assertEqual(len(shifted.wire_self_contacts), 1)
        self.assertGreater(shifted.route_diagnostics["self_contact_count"], 0)
        self.assertEqual(shifted.route_diagnostics["shifted_point_count"], 0)
        self.assertAlmostEqual(float(np.max(np.abs(shifted.wire_path.v[:, 2]))), 0.0)

    def test_3d_self_contact_shift_lifts_return_segment_only(self) -> None:
        vertices = np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        mesh = normalize_mesh(Mesh(vertices=vertices, faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)))
        wire = np.array(
            [
                [-0.8, -0.8, 0.0],
                [0.8, 0.8, 0.0],
                [-0.8, 0.8, 0.0],
                [0.8, -0.8, 0.0],
            ],
            dtype=float,
        )
        part = CoilPart(mesh=mesh)
        part.wire_path = WirePath(v=wire)
        part.connected_groups = [
            ConnectedGroup(
                v=wire,
                return_path_v=wire[2:3],
                spiral_in_v=wire,
                spiral_out_v=wire,
            )
        ]

        shifted = shift_return_paths(
            [part],
            CoilGenConfig(normal_shift_length=0.01, conductor_cross_section_width=0.001),
        )[0]
        shifted = cleanup_wire_paths([shifted])[0]

        self.assertEqual(len(shifted.wire_self_contacts), 1)
        self.assertGreater(shifted.route_diagnostics["shifted_point_count"], 0)
        self.assertGreater(float(np.max(np.abs(shifted.wire_path.v[:, 2]))), 0.0)

    def test_crossing_inter_group_bridges_raise_only_one_local_overpass(self) -> None:
        mesh = Mesh(
            vertices=np.array([[-4.0, -4.0, 0.0], [4.0, -4.0, 0.0], [4.0, 4.0, 0.0], [-4.0, 4.0, 0.0]], dtype=float),
            faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        )
        connection = np.array([[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0]], dtype=float)
        closing_connection = np.array([[-1.0, 1.0, 0.0], [1.0, -1.0, 0.0]], dtype=float)
        part = CoilPart(mesh=mesh)
        part.wire_path = WirePath(
            v=np.array(
                [
                    [-1.0, -1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [3.0, 1.0, 0.0],
                    [3.0, 3.0, 0.0],
                    [-1.0, 3.0, 0.0],
                    [-1.0, 1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [3.0, -1.0, 0.0],
                    [3.0, -3.0, 0.0],
                    [-1.0, -3.0, 0.0],
                    [-1.0, -1.0, 0.0],
                ],
                dtype=float,
            )
        )
        part.opening_cuts_among_groups = [
            InterGroupCut(
                cut1_v=np.zeros((0, 3), dtype=float),
                cut2_v=np.zeros((0, 3), dtype=float),
                connection_path_v=connection,
                closing_connection_path_v=closing_connection,
                is_return_connection=False,
            )
        ]

        shifted = shift_return_paths(
            [part],
            CoilGenConfig(
                normal_shift_length=0.01,
                normal_shift_smooth_factors=(1.0, 1.0),
                min_turn_distance=0.001,
            ),
        )[0]

        primary_z = shifted.opening_cuts_among_groups[0].connection_path_v[:, 2]
        closing_z = shifted.opening_cuts_among_groups[0].closing_connection_path_v[:, 2]
        np.testing.assert_allclose(primary_z, 0.0, atol=1e-12)
        self.assertGreater(float(np.min(closing_z)), 0.0)
        self.assertLessEqual(float(np.max(shifted.shift_array)), 1.0)

    def test_crossing_connectors_from_different_cuts_raise_only_one_overpass(self) -> None:
        mesh = Mesh(
            vertices=np.array(
                [[-4.0, -4.0, 0.0], [4.0, -4.0, 0.0], [4.0, 4.0, 0.0], [-4.0, 4.0, 0.0]],
                dtype=float,
            ),
            faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        )
        first = np.array([[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0]], dtype=float)
        second = np.array([[-1.0, 1.0, 0.0], [1.0, -1.0, 0.0]], dtype=float)
        part = CoilPart(mesh=mesh)
        part.wire_path = WirePath(
            v=np.array(
                [
                    [-1.0, -1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [3.0, 1.0, 0.0],
                    [3.0, 3.0, 0.0],
                    [-1.0, 3.0, 0.0],
                    [-1.0, 1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [3.0, -1.0, 0.0],
                    [3.0, -3.0, 0.0],
                    [-1.0, -3.0, 0.0],
                    [-1.0, -1.0, 0.0],
                ],
                dtype=float,
            )
        )
        part.opening_cuts_among_groups = [
            InterGroupCut(np.zeros((0, 3)), np.zeros((0, 3)), connection_path_v=first),
            InterGroupCut(np.zeros((0, 3)), np.zeros((0, 3)), connection_path_v=second),
        ]

        shifted = shift_return_paths(
            [part],
            CoilGenConfig(
                normal_shift_length=0.01,
                normal_shift_smooth_factors=(1.0, 1.0),
                min_turn_distance=0.001,
            ),
        )[0]

        first_z = shifted.opening_cuts_among_groups[0].connection_path_v[:, 2]
        second_z = shifted.opening_cuts_among_groups[1].connection_path_v[:, 2]
        np.testing.assert_allclose(first_z, 0.0, atol=1e-12)
        self.assertGreater(float(np.min(second_z)), 0.0)
        self.assertLessEqual(float(np.max(shifted.shift_array)), 1.0)

    def test_within_group_crossing_raises_long_return_not_contour_connection(self) -> None:
        mesh = Mesh(
            vertices=np.array(
                [[-4.0, -4.0, 0.0], [4.0, -4.0, 0.0], [4.0, 4.0, 0.0], [-4.0, 4.0, 0.0]],
                dtype=float,
            ),
            faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        )
        contour_connection = np.array([[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0]], dtype=float)
        long_return = np.array([[-1.0, 1.0, 0.0], [1.0, -1.0, 0.0]], dtype=float)
        wire = np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [3.0, 1.0, 0.0],
                [3.0, 3.0, 0.0],
                [-1.0, 3.0, 0.0],
                [-1.0, 1.0, 0.0],
                [1.0, -1.0, 0.0],
                [3.0, -1.0, 0.0],
                [3.0, -3.0, 0.0],
                [-1.0, -3.0, 0.0],
                [-1.0, -1.0, 0.0],
            ],
            dtype=float,
        )
        part = CoilPart(mesh=mesh)
        part.wire_path = WirePath(v=wire)
        part.connected_groups = [
            ConnectedGroup(
                v=wire,
                return_path_v=long_return,
                spiral_in_v=contour_connection,
                spiral_out_v=contour_connection,
            )
        ]

        shifted = shift_return_paths(
            [part],
            CoilGenConfig(
                normal_shift_length=0.01,
                normal_shift_smooth_factors=(1.0, 1.0),
                min_turn_distance=0.001,
            ),
        )[0]

        np.testing.assert_allclose(shifted.wire_path.v[[0, 1], 2], 0.0, atol=1e-12)
        np.testing.assert_allclose(shifted.wire_path.v[[5, 6], 2], 0.01, atol=1e-12)
        self.assertLessEqual(float(np.max(shifted.shift_array)), 1.0)

    def test_long_return_has_overpass_precedence_over_inter_group_connector(self) -> None:
        mesh = Mesh(
            vertices=np.array(
                [[-4.0, -4.0, 0.0], [4.0, -4.0, 0.0], [4.0, 4.0, 0.0], [-4.0, 4.0, 0.0]],
                dtype=float,
            ),
            faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        )
        inter_group_connector = np.array([[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0]], dtype=float)
        long_return = np.array([[-1.0, 1.0, 0.0], [1.0, -1.0, 0.0]], dtype=float)
        wire = np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [3.0, 1.0, 0.0],
                [3.0, 3.0, 0.0],
                [-1.0, 3.0, 0.0],
                [-1.0, 1.0, 0.0],
                [1.0, -1.0, 0.0],
                [3.0, -1.0, 0.0],
                [3.0, -3.0, 0.0],
                [-1.0, -3.0, 0.0],
                [-1.0, -1.0, 0.0],
            ],
            dtype=float,
        )
        part = CoilPart(mesh=mesh)
        part.wire_path = WirePath(v=wire)
        part.connected_groups = [
            ConnectedGroup(
                v=wire,
                return_path_v=long_return,
                spiral_in_v=wire,
                spiral_out_v=wire,
            )
        ]
        part.opening_cuts_among_groups = [
            InterGroupCut(
                np.zeros((0, 3), dtype=float),
                np.zeros((0, 3), dtype=float),
                connection_path_v=inter_group_connector,
            )
        ]

        shifted = shift_return_paths(
            [part],
            CoilGenConfig(normal_shift_length=0.01, min_turn_distance=0.001),
        )[0]

        np.testing.assert_allclose(shifted.opening_cuts_among_groups[0].connection_path_v[:, 2], 0.0, atol=1e-12)
        np.testing.assert_allclose(shifted.wire_path.v[[5, 6], 2], 0.01, atol=1e-12)

    def test_inter_group_connector_without_crossing_remains_on_support(self) -> None:
        mesh = Mesh(
            vertices=np.array(
                [[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [2.0, 2.0, 0.0], [-2.0, 2.0, 0.0]],
                dtype=float,
            ),
            faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        )
        wire = np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [-1.0, -1.0, 0.0],
            ],
            dtype=float,
        )
        part = CoilPart(mesh=mesh)
        part.wire_path = WirePath(v=wire)
        part.opening_cuts_among_groups = [
            InterGroupCut(
                np.zeros((0, 3), dtype=float),
                np.zeros((0, 3), dtype=float),
                connection_path_v=wire[:2],
            )
        ]

        shifted = shift_return_paths(
            [part],
            CoilGenConfig(normal_shift_length=0.01, min_turn_distance=0.001),
        )[0]

        np.testing.assert_allclose(shifted.wire_path.v[:, 2], 0.0, atol=1e-12)
        self.assertFalse(np.any(shifted.points_to_shift))

    def test_crossing_route_identity_extends_through_connector_approach(self) -> None:
        route_a = _NormalShiftRoute(
            key=("inter_group_connector", 0, 0),
            priority=1,
            order=0,
            segment_indices=(0,),
        )
        route_b = _NormalShiftRoute(
            key=("inter_group_connector", 1, 0),
            priority=1,
            order=1,
            segment_indices=(3,),
        )
        routes = [[route_a], [], [], [route_b], [], []]
        wire = np.column_stack(
            (
                np.arange(7, dtype=float) * 0.001,
                np.zeros(7, dtype=float),
                np.zeros(7, dtype=float),
            )
        )

        extended = _extend_normal_shift_routes_by_arclength(routes, wire, maximum_distance=0.002)

        self.assertEqual(extended[4][0].key, route_b.key)
        self.assertEqual(_select_crossing_overpass(0, 3, extended), (3, 0))
        self.assertEqual(_select_crossing_overpass(0, 4, extended), (4, 0))

    def test_contact_cluster_uses_one_winner_for_core_and_approach_pairs(self) -> None:
        route_a = _NormalShiftRoute(
            key=("inter_group_connector", 0, 0),
            priority=1,
            order=0,
            segment_indices=(0,),
        )
        route_b = _NormalShiftRoute(
            key=("inter_group_connector", 1, 0),
            priority=1,
            order=1,
            segment_indices=(6,),
        )
        routes = [[route_a], [], [], [], [], [], [route_b], [], [], []]
        wire = np.column_stack(
            (
                np.arange(11, dtype=float) * 0.001,
                np.zeros(11, dtype=float),
                np.zeros(11, dtype=float),
            )
        )
        routes = _extend_normal_shift_routes_by_arclength(routes, wire, maximum_distance=0.002)
        contacts = [(0, 6, 0.0), (0, 7, 0.001), (1, 6, 0.001)]

        selections = _crossing_event_segment_selections(
            contacts,
            routes,
            wire,
            cluster_arclength=0.003,
        )

        self.assertEqual(selections, [((6, 7), (0, 1))])

    def test_underpass_window_is_smooth_not_single_point_inward_notch(self) -> None:
        wire = np.column_stack(
            (
                np.arange(13, dtype=float) * 0.001,
                np.zeros(13, dtype=float),
                np.zeros(13, dtype=float),
            )
        )
        base_levels = np.ones(len(wire), dtype=float)
        overpass = np.zeros(len(wire), dtype=bool)
        underpass = np.zeros(len(wire), dtype=bool)
        underpass[6] = True

        profile = _crossing_shift_profile(
            base_levels,
            overpass,
            underpass,
            wire,
            transition_length=0.005,
            smooth_factors=(1.0, 1.0),
        )

        self.assertAlmostEqual(float(profile[6]), 0.0)
        np.testing.assert_allclose(profile, profile[::-1], atol=1e-12)
        self.assertTrue(np.all(np.diff(profile[:7]) <= 1e-12))
        self.assertTrue(np.all(np.diff(profile[6:]) >= -1e-12))
        self.assertLess(float(np.max(np.abs(np.diff(profile)))), 0.35)

    def test_cleanup_wire_paths_records_3d_route_diagnostics(self) -> None:
        points = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            dtype=float,
        )
        part = CoilPart(mesh=Mesh(vertices=points[:3], faces=np.array([[0, 1, 2]], dtype=np.int64)))
        part.wire_path = WirePath(v=points)

        part = cleanup_wire_paths([part])[0]

        self.assertEqual(len(part.wire_path.v), 4)
        self.assertEqual(part.route_diagnostics["point_count"], 4)
        self.assertEqual(part.route_diagnostics["segment_count"], 3)
        self.assertTrue(part.route_diagnostics["closed_v"] is True)
        self.assertEqual(part.route_diagnostics["inter_group_cut_count"], 0)
        self.assertEqual(part.route_diagnostics["inter_group_connector_point_count"], 0)
        self.assertEqual(part.route_diagnostics["min_wire_wire_segment_a"], -1)
        self.assertEqual(part.route_diagnostics["min_wire_wire_segment_b"], -1)
        self.assertEqual(part.route_diagnostics["min_wire_wire_distance"], 0.0)
        self.assertEqual(part.route_diagnostics["opening_cut_count"], 0)
        self.assertEqual(part.route_diagnostics["degenerate_opening_cut_count"], 0)

    def test_cleanup_wire_paths_removes_exact_retraced_spur(self) -> None:
        mesh = Mesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
        )
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        part = CoilPart(mesh=mesh)
        part.wire_path = WirePath(v=np.vstack((a, b, a, b, c, a)))

        cleaned = cleanup_wire_paths([part])[0]

        self.assertTrue(np.allclose(cleaned.wire_path.v, np.vstack((a, b, c, a))))
        self.assertEqual(cleaned.route_diagnostics["wire_clearance_failure_count"], 0)

    def test_shifted_wire_repair_collapses_short_offset_reversal_spur(self) -> None:
        points = np.array(
            [
                [0.0000, 0.0000, 0.005],
                [0.0030, 0.0000, 0.005],
                [0.0005, 0.0002, 0.005],
                [0.0040, 0.0000, 0.005],
                [0.0120, 0.0040, 0.005],
                [0.0200, 0.0100, 0.005],
            ],
            dtype=float,
        )
        shifts = np.ones(len(points), dtype=float)

        repaired, repaired_shifts = _repair_shifted_wire_reversal_spurs(
            points,
            shifts,
            CoilGenConfig(
                interconnection_cut_width=0.005,
                within_group_cut_width=0.005,
                inter_group_cut_width=0.005,
                sweep_min_bend_radius=0.003,
            ),
        )

        self.assertEqual(repaired.shape, (4, 3))
        self.assertTrue(np.allclose(repaired, points[[0, 3, 4, 5]]))
        self.assertTrue(np.allclose(repaired_shifts, 1.0))

    def test_cleanup_wire_paths_uses_configurable_route_thresholds(self) -> None:
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
        part = CoilPart(
            mesh=Mesh(
                vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                faces=np.array([[0, 1, 2]], dtype=np.int64),
            )
        )
        part.wire_path = WirePath(v=points)
        part.opening_cuts_among_groups = [
            InterGroupCut(
                cut1_v=np.array([[0.0, 0.0, 0.0], [0.0, 0.01, 0.0]], dtype=float),
                cut2_v=np.array([[1.0, 0.0, 0.0], [1.0, 0.01, 0.0]], dtype=float),
                connection_path_v=np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float),
                direct_distance=1.0,
                connector_distance=3.0,
                connection_method="direct_3d",
                used_clearance_fallback=True,
            )
        ]

        part = cleanup_wire_paths([part], config=CoilGenConfig(suspicious_inter_group_connector_ratio=1.5))[0]

        self.assertEqual(part.route_diagnostics["suspicious_inter_group_connector_count"], 1)
        self.assertEqual(part.route_diagnostics["inter_group_clearance_fallback_count"], 1)
        self.assertTrue(part.route_diagnostics["route_quality_ok"])
        self.assertEqual(part.route_diagnostics["route_quality_status"], "warn")

    def test_route_diagnostics_include_closing_inter_group_bridge_clearance(self) -> None:
        mesh = Mesh(
            vertices=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=float),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
        )
        part = CoilPart(
            mesh=mesh,
            contour_lines=[ContourLine(v=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float), potential=0.0)],
        )
        part.wire_path = WirePath(v=np.array([[0.0, 1.0, 0.0], [2.0, 1.0, 0.0]], dtype=float))
        part.opening_cuts_among_groups = [
            InterGroupCut(
                cut1_v=np.array([[0.0, 0.0, 0.0], [0.0, 0.01, 0.0]], dtype=float),
                cut2_v=np.array([[2.0, 0.0, 0.0], [2.0, 0.01, 0.0]], dtype=float),
                connection_path_v=np.array([[0.0, 1.0, 0.0], [2.0, 1.0, 0.0]], dtype=float),
                closing_connection_path_v=np.array([[1.0, -1.0, 0.0], [1.0, 1.0, 0.0]], dtype=float),
            )
        ]

        part = cleanup_wire_paths([part], config=CoilGenConfig(min_inter_group_connector_clearance=0.05))[0]

        self.assertEqual(part.route_diagnostics["inter_group_connector_path_count"], 2)
        self.assertEqual(part.route_diagnostics["inter_group_connector_clearance_violation_count"], 1)
        self.assertFalse(part.route_diagnostics["route_quality_ok"])

    def test_cleanup_wire_paths_marks_min_wire_distance_as_route_failure(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.001, 0.0],
                [0.0, 0.001, 0.0],
            ],
            dtype=float,
        )
        part = CoilPart(mesh=Mesh(vertices=points[:3], faces=np.array([[0, 1, 2]], dtype=np.int64)))
        part.wire_path = WirePath(v=points)

        part = cleanup_wire_paths([part], config=CoilGenConfig(min_wire_distance_m=0.005))[0]

        self.assertEqual(part.route_diagnostics["wire_clearance_failure_count"], 1)
        self.assertAlmostEqual(part.route_diagnostics["wire_clearance_failure_threshold"], 0.005)
        self.assertLess(part.route_diagnostics["min_wire_wire_distance"], 0.005)
        self.assertFalse(part.route_diagnostics["route_quality_ok"])
        self.assertEqual(part.route_diagnostics["route_quality_status"], "fail")
        self.assertEqual(part.route_diagnostics["route_quality_failure_reasons"], "wire_clearance_target")
        self.assertTrue(part.route_diagnostics["route_topology_ok"])
        self.assertEqual(part.route_diagnostics["route_topology_failure_count"], 0)

    def test_inter_group_selection_skips_unclear_nearest_candidate(self) -> None:
        part, groups = _clearance_selection_part()

        connection = _select_group_connection(
            part,
            groups,
            [0, 1],
            np.zeros((0, 3), dtype=float),
            CoilGenConfig(
                inter_group_connection_candidate_count=2,
                min_inter_group_connector_clearance=0.05,
                suspicious_inter_group_connector_ratio=10.0,
            ),
        )

        self.assertIsNotNone(connection)
        self.assertFalse(connection.used_clearance_fallback)
        self.assertAlmostEqual(connection.point_a_v[1], 1.0)
        self.assertAlmostEqual(connection.point_b_v[1], 1.0)

    def test_inter_group_selection_samples_outer_contour_before_clearance_fallback(self) -> None:
        part, groups = _clearance_selection_part()

        connection = _select_group_connection(
            part,
            groups,
            [0, 1],
            np.zeros((0, 3), dtype=float),
            CoilGenConfig(
                inter_group_connection_candidate_count=1,
                min_inter_group_connector_clearance=0.05,
                suspicious_inter_group_connector_ratio=10.0,
            ),
        )

        self.assertIsNotNone(connection)
        self.assertFalse(connection.used_clearance_fallback)
        self.assertAlmostEqual(connection.point_a_v[1], 1.0)
        self.assertAlmostEqual(connection.point_b_v[1], 1.0)

    def test_inter_group_selection_never_uses_closer_inner_contours(self) -> None:
        part, groups = _outer_only_selection_part()

        search_a = _outer_group_connection_search_path(
            part,
            groups[0],
            0,
            np.zeros((0, 3), dtype=float),
            boundary_threshold=2,
        )
        search_b = _outer_group_connection_search_path(
            part,
            groups[1],
            1,
            np.zeros((0, 3), dtype=float),
            boundary_threshold=2,
        )
        self.assertTrue(np.all(search_a.loop_indices == 0))
        self.assertTrue(np.all(search_b.loop_indices == 2))

        connection = _select_group_connection(
            part,
            groups,
            [0, 1],
            np.zeros((0, 3), dtype=float),
            CoilGenConfig(
                inter_group_connection_candidate_count=4,
                interconnection_cut_width=0.05,
                suspicious_inter_group_connector_ratio=100.0,
            ),
        )

        self.assertIsNotNone(connection)
        self.assertEqual(connection.loop_a_index, 0)
        self.assertEqual(connection.loop_b_index, 2)

    def test_cylindrical_inter_group_selection_aligns_tied_nearest_cuts(self) -> None:
        part, groups = _cylindrical_two_ring_connection_part()

        connection = _select_group_connection(
            part,
            groups,
            [0, 1],
            np.zeros((0, 3), dtype=float),
            CoilGenConfig(
                surface_is_cylinder_flag=True,
                inter_group_connection_candidate_count=1,
                inter_group_cut_alignment="auto",
                inter_group_cut_alignment_distance_rel_tol=1.0e-9,
                inter_group_cut_alignment_candidate_factor=1,
                interconnection_cut_width=0.04,
                suspicious_inter_group_connector_ratio=10.0,
            ),
        )

        self.assertIsNotNone(connection)
        midpoint = (connection.point_a_v + connection.point_b_v) / 2.0
        radial = midpoint[:2] / np.linalg.norm(midpoint[:2])
        self.assertGreater(radial[0], 0.98)
        self.assertLess(abs(radial[1]), 0.05)

    def test_cylindrical_inter_group_selection_uses_return_corridor_when_available(self) -> None:
        part, groups = _cylindrical_two_ring_connection_part()
        radius = 0.25
        groups[0].return_path_v = np.array([[0.0, radius, -0.08], [0.0, radius, -0.04]], dtype=float)
        groups[1].return_path_v = np.array([[0.0, -radius, 0.04], [0.0, -radius, 0.08]], dtype=float)
        part.connected_groups = groups

        connection = _select_group_connection(
            part,
            groups,
            [0, 1],
            np.zeros((0, 3), dtype=float),
            CoilGenConfig(
                surface_is_cylinder_flag=True,
                inter_group_connection_candidate_count=1,
                inter_group_cut_alignment="auto",
                inter_group_cut_alignment_distance_rel_tol=1.0e-9,
                inter_group_cut_alignment_candidate_factor=1,
                interconnection_cut_width=0.04,
                suspicious_inter_group_connector_ratio=10.0,
            ),
        )

        self.assertIsNotNone(connection)
        midpoint = (connection.point_a_v + connection.point_b_v) / 2.0
        radial = midpoint[:2] / np.linalg.norm(midpoint[:2])
        self.assertGreater(radial[1], 0.98)
        self.assertLess(abs(radial[0]), 0.05)

    def test_combined_return_group_fusion_uses_clean_return_jump(self) -> None:
        mesh = Mesh(
            vertices=np.array(
                [
                    [-0.1, -0.1, 0.0],
                    [2.3, -0.1, 0.0],
                    [-0.1, 0.3, 0.0],
                    [2.3, 0.3, 0.0],
                ],
                dtype=float,
            ),
            faces=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64),
        )
        part = CoilPart(mesh=mesh)
        group_a_spiral = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
        group_a_return = np.array([[1.0, 0.2, 0.0], [0.0, 0.2, 0.0]], dtype=float)
        group_b_spiral = np.array([[1.2, 0.0, 0.0], [2.2, 0.0, 0.0]], dtype=float)
        group_b_return = np.array([[2.2, 0.2, 0.0], [1.2, 0.2, 0.0]], dtype=float)
        contour_a = ContourLine(
            v=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.0]],
                dtype=float,
            ),
            potential=0.0,
        )
        contour_b = ContourLine(
            v=np.array(
                [[1.2, 0.0, 0.0], [2.2, 0.0, 0.0], [2.2, 0.2, 0.0], [1.2, 0.2, 0.0], [1.2, 0.0, 0.0]],
                dtype=float,
            ),
            potential=1.0,
        )
        part.contour_lines = [contour_a, contour_b]
        part.groups = [
            LoopGroup(loops=[contour_a], loop_indices=[0], outer_loop_indices=(0,)),
            LoopGroup(loops=[contour_b], loop_indices=[1], outer_loop_indices=(1,)),
        ]
        group_a = ConnectedGroup(
            v=np.vstack((group_a_spiral, group_a_return, group_a_spiral[:1])),
            return_path_v=group_a_return,
            spiral_in_v=group_a_spiral,
            spiral_out_v=group_a_spiral,
            outer_loop_indices=(0,),
        )
        group_b = ConnectedGroup(
            v=np.vstack((group_b_spiral, group_b_return, group_b_spiral[:1])),
            return_path_v=group_b_return,
            spiral_in_v=group_b_spiral,
            spiral_out_v=group_b_spiral,
            outer_loop_indices=(1,),
        )

        result = _combined_return_group_fusion_candidate(
            part,
            group_a,
            group_b,
            0,
            1,
            CoilGenConfig(interconnection_cut_width=0.02),
            None,
            allow_direct_fallback=True,
        )

        self.assertIsNotNone(result)
        self.assertTrue(result.combined_return)
        np.testing.assert_allclose(result.connection_path_v[0], group_b_return[-1])
        np.testing.assert_allclose(result.connection_path_v[-1], group_a_return[0])
        np.testing.assert_allclose(result.selected_point1_v, group_a_return[0])
        np.testing.assert_allclose(result.selected_point2_v, group_b_return[-1])
        self.assertLess(result.path_length, 5.0)

    def test_combined_return_group_fusion_selects_shortest_spiral_variant_pair(self) -> None:
        mesh = Mesh(
            vertices=np.array(
                [
                    [-0.1, -0.1, 0.0],
                    [4.2, -0.1, 0.0],
                    [-0.1, 1.2, 0.0],
                    [4.2, 1.2, 0.0],
                ],
                dtype=float,
            ),
            faces=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64),
        )
        part = CoilPart(mesh=mesh)
        contour_a = ContourLine(
            v=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [3.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [1.0, 0.2, 0.0],
                    [0.0, 0.2, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=float,
            ),
            potential=0.0,
        )
        contour_b = ContourLine(
            v=np.array(
                [
                    [3.9, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [1.1, 1.0, 0.0],
                    [1.0, 0.2, 0.0],
                    [1.1, 0.2, 0.0],
                    [3.9, 0.0, 0.0],
                ],
                dtype=float,
            ),
            potential=1.0,
        )
        part.contour_lines = [contour_a, contour_b]
        part.groups = [
            LoopGroup(loops=[contour_a], loop_indices=[0], outer_loop_indices=(0,)),
            LoopGroup(loops=[contour_b], loop_indices=[1], outer_loop_indices=(1,)),
        ]
        group_a = ConnectedGroup(
            v=np.zeros((0, 3), dtype=float),
            return_path_v=np.array([[1.0, 0.2, 0.0], [0.0, 0.2, 0.0]], dtype=float),
            spiral_in_v=np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float),
            spiral_out_v=np.array([[3.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=float),
            outer_loop_indices=(0,),
        )
        group_b = ConnectedGroup(
            v=np.zeros((0, 3), dtype=float),
            return_path_v=np.array([[1.1, 0.2, 0.0], [1.0, 0.2, 0.0]], dtype=float),
            spiral_in_v=np.array([[3.9, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float),
            spiral_out_v=np.array([[1.1, 1.0, 0.0], [1.0, 0.2, 0.0]], dtype=float),
            outer_loop_indices=(1,),
        )

        result = _combined_return_group_fusion_candidate(
            part,
            group_a,
            group_b,
            0,
            1,
            CoilGenConfig(interconnection_cut_width=0.02),
            None,
            allow_direct_fallback=True,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.group1_variant_name, "spiral_out")
        self.assertEqual(result.group2_variant_name, "spiral_out")
        self.assertLess(result.path_length, 9.0)

    def test_combined_return_group_fusion_uses_outer_loops_for_both_group_bridges(self) -> None:
        mesh = Mesh(
            vertices=np.array(
                [[-0.5, -0.2, 0.0], [1.5, -0.2, 0.0], [-0.5, 1.2, 0.0], [1.5, 1.2, 0.0]],
                dtype=float,
            ),
            faces=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64),
        )
        outer_a = ContourLine(v=np.array([[0.0, -0.1, 0.0], [0.0, 0.1, 0.0]]), potential=0.0)
        inner_a = ContourLine(v=np.array([[0.0, 0.9, 0.0], [0.0, 1.1, 0.0]]), potential=1.0)
        outer_b = ContourLine(v=np.array([[1.0, -0.1, 0.0], [1.0, 0.1, 0.0]]), potential=2.0)
        inner_b = ContourLine(v=np.array([[1.0, 0.9, 0.0], [1.0, 1.1, 0.0]]), potential=3.0)
        part = CoilPart(mesh=mesh, contour_lines=[outer_a, inner_a, outer_b, inner_b])
        part.groups = [
            LoopGroup(loops=[outer_a, inner_a], loop_indices=[0, 1], outer_loop_indices=(0,)),
            LoopGroup(loops=[outer_b, inner_b], loop_indices=[2, 3], outer_loop_indices=(2,)),
        ]
        group_a = ConnectedGroup(
            v=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
            return_path_v=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
            spiral_in_v=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            spiral_out_v=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
            outer_loop_indices=(0,),
        )
        group_b = ConnectedGroup(
            v=np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
            return_path_v=np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
            spiral_in_v=np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
            spiral_out_v=np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
            outer_loop_indices=(2,),
        )

        result = _combined_return_group_fusion_candidate(
            part,
            group_a,
            group_b,
            0,
            1,
            CoilGenConfig(interconnection_cut_width=0.02),
            None,
            allow_direct_fallback=True,
        )

        self.assertIsNotNone(result)
        self.assertEqual({result.group1_variant_name, result.group2_variant_name}, {"spiral_in", "spiral_out"})
        self.assertEqual(result.loop1_index, 0)
        self.assertEqual(result.loop2_index, 2)
        np.testing.assert_allclose(result.connection_path_v[:, 1], 0.0)
        np.testing.assert_allclose(result.closing_connection_path_v[:, 1], 0.0)

    def test_combined_return_must_shorten_complete_conductor(self) -> None:
        opened_cut_route = WirePath(v=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float))
        combined = GroupFusionResult(
            fused=WirePath(v=np.zeros((0, 3), dtype=float)),
            connection_path_v=np.zeros((0, 3), dtype=float),
            closing_connection_path_v=np.zeros((0, 3), dtype=float),
            cut1_v=np.zeros((0, 3), dtype=float),
            cut2_v=np.zeros((0, 3), dtype=float),
            selected_point1_v=np.zeros(3, dtype=float),
            selected_point2_v=np.zeros(3, dtype=float),
            direct_distance=0.1,
            geodesic_distance=0.1,
            connector_distance=0.1,
            used_direct_fallback=False,
            path_length=1.1,
        )

        self.assertFalse(_combined_return_is_shorter(combined, opened_cut_route))
        combined.path_length = 0.9
        self.assertTrue(_combined_return_is_shorter(combined, opened_cut_route))
        combined.path_length = 1.0
        self.assertFalse(_combined_return_is_shorter(combined, opened_cut_route))

    def test_cleanup_wire_paths_can_fail_on_route_quality_issues(self) -> None:
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
        part = CoilPart(
            mesh=Mesh(
                vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                faces=np.array([[0, 1, 2]], dtype=np.int64),
            )
        )
        part.wire_path = WirePath(v=points)
        part.opening_cuts_among_groups = [
            InterGroupCut(
                cut1_v=np.zeros((0, 3), dtype=float),
                cut2_v=np.zeros((0, 3), dtype=float),
                connection_path_v=np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float),
                direct_distance=1.0,
                connector_distance=3.0,
                connection_method="direct_3d",
            )
        ]

        with self.assertRaisesRegex(ValueError, "Route quality checks failed"):
            cleanup_wire_paths([part], config=CoilGenConfig(fail_on_route_quality_issues=True))

    def test_small_y_gradient_run(self) -> None:
        result = run_coilgen(_small_y_gradient_config(skip_sweep=True))

        self.assertEqual(len(result.coil_parts), 1)
        self.assertGreater(len(result.target_field.coords), 0)
        self.assertTrue(np.isfinite(result.potential_step))
        self.assertIsInstance(result.error_vals, dict)
        self.assertEqual(result.inductance_matrix.shape, (1, 1))
        self.assertEqual(result.timing_summary[0]["stage"], "start")
        self.assertEqual(result.timing_summary[-1]["stage"], "evaluated field errors")
        self.assertGreaterEqual(result.timing_summary[-1]["elapsed_seconds"], 0.0)

    def test_result_summary_and_performance_table_rows_include_scientific_metrics(self) -> None:
        result = run_coilgen(_small_y_gradient_config(skip_sweep=True))

        summary = result_summary_dict(result)
        self.assertIn("layout_gradient_efficiency_mT_per_m_per_A", summary)
        self.assertIn("coil_inductance_same_current_H", summary)
        self.assertIn("unconnected_contours_min_wire_distance_m", summary)
        self.assertIn("concomitant_rms_T", summary)
        self.assertIn("concomitant_max_abs_ppm_of_B0", summary)
        self.assertGreater(summary["layout_fit_sample_count"], 0)
        self.assertGreater(summary["concomitant_sample_count"], 0)

        table_rows = _performance_table_rows(result)
        quantities = {row[1] for row in table_rows}
        self.assertIn("connected-wire gradient efficiency", quantities)
        # The distance calculation is disabled in this lightweight fixture;
        # unavailable quantities are intentionally omitted from display rows.
        self.assertNotIn("minimum unconnected-contour wire clearance", quantities)
        self.assertEqual(summary["coil_electrical_calculation_status"], "skipped")
        self.assertTrue(np.isnan(summary["coil_inductance_H"]))
        self.assertNotIn("self inductance", quantities)
        status = next(row[2] for row in table_rows if row[1] == "resistance/inductance calculation")
        self.assertEqual(status, "skipped")

    def test_layout_vs_stream_error_uses_stream_current_convention(self) -> None:
        errors = _field_error_values(
            target_z=np.array([1.0, -1.0], dtype=float),
            sf_z=np.array([0.10, -0.10], dtype=float),
            layout_z=np.array([1.0, -1.0], dtype=float),
            loop_z=np.array([1.0, -1.0], dtype=float),
            layout_stream_z=np.array([0.11, -0.09], dtype=float),
            loop_stream_z=np.array([0.10, -0.10], dtype=float),
        )

        self.assertAlmostEqual(errors["max_rel_error_layout_vs_target"], 0.0)
        self.assertAlmostEqual(errors["max_rel_error_layout_vs_stream_function_field"], 10.0)
        self.assertAlmostEqual(errors["mean_rel_error_unconnected_contours_vs_stream_function_field"], 0.0)

    def test_result_plot_specs_define_reusable_plot_registry(self) -> None:
        result = run_coilgen(_small_y_gradient_config(skip_sweep=True))

        specs = build_result_plot_specs(
            result, include=("problem_setup_geometry", "wire", "performance_table", "regularization")
        )

        self.assertTrue(all(isinstance(spec, MatplotlibPlotSpec) for spec in specs))
        self.assertEqual(
            [spec.key for spec in specs],
            [
                "wire",
                "wire_sequence",
                "contour_current_orientation",
                "problem_setup_geometry",
                "performance_table",
                "regularization",
            ],
        )
        self.assertEqual(specs[0].category, "layout")
        self.assertEqual(specs[1].category, "routing")
        self.assertEqual(specs[2].category, "contours")
        self.assertEqual(specs[3].category, "layout")
        self.assertEqual(specs[4].category, "metrics")
        direction_specs = build_result_plot_specs(
            result,
            include=("wire_sequence", "contour_current_orientation"),
        )
        self.assertEqual(
            [spec.key for spec in direction_specs],
            ["wire_sequence", "contour_current_orientation"],
        )
        contour_anchor_keys = [spec.key for spec in build_result_plot_specs(result, include=("raw_contours",))]
        self.assertEqual(
            contour_anchor_keys,
            ["wire_sequence", "raw_contours", "contour_current_orientation"],
        )
        default_keys = [spec.key for spec in build_result_plot_specs(result)]
        self.assertIn("wire_sequence", default_keys)
        self.assertIn("contour_current_orientation", default_keys)

    def test_suppression_field_summary_reports_rms_and_peak_at_fitted_current(self) -> None:
        summary = _suppression_field_summary(
            "layout",
            np.array([[0.0, 0.0, 1e-6], [0.0, 0.0, -3e-6]], dtype=float),
            suppressed_mask=np.array([False, True]),
            current=2.0,
        )

        self.assertAlmostEqual(summary["layout_suppressed_rms_Bz_T_per_A"], 3e-6)
        self.assertAlmostEqual(summary["layout_suppressed_peak_abs_Bz_T_per_A"], 3e-6)
        self.assertAlmostEqual(summary["layout_suppressed_rms_Bmag_T_per_A"], 3e-6)
        self.assertAlmostEqual(summary["layout_suppressed_peak_Bmag_T_per_A"], 3e-6)
        self.assertAlmostEqual(summary["layout_suppressed_rms_Bz_T_at_fitted_current"], 6e-6)
        self.assertAlmostEqual(summary["layout_suppressed_peak_abs_Bz_T_at_fitted_current"], 6e-6)
        self.assertAlmostEqual(summary["layout_suppressed_rms_Bmag_T_at_fitted_current"], 6e-6)
        self.assertAlmostEqual(summary["layout_suppressed_peak_Bmag_T_at_fitted_current"], 6e-6)

    def test_dsv_errors_ignore_suppression_points_and_fringe_metrics_use_suppression_points(self) -> None:
        result = _minimal_result_with_suppression_points()

        dsv_mask = dsv_target_mask(result.target_field)
        suppression_mask = suppression_target_mask(result.target_field)
        self.assertEqual(np.count_nonzero(dsv_mask), 2)
        self.assertEqual(np.count_nonzero(suppression_mask), 2)

        relative_error = field_relative_error_scalar_maps(result)["layout_vs_target"].values
        self.assertTrue(np.all(np.isfinite(relative_error[dsv_mask])))
        self.assertTrue(np.all(np.isnan(relative_error[suppression_mask])))

        gradient_map = gradient_component_scalar_maps(result, component="dbz_dy")["layout"].values
        self.assertTrue(np.all(np.isfinite(gradient_map[dsv_mask])))
        self.assertTrue(np.all(np.isnan(gradient_map[suppression_mask])))

        suppression_maps = suppression_field_scalar_maps(result)
        self.assertTrue(np.all(np.isnan(suppression_maps["layout_bmag"].values[dsv_mask])))
        self.assertAlmostEqual(float(np.nanmax(suppression_maps["layout_bmag"].values)), 10.0)

        summary = result_summary_dict(result)
        self.assertEqual(summary["dsv_target_point_count"], 2)
        self.assertEqual(summary["suppression_point_count"], 2)
        self.assertAlmostEqual(summary["layout_mean_rel_error_percent"], 0.0)
        self.assertAlmostEqual(summary["layout_suppressed_peak_Bmag_T_per_A"], 10.0)

    def test_active_performance_table_plot_opens_when_matplotlib_is_available(self) -> None:
        result = run_coilgen(_small_y_gradient_config(skip_sweep=True))

        try:
            fig = plot_performance_table_matplotlib(result, show=False)
        except ImportError as exc:
            self.skipTest(str(exc))
        self.assertEqual(fig.axes[0].get_title(), "Coil Design Performance Summary")
        self.assertTrue(fig.axes[0].tables)

    def test_l_curve_regularization_is_default_and_records_knee(self) -> None:
        config = _small_y_gradient_config(skip_sweep=True)
        config.l_curve_factor_count = 7
        config.l_curve_factor_min_multiplier = 1e-2
        config.l_curve_factor_max_multiplier = 1e2

        result = run_coilgen(config)

        diagnostics = result.regularization_diagnostics
        self.assertEqual(config.regularization_selection_method, "l_curve")
        self.assertEqual(diagnostics["selection_method"], "l_curve")
        self.assertEqual(diagnostics["candidate_solver"], "normal_equations")
        self.assertEqual(diagnostics["l_curve_scan_method"], "normal_equations")
        self.assertEqual(diagnostics["l_curve_curvature_smoothing_window"], 5)
        self.assertGreater(result.regularization_factor, 0.0)
        self.assertGreater(result.regularization_lambda, 0.0)
        self.assertEqual(len(diagnostics["candidate_factors"]), 7)
        self.assertEqual(len(diagnostics["residual_norms"]), 7)
        self.assertGreaterEqual(diagnostics["selected_index"], 0)
        self.assertLess(diagnostics["selected_index"], 7)
        self.assertTrue(diagnostics["fallback"])
        self.assertIn("insufficient dynamic range", diagnostics["fallback_reason"])

    def test_stream_function_optimization_applies_target_weights(self) -> None:
        mesh = Mesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
        )
        part = CoilPart(mesh=mesh)
        part.sensitivity_matrix = np.zeros((3, 2, 3), dtype=float)
        part.sensitivity_matrix[2, 0, :] = np.array([1.0, 0.0, 0.0])
        part.sensitivity_matrix[2, 1, :] = np.array([0.0, 1.0, 0.0])
        part.resistance_matrix = np.eye(3)
        target = TargetField(
            coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float),
            b=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], dtype=float),
            weights=np.array([1.0, 4.0], dtype=float),
            suppressed=np.array([False, True]),
            target_field_group_inds=np.array([1, 2], dtype=np.int64),
        )

        problem = StreamFunctionOptimizationProblem.from_parts([part], target)
        ata, _rtr, rhs = problem.normal_equation_matrices()

        expected_unweighted = part.sensitivity_matrix[2] @ problem.reducer.expansion_matrix
        expected_weighted = expected_unweighted * np.sqrt(target.weights)[:, None]
        np.testing.assert_allclose(problem.unweighted_reduced_sensitivity, expected_unweighted)
        np.testing.assert_allclose(problem.reduced_sensitivity, expected_weighted)
        np.testing.assert_allclose(problem.target_z, [1.0, 6.0])
        np.testing.assert_allclose(ata, expected_weighted.T @ expected_weighted)
        np.testing.assert_allclose(rhs, expected_weighted.T @ problem.target_z)

    def test_stream_function_spectral_basis_reduces_optimizer_unknowns(self) -> None:
        mesh = Mesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.5, 0.5, 0.0],
                ],
                dtype=float,
            ),
            faces=np.array(
                [
                    [0, 1, 4],
                    [1, 2, 4],
                    [2, 3, 4],
                    [3, 0, 4],
                ],
                dtype=np.int64,
            ),
        )
        part = CoilPart(mesh=mesh)
        part.sensitivity_matrix = np.zeros((3, 3, mesh.n_vertices), dtype=float)
        part.sensitivity_matrix[2, :, :3] = np.eye(3)
        part.resistance_matrix = np.eye(mesh.n_vertices)
        target = TargetField(
            coords=np.array([[0.0, 0.0, 0.1], [0.5, 0.5, 0.1], [1.0, 1.0, 0.1]], dtype=float),
            b=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.5], [0.0, 0.0, -0.2]], dtype=float),
            weights=np.ones(3, dtype=float),
            suppressed=np.zeros(3, dtype=bool),
            target_field_group_inds=np.ones(3, dtype=np.int64),
        )
        config = CoilGenConfig(
            stream_function_basis="cotangent_spectral_lowpass",
            stream_function_spectral_cutoff_length_m=0.0,
            stream_function_spectral_min_modes=2,
            stream_function_spectral_max_modes=2,
        )

        problem = StreamFunctionOptimizationProblem.from_parts([part], target, config)
        diagnostics = problem.symmetry_diagnostics or {}

        self.assertTrue(diagnostics["stream_function_spectral_active"])
        self.assertTrue(diagnostics["stream_function_gauge_removed"])
        self.assertEqual(diagnostics["stream_function_gauge_mode_count"], 1)
        self.assertEqual(diagnostics["stream_function_spectral_unknown_count_before"], mesh.n_vertices - 1)
        self.assertEqual(problem.reduced_sensitivity.shape[1], 2)
        expanded = problem.expand(np.array([1.0, -0.5], dtype=float))
        self.assertEqual(expanded.shape, (mesh.n_vertices,))
        _stiffness, mass = cotangent_stiffness_and_lumped_mass(mesh)
        physical_basis = problem.reducer.expansion_matrix
        np.testing.assert_allclose(mass @ physical_basis, 0.0, atol=1.0e-12)
        np.testing.assert_allclose(
            physical_basis.T @ (mass[:, None] * physical_basis),
            np.eye(physical_basis.shape[1]),
            atol=1.0e-10,
        )

    def test_stream_function_hard_spectral_cutoff_never_restores_shorter_modes(self) -> None:
        mesh = _grid_mesh(5, 5)
        part = CoilPart(mesh=mesh)
        part.sensitivity_matrix = np.zeros((3, 3, mesh.n_vertices), dtype=float)
        part.sensitivity_matrix[2, :, :3] = np.eye(3)
        part.resistance_matrix = np.eye(mesh.n_vertices)
        target = TargetField(
            coords=np.array([[0.0, 0.0, 0.1], [0.5, 0.5, 0.1], [1.0, 1.0, 0.1]], dtype=float),
            b=np.column_stack((np.zeros(3), np.zeros(3), np.array([1.0, 0.5, -0.2]))),
            weights=np.ones(3),
            suppressed=np.zeros(3, dtype=bool),
            target_field_group_inds=np.ones(3, dtype=np.int64),
        )
        cutoff = 1.5
        config = CoilGenConfig(
            stream_function_basis="cotangent_spectral_lowpass",
            stream_function_spectral_cutoff_length_m=cutoff,
            stream_function_spectral_cutoff_policy="hard",
            stream_function_spectral_min_modes=20,
            stream_function_spectral_max_modes=20,
        )

        problem = StreamFunctionOptimizationProblem.from_parts([part], target, config)
        diagnostics = problem.symmetry_diagnostics or {}

        self.assertEqual(diagnostics["stream_function_spectral_cutoff_policy"], "hard")
        self.assertFalse(diagnostics["stream_function_spectral_min_modes_overrode_cutoff"])
        self.assertGreaterEqual(
            diagnostics["stream_function_spectral_min_retained_wavelength_m"],
            cutoff * (1.0 - 1.0e-10),
        )

    def test_fem_biharmonic_metric_is_psd_and_annihilates_constant_stream_function(self) -> None:
        mesh = _grid_mesh(5, 4)
        metric = _surface_biharmonic_metric(mesh)
        eigenvalues = np.linalg.eigvalsh(metric)

        self.assertGreaterEqual(float(np.min(eigenvalues)), -1.0e-9 * float(np.max(np.abs(eigenvalues))))
        np.testing.assert_allclose(metric @ np.ones(mesh.n_vertices), 0.0, atol=1.0e-9)

    def test_resistance_matrix_integrates_current_density_over_area_once(self) -> None:
        mesh = normalize_mesh(
            Mesh(
                vertices=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=float),
                faces=np.array([[0, 1, 2]], dtype=np.int64),
            )
        )
        part = calculate_basis_functions(calculate_one_ring_by_mesh([CoilPart(mesh=mesh)]))[0]
        area = float(part.basis_elements[0].area[0])
        self.assertNotAlmostEqual(area, area * area)

        part = calculate_resistance_matrix(
            [part],
            CoilGenConfig(specific_conductivity_conductor=1.0, conductor_thickness=1.0),
        )[0]

        expected = np.zeros((3, 3), dtype=float)
        for node_a in range(3):
            current_a = part.basis_elements[node_a].current[0]
            for node_b in range(3):
                current_b = part.basis_elements[node_b].current[0]
                expected[node_a, node_b] = float(np.dot(current_a, current_b) * area)
        np.testing.assert_allclose(part.resistance_matrix, expected, atol=1e-12)

    def test_stream_function_regularization_uses_resistance_quadratic_directly(self) -> None:
        mesh = Mesh(
            vertices=np.zeros((2, 3), dtype=float),
            faces=np.array([[0, 1, 1]], dtype=np.int64),
        )
        reducer = BoundaryReducer(
            full_node_count=2,
            boundary_nodes=[],
            non_boundary_nodes=np.arange(2),
            expansion_matrix=np.eye(2),
        )
        resistance_quadratic = np.diag([2.0, 3.0])
        sensitivity = np.zeros((3, 2, 2), dtype=float)
        sensitivity[2] = np.eye(2)
        problem = StreamFunctionOptimizationProblem(
            sensitivity_matrix=sensitivity,
            resistance_matrix=resistance_quadratic,
            combined_mesh=mesh,
            reducer=reducer,
            target_z=np.zeros(2, dtype=float),
            reduced_sensitivity=np.eye(2),
            reduced_resistance=resistance_quadratic,
        )

        _ata, regularization_quadratic, _rhs = problem.normal_equation_matrices()

        np.testing.assert_allclose(regularization_quadratic, resistance_quadratic)
        self.assertAlmostEqual(problem.resistance_norm(np.array([2.0, 1.0])), math.sqrt(11.0))

    def test_flanged_split_bore_suppression_surface_generation(self) -> None:
        config = CoilGenConfig(
            coil_mesh_file="create flanged split bore mesh",
            flanged_split_bore_mesh_parameter_list=(0.05, 0.2, 0.3, 0.1, 0.2),
            create_flanged_split_bore_field_suppression_surface="true",
            flanged_split_bore_area_suppression_parameters=(0.05, 0.4, 0.5, 0.1, 0.2),
        )

        mesh = load_secondary_target_mesh(config, Path("."))

        self.assertIsNotNone(mesh)
        self.assertGreater(mesh.n_vertices, 0)
        self.assertGreater(mesh.n_faces, 0)
        radii = np.linalg.norm(mesh.vertices[:, :2], axis=1)
        self.assertGreater(float(np.max(radii)), 0.49)

    def test_fixed_regularization_preserves_configured_factor(self) -> None:
        config = _small_y_gradient_config(skip_sweep=True)
        config.regularization_selection_method = "fixed"
        config.tikonov_reg_factor = 12.5

        result = run_coilgen(config)

        self.assertEqual(result.regularization_diagnostics["selection_method"], "fixed")
        self.assertAlmostEqual(result.regularization_factor, 12.5)

    def test_min_turn_distance_limits_contour_density(self) -> None:
        baseline_config = _small_y_gradient_config(skip_sweep=True)
        baseline_config.cylinder_mesh_parameter_list = (0.25, 0.1, 10, 5, 1, 0, 0, 0)
        baseline_config.levels = 6
        baseline_config.regularization_selection_method = "fixed"
        baseline_config.tikonov_reg_factor = 2.0
        baseline_config.skip_postprocessing = True
        baseline_config.skip_calculation_min_winding_distance = False

        spacing_config = _small_y_gradient_config(skip_sweep=True)
        spacing_config.cylinder_mesh_parameter_list = baseline_config.cylinder_mesh_parameter_list
        spacing_config.levels = baseline_config.levels
        spacing_config.regularization_selection_method = baseline_config.regularization_selection_method
        spacing_config.tikonov_reg_factor = baseline_config.tikonov_reg_factor
        spacing_config.skip_postprocessing = True
        spacing_config.skip_calculation_min_winding_distance = False
        spacing_config.min_turn_distance = 0.02

        baseline = run_coilgen(baseline_config)
        constrained = run_coilgen(spacing_config)
        diagnostics = constrained.regularization_diagnostics

        self.assertLess(constrained.num_levels, baseline.num_levels)
        self.assertGreater(
            constrained.coil_parts[0].minimal_contour_distance,
            baseline.coil_parts[0].minimal_contour_distance,
        )
        self.assertGreaterEqual(diagnostics["measured_min_contour_distance"], spacing_config.min_turn_distance)
        self.assertTrue(diagnostics["turn_spacing_measured_threshold_met"])
        self.assertFalse(diagnostics["turn_spacing_penalty_enabled"])
        self.assertFalse(diagnostics["turn_spacing_threshold_met"])

    def test_turn_spacing_density_percentile_ignores_isolated_current_spike(self) -> None:
        mesh = Mesh(
            vertices=np.zeros((3, 3), dtype=float),
            faces=np.tile(np.array([[0, 1, 2]], dtype=np.int64), (101, 1)),
        )
        density = np.zeros((101, 3), dtype=float)
        density[:, 0] = 1.0
        density[-1, 0] = 1000.0
        part = CoilPart(mesh=mesh)
        part.current_density = density

        strict = CoilGenConfig(min_turn_distance=1.0, turn_spacing_density_percentile=100.0)
        robust = CoilGenConfig(min_turn_distance=1.0, turn_spacing_density_percentile=95.0)

        strict_count, strict_diag = _spacing_limited_level_count_and_diagnostics(
            [part],
            stream_function_range=10.0,
            requested_levels=10,
            level_offset=0.25,
            config=strict,
        )
        robust_count, robust_diag = _spacing_limited_level_count_and_diagnostics(
            [part],
            stream_function_range=10.0,
            requested_levels=10,
            level_offset=0.25,
            config=robust,
        )

        self.assertEqual(strict_count, 1)
        self.assertEqual(robust_count, 10)
        self.assertEqual(strict_diag["turn_spacing_reference_current_density_A_per_m"], 1000.0)
        self.assertEqual(robust_diag["turn_spacing_reference_current_density_A_per_m"], 1.0)
        self.assertEqual(robust_diag["turn_spacing_max_current_density_A_per_m"], 1000.0)

    def test_engineering_contouring_uses_min_wire_distance_and_current_density(self) -> None:
        mesh = Mesh(vertices=np.zeros((3, 3), dtype=float), faces=np.array([[0, 1, 2]], dtype=np.int64))
        part = CoilPart(
            mesh=mesh,
            stream_function=np.array([-1.0, 0.0, 1.0], dtype=float),
            current_density=np.array([[2.0, 0.0, 0.0]], dtype=float),
        )

        parts, primary_ind = calc_potential_levels(
            [part],
            mesh,
            CoilGenConfig(
                contour_discretization_method="engineering",
                max_current_A=1.0,
                min_wire_distance_m=0.25,
                pot_offset_factor=0.25,
                level_set_method="primary",
            ),
        )

        self.assertEqual(primary_ind, 0)
        self.assertAlmostEqual(parts[0].contour_step, 0.5)
        np.testing.assert_allclose(parts[0].potential_level_list, np.array([-0.75, -0.25, 0.25, 0.75]))
        self.assertEqual(parts[0].contour_level_diagnostics["method"], "engineering_current_spacing")
        self.assertTrue(parts[0].contour_level_diagnostics["engineering_current_limit_feasible"])

    def test_engineering_contouring_reports_infeasible_current_cap(self) -> None:
        mesh = Mesh(vertices=np.zeros((3, 3), dtype=float), faces=np.array([[0, 1, 2]], dtype=np.int64))
        part = CoilPart(
            mesh=mesh,
            stream_function=np.array([-1.0, 0.0, 1.0], dtype=float),
            current_density=np.array([[10.0, 0.0, 0.0]], dtype=float),
        )

        parts, _primary_ind = calc_potential_levels(
            [part],
            mesh,
            CoilGenConfig(
                contour_discretization_method="engineering",
                max_current_A=1.0,
                min_wire_distance_m=0.25,
                pot_offset_factor=0.25,
                level_set_method="primary",
            ),
        )

        self.assertAlmostEqual(parts[0].contour_step, 1.0)
        self.assertFalse(parts[0].contour_level_diagnostics["engineering_current_limit_feasible"])
        self.assertAlmostEqual(parts[0].contour_level_diagnostics["engineering_required_contour_step_A"], 2.5)

    def test_engineering_current_density_limit_scales_stream_function_before_contouring(self) -> None:
        mesh = Mesh(
            vertices=np.zeros((3, 3), dtype=float),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
            stream_function=np.array([2.0, 0.0, -2.0], dtype=float),
        )
        part = CoilPart(
            mesh=mesh,
            stream_function=mesh.stream_function.copy(),
            current_density=np.array([[10.0, 0.0, 0.0]], dtype=float),
        )
        config = CoilGenConfig(
            contour_discretization_method="engineering",
            max_current_A=1.0,
            min_wire_distance_m=0.5,
        )

        parts, combined_mesh, b_field, diagnostics = enforce_stream_function_current_density_limit(
            [part],
            mesh,
            np.ones((2, 3), dtype=float),
            config,
        )

        self.assertTrue(diagnostics["surface_current_density_scaling_applied"])
        self.assertAlmostEqual(diagnostics["surface_current_density_scaling_factor"], 0.2)
        self.assertFalse(diagnostics["gradient_hardware_feasible"])
        self.assertEqual(diagnostics["gradient_hardware_infeasible_policy"], "scale_to_limit")
        self.assertAlmostEqual(diagnostics["gradient_hardware_scale_factor"], 0.2)
        self.assertAlmostEqual(diagnostics["gradient_hardware_effective_target_gradient_strength"], 0.2)
        np.testing.assert_allclose(parts[0].stream_function, np.array([0.4, 0.0, -0.4]))
        np.testing.assert_allclose(parts[0].current_density, np.array([[2.0, 0.0, 0.0]]))
        np.testing.assert_allclose(combined_mesh.stream_function, np.array([0.4, 0.0, -0.4]))
        np.testing.assert_allclose(b_field, 0.2 * np.ones((2, 3), dtype=float))

    def test_engineering_current_density_limit_can_report_without_scaling(self) -> None:
        mesh = Mesh(
            vertices=np.zeros((3, 3), dtype=float),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
            stream_function=np.array([2.0, 0.0, -2.0], dtype=float),
        )
        part = CoilPart(
            mesh=mesh,
            stream_function=mesh.stream_function.copy(),
            current_density=np.array([[10.0, 0.0, 0.0]], dtype=float),
        )
        config = CoilGenConfig(
            contour_discretization_method="engineering",
            max_current_A=1.0,
            min_wire_distance_m=0.5,
            gradient_hardware_infeasible_policy="report_only",
        )

        parts, combined_mesh, b_field, diagnostics = enforce_stream_function_current_density_limit(
            [part],
            mesh,
            np.ones((2, 3), dtype=float),
            config,
        )

        self.assertFalse(diagnostics["surface_current_density_scaling_applied"])
        self.assertFalse(diagnostics["gradient_hardware_feasible"])
        self.assertAlmostEqual(diagnostics["gradient_hardware_scale_factor"], 0.2)
        np.testing.assert_allclose(parts[0].stream_function, np.array([2.0, 0.0, -2.0]))
        np.testing.assert_allclose(combined_mesh.stream_function, np.array([2.0, 0.0, -2.0]))
        np.testing.assert_allclose(b_field, np.ones((2, 3), dtype=float))

    def test_contour_path_resampling_keeps_inserted_points_on_mesh_faces(self) -> None:
        mesh, contour = _single_triangle_contour()
        part = CoilPart(mesh=mesh, contour_lines=[contour])
        config = CoilGenConfig(
            contour_path_regularization_mode="resample",
            contour_resample_target_segment_length=0.1,
            contour_only_sweep_mode=True,
            min_point_loop_number=3,
        )

        regularize_contour_paths([part], config)

        resampled = part.contour_lines[0]
        self.assertGreater(len(resampled.v), len(contour.v))
        self.assertIsNotNone(resampled.segment_faces)
        self.assertIsNotNone(resampled.segment_barycentric)
        self.assertLessEqual(float(np.max(np.linalg.norm(np.diff(resampled.v, axis=0), axis=1))), 0.1 + 1e-12)
        np.testing.assert_allclose(np.sum(resampled.segment_barycentric, axis=2), 1.0, atol=1e-10)
        for face_ind, bary, point in zip(resampled.surface_faces, resampled.surface_barycentric, resampled.v):
            reconstructed = bary @ mesh.vertices[mesh.faces[int(face_ind)]]
            np.testing.assert_allclose(point, reconstructed, atol=1e-12)

    def test_contour_path_resampling_defers_before_routing(self) -> None:
        mesh, contour = _single_triangle_contour()
        part = CoilPart(mesh=mesh, contour_lines=[contour])
        config = CoilGenConfig(
            contour_path_regularization_mode="resample",
            contour_resample_target_segment_length=0.1,
            skip_postprocessing=False,
            contour_only_sweep_mode=False,
            min_point_loop_number=3,
        )

        regularize_contour_paths([part], config)

        diagnostics = part.contour_level_diagnostics
        self.assertEqual(len(part.contour_lines[0].v), len(contour.v))
        self.assertFalse(diagnostics["contour_path_resampling_applied"])
        self.assertTrue(diagnostics["contour_path_resampling_deferred_until_after_routing"])

    def test_parameterized_bspline_smoothing_maps_regularized_segments_to_faces(self) -> None:
        mesh, contour = _single_triangle_contour()
        part = CoilPart(mesh=mesh, contour_lines=[contour])
        config = CoilGenConfig(
            contour_path_regularization_mode="parameterized_bspline",
            contour_smoothing_factor=0.5,
            contour_smoothing_max_displacement=0.05,
            min_point_loop_number=3,
        )

        regularize_contour_paths([part], config)

        smoothed = part.contour_lines[0]
        self.assertTrue(part.contour_level_diagnostics["contour_path_smoothing_applied"])
        self.assertGreater(len(smoothed.v), len(contour.v))
        self.assertIsNone(smoothed.surface_edge_vertices)
        self.assertIsNone(smoothed.surface_edge_t)
        for face_ind, barycentric, point in zip(
            smoothed.surface_faces,
            smoothed.surface_barycentric,
            smoothed.v,
        ):
            reconstructed = barycentric @ mesh.vertices[mesh.faces[int(face_ind)]]
            np.testing.assert_allclose(point, reconstructed, atol=1e-12)
        np.testing.assert_allclose(np.sum(smoothed.segment_barycentric, axis=2), 1.0, atol=1e-10)

    def test_contour_smoothing_spacing_guard_measures_segment_to_segment_distance(self) -> None:
        horizontal = np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
        vertical = np.asarray([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)

        self.assertAlmostEqual(_contour_pair_min_distance(horizontal, vertical), 0.0, places=12)

    def test_stream_function_optimization_reports_scheme_and_cost_function(self) -> None:
        config = _small_y_gradient_config(skip_sweep=True)
        config.sf_opt_method = "tikhonov"
        config.l_curve_factor_count = 7

        result = run_coilgen(config)

        diagnostics = result.regularization_diagnostics
        self.assertEqual(diagnostics["stream_function_scheme"], "tikhonov")
        self.assertEqual(diagnostics["stream_function_cost_function"], "field_error_plus_resistance")
        self.assertIn("A x - b", diagnostics["cost_function_formula"])
        self.assertGreater(diagnostics["unknown_count"], 0)
        self.assertGreater(diagnostics["target_sample_count"], 0)

    def test_unknown_stream_function_cost_function_fails_clearly(self) -> None:
        config = _small_y_gradient_config(skip_sweep=True)
        config.stream_function_cost_function = "not_a_physical_objective"

        with self.assertRaisesRegex((NotImplementedError, ValueError), "cost function|stream_function_cost_function"):
            run_coilgen(config)

    def test_small_y_gradient_can_use_geodesic_inter_group_paths(self) -> None:
        config = _small_y_gradient_config(skip_sweep=True)
        config.inter_group_connection_path_method = "geodesic"

        result = run_coilgen(config)

        part = result.coil_parts[0]
        self.assertIsNotNone(part.wire_path)
        self.assertTrue(any(len(cut.connection_path_v) for cut in part.opening_cuts_among_groups))
        self.assertEqual(part.route_diagnostics["inter_group_cut_count"], len(part.opening_cuts_among_groups))
        self.assertGreater(part.route_diagnostics["inter_group_geodesic_connector_count"], 0)
        self.assertEqual(part.route_diagnostics["inter_group_direct_fallback_count"], 0)
        self.assertGreater(part.route_diagnostics["max_inter_group_direct_distance"], 0.0)
        self.assertGreater(part.route_diagnostics["max_inter_group_geodesic_distance"], 0.0)
        self.assertGreaterEqual(part.route_diagnostics["min_wire_wire_distance"], 0.0)
        self.assertGreaterEqual(part.route_diagnostics["min_inter_group_connector_to_contour_distance"], 0.0)
        self.assertGreaterEqual(part.route_diagnostics["max_inter_group_connector_ratio"], 1.0)
        self.assertEqual(part.route_diagnostics["suspicious_inter_group_connector_count"], 0)
        self.assertGreater(part.route_diagnostics["opening_cut_count"], 0)
        self.assertEqual(part.route_diagnostics["degenerate_opening_cut_count"], 0)
        self.assertGreater(part.route_diagnostics["min_opening_cut_span"], 0.0)
        for cut in part.opening_cuts_among_groups:
            self.assertGreaterEqual(cut.group1_index, 0)
            self.assertGreaterEqual(cut.group2_index, 0)
            self.assertEqual(cut.selected_point1_v.shape, (1, 3))
            self.assertEqual(cut.selected_point2_v.shape, (1, 3))
            self.assertGreater(cut.direct_distance, 0.0)
            self.assertTrue(np.isfinite(cut.geodesic_distance))
            self.assertEqual(cut.connection_method, "geodesic")

    def test_native_inductance_solver_for_circular_loop(self) -> None:
        angle = np.linspace(0.0, 2.0 * np.pi, 65)
        radius = 0.05
        points = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), np.zeros_like(angle)))
        part = CoilPart(mesh=Mesh(vertices=points[:3], faces=np.array([[0, 1, 2]], dtype=np.int64)))
        part.wire_path = WirePath(v=points)
        config = CoilGenConfig(
            skip_inductance_calculation=False,
            conductor_cross_section_width=0.001,
            conductor_cross_section_height=0.001,
            specific_conductivity_conductor=1.724e-8,
        )

        parts, inductance_matrix, resistance, lengths, cross_sections = calculate_inductance_by_coil_layout(
            [part], config
        )

        self.assertGreater(parts[0].coil_inductance, 0.0)
        self.assertGreater(parts[0].coil_resistance, 0.0)
        self.assertAlmostEqual(lengths[0], 2.0 * np.pi * radius, delta=2e-3)
        self.assertAlmostEqual(cross_sections[0], 1e-6)
        self.assertGreater(inductance_matrix[0, 0], 0.0)
        self.assertGreater(resistance[0], 0.0)

    def test_native_inductance_solver_reports_mutual_matrix_for_multiple_parts(self) -> None:
        angle = np.linspace(0.0, 2.0 * np.pi, 33)
        radius = 0.04
        loop_a = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), np.zeros_like(angle)))
        loop_b = loop_a + np.array([0.0, 0.0, 0.03])
        part_a = CoilPart(mesh=Mesh(vertices=loop_a[:3], faces=np.array([[0, 1, 2]], dtype=np.int64)))
        part_b = CoilPart(mesh=Mesh(vertices=loop_b[:3], faces=np.array([[0, 1, 2]], dtype=np.int64)))
        part_a.wire_path = WirePath(v=loop_a)
        part_b.wire_path = WirePath(v=loop_b)
        config = CoilGenConfig(
            skip_inductance_calculation=False,
            conductor_cross_section_width=0.001,
            conductor_cross_section_height=0.001,
            specific_conductivity_conductor=1.724e-8,
        )

        _parts, inductance_matrix, _resistance, _lengths, _cross_sections = calculate_inductance_by_coil_layout(
            [part_a, part_b],
            config,
        )
        summary = inductance_matrix_summary_dict(inductance_matrix)

        self.assertEqual(inductance_matrix.shape, (2, 2))
        self.assertAlmostEqual(inductance_matrix[0, 1], inductance_matrix[1, 0])
        self.assertGreater(abs(inductance_matrix[0, 1]), 0.0)
        self.assertIn("coil_mutual_inductance_part1_part2_H", summary)
        self.assertAlmostEqual(summary["coil_mutual_inductance_part1_part2_H"], inductance_matrix[0, 1])

    def test_swept_conductor_body_and_stl_export(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                [0.05, 0.05, 0.0],
            ],
            dtype=float,
        )
        mesh = normalize_mesh(
            Mesh(
                vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                faces=np.array([[0, 1, 2]], dtype=np.int64),
            )
        )
        part = CoilPart(mesh=mesh)
        part.wire_path = WirePath(v=points)
        config = CoilGenConfig(
            skip_sweep=False,
            conductor_cross_section_width=0.002,
            conductor_cross_section_height=0.001,
            # This serialization fixture intentionally retains a 90-degree miter.
            sweep_max_frame_rotation_deg=90.0,
        )

        part = create_swept_coil_bodies([part], config)[0]

        self.assertIsNotNone(part.swept_body)
        self.assertGreater(part.swept_body.n_vertices, len(points))
        self.assertGreater(part.swept_body.n_faces, 0)
        self.assertEqual(part.manufacturing_diagnostics["sweep_geometry_backend"], "triangular_mesh")
        self.assertEqual(part.manufacturing_diagnostics["sweep_geometry_role"], "visualization_mesh_export")
        self.assertFalse(part.manufacturing_diagnostics["analytic_solid_created"])
        self.assertTrue(part.manufacturing_diagnostics["manufacturing_ok"])

        with TemporaryDirectory() as tmp:
            result = run_coilgen(_small_y_gradient_config(skip_sweep=False))
            # This compact numerical fixture does not satisfy the physical
            # conductor-envelope clearance gate. Bypass manufacturing readiness
            # here because this test exercises serialization only.
            files = save_swept_bodies_as_stl(result, tmp, prefix="body", fail_on_violations=False)
            self.assertEqual(len(files), 2)
            self.assertEqual(files[0].name, "body_part1.stl")
            self.assertEqual(files[1].name, "body_part1_current_carrying_surface.stl")
            volume_text = files[0].read_text(encoding="ascii")
            surface_text = files[1].read_text(encoding="ascii")
            self.assertIn("solid body_part1", volume_text)
            self.assertIn("facet normal", volume_text)
            self.assertIn("solid body_part1_current_carrying_surface", surface_text)
            self.assertIn("facet normal", surface_text)
            self.assertEqual(surface_text.count("facet normal"), result.coil_parts[0].mesh.n_faces)
            step_files = save_swept_bodies_as_step(result, tmp, prefix="body", fail_on_violations=False)
            self.assertEqual(len(step_files), 1)
            step_text = step_files[0].read_text(encoding="ascii")
            self.assertIn("ISO-10303-21", step_text)
            self.assertIn("MANIFOLD_SOLID_BREP('body_part1'", step_text)
            self.assertIn("ADVANCED_FACE", step_text)
            self.assertIn("PLANE", step_text)
            self.assertIn("CLOSED_SHELL", step_text)
            self.assertIn("END-ISO-10303-21", step_text)

    def test_closed_sweep_is_periodic_without_coincident_end_caps(self) -> None:
        angles = np.linspace(0.0, 2.0 * np.pi, 9)
        points = np.column_stack((0.05 * np.cos(angles), 0.05 * np.sin(angles), np.zeros_like(angles)))
        mesh = normalize_mesh(
            Mesh(
                vertices=np.array(
                    [
                        [-0.1, -0.1, 0.0],
                        [0.1, -0.1, 0.0],
                        [0.1, 0.1, 0.0],
                        [-0.1, 0.1, 0.0],
                    ],
                    dtype=float,
                ),
                faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
            )
        )
        part = CoilPart(mesh=mesh)
        part.wire_path = WirePath(v=points)
        config = CoilGenConfig(
            skip_sweep=False,
            conductor_cross_section_width=0.001,
            conductor_cross_section_height=0.001,
            sweep_frame_method="surface_constrained_parallel_transport",
            # The authoritative input is an octagon, not an analytic circle.
            sweep_max_frame_rotation_deg=45.0,
        )

        part = create_swept_coil_bodies([part], config)[0]
        diagnostics = part.manufacturing_diagnostics

        self.assertTrue(diagnostics["sweep_closed_path"])
        self.assertEqual(diagnostics["sweep_end_cap_count"], 0)
        self.assertGreater(diagnostics["sweep_ring_count"], len(points) - 1)
        self.assertGreater(diagnostics["sweep_adaptive_points_inserted"], 0)
        self.assertEqual(part.swept_body.n_vertices, 4 * diagnostics["sweep_ring_count"])
        self.assertEqual(part.swept_body.n_faces, 8 * diagnostics["sweep_ring_count"])
        self.assertEqual(diagnostics["misoriented_edge_count"], 0)
        self.assertTrue(diagnostics["positive_volume"])
        self.assertTrue(diagnostics["manufacturing_ok"])

    def test_mesh_diagnostics_reject_inconsistent_face_orientation(self) -> None:
        points = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=float)
        support = normalize_mesh(
            Mesh(
                vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                faces=np.array([[0, 1, 2]], dtype=np.int64),
            )
        )
        part = CoilPart(mesh=support)
        part.wire_path = WirePath(v=points)
        part = create_swept_coil_bodies(
            [part],
            CoilGenConfig(
                skip_sweep=False,
                conductor_cross_section_width=0.002,
                conductor_cross_section_height=0.001,
            ),
        )[0]
        faces = part.swept_body.faces.copy()
        faces[0] = faces[0, ::-1]

        diagnostics = mesh_manufacturing_diagnostics(Mesh(vertices=part.swept_body.vertices, faces=faces))

        self.assertGreater(diagnostics["misoriented_edge_count"], 0)
        self.assertFalse(diagnostics["manufacturing_ok"])

    def test_infeasible_minimum_bend_radius_blocks_manufacturing_export(self) -> None:
        points = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.001, 0.001, 0.0]], dtype=float)
        support = normalize_mesh(
            Mesh(
                vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                faces=np.array([[0, 1, 2]], dtype=np.int64),
            )
        )
        part = CoilPart(mesh=support)
        part.wire_path = WirePath(v=points)
        part = create_swept_coil_bodies(
            [part],
            CoilGenConfig(
                skip_sweep=False,
                conductor_cross_section_width=0.0002,
                conductor_cross_section_height=0.0002,
                sweep_bend_mode="fillet",
                sweep_min_bend_radius=0.003,
            ),
        )[0]

        self.assertTrue(part.manufacturing_diagnostics["sweep_min_bend_radius_violation"])
        self.assertIn("minimum_bend_radius_violation", part.manufacturing_diagnostics["manufacturing_failure_reasons"])
        self.assertFalse(part.manufacturing_diagnostics["manufacturing_ok"])
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "minimum_bend_radius_violation"):
                save_swept_bodies_as_stl(
                    SimpleNamespace(coil_parts=[part]), tmp, prefix="infeasible", fail_on_violations=True
                )

    def test_mitered_corner_violates_positive_minimum_bend_radius(self) -> None:
        points = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.05, 0.05, 0.0]], dtype=float)
        support = normalize_mesh(
            Mesh(
                vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                faces=np.array([[0, 1, 2]], dtype=np.int64),
            )
        )
        part = CoilPart(mesh=support)
        part.wire_path = WirePath(v=points)

        part = create_swept_coil_bodies(
            [part],
            CoilGenConfig(
                skip_sweep=False,
                conductor_cross_section_width=0.002,
                conductor_cross_section_height=0.001,
                sweep_bend_mode="miter",
                sweep_min_bend_radius=0.003,
            ),
        )[0]

        self.assertTrue(part.manufacturing_diagnostics["sweep_min_bend_radius_violation"])
        self.assertFalse(part.manufacturing_diagnostics["manufacturing_ok"])

    def test_capsule_collision_uses_bvh_and_detects_remote_crossing(self) -> None:
        points = np.array(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 2.0, 0.0],
                [-1.0, 2.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )

        diagnostics = _capsule_clearance_diagnostics(points, 0.2, 0.1, closed=False)

        self.assertEqual(diagnostics["sweep_capsule_broad_phase"], "bvh")
        self.assertTrue(diagnostics["sweep_capsule_self_intersection"])
        self.assertEqual(diagnostics["sweep_capsule_min_centerline_distance_m"], 0.0)
        self.assertGreater(diagnostics["sweep_capsule_evaluated_pair_count"], 0)

    def test_capsule_bvh_does_not_skip_routes_above_legacy_limit(self) -> None:
        points = np.column_stack(
            (
                np.arange(3202, dtype=float),
                np.zeros(3202, dtype=float),
                np.zeros(3202, dtype=float),
            )
        )

        diagnostics = _capsule_clearance_diagnostics(points, 0.02, 0.01, closed=False)
        all_pair_count = 3201 * 3200 // 2

        self.assertEqual(diagnostics["sweep_capsule_broad_phase"], "bvh")
        self.assertFalse(diagnostics["sweep_capsule_self_intersection"])
        self.assertLess(diagnostics["sweep_capsule_candidate_pair_count"], all_pair_count // 100)

    def test_capsule_local_exclusion_uses_gap_between_segments(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0001, 0.0],
                [2.0, 0.0001, 0.0],
            ],
            dtype=float,
        )

        diagnostics = _capsule_clearance_diagnostics(points, 0.01, 0.005, closed=False)

        self.assertFalse(diagnostics["sweep_capsule_self_intersection"])
        self.assertEqual(diagnostics["sweep_capsule_evaluated_pair_count"], 0)

    def test_unspecified_route_failure_blocks_export_even_after_mesh_validation(self) -> None:
        points = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=float)
        support = normalize_mesh(
            Mesh(
                vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                faces=np.array([[0, 1, 2]], dtype=np.int64),
            )
        )
        part = CoilPart(mesh=support)
        part.wire_path = WirePath(v=points)
        part = create_swept_coil_bodies(
            [part],
            CoilGenConfig(
                skip_sweep=False,
                conductor_cross_section_width=0.002,
                conductor_cross_section_height=0.001,
            ),
        )[0]
        self.assertTrue(part.manufacturing_diagnostics["mesh_solid_ok"])
        part.route_diagnostics["route_quality_status"] = "fail"

        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "route_topology_failure"):
                save_swept_bodies_as_stl(
                    SimpleNamespace(coil_parts=[part]), tmp, prefix="route_failure", fail_on_violations=True
                )

        self.assertIn("route_topology_failure", part.manufacturing_diagnostics["manufacturing_failure_reasons"])
        self.assertEqual(
            part.manufacturing_diagnostics["manufacturing_route_topology_failure_reasons"],
            "unspecified_route_failure",
        )
        self.assertFalse(part.manufacturing_diagnostics["manufacturing_ok"])

    def test_design_clearance_target_failure_does_not_replace_physical_sweep_clearance(self) -> None:
        points = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=float)
        support = normalize_mesh(
            Mesh(
                vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                faces=np.array([[0, 1, 2]], dtype=np.int64),
            )
        )
        part = CoilPart(mesh=support)
        part.wire_path = WirePath(v=points)
        part.route_diagnostics.update(
            {
                "route_quality_status": "fail",
                "route_quality_failure_count": 1,
                "route_quality_failure_reasons": "wire_clearance_target",
                "wire_clearance_failure_count": 1,
                "min_wire_wire_distance": 0.01,
                "min_wire_wire_segment_a": 0,
                "min_wire_wire_segment_b": 2,
            }
        )

        part = create_swept_coil_bodies(
            [part],
            CoilGenConfig(
                skip_sweep=False,
                conductor_cross_section_width=0.002,
                conductor_cross_section_height=0.001,
            ),
        )[0]
        diagnostics = part.manufacturing_diagnostics

        self.assertTrue(diagnostics["route_design_target_failure"])
        self.assertFalse(diagnostics["manufacturing_route_topology_failure"])
        self.assertFalse(diagnostics["sweep_route_clearance_violation"])
        self.assertFalse(diagnostics["sweep_self_intersection_skipped"])
        self.assertTrue(diagnostics["manufacturing_ok"])
        with TemporaryDirectory() as tmp:
            files = save_swept_bodies_as_stl(SimpleNamespace(coil_parts=[part]), tmp, prefix="target_only")
        self.assertEqual(len(files), 2)

    def test_physical_route_clearance_failure_still_blocks_export(self) -> None:
        points = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=float)
        support = normalize_mesh(
            Mesh(
                vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                faces=np.array([[0, 1, 2]], dtype=np.int64),
            )
        )
        part = CoilPart(mesh=support)
        part.wire_path = WirePath(v=points)
        part.route_diagnostics.update(
            {
                "route_quality_status": "fail",
                "route_quality_failure_count": 1,
                "route_quality_failure_reasons": "wire_clearance_target",
                "wire_clearance_failure_count": 1,
                "min_wire_wire_distance": 0.001,
                "min_wire_wire_segment_a": 0,
                "min_wire_wire_segment_b": 2,
            }
        )

        part = create_swept_coil_bodies(
            [part],
            CoilGenConfig(
                skip_sweep=False,
                conductor_cross_section_width=0.002,
                conductor_cross_section_height=0.001,
            ),
        )[0]
        diagnostics = part.manufacturing_diagnostics

        self.assertTrue(diagnostics["sweep_route_clearance_violation"])
        self.assertFalse(diagnostics["sweep_self_intersection_skipped"])
        self.assertFalse(diagnostics["sweep_self_intersection_skipped_due_route_precheck_failure"])
        self.assertIn("route_clearance_violation", diagnostics["manufacturing_failure_reasons"])
        self.assertNotIn("self_intersection_check_skipped", diagnostics["manufacturing_failure_reasons"])
        self.assertFalse(diagnostics["manufacturing_ok"])
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "route_clearance_violation"):
                save_swept_bodies_as_stl(
                    SimpleNamespace(coil_parts=[part]), tmp, prefix="overlap", fail_on_violations=True
                )

    def test_sweep_path_simplification_removes_redundant_straight_rings(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.025, 0.0, 0.0],
                [0.05, 0.0, 0.0],
            ],
            dtype=float,
        )
        mesh = normalize_mesh(
            Mesh(
                vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                faces=np.array([[0, 1, 2]], dtype=np.int64),
            )
        )
        part = CoilPart(mesh=mesh)
        part.wire_path = WirePath(v=points)
        config = CoilGenConfig(
            skip_sweep=False,
            conductor_cross_section_width=0.002,
            conductor_cross_section_height=0.001,
            sweep_path_simplification="exact",
        )

        part = create_swept_coil_bodies([part], config)[0]

        self.assertEqual(part.manufacturing_diagnostics["sweep_points_removed"], 1)
        self.assertGreater(part.manufacturing_diagnostics["sweep_ring_count"], 2)
        self.assertGreater(part.manufacturing_diagnostics["sweep_adaptive_points_inserted"], 0)
        self.assertEqual(
            part.swept_body.n_vertices,
            4 * part.manufacturing_diagnostics["sweep_ring_count"] + 2,
        )
        self.assertTrue(part.manufacturing_diagnostics["manufacturing_ok"])

    def test_sweep_profile_axes_are_transverse_width_and_surface_normal_height(self) -> None:
        support = normalize_mesh(
            Mesh(
                vertices=np.array(
                    [[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.1, 0.1, 0.0], [-0.1, 0.1, 0.0]],
                    dtype=float,
                ),
                faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
            )
        )
        part = CoilPart(mesh=support)
        part.wire_path = WirePath(v=np.array([[-0.02, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=float))
        part = create_swept_coil_bodies(
            [part],
            CoilGenConfig(
                skip_sweep=False,
                conductor_cross_section_width=0.004,
                conductor_cross_section_height=0.001,
            ),
        )[0]

        ring_vertices = part.swept_body.vertices[: 4 * part.manufacturing_diagnostics["sweep_ring_count"]]
        self.assertAlmostEqual(float(np.ptp(ring_vertices[:, 1])), 0.004, places=10)
        self.assertAlmostEqual(float(np.ptp(ring_vertices[:, 2])), 0.001, places=10)
        self.assertEqual(part.manufacturing_diagnostics["sweep_cross_section_axis_0"], "surface_transverse")
        self.assertEqual(part.manufacturing_diagnostics["sweep_cross_section_axis_1"], "surface_normal")

    def test_sweep_rejects_cross_sections_that_cap_fan_cannot_safely_triangulate(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly convex"):
            conductor_cross_section_points(
                CoilGenConfig(
                    cross_sectional_points=(
                        (-0.002, -0.001),
                        (0.002, -0.001),
                        (0.0, 0.0),
                        (0.002, 0.001),
                        (-0.002, 0.001),
                    )
                )
            )
        with self.assertRaisesRegex(ValueError, "non-self-intersecting"):
            conductor_cross_section_points(
                CoilGenConfig(
                    cross_sectional_points=(
                        (-0.002, -0.001),
                        (0.002, 0.001),
                        (-0.002, 0.001),
                        (0.002, -0.001),
                    )
                )
            )

    def test_surface_adaptive_sampling_replaces_a_cylinder_chord(self) -> None:
        radius = 0.1
        support = normalize_mesh(build_cylinder_mesh(0.2, radius, 64, 8, 1, 0, 0, 0))
        angles = np.array([-0.2, 0.2])
        points = np.column_stack(
            (
                radius * np.sin(angles),
                radius * np.cos(angles),
                np.zeros_like(angles),
            )
        )
        part = CoilPart(mesh=support)
        part.wire_path = WirePath(v=points)
        part = create_swept_coil_bodies(
            [part],
            CoilGenConfig(
                skip_sweep=False,
                conductor_cross_section_width=0.001,
                conductor_cross_section_height=0.001,
                sweep_max_segment_length=0.1,
                sweep_max_chord_error=1e-5,
                sweep_max_frame_rotation_deg=5.0,
            ),
        )[0]

        diagnostics = part.manufacturing_diagnostics
        self.assertGreater(diagnostics["sweep_adaptive_points_inserted"], 0)
        self.assertLessEqual(diagnostics["sweep_adaptive_max_output_segment_length_m"], 0.02)
        self.assertGreater(diagnostics["sweep_adaptive_max_chord_correction_m"], 1e-4)
        self.assertFalse(diagnostics["sweep_centerline_support_domain_violation"])
        self.assertTrue(diagnostics["manufacturing_ok"])

    def test_surface_adaptive_sampling_preserves_direct_connector_chord(self) -> None:
        radius = 0.1
        support = normalize_mesh(build_cylinder_mesh(0.2, radius, 64, 8, 1, 0, 0, 0))
        angles = np.array([-0.2, 0.2])
        points = np.column_stack(
            (
                radius * np.sin(angles),
                radius * np.cos(angles),
                np.zeros_like(angles),
            )
        )
        part = CoilPart(mesh=support)
        part.wire_path = WirePath(
            v=points,
            segment_owners=[WireSegmentOwner(kind="inter_group_connector", connector_index=0)],
            segment_geometry=[WirePathGeometry.FREE_3D],
        )
        part.opening_cuts_among_groups = [
            InterGroupCut(
                cut1_v=points,
                cut2_v=points,
                connection_path_v=points,
                connection_method="direct_3d",
            )
        ]

        part = create_swept_coil_bodies(
            [part],
            CoilGenConfig(
                skip_sweep=False,
                conductor_cross_section_width=0.001,
                conductor_cross_section_height=0.001,
                sweep_max_segment_length=0.01,
                sweep_max_chord_error=1e-5,
                sweep_max_frame_rotation_deg=5.0,
            ),
        )[0]
        diagnostics = part.manufacturing_diagnostics

        self.assertEqual(diagnostics["sweep_adaptive_surface_segment_count"], 0)
        self.assertEqual(diagnostics["sweep_adaptive_linear_segment_count"], 1)
        self.assertEqual(diagnostics["sweep_adaptive_max_chord_correction_m"], 0.0)
        self.assertGreater(diagnostics["sweep_adaptive_points_inserted"], 0)

    def test_support_domain_check_rejects_direct_route_beyond_mesh_boundary(self) -> None:
        support = normalize_mesh(
            Mesh(
                vertices=np.array(
                    [[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.1, 0.1, 0.0], [-0.1, 0.1, 0.0]],
                    dtype=float,
                ),
                faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
            )
        )
        points = np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]], dtype=float)
        part = CoilPart(mesh=support)
        part.wire_path = WirePath(
            v=points,
            segment_owners=[WireSegmentOwner(kind="inter_group_connector", connector_index=0)],
        )
        part.opening_cuts_among_groups = [
            InterGroupCut(
                cut1_v=points,
                cut2_v=points,
                connection_path_v=points,
                connection_method="direct_3d",
            )
        ]

        part = create_swept_coil_bodies(
            [part],
            CoilGenConfig(
                skip_sweep=False,
                conductor_cross_section_width=0.001,
                conductor_cross_section_height=0.001,
            ),
        )[0]

        self.assertTrue(part.manufacturing_diagnostics["sweep_centerline_support_domain_violation"])
        self.assertIn(
            "centerline_outside_support_domain",
            part.manufacturing_diagnostics["manufacturing_failure_reasons"],
        )

    def test_support_domain_check_accepts_normal_lift_over_cylinder(self) -> None:
        radius = 0.1
        support = normalize_mesh(build_cylinder_mesh(0.2, radius, 64, 8, 1, 0, 0, 0))
        angles = np.array([-0.2, 0.0, 0.2])
        lifted_radius = radius + 0.003
        points = np.column_stack(
            (
                lifted_radius * np.sin(angles),
                lifted_radius * np.cos(angles),
                np.zeros_like(angles),
            )
        )
        owner = WireSegmentOwner(kind="contour_group", group_index=0)
        part = CoilPart(mesh=support)
        part.wire_path = WirePath(
            v=points,
            segment_owners=[owner, owner],
            segment_geometry=[WirePathGeometry.OFFSET_SURFACE] * 2,
        )

        part = create_swept_coil_bodies(
            [part],
            CoilGenConfig(
                skip_sweep=False,
                conductor_cross_section_width=0.001,
                conductor_cross_section_height=0.001,
                sweep_max_segment_length=0.01,
                sweep_max_chord_error=1e-5,
            ),
        )[0]

        diagnostics = part.manufacturing_diagnostics
        self.assertFalse(diagnostics["sweep_centerline_support_domain_violation"])
        self.assertEqual(diagnostics["sweep_adaptive_surface_segment_count"], 0)
        self.assertEqual(diagnostics["sweep_adaptive_offset_surface_segment_count"], 2)
        self.assertEqual(diagnostics["sweep_adaptive_max_chord_correction_m"], 0.0)

    def test_lifted_route_crossing_coarse_cylinder_vertex_is_not_reprojected(self) -> None:
        radius = 0.1
        support = normalize_mesh(build_cylinder_mesh(0.2, radius, 42, 8, 1, 0, 0, 0))
        angles = np.array([-0.002, 0.002])
        lifted_radius = radius + 0.006
        points = np.column_stack(
            (
                lifted_radius * np.sin(angles),
                lifted_radius * np.cos(angles),
                np.zeros_like(angles),
            )
        )
        part = CoilPart(mesh=support)
        part.wire_path = WirePath(
            v=points,
            segment_geometry=[WirePathGeometry.OFFSET_SURFACE],
        )

        part = create_swept_coil_bodies(
            [part],
            CoilGenConfig(
                skip_sweep=False,
                conductor_cross_section_width=0.001,
                conductor_cross_section_height=0.001,
                sweep_max_segment_length=5e-5,
                sweep_max_chord_error=1e-5,
            ),
        )[0]
        diagnostics = part.manufacturing_diagnostics
        ring_count = int(diagnostics["sweep_ring_count"])
        rings = part.swept_body.vertices[: 4 * ring_count].reshape((ring_count, 4, 3))
        centers = np.mean(rings, axis=1)
        chord = points[1] - points[0]
        distance_to_input_chord = np.linalg.norm(np.cross(centers - points[0], chord), axis=1) / np.linalg.norm(chord)

        self.assertGreater(diagnostics["sweep_adaptive_points_inserted"], 0)
        self.assertEqual(diagnostics["sweep_adaptive_max_chord_correction_m"], 0.0)
        self.assertEqual(diagnostics["sweep_adaptive_depth_limit_count"], 0)
        self.assertEqual(diagnostics["sweep_adaptive_unresolved_count"], 0)
        self.assertEqual(diagnostics["sweep_folded_quad_count"], 0)
        self.assertLess(float(np.max(distance_to_input_chord)), 1e-10)

    def test_parallel_transport_does_not_flip_a_valid_normal_past_ninety_degrees(self) -> None:
        angle = np.deg2rad(100.0)
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0 + np.cos(angle), np.sin(angle), 0.0],
            ],
            dtype=float,
        )
        tangents = np.array(
            [
                [1.0, 0.0, 0.0],
                [np.cos(angle), np.sin(angle), 0.0],
                [np.cos(angle), np.sin(angle), 0.0],
            ],
            dtype=float,
        )
        support = normalize_mesh(
            Mesh(
                vertices=np.array([[-1.0, 0.0, -1.0], [2.0, 0.0, -1.0], [-1.0, 0.0, 1.0]], dtype=float),
                faces=np.array([[0, 1, 2]], dtype=np.int64),
            )
        )
        part = CoilPart(mesh=support)
        geometry = [WirePathGeometry.FREE_3D, WirePathGeometry.FREE_3D]

        normals, binormals = _parallel_transport_path_frames(
            part,
            points,
            tangents,
            geometry,
            CoilGenConfig(sweep_surface_normal_constraint_strength=0.35),
            closed=False,
        )
        expected = _rotate_between_vectors(normals[0], tangents[0], tangents[1])

        self.assertGreater(float(np.dot(normals[1], expected)), 1.0 - 1e-12)
        self.assertLess(float(np.dot(normals[0], normals[1])), 0.0)
        np.testing.assert_allclose(np.cross(tangents, normals), binormals, atol=1e-12)

    def test_crossing_swept_wire_is_rejected_by_centerline_and_triangle_bvhs(self) -> None:
        support = normalize_mesh(
            Mesh(
                vertices=np.array(
                    [[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.1, 0.1, 0.0], [-0.1, 0.1, 0.0]],
                    dtype=float,
                ),
                faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
            )
        )
        part = CoilPart(mesh=support)
        part.wire_path = WirePath(
            v=np.array(
                [
                    [-0.04, -0.02, 0.0],
                    [0.04, 0.02, 0.0],
                    [0.04, -0.02, 0.0],
                    [-0.04, 0.02, 0.0],
                ],
                dtype=float,
            )
        )
        part = create_swept_coil_bodies(
            [part],
            CoilGenConfig(
                skip_sweep=False,
                conductor_cross_section_width=0.001,
                conductor_cross_section_height=0.001,
            ),
        )[0]

        diagnostics = part.manufacturing_diagnostics
        self.assertTrue(diagnostics["sweep_capsule_self_intersection"])
        self.assertTrue(diagnostics["sweep_mesh_self_intersection"])
        self.assertIn("swept_mesh_self_intersection", diagnostics["manufacturing_failure_reasons"])
        self.assertFalse(diagnostics["manufacturing_ok"])

    def test_surface_outer_mounting_keeps_asymmetric_profile_outside_support(self) -> None:
        support = normalize_mesh(
            Mesh(
                vertices=np.array(
                    [[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.1, 0.1, 0.0], [-0.1, 0.1, 0.0]],
                    dtype=float,
                ),
                faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
            )
        )
        part = CoilPart(mesh=support)
        part.wire_path = WirePath(v=np.array([[-0.02, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=float))
        part = create_swept_coil_bodies(
            [part],
            CoilGenConfig(
                skip_sweep=False,
                conductor_mounting_mode="surface_outer",
                cross_sectional_points=(
                    (-0.002, -0.0002),
                    (0.002, -0.0002),
                    (0.001, 0.0008),
                    (-0.001, 0.0008),
                ),
            ),
        )[0]

        diagnostics = part.manufacturing_diagnostics
        self.assertAlmostEqual(diagnostics["sweep_centerline_offset_m"], 0.0002)
        self.assertEqual(diagnostics["sweep_support_penetration_count"], 0)
        self.assertFalse(diagnostics["sweep_support_penetration_violation"])
        self.assertTrue(diagnostics["manufacturing_ok"])

    def test_sweep_automatically_repairs_collinear_short_lateral_edges(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [0.0100002, 0.0, 0.0],
                [0.02, 0.0, 0.0],
            ],
            dtype=float,
        )
        support = normalize_mesh(
            Mesh(
                vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                faces=np.array([[0, 1, 2]], dtype=np.int64),
            )
        )
        part = CoilPart(mesh=support)
        part.wire_path = WirePath(v=points)

        part = create_swept_coil_bodies(
            [part],
            CoilGenConfig(
                skip_sweep=False,
                sweep_path_simplification="off",
                conductor_cross_section_width=0.002,
                conductor_cross_section_height=0.001,
            ),
        )[0]
        diagnostics = part.manufacturing_diagnostics

        self.assertEqual(diagnostics["sweep_points_removed"], 0)
        self.assertEqual(diagnostics["sweep_degenerate_ring_points_removed"], 1)
        self.assertEqual(diagnostics["sweep_unrepaired_short_lateral_edge_count"], 0)
        self.assertEqual(diagnostics["short_edge_count"], 0)
        self.assertTrue(diagnostics["mesh_solid_ok"])
        self.assertTrue(diagnostics["manufacturing_ok"])

    def test_sweep_coarsens_noncollinear_micro_spur_within_chord_tolerance(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [0.0100002, 0.0000002, 0.0],
                [0.02, 0.0, 0.0],
            ],
            dtype=float,
        )
        support = normalize_mesh(
            Mesh(
                vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                faces=np.array([[0, 1, 2]], dtype=np.int64),
            )
        )
        part = CoilPart(mesh=support)
        part.wire_path = WirePath(v=points)

        part = create_swept_coil_bodies(
            [part],
            CoilGenConfig(
                skip_sweep=False,
                sweep_path_simplification="off",
                conductor_cross_section_width=0.002,
                conductor_cross_section_height=0.001,
            ),
        )[0]
        diagnostics = part.manufacturing_diagnostics

        self.assertEqual(diagnostics["sweep_degenerate_ring_points_removed"], 0)
        self.assertEqual(diagnostics["sweep_quality_coarsening_points_removed"], 1)
        self.assertEqual(diagnostics["sweep_unrepaired_short_lateral_edge_count"], 0)
        self.assertEqual(diagnostics["short_edge_count"], 0)
        self.assertTrue(diagnostics["mesh_solid_ok"])
        self.assertTrue(diagnostics["manufacturing_ok"])

    def test_sweep_preserves_spatial_curvature_during_quality_repair(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [0.01002, 0.00002, 0.0],
                [0.02, 0.001, 0.0],
            ],
            dtype=float,
        )
        support = normalize_mesh(
            Mesh(
                vertices=np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]], dtype=float),
                faces=np.array([[0, 1, 2]], dtype=np.int64),
            )
        )
        part = CoilPart(mesh=support)
        part.wire_path = WirePath(
            v=points,
            segment_geometry=[
                WirePathGeometry.FREE_3D,
                WirePathGeometry.FREE_3D,
                WirePathGeometry.OFFSET_SURFACE,
            ],
        )

        part = create_swept_coil_bodies(
            [part],
            CoilGenConfig(
                skip_sweep=False,
                conductor_cross_section_width=0.0017,
                conductor_cross_section_height=0.0017,
                sweep_path_simplification="off",
                sweep_max_chord_error=5e-5,
            ),
        )[0]
        diagnostics = part.manufacturing_diagnostics

        # A free/offset bend cannot be recovered by projecting a simplified
        # chord. Triangle-quality repair must preserve this spatial curve.
        self.assertEqual(diagnostics["sweep_quality_coarsening_points_removed"], 0)
        self.assertEqual(diagnostics["sweep_ring_points_removed"], 0)
        self.assertEqual(diagnostics["sweep_folded_quad_count"], 0)
        self.assertEqual(diagnostics["sweep_fold_repair_points_removed"], 0)
        self.assertEqual(diagnostics["sweep_ring_count"], 4)
        np.testing.assert_array_equal(part.wire_path.v, points)

    def test_sweep_does_not_coarsen_across_a_true_surface_boundary(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [0.01002, 0.00002, 0.0],
                [0.02, 0.001, 0.0],
            ],
            dtype=float,
        )
        support = normalize_mesh(
            Mesh(
                vertices=np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]], dtype=float),
                faces=np.array([[0, 1, 2]], dtype=np.int64),
            )
        )
        part = CoilPart(mesh=support)
        part.wire_path = WirePath(
            v=points,
            segment_geometry=[
                WirePathGeometry.SURFACE,
                WirePathGeometry.FREE_3D,
                WirePathGeometry.SURFACE,
            ],
        )

        part = create_swept_coil_bodies(
            [part],
            CoilGenConfig(
                skip_sweep=False,
                conductor_cross_section_width=0.0017,
                conductor_cross_section_height=0.0017,
                sweep_path_simplification="off",
                sweep_max_chord_error=5e-5,
            ),
        )[0]
        diagnostics = part.manufacturing_diagnostics

        self.assertEqual(diagnostics["sweep_quality_coarsening_points_removed"], 0)
        self.assertEqual(diagnostics["sweep_ring_count"], len(points))
        np.testing.assert_array_equal(part.wire_path.v, points)
        self.assertTrue(diagnostics["sweep_final_frame_rotation_violation"])
        self.assertFalse(diagnostics["manufacturing_ok"])

    def test_sweep_fillet_bend_inserts_arc_rings(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                [0.05, 0.05, 0.0],
            ],
            dtype=float,
        )
        mesh = normalize_mesh(
            Mesh(
                vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                faces=np.array([[0, 1, 2]], dtype=np.int64),
            )
        )
        part = CoilPart(mesh=mesh)
        part.wire_path = WirePath(v=points)
        config = CoilGenConfig(
            skip_sweep=False,
            conductor_cross_section_width=0.002,
            conductor_cross_section_height=0.001,
            sweep_bend_mode="fillet",
            sweep_min_bend_radius=0.01,
            sweep_bend_arc_angle_deg=15.0,
            sweep_frame_method="surface_constrained_parallel_transport",
        )

        part = create_swept_coil_bodies([part], config)[0]

        self.assertGreater(part.manufacturing_diagnostics["sweep_bend_points_inserted"], 0)
        self.assertGreater(part.manufacturing_diagnostics["sweep_ring_count"], len(points))
        self.assertAlmostEqual(part.manufacturing_diagnostics["sweep_min_effective_bend_radius_m"], 0.01)
        self.assertEqual(part.manufacturing_diagnostics["sweep_corner_candidate_count"], 1)
        self.assertEqual(part.manufacturing_diagnostics["sweep_corner_resolved_count"], 1)
        self.assertLess(part.manufacturing_diagnostics["sweep_ring_cross_section_area_max_relative_error"], 1e-12)
        self.assertTrue(part.manufacturing_diagnostics["manufacturing_ok"])

    def test_sweep_fillet_center_is_correct_for_non_right_angle(self) -> None:
        turn_angle = np.deg2rad(30.0)
        corner = np.array([0.05, 0.0, 0.0])
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                corner,
                corner + 0.05 * np.array([np.cos(turn_angle), np.sin(turn_angle), 0.0]),
            ],
            dtype=float,
        )
        support = normalize_mesh(
            Mesh(
                vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                faces=np.array([[0, 1, 2]], dtype=np.int64),
            )
        )
        part = CoilPart(mesh=support, wire_path=WirePath(v=points))
        required_radius = 0.01
        part = create_swept_coil_bodies(
            [part],
            CoilGenConfig(
                skip_sweep=False,
                conductor_cross_section_width=0.002,
                conductor_cross_section_height=0.001,
                sweep_bend_mode="fillet",
                sweep_min_bend_radius=required_radius,
                sweep_bend_arc_angle_deg=2.0,
            ),
        )[0]

        diagnostics = part.manufacturing_diagnostics
        self.assertEqual(diagnostics["sweep_corner_resolved_count"], 1)
        self.assertGreaterEqual(diagnostics["sweep_min_resampled_bend_radius_m"], 0.999 * required_radius)
        self.assertLessEqual(diagnostics["sweep_min_resampled_bend_radius_m"], 1.01 * required_radius)
        self.assertEqual(diagnostics["sweep_folded_quad_count"], 0)

    def test_sweep_fillet_is_independent_of_collinear_input_sampling(self) -> None:
        coarse = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                [0.05, 0.05, 0.0],
            ],
            dtype=float,
        )
        dense = np.vstack(
            (
                np.column_stack((np.linspace(0.0, 0.05, 11), np.zeros(11), np.zeros(11))),
                np.column_stack(
                    (
                        np.full(10, 0.05),
                        np.linspace(0.005, 0.05, 10),
                        np.zeros(10),
                    )
                ),
            )
        )
        support = normalize_mesh(
            Mesh(
                vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                faces=np.array([[0, 1, 2]], dtype=np.int64),
            )
        )
        config = CoilGenConfig(
            skip_sweep=False,
            conductor_cross_section_width=0.002,
            conductor_cross_section_height=0.001,
            sweep_path_simplification="off",
            sweep_bend_mode="fillet",
            sweep_min_bend_radius=0.01,
            sweep_bend_arc_angle_deg=15.0,
        )
        parts = []
        for points in (coarse, dense):
            part = CoilPart(mesh=support)
            part.wire_path = WirePath(v=points)
            parts.append(create_swept_coil_bodies([part], config)[0])

        coarse_part, dense_part = parts
        self.assertEqual(coarse_part.manufacturing_diagnostics["sweep_corner_resolved_count"], 1)
        self.assertEqual(dense_part.manufacturing_diagnostics["sweep_corner_resolved_count"], 1)
        self.assertGreater(dense_part.manufacturing_diagnostics["sweep_bend_analysis_points_removed"], 0)
        self.assertEqual(
            coarse_part.manufacturing_diagnostics["sweep_ring_count"],
            dense_part.manufacturing_diagnostics["sweep_ring_count"],
        )
        np.testing.assert_allclose(
            coarse_part.swept_body.vertices,
            dense_part.swept_body.vertices,
            rtol=0.0,
            atol=1e-12,
        )

    def test_sweep_fillet_uses_multisegment_support_and_is_reflection_invariant(self) -> None:
        support = normalize_mesh(
            Mesh(
                vertices=np.array(
                    [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
                    dtype=float,
                ),
                faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
            )
        )
        config = CoilGenConfig(
            skip_sweep=False,
            conductor_cross_section_width=0.001,
            conductor_cross_section_height=0.001,
            sweep_bend_mode="fillet",
            sweep_min_bend_radius=0.005,
            sweep_max_chord_error=1e-5,
            sweep_tangent_angle_tolerance_deg=2.0,
            sweep_corner_detection_angle_deg=15.0,
        )
        diagnostics = []
        for sign in (1.0, -1.0):
            points = np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [0.02, 0.0, 0.0],
                    [0.02002, sign * 0.001, 0.0],
                    [0.02, sign * 0.02, 0.0],
                ],
                dtype=float,
            )
            part = create_swept_coil_bodies(
                [CoilPart(mesh=support, wire_path=WirePath(v=points))],
                config,
            )[0]
            diagnostics.append(part.manufacturing_diagnostics)

        for values in diagnostics:
            self.assertEqual(values["sweep_corner_candidate_count"], 1)
            self.assertEqual(values["sweep_corner_resolved_count"], 1)
            self.assertEqual(values["sweep_bend_trim_conflict_count"], 0)
            self.assertEqual(values["sweep_bend_multisegment_support_count"], 1)
            self.assertEqual(values["sweep_bend_support_max_span_segments"], 2)
            self.assertAlmostEqual(values["sweep_bend_support_max_deviation_m"], 2e-5)
            self.assertFalse(values["sweep_min_bend_radius_violation"])
            self.assertEqual(values["sweep_folded_quad_count"], 0)
            self.assertTrue(values["manufacturing_ok"])

    def test_sweep_fillet_can_borrow_support_across_a_connector_transition(self) -> None:
        support = normalize_mesh(
            Mesh(
                vertices=np.array(
                    [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
                    dtype=float,
                ),
                faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
            )
        )
        points = np.asarray(
            [
                [-0.02, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.02, 0.0, 0.0],
                [0.02, 0.0002, 0.0],
                [0.02, 0.02, 0.0],
            ],
            dtype=float,
        )
        geometry = [
            WirePathGeometry.FREE_3D,
            WirePathGeometry.FREE_3D,
            WirePathGeometry.FREE_3D,
            WirePathGeometry.OFFSET_SURFACE,
        ]
        config = CoilGenConfig(
            skip_sweep=False,
            conductor_cross_section_width=0.001,
            conductor_cross_section_height=0.001,
            sweep_bend_mode="fillet",
            sweep_min_bend_radius=0.005,
            sweep_max_chord_error=1e-5,
            sweep_tangent_angle_tolerance_deg=2.0,
            sweep_corner_detection_angle_deg=15.0,
        )

        diagnostics = []
        for route_points, route_geometry in (
            (points, geometry),
            (points[::-1], list(reversed(geometry))),
        ):
            part = create_swept_coil_bodies(
                [
                    CoilPart(
                        mesh=support,
                        wire_path=WirePath(
                            v=route_points,
                            segment_geometry=route_geometry,
                        ),
                    )
                ],
                config,
            )[0]
            diagnostics.append(part.manufacturing_diagnostics)

        for values in diagnostics:
            self.assertEqual(values["sweep_corner_candidate_count"], 1)
            self.assertEqual(values["sweep_corner_resolved_count"], 1)
            self.assertEqual(values["sweep_bend_support_unavailable_count"], 0)
            self.assertEqual(values["sweep_bend_multisegment_support_count"], 1)
            self.assertFalse(values["sweep_min_bend_radius_violation"])
            self.assertEqual(values["sweep_folded_quad_count"], 0)
            self.assertTrue(values["manufacturing_ok"])

    def test_sweep_fillet_envelops_the_full_distributed_corner_event(self) -> None:
        support = normalize_mesh(
            Mesh(
                vertices=np.array(
                    [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
                    dtype=float,
                ),
                faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
            )
        )
        points = [np.array([-0.01, 0.0, 0.0]), np.zeros(3)]
        direction = 0.0
        for turn_deg in (15.0, 23.0, 15.0):
            direction += np.deg2rad(turn_deg)
            points.append(
                points[-1]
                + 0.0009
                * np.array(
                    [np.cos(direction), np.sin(direction), 0.0],
                    dtype=float,
                )
            )
        points.append(
            points[-1]
            + 0.01
            * np.array(
                [np.cos(direction), np.sin(direction), 0.0],
                dtype=float,
            )
        )
        points_array = np.asarray(points)
        config = CoilGenConfig(
            skip_sweep=False,
            conductor_cross_section_width=0.0005,
            conductor_cross_section_height=0.0005,
            sweep_bend_mode="fillet",
            sweep_min_bend_radius=0.0025,
            sweep_bend_arc_angle_deg=2.0,
            sweep_max_chord_error=5e-5,
            sweep_tangent_angle_tolerance_deg=0.5,
            sweep_corner_detection_angle_deg=15.0,
        )

        for sign in (1.0, -1.0):
            reflected = points_array.copy()
            reflected[:, 1] *= sign
            part = create_swept_coil_bodies(
                [
                    CoilPart(
                        mesh=support,
                        wire_path=WirePath(
                            v=reflected,
                            segment_geometry=[WirePathGeometry.SURFACE] * (len(reflected) - 1),
                        ),
                    )
                ],
                config,
            )[0]
            diagnostics = part.manufacturing_diagnostics

            self.assertEqual(diagnostics["sweep_corner_candidate_count"], 1)
            self.assertEqual(diagnostics["sweep_corner_resolved_count"], 1)
            self.assertEqual(diagnostics["sweep_bend_window_envelope_count"], 1)
            self.assertEqual(diagnostics["sweep_bend_multisegment_support_count"], 1)
            self.assertGreaterEqual(diagnostics["sweep_min_resampled_bend_radius_m"], 0.999 * 0.0025)
            self.assertFalse(diagnostics["sweep_min_bend_radius_violation"])
            self.assertEqual(diagnostics["sweep_folded_quad_count"], 0)
            self.assertTrue(diagnostics["manufacturing_ok"])

    def test_route_clearance_warning_does_not_disable_sweep_conditioning(self) -> None:
        support = normalize_mesh(
            Mesh(
                vertices=np.array(
                    [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
                    dtype=float,
                ),
                faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
            )
        )
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.02, 0.0, 0.0],
                [0.04, 0.0, 0.0],
                [0.06, 0.0, 0.0],
            ],
            dtype=float,
        )
        part = CoilPart(
            mesh=support,
            wire_path=WirePath(
                v=points,
                segment_geometry=[WirePathGeometry.FREE_3D] * 3,
            ),
        )
        part.route_diagnostics = {
            "min_wire_wire_distance": 1e-4,
            "min_wire_wire_segment_a": 0,
            "min_wire_wire_segment_b": 2,
        }

        part = create_swept_coil_bodies(
            [part],
            CoilGenConfig(
                skip_sweep=False,
                conductor_cross_section_width=0.001,
                conductor_cross_section_height=0.001,
                sweep_max_segment_length=0.002,
                sweep_self_intersection_check="capsule_fast",
            ),
        )[0]
        diagnostics = part.manufacturing_diagnostics

        self.assertTrue(diagnostics["manufacturing_route_precheck_failure"])
        self.assertEqual(
            diagnostics["manufacturing_route_precheck_failure_reasons"],
            "route_clearance_violation",
        )
        self.assertFalse(diagnostics["sweep_adaptive_sampling_skipped_due_route_precheck_failure"])
        self.assertGreater(diagnostics["sweep_adaptive_points_inserted"], 0)
        self.assertFalse(diagnostics["sweep_diagonal_optimization_skipped"])
        self.assertEqual(diagnostics["sweep_diagonal_optimization_skip_reason"], "")
        self.assertFalse(diagnostics["sweep_support_conformity_skipped_due_route_precheck_failure"])
        self.assertFalse(diagnostics["sweep_self_intersection_skipped"])
        self.assertFalse(diagnostics["sweep_self_intersection_skipped_due_route_precheck_failure"])
        self.assertEqual(diagnostics["sweep_folded_quad_count"], 0)
        self.assertEqual(diagnostics["manufacturing_failure_reasons"], "route_clearance_violation")

    def test_sweep_multisegment_support_does_not_flatten_large_route_deviation(self) -> None:
        support = normalize_mesh(
            Mesh(
                vertices=np.array(
                    [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
                    dtype=float,
                ),
                faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
            )
        )
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.02, 0.0, 0.0],
                [0.0207, 0.001, 0.0],
                [0.02, 0.02, 0.0],
            ],
            dtype=float,
        )
        part = create_swept_coil_bodies(
            [CoilPart(mesh=support, wire_path=WirePath(v=points))],
            CoilGenConfig(
                skip_sweep=False,
                conductor_cross_section_width=0.001,
                conductor_cross_section_height=0.001,
                sweep_bend_mode="fillet",
                sweep_min_bend_radius=0.005,
                sweep_max_chord_error=1e-5,
                sweep_tangent_angle_tolerance_deg=40.0,
            ),
        )[0]
        diagnostics = part.manufacturing_diagnostics

        self.assertEqual(diagnostics["sweep_corner_candidate_count"], 1)
        self.assertEqual(diagnostics["sweep_corner_resolved_count"], 0)
        self.assertEqual(diagnostics["sweep_bend_support_unavailable_count"], 1)
        self.assertEqual(diagnostics["sweep_bend_multisegment_support_count"], 0)
        self.assertTrue(diagnostics["sweep_min_bend_radius_violation"])
        self.assertFalse(diagnostics["manufacturing_ok"])

    def test_closed_multisegment_fillet_is_seam_invariant(self) -> None:
        support = normalize_mesh(
            Mesh(
                vertices=np.array(
                    [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
                    dtype=float,
                ),
                faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
            )
        )
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.04, 0.0, 0.0],
                [0.04002, 0.001, 0.0],
                [0.04, 0.04, 0.0],
                [0.0, 0.04, 0.0],
            ],
            dtype=float,
        )
        geometry = [WirePathGeometry.SURFACE] * len(points)
        config = CoilGenConfig(
            sweep_bend_mode="fillet",
            sweep_min_bend_radius=0.005,
            sweep_max_chord_error=1e-5,
            sweep_tangent_angle_tolerance_deg=2.0,
        )
        cross_section = conductor_cross_section_points(config)
        baseline: tuple[int, int, int, int] | None = None

        for shift in range(len(points)):
            shifted_points = np.roll(points, -shift, axis=0)
            shifted_geometry = geometry[shift:] + geometry[:shift]
            resolved, resolved_geometry, diagnostics = _resolve_sweep_centerline_bends(
                CoilPart(mesh=support),
                shifted_points,
                shifted_geometry,
                cross_section,
                config,
                closed=True,
            )
            summary = (
                diagnostics["sweep_corner_candidate_count"],
                diagnostics["sweep_corner_resolved_count"],
                diagnostics["sweep_bend_trim_conflict_count"],
                diagnostics["sweep_bend_multisegment_support_count"],
            )
            if baseline is None:
                baseline = summary
            self.assertEqual(summary, baseline)
            self.assertEqual(len(resolved_geometry), len(resolved))

        self.assertEqual(baseline, (4, 4, 0, 1))

    def test_sweep_corner_detection_catches_distributed_tangent_jump(self) -> None:
        def sampled_turn(turns_deg: list[float], segment_length: float) -> np.ndarray:
            direction = 0.0
            point = np.zeros(3, dtype=float)
            points = [point.copy()]
            for turn_deg in [0.0, *turns_deg]:
                direction += np.deg2rad(turn_deg)
                point = point + segment_length * np.array(
                    [np.cos(direction), np.sin(direction), 0.0],
                    dtype=float,
                )
                points.append(point.copy())
            return np.asarray(points)

        config = CoilGenConfig(
            sweep_min_bend_radius=0.0025,
            sweep_corner_detection_angle_deg=15.0,
        )
        single = sampled_turn([60.0], 0.01)
        distributed = sampled_turn([20.0, 20.0, 20.0], 0.01)
        densely_distributed = sampled_turn([18.0] * 5, 0.01)

        single_candidates = _sweep_corner_candidates(
            single,
            [WirePathGeometry.FREE_3D] * (len(single) - 1),
            0.0025,
            config,
            closed=False,
        )
        distributed_candidates = _sweep_corner_candidates(
            distributed,
            [WirePathGeometry.FREE_3D] * (len(distributed) - 1),
            0.0025,
            config,
            closed=False,
        )
        dense_candidates = _sweep_corner_candidates(
            densely_distributed,
            [WirePathGeometry.FREE_3D] * (len(densely_distributed) - 1),
            0.0025,
            config,
            closed=False,
        )

        self.assertEqual(len(single_candidates), 1)
        self.assertEqual(len(distributed_candidates), 3)
        self.assertEqual(len(dense_candidates), 5)
        self.assertTrue(all("window_radius" in candidate.detection_reasons for candidate in distributed_candidates))
        self.assertTrue(all("window_radius" in candidate.detection_reasons for candidate in dense_candidates))

    def test_sweep_corner_detection_promotes_every_undersized_sample(self) -> None:
        segment_length = 0.0005
        direction = 0.0
        point = np.zeros(3, dtype=float)
        points = [point.copy()]
        for turn_deg in [0.0, 20.0, 20.0, 20.0]:
            direction += np.deg2rad(turn_deg)
            point = point + segment_length * np.array(
                [np.cos(direction), np.sin(direction), 0.0],
                dtype=float,
            )
            points.append(point.copy())
        points_array = np.asarray(points)
        candidates = _sweep_corner_candidates(
            points_array,
            [WirePathGeometry.FREE_3D] * (len(points_array) - 1),
            0.0025,
            CoilGenConfig(sweep_min_bend_radius=0.0025),
            closed=False,
        )

        self.assertEqual(len(candidates), 3)
        self.assertTrue(all("minimum_radius" in candidate.detection_reasons for candidate in candidates))
        self.assertTrue(all(candidate.discrete_radius_m < 0.0025 for candidate in candidates))

    def test_sweep_corner_detection_is_reflection_and_closed_seam_invariant(self) -> None:
        points = np.asarray(
            [
                [-0.030, -0.010, 0.0],
                [0.000, -0.010, 0.0],
                [0.002, -0.006, 0.0],
                [0.006, -0.002, 0.0],
                [0.030, 0.000, 0.0],
                [0.008, 0.020, 0.0],
                [-0.025, 0.015, 0.0],
            ],
            dtype=float,
        )
        geometry = [WirePathGeometry.FREE_3D] * len(points)
        config = CoilGenConfig(
            sweep_min_bend_radius=0.0025,
            sweep_corner_detection_angle_deg=15.0,
        )
        baseline = _sweep_corner_candidates(
            points,
            geometry,
            0.0025,
            config,
            closed=True,
        )
        baseline_indices = {candidate.index for candidate in baseline}

        reflected = points.copy()
        reflected[:, 0] *= -1.0
        reflected_candidates = _sweep_corner_candidates(
            reflected,
            geometry,
            0.0025,
            config,
            closed=True,
        )
        self.assertEqual(
            [(candidate.index, candidate.detection_reasons) for candidate in reflected_candidates],
            [(candidate.index, candidate.detection_reasons) for candidate in baseline],
        )

        for shift in range(1, len(points)):
            shifted = np.roll(points, -shift, axis=0)
            shifted_candidates = _sweep_corner_candidates(
                shifted,
                geometry[shift:] + geometry[:shift],
                0.0025,
                config,
                closed=True,
            )
            mapped_indices = {(candidate.index + shift) % len(points) for candidate in shifted_candidates}
            self.assertEqual(mapped_indices, baseline_indices)

    def test_sweep_fillet_resolves_surface_to_free_space_corner(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                [0.05, 0.05, 0.02],
            ],
            dtype=float,
        )
        support = normalize_mesh(
            Mesh(
                vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                faces=np.array([[0, 1, 2]], dtype=np.int64),
            )
        )
        part = CoilPart(mesh=support)
        part.wire_path = WirePath(
            v=points,
            segment_geometry=[WirePathGeometry.SURFACE, WirePathGeometry.FREE_3D],
        )
        part = create_swept_coil_bodies(
            [part],
            CoilGenConfig(
                skip_sweep=False,
                conductor_cross_section_width=0.002,
                conductor_cross_section_height=0.001,
                sweep_bend_mode="fillet",
                sweep_min_bend_radius=0.01,
                sweep_bend_arc_angle_deg=10.0,
            ),
        )[0]

        diagnostics = part.manufacturing_diagnostics
        self.assertEqual(diagnostics["sweep_corner_candidate_count"], 1)
        self.assertEqual(diagnostics["sweep_corner_geometry_transition_count"], 1)
        self.assertEqual(diagnostics["sweep_corner_resolved_count"], 1)
        self.assertEqual(diagnostics["sweep_bend_infeasible_count"], 0)
        self.assertFalse(diagnostics["sweep_min_bend_radius_violation"])
        self.assertEqual(diagnostics["open_edge_count"], 0)
        self.assertEqual(diagnostics["misoriented_edge_count"], 0)

    def test_curved_surface_corner_requires_intrinsic_reroute(self) -> None:
        support = normalize_mesh(build_cylinder_mesh(0.4, 1.0, 64, 8, 1, 0, 0, 0))
        part = CoilPart(mesh=support)
        part.wire_path = WirePath(
            v=np.asarray(
                [
                    [0.0, 1.0, -0.1],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                ],
                dtype=float,
            ),
            segment_geometry=[WirePathGeometry.SURFACE, WirePathGeometry.SURFACE],
        )
        part = create_swept_coil_bodies(
            [part],
            CoilGenConfig(
                skip_sweep=False,
                conductor_cross_section_width=0.02,
                conductor_cross_section_height=0.01,
                sweep_bend_mode="fillet",
                sweep_min_bend_radius=0.05,
                sweep_corner_detection_angle_deg=15.0,
            ),
        )[0]

        diagnostics = part.manufacturing_diagnostics
        self.assertEqual(diagnostics["sweep_corner_candidate_count"], 1)
        self.assertEqual(diagnostics["sweep_surface_corner_requires_intrinsic_reroute_count"], 1)
        self.assertEqual(diagnostics["sweep_corner_resolved_count"], 0)
        self.assertTrue(diagnostics["sweep_min_bend_radius_violation"])
        self.assertIn("minimum_bend_radius_violation", diagnostics["manufacturing_failure_reasons"])

    def test_analytic_cylinder_resolves_surface_to_surface_corner(self) -> None:
        radius = 0.084
        required_bend_radius = 0.0025
        rotation_angle = np.pi / 2.0
        parameters = (0.2, radius, 42, 12, 1, 0, 0, rotation_angle)
        support = normalize_mesh(build_cylinder_mesh(*parameters))
        corner = np.array([0.0, 0.0, radius])
        circumferential_angle = 0.30
        points = np.asarray(
            [
                corner + np.array([0.0, 0.03, 0.0]),
                corner,
                [
                    radius * np.sin(circumferential_angle),
                    0.0,
                    radius * np.cos(circumferential_angle),
                ],
            ],
            dtype=float,
        )
        part = CoilPart(mesh=support)
        part.wire_path = WirePath(
            v=points,
            segment_geometry=[WirePathGeometry.SURFACE, WirePathGeometry.SURFACE],
        )
        part = create_swept_coil_bodies(
            [part],
            CoilGenConfig(
                skip_sweep=False,
                coil_mesh_file="create cylinder mesh",
                surface_is_cylinder_flag=True,
                cylinder_mesh_parameter_list=parameters,
                conductor_cross_section_width=0.0017,
                conductor_cross_section_height=0.0017,
                sweep_bend_mode="fillet",
                sweep_min_bend_radius=required_bend_radius,
                sweep_corner_detection_angle_deg=15.0,
            ),
        )[0]

        diagnostics = part.manufacturing_diagnostics
        self.assertEqual(diagnostics["sweep_surface_fillet_backend"], "analytic_cylinder")
        self.assertEqual(diagnostics["sweep_surface_corner_eligible_count"], 1)
        self.assertEqual(diagnostics["sweep_surface_corner_resolved_count"], 1)
        self.assertEqual(diagnostics["sweep_surface_corner_validation_failure_count"], 0)
        self.assertEqual(diagnostics["sweep_surface_corner_requires_intrinsic_reroute_count"], 0)
        self.assertFalse(diagnostics["sweep_min_bend_radius_violation"])
        self.assertGreaterEqual(
            diagnostics["sweep_min_resampled_bend_radius_m"],
            0.99 * required_bend_radius,
        )
        self.assertLessEqual(diagnostics["sweep_surface_fillet_radius_inflation_max"], 1.05)
        self.assertGreater(diagnostics["sweep_surface_bend_projection_max_m"], 0.0)

    def test_analytic_cylinder_retracts_faceted_surface_trace(self) -> None:
        radius = 0.084
        parameters = (0.2, radius, 42, 12, 1, 0, 0, 0)
        support = normalize_mesh(build_cylinder_mesh(*parameters))
        facet_midpoint = 0.5 * (support.vertices[0] + support.vertices[1])
        points = np.asarray(
            [
                facet_midpoint + np.array([0.0, 0.0, -0.02]),
                facet_midpoint,
                facet_midpoint + np.array([0.0, 0.0, 0.02]),
            ]
        )
        part = CoilPart(mesh=support)
        config = CoilGenConfig(
            coil_mesh_file="create cylinder mesh",
            surface_is_cylinder_flag=True,
            cylinder_mesh_parameter_list=parameters,
            sweep_bend_mode="fillet",
            sweep_min_bend_radius=0.0025,
        )

        retracted, _geometry, diagnostics = _resolve_sweep_centerline_bends(
            part,
            points,
            [WirePathGeometry.SURFACE, WirePathGeometry.SURFACE],
            conductor_cross_section_points(config),
            config,
            closed=False,
        )

        np.testing.assert_allclose(np.linalg.norm(retracted[:, :2], axis=1), radius, atol=1e-12)
        self.assertEqual(diagnostics["sweep_surface_fillet_backend"], "analytic_cylinder")
        self.assertEqual(diagnostics["sweep_analytic_surface_retraction_point_count"], 3)
        self.assertGreater(diagnostics["sweep_analytic_surface_retraction_max_m"], 0.0)

    def test_interacting_sharp_bends_block_radius_controlled_manufacturing(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.005, 0.0, 0.0],
                [0.005, 0.005, 0.0],
                [0.010, 0.005, 0.0],
            ],
            dtype=float,
        )
        support = normalize_mesh(
            Mesh(
                vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                faces=np.array([[0, 1, 2]], dtype=np.int64),
            )
        )
        part = CoilPart(mesh=support)
        part.wire_path = WirePath(v=points)
        part = create_swept_coil_bodies(
            [part],
            CoilGenConfig(
                skip_sweep=False,
                conductor_cross_section_width=0.002,
                conductor_cross_section_height=0.001,
                sweep_bend_mode="fillet",
                sweep_min_bend_radius=0.01,
            ),
        )[0]

        diagnostics = part.manufacturing_diagnostics
        self.assertEqual(diagnostics["sweep_corner_candidate_count"], 2)
        self.assertEqual(diagnostics["sweep_corner_resolved_count"], 0)
        self.assertEqual(diagnostics["sweep_bend_trim_conflict_count"], 2)
        self.assertTrue(diagnostics["sweep_min_bend_radius_violation"])
        self.assertIn("minimum_bend_radius_violation", diagnostics["manufacturing_failure_reasons"])
        self.assertFalse(diagnostics["manufacturing_ok"])

    def test_manufacturing_diagnostics_exports_and_stl_clearance_gate(self) -> None:
        result = run_coilgen(_small_y_gradient_config(skip_sweep=False))
        part = result.coil_parts[0]

        diagnostics = mesh_manufacturing_diagnostics(part.swept_body)

        # Ring repair now produces a clean closed shell for this compact
        # fixture, while the independent collision/clearance gates still reject
        # it as a physical conductor body.
        self.assertTrue(diagnostics["manufacturing_ok"])
        self.assertEqual(diagnostics["short_edge_count"], 0)
        self.assertEqual(diagnostics["open_edge_count"], 0)
        self.assertFalse(part.manufacturing_diagnostics["manufacturing_ok"])
        self.assertTrue(part.manufacturing_diagnostics["sweep_route_clearance_violation"])
        with TemporaryDirectory() as tmp:
            files = save_manufacturing_diagnostics(result, tmp, prefix="mfg")
            rows = manufacturing_diagnostics_rows(result)
            csv_text = (Path(tmp) / "mfg_manufacturing_diagnostics.csv").read_text(encoding="utf-8")

        expected_names = {"mfg_manufacturing_diagnostics.json", "mfg_manufacturing_diagnostics.csv"}
        for index, exported_part in enumerate(result.coil_parts, 1):
            if exported_part.normal_shift_provenance:
                expected_names.add(f"mfg_part{index:03d}_normal_shift_provenance.npz")
            if exported_part.joint_route_diagnostics:
                expected_names.add(f"mfg_part{index:03d}_joint_routing.json")
        self.assertEqual({path.name for path in files}, expected_names)
        self.assertEqual(len(rows), len(result.coil_parts))
        self.assertIn("manufacturing_ok", csv_text)

        part.route_diagnostics["min_wire_wire_distance"] = 0.0
        part.route_diagnostics["min_wire_wire_segment_a"] = 0
        part.route_diagnostics["min_wire_wire_segment_b"] = 2
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "not ready for STL export"):
                save_swept_bodies_as_stl(result, tmp, prefix="body", min_clearance=1e-3, fail_on_violations=True)

    def test_projection_visualization_exports(self) -> None:
        result = run_coilgen(_small_y_gradient_config(skip_sweep=True))

        with TemporaryDirectory() as tmp:
            files = save_result_visualizations(result, tmp, prefix="test")
            self.assertEqual(len(files), 7)
            self.assertTrue(any(file.name.endswith("_contour_groups_iso.svg") for file in files))
            for file in files:
                text = file.read_text(encoding="utf-8")
                self.assertIn("<svg", text)
                self.assertGreater(len(text), 200)

    def test_plot_gallery_exports_full_inspection_set(self) -> None:
        result = run_coilgen(_small_y_gradient_config(skip_sweep=True))

        with TemporaryDirectory() as tmp:
            files = save_plot_gallery(result, tmp, prefix="gallery")
            names = {file.name for file in files}

            required = {
                "gallery_part1_wire_iso.svg",
                "gallery_part1_wire_yz.svg",
                "gallery_part1_contours_iso.svg",
                "gallery_part1_stream_function_xy.svg",
                "gallery_part1_stream_function_xz.svg",
                "gallery_part1_stream_function_yz.svg",
                "gallery_part1_stream_function_iso.svg",
                "gallery_part1_current_density_xy.svg",
                "gallery_part1_current_density_xz.svg",
                "gallery_part1_current_density_yz.svg",
                "gallery_part1_current_density_iso.svg",
                "gallery_target_field_target_xy.svg",
                "gallery_target_field_target_iso.svg",
                "gallery_target_field_layout_xy.svg",
                "gallery_target_field_stream_xy.svg",
                "gallery_target_field_error_xy.svg",
                "gallery_target_field_error_iso.svg",
                "gallery_gradient_layout_magnitude_iso.svg",
                "gallery_gradient_error_dbz_dy_iso.svg",
                "gallery_relative_gradient_error_layout_vs_target_dbz_dy_iso.svg",
                "gallery_error_map_layout_vs_target_xy.svg",
                "gallery_error_map_layout_vs_stream_iso.svg",
                "gallery_relative_error_map_layout_vs_target_xy.svg",
                "gallery_relative_error_map_layout_vs_stream_iso.svg",
                "gallery_error_metrics.svg",
                "gallery_performance_summary.svg",
                "gallery_regularization_l_curve.svg",
                "gallery_route_diagnostics.csv",
                "gallery_topology_diagnostics.csv",
            }
            self.assertTrue(required.issubset(names))
            for name in required:
                file = Path(tmp) / name
                if file.suffix == ".svg":
                    text = file.read_text(encoding="utf-8")
                    self.assertIn("<svg", text)
                    self.assertGreater(len(text), 200)

    def test_matplotlib_colorbar_ticks_are_explicit_and_scientific(self) -> None:
        ticks = _colorbar_tick_values(1.0e-4, 3.0e-4)
        self.assertEqual(len(ticks), 5)
        self.assertEqual(_format_colorbar_tick(ticks[0]), "1.000e-04")
        self.assertEqual(_format_colorbar_tick(ticks[-1]), "3.000e-04")

        constant_ticks = _colorbar_tick_values(2.0e-4, 2.0e-4)
        self.assertEqual(constant_ticks.tolist(), [2.0e-4])
        self.assertEqual(_format_colorbar_tick(constant_ticks[0]), "2.000e-04")

        self.assertEqual(len(_colorbar_tick_values(float("nan"), 1.0)), 0)

    def test_matplotlib_titles_wrap_for_dense_scientific_labels(self) -> None:
        title = "Connected Wire - Target Bz Gradient dBz/dy Relative Error [%]"

        wrapped = _wrap_plot_title(title, width=24)

        self.assertIn("\n", wrapped)
        self.assertLessEqual(max(len(line) for line in wrapped.splitlines()), 24)

    def test_l_curve_regularization_scan_arrays_are_available(self) -> None:
        config = _small_y_gradient_config(skip_sweep=True)
        config.l_curve_factor_count = 7

        result = run_coilgen(config)
        scan = _regularization_scan_arrays(result)

        self.assertEqual(len(scan["factors"]), 7)
        self.assertEqual(len(scan["residuals"]), 7)
        self.assertEqual(len(scan["regularization_norms"]), 7)
        self.assertTrue(scan["valid"][scan["selected_index"]])

    def test_regularization_selection_report_and_candidate_rows_are_scientific(self) -> None:
        config = _small_y_gradient_config(skip_sweep=True)
        config.l_curve_factor_count = 7

        result = run_coilgen(config)
        report = regularization_selection_report(result)
        rows = regularization_candidate_rows(result)

        self.assertEqual(report.selection_method, "l_curve")
        self.assertEqual(report.candidate_count, 7)
        self.assertEqual(len(rows), 7)
        self.assertIn(report.knee_type, {"first_tradeoff_knee", "tradeoff_knee", "overregularized_collapse"})
        self.assertIn("residual_norm_T_per_A", rows[0])
        self.assertIn("resistance_seminorm_ohm_A", rows[0])
        self.assertTrue(any(row["selected"] for row in rows))

    def test_contour_stability_warnings_flag_zero_crossing_odd_levels(self) -> None:
        result = _minimal_result_with_suppression_points()
        result.coil_parts[0].potential_level_list = np.array([-1.0, 0.0, 1.0], dtype=float)

        warnings = contour_stability_warnings(result)

        self.assertTrue(any("odd number" in warning for warning in warnings))
        self.assertTrue(any("level at zero" in warning for warning in warnings))

        result.coil_parts[0].potential_level_list = np.array([-1.0, -0.1, 1.0], dtype=float)
        result.coil_parts[0].contour_level_diagnostics["zero_crossing_level_strategy"] = "full_range"
        warnings = contour_stability_warnings(result)
        self.assertFalse(any("odd number" in warning for warning in warnings))

    def test_geometry_factory_registry_exposes_builtins_and_custom_factory(self) -> None:
        builtins = available_geometry_factories()
        self.assertIn("create cylinder mesh", {factory.name for factory in builtins})
        self.assertIsNotNone(geometry_factory("create cylinder mesh"))

        register_geometry_factory(
            GeometryFactory(
                name="create unit test triangle mesh",
                description="single triangle for registry tests",
                parameter_attribute="none",
                builder=lambda _config: Mesh(
                    vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
                    faces=np.array([[0, 1, 2]], dtype=np.int64),
                ),
            ),
            replace=True,
        )

        mesh = load_mesh_from_config(CoilGenConfig(coil_mesh_file="create unit test triangle mesh"), Path.cwd())

        self.assertEqual(mesh.n_vertices, 3)
        self.assertEqual(mesh.n_faces, 1)
        self.assertIsNotNone(mesh.face_normals)

    def test_typed_study_result_and_scheme_registry_are_available(self) -> None:
        result = _minimal_result_with_suppression_points()
        result.input_data.field_shape_function = "y"
        study_result = study_result_from_coilgen_result("case_a", {"gap_m": 0.8}, result)
        table = DesignStudyTable.from_results([study_result])

        self.assertEqual(table.rows()[0]["parameter.gap_m"], 0.8)
        self.assertIs(table.best_by("layout_gradient_efficiency_mT_per_m_per_A"), study_result)
        self.assertIsNone(table.best_by("not_a_metric"))
        self.assertIsNotNone(stream_function_optimization_scheme("tikhonov"))
        self.assertIsNotNone(stream_function_optimization_scheme("target_error_constrained_efficiency"))
        self.assertIsNotNone(stream_function_optimization_scheme("constrained_quadratic"))
        self.assertTrue(available_stream_function_optimization_schemes())
        self.assertEqual(StudyMetric("x", 1.0, "m").as_value(), 1.0)

    def test_plot_book_export_filename_stem_is_safe(self) -> None:
        self.assertEqual(_safe_file_stem("Gradient Bz [T/(A m)]"), "Gradient_Bz_T_A_m")
        self.assertEqual(_safe_file_stem(""), "plot")

    def test_plot_book_open_alias_shows_retained_book(self) -> None:
        class FakePlotBook(MatplotlibPlotBook):
            def __init__(self) -> None:
                super().__init__(title="fake")
                self.shown_blocks: list[bool] = []
                self.selected_pages: list[int] = []

            @property
            def page_count(self) -> int:
                return 3

            def show(self, *, block: bool = False) -> None:
                self.shown_blocks.append(bool(block))

            def close(self) -> None:
                pass

            def show_page(self, index: int) -> None:
                self.selected_pages.append(int(index))

            def _export_page(self, index: int, directory: str | Path) -> Path:
                raise NotImplementedError

        book = FakePlotBook()

        book.open()
        book.open(page=2, block=True)
        book.reopen(page=1)

        self.assertEqual(book.shown_blocks, [False, True, False])
        self.assertEqual(book.selected_pages, [2, 1])

    def test_scrollable_plot_book_uses_native_figure_size(self) -> None:
        class FakeFigure:
            @staticmethod
            def get_size_inches() -> np.ndarray:
                return np.array([13.5, 38.0])

            @staticmethod
            def get_dpi() -> float:
                return 100.0

        class CompressedCanvas:
            @staticmethod
            def get_width_height() -> tuple[int, int]:
                return (900, 700)

        self.assertEqual(_native_figure_canvas_size(FakeFigure(), CompressedCanvas()), (1350, 3800))

    def test_auto_l_curve_candidate_range_uses_problem_singular_values(self) -> None:
        singular_values = np.array([0.1, 1.0, 10.0, 100.0], dtype=float)
        reduced_sensitivity = np.diag(singular_values)
        reduced_resistance = np.eye(4)
        mesh = Mesh(
            vertices=np.zeros((4, 3), dtype=float),
            faces=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        )
        reducer = BoundaryReducer(
            full_node_count=4,
            boundary_nodes=[],
            non_boundary_nodes=np.arange(4),
            expansion_matrix=np.eye(4),
        )
        problem = StreamFunctionOptimizationProblem(
            sensitivity_matrix=np.zeros((3, 4, 4), dtype=float),
            resistance_matrix=reduced_resistance,
            combined_mesh=mesh,
            reducer=reducer,
            target_z=np.ones(4, dtype=float),
            reduced_sensitivity=reduced_sensitivity,
            reduced_resistance=reduced_resistance,
        )
        config = CoilGenConfig(
            regularization_selection_method="l_curve",
            l_curve_scan_method="auto",
            tikonov_reg_factor=1e-6,
            l_curve_factor_count=17,
            l_curve_factor_min_multiplier=0.1,
            l_curve_factor_max_multiplier=10.0,
            progress_flag=False,
        )

        solution = solve_tikhonov_stream_function(problem, config)
        diagnostics = solution.diagnostics

        self.assertFalse(diagnostics["fallback"])
        self.assertEqual(len(diagnostics["candidate_factors"]), 17)
        self.assertGreater(max(diagnostics["candidate_factors"]), 1.0)
        self.assertIn("generalized singular-value", " ".join(diagnostics["candidate_notes"]))

    def test_regularization_auto_range_uses_problem_scale_with_normal_scan(self) -> None:
        singular_values = np.array([0.1, 1.0, 10.0, 100.0], dtype=float)
        reduced_sensitivity = np.diag(singular_values)
        reduced_resistance = np.eye(4)
        mesh = Mesh(
            vertices=np.zeros((4, 3), dtype=float),
            faces=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        )
        reducer = BoundaryReducer(
            full_node_count=4,
            boundary_nodes=[],
            non_boundary_nodes=np.arange(4),
            expansion_matrix=np.eye(4),
        )
        problem = StreamFunctionOptimizationProblem(
            sensitivity_matrix=np.zeros((3, 4, 4), dtype=float),
            resistance_matrix=reduced_resistance,
            combined_mesh=mesh,
            reducer=reducer,
            target_z=np.ones(4, dtype=float),
            reduced_sensitivity=reduced_sensitivity,
            reduced_resistance=reduced_resistance,
        )
        config = CoilGenConfig(
            regularization_selection_method="l_curve",
            l_curve_scan_method="normal_equations",
            regularization_auto_range=True,
            tikonov_reg_factor=1e12,
            l_curve_factor_count=17,
            l_curve_factor_min_multiplier=10.0,
            l_curve_factor_max_multiplier=100.0,
            progress_flag=False,
        )

        solution = solve_tikhonov_stream_function(problem, config)
        diagnostics = solution.diagnostics

        self.assertEqual(diagnostics["candidate_solver"], "normal_equations")
        self.assertTrue(diagnostics["regularization_auto_range"])
        self.assertIn("generalized singular-value", " ".join(diagnostics["candidate_notes"]))
        self.assertEqual(len(diagnostics["candidate_factors"]), 17)
        self.assertLess(min(diagnostics["candidate_factors"]), 1.0)
        self.assertLess(max(diagnostics["candidate_factors"]), 1.0e8)

    def test_regularization_auto_range_handles_semidefinite_resistance_metric(self) -> None:
        reduced_sensitivity = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 10.0],
                [0.0, 0.2, 0.5],
            ],
            dtype=float,
        )
        reduced_resistance = np.diag([0.0, 1.0, 4.0])
        mesh = Mesh(
            vertices=np.zeros((3, 3), dtype=float),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
        )
        reducer = BoundaryReducer(
            full_node_count=3,
            boundary_nodes=[],
            non_boundary_nodes=np.arange(3),
            expansion_matrix=np.eye(3),
        )
        problem = StreamFunctionOptimizationProblem(
            sensitivity_matrix=np.zeros((3, 3, 3), dtype=float),
            resistance_matrix=reduced_resistance,
            combined_mesh=mesh,
            reducer=reducer,
            target_z=np.array([1.0, 1.0, 0.0], dtype=float),
            reduced_sensitivity=reduced_sensitivity,
            reduced_resistance=reduced_resistance,
        )
        config = CoilGenConfig(
            regularization_selection_method="l_curve",
            l_curve_scan_method="normal_equations",
            regularization_auto_range=True,
            regularization_auto_range_min_factor=1.0e-2,
            l_curve_factor_count=17,
            progress_flag=False,
        )

        solution = solve_tikhonov_stream_function(problem, config)
        diagnostics = solution.diagnostics

        notes = " ".join(diagnostics["candidate_notes"])
        self.assertIn("generalized singular-value", notes)
        self.assertIn("R rank 2/3", notes)
        self.assertGreaterEqual(min(diagnostics["candidate_factors"]), 1.0e-2)

    def test_regularization_auto_range_rejects_numerical_singular_tail(self) -> None:
        singular_values = np.array([1.0, 1.0e-6], dtype=float)
        reduced_sensitivity = np.diag(singular_values)
        reduced_resistance = np.eye(2)
        mesh = Mesh(
            vertices=np.zeros((2, 3), dtype=float),
            faces=np.array([[0, 1, 1]], dtype=np.int64),
        )
        reducer = BoundaryReducer(
            full_node_count=2,
            boundary_nodes=[],
            non_boundary_nodes=np.arange(2),
            expansion_matrix=np.eye(2),
        )
        problem = StreamFunctionOptimizationProblem(
            sensitivity_matrix=np.zeros((3, 2, 2), dtype=float),
            resistance_matrix=reduced_resistance,
            combined_mesh=mesh,
            reducer=reducer,
            target_z=np.array([0.0, 1.0], dtype=float),
            reduced_sensitivity=reduced_sensitivity,
            reduced_resistance=reduced_resistance,
        )
        config = CoilGenConfig(
            regularization_selection_method="l_curve",
            l_curve_scan_method="normal_equations",
            regularization_auto_range=True,
            l_curve_factor_count=17,
            progress_flag=False,
        )

        solution = solve_tikhonov_stream_function(problem, config)
        diagnostics = solution.diagnostics

        self.assertIn("retained 1/2 modes", " ".join(diagnostics["candidate_notes"]))
        self.assertGreaterEqual(min(diagnostics["candidate_factors"]), 0.01)
        self.assertLess(diagnostics["estimated_selected_contour_step_A"], 1.0)

    def test_regularization_auto_range_contour_step_guard_raises_lambda(self) -> None:
        singular_values = np.array([1.0, 1.0e-6], dtype=float)
        reduced_sensitivity = np.diag(singular_values)
        reduced_resistance = np.eye(2)
        mesh = Mesh(
            vertices=np.zeros((2, 3), dtype=float),
            faces=np.array([[0, 1, 1]], dtype=np.int64),
        )
        reducer = BoundaryReducer(
            full_node_count=2,
            boundary_nodes=[],
            non_boundary_nodes=np.arange(2),
            expansion_matrix=np.eye(2),
        )
        problem = StreamFunctionOptimizationProblem(
            sensitivity_matrix=np.zeros((3, 2, 2), dtype=float),
            resistance_matrix=reduced_resistance,
            combined_mesh=mesh,
            reducer=reducer,
            target_z=np.array([0.0, 1.0], dtype=float),
            reduced_sensitivity=reduced_sensitivity,
            reduced_resistance=reduced_resistance,
        )
        config = CoilGenConfig(
            regularization_selection_method="l_curve",
            l_curve_scan_method="normal_equations",
            regularization_auto_range=True,
            regularization_auto_range_min_singular_relative=1.0e-12,
            regularization_auto_range_target_projection_relative=1.0e-12,
            regularization_auto_range_lower_padding=1.0e-3,
            regularization_auto_range_upper_padding=1.0e3,
            regularization_auto_range_max_contour_step_A=10.0,
            l_curve_factor_count=21,
            levels=10,
            pot_offset_factor=0.5,
            progress_flag=False,
        )

        solution = solve_tikhonov_stream_function(problem, config)
        diagnostics = solution.diagnostics

        self.assertLessEqual(diagnostics["estimated_selected_contour_step_A"], 10.0)
        self.assertGreater(diagnostics["selected_factor"], 1.0e-6)
        self.assertIn("contour-step guard raised selected factor", " ".join(diagnostics["candidate_notes"]))

    def test_stream_function_smoothness_penalty_records_diagnostics(self) -> None:
        config = _small_y_gradient_config(skip_sweep=True)
        config.stream_function_smoothness_factor = 0.25

        result = run_coilgen(config)

        self.assertTrue(result.regularization_diagnostics["stream_function_smoothness_enabled"])
        self.assertAlmostEqual(result.regularization_diagnostics["stream_function_smoothness_factor"], 0.25)
        self.assertGreater(result.regularization_diagnostics["stream_function_smoothness_matrix_scale"], 0.0)

    def test_target_error_constrained_efficiency_solver_respects_error_budget(self) -> None:
        config = _small_y_gradient_config(skip_sweep=True)
        config.sf_opt_method = "target_error_constrained_efficiency"
        config.target_error_tolerance_percent = 25.0
        config.target_error_constraint_factor_count = 25
        config.progress_flag = False

        result = run_coilgen(config)
        diagnostics = result.regularization_diagnostics

        self.assertEqual(diagnostics["stream_function_scheme"], "target_error_constrained_efficiency")
        self.assertTrue(diagnostics["target_error_constraint_met"])
        self.assertLessEqual(diagnostics["target_error_constraint_relative_residual_percent"], 25.0 + 1e-6)
        self.assertGreater(diagnostics["selected_lambda"], 0.0)

    def test_target_error_constrained_efficiency_solver_respects_dsv_std_budget(self) -> None:
        mesh = Mesh(
            vertices=np.zeros((2, 3), dtype=float),
            faces=np.array([[0, 1, 1]], dtype=np.int64),
        )
        reducer = BoundaryReducer(
            full_node_count=2,
            boundary_nodes=[],
            non_boundary_nodes=np.arange(2),
            expansion_matrix=np.eye(2),
        )
        problem = StreamFunctionOptimizationProblem(
            sensitivity_matrix=np.zeros((3, 2, 2), dtype=float),
            resistance_matrix=np.eye(2),
            combined_mesh=mesh,
            reducer=reducer,
            target_z=np.array([1.0, -1.0], dtype=float),
            reduced_sensitivity=np.eye(2),
            reduced_resistance=np.eye(2),
        )
        config = CoilGenConfig(
            sf_opt_method="target_error_constrained_efficiency",
            target_error_constraint_metric="std",
            target_error_std_limit_T=0.2,
            target_error_constraint_factor_count=31,
            progress_flag=False,
        )

        solution = solve_stream_function_problem(problem, config)
        diagnostics = solution.diagnostics

        self.assertEqual(diagnostics["target_error_constraint_metric"], "std")
        self.assertEqual(diagnostics["dsv_constraint_metric"], "std")
        self.assertTrue(diagnostics["target_error_constraint_met"])
        self.assertLessEqual(diagnostics["dsv_constraint_std_T"], 0.2 + 1.0e-8)
        self.assertAlmostEqual(diagnostics["dsv_constraint_allowed_std_T"], 0.2)
        self.assertAlmostEqual(diagnostics["target_error_allowed_field_metric_T"], 0.2)
        self.assertLessEqual(diagnostics["dsv_constraint_residual_norm_T"], np.sqrt(2.0) * 0.2 + 1.0e-8)

    def test_dsv_std_constraint_removes_constant_residual_offset(self) -> None:
        mesh = Mesh(
            vertices=np.zeros((1, 3), dtype=float),
            faces=np.array([[0, 0, 0]], dtype=np.int64),
        )
        reducer = BoundaryReducer(
            full_node_count=1,
            boundary_nodes=[],
            non_boundary_nodes=np.arange(1),
            expansion_matrix=np.eye(1),
        )
        problem = StreamFunctionOptimizationProblem(
            sensitivity_matrix=np.zeros((3, 2, 1), dtype=float),
            resistance_matrix=np.eye(1),
            combined_mesh=mesh,
            reducer=reducer,
            target_z=np.array([10.0, 11.0], dtype=float),
            reduced_sensitivity=np.zeros((2, 1), dtype=float),
            reduced_resistance=np.eye(1),
        )
        config = CoilGenConfig(
            sf_opt_method="target_error_constrained_efficiency",
            target_error_constraint_metric="std",
            target_error_std_limit_T=0.5,
            progress_flag=False,
        )
        blocks = _target_error_constraint_blocks(problem, config)

        diagnostics = _residual_constraint_diagnostics(problem, np.zeros(1), blocks, config)

        self.assertTrue(diagnostics["dsv_constraint_met"])
        self.assertAlmostEqual(diagnostics["dsv_constraint_std_T"], 0.5)
        self.assertAlmostEqual(diagnostics["dsv_constraint_allowed_std_T"], 0.5)
        self.assertAlmostEqual(diagnostics["dsv_constraint_mean_T"], 0.0)
        self.assertGreater(problem.residual_norm(np.zeros(1)), diagnostics["dsv_constraint_residual_norm_T"])

    def test_gradient_linearity_constraints_use_requested_gradient_radius_budget(self) -> None:
        axis = np.array([-0.08, 0.08], dtype=float)
        coords = np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
        target_bz = 0.01 * coords[:, 0]
        reduced_sensitivity = np.column_stack([np.ones(coords.shape[0]), coords])
        mesh = Mesh(
            vertices=np.zeros((4, 3), dtype=float),
            faces=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        )
        reducer = BoundaryReducer(
            full_node_count=4,
            boundary_nodes=[],
            non_boundary_nodes=np.arange(4),
            expansion_matrix=np.eye(4),
        )
        problem = StreamFunctionOptimizationProblem(
            sensitivity_matrix=np.zeros((3, coords.shape[0], 4), dtype=float),
            resistance_matrix=np.eye(4),
            combined_mesh=mesh,
            reducer=reducer,
            target_z=target_bz,
            reduced_sensitivity=reduced_sensitivity,
            reduced_resistance=np.eye(4),
            unweighted_target_z=target_bz,
            unweighted_reduced_sensitivity=reduced_sensitivity,
            target_coords=coords,
            target_weights=np.ones(coords.shape[0], dtype=float),
            target_suppressed=np.zeros(coords.shape[0], dtype=bool),
        )
        config = CoilGenConfig(
            target_error_tolerance_percent=100.0,
            target_gradient_strength=0.01,
            linearity_reference_radius_m=0.08,
            linearity_rms_tolerance_percent=1.0,
            linearity_peak_tolerance_percent=5.0,
            linearity_cross_gradient_tolerance_percent=1.0,
            gradient_strength_tolerance_percent=1.0,
        )

        blocks = _target_error_constraint_blocks(problem, config)
        by_name = {block.name: block for block in blocks}
        diagnostics = _residual_constraint_diagnostics(problem, np.array([0.0, 0.01, 0.0, 0.0]), blocks, config)

        self.assertIn("linearity_rms", by_name)
        self.assertIn("linearity_peak", by_name)
        self.assertIn("gradient_strength", by_name)
        self.assertIn("cross_gradient", by_name)
        self.assertAlmostEqual(by_name["linearity_rms"].field_limit_T, 8.0e-6)
        self.assertAlmostEqual(by_name["linearity_rms"].allowed_residual, np.sqrt(coords.shape[0]) * 8.0e-6)
        self.assertAlmostEqual(by_name["linearity_peak"].allowed_residual, 4.0e-5)
        self.assertAlmostEqual(by_name["gradient_strength"].allowed_residual, 1.0e-4)
        self.assertAlmostEqual(by_name["cross_gradient"].allowed_residual, 1.0e-4)
        self.assertTrue(diagnostics["linearity_rms_constraint_met"])
        self.assertTrue(diagnostics["linearity_peak_constraint_met"])
        self.assertTrue(diagnostics["gradient_strength_constraint_met"])
        self.assertTrue(diagnostics["cross_gradient_constraint_met"])

    def test_gradient_linearity_report_uses_layout_operating_current(self) -> None:
        axis = np.array([-0.08, 0.0, 0.08], dtype=float)
        coords = np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
        target_b = np.zeros((coords.shape[0], 3), dtype=float)
        target_b[:, 2] = 0.01 * coords[:, 0]
        target = TargetField(
            coords=coords,
            b=target_b,
            weights=np.ones(coords.shape[0], dtype=float),
            suppressed=np.zeros(coords.shape[0], dtype=bool),
            target_field_group_inds=np.ones(coords.shape[0], dtype=np.int64),
        )
        config = CoilGenConfig(
            target_gradient_strength=0.01,
            linearity_reference_radius_m=0.08,
            linearity_rms_tolerance_percent=1.0,
            linearity_peak_tolerance_percent=5.0,
            linearity_cross_gradient_tolerance_percent=1.0,
            gradient_strength_tolerance_percent=1.0,
        )
        fitted_layout_current = 50.0
        stream_function_current = 0.14
        field_per_amp = target_b / fitted_layout_current
        result = SimpleNamespace(
            field_layout_per1Amp=field_per_amp,
            field_by_layout=field_per_amp * stream_function_current,
            needed_current_layout=stream_function_current,
            input_data=config,
            layout_gradient={
                "achieved_gradient": 0.01 / fitted_layout_current,
                "desired_gradient": 0.01,
            },
        )

        bad_summary = gradient_linearity_summary(result.field_by_layout, target, config)
        corrected_summary = gradient_linearity_summary(_gradient_linearity_field_for_result(result), target, config)

        self.assertEqual(bad_summary["linearity_status"], "FAIL")
        self.assertGreater(bad_summary["linearity_gradient_strength_error_percent"], 99.0)
        self.assertEqual(corrected_summary["linearity_status"], "PASS")
        self.assertAlmostEqual(corrected_summary["linearity_gradient_strength_error_percent"], 0.0)

    def test_target_error_solver_expands_below_too_large_scan_minimum(self) -> None:
        mesh = Mesh(
            vertices=np.zeros((1, 3), dtype=float),
            faces=np.array([[0, 0, 0]], dtype=np.int64),
        )
        reducer = BoundaryReducer(
            full_node_count=1,
            boundary_nodes=[],
            non_boundary_nodes=np.arange(1),
            expansion_matrix=np.eye(1),
        )
        problem = StreamFunctionOptimizationProblem(
            sensitivity_matrix=np.zeros((3, 1, 1), dtype=float),
            resistance_matrix=np.eye(1),
            combined_mesh=mesh,
            reducer=reducer,
            target_z=np.array([1.0], dtype=float),
            reduced_sensitivity=np.eye(1),
            reduced_resistance=np.eye(1),
        )
        config = CoilGenConfig(
            sf_opt_method="target_error_constrained_efficiency",
            tikonov_reg_factor=100.0,
            target_error_tolerance_percent=5.0,
            target_error_constraint_factor_count=5,
            target_error_constraint_factor_min_multiplier=10.0,
            target_error_constraint_factor_max_multiplier=1.0e4,
            progress_flag=False,
        )

        solution = solve_stream_function_problem(problem, config)
        diagnostics = solution.diagnostics

        self.assertTrue(diagnostics["target_error_constraint_met"])
        self.assertLess(diagnostics["selected_lambda"], 1.0)
        self.assertIn("expanded below configured scan minimum", diagnostics["target_error_constraint_selection_reason"])

    def test_target_error_auto_range_uses_problem_scale_before_expansion(self) -> None:
        mesh = Mesh(
            vertices=np.zeros((1, 3), dtype=float),
            faces=np.array([[0, 0, 0]], dtype=np.int64),
        )
        reducer = BoundaryReducer(
            full_node_count=1,
            boundary_nodes=[],
            non_boundary_nodes=np.arange(1),
            expansion_matrix=np.eye(1),
        )
        problem = StreamFunctionOptimizationProblem(
            sensitivity_matrix=np.zeros((3, 1, 1), dtype=float),
            resistance_matrix=np.eye(1),
            combined_mesh=mesh,
            reducer=reducer,
            target_z=np.array([1.0], dtype=float),
            reduced_sensitivity=np.eye(1),
            reduced_resistance=np.eye(1),
        )
        config = CoilGenConfig(
            sf_opt_method="target_error_constrained_efficiency",
            regularization_auto_range=True,
            tikonov_reg_factor=1.0e12,
            target_error_tolerance_percent=5.0,
            target_error_constraint_factor_count=17,
            target_error_constraint_factor_min_multiplier=10.0,
            target_error_constraint_factor_max_multiplier=1.0e4,
            progress_flag=False,
        )

        solution = solve_stream_function_problem(problem, config)
        diagnostics = solution.diagnostics

        self.assertTrue(diagnostics["regularization_auto_range"])
        self.assertTrue(diagnostics["target_error_constraint_met"])
        self.assertLess(max(diagnostics["candidate_lambdas"]), 1.0e8)
        self.assertNotIn(
            "expanded below configured scan minimum", diagnostics["target_error_constraint_selection_reason"]
        )

    def test_target_error_auto_range_min_factor_prevents_underregularized_expansion(self) -> None:
        mesh = Mesh(
            vertices=np.zeros((1, 3), dtype=float),
            faces=np.array([[0, 0, 0]], dtype=np.int64),
        )
        reducer = BoundaryReducer(
            full_node_count=1,
            boundary_nodes=[],
            non_boundary_nodes=np.arange(1),
            expansion_matrix=np.eye(1),
        )
        problem = StreamFunctionOptimizationProblem(
            sensitivity_matrix=np.zeros((3, 1, 1), dtype=float),
            resistance_matrix=np.eye(1),
            combined_mesh=mesh,
            reducer=reducer,
            target_z=np.array([1.0], dtype=float),
            reduced_sensitivity=np.eye(1),
            reduced_resistance=np.eye(1),
        )
        config = CoilGenConfig(
            sf_opt_method="target_error_constrained_efficiency",
            regularization_auto_range=True,
            regularization_auto_range_min_factor=1.0,
            target_error_tolerance_percent=5.0,
            target_error_constraint_factor_count=9,
            target_error_auto_retry_enabled=False,
            progress_flag=False,
        )

        solution = solve_stream_function_problem(problem, config)
        diagnostics = solution.diagnostics

        self.assertFalse(diagnostics["target_error_constraint_met"])
        self.assertGreaterEqual(diagnostics["selected_lambda"], 1.0)
        self.assertIn("regularization_auto_range_min_factor prevented expansion", diagnostics["fallback_reason"])

    def test_target_error_auto_retry_removes_auto_range_floor_to_meet_spec(self) -> None:
        mesh = Mesh(
            vertices=np.zeros((1, 3), dtype=float),
            faces=np.array([[0, 0, 0]], dtype=np.int64),
        )
        reducer = BoundaryReducer(
            full_node_count=1,
            boundary_nodes=[],
            non_boundary_nodes=np.arange(1),
            expansion_matrix=np.eye(1),
        )
        problem = StreamFunctionOptimizationProblem(
            sensitivity_matrix=np.zeros((3, 1, 1), dtype=float),
            resistance_matrix=np.eye(1),
            combined_mesh=mesh,
            reducer=reducer,
            target_z=np.array([1.0], dtype=float),
            reduced_sensitivity=np.eye(1),
            reduced_resistance=np.eye(1),
        )
        config = CoilGenConfig(
            sf_opt_method="target_error_constrained_efficiency",
            regularization_auto_range=True,
            regularization_auto_range_min_factor=1.0,
            target_error_tolerance_percent=5.0,
            target_error_constraint_factor_count=9,
            target_error_auto_retry_enabled=True,
            target_error_auto_retry_max_reruns=10,
            progress_flag=False,
        )

        solution = solve_stream_function_problem(problem, config)
        diagnostics = solution.diagnostics

        self.assertTrue(diagnostics["target_error_constraint_met"])
        self.assertGreaterEqual(diagnostics["target_error_auto_retry_count"], 1)
        self.assertEqual(diagnostics["target_error_auto_retry_selected_rerun"], 1)
        self.assertLess(diagnostics["selected_lambda"], 1.0)
        self.assertIn("automatic rerun", diagnostics["target_error_auto_retry_reason"])

    def test_target_error_auto_range_contour_step_guard_preserves_error_budget(self) -> None:
        mesh = Mesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
            faces=np.array([[0, 1, 1]], dtype=np.int64),
        )
        reducer = BoundaryReducer(
            full_node_count=2,
            boundary_nodes=[],
            non_boundary_nodes=np.arange(2),
            expansion_matrix=np.eye(2),
        )
        problem = StreamFunctionOptimizationProblem(
            sensitivity_matrix=np.zeros((3, 2, 2), dtype=float),
            resistance_matrix=np.eye(2),
            combined_mesh=mesh,
            reducer=reducer,
            target_z=np.array([1.0, 0.0], dtype=float),
            reduced_sensitivity=np.diag([1.0e-6, 1.0]),
            reduced_resistance=np.eye(2),
        )
        config = CoilGenConfig(
            sf_opt_method="target_error_constrained_efficiency",
            regularization_auto_range=True,
            regularization_auto_range_min_singular_relative=1.0e-12,
            regularization_auto_range_target_projection_relative=1.0e-12,
            regularization_auto_range_lower_padding=1.0e-2,
            regularization_auto_range_upper_padding=1.0e4,
            regularization_auto_range_max_contour_step_A=10.0,
            target_error_tolerance_percent=5.0,
            target_error_constraint_factor_count=25,
            levels=10,
            pot_offset_factor=0.5,
            progress_flag=False,
        )

        solution = solve_stream_function_problem(problem, config)
        diagnostics = solution.diagnostics

        notes = " ".join(diagnostics["candidate_notes"])
        self.assertTrue(diagnostics["target_error_constraint_met"])
        self.assertTrue(diagnostics["target_error_contour_step_guard_relaxed_for_constraint"])
        self.assertGreater(diagnostics["estimated_selected_contour_step_A"], 10.0)
        self.assertIn("contour-step guard", notes)
        self.assertIn("not enforced", notes)

    def test_constrained_quadratic_solver_respects_error_budget_or_fallback(self) -> None:
        mesh = Mesh(
            vertices=np.zeros((2, 3), dtype=float),
            faces=np.array([[0, 1, 1]], dtype=np.int64),
        )
        reducer = BoundaryReducer(
            full_node_count=2,
            boundary_nodes=[],
            non_boundary_nodes=np.arange(2),
            expansion_matrix=np.eye(2),
        )
        problem = StreamFunctionOptimizationProblem(
            sensitivity_matrix=np.zeros((3, 2, 2), dtype=float),
            resistance_matrix=np.eye(2),
            combined_mesh=mesh,
            reducer=reducer,
            target_z=np.array([1.0, -0.5], dtype=float),
            reduced_sensitivity=np.eye(2),
            reduced_resistance=np.eye(2),
        )
        config = CoilGenConfig(
            sf_opt_method="constrained_quadratic",
            target_error_tolerance_percent=1.0,
            target_error_constraint_factor_count=21,
            progress_flag=False,
        )

        solution = solve_stream_function_problem(problem, config)
        diagnostics = solution.diagnostics

        self.assertEqual(diagnostics["stream_function_scheme"], "constrained_quadratic")
        self.assertTrue(diagnostics["target_error_constraint_met"])
        self.assertLessEqual(diagnostics["target_error_constraint_relative_residual_percent"], 1.0 + 1e-6)
        self.assertIn(diagnostics["constrained_solver_backend"], {"cvxpy", "tikhonov_scan_fallback"})

    def test_constrained_quadratic_solver_respects_contour_step_guard(self) -> None:
        mesh = Mesh(
            vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
            faces=np.array([[0, 1, 1]], dtype=np.int64),
        )
        reducer = BoundaryReducer(
            full_node_count=2,
            boundary_nodes=[],
            non_boundary_nodes=np.arange(2),
            expansion_matrix=np.eye(2),
        )
        problem = StreamFunctionOptimizationProblem(
            sensitivity_matrix=np.zeros((3, 2, 2), dtype=float),
            resistance_matrix=np.eye(2),
            combined_mesh=mesh,
            reducer=reducer,
            target_z=np.array([1.0, -1.0], dtype=float),
            reduced_sensitivity=np.eye(2),
            reduced_resistance=np.eye(2),
        )
        config = CoilGenConfig(
            sf_opt_method="constrained_quadratic",
            target_error_tolerance_percent=50.0,
            regularization_auto_range_max_contour_step_A=1.0,
            levels=2,
            pot_offset_factor=0.0,
            contourability_repair_enabled=False,
            progress_flag=False,
        )

        solution = solve_stream_function_problem(problem, config)
        diagnostics = solution.diagnostics

        self.assertTrue(diagnostics["target_error_constraint_met"])
        if diagnostics["constrained_solver_backend"] == "cvxpy":
            self.assertTrue(diagnostics["contour_step_constraint_enabled"])
            self.assertLessEqual(diagnostics["estimated_selected_contour_step_A"], 1.0 + 1e-5)

    def test_constraint_diagnostics_share_solver_roundoff_tolerance(self) -> None:
        mesh = Mesh(
            vertices=np.zeros((1, 3), dtype=float),
            faces=np.array([[0, 0, 0]], dtype=np.int64),
        )
        reducer = BoundaryReducer(
            full_node_count=1,
            boundary_nodes=[],
            non_boundary_nodes=np.arange(1),
            expansion_matrix=np.eye(1),
        )
        problem = StreamFunctionOptimizationProblem(
            sensitivity_matrix=np.zeros((3, 1, 1), dtype=float),
            resistance_matrix=np.eye(1),
            combined_mesh=mesh,
            reducer=reducer,
            target_z=np.array([0.0], dtype=float),
            reduced_sensitivity=np.eye(1),
            reduced_resistance=np.eye(1),
        )
        block = ResidualConstraintBlock(
            name="dsv",
            matrix=np.eye(1),
            target=np.zeros(1, dtype=float),
            allowed_residual=1.0,
            reference_norm=20.0,
            relative_tolerance_percent=5.0,
            absolute_tolerance=1.0,
        )
        diagnostics = _residual_constraint_diagnostics(
            problem,
            np.array([1.0 + 5.0e-9], dtype=float),
            (block,),
            CoilGenConfig(sf_opt_method="constrained_quadratic"),
        )

        self.assertTrue(diagnostics["target_error_constraint_met"])
        self.assertTrue(diagnostics["dsv_constraint_met"])

    def test_fixed_error_solver_uses_separate_dsv_and_suppression_constraints(self) -> None:
        mesh = Mesh(
            vertices=np.zeros((2, 3), dtype=float),
            faces=np.array([[0, 1, 1]], dtype=np.int64),
        )
        reducer = BoundaryReducer(
            full_node_count=2,
            boundary_nodes=[],
            non_boundary_nodes=np.arange(2),
            expansion_matrix=np.eye(2),
        )
        problem = StreamFunctionOptimizationProblem(
            sensitivity_matrix=np.zeros((3, 2, 2), dtype=float),
            resistance_matrix=np.eye(2),
            combined_mesh=mesh,
            reducer=reducer,
            target_z=np.array([1.0, 0.0], dtype=float),
            reduced_sensitivity=np.eye(2),
            reduced_resistance=np.eye(2),
            unweighted_target_z=np.array([1.0, 0.0], dtype=float),
            unweighted_reduced_sensitivity=np.eye(2),
            target_coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float),
            target_weights=np.array([1.0, 1000.0], dtype=float),
            target_suppressed=np.array([False, True], dtype=bool),
        )
        config = CoilGenConfig(
            sf_opt_method="constrained_quadratic",
            target_error_constraint_mode="separate",
            target_error_tolerance_percent=2.0,
            suppression_error_tolerance_percent=1.0,
            suppression_field_rms_limit_T=0.2,
            target_error_constraint_factor_count=21,
            progress_flag=False,
        )

        solution = solve_stream_function_problem(problem, config)
        diagnostics = solution.diagnostics

        self.assertEqual(diagnostics["target_error_constraint_mode"], "separate")
        self.assertTrue(diagnostics["target_error_constraint_met"])
        self.assertTrue(diagnostics["dsv_constraint_met"])
        self.assertTrue(diagnostics["suppression_constraint_enabled"])
        self.assertTrue(diagnostics["suppression_constraint_met"])
        self.assertLessEqual(diagnostics["dsv_constraint_relative_residual_percent"], 2.0 + 1e-6)
        self.assertLessEqual(diagnostics["suppression_constraint_relative_residual_percent"], 1.0 + 1e-6)
        self.assertAlmostEqual(diagnostics["suppression_field_rms_limit_T"], 0.2)
        self.assertLessEqual(diagnostics["suppression_field_rms_T"], 0.2 + 1e-9)

    def test_smoothed_l_curve_knee_ignores_underregularized_local_wiggle(self) -> None:
        lambdas = np.logspace(-4.0, 4.0, 21)
        residual_svg_y = np.array(
            [
                424.000,
                414.281,
                377.891,
                336.283,
                302.200,
                279.784,
                266.008,
                261.292,
                252.365,
                226.330,
                197.282,
                183.227,
                178.022,
                165.232,
                145.104,
                136.669,
                132.152,
                113.240,
                92.224,
                85.319,
                84.000,
            ],
            dtype=float,
        )
        resistance_svg_y = np.array(
            [
                84.000,
                84.069,
                84.440,
                86.557,
                94.791,
                110.740,
                129.405,
                133.897,
                135.742,
                140.651,
                157.808,
                184.671,
                193.779,
                202.437,
                226.698,
                246.689,
                251.355,
                261.296,
                295.329,
                355.044,
                424.000,
            ],
            dtype=float,
        )
        log_residual = (424.0 - residual_svg_y) / 340.0
        log_resistance = (424.0 - resistance_svg_y) / 340.0

        selected_index, curvature, fallback_reason = _select_l_curve_knee(
            lambdas,
            np.exp(log_residual),
            np.exp(log_resistance),
            smoothing_window=5,
        )

        self.assertGreaterEqual(selected_index, 8)
        self.assertLessEqual(selected_index, 14)
        self.assertFalse(fallback_reason)
        self.assertTrue(np.isfinite(curvature[selected_index]))

    def test_l_curve_knee_rejects_flat_plateau_curvature_spike(self) -> None:
        lambdas = np.logspace(-3.0, 1.0, 161)
        log_lambdas = np.log(lambdas)
        transition = 1.0 / (1.0 + np.exp(-7.0 * (log_lambdas - np.log(3.0))))
        log_residual = 1.0e-6 + 1.2 * transition
        log_resistance = 1.0 - 0.95 * transition
        plateau_wiggle = np.exp(-0.5 * ((log_lambdas - np.log(0.0014)) / 0.018) ** 2)
        log_residual += 0.0012 * plateau_wiggle
        log_resistance -= 0.0012 * plateau_wiggle

        selected_index, curvature, fallback_reason = _select_l_curve_knee(
            lambdas,
            np.exp(log_residual),
            np.exp(log_resistance),
            smoothing_window=5,
        )

        raw_curvature_index = int(np.nanargmax(curvature))
        self.assertLess(lambdas[raw_curvature_index], 0.01)
        self.assertGreater(lambdas[selected_index], 1.0)
        self.assertFalse(fallback_reason)
        self.assertTrue(np.isfinite(curvature[selected_index]))

    def test_l_curve_knee_keeps_late_broad_knee_when_no_earlier_tradeoff(self) -> None:
        lambdas = np.logspace(-4.0, 6.0, 161)
        log_lambdas = np.log(lambdas)
        residual_transition = 1.0 / (1.0 + np.exp(-3.0 * (log_lambdas - np.log(1.1e3))))
        resistance_transition = 1.0 / (1.0 + np.exp(-1.5 * (log_lambdas - np.log(5.0e4))))
        log_residual = 0.08 * residual_transition
        log_resistance = 1.0 - 0.95 * resistance_transition

        selected_index, curvature, fallback_reason = _select_l_curve_knee(
            lambdas,
            np.exp(log_residual),
            np.exp(log_resistance),
            smoothing_window=15,
        )

        self.assertLess(lambdas[selected_index], 8.0e3)
        self.assertGreater(lambdas[selected_index], 2.0e3)
        self.assertFalse(fallback_reason)
        self.assertTrue(np.isfinite(curvature[selected_index]))

    def test_l_curve_knee_prefers_first_tradeoff_over_late_current_collapse(self) -> None:
        lambdas = np.logspace(-4.0, 6.0, 161)
        log_lambdas = np.log(lambdas)
        useful_tradeoff = 1.0 / (1.0 + np.exp(-2.8 * (log_lambdas - np.log(600.0))))
        current_collapse = 1.0 / (1.0 + np.exp(-1.3 * (log_lambdas - np.log(8.0e4))))
        log_residual = 0.105 * useful_tradeoff
        log_resistance = 1.0 - 0.20 * useful_tradeoff - 5.50 * current_collapse

        selected_index, curvature, fallback_reason = _select_l_curve_knee(
            lambdas,
            np.exp(log_residual),
            np.exp(log_resistance),
            smoothing_window=15,
        )

        self.assertGreater(lambdas[selected_index], 300.0)
        self.assertLess(lambdas[selected_index], 2.5e3)
        self.assertFalse(fallback_reason)
        self.assertTrue(np.isfinite(curvature[selected_index]))

    def test_l_curve_knee_rejects_scan_without_dynamic_range(self) -> None:
        lambdas = np.logspace(-3.0, 1.0, 81)
        residuals = 1.0 + 1.0e-8 * np.linspace(0.0, 1.0, len(lambdas))
        resistance = 2.0 - 1.0e-8 * np.linspace(0.0, 1.0, len(lambdas))

        selected_index, curvature, fallback_reason = _select_l_curve_knee(
            lambdas,
            residuals,
            resistance,
            smoothing_window=5,
        )

        self.assertEqual(selected_index, -1)
        self.assertIn("insufficient dynamic range", fallback_reason)
        self.assertEqual(len(curvature), len(lambdas))

    def test_field_gradient_and_error_maps_are_finite(self) -> None:
        result = run_coilgen(_small_y_gradient_config(skip_sweep=True))
        count = len(result.target_field.coords)

        field_maps = field_scalar_maps(result)
        error_maps = field_error_scalar_maps(result)
        relative_error_maps = field_relative_error_scalar_maps(result)
        gradient_maps = gradient_component_scalar_maps(result, component="magnitude")
        gradient_y_maps = gradient_component_scalar_maps(result, component="dbz_dy")
        relative_gradient_y_maps = gradient_relative_error_scalar_maps(result, component="dbz_dy")

        self.assertEqual(set(field_maps), {"target", "layout", "stream", "error"})
        self.assertEqual(set(error_maps), {"layout_vs_target", "stream_vs_target", "layout_vs_stream"})
        self.assertEqual(set(relative_error_maps), {"layout_vs_target", "stream_vs_target", "layout_vs_stream"})
        self.assertEqual(set(gradient_maps), {"target", "layout", "stream", "error"})
        self.assertEqual(set(relative_gradient_y_maps), {"layout_vs_target", "stream_vs_target", "layout_vs_stream"})
        self.assertEqual(field_maps["target"].unit, "T/A")
        self.assertEqual(error_maps["layout_vs_target"].unit, "T/A")
        self.assertEqual(relative_error_maps["layout_vs_target"].unit, "%")
        self.assertEqual(gradient_maps["layout"].unit, "T/(A m)")
        self.assertEqual(relative_gradient_y_maps["layout_vs_target"].unit, "%")
        self.assertIn("[T/(A m)]", gradient_maps["layout"].label_with_unit)
        self.assertIn("[%]", relative_error_maps["layout_vs_target"].label_with_unit)
        self.assertIn("[%]", relative_gradient_y_maps["layout_vs_target"].label_with_unit)
        for mapping in (
            field_maps,
            error_maps,
            relative_error_maps,
            gradient_maps,
            gradient_y_maps,
            relative_gradient_y_maps,
        ):
            for scalar_map in mapping.values():
                self.assertEqual(len(scalar_map.values), count)
                self.assertTrue(np.all(np.isfinite(scalar_map.values)))

        selection = central_slice_selection(result.target_field.coords, 2)
        self.assertEqual(len(selection.mask), count)
        self.assertGreater(np.count_nonzero(selection.mask), 0)

        with TemporaryDirectory() as tmp:
            gradient_file = save_gradient_map_isometric(
                result, Path(tmp) / "gradient.svg", source="layout", component="magnitude"
            )
            relative_gradient_file = save_gradient_relative_error_map_isometric(
                result,
                Path(tmp) / "gradient_relative.svg",
                source="layout_vs_target",
                component="magnitude",
            )
            error_file = save_field_error_map(result, Path(tmp) / "error.svg", source="layout_vs_target")
            relative_error_file = save_field_relative_error_map(
                result, Path(tmp) / "error_relative.svg", source="layout_vs_target"
            )
            gradient_text = gradient_file.read_text(encoding="utf-8")
            relative_gradient_text = relative_gradient_file.read_text(encoding="utf-8")
            error_text = error_file.read_text(encoding="utf-8")
            relative_error_text = relative_error_file.read_text(encoding="utf-8")
            self.assertIn("<svg", gradient_text)
            self.assertIn("T/(A m)", gradient_text)
            self.assertIn("<svg", relative_gradient_text)
            self.assertIn("[%]", relative_gradient_text)
            self.assertIn("<svg", error_text)
            self.assertIn("T/A", error_text)
            self.assertIn("<svg", relative_error_text)
            self.assertIn("[%]", relative_error_text)

    def test_projection_current_error_maps_use_final_field_units(self) -> None:
        result = _minimal_result_with_suppression_points()
        result.input_data = CoilGenConfig(field_evaluation_current_source="target_projection")
        result.needed_current_layout = 2.0
        result.potential_step = 4.0
        result.field_by_layout = result.target_field.b.copy()
        result.field_by_layout[:, 2] += 1.0
        result.field_layout_per1Amp = result.field_by_layout / result.needed_current_layout
        target_1a_b = result.target_field.b / result.potential_step
        result.target_field_1A = TargetField(
            coords=result.target_field.coords.copy(),
            b=target_1a_b,
            weights=result.target_field.weights.copy(),
            suppressed=result.target_field.suppressed.copy(),
            target_field_group_inds=result.target_field.target_field_group_inds.copy(),
        )

        error_map = field_error_scalar_maps(result)["layout_vs_target"]

        self.assertEqual(error_map.unit, "T")
        np.testing.assert_allclose(error_map.values[:2], np.ones(2))

    def test_inter_group_diagnostics_exports_csv_and_json(self) -> None:
        config = _small_y_gradient_config(skip_sweep=True)
        config.inter_group_connection_path_method = "geodesic"
        result = run_coilgen(config)

        with TemporaryDirectory() as tmp:
            files = save_inter_group_diagnostics(result, tmp, prefix="diag")
            rows = inter_group_diagnostics_rows(result)
            json_rows = json.loads((Path(tmp) / "diag_inter_group_diagnostics.json").read_text(encoding="utf-8"))
            csv_text = (Path(tmp) / "diag_inter_group_diagnostics.csv").read_text(encoding="utf-8")

        self.assertEqual(len(files), 2)
        self.assertEqual(len(rows), len(result.coil_parts[0].opening_cuts_among_groups))
        self.assertEqual(len(json_rows), len(rows))
        self.assertIn("connection_method", csv_text)
        self.assertIn("outer_contour_only", csv_text)
        self.assertIn("geodesic", csv_text)
        self.assertGreater(rows[0]["direct_distance"], 0.0)
        self.assertTrue(all(row["outer_contour_only"] for row in rows))

    def test_route_diagnostics_exports_csv_and_json(self) -> None:
        result = run_coilgen(_small_y_gradient_config(skip_sweep=True))

        with TemporaryDirectory() as tmp:
            files = save_route_diagnostics(result, tmp, prefix="route")
            rows = route_diagnostics_rows(result)
            json_rows = json.loads((Path(tmp) / "route_route_diagnostics.json").read_text(encoding="utf-8"))
            csv_text = (Path(tmp) / "route_route_diagnostics.csv").read_text(encoding="utf-8")

        self.assertEqual(len(files), 2)
        self.assertEqual(len(rows), len(result.coil_parts))
        self.assertEqual(len(json_rows), len(rows))
        self.assertIn("opening_cut_count", csv_text)
        self.assertIn("min_wire_wire_distance", csv_text)
        self.assertGreaterEqual(rows[0]["opening_cut_count"], 0)

    def test_topology_and_cut_diagnostics_exports_csv_and_json(self) -> None:
        result = run_coilgen(_small_y_gradient_config(skip_sweep=True))

        with TemporaryDirectory() as tmp:
            topo_files = save_topology_diagnostics(result, tmp, prefix="topo")
            cut_files = save_within_group_cut_diagnostics(result, tmp, prefix="cut")
            topo_rows = topology_diagnostics_rows(result)
            cut_rows = within_group_cut_diagnostics_rows(result)
            topo_text = (Path(tmp) / "topo_topology_diagnostics.csv").read_text(encoding="utf-8")
            cut_text = (Path(tmp) / "cut_within_group_cut_diagnostics.csv").read_text(encoding="utf-8")

        self.assertEqual(len(topo_files), 2)
        self.assertEqual(len(cut_files), 2)
        self.assertEqual(len(topo_rows), len(result.coil_parts[0].groups))
        self.assertIn("parent_group_index", topo_text)
        self.assertIn("selected_pair_indices", cut_text)
        self.assertTrue(any("line_deviation" in row for row in cut_rows))

    def test_field_and_contour_orientation_diagnostics_exports(self) -> None:
        result = run_coilgen(_small_y_gradient_config(skip_sweep=True))

        with TemporaryDirectory() as tmp:
            field_files = save_field_error_diagnostics(result, tmp, prefix="field")
            orientation_files = save_contour_orientation_diagnostics(result, tmp, prefix="orient")
            field_rows = field_error_diagnostics_rows(result)
            orientation_rows = contour_orientation_diagnostics_rows(result)
            orient_text = (Path(tmp) / "orient_contour_orientation_diagnostics.csv").read_text(encoding="utf-8")

        self.assertEqual(len(field_files), 2)
        self.assertEqual(len(orientation_files), 2)
        self.assertTrue(any(row["source"] == "field_error" for row in field_rows))
        self.assertEqual(len(orientation_rows), len(result.coil_parts[0].contour_lines))
        self.assertIn("chirality_defined", orient_text)
        self.assertIn("stream_current_alignment_normalized", orient_text)
        self.assertTrue(all(row["chirality_defined"] for row in orientation_rows))
        self.assertTrue(all(row["current_orientation_consistent"] for row in orientation_rows))

    def test_mesh_surface_diagnostics_reports_components_normals_and_self_intersection(self) -> None:
        mesh = Mesh(
            vertices=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.25, 0.25, -0.5],
                    [0.25, 0.25, 0.5],
                    [1.0, 1.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [2.0, 1.0, 0.0],
                ],
                dtype=float,
            ),
            faces=np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int64),
            face_normals=np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float),
        )

        diagnostics = mesh_surface_diagnostics(mesh, check_self_intersections=True)

        self.assertEqual(diagnostics["connected_face_component_count"], 3)
        self.assertGreaterEqual(diagnostics["boundary_loop_count"], 3)
        self.assertGreaterEqual(diagnostics["inconsistent_face_normal_count"], 1)
        self.assertGreaterEqual(diagnostics["self_intersection_count"], 1)

    def test_contour_orientation_diagnostics_are_defined_for_extracted_loops(self) -> None:
        result = run_coilgen(_small_y_gradient_config(skip_sweep=True))

        rows = contour_orientation_diagnostics(result.coil_parts[0])

        self.assertEqual(len(rows), len(result.coil_parts[0].contour_lines))
        self.assertTrue(all(row["chirality_defined"] for row in rows))
        self.assertTrue(all(row["stream_current_orientation_sign"] == 1 for row in rows))
        self.assertTrue(all(row["current_orientation_consistent"] for row in rows))

    def test_contour_chirality_is_undefined_when_surface_normals_cancel(self) -> None:
        part = _cylinder_z_ring_part(level_count=1)

        row = contour_orientation_diagnostics(part)[0]

        self.assertLess(row["surface_normal_coherence"], 1e-8)
        self.assertFalse(row["chirality_defined"])
        self.assertEqual(row["orientation_sign"], 0)

    def test_contour_current_orientation_is_corrected_from_stream_function(self) -> None:
        result = run_coilgen(_small_y_gradient_config(skip_sweep=True))
        part = result.coil_parts[0]
        contour = part.contour_lines[0]
        alignment, scale = contour_current_alignment(part, contour)
        self.assertGreater(scale, 0.0)
        self.assertGreater(alignment, 0.0)

        reversed_contour = reverse_contour_line(contour)
        reversed_alignment, reversed_scale = contour_current_alignment(part, reversed_contour)
        self.assertAlmostEqual(reversed_scale, scale)
        self.assertLess(reversed_alignment, 0.0)

        corrected = validate_and_correct_contour_currents(part, [reversed_contour])[0]
        corrected_alignment, corrected_scale = contour_current_alignment(part, corrected)
        self.assertAlmostEqual(corrected_scale, scale)
        self.assertGreater(corrected_alignment, 0.0)

    def test_standard_example_configs_smoke(self) -> None:
        configs = standard_example_configs()
        # These legacy 1 mm lift cases still exercise the EM pipeline. The two
        # direct cylinder fixtures penetrate their support before lifting.
        # Planar Y retains a finite-body return/contour collision at its 1 mm
        # lift; the bounded return generator cannot certify its reentrant entry.
        # The other listed fixtures retain exact source-route intersections.
        # Regular geodesic refinement now removes the cylinder Y intersection;
        # it retains the independently reported ownership/body-check warnings.
        # Require those real routing failures rather than loosening constraints.
        unresolved_reasons = {
            "cylinder_ygradient_direct3d": "connector_bridge_unresolved",
            "cylinder_xgradient_direct3d": "connector_bridge_unresolved",
            "planary_ygradient_direct3d": "connector_bridge_unresolved",
            "planary_xgradient_direct3d": "centerline_intersection",
            "planary_xgradient_geodesic": "centerline_intersection",
        }

        self.assertGreaterEqual(len(configs), 8)
        for name, config in configs:
            # These historical examples describe independent support coils and
            # skip the conductor sweep; they cannot certify a new series bridge.
            config.routing_circuit_scope = "per_part"
            result = run_coilgen(config)
            failures, _warnings = validate_generated_example_result(name, result)
            if name in unresolved_reasons:
                self.assertEqual(len(result.coil_parts), 1)
                part = result.coil_parts[0]
                self.assertTrue(part.normal_shift_diagnostics["normal_shift_unresolved"])
                self.assertTrue(part.normal_shift_diagnostics["normal_shift_quality_failure"])
                self.assertIn(unresolved_reasons[name], part.normal_shift_diagnostics["normal_shift_failure_reasons"])
                self.assertFalse(part.route_diagnostics["route_quality_ok"])
                self.assertEqual(
                    failures,
                    ["part 1: route_quality_failure_count=1 exceeds 0", "part 1: route_quality_ok is false"],
                    name,
                )
            else:
                self.assertEqual(failures, [], name)
                if name == "cylinder_ygradient_geodesic":
                    part = result.coil_parts[0]
                    self.assertTrue(part.route_diagnostics["route_quality_ok"])
                    self.assertEqual(part.route_diagnostics["route_quality_status"], "warn")
                    self.assertFalse(part.normal_shift_diagnostics["normal_shift_quality_failure"])
                    self.assertIn(
                        "crossing_ownership_conflict", part.normal_shift_diagnostics["normal_shift_failure_reasons"]
                    )
                    self.assertEqual(part.route_diagnostics["normal_shift_centerline_intersection_count"], 0)
                    self.assertEqual(find_wire_segment_contacts_3d(part.wire_path.v, threshold=1e-9), [])
                    self.assertEqual(
                        part.route_diagnostics["normal_shift_physical_clearance_status"], "requires_swept_body_check"
                    )

    def test_dense_generated_example_configs_are_self_contained(self) -> None:
        configs = dense_generated_example_configs()

        self.assertGreaterEqual(len(configs), 3)
        for name, config in configs:
            self.assertEqual(config.target_field_definition_file, "none", name)
            self.assertIn("create ", config.coil_mesh_file, name)
            if name.startswith("dense_") or name.startswith("halbach_"):
                self.assertGreaterEqual(config.levels, 10, name)
                self.assertGreaterEqual(config.within_group_cut_candidate_count, 8, name)

    def test_example_parameter_parser_fills_missing_defaults(self) -> None:
        defaults = {
            "coilgen": {"field_shape_function": "y", "levels": 8},
            "plots": {"show_active": False, "mode": "book"},
        }
        stream = io.StringIO()

        with redirect_stdout(stream):
            merged = parse_example_parameters(
                {"coilgen": {"field_shape_function": "x"}, "plots": {}},
                defaults,
                example_name="parser_test",
            )

        self.assertEqual(merged["coilgen"]["field_shape_function"], "x")
        self.assertEqual(merged["coilgen"]["levels"], 8)
        self.assertFalse(merged["plots"]["show_active"])
        self.assertEqual(merged["plots"]["mode"], "book")
        text = stream.getvalue()
        self.assertIn("using default values for missing parameter", text)
        self.assertIn("coilgen.levels = 8", text)
        self.assertIn("plots.show_active = False", text)
        self.assertIn("plots.mode = 'book'", text)

    def test_example_defaults_are_centralized_and_mergeable(self) -> None:
        self.assertIn("simple_ygradient", DEFAULTS_BY_EXAMPLE)
        defaults = default_parameters_for("simple_ygradient")

        stream = io.StringIO()
        with redirect_stdout(stream):
            merged = parse_example_parameters(
                {"coilgen": {"field_shape_function": "x"}, "plots": {"show_active": False}},
                defaults,
                example_name="simple_ygradient_test",
            )

        self.assertEqual(merged["coilgen"]["field_shape_function"], "x")
        self.assertEqual(merged["coilgen"]["levels"], defaults["coilgen"]["levels"])
        self.assertFalse(merged["plots"]["show_active"])
        self.assertEqual(merged["output"]["name"], defaults["output"]["name"])
        self.assertIn("using default values for missing parameter", stream.getvalue())

    def test_parallel_biot_savart_matches_serial(self) -> None:
        angle = np.linspace(0.0, 2.0 * np.pi, 49)
        wire = np.column_stack((0.1 * np.cos(angle), 0.1 * np.sin(angle), np.zeros_like(angle)))
        target = np.array([(x, y, z) for x in (-0.02, 0.0, 0.02) for y in (-0.03, 0.01) for z in (0.04, 0.08)])

        serial = biot_savart_wire_path(wire, target, chunk_size=5, worker_count=1, min_work_size=0)
        parallel = biot_savart_wire_path(wire, target, chunk_size=5, worker_count=2, min_work_size=0)

        np.testing.assert_allclose(parallel, serial, rtol=1e-14, atol=1e-18)

    def test_biot_savart_uses_exact_finite_straight_segment_formula(self) -> None:
        long_segment = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=float)
        subdivided = np.column_stack(
            (
                np.zeros(101, dtype=float),
                np.zeros(101, dtype=float),
                np.linspace(-1.0, 1.0, 101),
            )
        )
        target = np.array([[1.0, 0.0, 0.0]], dtype=float)

        long_field = biot_savart_wire_path(long_segment, target)
        subdivided_field = biot_savart_wire_path(subdivided, target)

        expected = np.array([[0.0, np.sqrt(2.0) * 1e-7, 0.0]], dtype=float)
        np.testing.assert_allclose(long_field, expected, rtol=1e-14, atol=1e-18)
        np.testing.assert_allclose(subdivided_field, expected, rtol=1e-14, atol=1e-18)

    def test_vectorized_wire_distance_matches_scalar_reference(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.3, 0.1, 0.0],
                [0.6, 0.0, 0.1],
                [0.2, 0.5, 0.2],
                [-0.1, 0.4, -0.1],
                [0.1, -0.2, 0.0],
                [0.45, -0.1, 0.3],
            ],
            dtype=float,
        )
        expected = (-1, -1, np.inf)
        for seg_a in range(len(points) - 1):
            for seg_b in range(seg_a + 2, len(points) - 1):
                distance = _segment_segment_distance_3d(
                    points[seg_a],
                    points[seg_a + 1],
                    points[seg_b],
                    points[seg_b + 1],
                )
                if distance < expected[2]:
                    expected = (seg_a, seg_b, distance)

        spatial_config = CoilGenConfig(wire_distance_method="spatial", wire_distance_chunk_size=2)
        all_pairs_config = CoilGenConfig(wire_distance_method="all_pairs", wire_distance_chunk_size=2)
        observed = find_min_wire_segment_distance_3d(points, config=spatial_config)
        all_pairs = find_min_wire_segment_distance_3d(points, config=all_pairs_config)

        self.assertEqual(observed[:2], expected[:2])
        self.assertAlmostEqual(observed[2], expected[2], places=14)
        self.assertEqual(observed[:2], all_pairs[:2])
        self.assertAlmostEqual(observed[2], all_pairs[2], places=14)

        threshold = expected[2] + 1e-12
        spatial_contacts = find_wire_segment_contacts_3d(points, threshold, config=spatial_config)
        all_pairs_contacts = find_wire_segment_contacts_3d(points, threshold, config=all_pairs_config)
        self.assertEqual(spatial_contacts, all_pairs_contacts)

    def test_closed_wire_distance_excludes_local_continuation_across_seam(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, -0.001, 0.0],
                [0.0, -0.010, 0.0],
                [0.020, -0.010, 0.0],
                [0.020, 0.010, 0.0],
                [0.0, 0.010, 0.0],
                [0.0, 0.009, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        for method in ("spatial", "all_pairs"):
            config = CoilGenConfig(
                wire_distance_method=method,
                min_wire_distance_m=0.005,
                min_turn_distance=0.005,
                wire_distance_chunk_size=2,
            )

            seg_a, seg_b, distance = find_min_wire_segment_distance_3d(points, config=config)

            self.assertNotEqual((seg_a, seg_b), (1, 6))
            self.assertGreaterEqual(distance, 0.01)

    def test_parallel_pipeline_matches_serial_small_case(self) -> None:
        serial_config = _small_y_gradient_config(skip_sweep=True)
        serial_config.parallel_worker_count = 1
        serial_config.parallel_min_work_size = 0
        parallel_config = _small_y_gradient_config(skip_sweep=True)
        parallel_config.parallel_worker_count = 2
        parallel_config.parallel_min_work_size = 0

        serial = run_coilgen(serial_config)
        parallel = run_coilgen(parallel_config)

        np.testing.assert_allclose(parallel.b_field_opt_sf, serial.b_field_opt_sf, rtol=1e-12, atol=1e-18)
        np.testing.assert_allclose(parallel.field_layout_per1Amp, serial.field_layout_per1Amp, rtol=1e-12, atol=1e-18)
        self.assertEqual(len(parallel.coil_parts[0].contour_lines), len(serial.coil_parts[0].contour_lines))

    def test_vectorized_sensitivity_matches_scalar_small_case(self) -> None:
        scalar_config = _small_y_gradient_config(skip_sweep=True)
        scalar_config.sensitivity_calculation_method = "scalar"
        scalar_config.optimization_solver = "pinv"
        vectorized_config = _small_y_gradient_config(skip_sweep=True)
        vectorized_config.sensitivity_calculation_method = "vectorized"
        vectorized_config.optimization_solver = "pinv"

        scalar = run_coilgen(scalar_config)
        vectorized = run_coilgen(vectorized_config)

        np.testing.assert_allclose(vectorized.b_field_opt_sf, scalar.b_field_opt_sf, rtol=1e-12, atol=1e-18)
        np.testing.assert_allclose(vectorized.field_layout_per1Amp, scalar.field_layout_per1Amp, rtol=1e-12, atol=1e-18)

    def test_generated_example_output_helper_saves_debug_views(self) -> None:
        result = run_coilgen(_small_y_gradient_config(skip_sweep=True))

        with TemporaryDirectory() as tmp:
            files = save_generated_example_outputs(result, tmp, prefix="generated")

        names = {file.name for file in files}
        self.assertTrue(any(name.endswith("_within_group_cut_choices_iso.svg") for name in names))
        self.assertTrue(any(name.endswith("_interconnect_debug_iso.svg") for name in names))
        self.assertTrue(any(name.endswith("_route_diagnostics.csv") for name in names))

    def test_contour_group_visualization_exports_topology_order(self) -> None:
        raw_part = _radial_contour_part(levels=(0.16, 0.49))
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "topological_loop_grouping"):
                save_contour_group_projection(raw_part, Path(tmp) / "raw_groups.svg")

            part = topological_loop_grouping([raw_part])[0]
            file = save_contour_group_projection(part, Path(tmp) / "groups.svg")
            iso_file = save_contour_group_isometric(part, Path(tmp) / "groups_iso.svg")
            text = file.read_text(encoding="utf-8")
            iso_text = iso_file.read_text(encoding="utf-8")

        self.assertIn("<svg", text)
        self.assertIn("Contour Groups XY", text)
        self.assertIn("G0", text)
        self.assertIn("2 loops", text)
        self.assertGreater(len(text), 500)
        self.assertIn("Contour Groups ISO", iso_text)
        self.assertGreater(len(iso_text), 500)

    def test_interconnect_debug_visualization_exports_cut_markers(self) -> None:
        result = run_coilgen(_small_y_gradient_config(skip_sweep=True))

        with TemporaryDirectory() as tmp:
            file = save_interconnect_debug_projection(result.coil_parts[0], Path(tmp) / "debug.svg")
            iso_file = save_interconnect_debug_isometric(result.coil_parts[0], Path(tmp) / "debug_iso.svg")
            text = file.read_text(encoding="utf-8")
            iso_text = iso_file.read_text(encoding="utf-8")

        self.assertIn("Interconnect Debug XY", text)
        self.assertIn("within-group opening cuts", text)
        self.assertIn("group return path", text)
        self.assertIn("<circle", text)
        self.assertIn("Interconnect Debug ISO", iso_text)
        self.assertGreater(len(iso_text), 500)

    def test_target_field_json_definition_loader(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "target.json"
            path.write_text(
                """
{
  "coords": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]],
  "b": [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
  "weights": [1.0, 0.5],
  "suppressed": [false, true],
  "target_field_group_inds": [1, 2]
}
""".strip(),
                encoding="utf-8",
            )
            target = load_target_field_definition(path, None, CoilGenConfig(target_gradient_strength=2.0))

        self.assertEqual(target.coords.shape, (2, 3))
        self.assertAlmostEqual(target.b[1, 2], 4.0)
        self.assertTrue(target.suppressed[1])
        self.assertAlmostEqual(target.weights[1], 0.5)

    def test_target_field_csv_definition_loader_with_named_field_and_gradient(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "target.csv"
            path.write_text(
                "\n".join(
                    [
                        "x_m,y_m,z_m,main_bx,main_by,main_bz,weight,suppressed,group,dbz_dx,dbz_dy,dbz_dz",
                        "0.0,0.0,0.0,0.0,0.0,1.0,1.0,false,1,0.1,0.2,0.3",
                        "0.0,0.0,0.1,0.0,0.0,2.0,0.5,true,2,0.4,0.5,0.6",
                    ]
                ),
                encoding="utf-8",
            )
            target = load_target_field_definition(
                path,
                None,
                CoilGenConfig(target_gradient_strength=3.0, target_field_definition_field_name="main"),
            )

        self.assertEqual(target.coords.shape, (2, 3))
        self.assertAlmostEqual(target.b[1, 2], 6.0)
        self.assertAlmostEqual(target.weights[1], 0.5)
        self.assertTrue(target.suppressed[1])
        self.assertEqual(target.target_field_group_inds[1], 2)
        self.assertTrue(np.allclose(target.target_gradient_dbdxyz[1], [1.2, 1.5, 1.8]))

    def test_target_field_csv_definition_loader_accepts_named_scalar_bz_map(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "target_scalar.csv"
            path.write_text(
                "\n".join(
                    [
                        "x_m,y_m,z_m,lin_y,weight",
                        "0.0,-0.1,0.0,-0.1,1.0",
                        "0.0,0.1,0.0,0.1,0.5",
                    ]
                ),
                encoding="utf-8",
            )
            target = load_target_field_definition(
                path,
                None,
                CoilGenConfig(target_gradient_strength=2.0, target_field_definition_field_name="lin_y"),
            )

        np.testing.assert_allclose(target.b[:, :2], 0.0)
        np.testing.assert_allclose(target.b[:, 2], [-0.2, 0.2])
        np.testing.assert_allclose(target.weights, [1.0, 0.5])

    def test_target_field_csv_definition_loader_accepts_hz_units(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "target_hz.csv"
            path.write_text(
                "\n".join(
                    [
                        "x_m,y_m,z_m,shim_bz,weight",
                        f"0.0,0.0,0.0,{PROTON_GYROMAGNETIC_RATIO_HZ_PER_T},1.0",
                        f"0.0,0.0,0.1,{-0.5 * PROTON_GYROMAGNETIC_RATIO_HZ_PER_T},0.5",
                    ]
                ),
                encoding="utf-8",
            )
            target = load_target_field_definition(
                path,
                None,
                CoilGenConfig(target_field_definition_field_name="shim_bz", target_field_units="Hz"),
            )

        np.testing.assert_allclose(target.b[:, 2], [1.0, -0.5])
        np.testing.assert_allclose(target.weights, [1.0, 0.5])

    def test_generated_target_field_can_use_hz_units(self) -> None:
        target = define_target_field(
            [],
            None,
            None,
            CoilGenConfig(
                field_shape_function="1",
                target_gradient_strength=PROTON_GYROMAGNETIC_RATIO_HZ_PER_T,
                target_field_units="Hz",
                target_region_radius=0.01,
                target_region_resolution=3,
            ),
        )

        self.assertEqual(target.b.shape, (1, 3))
        self.assertAlmostEqual(target.b[0, 2], 1.0)

    def test_mr_gradient_expression_target_normalizes_by_linear_fit(self) -> None:
        target = define_target_field(
            [],
            None,
            None,
            CoilGenConfig(
                design_context="mr_gradient",
                field_shape_function="2*y",
                target_gradient_strength=0.03,
                target_gradient_axis="y",
                target_region_radius=0.1,
                target_region_resolution=5,
            ),
        )

        np.testing.assert_allclose(target.b[:, 2], 0.03 * target.coords[:, 1], atol=1e-12)
        np.testing.assert_allclose(target.target_gradient_dbdxyz[:, 1], 0.03, atol=1e-12)
        self.assertEqual(target.target_diagnostics["target_gradient_normalization_method"], "linear_fit")
        self.assertEqual(target.target_diagnostics["target_gradient_axis"], "y")
        self.assertAlmostEqual(target.target_diagnostics["target_gradient_normalization_scale"], 0.015)
        self.assertAlmostEqual(target.target_diagnostics["target_gradient_effective_T_per_m"], 0.03)

    def test_constant_mr_target_does_not_amplify_roundoff_as_a_gradient(self) -> None:
        target = define_target_field(
            [],
            None,
            None,
            CoilGenConfig(
                design_context="mr_gradient",
                field_shape_function="1",
                target_gradient_strength=1.0e-4,
                target_region_radius=0.01,
                target_region_resolution=3,
            ),
        )

        np.testing.assert_allclose(target.b[:, 2], 1.0e-4, atol=1.0e-15)
        self.assertTrue(target.target_diagnostics["target_gradient_normalization_fallback"])
        self.assertEqual(target.target_diagnostics["target_gradient_raw_strength_T_per_m"], 0.0)

    def test_imported_mr_gradient_target_can_request_linear_fit_normalization(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "target_scalar.csv"
            path.write_text(
                "\n".join(
                    [
                        "x_m,y_m,z_m,lin_y,weight",
                        "0.0,-0.1,0.0,-0.1,1.0",
                        "0.0,0.1,0.0,0.1,0.5",
                    ]
                ),
                encoding="utf-8",
            )
            target = load_target_field_definition(
                path,
                None,
                CoilGenConfig(
                    target_gradient_strength=0.03,
                    target_gradient_normalization="linear_fit",
                    target_gradient_axis="y",
                    target_field_definition_field_name="lin_y",
                ),
            )

        np.testing.assert_allclose(target.b[:, 2], [-0.003, 0.003])
        np.testing.assert_allclose(target.target_gradient_dbdxyz[:, 1], 0.03)
        self.assertAlmostEqual(target.target_diagnostics["target_gradient_effective_T_per_m"], 0.03)

    def test_field_maps_and_summary_can_display_hz(self) -> None:
        result = _minimal_result_with_suppression_points()
        result.input_data = CoilGenConfig(field_display_units="Hz")

        maps = field_scalar_maps(result)
        summary = result_summary_dict(result)

        self.assertEqual(maps["target"].unit, "Hz/A")
        self.assertAlmostEqual(maps["target"].values[1], 0.5 * PROTON_GYROMAGNETIC_RATIO_HZ_PER_T)
        self.assertEqual(summary["field_display_units"], "Hz")
        self.assertAlmostEqual(summary["target_peak_Hz"], 0.5 * PROTON_GYROMAGNETIC_RATIO_HZ_PER_T)

    def test_shim_homogeneity_maps_stats_and_svg_mosaic_use_wire_layout_remaining_inhomogeneity(self) -> None:
        result = _minimal_result_with_suppression_points()
        result.field_by_layout[1, 2] = 0.25
        result.input_data = CoilGenConfig(
            design_context="mr_shim",
            field_display_units="Hz",
        )

        maps = shim_field_scalar_maps(result)
        stats = shim_homogeneity_stats(result, source="layout", domain="dsv")
        summary = result_summary_dict(result)
        selections = slice_mosaic_selections(
            result.target_field.coords,
            axis=1,
            slice_count=2,
            sample_mask=~result.target_field.suppressed,
        )

        self.assertEqual(maps["layout_residual"].unit, "Hz")
        self.assertAlmostEqual(maps["layout_residual"].values[1], 0.25 * PROTON_GYROMAGNETIC_RATIO_HZ_PER_T)
        self.assertAlmostEqual(
            maps["layout_remaining_inhomogeneity"].values[1], -0.25 * PROTON_GYROMAGNETIC_RATIO_HZ_PER_T
        )
        self.assertAlmostEqual(
            maps["layout_improvement_before_minus_after"].values[1],
            -0.25 * PROTON_GYROMAGNETIC_RATIO_HZ_PER_T,
        )
        self.assertEqual(stats["unit"], "Hz")
        self.assertAlmostEqual(stats["max_abs"], 0.25 * PROTON_GYROMAGNETIC_RATIO_HZ_PER_T)
        self.assertAlmostEqual(summary["shim_layout_residual_max_abs_Hz"], 0.25 * PROTON_GYROMAGNETIC_RATIO_HZ_PER_T)
        self.assertGreaterEqual(len(selections), 1)

        with TemporaryDirectory() as tmp:
            path = save_shim_homogeneity_mosaic(
                result,
                Path(tmp) / "shim_mosaic.svg",
                axis=1,
                slice_count=2,
                domain="all",
                min_points_per_slice=1,
            )
            text = path.read_text(encoding="utf-8")
            improvement_path = save_shim_homogeneity_mosaic(
                result,
                Path(tmp) / "shim_improvement_mosaic.svg",
                source="improvement",
                axis=1,
                slice_count=2,
                domain="all",
                min_points_per_slice=1,
            )
            improvement_text = improvement_path.read_text(encoding="utf-8")
            pointcloud_path = save_shim_pointcloud_before_after(
                result,
                Path(tmp) / "shim_pointcloud.svg",
                domain="all",
            )
            pointcloud_text = pointcloud_path.read_text(encoding="utf-8")

        self.assertIn("<svg", text)
        self.assertIn("B0 Inhomogeneity Filled Z-Slice Mosaic", text)
        self.assertIn("After Shimming Remaining B0 Inhomogeneity", text)
        self.assertIn("Shim Improvement Before - After", improvement_text)
        self.assertIn("Shim Target-Point B0 Inhomogeneity", pointcloud_text)
        self.assertIn("Before Shimming B0 Inhomogeneity", pointcloud_text)
        self.assertIn("After Shimming Remaining B0 Inhomogeneity", pointcloud_text)

    def test_spherical_harmonic_fit_recovers_known_basis_field(self) -> None:
        axis = np.linspace(-1.0, 1.0, 5)
        coords = np.asarray(
            [(x, y, z) for x in axis for y in axis for z in axis if x * x + y * y + z * z <= 1.0],
            dtype=float,
        )
        basis, terms, _, _ = solid_spherical_harmonic_basis(coords, max_order=2, center=np.zeros(3), radius=1.0)
        term_index = {(term.degree, term.order): index for index, term in enumerate(terms)}
        values = (
            11.0 * basis[:, term_index[(0, 0)]]
            - 7.0 * basis[:, term_index[(1, 1)]]
            + 3.0 * basis[:, term_index[(2, 0)]]
        )

        fit = fit_spherical_harmonics(coords, values, max_order=2, center=np.zeros(3), radius=1.0, unit="Hz")
        rows = spherical_harmonic_coefficient_rows(fit, min_contribution_fraction=0.05, max_terms=4)

        self.assertEqual(fit.unit, "Hz")
        np.testing.assert_allclose(fit.fitted_values, values, atol=1.0e-10)
        self.assertAlmostEqual(fit.coefficients[term_index[(0, 0)]], 11.0, places=10)
        self.assertAlmostEqual(fit.coefficients[term_index[(1, 1)]], -7.0, places=10)
        self.assertAlmostEqual(fit.coefficients[term_index[(2, 0)]], 3.0, places=10)
        self.assertTrue(any(row["degree"] == 2 and row["order"] == 0 for row in rows))

    def test_shim_spherical_harmonics_compare_before_after_hz(self) -> None:
        result = _minimal_result_with_suppression_points()
        result.input_data = CoilGenConfig(field_display_units="Hz")

        fits = shim_spherical_harmonic_fits(result, domain="dsv", max_order=1)
        rows = shim_spherical_harmonic_comparison(
            result, domain="dsv", max_order=1, min_contribution_fraction=0.0, max_terms=4
        )

        self.assertEqual(fits["before"].unit, "Hz")
        self.assertEqual(fits["before"].sample_count, 2)
        self.assertTrue(rows)
        self.assertGreater(max(float(row["contribution_rms_delta"]) for row in rows), 0.0)

    def test_coil_spherical_harmonics_decompose_target_layout_and_residual(self) -> None:
        axis = np.linspace(-1.0, 1.0, 5)
        coords = np.asarray(
            [(x, y, z) for x in axis for y in axis for z in axis if x * x + y * y + z * z <= 1.0],
            dtype=float,
        )
        basis, terms, _, _ = solid_spherical_harmonic_basis(coords, max_order=2, center=np.zeros(3), radius=1.0)
        term_index = {(term.degree, term.order): index for index, term in enumerate(terms)}
        target_values = 4.0 * basis[:, term_index[(1, 1)]] + 2.0 * basis[:, term_index[(2, 0)]]
        layout_values = target_values + 0.5 * basis[:, term_index[(0, 0)]]
        result = _result_from_bz_values(coords, target_values, layout_values)

        fits = coil_spherical_harmonic_fits(
            result,
            max_order=2,
            sources=("target", "layout", "layout_error"),
            center=np.zeros(3),
            radius=1.0,
        )
        rows = coil_spherical_harmonic_coefficient_rows(
            result,
            max_order=2,
            sources=("target", "layout", "layout_error"),
            min_contribution_fraction=0.0,
            max_terms=None,
            center=np.zeros(3),
            radius=1.0,
        )

        self.assertEqual(fits["target"].unit, "T")
        np.testing.assert_allclose(fits["target"].coefficients[term_index[(1, 1)]], 4.0, atol=1.0e-10)
        np.testing.assert_allclose(fits["target"].coefficients[term_index[(2, 0)]], 2.0, atol=1.0e-10)
        np.testing.assert_allclose(fits["layout_error"].coefficients[term_index[(0, 0)]], 0.5, atol=1.0e-10)
        self.assertTrue(any(row["degree"] == 1 and row["order"] == 1 for row in rows))

    def test_field_spherical_harmonics_plot_spec_is_in_default_result_book(self) -> None:
        result = _minimal_result_with_suppression_points()

        keys = [spec.key for spec in build_result_plot_specs(result, include=("spherical_harmonics",))]

        self.assertEqual(keys, ["spherical_harmonics"])

    def test_spherical_target_points_include_symmetric_boundary_points(self) -> None:
        target = define_target_field(
            [],
            None,
            None,
            CoilGenConfig(
                field_shape_function="y",
                target_region_radius=0.2,
                target_region_resolution=5,
            ),
        )

        rounded = {tuple(np.round(point, 12)) for point in target.coords}
        self.assertEqual(len(rounded), 7)
        for point in rounded:
            mirrored = tuple(np.round(-np.asarray(point), 12))
            self.assertIn(mirrored, rounded)

    def test_target_mode_efficiency_ignores_orthogonal_bz_noise(self) -> None:
        target_scale = 2.0
        target_mode = np.array([-1.0, 0.0, 1.0, 0.0])
        target_bz = target_scale * target_mode
        field_bz = 0.5 * target_mode + np.array([0.0, 40.0, 0.0, -40.0])
        field = np.column_stack((np.zeros_like(field_bz), np.zeros_like(field_bz), field_bz))

        fit = target_mode_fit_performance(field, target_bz, target_scale)

        self.assertAlmostEqual(fit["gradient_efficiency_mT_per_m_per_A"], 500.0)
        self.assertAlmostEqual(fit["needed_current_A"], 4.0)
        self.assertGreater(fit["mean_rel_error_percent"], 1000.0)

    def test_field_evaluation_current_can_use_stream_function_contour_step_for_shims(self) -> None:
        field = np.column_stack(
            (
                np.zeros(3, dtype=float),
                np.zeros(3, dtype=float),
                np.array([1.0, 2.0, 3.0], dtype=float),
            )
        )
        target = np.array([0.5, -1.0, 0.25], dtype=float)
        mask = np.ones(3, dtype=bool)

        current = _field_evaluation_current(
            field,
            target,
            1.0,
            mask,
            contour_step=7.5,
            source="stream_function",
        )
        projected = _field_evaluation_current(
            field,
            target,
            1.0,
            mask,
            contour_step=7.5,
            source="projection",
        )

        self.assertAlmostEqual(current, 7.5)
        self.assertAlmostEqual(projected, float((field[:, 2] @ target) / (field[:, 2] @ field[:, 2])))

    def test_field_evaluation_current_std_projection_minimizes_weighted_homogeneity(self) -> None:
        field_z = np.array([0.0, 1.0, 3.0, 8.0], dtype=float)
        target = np.array([10.0, 9.0, 5.0, -2.0], dtype=float)
        weights = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        field = np.column_stack((np.zeros(4), np.zeros(4), field_z))
        mask = np.ones(4, dtype=bool)

        current = _field_evaluation_current(
            field,
            target,
            1.0,
            mask,
            contour_step=1.0,
            source="target_std_projection",
            sample_weights=weights,
        )

        weighted_mean_field = float(np.dot(weights, field_z) / np.sum(weights))
        weighted_mean_target = float(np.dot(weights, target) / np.sum(weights))
        centered_field = field_z - weighted_mean_field
        centered_target = target - weighted_mean_target
        expected = float(
            np.dot(weights, centered_field * centered_target) / np.dot(weights, centered_field * centered_field)
        )
        self.assertAlmostEqual(current, expected)

        residual = target - current * field_z
        perturbed = target - (current + 0.1) * field_z
        residual_mean = float(np.dot(weights, residual) / np.sum(weights))
        perturbed_mean = float(np.dot(weights, perturbed) / np.sum(weights))
        residual_variance = float(np.dot(weights, (residual - residual_mean) ** 2))
        perturbed_variance = float(np.dot(weights, (perturbed - perturbed_mean) ** 2))
        self.assertLess(residual_variance, perturbed_variance)

    def test_preoptimized_stream_function_csv_bypasses_inverse_solve(self) -> None:
        mesh = normalize_mesh(build_cylinder_mesh(0.25, 0.1, 8, 4, 1, 0, 0, 0))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "preoptimized.csv"
            lines = ["part,vertex_index,x_m,y_m,z_m,stream_function"]
            for vertex_ind, vertex in enumerate(mesh.vertices):
                lines.append(f"1,{vertex_ind},{vertex[0]:.12g},{vertex[1]:.12g},{vertex[2]:.12g},{vertex[2]:.12g}")
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            config = _small_y_gradient_config(skip_sweep=True)
            config.field_shape_function = "1"
            config.preoptimized_stream_function_file = str(path)
            config.skip_postprocessing = True
            config.regularization_selection_method = "fixed"
            result = run_coilgen(config)

        part = result.coil_parts[0]
        self.assertEqual(result.regularization_diagnostics["selection_method"], "preoptimized")
        self.assertEqual(result.regularization_diagnostics["stream_function_scheme"], "preoptimized")
        np.testing.assert_allclose(part.stream_function, part.mesh.vertices[:, 2])
        self.assertGreater(len(part.contour_lines), 0)

    def test_field_shape_expression_accepts_common_product_syntax(self) -> None:
        points = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]])
        expected = np.array([2.0, 12.0])

        self.assertTrue(np.allclose(evaluate_field_expression("x*y", points), expected))
        self.assertTrue(np.allclose(evaluate_field_expression("xy", points), expected))
        self.assertTrue(np.allclose(evaluate_field_expression("x.*y", points), expected))
        self.assertTrue(np.allclose(evaluate_field_expression("2xy", points), 2.0 * expected))


def _clearance_selection_part() -> tuple[CoilPart, list[ConnectedGroup]]:
    mesh = Mesh(
        vertices=np.array(
            [
                [-0.1, -0.2, 0.0],
                [0.4, -0.2, 0.0],
                [-0.1, 1.2, 0.0],
                [0.4, 1.2, 0.0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64),
    )
    obstacle = ContourLine(
        v=np.array([[0.1, -0.1, 0.0], [0.1, 0.1, 0.0]], dtype=float),
        potential=0.0,
    )
    contour_a = ContourLine(v=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float), potential=1.0)
    contour_b = ContourLine(v=np.array([[0.2, 0.0, 0.0], [0.25, 1.0, 0.0]], dtype=float), potential=2.0)
    part = CoilPart(mesh=mesh, contour_lines=[obstacle, contour_a, contour_b])
    part.groups = [
        LoopGroup(loops=[contour_a], loop_indices=[1], outer_loop_indices=(1,)),
        LoopGroup(loops=[contour_b], loop_indices=[2], outer_loop_indices=(2,)),
    ]
    empty = np.zeros((0, 3), dtype=float)
    group_a_v = contour_a.v
    group_b_v = contour_b.v
    groups = [
        ConnectedGroup(
            v=group_a_v,
            return_path_v=empty,
            spiral_in_v=group_a_v,
            spiral_out_v=group_a_v,
            outer_loop_indices=(1,),
        ),
        ConnectedGroup(
            v=group_b_v,
            return_path_v=empty,
            spiral_in_v=group_b_v,
            spiral_out_v=group_b_v,
            outer_loop_indices=(2,),
        ),
    ]
    return part, groups


def _outer_only_selection_part() -> tuple[CoilPart, list[ConnectedGroup]]:
    mesh = Mesh(
        vertices=np.array(
            [
                [-5.0, -1.0, 0.0],
                [5.0, -1.0, 0.0],
                [-5.0, 1.0, 0.0],
                [5.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64),
    )

    def rectangle(x_min: float, x_max: float) -> np.ndarray:
        return np.array(
            [
                [x_min, -0.5, 0.0],
                [x_max, -0.5, 0.0],
                [x_max, 0.5, 0.0],
                [x_min, 0.5, 0.0],
                [x_min, -0.5, 0.0],
            ],
            dtype=float,
        )

    outer_a = ContourLine(v=rectangle(-4.0, -3.0), potential=0.0)
    inner_a = ContourLine(v=rectangle(-0.4, -0.2), potential=1.0)
    outer_b = ContourLine(v=rectangle(3.0, 4.0), potential=2.0)
    inner_b = ContourLine(v=rectangle(0.2, 0.4), potential=3.0)
    part = CoilPart(mesh=mesh, contour_lines=[outer_a, inner_a, outer_b, inner_b])
    part.loop_in_loop_matrix = np.zeros((4, 4), dtype=bool)
    part.loop_in_loop_matrix[0, 1] = True
    part.loop_in_loop_matrix[2, 3] = True
    part.groups = [
        LoopGroup(loops=[outer_a, inner_a], loop_indices=[0, 1], outer_loop_indices=(0,)),
        LoopGroup(loops=[outer_b, inner_b], loop_indices=[2, 3], outer_loop_indices=(2,)),
    ]
    empty = np.zeros((0, 3), dtype=float)
    group_a_v = np.vstack((outer_a.v, inner_a.v))
    group_b_v = np.vstack((outer_b.v, inner_b.v))
    groups = [
        ConnectedGroup(
            v=group_a_v,
            return_path_v=empty,
            spiral_in_v=group_a_v,
            spiral_out_v=group_a_v,
            outer_loop_indices=(0,),
        ),
        ConnectedGroup(
            v=group_b_v,
            return_path_v=empty,
            spiral_in_v=group_b_v,
            spiral_out_v=group_b_v,
            outer_loop_indices=(2,),
        ),
    ]
    return part, groups


def _rail_selection_group() -> tuple[LoopGroup, list[LoopCutShape]]:
    centers = [np.array([x, 0.0, 0.0], dtype=float) for x in (0.0, 1.0, 2.0)]
    radius = 0.2
    loops = [
        ContourLine(v=_circle_points_around(center, radius, samples=97), potential=float(ind))
        for ind, center in enumerate(centers)
    ]
    cut_infos: list[LoopCutShape] = []
    for ind, center in enumerate(centers):
        high = LoopCut(v=center + np.array([0.0, radius, 0.0]), segment_index=0)
        low_y = -radius if ind % 2 == 0 else radius
        low = LoopCut(v=center + np.array([0.0, low_y, 0.0]), segment_index=0)
        cut_infos.append(
            LoopCutShape(
                cut_points=[high, low],
                high_cut=high,
                low_cut=low,
                cut_width_high=0.04,
                cut_width_low=0.04,
            )
        )
    return LoopGroup(loops=loops, loop_indices=[0, 1, 2]), cut_infos


def _multi_pair_rail_selection_group() -> tuple[LoopGroup, list[LoopCutShape]]:
    centers = [np.array([x, 0.0, 0.0], dtype=float) for x in (0.0, 1.0, 2.0)]
    radius = 0.25
    loops = [
        ContourLine(v=_circle_points_around(center, radius, samples=129), potential=float(ind))
        for ind, center in enumerate(centers)
    ]

    def cut(center: np.ndarray, dx: float, dy: float) -> LoopCut:
        return LoopCut(v=center + np.array([dx, dy, 0.0]), segment_index=0)

    first_top = cut(centers[0], 0.0, radius)
    first_bottom = cut(centers[0], 0.0, -radius)
    second_right = cut(centers[1], radius, 0.0)
    second_left = cut(centers[1], -radius, 0.0)
    second_top = cut(centers[1], 0.0, radius)
    second_bottom = cut(centers[1], 0.0, -radius)
    third_top = cut(centers[2], 0.0, radius)
    third_bottom = cut(centers[2], 0.0, -radius)

    cut_infos = [
        LoopCutShape(
            cut_points=[first_top, first_bottom],
            high_cut=first_top,
            low_cut=first_bottom,
            cut_width_high=0.04,
            cut_width_low=0.04,
        ),
        LoopCutShape(
            cut_points=[second_right, second_left, second_top, second_bottom],
            high_cut=second_right,
            low_cut=second_left,
            cut_width_high=0.04,
            cut_width_low=0.04,
        ),
        LoopCutShape(
            cut_points=[third_top, third_bottom],
            high_cut=third_top,
            low_cut=third_bottom,
            cut_width_high=0.04,
            cut_width_low=0.04,
        ),
    ]
    return LoopGroup(loops=loops, loop_indices=[0, 1, 2]), cut_infos


def _circle_points_around(center: np.ndarray, radius: float, samples: int) -> np.ndarray:
    angle = np.linspace(0.0, 2.0 * np.pi, samples)
    return np.column_stack(
        (
            center[0] + radius * np.cos(angle),
            center[1] + radius * np.sin(angle),
            np.full_like(angle, center[2]),
        )
    )


def _small_y_gradient_config(*, skip_sweep: bool) -> CoilGenConfig:
    return CoilGenConfig(
        field_shape_function="y",
        coil_mesh_file="create cylinder mesh",
        cylinder_mesh_parameter_list=(0.25, 0.1, 8, 4, 1, 0, 0, 0),
        target_region_radius=0.04,
        target_region_resolution=5,
        levels=4,
        pot_offset_factor=0.25,
        min_point_loop_number=3,
        gauss_order=1,
        tikonov_reg_factor=10.0,
        plot_flag=False,
        save_stl_flag=False,
        skip_sweep=skip_sweep,
        skip_inductance_calculation=True,
    )


def _stream_function_range_part(
    stream_function_range: float,
    *,
    z: float = 0.0,
    stream_function: np.ndarray | None = None,
) -> CoilPart:
    mesh = _grid_mesh(2, 2)
    mesh.vertices[:, 2] = float(z)
    part = CoilPart(mesh=mesh)
    if stream_function is None:
        part.stream_function = np.linspace(0.0, float(stream_function_range), mesh.n_vertices)
    else:
        part.stream_function = np.asarray(stream_function, dtype=float)
    return part


def _assign_symmetric_optimization_matrices(parts: list[CoilPart]) -> None:
    for part in parts:
        n_vertices = part.mesh.n_vertices
        sensitivity = np.zeros((3, 2, n_vertices), dtype=float)
        for vertex_ind in range(n_vertices):
            sensitivity[2, :, vertex_ind] = float(vertex_ind + 1)
        part.sensitivity_matrix = sensitivity
        part.resistance_matrix = np.eye(n_vertices)


def _two_point_symmetric_target(bz_values: np.ndarray) -> TargetField:
    coords = np.array([[0.0, 0.0, 0.25], [0.0, 0.0, -0.25]], dtype=float)
    b = np.zeros((2, 3), dtype=float)
    b[:, 2] = np.asarray(bz_values, dtype=float)
    return TargetField(
        coords=coords,
        b=b,
        weights=np.ones(2, dtype=float),
        suppressed=np.zeros(2, dtype=bool),
        target_field_group_inds=np.ones(2, dtype=np.int64),
    )


def _single_mesh_x_mirror_part() -> CoilPart:
    vertices = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 3], [0, 3, 2]], dtype=np.int64)
    part = CoilPart(mesh=Mesh(vertices=vertices, faces=faces))
    sensitivity = np.zeros((3, 2, 4), dtype=float)
    sensitivity[2, :, 0] = [1.0, 2.0]
    sensitivity[2, :, 1] = [2.0, 1.0]
    sensitivity[2, :, 2] = [3.0, 4.0]
    sensitivity[2, :, 3] = [4.0, 3.0]
    part.sensitivity_matrix = sensitivity
    part.resistance_matrix = np.eye(4)
    return part


def _two_point_x_symmetric_target(bz_values: np.ndarray) -> TargetField:
    coords = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=float)
    b = np.zeros((2, 3), dtype=float)
    b[:, 2] = np.asarray(bz_values, dtype=float)
    return TargetField(
        coords=coords,
        b=b,
        weights=np.ones(2, dtype=float),
        suppressed=np.zeros(2, dtype=bool),
        target_field_group_inds=np.ones(2, dtype=np.int64),
    )


def _plane_crossing_square_mesh() -> Mesh:
    return Mesh(
        vertices=np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
    )


def _centered_plane_grid_mesh(nx: int, ny: int) -> Mesh:
    x_values = np.linspace(-1.0, 1.0, int(nx))
    y_values = np.linspace(-0.5, 0.5, int(ny))
    vertices = np.array([(x, y, 0.0) for y in y_values for x in x_values], dtype=float)
    faces = []
    for row in range(int(ny) - 1):
        for col in range(int(nx) - 1):
            a = row * int(nx) + col
            b = a + 1
            c = a + int(nx)
            d = c + 1
            faces.append((a, b, d))
            faces.append((a, d, c))
    return Mesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64))


def _auto_symmetry_parameters(*, require_parity: bool) -> dict[str, Any]:
    return {
        "mesh": {
            "auto_symmetry_candidate_axes": ("x",),
            "auto_symmetry_max_samples": 1000,
            "auto_symmetry_geometry_tolerance_m": 1.0e-6,
            "auto_symmetry_mesh_coverage_min": 0.99,
            "auto_symmetry_target_coverage_min": 0.99,
            "auto_symmetry_target_parity_tolerance": 0.1,
            "auto_symmetry_min_parity_margin": 0.1,
            "auto_symmetry_require_target_parity": bool(require_parity),
            "symmetrize_current_surface": True,
        },
        "field": {
            "field_name": "shim_bz",
            "symmetrize_target_field": True,
            "target_symmetry_interpolation_neighbors": 1,
            "target_symmetry_max_distance_factor": 1.5,
        },
        "coilgen": {
            "symmetry_reduction": "strict",
            "symmetry_reduction_mirror_plane_point": (0.0, 0.0, 0.0),
            "symmetry_reduction_mirror_plane_normal": (1.0, 0.0, 0.0),
            "contour_symmetry_validation": "strict",
        },
    }


def _half_reconstructed_square_with_pairs() -> tuple[Mesh, np.ndarray]:
    mesh, _summary = mirror_symmetric_mesh_from_half(
        _plane_crossing_square_mesh(),
        plane_point=(0.0, 0.0, 0.0),
        plane_normal=(1.0, 0.0, 0.0),
        half_space="positive",
        clip_tolerance=1e-12,
    )
    matches = _assert_exact_x_mirror_vertex_pairs(mesh)
    return mesh, matches


def _assert_exact_x_mirror_vertex_pairs(mesh: Mesh) -> np.ndarray:
    reflected = symmetry_reflect_points(
        mesh.vertices,
        np.array([0.0, 0.0, 0.0], dtype=float),
        np.array([1.0, 0.0, 0.0], dtype=float),
    )
    matches = symmetry_nearest_indices(reflected, mesh.vertices)
    if matches is None:
        raise AssertionError("Mirror vertex matching unexpectedly failed.")
    matches = np.asarray(matches, dtype=np.int64)
    distances = np.linalg.norm(reflected - mesh.vertices[matches], axis=1)
    np.testing.assert_allclose(distances, 0.0, atol=1e-12)
    if len(np.unique(matches)) != len(matches):
        raise AssertionError("Mirror vertex mapping is not one-to-one.")
    return matches


def _assign_x_mirror_even_sensitivity(part: CoilPart, matches: np.ndarray) -> None:
    n_vertices = part.mesh.n_vertices
    sensitivity = np.zeros((3, 2, n_vertices), dtype=float)
    visited: set[int] = set()
    for vertex_ind in range(n_vertices):
        if vertex_ind in visited:
            continue
        mirror_ind = int(matches[vertex_ind])
        visited.add(vertex_ind)
        visited.add(mirror_ind)
        if mirror_ind == vertex_ind:
            value = 0.5 + 0.05 * vertex_ind
            sensitivity[2, :, vertex_ind] = [value, value]
            continue
        left_value = 1.0 + 0.1 * min(vertex_ind, mirror_ind)
        right_value = 1.5 + 0.1 * max(vertex_ind, mirror_ind)
        sensitivity[2, :, vertex_ind] = [left_value, right_value]
        sensitivity[2, :, mirror_ind] = [right_value, left_value]
    part.sensitivity_matrix = sensitivity
    part.resistance_matrix = np.eye(n_vertices)


def _minimal_result_with_suppression_points() -> CoilGenResult:
    coords = np.array(
        [
            [0.0, -0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, -2.0],
        ],
        dtype=float,
    )
    target_b = np.zeros((4, 3), dtype=float)
    target_b[:, 2] = [-0.5, 0.5, 0.0, 0.0]
    suppressed = np.array([False, False, True, True], dtype=bool)
    target = TargetField(
        coords=coords,
        b=target_b,
        weights=np.ones(4, dtype=float),
        suppressed=suppressed,
        target_field_group_inds=np.array([1, 1, 2, 2], dtype=np.int64),
    )
    field = target_b.copy()
    field[:, 2] = [-0.5, 0.5, 10.0, -10.0]
    mesh = Mesh(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
    )
    part = CoilPart(mesh=mesh)
    return CoilGenResult(
        coil_parts=[part],
        num_levels=0,
        potential_step=1.0,
        primary_surface=0,
        target_field=target,
        field_by_layout=field.copy(),
        field_by_unconnected_loops=field.copy(),
        field_layout_per1Amp=field.copy(),
        field_loops_per1Amp=field.copy(),
        needed_current_layout=1.0,
        b_field_opt_sf=field.copy(),
        b_field_opt_sf_1A=field.copy(),
        regularization_factor=1.0,
        regularization_lambda=1.0,
        regularization_diagnostics={},
        target_field_1A=target,
        error_vals={
            "mean_rel_error_layout_vs_target": 0.0,
            "max_rel_error_layout_vs_target": 0.0,
        },
        input_data=CoilGenConfig(target_gradient_strength=1.0),
        layout_gradient=None,
        combined_mesh=mesh,
        inductance_matrix=np.array([[1.0]], dtype=float),
        resistance=np.array([1.0], dtype=float),
        coil_lengths=np.array([1.0], dtype=float),
        coil_cross_sections=np.array([1.0], dtype=float),
    )


def _result_from_bz_values(coords: np.ndarray, target_bz: np.ndarray, layout_bz: np.ndarray) -> CoilGenResult:
    coords = np.asarray(coords, dtype=float)
    target_bz = np.asarray(target_bz, dtype=float).reshape(-1)
    layout_bz = np.asarray(layout_bz, dtype=float).reshape(-1)
    target_b = np.zeros((len(coords), 3), dtype=float)
    target_b[:, 2] = target_bz
    layout_b = np.zeros((len(coords), 3), dtype=float)
    layout_b[:, 2] = layout_bz
    target = TargetField(
        coords=coords,
        b=target_b,
        weights=np.ones(len(coords), dtype=float),
        suppressed=np.zeros(len(coords), dtype=bool),
        target_field_group_inds=np.ones(len(coords), dtype=np.int64),
    )
    mesh = Mesh(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
    )
    return CoilGenResult(
        coil_parts=[CoilPart(mesh=mesh)],
        num_levels=0,
        potential_step=1.0,
        primary_surface=0,
        target_field=target,
        field_by_layout=layout_b.copy(),
        field_by_unconnected_loops=layout_b.copy(),
        field_layout_per1Amp=layout_b.copy(),
        field_loops_per1Amp=layout_b.copy(),
        needed_current_layout=1.0,
        b_field_opt_sf=layout_b.copy(),
        b_field_opt_sf_1A=layout_b.copy(),
        regularization_factor=1.0,
        regularization_lambda=1.0,
        regularization_diagnostics={},
        target_field_1A=target,
        error_vals={},
        input_data=CoilGenConfig(field_display_units="T", target_gradient_strength=1.0),
        layout_gradient=None,
        combined_mesh=mesh,
        inductance_matrix=np.array([[1.0]], dtype=float),
        resistance=np.array([1.0], dtype=float),
        coil_lengths=np.array([1.0], dtype=float),
        coil_cross_sections=np.array([1.0], dtype=float),
    )


def _grid_mesh(nx: int, ny: int) -> Mesh:
    vertices = np.array([(float(x), float(y), 0.0) for y in range(ny) for x in range(nx)], dtype=float)
    faces = []
    for y in range(ny - 1):
        for x in range(nx - 1):
            a = y * nx + x
            b = a + 1
            c = a + nx
            d = c + 1
            faces.append((a, b, d))
            faces.append((a, d, c))
    return Mesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64))


def _mean_face_normal_z_at(mesh: Mesh, z_value: float) -> float:
    face_centers = mesh.vertices[mesh.faces].mean(axis=1)
    face_inds = np.flatnonzero(np.isclose(face_centers[:, 2], float(z_value), rtol=0.0, atol=1e-12))
    normals = []
    for face_ind in face_inds:
        points = mesh.vertices[mesh.faces[face_ind]]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        norm = float(np.linalg.norm(normal))
        if norm > 0.0:
            normals.append(normal / norm)
    if not normals:
        raise AssertionError(f"No faces found at z={z_value}.")
    return float(np.mean(np.asarray(normals)[:, 2]))


def _radial_contour_part(levels: tuple[float, ...]) -> CoilPart:
    axis = np.linspace(-1.0, 1.0, 37)
    vertices = np.array([(x, y, 0.0) for y in axis for x in axis], dtype=float)
    faces = []
    width = len(axis)
    for row in range(width - 1):
        for col in range(width - 1):
            a = row * width + col
            b = a + 1
            c = a + width
            d = c + 1
            faces.append((a, b, d))
            faces.append((a, d, c))
    mesh = normalize_mesh(Mesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64)))
    part = CoilPart(mesh=mesh)
    part.stream_function = np.sum(mesh.vertices[:, :2] ** 2, axis=1)
    part.potential_level_list = np.asarray(levels, dtype=float)
    part.contour_step = float(np.diff(part.potential_level_list).mean()) if len(levels) > 1 else 1.0
    return extract_contours([part], CoilGenConfig(min_point_loop_number=8))[0]


def _cylinder_z_ring_part(*, level_count: int) -> CoilPart:
    mesh = normalize_mesh(build_cylinder_mesh(0.4, 0.2, 24, 8, 1, 0, 0, 0))
    part = CoilPart(mesh=mesh)
    z_values = mesh.vertices[:, 2]
    z_min = float(np.min(z_values))
    z_span = float(np.ptp(z_values))
    fractions = np.linspace(0.25, 0.75, level_count)
    part.stream_function = z_values
    part.potential_level_list = np.asarray([z_min + fraction * z_span for fraction in fractions], dtype=float)
    part.contour_step = float(np.diff(part.potential_level_list).mean()) if level_count > 1 else 1.0
    part = extract_contours([part], CoilGenConfig(min_point_loop_number=8))[0]
    if len(part.contour_lines) != level_count:
        raise AssertionError(f"Expected {level_count} cylinder ring contours, got {len(part.contour_lines)}.")
    return part


def _cylindrical_two_ring_connection_part() -> tuple[CoilPart, list[ConnectedGroup]]:
    radius = 0.25
    samples = 28
    mesh = normalize_mesh(build_cylinder_mesh(0.5, radius, samples, 4, 1, 0, 0, 0))
    part = CoilPart(mesh=mesh)
    rings = [_cylindrical_ring_path(radius, -0.04, samples), _cylindrical_ring_path(radius, 0.04, samples)]
    part.contour_lines = [ContourLine(v=ring, potential=float(index)) for index, ring in enumerate(rings)]
    part.groups = [LoopGroup(loops=[contour], loop_indices=[index]) for index, contour in enumerate(part.contour_lines)]
    return_paths = np.zeros((0, 3), dtype=float)
    groups = [ConnectedGroup(v=ring, return_path_v=return_paths, spiral_in_v=ring, spiral_out_v=ring) for ring in rings]
    return part, groups


def _cylindrical_ring_path(radius: float, z: float, samples: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    points = np.column_stack(
        (
            radius * np.sin(angles),
            radius * np.cos(angles),
            np.full(samples, z, dtype=float),
        )
    )
    return np.vstack((points, points[:1]))


def _single_triangle_contour() -> tuple[Mesh, ContourLine]:
    mesh = Mesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
    )
    edge_vertices = np.array([[0, 1], [1, 2], [2, 0], [0, 1]], dtype=np.int64)
    edge_t = np.array([0.2, 0.2, 0.2, 0.2], dtype=float)
    points = mesh.vertices[edge_vertices[:, 0]] + edge_t[:, None] * (
        mesh.vertices[edge_vertices[:, 1]] - mesh.vertices[edge_vertices[:, 0]]
    )
    b0 = np.array([0.8, 0.2, 0.0], dtype=float)
    b1 = np.array([0.0, 0.8, 0.2], dtype=float)
    b2 = np.array([0.2, 0.0, 0.8], dtype=float)
    bary = np.vstack((b0, b1, b2, b0))
    return mesh, ContourLine(
        v=points,
        potential=0.0,
        surface_faces=np.zeros(4, dtype=np.int64),
        surface_barycentric=bary,
        surface_edge_vertices=edge_vertices,
        surface_edge_t=edge_t,
        segment_faces=np.zeros(3, dtype=np.int64),
        segment_barycentric=np.asarray([[b0, b1], [b1, b2], [b2, b0]], dtype=float),
    )


def _circle_loop(radius: float, samples: int) -> ContourLine:
    angle = np.linspace(0.0, 2.0 * np.pi, samples)
    points = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), np.zeros_like(angle)))
    return ContourLine(v=points, potential=0.0)


def _square_loop_at(center_x: float, center_y: float, width: float, z: float) -> ContourLine:
    half = width / 2.0
    points = np.array(
        [
            [center_x - half, center_y - half, z],
            [center_x + half, center_y - half, z],
            [center_x + half, center_y + half, z],
            [center_x - half, center_y + half, z],
            [center_x - half, center_y - half, z],
        ],
        dtype=float,
    )
    return ContourLine(v=points, potential=z)


def _mobius_strip_mesh(segments: int = 7, half_width: float = 0.12) -> Mesh:
    vertices = []
    for idx in range(segments):
        theta = 2.0 * np.pi * idx / segments
        center = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=float)
        radial = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=float)
        vertical = np.array([0.0, 0.0, 1.0], dtype=float)
        offset_direction = np.cos(theta / 2.0) * radial + np.sin(theta / 2.0) * vertical
        vertices.append(center - half_width * offset_direction)
        vertices.append(center + half_width * offset_direction)

    def vid(segment: int, side: int) -> int:
        return 2 * (segment % segments) + side

    faces = []
    for idx in range(segments):
        if idx == segments - 1:
            next_low = vid(0, 1)
            next_high = vid(0, 0)
        else:
            next_low = vid(idx + 1, 0)
            next_high = vid(idx + 1, 1)
        low = vid(idx, 0)
        high = vid(idx, 1)
        faces.append((low, next_low, next_high))
        faces.append((low, next_high, high))
    return Mesh(vertices=np.asarray(vertices, dtype=float), faces=np.asarray(faces, dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
