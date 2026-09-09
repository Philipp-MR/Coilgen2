"""Compatibility records captured before consolidating the public adapters."""

from dataclasses import fields
import inspect

import pytest

import coilgen
import coilgen.api as api
import coilgen.designs as designs
from coilgen.config import CoilGenConfig
from coilgen.problem import PipelineGeometrySpec


# Keep the curated API's exact ordering as well as its membership.
EXPECTED_API_EXPORTS = """
CoilGenConfig CoilApplication CoilDesignContext EvaluationFeature design_context_key
normalize_coil_application normalize_design_context normalize_evaluation_feature CoilGenResult run_coilgen
save_plot_gallery save_shim_homogeneity_mosaic save_shim_pointcloud_before_after create_swept_contour_meshes
manufacturing_diagnostics_rows mesh_manufacturing_diagnostics save_contour_traces_as_step
save_manufacturing_diagnostics save_mesh_as_step save_mesh_as_stl save_swept_bodies_as_step
save_swept_bodies_as_stl save_wire_trace_artifacts save_wire_trace_as_csv save_wire_trace_as_step
wire_trace_rows plot_result_matplotlib plot_result_matplotlib_book plot_field_spherical_harmonics
plot_shim_before_after_mosaics plot_shim_field_histograms plot_shim_homogeneity_mosaic
plot_shim_pointclouds_3d plot_shim_spherical_harmonics plot_matplotlib_figures_book
plot_layout_field_overview_3d plot_problem_setup_geometry_3d close_matplotlib_plot_books
plot_performance_table_matplotlib MatplotlibPlotBook QtMatplotlibPlotBook ImageMatplotlibPlotBook
MatplotlibPlotBookBuilder shim_field_scalar_maps coil_field_spherical_harmonic_maps
coil_spherical_harmonic_coefficient_rows coil_spherical_harmonic_fits shim_homogeneity_stats
shim_spherical_harmonic_comparison shim_spherical_harmonic_fits interpolated_slice_mosaic InterpolatedSlice
slice_mosaic_selections SphericalHarmonicFit SphericalHarmonicTerm compare_spherical_harmonic_fits
fit_spherical_harmonics solid_spherical_harmonic_basis solid_spherical_harmonic_terms
spherical_harmonic_coefficient_rows build_mesh_topology_suppression_surface mirror_symmetric_mesh_from_half
ConcomitantFieldReport CurrentDensityReport ReadinessReport ThermalScreeningReport mesh_surface_diagnostics
mesh_quality_report MeshQualityReport coil_design_problem_to_config DesignMetadata SuppressionRegion
TargetFieldSpec ConductorSpec SupportMesh CoilSupportSpec OptimizationSpec PipelineGeometrySpec LayoutSpec
EvaluationSpec CoilDesignProblem CoilChannelSpec CoilChannel CoilChannelRun CoilProject CoilProjectRun
CoilProjectParameters GradientLinearitySpec gradient_linearity_summary CoilParameterSection MeshParameters
TargetParameters OptimizationParameters ConductorParameters SystemParameters NumericsParameters
plot_project_run_book write_project_summary MetricDefinition MetricRegistry DEFAULT_METRIC_DEFINITIONS
DEFAULT_METRIC_REGISTRY RECOMMENDED_SETTINGS_PRESETS RecommendedSettingsPreset apply_recommended_settings
recommended_settings SolverResult CoilDesignSolver PipelineCoilDesignSolver HardwareSpec
HardwareCompilerOptions ProblemAnalysis ContinuousCandidateMetrics ContinuousCandidate HardwareCompilerReport
DiscreteHardwareMetrics HardwareConstrainedCoilDesignSolver analyze_stream_function_problem
compile_stream_function_problem compile_problem_to_hardware_config evaluate_discrete_hardware
register_in_memory_mesh get_in_memory_mesh register_in_memory_target_field get_in_memory_target_field
clear_in_memory_objects MatplotlibPlotSpec PlotBookEntry PlotBookPage StudyPlotPage StudyPlotSection
StudyPlotBookDefinition plot_study_plot_book PROTON_GYROMAGNETIC_RATIO_HZ_PER_T normalize_field_units
field_values_to_tesla field_values_from_tesla field_unit_label config_target_field_units
config_field_display_units ParameterAxis ParameterStudySpec DesignStudyCase DesignStudyResult DesignStudyTable
StudyMetric ConfigValidationMessage ConfigValidationReport validate_config CoilDesign CoilGenConfigDesign
CylinderCoilDesign PlanarCoilDesign CircularDiscCoilDesign AnnularDiscCoilDesign PartialCylinderCoilDesign
EllipticalCylinderCoilDesign SurfaceOfRevolutionCoilDesign SuperellipseCylinderCoilDesign
SphericalCapCoilDesign EllipsoidalCapCoilDesign ConicalFrustumCoilDesign ToroidalSegmentCoilDesign
RuledSurfaceCoilDesign ParametricSurfaceCoilDesign ShieldedCoilDesign BiplanarCoilDesign
CircularBiplanarCoilDesign ShieldedCircularBiplanarCoilDesign FlangedSplitBoreCoilDesign
ShieldedFlangedSplitBoreCoilDesign ImportedMeshCoilDesign coil_geometry_api_template
write_coil_geometry_api_template ResultQuality concomitant_field_report_dict concomitant_field_summary_dict
current_density_report_dict current_density_summary_dict readiness_report_dict readiness_summary_dict
result_quality_status result_summary_dict thermal_screening_summary_dict write_design_report
write_design_summary_csv REPORT_SCHEMA_VERSION REQUIRED_MARKDOWN_SECTIONS REQUIRED_SUMMARY_KEYS
ReportSchemaValidation validate_markdown_sections validate_summary_schema CoilDesignRunArtifacts
CoilDesignRunOptions run_coil_design run_user_design ActiveShieldBenchmarkResult BiplanarGradientScalingResult
CircularLoopValidationResult SolenoidValidationResult anti_helmholtz_biplanar_gradient_per_amp
circular_loop_bz_on_axis_per_amp validate_active_shield_suppression validate_biplanar_gradient_scaling
validate_circular_loop_biot_savart validate_solenoid_against_analytical_formulas
""".split()

# Deliberately distinct, nondefault values catch axis/parameter reordering.
GEOMETRY_INPUTS = {
    "angular_divisions": 5,
    "azimuthal_divisions": 6,
    "cap_polar_angle_rad": 0.039,
    "center_m": (0.012, -0.023, 0.034),
    "circular_divisions": 9,
    "edge_length_m": 0.078,
    "end_angle_rad": 0.091,
    "exponent": 0.104,
    "height_m": 0.117,
    "inner_radius_m": 0.13,
    "lateral_divisions": 15,
    "length_divisions": 16,
    "length_m": 0.169,
    "longitudinal_divisions": 18,
    "major_divisions": 19,
    "major_radius_m": 0.208,
    "minor_divisions": 21,
    "minor_radius_m": 0.234,
    "normal": (0.2, 0.3, 0.4),
    "outer_radius_m": 0.26,
    "plane_distance_m": 0.273,
    "polar_divisions": 26,
    "primary_edge_length_m": 0.299,
    "primary_outer_radius_m": 0.312,
    "radial_divisions": 29,
    "radius_bottom_m": 0.338,
    "radius_m": 0.351,
    "radius_top_m": 0.364,
    "radius_x_m": 0.377,
    "radius_y_m": 0.39,
    "radius_z_m": 0.403,
    "rotation_angle_rad": 0.416,
    "rotation_axis": (0.2, 0.3, 0.4),
    "shield_edge_length_m": 0.442,
    "shield_gap_m": 0.455,
    "shield_outer_radius_m": 0.468,
    "start_angle_rad": 0.481,
    "tube_end_angle_rad": 0.494,
    "tube_start_angle_rad": 0.507,
    "width_m": 0.52,
    "z_radius_pairs_m": ((-0.21, 0.14), (0.01, 0.18), (0.26, 0.13)),
}

# Old implementation output: class, factory, mesh name, cylinder flag,
# parameter key, nondefault tuple, default tuple.
GEOMETRY_CASES = [
    (
        "CylinderCoilDesign",
        "cylinder",
        "create cylinder mesh",
        True,
        "cylinder_mesh_parameter_list",
        (0.169, 0.351, 5, 16, 0.2, 0.3, 0.4, 0.416),
        (0.5, 0.25, 40, 40, 1.0, 0.0, 0.0, 0.0),
    ),
    (
        "PlanarCoilDesign",
        "planar",
        "create planary mesh",
        False,
        "planar_mesh_parameter_list",
        (0.117, 0.52, 15, 18, 0.2, 0.3, 0.4, 0.416, 0.012, -0.023, 0.034),
        (0.25, 0.25, 40, 40, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    (
        "CircularDiscCoilDesign",
        "circular",
        "create circular mesh",
        False,
        "circular_mesh_parameter_list",
        (0.351, 29, 0.2, 0.3, 0.4, 0.416, 0.012, -0.023, 0.034),
        (0.25, 40, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    (
        "AnnularDiscCoilDesign",
        "annular_disc",
        "create annular disc mesh",
        False,
        "annular_disc_mesh_parameter_list",
        (0.13, 0.26, 29, 9, 0.2, 0.3, 0.4, 0.012, -0.023, 0.034),
        (0.08, 0.25, 6, 48, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    ),
    (
        "PartialCylinderCoilDesign",
        "partial_cylinder",
        "create partial cylinder mesh",
        True,
        "partial_cylinder_mesh_parameter_list",
        (0.169, 0.351, 0.481, 0.091, 5, 16, 0.2, 0.3, 0.4, 0.416, 0.012, -0.023, 0.034),
        (0.5, 0.25, -1.5707963267948966, 1.5707963267948966, 36, 24, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    (
        "EllipticalCylinderCoilDesign",
        "elliptical_cylinder",
        "create elliptical cylinder mesh",
        True,
        "elliptical_cylinder_mesh_parameter_list",
        (0.169, 0.377, 0.39, 5, 16, 0.2, 0.3, 0.4, 0.416, 0.012, -0.023, 0.034),
        (0.5, 0.28, 0.18, 48, 24, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    (
        "SurfaceOfRevolutionCoilDesign",
        "surface_of_revolution",
        "create surface of revolution mesh",
        True,
        "surface_of_revolution_mesh_parameter_list",
        (9.0, -0.21, 0.14, 0.01, 0.18, 0.26, 0.13),
        (48.0, -0.25, 0.18, 0.0, 0.25, 0.25, 0.18),
    ),
    (
        "SuperellipseCylinderCoilDesign",
        "superellipse_cylinder",
        "create superellipse cylinder mesh",
        True,
        "superellipse_cylinder_mesh_parameter_list",
        (0.169, 0.377, 0.39, 0.104, 5, 16, 0.2, 0.3, 0.4, 0.416, 0.012, -0.023, 0.034),
        (0.5, 0.28, 0.18, 4.0, 48, 24, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    (
        "SphericalCapCoilDesign",
        "spherical_cap",
        "create spherical cap mesh",
        False,
        "spherical_cap_mesh_parameter_list",
        (0.351, 0.039, 6, 26, 0.2, 0.3, 0.4, 0.416, 0.012, -0.023, 0.034),
        (0.25, 1.5707963267948966, 48, 16, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    (
        "EllipsoidalCapCoilDesign",
        "ellipsoidal_cap",
        "create ellipsoidal cap mesh",
        False,
        "ellipsoidal_cap_mesh_parameter_list",
        (0.377, 0.39, 0.403, 0.039, 6, 26, 0.2, 0.3, 0.4, 0.416, 0.012, -0.023, 0.034),
        (0.25, 0.2, 0.18, 1.5707963267948966, 48, 16, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    (
        "ConicalFrustumCoilDesign",
        "conical_frustum",
        "create conical frustum mesh",
        True,
        "conical_frustum_mesh_parameter_list",
        (0.169, 0.338, 0.364, 5, 16, 0.2, 0.3, 0.4, 0.416, 0.012, -0.023, 0.034),
        (0.5, 0.22, 0.16, 48, 24, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    (
        "ToroidalSegmentCoilDesign",
        "toroidal_segment",
        "create toroidal segment mesh",
        False,
        "toroidal_segment_mesh_parameter_list",
        (0.208, 0.234, 0.481, 0.091, 0.507, 0.494, 19, 21, 0.2, 0.3, 0.4, 0.416, 0.012, -0.023, 0.034),
        (
            0.35,
            0.08,
            -1.5707963267948966,
            1.5707963267948966,
            0.0,
            6.283185307179586,
            36,
            16,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    ),
    (
        "BiplanarCoilDesign",
        "biplanar",
        "create bi-planary mesh",
        False,
        "biplanar_mesh_parameter_list",
        (0.117, 0.52, 15, 18, 0.2, 0.3, 0.4, 0.012, -0.023, 0.034, 0.273),
        (0.25, 0.25, 40, 40, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.2),
    ),
    (
        "CircularBiplanarCoilDesign",
        "circular_biplanar",
        "create circular bi-planary mesh",
        False,
        "circular_biplanar_mesh_parameter_list",
        (0.078, 0.26, 0.2, 0.3, 0.4, 0.012, -0.023, 0.034, 0.273),
        (0.08, 0.7, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    ),
    (
        "ShieldedCircularBiplanarCoilDesign",
        "shielded_circular_biplanar",
        "create shielded bi-planary mesh",
        False,
        "shielded_circular_biplanar_mesh_parameter_list",
        (0.312, 0.299, 0.468, 0.442, 0.2, 0.3, 0.4, 0.012, -0.023, 0.034, 0.273, 0.455),
        (0.7, 0.08, 0.8, 0.08, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.08),
    ),
]


def test_curated_api_keeps_exports_attributes_and_object_identity():
    assert api.__all__ == EXPECTED_API_EXPORTS
    assert {name for name in dir(api) if not name.startswith("_")} == set(EXPECTED_API_EXPORTS) | {"annotations"}
    namespace = {}
    exec("from coilgen.api import *", namespace)
    assert set(namespace) - {"__builtins__"} == set(EXPECTED_API_EXPORTS)
    for name in EXPECTED_API_EXPORTS:
        assert getattr(api, name) is getattr(coilgen, name)
        assert namespace[name] is getattr(coilgen, name)
        imported = __import__("coilgen.api", fromlist=[name])
        assert getattr(imported, name) is getattr(coilgen, name)
    with pytest.raises(AttributeError):
        getattr(api, "not_a_coilgen_export")


SETTINGS_PROPERTIES = (
    "mesh_settings",
    "target_settings",
    "optimization_settings",
    "routing_settings",
    "performance_settings",
    "manufacturing_settings",
)


def test_settings_project_every_field_without_copying_or_cross_wiring():
    config = CoilGenConfig()
    for property_name in SETTINGS_PROPERTIES:
        view = getattr(config, property_name)
        for item in fields(view):
            setattr(config, item.name, object())
    for property_name in SETTINGS_PROPERTIES:
        view = getattr(config, property_name)
        assert view is not getattr(config, property_name)
        for item in fields(view):
            assert getattr(view, item.name) is getattr(config, item.name)


def test_settings_keep_mutable_aliases_and_as_dict_copies():
    points = [[0.0, 0.0], [0.001, 0.0], [0.0, 0.001]]
    physics = {"objective": "mixture", "objective_weights": {"power": 1.0}}
    config = CoilGenConfig(cross_sectional_points=points, research_physics=physics)
    manufacturing = config.manufacturing_settings
    optimization = config.optimization_settings
    copied = manufacturing.as_dict()
    points[0][0] = 0.002
    physics["objective_weights"]["power"] = 2.0
    assert manufacturing.cross_sectional_points is points
    assert optimization.research_physics is physics
    assert manufacturing.cross_sectional_points[0][0] == 0.002
    assert optimization.research_physics["objective_weights"]["power"] == 2.0
    assert copied["cross_sectional_points"][0][0] == 0.0


@pytest.mark.parametrize("classname,factory_name,mesh_name,cylinder,key,expected,default", GEOMETRY_CASES)
def test_generated_geometry_serialization_matches_pre_cleanup_outputs(
    classname, factory_name, mesh_name, cylinder, key, expected, default
):
    design_class = getattr(designs, classname)
    factory = getattr(PipelineGeometrySpec, factory_name)
    kwargs = {name: GEOMETRY_INPUTS[name] for name in inspect.signature(factory).parameters}
    expected_config = {"coil_mesh_file": mesh_name, "surface_is_cylinder_flag": cylinder, key: expected}
    assert design_class(**kwargs).geometry_config() == expected_config
    assert factory(**kwargs).to_config_overrides() == expected_config
    assert isinstance(design_class(**kwargs).geometry_config()[key], tuple)
    assert design_class().geometry_config() == {
        "coil_mesh_file": mesh_name,
        "surface_is_cylinder_flag": cylinder,
        key: default,
    }
    for name in set(kwargs) & {"rotation_axis", "normal", "center_m", "z_radius_pairs_m"}:
        # The former tuple encoders also rejected noniterable sequences.
        with pytest.raises(TypeError):
            design_class(**{**kwargs, name: None}).geometry_config()
    for name in set(kwargs) & {"rotation_axis", "normal", "center_m"}:
        sequence = list(kwargs[name])
        design = design_class(**{**kwargs, name: sequence})
        encoded = design.geometry_config()
        sequence[0] += 0.1
        assert getattr(design, name) is sequence
        assert encoded == expected_config
        assert design.geometry_config() != encoded
