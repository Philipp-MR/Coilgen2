"""Multipart single-circuit examples retain their required physical audit."""

from copy import deepcopy

from coilgen import BiplanarCoilDesign
from projects.general_examples.parameters.defaults import RUN_STANDARD_EXAMPLES_DEFAULT_PARAMETERS
from projects.general_examples.scripts import open_biplanar_gradient, run_standard_examples


def test_standard_biplanar_definition_and_fallback_enable_required_single_circuit_sweep():
    shipped = run_standard_examples.get_parameters()
    for parameters in (shipped, deepcopy(RUN_STANDARD_EXAMPLES_DEFAULT_PARAMETERS)):
        configs = dict(run_standard_examples.standard_example_configs(parameters))
        biplanar = configs["biplanar_ygradient_direct3d"]
        assert biplanar.routing_circuit_scope == "single"
        assert not biplanar.skip_postprocessing
        assert not biplanar.skip_sweep
        assert biplanar.biplanar_mesh_parameter_list == (0.25, 0.25, 10, 10, 1, 0, 0, 0, 0, 0, 0.2)
        assert biplanar.field_shape_function == "y"
        assert biplanar.conductor_cross_section_width == shipped["base_config"]["conductor_cross_section_width"]
        assert not biplanar.save_stl_flag
        assert configs["cylinder_ygradient_direct3d"].skip_sweep
        dense = dict(run_standard_examples.dense_generated_example_configs(parameters))
        split_bore = dense["split_bore_flange_ygradient_direct3d"]
        assert split_bore.routing_circuit_scope == "single"
        assert not split_bore.skip_postprocessing
        assert not split_bore.skip_sweep
        assert split_bore.flanged_split_bore_mesh_parameter_list == (0.12, 0.2, 0.28, 0.08, 0.2)
        assert split_bore.target_region_radius == 0.05
        assert split_bore.levels == 4


def test_open_biplanar_retains_single_circuit_and_original_field_geometry():
    config = BiplanarCoilDesign(**open_biplanar_gradient.PARAMETERS.design.to_dict()).to_config()
    assert config.routing_circuit_scope == "single"
    assert not config.skip_postprocessing
    assert not config.skip_sweep
    assert config.field_shape_function == "y"
    assert config.biplanar_mesh_parameter_list[-1] == 0.26
    assert config.target_region_radius == 0.07
    assert config.levels == 8
