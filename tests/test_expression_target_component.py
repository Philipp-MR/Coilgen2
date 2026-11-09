"""A field expression never silently becomes an all-zero target for another component."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.application.presets import preset_config
from coilgen.config_validation import validate_config
from coilgen.field_components import field_component_values
from coilgen.geometry import load_mesh_from_config, split_disconnected_mesh
from coilgen.research.application import _target_analysis_recipe
from coilgen.research.requests import ResearchRequest
from coilgen.research.validation import sample_independent_target
from coilgen.target import define_target_field


def _target(**changes):
    config = replace(preset_config("cylinder"), field_shape_function="y", **changes)
    parts = split_disconnected_mesh(load_mesh_from_config(config, None))
    return config, define_target_field(parts, None, None, config)


def _component_errors(config):
    return [item for item in validate_config(config).errors if item.field == "target_field_component"]


@pytest.mark.parametrize("component", ["Bz", "vector"])
def test_bz_expression_is_preserved_for_scalar_and_full_vector_targets(component):
    config, target = _target(target_field_component=component)
    assert np.linalg.norm(target.b[:, 2]) > 0
    np.testing.assert_array_equal(target.b[:, :2], 0.0)
    np.testing.assert_array_equal(target.b, _target()[1].b)
    assert not _component_errors(config)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_b_parallel_b0_along_z_carries_the_expression(sign):
    config, target = _target(target_field_component="BdotB0", b_0_direction=(0.0, 0.0, 2.0 * sign))
    reference = _target()[1]
    useful = field_component_values(target.b, "BdotB0", b0_direction=config.b_0_direction)
    np.testing.assert_allclose(useful, reference.b[:, 2], atol=1e-18)
    assert not _component_errors(config)


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("normalization", ["none", "linear_fit"])
def test_independent_target_preserves_selected_component_and_normalization(sign, normalization):
    config, target = _target(
        target_field_component="BdotB0",
        b_0_direction=(0.0, 0.0, sign),
        target_gradient_normalization=normalization,
        target_gradient_strength=0.02,
        set_roi_into_mesh_center=False,
    )
    result = SimpleNamespace(input_data=config, regularization_diagnostics=target.target_diagnostics)
    independent = sample_independent_target(result, resolution=6)
    scale = 0.02 if normalization == "linear_fit" else 1.0
    for sampled in (target, independent):
        useful = field_component_values(sampled.b, "BdotB0", b0_direction=config.b_0_direction)
        np.testing.assert_allclose(useful, scale * sampled.coords[:, 1], atol=1e-18)
    np.testing.assert_allclose(target.target_gradient_dbdxyz[:, 1], sign * scale)
    recipe = _target_analysis_recipe(ResearchRequest.from_config(config), target)["target_recipe"]
    assert recipe["expression_bz_sign"] == sign


def test_legacy_independent_target_retains_cartesian_bz_expression():
    config = replace(
        preset_config("cylinder"),
        field_shape_function="y",
        target_field_component="BdotB0",
        b_0_direction=(0.0, 0.0, -1.0),
        set_roi_into_mesh_center=False,
    )
    diagnostics = {"target_gradient_normalization_scale": 0.02}
    result = SimpleNamespace(input_data=config, regularization_diagnostics=diagnostics)
    independent = sample_independent_target(result, resolution=6)
    np.testing.assert_allclose(independent.b[:, 2], 0.02 * independent.coords[:, 1], atol=1e-18)


@pytest.mark.parametrize(
    "changes",
    [
        {"target_field_component": "BdotB0", "b_0_direction": (1.0, 0.0, 0.0)},
        {"target_field_component": "Bx"},
        {"target_field_component": "By"},
    ],
)
def test_other_components_are_rejected_instead_of_zero(changes):
    config = replace(preset_config("cylinder"), field_shape_function="y", **changes)
    errors = _component_errors(config)
    assert errors and "rotate the support" in errors[0].suggestion
    with pytest.raises(ValueError, match="cannot be set by a field expression"):
        _target(**changes)


def test_imported_maps_keep_their_components():
    config = replace(
        preset_config("cylinder"), target_field_component="Bx", target_field_definition_file="inputs/map.csv"
    )
    assert not _component_errors(config)
