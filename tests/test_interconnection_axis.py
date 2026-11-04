"""Routing follows a configurable axis; field physics keeps the true B0."""

from dataclasses import replace
import math

import numpy as np
import pytest

from coilgen.application.presets import preset_config
from coilgen.application.templates import HALBACH_BORE_AXIS, HALBACH_INTERCONNECTION_AXIS, template_assembly
from coilgen.config import CoilGenConfig
from coilgen.config_validation import validate_config
from coilgen.geometry import load_mesh_from_config, split_disconnected_mesh
from coilgen.layout import interconnection_axis
from coilgen.models import Mesh
from coilgen.pipeline_stages import PipelineStage, config_field_stage
from coilgen.target import define_target_field


def _halbach_cylinder_config(axis_setting="b0"):
    base = preset_config("cylinder")
    length, radius, angular, axial = base.cylinder_mesh_parameter_list[:4]
    return replace(
        base,
        cylinder_mesh_parameter_list=(length, radius, angular, axial, 1, 0, 0, math.pi / 2),
        interconnection_axis=axis_setting,
    )


def _part(config):
    return split_disconnected_mesh(load_mesh_from_config(config, None))[0]


def test_default_axis_is_b0_for_every_support():
    config = replace(preset_config("cylinder"), b_0_direction=(0.0, 0.0, 2.0))
    np.testing.assert_allclose(interconnection_axis(config, _part(config)), (0.0, 0.0, 2.0))


def test_support_axis_follows_a_rotated_bore():
    config = _halbach_cylinder_config("support_axis")
    axis = interconnection_axis(config, _part(config))
    assert abs(float(axis @ np.asarray(HALBACH_BORE_AXIS))) == pytest.approx(1.0)


def test_explicit_vector_and_closed_support_fallback():
    config = CoilGenConfig(interconnection_axis=(1.0, 0.0, 0.0))
    np.testing.assert_allclose(interconnection_axis(config), (1.0, 0.0, 0.0))
    closed = Mesh(vertices=np.eye(3), faces=np.array([[0, 1, 2]]))
    config = CoilGenConfig(interconnection_axis="support_axis", b_0_direction=(0.0, 1.0, 0.0))
    np.testing.assert_allclose(interconnection_axis(config, closed), (0.0, 1.0, 0.0))


@pytest.mark.parametrize(
    ("value", "valid"),
    [("b0", True), ("support_axis", True), ((0, 1, 0), True), ("bore", False), ((0, 0, 0), False), ((1, 2), False)],
)
def test_interconnection_axis_validation(value, valid):
    errors = [item for item in validate_config(CoilGenConfig(interconnection_axis=value)).errors]
    assert (not any(item.field == "interconnection_axis" for item in errors)) is valid


def test_axis_is_a_routing_input_not_a_sheet_input():
    stage = config_field_stage("interconnection_axis", preset_config("cylinder"))
    assert stage in {PipelineStage.CONTOURS, PipelineStage.ROUTED}


@pytest.mark.parametrize("axis", "xyz")
def test_halbach_templates_keep_b0_physics_and_route_along_the_bore(axis):
    project = template_assembly(f"halbach_g{axis}").channels[0].project
    config = project.config
    mesh = load_mesh_from_config(config, project.project_root)
    extent = np.ptp(mesh.vertices, axis=0)
    assert int(np.argmax(extent)) == 1
    assert config.interconnection_axis == HALBACH_INTERCONNECTION_AXIS[axis]
    np.testing.assert_allclose(config.b_0_direction, (0.0, 0.0, 1.0))
    target = define_target_field(split_disconnected_mesh(mesh), None, None, config)
    np.testing.assert_allclose(target.b[:, 2], 0.02 * target.coords[:, "xyz".index(axis)], atol=1e-15)
    assert not validate_config(config).errors
