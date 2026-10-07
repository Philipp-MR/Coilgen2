"""Typed analytic-cylinder reference chords never relax original source checks."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

import coilgen.group_route_support as NEW
from coilgen.config import CoilGenConfig
from coilgen.connector_bridge_validation import _SpatialSupport
from coilgen.geometry import build_cylinder_mesh, normalize_mesh
from coilgen.manufacturing import _SurfaceProjector
from coilgen.models import WirePathGeometry

from test_group_route_support_bvh import cylinder, plane


def fixture(angle=0.0012, geometry=WirePathGeometry.SURFACE):
    config = CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        surface_is_cylinder_flag=True,
        cylinder_mesh_parameter_list=[0.06, 0.02, 48, 2, 0, 0, 1, 0],
        normal_shift_length=0.006,
        sweep_max_chord_error=5e-5,
    )
    origins = 0.02 * np.array([[np.cos(angle), -np.sin(angle), 0], [np.cos(angle), np.sin(angle), 0]])
    part = SimpleNamespace(mesh=cylinder(), fabrication_path=SimpleNamespace(closed=False, segment_geometry=[geometry]))
    diagnostics = {}
    allowances = NEW._analytic_surface_chord_allowances(part, config, origins, 1e-8, diagnostics)
    return part, config, origins, allowances, diagnostics


def checker(mesh):
    return NEW._CenteredSpatialSupport(_SurfaceProjector(mesh), 0.006, 1.0, 1e-8)


def test_verified_tiny_analytic_chord_facet_hit_has_chord_specific_deficit_and_unchanged_samples():
    part, _, origins, allowances, diagnostics = fixture()
    segments = origins[None]
    assert 1e-8 < allowances[0] < 2e-8
    strict = checker(part.mesh)
    typed = checker(part.mesh)
    assert strict.crosses_surface(segments)
    assert not typed.crosses_surface(segments, allowances)
    assert typed.analytic_tessellation_facet_hit_count > 0
    assert typed.validate(segments, analytic_chord_allowances=allowances) == []
    assert checker(part.mesh).validate(segments) == ["support_surface_intersection"]
    midpoint = origins.mean(axis=0)
    assert typed.sample(midpoint) == strict.sample(midpoint)
    assert typed.sample(midpoint)[2] < -1e-8
    assert diagnostics["analytic_reference_surface_chord_count"] == 1


@pytest.mark.parametrize(
    "defect",
    [
        "free",
        "off_radius_endpoint",
        "large_deficit",
        "counterfeit_mesh",
        "inward_mesh",
        "mesh_realization",
        "negative_lift",
    ],
)
def test_only_verified_typed_small_deficit_chords_receive_allowance(defect):
    part, config, origins, _, _ = fixture(0.1 if defect == "large_deficit" else 0.0012)
    if defect == "free":
        part.fabrication_path.segment_geometry = [WirePathGeometry.FREE_3D]
    elif defect == "off_radius_endpoint":
        origins[0] *= 1.00001
    elif defect == "counterfeit_mesh":
        part.mesh.vertices[0, 0] += 2e-8
    elif defect == "inward_mesh":
        part.mesh.faces = part.mesh.faces[:, ::-1]
    elif defect == "mesh_realization":
        config.sweep_surface_realization = "mesh"
    elif defect == "negative_lift":
        config.normal_shift_length = -0.006
    result = NEW._analytic_surface_chord_allowances(part, config, origins, 1e-8, {})
    assert result is None or np.array_equal(result, [0.0])


def test_same_raw_free_chord_and_coincident_raw_surface_vertex_remain_strict():
    part, _, origins, allowances, _ = fixture()
    midpoint = origins.mean(axis=0)
    raw_vertex = np.repeat(midpoint[None], 2, axis=0)
    combined = np.stack((origins, origins, raw_vertex))
    assert checker(part.mesh).crosses_surface(combined, [allowances[0], 0.0, 0.0])
    assert checker(part.mesh).validate(raw_vertex[None], analytic_chord_allowances=[0.0]) == ["lift_limit_violation"]


def test_completed_crossing_cache_isolated_by_chord_allowance_not_point_identity():
    part, _, origins, allowances, _ = fixture()
    probe = checker(part.mesh)
    assert not probe.crosses_surface(origins[None], allowances)
    work = probe.box_work
    assert not probe.crosses_surface(origins[None].copy(), allowances.copy())
    assert probe.box_work == work
    assert probe.crosses_surface(origins[None])
    assert probe.box_work > work


@pytest.mark.parametrize("kind", ["contact", "wrong_side", "boundary", "crossing"])
def test_optional_none_keeps_exact_original_strict_validation(kind):
    mesh = plane()
    points = np.array([[[-0.005, 0, 0.001], [0.005, 0, 0.001]]])
    if kind == "contact":
        points[:, :, 2] = 0
    elif kind == "wrong_side":
        points[:, :, 2] = -2e-8
    elif kind == "boundary":
        points[0, :, 0] = [0.027, 0.032]
    else:
        points[0, :, 2] = [-0.001, 0.001]
    old = _SpatialSupport(_SurfaceProjector(mesh), 0.006, 1.0, 1e-8)
    new = checker(mesh)
    assert new.validate(points, analytic_chord_allowances=None) == old.validate(points)
    assert (new.evaluations, new.minimum_height, new.maximum_height, new.maximum_distance) == (
        old.evaluations,
        old.minimum_height,
        old.maximum_height,
        old.maximum_distance,
    )


def test_actual_y_first_facet_hit_retains_typed_chord_scope():
    # Original standard-Y captured trace interval9, with exact original points.
    # Rebuild its unchanged support rather than depend on a whole-coil pickle.
    parameters = [0.25, 0.066, 28, 20, 1.0, 0.0, 0.0, 0.0]
    config = CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        surface_is_cylinder_flag=True,
        cylinder_mesh_parameter_list=parameters,
        normal_shift_length=0.006,
        sweep_max_chord_error=5e-5,
    )
    points = np.array(
        [
            [0.05944112831559965, -0.02868365849344216, 0.08501142259985849],
            [0.05947952534182201, -0.028603951914229493, 0.08493205194796993],
        ]
    )
    part = SimpleNamespace(
        mesh=normalize_mesh(build_cylinder_mesh(*parameters)),
        fabrication_path=SimpleNamespace(closed=False, segment_geometry=[WirePathGeometry.SURFACE]),
    )
    allowances = NEW._analytic_surface_chord_allowances(part, config, points, 1e-8, {})
    assert 1e-8 < allowances[0] < 2e-8
    strict = checker(part.mesh)
    assert strict.crosses_surface(points[None])
    typed = checker(part.mesh)
    assert not typed.crosses_surface(points[None], allowances)
    assert typed.analytic_tessellation_facet_hit_count > 0
    assert typed.validate(points[None], analytic_chord_allowances=allowances) == []
    altered = deepcopy(part)
    changed_points = points.copy()
    changed_points[0] *= 1.0001
    assert np.array_equal(NEW._analytic_surface_chord_allowances(altered, config, changed_points, 1e-8, {}), [0.0])
