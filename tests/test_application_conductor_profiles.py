"""Profile editing preserves physical local axes, original origin, and SI units."""

import numpy as np
import pytest

from coilgen.application import ProjectDocument
from coilgen.application.conductor_profiles import (
    canonical_profile_points,
    is_rectangular_profile,
    validate_conductor_profile,
)
from coilgen.application.presets import preset_config
from coilgen.config import CoilGenConfig
from coilgen.geometry import normalize_mesh
from coilgen.inductance import calculate_inductance_by_coil_layout, conductor_cross_section_area
from coilgen.manufacturing import conductor_cross_section_points, create_swept_coil_bodies
from coilgen.models import CoilPart, Mesh, WirePath


# Deliberately offset from the wire origin, asymmetric, and provided clockwise.
TRAPEZOID = np.array([[1, -1], [2, 1], [4, 1], [5, -1]], dtype=float) * 1e-3


@pytest.mark.parametrize("legacy_orientation", [False, True])
@pytest.mark.parametrize("closing_point", [False, True])
def test_profile_contract_preserves_order_and_origin(legacy_orientation, closing_point):
    supplied = np.vstack((TRAPEZOID, TRAPEZOID[0])) if closing_point else TRAPEZOID
    if legacy_orientation:
        supplied = supplied.T
    metrics = validate_conductor_profile(supplied)
    np.testing.assert_array_equal(metrics.points_m, TRAPEZOID)
    assert metrics.clockwise
    assert metrics.area_m2 == pytest.approx(6e-6)
    assert metrics.width_m == pytest.approx(4e-3)
    assert metrics.height_m == pytest.approx(2e-3)
    assert metrics.centroid_m == pytest.approx((3e-3, -1e-3 / 9))
    assert metrics.bounds_min_m == pytest.approx((1e-3, -1e-3))
    assert metrics.bounds_max_m == pytest.approx((5e-3, 1e-3))
    np.testing.assert_array_equal(canonical_profile_points(supplied), TRAPEZOID)
    config = CoilGenConfig(cross_sectional_points=metrics.points_m)
    assert conductor_cross_section_area(config) == pytest.approx(metrics.area_m2)
    np.testing.assert_array_equal(conductor_cross_section_points(config), TRAPEZOID[::-1])


def test_rectangle_sentinel_expands_only_for_metrics():
    sentinel = ((0.0, 0.0),)
    assert is_rectangular_profile(sentinel)
    assert canonical_profile_points(sentinel) == sentinel
    metrics = validate_conductor_profile(sentinel, width_m=4e-3, height_m=2e-3)
    assert metrics.area_m2 == pytest.approx(8e-6)
    assert metrics.centroid_m == (0.0, 0.0)
    assert metrics.bounds_min_m == (-2e-3, -1e-3)
    config = CoilGenConfig(conductor_cross_section_width=4e-3, conductor_cross_section_height=2e-3)
    np.testing.assert_array_equal(metrics.points_m, conductor_cross_section_points(config))
    assert not is_rectangular_profile([[0, 0], [1, 0]])
    assert not is_rectangular_profile([[0, np.nan]])
    assert not is_rectangular_profile("bad input")


@pytest.mark.parametrize("width,height", [(None, None), (0, 1), (1, -1), (np.inf, 1), (1, np.nan), ("bad", 1)])
def test_rectangle_requires_physical_dimensions(width, height):
    with pytest.raises(ValueError, match="positive finite"):
        validate_conductor_profile(((0, 0),), width_m=width, height_m=height)


@pytest.mark.parametrize(
    "points,message",
    [
        ([], "two coordinates"),
        ([[0, 0], [1, 0]], "at least three"),
        ([[0, 0], [1, 0], [np.nan, 1]], "finite"),
        ([[0, 0], [1, 0], [0, np.inf]], "finite"),
        ([[0, 0], [1, 0], ["bad", 1]], "numeric"),
        ([[0, 0], [1, 0], [1, 0], [0, 1]], "zero-length"),
        ([[0, 0], [1, 1], [0, 1], [1, 0]], "intersect"),
        # Collinear non-neighbouring segments overlap, without a proper crossing.
        ([[0, 0], [3, 0], [3, 2], [1, 0], [2, 0], [0, 2]], "overlap"),
        # A non-neighbouring edge touches another vertex.
        ([[0, 0], [2, 0], [2, 2], [1, 0], [0, 2]], "touch"),
        ([[0, 0], [1, 0], [2, 0]], "zero or numerically"),
        ([[0, 0], [2, 0], [1, 1], [2, 2], [0, 2]], "strictly convex"),
        ([[0, 0], [1, 0], [2, 0], [2, 2], [0, 2]], "collinear"),
        ([[-1e308, 0], [1e308, 0], [0, 1]], "span is too large"),
    ],
)
def test_invalid_geometry_has_actionable_error(points, message):
    with pytest.raises(ValueError, match=message):
        validate_conductor_profile(points)


def test_si_profile_reaches_sweep_centroid_and_resistance(tmp_path):
    config = preset_config("cylinder")
    project = ProjectDocument.from_config(config).update_config(
        {
            "cross_sectional_points": validate_conductor_profile(TRAPEZOID).points_m,
            "conductor_mounting_mode": "centered",
            "conductor_cross_section_width": 1e-3,
            "conductor_cross_section_height": 1e-3,
            "sweep_max_segment_length": 0.0,
            "sweep_bend_mode": "miter",
            "sweep_insulation_thickness": 0.1e-3,
            "inductance_source": "fabricated_centroid",
        }
    )
    path = project.save(tmp_path / "profile.coilgen.json")
    restored = ProjectDocument.load(path)
    assert restored.fingerprint == project.fingerprint
    np.testing.assert_array_equal(restored.values["cross_sectional_points"], TRAPEZOID)
    config = restored.config
    mesh = normalize_mesh(
        Mesh(
            np.array([[-0.1, -0.1, 0], [0.1, -0.1, 0], [0.1, 0.1, 0], [-0.1, 0.1, 0]]),
            np.array([[0, 1, 2], [0, 2, 3]]),
        )
    )
    route = np.array([[0, 0, 0], [0.06, 0, 0]], dtype=float)
    part = CoilPart(mesh=mesh, wire_path=WirePath(route))
    create_swept_coil_bodies([part], config)
    assert part.manufacturing_diagnostics["manufacturing_ok"]
    assert part.manufacturing_diagnostics["sweep_nominal_cross_section_area_m2"] == pytest.approx(6e-6)
    np.testing.assert_array_equal(part.fabrication_path.cross_section_m, TRAPEZOID[::-1])
    count = len(part.fabrication_path.points_m)
    np.testing.assert_allclose(part.fabrication_path.normals, np.tile([0, 0, 1], (count, 1)))
    np.testing.assert_allclose(part.fabrication_path.transverse, np.tile([0, -1, 0], (count, 1)))
    np.testing.assert_allclose(part.fabrication_path.points_m[[0, -1]], route + [0, -3e-3, -1e-3 / 9], atol=1e-15)
    np.testing.assert_allclose(part.fabrication_path.points_m[:, 1:], np.tile([-3e-3, -1e-3 / 9], (count, 1)))
    _, inductance, resistance, lengths, areas = calculate_inductance_by_coil_layout([part], config)
    assert areas[0] == pytest.approx(6e-6)
    assert lengths[0] == pytest.approx(0.06)
    assert resistance[0] == pytest.approx(config.specific_conductivity_conductor * 0.06 / 6e-6)
    assert inductance[0, 0] > 0
