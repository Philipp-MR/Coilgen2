"""Round wire point count, U-shape preset and u, n table import."""

import numpy as np
import pytest

from coilgen import CoilGenConfig, CoilPart, Mesh, WirePath, create_swept_coil_meshes
from coilgen.application.conductor_profiles import (
    equal_area_radius_scale,
    parse_profile_table,
    read_profile_table,
    round_wire_profile,
    u_shape_profile,
    validate_conductor_profile,
)
from coilgen.application.session import WorkbenchSession
from coilgen.inductance import conductor_cross_section_area


@pytest.mark.parametrize("segments", [6, 8, 12, 16, 64])
def test_round_wire_point_count_and_equal_area(segments):
    plain = validate_conductor_profile(round_wire_profile(0.002, segments))
    assert len(plain.points_m) == segments
    expected = segments / 2 * np.sin(2 * np.pi / segments) * 1e-6
    assert plain.area_m2 == pytest.approx(expected)
    assert plain.width_m <= 0.002 + 1e-15
    corrected = validate_conductor_profile(round_wire_profile(0.002, segments, equal_area=True))
    assert len(corrected.points_m) == segments
    assert corrected.area_m2 == pytest.approx(np.pi * 1e-6, rel=1e-12)
    assert corrected.centroid_m == pytest.approx((0, 0), abs=1e-15)
    assert equal_area_radius_scale(segments) > 1


def test_default_round_wire_is_unchanged():
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    expected = np.column_stack((np.cos(angles), np.sin(angles))) * 75e-5
    np.testing.assert_allclose(round_wire_profile(0.0015), expected, rtol=1e-15, atol=1e-18)
    # Trigonometric round-off on the axes is stored as exact zeros.
    assert round_wire_profile(0.0015)[4] == (0.0, 0.00075)


@pytest.mark.parametrize("segments", [5, 257, 8.5, "8", True, None])
def test_round_wire_rejects_bad_point_counts(segments):
    with pytest.raises(ValueError, match="circle points|points around a circle"):
        round_wire_profile(0.002, segments)


@pytest.mark.parametrize("arc_side", ["top", "bottom"])
@pytest.mark.parametrize("segments", [6, 8, 16, 32])
@pytest.mark.parametrize("height", [0.001, 0.0015, 0.003])
def test_u_shape_is_a_valid_rectangle_plus_half_circle(arc_side, segments, height):
    width = 0.002
    points = u_shape_profile(width, height, segments, arc_side=arc_side)
    summary = validate_conductor_profile(points)
    straight = height - width / 2
    arc_segments = segments // 2
    # Half polygon of the inscribed full circle plus the straight part.
    expected_area = width * straight + arc_segments / 2 * np.sin(np.pi / arc_segments) * (width / 2) ** 2
    assert summary.area_m2 == pytest.approx(expected_area)
    assert len(points) == arc_segments + (3 if straight > 0 else 1)
    assert summary.width_m == pytest.approx(width)
    flat = summary.bounds_min_m[1] if arc_side == "top" else summary.bounds_max_m[1]
    assert flat == pytest.approx(-height / 2 if arc_side == "top" else height / 2)
    if arc_segments % 2 == 0:  # an arc vertex sits on the apex
        assert summary.height_m == pytest.approx(height)
    on_flat_side = [point for point in points if point[1] == pytest.approx(flat)]
    assert len(on_flat_side) == 2


def test_u_shape_needs_height_of_at_least_the_radius():
    with pytest.raises(ValueError, match="at least half its width"):
        u_shape_profile(0.002, 0.0009)
    with pytest.raises(ValueError, match="arc side"):
        u_shape_profile(0.002, 0.002, arc_side="left")


@pytest.mark.parametrize("make_part", ["open", "closed"])
def test_u_shape_sweeps_watertight_with_its_area(make_part):
    support = Mesh(
        vertices=np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]), faces=np.array([[0, 1, 2], [0, 2, 3]])
    )
    if make_part == "open":
        route = np.array([[-0.01, 0, 0], [0.01, 0, 0]])
    else:
        angle = np.linspace(0, 2 * np.pi, 65)
        route = np.column_stack((0.03 * np.cos(angle), 0.03 * np.sin(angle), np.zeros_like(angle)))
        route[-1] = route[0]
    part = CoilPart(mesh=support, wire_path=WirePath(v=route), contour_step=1.0)
    profile = u_shape_profile(0.001, 0.0012, 8)
    config = CoilGenConfig(
        conductor_cross_section_width=0.001,
        conductor_cross_section_height=0.0012,
        sweep_max_chord_error=0,
        cross_sectional_points=profile,
    )
    create_swept_coil_meshes([part], config)
    faces = np.asarray(part.swept_body.faces)
    edges = np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    assert np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2)
    assert conductor_cross_section_area(config) == pytest.approx(validate_conductor_profile(profile).area_m2)


@pytest.mark.parametrize(
    "text",
    [
        "u;n\n0;0\n1,5;0\n1,5;2\n",  # German spreadsheet export
        "u [mm]\tn [mm]\n0\t0\n1.5\t0\n1.5\t2\n",
        "# exported profile\n0, 0\n1.5, 0\n1.5, 2\n\n",
        "0 0\n1.5 0\n1.5 2\n0 0\n",  # explicit closing point
    ],
)
def test_profile_table_formats(text):
    assert parse_profile_table(text) == ((0.0, 0.0), (0.0015, 0.0), (0.0015, 0.002))


def test_profile_table_units_and_errors(tmp_path):
    assert parse_profile_table("0 0\n1 0\n0 1", length_scale_to_m=1.0)[1] == (1.0, 0.0)
    with pytest.raises(ValueError, match="Line 3: expected two numbers"):
        parse_profile_table("u n\n0 0\nx 1\n1 1\n")
    with pytest.raises(ValueError, match="Line 2: expected two columns"):
        parse_profile_table("0 0\n1 0 5\n1 1\n")
    with pytest.raises(ValueError, match="at least three"):
        parse_profile_table("u;n\n0;0\n1;1\n")
    with pytest.raises(ValueError, match="finite"):
        parse_profile_table("0 0\n1 nan\n1 1\n")
    path = tmp_path / "profile.csv"
    path.write_bytes("u;n\n0;0\n2;0\n1;1,5\n".encode("latin-1"))
    assert read_profile_table(path) == ((0.0, 0.0), (0.002, 0.0), (0.001, 0.0015))


def test_session_presets_use_point_count_and_table_import(tmp_path):
    session = WorkbenchSession(tmp_path)
    session.load_template("cylinder")
    config = session.project.config
    session.apply_profile_preset("round", segments=8, equal_area=True)
    points = session.project.config.cross_sectional_points
    assert len(points) == 8
    assert validate_conductor_profile(points).area_m2 == pytest.approx(
        np.pi * (config.conductor_cross_section_width / 2) ** 2
    )
    session.apply_profile_preset("u_shape", segments=8)
    summary = validate_conductor_profile(session.project.config.cross_sectional_points)
    assert summary.height_m == pytest.approx(config.conductor_cross_section_height)
    table = tmp_path / "holder.txt"
    table.write_text("u n\n-1 -1\n1 -1\n1 1\n0 2\n-1 1\n", encoding="utf-8")
    session.import_profile_table(table)
    assert tuple(session.project.config.cross_sectional_points[3]) == (0.0, 0.002)
    crossing = tmp_path / "crossing.csv"
    crossing.write_text("0,0\n1,1\n0,1\n1,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="intersect"):
        session.import_profile_table(crossing)
    session.undo()
    assert len(session.project.config.cross_sectional_points) == len(summary.points_m)
