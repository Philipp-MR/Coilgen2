"""Rectangular openings in generated cylinders: exact edges, area, topology and a sheet solve."""

import math

import numpy as np
import pytest

from coilgen.application.cylinder_pair import format_openings, parse_openings
from coilgen.application.gradient_geometry import CylinderOpening, CylinderSurface, CylindricalGeometry
from coilgen.application.presets import preset_config
from coilgen.application.project import ProjectDocument
from coilgen.application.service import DesignService
from coilgen.geometry import load_mesh_from_config, split_disconnected_mesh
from coilgen.pipeline_stages import PipelineCheckpoint

WINDOW = CylinderOpening(math.pi / 2, math.radians(40), 0.0, 0.08)


def _area(mesh):
    triangles = mesh.vertices[mesh.faces]
    return (
        0.5
        * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1).sum()
    )


def _polygon_side_area(radius, length, angles):
    """Area of the inscribed polygonal tube between sorted angles (one wrap)."""
    steps = np.diff(np.concatenate((angles, angles[:1] + 2 * math.pi)))
    return float(np.sum(2 * radius * np.sin(steps / 2)) * length)


def test_opening_edges_lie_on_nodes_and_add_one_boundary_loop():
    surface = CylinderSurface(0.08, 0.3, 16, 8, openings=(WINDOW,))
    mesh = surface.mesh()
    assert len(mesh.boundaries) == 3
    loops = sorted(mesh.boundaries, key=len)
    window = mesh.vertices[loops[0]]
    phi = np.degrees(np.arctan2(window[:, 1], window[:, 0]))
    # Exact edges, not a staircase: every loop vertex is on one of the four sides.
    on_side = np.isclose(phi, 70, atol=1e-9) | np.isclose(phi, 110, atol=1e-9)
    on_end = np.isclose(window[:, 2], -0.04, atol=1e-14) | np.isclose(window[:, 2], 0.04, atol=1e-14)
    assert np.all(on_side | on_end)
    assert phi.min() == pytest.approx(70) and phi.max() == pytest.approx(110)
    np.testing.assert_allclose(np.linalg.norm(mesh.vertices[:, :2], axis=1), 0.08, atol=1e-14)
    assert np.ptp(mesh.vertices[:, 2]) == pytest.approx(0.3)
    assert len(split_disconnected_mesh(mesh)) == 1


def test_area_drops_by_the_window_and_two_windows_add_two_loops():
    plain = CylinderSurface(0.08, 0.3, 16, 8).mesh()
    surface = CylinderSurface(0.08, 0.3, 16, 8, openings=(WINDOW,))
    mesh = surface.mesh()
    end = mesh.vertices[np.isclose(mesh.vertices[:, 2], -0.15)]
    angles = np.sort(np.arctan2(end[:, 1], end[:, 0]) % (2 * math.pi))
    steps = np.diff(np.concatenate((angles, angles[:1] + 2 * math.pi)))
    chords = 2 * 0.08 * np.sin(steps / 2)
    middles = angles + steps / 2
    inside = np.abs(middles - math.pi / 2) < math.radians(20)
    # Polygonal tube over the refined columns, minus the flat window panels.
    assert np.sum(steps[inside]) == pytest.approx(math.radians(40))
    assert _area(mesh) == pytest.approx(chords.sum() * 0.3 - chords[inside].sum() * 0.08, rel=1e-12)
    assert _polygon_side_area(0.08, 0.3, angles) == pytest.approx(chords.sum() * 0.3)
    assert _area(mesh) < _area(plain)
    assert _area(mesh) == pytest.approx(2 * math.pi * 0.08 * 0.3 - 0.08 * math.radians(40) * 0.08, rel=0.02)
    second = CylinderOpening(-math.pi / 2, math.radians(40), 0.05, 0.06)
    assert len(CylinderSurface(0.08, 0.3, 16, 8, openings=(WINDOW, second)).mesh().boundaries) == 4


def test_openings_round_trip_through_project_metadata(tmp_path):
    primary = CylinderSurface(0.08, 0.3, 16, 8, azimuth_rad=0.3, openings=(WINDOW,))
    shield = CylinderSurface(0.11, 0.3, 16, 8, openings=(WINDOW,))
    geometry = CylindricalGeometry(primary, shield, CylinderSurface(0.15, 0.4, 24, 12))
    assert CylindricalGeometry.from_dict(geometry.to_dict()) == geometry
    # A surface without openings keeps its original portable form.
    assert "openings" not in CylinderSurface(0.08, 0.3).to_dict()
    project = geometry.apply(ProjectDocument.from_config(preset_config("cylinder"), project_root=tmp_path))
    reopened = ProjectDocument.load(project.save(tmp_path / "windows.coilgen.json"))
    assert CylindricalGeometry.from_project(reopened) == geometry
    mesh = load_mesh_from_config(reopened.config, tmp_path)
    assert sum(len(part.boundaries) for part in split_disconnected_mesh(mesh)) == 6
    assert mesh.metadata["part_roles"] == ["primary", "active_shield"]


def test_opening_text_round_trip_merges_both_surfaces():
    openings = parse_openings("90, 40, 0, 80, both; 270, 20, 50, 30, shield")
    assert openings["primary"] == (CylinderOpening(math.radians(90), math.radians(40), 0.0, 0.08),)
    assert len(openings["shield"]) == 2
    geometry = CylindricalGeometry(
        CylinderSurface(0.08, 0.3, 16, 8, openings=openings["primary"]),
        CylinderSurface(0.11, 0.3, 16, 8, openings=openings["shield"]),
    )
    assert format_openings(geometry) == "90, 40, 0, 80, both; -90, 20, 50, 30, shield"
    with pytest.raises(ValueError, match="no shield"):
        parse_openings("90, 40, 0, 80, shield", has_shield=False)
    with pytest.raises(ValueError, match="Opening 1"):
        parse_openings("90, 40, 0")


@pytest.mark.parametrize(
    "openings,match",
    (
        (((0, math.radians(350), 0, 0.05),), "full circumference"),
        (((0, 0.5, 0, 0.3),), "end"),
        (((0, 0.5, 0.13, 0.03),), "end"),
        (((0, 0.5, 0, 0.05), (0.6, 0.5, 0, 0.05)), "Openings 1 and 2"),
        (((0, 0.5, 0, 0.05), (0, 0.5, 0.06, 0.05)), "Openings 1 and 2"),
        (((0, -0.5, 0, 0.05),), "width must be positive"),
    ),
)
def test_openings_keep_ligaments_and_stay_inside(openings, match):
    with pytest.raises(ValueError, match=match):
        CylinderSurface(0.08, 0.3, 16, 8, openings=openings)


def test_closed_and_suppression_cylinders_have_no_openings():
    with pytest.raises(ValueError, match="closed cylinder"):
        CylinderSurface(0.08, 0.3, 16, 8, openings=(WINDOW,)).mesh(closed=True)
    with pytest.raises(ValueError, match="suppression cylinder"):
        CylindricalGeometry(CylinderSurface(0.08, 0.3), suppression=CylinderSurface(0.15, 0.4, openings=(WINDOW,)))


def test_sheet_solve_on_a_cylinder_with_one_window(tmp_path):
    project = ProjectDocument.from_config(preset_config("cylinder"), project_root=tmp_path)
    project = CylindricalGeometry(CylinderSurface(0.08, 0.3, 16, 8, openings=(WINDOW,))).apply(project)
    result = DesignService.run(
        project, tmp_path / "sheet", threads=1, pipeline_stage="sheet", fabricate=False, validation_resolution=8
    )
    assert result["summary"]["sheet_accepted"] is True
    assert result["validation"]["passed"] is True
    part = PipelineCheckpoint.load(tmp_path / "sheet" / "pipeline_checkpoint.cgstage").state.parts[0]
    assert len(part.mesh.boundaries) == 3
    # Each boundary loop carries one stream-function value (one degree of freedom).
    for loop in part.mesh.boundaries:
        values = part.stream_function[np.asarray(loop)]
        assert np.ptp(values) <= 1e-9 * max(np.ptp(part.stream_function), 1.0)
    assert np.ptp(part.stream_function) > 0
