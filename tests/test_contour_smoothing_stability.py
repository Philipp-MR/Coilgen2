from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from coilgen import CoilGenConfig, CoilPart, ContourLine
from coilgen.contour_smoothing import (
    MeshContourBSplineSmoother,
    _TopologyWorkLimit,
    _circular_cylinder_frame,
    _polyline_self_intersects,
)
from coilgen.contours import extract_contours
from coilgen.geometry import build_cylinder_mesh, normalize_mesh
from coilgen.progress import DesignCancelled, progress_context
from coilgen.config_validation import validate_config


def _seam_crossing_loop() -> tuple[CoilPart, ContourLine]:
    mesh = normalize_mesh(build_cylinder_mesh(0.2, 0.08, 28, 16, 1, 0, 0, 0))
    frame = _circular_cylinder_frame(mesh)
    assert frame is not None
    center, axis, first_radial, second_radial, _ = frame
    relative = mesh.vertices - center
    angle = np.arctan2(relative @ second_radial, relative @ first_radial)
    axial = relative @ axis
    part = CoilPart(mesh=mesh)
    part.stream_function = -np.cos(angle) * (1.0 - (axial / np.max(np.abs(axial))) ** 2)
    part.potential_level_list = np.array([0.3])
    extract_contours([part], CoilGenConfig(min_point_loop_number=8))
    assert len(part.contour_lines) == 1
    contour = part.contour_lines[0]
    contour_angle = np.arctan2((contour.v - center) @ second_radial, (contour.v - center) @ first_radial)
    assert np.any(np.abs(np.diff(contour_angle)) > np.pi)
    return part, contour


def _assert_surface_reconstruction(part: CoilPart, contour: ContourLine) -> None:
    assert contour.segment_faces is not None
    assert contour.segment_barycentric is not None
    triangles = part.mesh.vertices[part.mesh.faces[contour.segment_faces]]
    reconstructed = np.einsum("nki,nij->nkj", contour.segment_barycentric, triangles)
    endpoints = np.stack((contour.v[:-1], contour.v[1:]), axis=1)
    np.testing.assert_allclose(reconstructed, endpoints, rtol=0, atol=2.0e-12)
    np.testing.assert_allclose(contour.segment_barycentric.sum(axis=2), 1.0, rtol=0, atol=2.0e-14)
    assert np.min(contour.segment_barycentric) >= 0.0
    np.testing.assert_array_equal(contour.v[0], contour.v[-1])
    assert np.min(np.linalg.norm(np.diff(contour.v, axis=0), axis=1)) > 0.0


@pytest.mark.parametrize("reverse", [False, True])
def test_seam_crossing_smoothing_retains_circulation_and_exact_surface_geometry(reverse: bool) -> None:
    part, source = _seam_crossing_loop()
    if reverse:
        source = ContourLine(
            v=source.v[::-1],
            potential=source.potential,
            current_orientation=-1.0,
            segment_faces=source.segment_faces[::-1],
            segment_barycentric=source.segment_barycentric[::-1, ::-1],
            turn_multiplicity=3,
        )
    before = source.v.copy()
    result = MeshContourBSplineSmoother(part.mesh).smooth(
        source,
        smoothing_factor=0.0,
        scale_mode="mesh_coupled",
        mesh_gain=0.1,
        max_displacement_m=0.001,
        target_segment_length_m=0.001,
    )
    assert result.applied, result.reason
    assert result.parameterization == "developed_circular_cylinder"
    assert result.max_displacement_m <= 0.001 * (1.0 + 1.0e-6)
    assert result.contour.potential == source.potential
    assert result.contour.current_orientation == source.current_orientation
    assert result.contour.turn_multiplicity == source.turn_multiplicity
    np.testing.assert_array_equal(source.v, before)
    _assert_surface_reconstruction(part, result.contour)

    # Area in developed cylinder coordinates gives an independent circulation
    # check even when the geometry straddles atan2's angular branch cut.
    frame = _circular_cylinder_frame(part.mesh)
    assert frame is not None
    center, axis, first_radial, second_radial, radius = frame
    areas = []
    for contour in (source, result.contour):
        relative = contour.v - center
        angle = np.unwrap(np.arctan2(relative @ second_radial, relative @ first_radial))
        xy = np.column_stack((radius * angle, relative @ axis))
        areas.append(np.sum(xy[:-1, 0] * xy[1:, 1] - xy[1:, 0] * xy[:-1, 1]))
    assert areas[0] * areas[1] > 0.0


def test_disabled_smoothing_is_an_exact_identity() -> None:
    part, source = _seam_crossing_loop()
    result = MeshContourBSplineSmoother(part.mesh).smooth(source, smoothing_factor=0.0)
    assert not result.applied
    assert result.contour is source
    assert result.reason == "smoothing_disabled"


@pytest.mark.parametrize("damage", ["negative_face", "outside_face", "nan_barycentric", "wrong_geometry", "nan_point"])
def test_invalid_surface_input_returns_original_without_aborting_the_design(damage: str) -> None:
    part, source = _seam_crossing_loop()
    source = replace(
        source,
        v=source.v.copy(),
        segment_faces=source.segment_faces.copy(),
        segment_barycentric=source.segment_barycentric.copy(),
    )
    if damage == "negative_face":
        source.segment_faces[0] = -1
    elif damage == "outside_face":
        source.segment_faces[0] = len(part.mesh.faces)
    elif damage == "nan_barycentric":
        source.segment_barycentric[0, 0, 0] = np.nan
    elif damage == "wrong_geometry":
        source.segment_barycentric[0, 0] = [1.0, 0.0, 0.0]
    else:
        source.v[1, 0] = np.nan
    result = MeshContourBSplineSmoother(part.mesh).smooth(source, smoothing_factor=0.25)
    assert not result.applied
    assert result.contour is source
    assert result.reason in {"invalid_surface_provenance", "nonfinite_contour_points"}


def test_tight_displacement_cap_explains_fallback_without_relocating_original() -> None:
    part, source = _seam_crossing_loop()
    before = source.v.copy()
    result = MeshContourBSplineSmoother(part.mesh).smooth(
        source,
        smoothing_factor=0.0,
        scale_mode="mesh_coupled",
        mesh_gain=0.1,
        max_displacement_m=1.0e-9,
        target_segment_length_m=0.001,
    )
    assert not result.applied
    assert result.contour is source
    assert result.reason == "displacement_limit_exceeded"
    np.testing.assert_array_equal(source.v, before)


@pytest.mark.parametrize("limit", [float("nan"), float("inf"), -float("inf"), -0.001])
def test_invalid_displacement_limit_cannot_silently_disable_the_cap(limit: float) -> None:
    part, source = _seam_crossing_loop()
    result = MeshContourBSplineSmoother(part.mesh).smooth(
        source,
        smoothing_factor=0.25,
        max_displacement_m=limit,
    )
    assert not result.applied
    assert result.contour is source
    assert result.reason == "invalid_displacement_limit"
    report = validate_config(CoilGenConfig(contour_smoothing_max_displacement=limit))
    assert any(message.field == "contour_smoothing_max_displacement" for message in report.errors)


def test_topology_work_limit_keeps_original_with_explicit_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    part, source = _seam_crossing_loop()

    def limited_topology(*args: object, **kwargs: object) -> None:
        raise _TopologyWorkLimit("cell workspace exceeded")

    monkeypatch.setattr("coilgen.contour_smoothing._polyline_self_intersects", limited_topology)
    result = MeshContourBSplineSmoother(part.mesh).smooth(source, smoothing_factor=0.25)
    assert not result.applied
    assert result.contour is source
    assert result.reason == "topology_check_work_limit_exceeded"


def test_pathological_sampling_has_a_bounded_topology_pair_workspace() -> None:
    # This open curve is simple, but almost all vertices cluster in one cell.
    # Testing every pair would require tens of millions of comparisons.
    x = np.r_[np.linspace(0.0, 1.0e-8, 8192), 1.0]
    points = np.column_stack((x, np.zeros_like(x)))
    with pytest.raises(_TopologyWorkLimit, match="pair workspace"):
        _polyline_self_intersects(points)


def test_long_diagonal_has_a_bounded_topology_cell_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    # A small explicit budget proves that rectangular cell expansion is
    # checked before allocating the long diagonal's entire key list.
    monkeypatch.setattr("coilgen.contour_smoothing._MAX_TOPOLOGY_CELL_REFERENCES", 32)
    points = np.array([[0, 0], [1, 1], [1.1, 1], [1.1, 1.1]], dtype=float)
    with pytest.raises(_TopologyWorkLimit, match="cell workspace"):
        _polyline_self_intersects(points)


def test_fitting_failure_keeps_original_contour(monkeypatch: pytest.MonkeyPatch) -> None:
    part, source = _seam_crossing_loop()

    def failed_fit(*args: object, **kwargs: object) -> None:
        raise ValueError("FITPACK could not fit this contour")

    monkeypatch.setattr("coilgen.contour_smoothing._periodic_bspline", failed_fit)
    result = MeshContourBSplineSmoother(part.mesh).smooth(source, smoothing_factor=0.25)
    assert not result.applied
    assert result.contour is source
    assert result.reason == "spline_fit_failed"


def test_user_cancellation_inside_numerical_fit_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    part, source = _seam_crossing_loop()

    def cancelled_fit(*args: object, **kwargs: object) -> None:
        raise DesignCancelled("cancelled inside fitting")

    monkeypatch.setattr("coilgen.contour_smoothing._periodic_bspline", cancelled_fit)
    with pytest.raises(DesignCancelled, match="inside fitting"):
        MeshContourBSplineSmoother(part.mesh).smooth(source, smoothing_factor=0.25)


def test_dense_mapping_cooperatively_observes_worker_cancellation() -> None:
    part, source = _seam_crossing_loop()
    before = source.v.copy()
    messages: list[str] = []
    with progress_context(
        lambda event: messages.append(event["message"]),
        cancel_check=lambda: bool(messages and messages[-1] == "mapping fitted contour to surface"),
    ):
        with pytest.raises(DesignCancelled):
            MeshContourBSplineSmoother(part.mesh).smooth(
                source,
                smoothing_factor=0.0,
                scale_mode="mesh_coupled",
                mesh_gain=0.1,
                target_segment_length_m=0.001,
            )
    assert "mapping fitted contour to surface" in messages
    np.testing.assert_array_equal(source.v, before)


def test_full_resolution_topology_guard_finds_crossing_hidden_between_decimated_samples() -> None:
    # A local bow tie occupies four of 2049 samples. Uniform decimation to
    # 513 points skips it entirely and would accept this invalid contour.
    points = np.column_stack((np.arange(2049, dtype=float), np.zeros(2049)))
    points[1001:1005] = [[1001.0, 1.0], [1003.0, -1.0], [1001.0, -1.0], [1003.0, 1.0]]
    assert _polyline_self_intersects(points)
    assert not _polyline_self_intersects(points[np.linspace(0, 2048, 513, dtype=int)])


def test_topology_guard_accepts_dense_simple_loop_and_rejects_nonadjacent_touch() -> None:
    angle = np.linspace(0, 2.0 * np.pi, 8193)
    circle = np.column_stack((np.cos(angle), np.sin(angle)))
    circle[-1] = circle[0]
    assert not _polyline_self_intersects(circle)
    # Two nonadjacent portions sharing one point are not a simple contour.
    touching = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [1, 0], [2, 0]], dtype=float)
    assert _polyline_self_intersects(touching)
