"""Preserve directed source sampling of explicitly enabled mutable cylinder joins."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.geometry import normalize_mesh
from coilgen.manufacturing import (
    _AnalyticCylinderSurface,
    _prepare_sweep_centerline,
    _path_tangents,
    _path_frames,
    _frame_rotation_angles_deg,
    _final_path_bend_diagnostics,
)
from coilgen.models import CoilPart, Mesh, WirePath, WirePathGeometry, WireSegmentOwner

PROFILE = np.array([[-0.0001, -0.0001], [0.0001, -0.0001], [0.0001, 0.0001], [-0.0001, 0.0001]])


def fixture(*, include_sharp=True, detection_angle=15.0):
    radius = 0.066
    theta = np.arange(64) * 2 * np.pi / 64
    ring = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), np.zeros(64)))
    vertices = np.vstack((ring + [0, 0, -0.1], ring + [0, 0, 0.1]))
    faces = np.asarray(
        [[a, (a + 1) % 64, (a + 1) % 64 + 64] for a in range(64)] + [[a, (a + 1) % 64 + 64, a + 64] for a in range(64)]
    )
    mesh = normalize_mesh(Mesh(vertices, faces))
    surface = _AnalyticCylinderSurface(np.zeros(3), np.array([0.0, 0.0, 1.0]), radius, -0.1, 0.1)
    angle = np.linspace(np.pi, np.pi / 2, 91)
    arc = np.array([0.002, 0.005]) + 0.002 * np.c_[np.cos(angle), np.sin(angle)]
    chart = (
        np.vstack(([-0.01, 0], [0, 0], arc, [0.012, 0.007]))
        if include_sharp
        else np.vstack(([0, 0], arc, [0.012, 0.007]))
    )
    points = surface.exp_map(np.array([radius, 0, 0]), chart)
    labels = [WirePathGeometry.SURFACE] * (len(points) - 1)
    arc_start = 2 if include_sharp else 1
    owners = [
        WireSegmentOwner("contour_group", group_index=0, connector_role="joint_spiral_join")
        if arc_start <= a < arc_start + 90
        else WireSegmentOwner("contour_loop", group_index=0, loop_index=1)
        for a in range(len(labels))
    ]
    part = CoilPart(mesh, wire_path=WirePath(points.copy(), segment_owners=owners, segment_geometry=labels))
    config = CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        surface_is_cylinder_flag=True,
        cylinder_mesh_parameter_list=(0.2, radius, 64, 10, 0, 0, 1, 0),
        sweep_bend_mode="fillet",
        sweep_min_bend_radius=0.0014,
        sweep_max_frame_rotation_deg=10,
        sweep_corner_detection_angle_deg=detection_angle,
        sweep_max_chord_error=5e-5,
        conductor_cross_section_width=0.0002,
        conductor_cross_section_height=0.0002,
    )
    return part, points, labels, config


def prepare(part, points, labels, config):
    before = deepcopy(part.wire_path)
    prepared, geometry, diagnostics = _prepare_sweep_centerline(part, points, labels, PROFILE, config, closed=False)
    tangents = _path_tangents(prepared, closed=False)
    normal, binormal = _path_frames(part, prepared, tangents, geometry, config, closed=False)
    angles = _frame_rotation_angles_deg(tangents, normal, binormal, closed=False)
    bends = _final_path_bend_diagnostics(prepared, PROFILE, config, closed=False)
    np.testing.assert_array_equal(part.wire_path.v, before.v)
    assert part.wire_path.segment_owners == before.segment_owners
    assert part.wire_path.segment_geometry == before.segment_geometry
    return prepared, geometry, diagnostics, angles, bends


@pytest.mark.parametrize("include_sharp", [False, True])
@pytest.mark.parametrize("role", ["joint_spiral_join", "joint_return"])
@pytest.mark.parametrize("reflection", [-1, 1])
def test_certified_dense_c1_mutable_join_keeps_frame_and_radius_bounds(include_sharp, role, reflection):
    part, points, labels, config = fixture(include_sharp=include_sharp)
    points = points.copy()
    points[:, 1] *= reflection
    part.wire_path.v = points.copy()
    for owner in part.wire_path.segment_owners:
        if owner.connector_role:
            owner.connector_role = role
    config.sweep_corner_cluster_max_source_deviation_m = 0.001
    if not include_sharp:
        tangent = _path_tangents(points, closed=False)
        normal, binormal = _path_frames(part, points, tangent, labels, config, closed=False)
        assert np.max(_frame_rotation_angles_deg(tangent, normal, binormal, closed=False)) <= 10 + 1e-6
    _prepared, _geometry, diagnostics, angles, bends = prepare(part, points, labels, config)
    assert np.max(angles) <= config.sweep_max_frame_rotation_deg + 1e-6
    assert not bends["sweep_final_min_bend_radius_violation"]
    assert not bends["sweep_final_curvature_envelope_violation"]
    assert diagnostics["sweep_bend_infeasible_count"] == 0
    if include_sharp:
        assert diagnostics["sweep_corner_resolved_count"] >= 1


@pytest.mark.parametrize(
    "defect",
    [
        "missing_owners",
        "mismatched_source_sequence",
        "mismatched_geometry",
        "unknown_role",
        "protected_contour",
        "non_surface_curve",
    ],
)
def test_curve_retention_requires_exact_source_provenance_and_mutable_surface_role(defect):
    part, points, labels, config = fixture(include_sharp=False)
    if defect == "missing_owners":
        part.wire_path.segment_owners = []
    elif defect == "mismatched_source_sequence":
        part.wire_path.v = part.wire_path.v.copy()
        part.wire_path.v[30, 2] += 0.0003
    elif defect == "mismatched_geometry":
        part.wire_path.segment_geometry[30] = WirePathGeometry.FREE_3D
    elif defect in {"unknown_role", "protected_contour"}:
        for owner in part.wire_path.segment_owners:
            if owner.connector_role:
                if defect == "unknown_role":
                    owner.connector_role = "legacy_unverified"
                else:
                    owner.kind = "contour_loop"
                    owner.loop_index = 2
    else:
        labels = [
            WirePathGeometry.FREE_3D if owner.connector_role else label
            for owner, label in zip(part.wire_path.segment_owners, labels)
        ]
        part.wire_path.segment_geometry = labels.copy()
    baseline = prepare(part, points, labels, config)
    config.sweep_corner_cluster_max_source_deviation_m = 0.001
    enabled = prepare(part, points, labels, config)
    np.testing.assert_array_equal(enabled[0], baseline[0])
    assert enabled[1] == baseline[1]
    np.testing.assert_array_equal(enabled[3], baseline[3])
    assert enabled[4] == baseline[4]


def test_zero_budget_keeps_existing_fillet_output_and_miter_bypasses_the_new_source_sampling_path():
    part, points, labels, config = fixture(include_sharp=False)
    baseline = prepare(part, points, labels, config)
    assert len(baseline[0]) == 16
    assert baseline[2]["sweep_bend_analysis_points_removed"] == 84
    assert float(np.max(baseline[3])) == pytest.approx(14.526477228087563, abs=1e-7)
    config.sweep_bend_mode = "miter"
    config.sweep_min_bend_radius = 0
    miter_zero = prepare(part, points, labels, config)
    config.sweep_corner_cluster_max_source_deviation_m = 0.001
    miter_enabled = prepare(part, points, labels, config)
    np.testing.assert_array_equal(miter_enabled[0], miter_zero[0])
    assert miter_enabled[1:3] == miter_zero[1:3]
    np.testing.assert_array_equal(miter_enabled[3], miter_zero[3])
    assert miter_enabled[4] == miter_zero[4]


@pytest.mark.parametrize("radius", [0.0014, 0.01])
def test_preserving_a_mutable_source_curve_does_not_skip_real_corner_feasibility(radius):
    part, _points, _labels, config = fixture()
    surface = _AnalyticCylinderSurface(np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.066, -0.1, 0.1)
    chart = 0.001 * np.array([[-5, 0], [0, 0], [5, 0], [5, 5], [10, 5], [15, 5]])
    points = surface.exp_map(np.array([0.066, 0, 0]), chart)
    labels = [WirePathGeometry.SURFACE] * (len(points) - 1)
    part.wire_path = WirePath(
        points.copy(),
        segment_geometry=labels,
        segment_owners=[
            WireSegmentOwner("contour_group", group_index=0, connector_role="joint_return") for _ in labels
        ],
    )
    config = replace(config, sweep_min_bend_radius=radius, sweep_corner_cluster_max_source_deviation_m=0.001)
    _points, _geometry, diagnostics, angles, bends = prepare(part, points, labels, config)
    if radius == 0.01:
        assert diagnostics["sweep_min_bend_radius_violation"]
        assert bends["sweep_final_min_bend_radius_violation"]
        assert diagnostics["sweep_corner_resolved_count"] == 0
    else:
        assert diagnostics["sweep_corner_resolved_count"] >= 2
        assert not diagnostics["sweep_min_bend_radius_violation"]
        assert not bends["sweep_final_min_bend_radius_violation"]
        assert np.max(angles) <= config.sweep_max_frame_rotation_deg + 1e-6


def test_eligible_source_with_no_detected_corners_retains_initialized_repair_state_and_hard_frame_gate(monkeypatch):
    import coilgen.manufacturing as manufacturing

    part, points, labels, config = fixture(include_sharp=True)
    config.sweep_corner_cluster_max_source_deviation_m = 0.001
    # Model a detector with no initial corner proposals, while independent
    # tangent/frame sampling still sees the actual sharp source geometry.
    monkeypatch.setattr(manufacturing, "_sweep_corner_candidates", lambda *_args, **_kwargs: [])
    _prepared, _geometry, diagnostics, angles, _bends = prepare(part, points, labels, config)
    assert diagnostics["sweep_corner_candidate_count"] == 0
    assert diagnostics["sweep_corner_resolved_count"] == 0
    assert np.max(angles) > config.sweep_max_frame_rotation_deg
    # An empty detector is never a physical sampling certificate.
    assert diagnostics.get("sweep_source_frame_repair_pass_count", 0) <= 2
