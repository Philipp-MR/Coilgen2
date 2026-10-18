"""Physical plotting coordinates, exact provenance and run-bound scope."""

import numpy as np
import pytest
from dataclasses import replace

from coilgen.application.diagnostics import ProblemIssue
from coilgen.application.routing_analysis import (
    SupportView,
    analyze_routes,
    draw_routing_analysis,
    draw_routing_closeup,
    routing_analysis_available,
)
from coilgen.application.service import RunRecord
from coilgen.models import CoilPart, Mesh, WirePath, WireSegmentOwner, FabricationPath
from coilgen.scientific_arrays import fabrication_frame_arrays, path_provenance_arrays, routing_evidence


def plane(offset=0.0):
    vertices = np.array([[offset, -0.1, -0.1], [offset, 0.1, -0.1], [offset, 0.1, 0.1], [offset, -0.1, 0.1]])
    return vertices, np.array([[0, 1, 2], [0, 2, 3]])


def shell(bottom_radius=0.08, top_radius=0.08, length=0.2, count=12):
    angles = np.arange(count) * 2 * np.pi / count
    bottom = np.column_stack((bottom_radius * np.cos(angles), bottom_radius * np.sin(angles), np.zeros(count)))
    top = np.column_stack((top_radius * np.cos(angles), top_radius * np.sin(angles), np.full(count, length)))
    faces = []
    for index in range(count):
        following = (index + 1) % count
        faces.extend(((index, following, count + index), (following, count + following, count + index)))
    return np.vstack((bottom, top)), np.asarray(faces)


def record(tmp_path, *, two_planes=False, frames=False):
    vertices, faces = plane(0.05)
    points = np.array(
        [[0.05, -0.02, -0.02], [0.05, 0.02, 0.02], [0.05, -0.02, 0.02], [0.05, 0.02, -0.02], [0.05, -0.02, -0.02]]
    )
    arrays = {
        "vertices_m": vertices,
        "faces": faces,
        "support_0_vertices_m": vertices,
        "support_0_faces": faces,
        "route_0_points_m": points,
        "route_0_owner_kind": np.array(
            ["contour_loop", "contour_group", "within_group_return", "inter_group_connector"]
        ),
        "route_0_owner_connector_role": np.array(["", "joint_spiral_join", "", "primary"]),
        "route_0_owner_group_index": np.array([0, 0, 0, -1]),
    }
    metadata = {
        "name": "Exact routing evidence",
        "summary": {"accepted": False},
        "fabrication": {"ready": False},
        "routing": {
            "parts": [{"part_index": 0, "current_A": -2.0, "joint_routing": {"selected_route_status": "invalid"}}]
        },
    }
    if two_planes:
        arrays.update(
            {
                "support_1_vertices_m": vertices - [0.1, 0, 0],
                "support_1_faces": faces,
                "route_1_points_m": points - [0.1, 0, 0],
            }
        )
    if frames:
        path = FabricationPath(
            points[:-1] + [0.001, 0, 0],
            np.tile([0, 0, 1.0], (4, 1)),
            np.tile([1.0, 0, 0], (4, 1)),
            np.tile([0, 1.0, 0], (4, 1)),
            np.array([[0.0, 0.0], [0.002, 0.0], [0.0, 0.001]]),
            points[:-1],
            closed=True,
        )
        prefix = "review_conductor_0"
        arrays[prefix + "_points_m"] = path.wire_points_m
        arrays.update(fabrication_frame_arrays(prefix, path))
        metadata["fabrication"]["paths"] = [
            {
                "body_id": "part001_body0001",
                "part_index": 1,
                "diagnostics": {
                    "manufacturing_ok": False,
                    "manufacturing_failure_reasons": "minimum_bend_radius_violation",
                    "sweep_ring_count": 4,
                    "sweep_final_bend_point_count": 4,
                    "sweep_final_min_bend_point_index": 1,
                    "sweep_final_min_bend_radius_m": 0.001,
                    "sweep_required_bend_radius_m": 0.006,
                },
            }
        ]
    return RunRecord(tmp_path, metadata, arrays)


@pytest.mark.parametrize("radii", [(0.08, 0.08), (0.09, 0.05)])
def test_full_facet_development_preserves_every_panel_distance(radii):
    vertices, faces = shell(*radii)
    support = SupportView.from_mesh(vertices, faces)
    assert support.kind == "shell" and len(support.panels_3d) == 12
    for actual, developed in zip(support.panels_3d, support.panels_2d):
        np.testing.assert_allclose(
            np.linalg.norm(actual[:, None] - actual, axis=2),
            np.linalg.norm(developed[:, None] - developed, axis=2),
            rtol=1e-12,
            atol=1e-14,
        )
    np.testing.assert_allclose(support.panels_2d[:-1, 1:3], support.panels_2d[1:, [0, 3]], atol=1e-13)
    _, height, _ = support.coordinates(vertices)
    np.testing.assert_allclose(height, 0.0, atol=1e-14)
    assert "crossings here do not prove collisions" in support.scope


def test_cylinder_developed_extent_uses_polygon_perimeter_not_angle_or_ideal_radius():
    vertices, faces = shell(count=8)
    support = SupportView.from_mesh(vertices, faces)
    expected = np.linalg.norm(np.roll(vertices[:8], -1, axis=0) - vertices[:8], axis=1).sum()
    assert np.ptp(support.panels_2d[..., 0]) == pytest.approx(expected)
    assert expected != pytest.approx(2 * np.pi * 0.08)


def test_signed_height_and_planar_coordinates_follow_actual_transformed_faces():
    vertices, faces = plane()
    support = SupportView.from_mesh(vertices, faces)
    points = np.array([[0.003, 0.02, 0.04], [-0.002, -0.02, 0.01]])
    xy, height, _ = support.coordinates(points)
    np.testing.assert_allclose(xy, points[:, 1:])
    np.testing.assert_allclose(height, [0.003, -0.002])
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    translation = np.array([4.0, -3.0, 1.0])
    transformed = SupportView.from_mesh(vertices @ rotation.T + translation, faces)
    _, actual, _ = transformed.coordinates(points @ rotation.T + translation)
    np.testing.assert_allclose(actual, height, atol=1e-14)


def test_display_cut_is_omitted_without_changing_saved_coordinates():
    support = SupportView.from_mesh(*shell(count=8))
    assert support.visible_intervals(np.array([0, 7, 6, 5])).tolist() == [False, True, True]


def test_exact_sources_snapshot_provenance_and_separate_supports_are_immutable(tmp_path):
    saved = record(tmp_path, two_planes=True)
    before = saved.arrays
    analysis = analyze_routes(saved)
    assert analysis.fabrication_status == "Failed"
    assert len(analysis.sources) == 2
    first, second = analysis.sources
    assert first.roles == ("contour", "join", "return", "connector")
    assert first.current_A == -2 and first.status == "Retained baseline · invalid"
    assert first.support.origin_m[0] == pytest.approx(0.05)
    assert second.support.origin_m[0] == pytest.approx(-0.05)
    assert first.issues == second.issues == ()  # A crossing in projection cannot invent a collision.
    for value in (first.points_m, first.group_indices, first.support.vertices_m):
        assert not value.flags.writeable
    first.support.coordinates(first.points_m)[0][:] = 0.0
    for key, value in before.items():
        np.testing.assert_array_equal(saved.array(key), value)


def test_historical_merged_biplanar_support_is_split_by_connectivity_and_geometry(tmp_path):
    saved = record(tmp_path, two_planes=True)
    arrays = {key: value for key, value in saved.arrays.items() if not key.startswith("support_")}
    first, faces = plane(0.05)
    second, _ = plane(-0.05)
    arrays.update(vertices_m=np.vstack((second, first)), faces=np.vstack((faces, faces + 4)))
    analysis = analyze_routes(RunRecord(tmp_path, saved.metadata, arrays))
    assert [source.support.origin_m[0] for source in analysis.sources] == pytest.approx([0.05, -0.05])


def test_physical_profile_uses_area_centroid_and_saved_bend_triplet(tmp_path):
    analysis = analyze_routes(record(tmp_path, frames=True))
    source = analysis.sources[-1]
    assert source.rings_m is not None and source.rings_m.shape == (5, 3, 3)
    expected = source.points_m[0] - [0.001 / 3, 0.002 / 3, 0.0]
    np.testing.assert_allclose(source.rings_m[0, 0], expected)
    np.testing.assert_array_equal(source.rings_m[0], source.rings_m[-1])
    assert not source.rings_m.flags.writeable
    issue = source.issues[0]
    assert len(issue.points_m) == 3 and issue.segments == ((0, 1), (1, 2))
    np.testing.assert_array_equal(issue.points_m, source.points_m[:3])
    assert "Saved minimum-bend triplet" in issue.location_description


def test_mismatched_frame_coordinates_are_rejected_instead_of_transferring_labels(tmp_path):
    saved = record(tmp_path, frames=True)
    arrays = saved.arrays
    arrays["review_conductor_0_frame_points_m"][0, 0] += 1e-9
    with pytest.raises(ValueError, match="exactly correspond"):
        analyze_routes(RunRecord(tmp_path, saved.metadata, arrays))


def test_invalid_interval_provenance_is_rejected(tmp_path):
    saved = record(tmp_path)
    arrays = saved.arrays
    arrays["route_0_owner_kind"] = arrays["route_0_owner_kind"][:-1]
    with pytest.raises(ValueError, match="one typed string"):
        analyze_routes(RunRecord(tmp_path, saved.metadata, arrays))


def test_finite_work_budget_never_decimates_a_long_path(tmp_path):
    saved = record(tmp_path)
    arrays = saved.arrays
    arrays["route_0_points_m"] = np.tile([[0.05, 0.0, 0.0]], (100001, 1))
    with pytest.raises(ValueError, match="100,000-point.*no geometry is decimated"):
        analyze_routes(RunRecord(tmp_path, saved.metadata, arrays))


def test_full_figure_has_physical_distance_height_axes_and_no_smoothed_data(tmp_path):
    from matplotlib.figure import Figure

    source = analyze_routes(record(tmp_path, frames=True)).sources[-1]
    figure = Figure()
    axes = draw_routing_analysis(figure, source, issue=source.issues[0], envelopes=True)
    assert len(axes) == 3
    assert axes[0].get_xlabel() == "Developed distance [mm]"
    assert axes[2].get_ylabel() == "Signed support-face height [mm]"
    segments = axes[0].collections[0].get_segments()
    xy = source.support.coordinates(source.points_m)[0]
    np.testing.assert_array_equal(segments, np.stack((xy[:-1], xy[1:]), axis=1) * 1e3)


def test_serialization_retains_typed_ownership_and_original_geometry():
    vertices, faces = plane()
    owners = [
        WireSegmentOwner(kind="contour_loop", group_index=0, loop_index=1),
        WireSegmentOwner(kind="within_group_return", group_index=0),
    ]
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.01, 0.01]])
    path = WirePath(points, segment_owners=owners)
    arrays = path_provenance_arrays("route_0", path)
    assert arrays["route_0_owner_kind"].tolist() == ["contour_loop", "within_group_return"]
    assert arrays["route_0_owner_group_index"].tolist() == [0, 0]
    part = CoilPart(
        Mesh(vertices, faces),
        contour_step=3.0,
        wire_path=path,
        joint_route_diagnostics={"selected_route_status": "invalid", "selection_policy": "regular"},
    )
    evidence = routing_evidence([part])
    assert evidence["parts"][0]["current_A"] == 3.0
    assert evidence["parts"][0]["joint_routing"]["selected_route_status"] == "invalid"
    np.testing.assert_array_equal(path.v, points)


def test_capability_check_borrows_only_saved_names(tmp_path):
    class NamesOnly(RunRecord):
        @property
        def arrays(self):
            raise AssertionError("Availability must not copy geometry arrays.")

        def array(self, *args, **kwargs):
            raise AssertionError("Availability must not load geometry.")

    saved = record(tmp_path)
    assert routing_analysis_available(NamesOnly(tmp_path, saved.metadata, saved.arrays))
    assert not routing_analysis_available(RunRecord(tmp_path, {}, {"target_points_m": np.zeros((2, 3))}))


@pytest.mark.parametrize(
    "names,expected",
    [
        (("route_0_points_m",), True),
        (("fabrication_0_1_points_m",), True),
        (("review_conductor_2_points_m",), True),
        (("target_points_m",), False),
        (("route_0_owner_kind", 1), False),
    ],
)
def test_legacy_record_capability_uses_available_array_keys(names, expected):
    from types import SimpleNamespace

    record = SimpleNamespace(metadata={}, arrays=dict.fromkeys(names))
    assert bool(routing_analysis_available(record)) is expected
    assert not routing_analysis_available(None)


def test_unclassified_added_span_keeps_ownership_without_inventing_join_role(tmp_path):
    saved = record(tmp_path)
    arrays = saved.arrays
    arrays["route_0_owner_connector_role"][1] = ""
    source = analyze_routes(RunRecord(tmp_path, saved.metadata, arrays)).sources[0]
    assert source.roles[1] == "unclassified_added"
    assert arrays["route_0_owner_kind"][1] == "contour_group"
    assert arrays["route_0_owner_connector_role"][1] == ""


@pytest.mark.parametrize(
    "complete,drive,expected",
    [
        (True, False, "Closed mathematical winding"),
        (False, False, "not completely designed"),
        (True, True, "Explicit drive paths recorded"),
    ],
)
def test_mathematical_closure_is_distinct_from_feed_design(tmp_path, complete, drive, expected):
    saved = record(tmp_path, frames=True)
    metadata = saved.metadata
    metadata["fabrication"]["winding"] = [
        {"body_id": "part001_body0001", "circuit_complete": complete, "drive_feed_designed": drive}
    ]
    analysis = analyze_routes(RunRecord(tmp_path, metadata, saved.arrays))
    assert all(expected in source.circuit_status for source in analysis.sources)
    assert metadata["fabrication"]["winding"][0]["drive_feed_designed"] is drive


def test_closeup_retains_exact_long_intervals_crossing_local_window_and_reports_limits(tmp_path):
    from matplotlib.figure import Figure

    source = analyze_routes(record(tmp_path, frames=True)).sources[-1]
    issue = source.issues[0]
    snapshot = source.points_m.copy()
    figure = Figure(figsize=(13, 8.3))
    axes = draw_routing_closeup(figure, source, issue, radius_m=0.003)
    assert axes[0].get_title() == "Local tangent projection · exact samples"
    assert axes[0].get_xlim() == pytest.approx((-3.0, 3.0))
    assert len(axes[0].collections[0].get_segments()) >= 2
    # A camera crop retains entire saved intervals; it never resamples the winding.
    spatial_segments = axes[1].collections[0]._segments3d
    original = np.stack((source.points_m[:-1], source.points_m[1:]), axis=1) * 1e3
    assert all(any(np.array_equal(segment, candidate) for candidate in original) for segment in spatial_segments)
    text = axes[2].texts[0].get_text()
    assert "Measured 1 mm / limit 6 mm" in text
    assert "Saved body: Final centroid bend radius / required minimum: 1 / 6 mm" in text
    assert "conservative screen, not an exact solid intersection" in text
    np.testing.assert_array_equal(source.points_m, snapshot)


@pytest.mark.parametrize("radius", [0.0, -1.0, np.nan, np.inf, True])
def test_closeup_rejects_invalid_camera_scale(tmp_path, radius):
    from matplotlib.figure import Figure

    source = analyze_routes(record(tmp_path, frames=True)).sources[-1]
    with pytest.raises(ValueError, match="finite positive"):
        draw_routing_closeup(Figure(), source, source.issues[0], radius_m=radius)


def test_closeup_requires_a_located_issue_from_selected_geometry(tmp_path):
    from matplotlib.figure import Figure

    source = analyze_routes(record(tmp_path, frames=True)).sources[-1]
    with pytest.raises(ValueError, match="belong"):
        draw_routing_closeup(Figure(), source, replace(source.issues[0], body_id="unrelated"))
    missing = ProblemIssue("unlocated", "No saved location", "No point was recorded.")
    with pytest.raises(ValueError, match="located recorded"):
        draw_routing_closeup(Figure(), replace(source, issues=(missing,)), missing)


def test_overview_marks_every_located_recorded_issue_and_does_not_infer_missing_ones(tmp_path):
    from matplotlib.collections import PathCollection
    from matplotlib.figure import Figure

    source = analyze_routes(record(tmp_path, frames=True)).sources[-1]
    located = replace(source.issues[0], key="second", points_m=(tuple(source.points_m[2]),), segments=())
    missing = ProblemIssue("unlocated", "No saved location", "No point was recorded.")
    source = replace(source, issues=(*source.issues, located, missing))
    figure = Figure()
    axes = draw_routing_analysis(figure, source)
    assert len([item for item in axes[0].collections if isinstance(item, PathCollection)]) == 2
    assert any("All 2 located saved check witnesses" in text.get_text() for text in figure.texts)


def test_negative_current_3d_arrows_stay_on_the_saved_intervals(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D

    source = analyze_routes(record(tmp_path)).sources[0]
    calls = []
    original = Axes3D.quiver

    def capture(axis, *args, **kwargs):
        calls.append(np.asarray(args))
        return original(axis, *args, **kwargs)

    monkeypatch.setattr(Axes3D, "quiver", capture)
    draw_routing_analysis(Figure(), source, envelopes=False)
    assert source.current_A < 0 and len(calls) == 4
    for index, arrow in enumerate(calls):
        np.testing.assert_array_equal(arrow[:3], source.points_m[index + 1] * 1e3)
        np.testing.assert_array_equal(arrow[:3] + arrow[3:], source.points_m[index] * 1e3)
