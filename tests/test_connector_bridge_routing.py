"""Ownership, provenance and stored-path regressions for inserted connector bridges."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.geometry import normalize_mesh
from coilgen.layout import (
    _bridge_spans_requiring_reconstruction,
    _build_owned_crossing_bridges,
    _map_reference_path_to_shifted_wire,
    _owned_crossing_bridge_spans,
    shift_return_paths,
)
from coilgen.models import CoilPart, InterGroupCut, Mesh, WirePath, WirePathGeometry, WireSegmentOwner
from coilgen.manufacturing import save_manufacturing_diagnostics


def _crossing_part(*, reverse_stored=False):
    # The closing connector has a 39.9 mm segment and 0.1 mm landing stub.
    # It crosses a remote primary connector; the outer contour closes far away.
    points = np.array(
        [
            [-0.04, 0.0, 0.0],
            [-0.02, 0.0, 0.0],
            [0.0199, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [0.04, 0.0, 0.0],
            [0.06, 0.04, 0.0],
            [0.0, 0.04, 0.0],
            [0.0, -0.04, 0.0],
            [-0.06, -0.04, 0.0],
            [-0.04, 0.0, 0.0],
        ]
    )
    contour = WireSegmentOwner("contour_loop", group_index=0, loop_index=0)
    closing = WireSegmentOwner("inter_group_connector", connector_index=0, connector_role="closing")
    primary = WireSegmentOwner("inter_group_connector", connector_index=0, connector_role="primary")
    owners = [contour] * (len(points) - 1)
    owners[1:3] = [closing] * 2
    owners[6] = primary
    geometry = [WirePathGeometry.SURFACE] * (len(points) - 1)
    geometry[1:3] = [WirePathGeometry.FREE_3D] * 2
    geometry[6] = WirePathGeometry.FREE_3D
    support = normalize_mesh(
        Mesh(
            np.array([[-0.2, -0.2, 0.0], [0.2, -0.2, 0.0], [0.2, 0.2, 0.0], [-0.2, 0.2, 0.0]]),
            np.array([[0, 1, 2], [0, 2, 3]]),
        )
    )
    stored = points[1:4][::-1] if reverse_stored else points[1:4]
    part = CoilPart(
        mesh=support,
        wire_path=WirePath(points, segment_owners=owners, segment_geometry=geometry),
        opening_cuts_among_groups=[
            InterGroupCut(
                np.zeros((0, 3)),
                np.zeros((0, 3)),
                connection_path_v=points[6:8].copy(),
                closing_connection_path_v=stored.copy(),
                normal_shift_paths_v=[stored.copy()],
            )
        ],
    )
    return part


def _config(**overrides):
    values = dict(
        normal_shift_length=0.003,
        normal_sweep_smooth_factor=5.0,
        min_wire_distance_m=0.002,
        conductor_cross_section_width=0.001,
        conductor_cross_section_height=0.001,
        sweep_min_bend_radius=0.0025,
        sweep_max_segment_length=0.001,
        sweep_max_chord_error=1e-5,
    )
    values.update(overrides)
    return CoilGenConfig(**values)


def _source_positions(part):
    provenance = part.normal_shift_provenance
    return provenance["output_source_segment_indices"] + provenance["output_source_segment_fractions"]


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_shift_inserts_owned_bridge_preserving_contours_and_provenance(sign):
    part = _crossing_part()
    source = part.wire_path.v.copy()
    original_owners = list(part.wire_path.segment_owners)
    shift_return_paths([part], _config(normal_shift_length=sign * 0.003))
    provenance = part.normal_shift_provenance
    assert part.normal_shift_diagnostics["normal_shift_bridge_count"] == 1
    assert provenance["schema_version"] == 2
    assert len(part.wire_path.v) > len(source)
    positions = _source_positions(part)
    assert np.all(np.diff(positions) > 0)
    outside = (positions <= 1) | (positions >= 3)
    np.testing.assert_array_equal(part.wire_path.v[outside], source[np.rint(positions[outside]).astype(int)])
    inserted = (positions > 1) & (positions < 3)
    assert np.max(sign * part.wire_path.v[inserted, 2]) <= 0.003 + 1e-12
    assert np.min(sign * part.wire_path.v[inserted, 2]) >= 0.0
    assert len(provenance["lift_factors"]) == len(part.wire_path.v)
    assert len(provenance["normals"]) == len(source)  # Explicit input-space field in schema 2.
    assert len(provenance["output_segment_owners"]) == len(part.wire_path.v) - 1
    assert len(provenance["output_segment_geometry"]) == len(part.wire_path.v) - 1
    for index, owner in enumerate(part.wire_path.segment_owners):
        source_index = int(np.floor(positions[index]))
        assert owner == original_owners[source_index]
        if 1 <= positions[index] < 3:
            assert part.wire_path.segment_geometry[index] == WirePathGeometry.FREE_3D
    assert part.normal_shift_diagnostics["normal_shift_introduced_backtrack_count"] == 0
    np.testing.assert_allclose(provenance["lift_distance_m"], part.wire_path.v[:, 2], atol=1e-12)


@pytest.mark.parametrize("reverse", [False, True])
def test_stored_connector_contains_every_inserted_sample_in_original_direction(reverse):
    part = _crossing_part(reverse_stored=reverse)
    shift_return_paths([part], _config())
    positions = _source_positions(part)
    expected = part.wire_path.v[(positions >= 1) & (positions <= 3)]
    if reverse:
        expected = expected[::-1]
    cut = part.opening_cuts_among_groups[0]
    np.testing.assert_array_equal(cut.closing_connection_path_v, expected)
    np.testing.assert_array_equal(cut.normal_shift_paths_v[0], expected)
    np.testing.assert_array_equal(cut.connection_path_v, part.normal_shift_provenance["points_before_m"][6:8])


def test_rejected_bridge_retains_sparse_source_and_reports_unresolved():
    part = _crossing_part()
    source = part.wire_path.v.copy()
    shift_return_paths([part], _config(sweep_min_bend_radius=0.5))
    np.testing.assert_array_equal(part.wire_path.v, source)
    assert part.normal_shift_diagnostics["normal_shift_bridge_count"] == 0
    assert part.normal_shift_diagnostics["normal_shift_rejected_bridge_count"] == 1
    assert "connector_bridge_unresolved" in part.normal_shift_diagnostics["normal_shift_failure_reasons"]
    event = part.normal_shift_provenance["bridge_events"][0]
    assert not event["accepted"]
    assert "insufficient_bend_radius" in event["generation_rejections"]
    np.testing.assert_array_equal(part.shift_array, np.zeros(len(source)))


def test_missing_support_rejects_bridge_without_crashing_or_moving_source():
    part = _crossing_part()
    source = part.wire_path.v.copy()
    part.mesh = Mesh(np.array([[0.0, 0.0, 0.0]]), np.zeros((0, 3), dtype=np.int64))
    shift_return_paths([part], _config())
    np.testing.assert_array_equal(part.wire_path.v, source)
    assert part.normal_shift_diagnostics["normal_shift_invalid_normal_count"] > 0
    assert part.normal_shift_diagnostics["normal_shift_quality_failure"]
    assert "undefined_lift_normal" in part.normal_shift_diagnostics["normal_shift_failure_reasons"]
    # The normal query already suppresses the invalid lift, so reconstruction
    # need not run. Its independent support guard must also remain safe.
    output, _lift, _positions, events = _build_owned_crossing_bridges(
        part, part.wire_path, [(1, 3, (6,))], np.zeros(len(source)), _config()
    )
    np.testing.assert_array_equal(output, source)
    assert not events[0]["accepted"]
    assert events[0]["reason"] == "undefined_support_normal"


@pytest.mark.parametrize("reverse", [False, True])
def test_connector_touching_closed_seam_uses_cyclic_tangents_and_preserves_contours(reverse):
    part = _crossing_part()
    part.wire_path.segment_owners[0] = part.wire_path.segment_owners[1]
    part.wire_path.segment_geometry[0] = WirePathGeometry.FREE_3D
    if reverse:
        part.wire_path = WirePath(
            part.wire_path.v[::-1].copy(),
            segment_owners=part.wire_path.segment_owners[::-1],
            segment_geometry=part.wire_path.segment_geometry[::-1],
        )
    source = part.wire_path.v.copy()
    original_owners = list(part.wire_path.segment_owners)
    shift_return_paths([part], _config())
    event = part.normal_shift_provenance["bridge_events"][0]
    assert event["accepted"], event
    assert len(part.wire_path.v) > len(source)
    np.testing.assert_array_equal(part.wire_path.v[[0, -1]], source[[0, -1]])
    positions = _source_positions(part)
    start, end = event["source_start_index"], event["source_end_index"]
    assert start == 0 if not reverse else end == len(source) - 1
    outside = (positions <= start) | (positions >= end)
    np.testing.assert_array_equal(part.wire_path.v[outside], source[np.rint(positions[outside]).astype(int)])
    assert np.all(np.diff(positions) > 0)
    for index, owner in enumerate(part.wire_path.segment_owners):
        assert owner == original_owners[int(np.floor(positions[index]))]
    assert np.max(part.wire_path.v[:, 2]) <= 0.003 + 1e-12
    assert event["local_clearance_m"] >= np.sqrt(2) * 0.001 - 1e-8
    assert part.normal_shift_diagnostics["normal_shift_introduced_backtrack_count"] == 0


def test_span_selection_requires_owner_geometry_and_direct_contact():
    part = _crossing_part()
    events = [((1, 2), (6,))]
    assert _owned_crossing_bridge_spans(part.wire_path, events, [(1, 6, 0.0)]) == [(1, 3, (6,))]
    assert _owned_crossing_bridge_spans(part.wire_path, events, []) == []
    part.wire_path.segment_geometry[1:3] = [WirePathGeometry.SURFACE] * 2
    assert _owned_crossing_bridge_spans(part.wire_path, events, [(1, 6, 0.0)]) == []


def test_span_with_overpass_and_underpass_roles_is_rejected_without_overriding_protection():
    part = _crossing_part()
    old = part.wire_path.v
    source = np.vstack((old[:8], [[0.008, -0.04, 0], [0.008, 0.04, 0], [-0.06, 0.04, 0], [-0.06, -0.04, 0]], old[-1:]))
    owners = [WireSegmentOwner("contour_loop", loop_index=0)] * (len(source) - 1)
    owners[1:3] = [WireSegmentOwner("inter_group_connector", connector_index=0, connector_role="closing")] * 2
    owners[6] = WireSegmentOwner("inter_group_connector", connector_index=0, connector_role="primary")
    owners[8] = WireSegmentOwner("inter_group_connector", connector_index=1, connector_role="closing")
    geometry = [WirePathGeometry.SURFACE] * (len(source) - 1)
    geometry[1:3] = [WirePathGeometry.FREE_3D] * 2
    geometry[6] = WirePathGeometry.FREE_3D
    part.wire_path = WirePath(source, segment_owners=owners, segment_geometry=geometry)
    shift_return_paths([part], _config())
    events = part.normal_shift_provenance["crossing_events"]
    assert any(1 in event["overpass_segments"] for event in events)
    assert any(1 in event["underpass_segments"] for event in events)
    np.testing.assert_array_equal(part.wire_path.v, source)
    assert part.normal_shift_diagnostics["normal_shift_introduced_backtrack_count"] == 0
    assert "connector_bridge_unresolved" in part.normal_shift_diagnostics["normal_shift_failure_reasons"]
    # A genuine interior underpass prevents this single-hump repair. The
    # crossing-coordinate guard and the legacy segment guard must both agree.
    output, _lift, _positions, bridge_events = _build_owned_crossing_bridges(
        part,
        part.wire_path,
        [(1, 3, (6,))],
        np.zeros(len(source)),
        _config(),
        protected_underpass_segments={1},
    )
    assert not bridge_events[0]["accepted"]
    assert bridge_events[0]["reason"] == "conflicting_crossing_roles"
    np.testing.assert_array_equal(output, source)


@pytest.mark.parametrize("positions, accepted", [((1.0, 3.0), True), ((2.0,), False)])
def test_bridge_protects_actual_interior_underpass_but_keeps_fixed_endpoint_roles(positions, accepted):
    part = _crossing_part()
    source = part.wire_path.v.copy()
    output, _lift, source_positions, events = _build_owned_crossing_bridges(
        part,
        part.wire_path,
        [(1, 3, (6,))],
        np.zeros(len(source)),
        _config(),
        protected_underpass_positions=positions,
    )
    assert events[0]["accepted"] is accepted, events[0]
    if accepted:
        assert len(output) > len(source)
        for position in positions:
            np.testing.assert_array_equal(output[source_positions == position], source[int(position)][None])
    else:
        assert events[0]["reason"] == "conflicting_crossing_roles"
        np.testing.assert_array_equal(output, source)


def _run_spacing_fallback_policy(monkeypatch, *, old_gap=0.0, physical_regressions=0, strict_alternative=False):
    from coilgen.connector_bridge_validation import BridgeValidation
    from coilgen.connector_bridges import BridgeGeneration

    part = _crossing_part()
    part.wire_path.v[6:8, 2] = old_gap
    source = part.wire_path.v.copy()
    proposals = []
    for index, height in enumerate([0.002, 0.0025] if strict_alternative else [0.002]):
        points = np.array([source[1], [0, 0, height], source[3]])
        proposals.append(
            SimpleNamespace(
                points=points,
                source_fractions=np.array([0.0, 0.5, 1.0]),
                lift_distances_m=np.array([0.0, height, 0.0]),
                local_clearance_m=0.002,
                maximum_deviation_m=height,
                minimum_radius_m=0.003,
                peak_parameter=0.5,
                height_m=height,
                profile_family="quintic",
            )
        )

    def generate(*_args, **kwargs):
        return (
            BridgeGeneration((), "no_spatial_candidate")
            if kwargs.get("spatial_mode")
            else BridgeGeneration(tuple(proposals), "candidates_available")
        )

    calls = []

    def validate(_original, _start, _end, points, **_kwargs):
        index = next(i for i, proposal in enumerate(proposals) if points is proposal.points)
        calls.append(index)
        accepted = bool(index == 1 and strict_alternative)
        return BridgeValidation(
            accepted,
            () if accepted else ("clearance_regression",),
            {
                "clearance_regression_count": 0 if accepted else 1,
                "physical_clearance_regression_count": 0 if accepted else physical_regressions,
                "maximum_gap_regression_m": 0.0 if accepted else 0.0004,
            },
        )

    monkeypatch.setattr("coilgen.connector_bridges.generate_connector_bridges", generate)
    monkeypatch.setattr("coilgen.connector_bridge_validation.validate_connector_bridge", validate)
    output, _lifts, _positions, events = _build_owned_crossing_bridges(
        part, part.wire_path, [(1, 3, (6,))], np.zeros(len(source)), _config(min_wire_distance_m=0.006)
    )
    return source, output, events[0], calls


def test_design_spacing_fallback_repairs_existing_intersection_with_explicit_diagnostics(monkeypatch):
    _source, output, event, calls = _run_spacing_fallback_policy(monkeypatch)
    assert event["accepted"]
    assert event["design_spacing_relaxed_for_intersection_repair"]
    assert event["design_spacing_regression_count"] == 1
    assert event["maximum_design_spacing_regression_m"] == pytest.approx(0.0004)
    assert output[2, 2] == pytest.approx(0.002)
    assert calls == [0]


@pytest.mark.parametrize("old_gap,physical_regressions", [(0.0002, 0), (0.0, 1)])
def test_spacing_fallback_rejects_positive_old_gap_or_new_physical_regression(
    monkeypatch, old_gap, physical_regressions
):
    def unnecessary_preparation(*args, **kwargs):
        raise AssertionError("A rejected spacing candidate needs no sweep preparation.")

    monkeypatch.setattr("coilgen.bridge_realization.bridge_realization_check", unnecessary_preparation)
    source, output, event, _calls = _run_spacing_fallback_policy(
        monkeypatch, old_gap=old_gap, physical_regressions=physical_regressions
    )
    assert not event["accepted"]
    assert "design_spacing_relaxed_for_intersection_repair" not in event
    np.testing.assert_array_equal(output, source)


def test_every_strict_candidate_is_preferred_over_spacing_fallback(monkeypatch):
    _source, output, event, calls = _run_spacing_fallback_policy(monkeypatch, strict_alternative=True)
    assert event["accepted"]
    assert "design_spacing_relaxed_for_intersection_repair" not in event
    assert output[2, 2] == pytest.approx(0.0025)
    assert calls == [0, 1]


@pytest.mark.parametrize("strict_alternative", [False, True])
def test_realization_failure_cannot_be_accepted_or_used_as_spacing_fallback(monkeypatch, strict_alternative):
    def reject_attachment(*args, **kwargs):
        return False, {
            "bridge_realization_checked": True,
            "bridge_realization_reason": "realized_attachment_curvature",
        }

    monkeypatch.setattr("coilgen.bridge_realization.bridge_realization_check", reject_attachment)
    source, output, event, calls = _run_spacing_fallback_policy(monkeypatch, strict_alternative=strict_alternative)
    assert not event["accepted"]
    assert "design_spacing_relaxed_for_intersection_repair" not in event
    assert calls == ([0, 1] if strict_alternative else [0])
    assert all("realized_attachment_curvature" in attempt["reasons"] for attempt in event["attempts"])
    np.testing.assert_array_equal(output, source)


def test_reconstruction_trigger_preserves_valid_curved_nonplanar_span():
    theta = np.linspace(0.0, np.pi / 3, 9)
    points = np.column_stack((0.05 * np.cos(theta), 0.05 * np.sin(theta), 0.01 * theta))
    trial = points.copy()
    trial[1:8, 2] += 0.001 * np.sin(np.linspace(0.0, np.pi, 7)) ** 2
    path = WirePath(points)
    assert _bridge_spans_requiring_reconstruction(path, [(1, 7, (0,))], trial, _config()) == []


def test_reconstruction_trigger_does_not_repair_unchanged_inherited_trim_conflict():
    source = np.array([[-0.02, 0.0, 0.0], [0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.001, 0.02, 0.0], [0.001, 0.04, 0.0]])
    # The 1 mm source interval cannot fit its existing 2.5 mm-radius corner;
    # normal shifting has introduced no change and must not claim this repair.
    path = WirePath(source)
    assert _bridge_spans_requiring_reconstruction(path, [(1, 3, (0,))], source.copy(), _config()) == []


def test_reconstruction_trigger_does_not_select_unchanged_long_single_interval():
    source = np.array([[-0.08, 0.0, 0.0], [-0.06, 0.0, 0.0], [0.06, 0.0, 0.0], [0.08, 0.0, 0.0]])
    path = WirePath(source)
    assert _bridge_spans_requiring_reconstruction(path, [(1, 2, (0,))], source.copy(), _config()) == []


def test_reconstruction_trigger_catches_new_trim_overlap_below_backtrack_threshold():
    source = np.array([[-0.01, 0.0, 0.0], [0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])
    trial = source.copy()
    trial[2, 2] = 0.0008
    directions = np.diff(trial, axis=0)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    angles = np.degrees(np.arccos(np.clip(np.sum(directions[:-1] * directions[1:], axis=1), -1, 1)))
    assert np.max(angles) < 50.0  # No >100-degree reversal; bend packing alone fails.
    path = WirePath(source)
    assert _bridge_spans_requiring_reconstruction(path, [(1, 3, (0,))], trial, _config()) == [(1, 3, (0,))]


@pytest.mark.parametrize("reverse", [False, True])
def test_reference_mapping_does_not_drop_samples_when_old_vertex_was_removed(reverse):
    before = np.array([[0.0, 0.0, 0.0], [0.999, 0.0, 0.0], [1.0, 0.0, 0.0]])
    positions = np.array([0.0, 0.2, 0.6, 0.9, 1.5, 2.0])
    after = np.column_stack((positions, np.sin(positions), np.zeros(len(positions))))
    reference = before[::-1] if reverse else before
    expected = after[::-1] if reverse else after
    actual = _map_reference_path_to_shifted_wire(reference, before, after, source_positions=positions)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("reverse", [False, True])
def test_reference_mapping_across_closed_seam_preserves_all_new_samples(reverse):
    before = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    positions = np.array([0.0, 0.4, 1.0, 2.0, 2.2, 2.8, 3.4, 4.0])
    after = np.column_stack((positions, positions**2, np.zeros(len(positions))))
    after[-1] = after[0]
    reference = before[[2, 3, 0, 1]]
    expected = after[[3, 4, 5, 6, 7, 1, 2]]
    if reverse:
        reference, expected = reference[::-1], expected[::-1]
    actual = _map_reference_path_to_shifted_wire(reference, before, after, source_positions=positions)
    np.testing.assert_array_equal(actual, expected)


def test_bridge_provenance_schema_two_roundtrips_numeric_input_and_output_arrays(tmp_path):
    part = _crossing_part()
    source = part.wire_path.v.copy()
    shift_return_paths([part], _config())
    files = save_manufacturing_diagnostics(SimpleNamespace(coil_parts=[part]), tmp_path, prefix="bridge")
    assert len(files) == 3
    with np.load(files[-1], allow_pickle=False) as saved:
        for key in saved.files:
            assert saved[key].dtype != object
        metadata = json.loads(str(saved["metadata_json"]))
        np.testing.assert_array_equal(saved["points_before_m"], source)
        np.testing.assert_array_equal(saved["points_after_m"], part.wire_path.v)
        assert len(saved["points_after_m"]) > len(saved["points_before_m"])
        assert saved["normals"].shape == source.shape
        assert saved["allowed_points"].shape == (len(source),)
        assert saved["points_after_lift_before_repair_m"].shape == source.shape
        assert saved["lift_factors"].shape == (len(part.wire_path.v),)
        positions = saved["output_source_segment_indices"] + saved["output_source_segment_fractions"]
        np.testing.assert_array_equal(positions, _source_positions(part))
        assert np.all(np.diff(positions) > 0)
    assert metadata["schema_version"] == 2
    assert metadata["length_unit"] == "m"
    assert len(metadata["segment_owners"]) == len(source) - 1
    assert len(metadata["output_segment_owners"]) == len(part.wire_path.v) - 1
    assert len(metadata["output_segment_geometry"]) == len(part.wire_path.v) - 1
    assert metadata["bridge_events"][0]["accepted"]
    assert "input:" in metadata["index_convention"] and "output:" in metadata["index_convention"]
