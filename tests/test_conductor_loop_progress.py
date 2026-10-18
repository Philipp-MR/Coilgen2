"""Long conductor loops expose real work and cooperate with cancellation."""

from copy import deepcopy

import numpy as np
import pytest

from coilgen import CoilGenConfig
from coilgen.geometry import normalize_mesh
from coilgen.manufacturing import create_swept_coil_meshes
from coilgen.models import CoilPart, Mesh, WirePath, WirePathGeometry
from coilgen.progress import DesignCancelled, progress_context, progress_scope
from coilgen.wire_provenance import nearest_source_segment_indices


def _corner_part():
    mesh = normalize_mesh(
        Mesh(
            np.array([[-0.1, -0.1, 0], [0.1, -0.1, 0], [0.1, 0.1, 0], [-0.1, 0.1, 0]]),
            np.array([[0, 1, 2], [0, 2, 3]]),
        )
    )
    points = np.array([[0, 0, 0.01], [0.05, 0, 0.01], [0.05, 0.05, 0.01]])
    part = CoilPart(mesh=mesh, wire_path=WirePath(points, segment_geometry=[WirePathGeometry.FREE_3D] * 2))
    config = CoilGenConfig(skip_sweep=False, sweep_bend_mode="fillet", sweep_min_bend_radius=0.005)
    return part, config


def test_counted_fillet_and_adaptive_work_preserves_exact_conductor_geometry():
    expected, config = _corner_part()
    actual = deepcopy(expected)
    create_swept_coil_meshes([expected], config)
    events = []
    with progress_context(events.append), progress_scope(0, 100, overall=True):
        create_swept_coil_meshes([actual], config)

    np.testing.assert_array_equal(actual.fabrication_path.points_m, expected.fabrication_path.points_m)
    np.testing.assert_array_equal(actual.swept_body.vertices, expected.swept_body.vertices)
    np.testing.assert_array_equal(actual.swept_body.faces, expected.swept_body.faces)
    assert actual.manufacturing_diagnostics == expected.manufacturing_diagnostics
    for label in ("Conductor bends", "Conductor sampling"):
        work = [event for event in events if event["label"] == label]
        assert work and work[0]["completed"] == 0
        assert work[-1]["completed"] == work[-1]["total"] > 0
    measured = [event["overall_percent"] for event in events]
    assert measured == sorted(measured) and measured[-1] == 100


@pytest.mark.parametrize("phase", ("Conductor bends", "Conductor sampling"))
def test_cancellation_inside_conductor_loops_retains_no_completed_body(phase):
    part, config = _corner_part()
    events, cancelled = [], False

    def capture(event):
        nonlocal cancelled
        events.append(event)
        if event["label"] == phase:
            cancelled = True

    with pytest.raises(DesignCancelled):
        with progress_context(capture, lambda: cancelled), progress_scope(0, 100, overall=True):
            create_swept_coil_meshes([part], config)
    assert part.swept_body is None
    assert max(event.get("overall_percent", 0) for event in events) < 100
    assert not any(event.get("completed_stage") for event in events)


@pytest.mark.parametrize("size", (129, 1025))
def test_provenance_counts_do_not_prematurely_complete_enclosing_operation(size):
    rng = np.random.default_rng(148)
    source = rng.normal(size=(size, 3)).cumsum(axis=0)
    target = source + [0, 0, 0.001]
    expected = nearest_source_segment_indices(source, target, chunk_size=8)
    events = []
    with progress_context(events.append), progress_scope(10, 90, overall=True):
        actual = nearest_source_segment_indices(source, target, chunk_size=8)
        assert max(event.get("overall_percent", 0) for event in events) == 10
    np.testing.assert_array_equal(actual, expected)
    work = [event for event in events if event["label"] == "Path provenance"]
    assert work[0]["completed"] == 0 and work[-1]["completed"] == size - 1
    assert all(event.get("phase_percent") is None for event in work)
    assert events[-1]["overall_percent"] == 90


@pytest.mark.parametrize("size", (129, 1025))
def test_provenance_loops_check_cancellation_before_returning_a_map(size):
    rng = np.random.default_rng(148)
    source = rng.normal(size=(size, 3)).cumsum(axis=0)
    target = source + [0, 0, 0.001]
    snapshots = source.copy(), target.copy()
    events, cancelled = [], False

    def capture(event):
        nonlocal cancelled
        events.append(event)
        if event["label"] == "Path provenance":
            cancelled = True

    with pytest.raises(DesignCancelled):
        with progress_context(capture, lambda: cancelled):
            nearest_source_segment_indices(source, target, chunk_size=8)
    assert not any(event.get("completed", 0) == size - 1 for event in events)
    np.testing.assert_array_equal(source, snapshots[0])
    np.testing.assert_array_equal(target, snapshots[1])
