from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.geometry import normalize_mesh
from coilgen import layout
from coilgen.manufacturing import create_swept_coil_meshes
from coilgen.models import CoilPart, ContourLine, Mesh, WirePath
from coilgen.progress import DesignCancelled, ProgressReporter, progress_context, progress_scope


def _parts():
    mesh = normalize_mesh(
        Mesh(
            np.array([[-0.1, -0.1, 0], [0.1, -0.1, 0], [0.1, 0.1, 0], [-0.1, 0.1, 0]]),
            np.array([[0, 1, 2], [0, 2, 3]]),
        )
    )
    path = np.column_stack((np.linspace(-0.02, 0.02, 8), np.zeros((8, 2))))
    return [CoilPart(mesh=deepcopy(mesh), wire_path=WirePath(path + [0, y, 0])) for y in (-0.02, 0.02)]


def test_contour_distance_counts_pairs_across_all_supports(monkeypatch):
    parts = _parts()
    for part in parts:
        part.contour_lines = [
            ContourLine(part.wire_path.v, potential=0),
            ContourLine(part.wire_path.v + [0, 0.01, 0], potential=1),
        ]
    monkeypatch.setattr(layout, "SurfaceGeodesicSolver", lambda mesh: None)
    monkeypatch.setattr(layout, "contour_evaluation_points", lambda mesh, contour, config: contour.v)
    observed = []

    def distance(*args):
        observed.append(max([event["percent"] for event in events if event.get("percent") is not None], default=0))
        return SimpleNamespace(distance=0.01)

    monkeypatch.setattr(layout, "_find_min_geodesic_loop_distance", distance)
    events = []
    with progress_context(events.append):
        layout.find_minimal_contour_distance(
            parts, CoilGenConfig(skip_calculation_min_winding_distance=False), ProgressReporter()
        )
    assert observed == [0, 50]
    assert events[-1]["percent"] == 100
    for part in parts:
        np.testing.assert_array_equal(part.contour_distance_matrix, [[0, 0.01], [0.01, 0]])


def test_topology_nested_tasks_cannot_finish_before_remaining_supports(monkeypatch):
    parts = _parts()
    for part in parts:
        part.contour_lines = [
            ContourLine(part.wire_path.v, potential=0),
            ContourLine(part.wire_path.v + [0, 0.01, 0], potential=1),
        ]
    arrivals, events = [], []

    def validate(part, *, check_separating, progress):
        arrivals.append(max([item.get("overall_percent", 0) for item in events], default=0))
        progress.report(100, "Nested validation complete")
        return SimpleNamespace(as_dict=lambda: {}, raise_for_errors=lambda: None)

    def inclusion(part, *, validate, progress):
        progress.report(100, "Nested inclusion complete")
        return np.zeros((len(part.contour_lines), len(part.contour_lines)), bool)

    monkeypatch.setattr(layout, "validate_intrinsic_topology_inputs", validate)
    monkeypatch.setattr(layout, "build_intrinsic_loop_inclusion_matrix", inclusion)
    with progress_context(events.append), progress_scope(0, 100, overall=True):
        layout.topological_loop_grouping(parts, CoilGenConfig(topological_grouping_method="legacy"), ProgressReporter())
    assert arrivals == [0, 50]
    assert events[-1]["overall_percent"] == 100
    measured = [item["overall_percent"] for item in events]
    assert measured == sorted(measured)
    assert all(len(part.groups) > 0 for part in parts)


def test_manufacturing_scopes_preserve_meshes_and_publish_intermediate_work():
    config = CoilGenConfig(
        skip_sweep=False, progress_flag=False, sweep_path_simplification="off", layout_terminal_strategy="waived"
    )
    expected, actual = _parts(), _parts()
    create_swept_coil_meshes(expected, config)
    events = []

    def capture(event):
        events.append(event)
        if event.get("overall_percent", 0) >= 99:
            assert actual[-1].swept_body is not None

    with progress_context(capture), progress_scope(0, 100, overall=True):
        create_swept_coil_meshes(actual, config)
    measured = [event["overall_percent"] for event in events]
    assert measured == sorted(measured) and measured[-1] == 100
    assert len(set(measured)) >= 10
    assert any(event.get("progress_indeterminate") for event in events)
    for wanted, saved in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(saved.swept_body.vertices, wanted.swept_body.vertices)
        np.testing.assert_array_equal(saved.swept_body.faces, wanted.swept_body.faces)
        assert saved.manufacturing_diagnostics == wanted.manufacturing_diagnostics


def test_manufacturing_cancelled_phase_cannot_report_completion():
    parts, events, cancelled = _parts(), [], False

    def capture(event):
        nonlocal cancelled
        events.append(event)
        if event.get("overall_percent", 0) >= 10:
            cancelled = True

    with pytest.raises(DesignCancelled):
        with progress_context(capture, cancel_check=lambda: cancelled), progress_scope(0, 100, overall=True):
            create_swept_coil_meshes(parts, CoilGenConfig(skip_sweep=False, progress_flag=False))
    assert max(event.get("overall_percent", 0) for event in events) < 100
