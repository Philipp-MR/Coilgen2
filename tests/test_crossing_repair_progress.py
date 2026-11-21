"""Long repair work stays cancellable without claiming pipeline completion."""

import numpy as np
import pytest

from coilgen import crossing_repair
from coilgen.config import CoilGenConfig
from coilgen.crossing_relaxation import _CutMotion
from coilgen.progress import DesignCancelled, ProgressReporter, progress_context, progress_scope


def test_candidate_sweeps_do_not_complete_the_repair_allocation(monkeypatch):
    events = []
    parts = [object()]

    def search(received, config):
        assert received is parts
        reporter = ProgressReporter()
        reporter.report(100, "Candidate sweep complete")
        reporter.report(None, "Next repair candidate")
        return received

    monkeypatch.setattr(crossing_repair, "_repair_swept_crossings", search)
    config = CoilGenConfig(skip_sweep=False, normal_shift_length=0.002, progress_flag=False)
    with progress_context(events.append), progress_scope(10, 90, "Repair", overall=True):
        assert crossing_repair.repair_swept_crossings(parts, config) is parts
    candidate = next(event for event in events if event["message"] == "Candidate sweep complete")
    following = next(event for event in events if event["message"] == "Next repair candidate")
    assert candidate["overall_percent"] == following["overall_percent"] == 10
    assert following["progress_indeterminate"]
    assert events[-1]["overall_percent"] == 90


def test_cancel_during_sequential_separation_is_bounded_and_preserves_input(monkeypatch):
    motion = object.__new__(_CutMotion)
    motion.config = CoilGenConfig(progress_flag=False)
    motion.origin = np.array([[0.0, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [0.03, 0, 0]])
    motion.normals = np.tile([0.0, 0, 1.0], (4, 1))
    motion.lateral_limit = motion.height_limit = np.full(4, 0.002)
    motion.sign, motion.closed = 1, False
    wire = motion.origin.copy()
    before = wire.copy()
    first = (np.array([1]), np.array([1.0]))
    second = (np.array([2]), np.array([1.0]))
    constraint = (
        0.001,
        np.array([0.0, 1.0, 0]),
        0.002,
        0,
        0.5,
        np.array([0.0, 0.001, 0]),
        first,
        2,
        0.5,
        np.array([0.0, -0.001, 0]),
        second,
    )
    calls, events = [], []
    original = motion.radial_limit

    def counted(points, indices):
        calls.append(len(indices))
        return original(points, indices)

    monkeypatch.setattr(motion, "radial_limit", counted)
    with progress_context(events.append, cancel_check=lambda: bool(events)):
        with pytest.raises(DesignCancelled):
            motion.step(wire, [constraint] * 256, 1.0)
    # Two bounded point updates per constraint; cancellation is checked at 64.
    assert 0 < len(calls) <= 128
    np.testing.assert_array_equal(wire, before)
    assert not any(event.get("scope_completed") for event in events)
