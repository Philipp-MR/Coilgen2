"""Cut volumes preserve directed source contours and atomic edit contracts."""

from copy import deepcopy
import numpy as np
import pytest
from coilgen.models import ContourLine
from coilgen.routing_ports import ContourArc
from coilgen.winding_cut_shapes import CutShape
from coilgen.interconnection_editor import InterconnectionEditor
from _coilgen_test_support.interconnection_editor import routed as routed, inner_cut


@pytest.mark.parametrize("kind", ["plane", "sphere", "box"])
def test_cut_shape_exact_boundaries_and_circular_seam(kind):
    arc = ContourArc(ContourLine(np.array([[-1.0, -1.0, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [-1, -1, 0]]), 1.0))
    shape = CutShape.from_dict(dict(kind=kind, center=[-1, -1, 0], normal=[1, 1, 0], size=[0.2, 0.2, 0.2]))
    intervals = shape.intervals(arc)
    assert len(intervals) == 1
    start, end = intervals[0]
    assert start < arc.perimeter < end
    expected = 0.2 if kind == "sphere" else 0.2 * np.sqrt(2)
    assert end - start == pytest.approx(2 * expected, abs=1e-12)
    for s in [start, end]:
        p = arc.point(s) - shape.center
        if kind == "sphere":
            assert np.linalg.norm(p) == pytest.approx(0.2)
        else:
            assert max(abs(p @ shape.axes.T)[: 1 if kind == "plane" else 3]) == pytest.approx(0.2)


@pytest.mark.parametrize("kind", ["plane", "sphere", "box"])
def test_cut_shapes_apply_exact_undo_persist_and_lock(routed, kind):
    editor = InterconnectionEditor(routed.part)
    cut = inner_cut(editor)
    before = editor.to_plan()
    shape = editor.default_cut_shape(kind, cut.key)
    shape["size"][0] *= 0.9
    preview = editor.preview_cut_shape(shape, [cut.key])
    assert len(preview["cuts"]) == 1 and editor.to_plan() == before
    editor.apply_seam(preview)
    assert editor.undo_label == "Apply cut shape" and editor.connectivity["complete"]
    assert editor.section(cut.key).cut_shape["kind"] == kind
    saved = editor.to_plan()
    replay = InterconnectionEditor(routed.part)
    replay.load_plan(saved)
    np.testing.assert_array_equal(editor.wire_path().v, replay.wire_path().v)
    assert replay.section(cut.key).cut_shape == editor.section(cut.key).cut_shape
    for a, b in zip(editor.source.contour_lines, routed.part.contour_lines):
        np.testing.assert_array_equal(a.v, b.v)
    editor.undo()
    assert editor.to_plan() == before
    editor.redo()
    assert editor.to_plan() == saved
    with pytest.raises(ValueError, match="stale"):
        editor.apply_seam(preview)
    editor.set_locked(cut.key, True)
    blocked = editor.to_plan()
    preview = editor.preview_cut_shape(shape, [cut.key])
    with pytest.raises(ValueError, match="unlocked"):
        editor.apply_seam(preview)
    assert editor.to_plan() == blocked


def test_cut_shape_tamper_and_full_turn_rejected(routed):
    editor = InterconnectionEditor(routed.part)
    cut = inner_cut(editor)
    shape = editor.default_cut_shape("sphere", cut.key)
    editor.apply_seam(editor.preview_cut_shape(shape, [cut.key]))
    saved = editor.to_plan()
    bad = deepcopy(saved)
    row = next(r for r in bad["sections"] if r.get("cut_shape"))
    row["cut_shape"]["center"][0] += 1
    with pytest.raises(ValueError, match="does not match"):
        editor.load_plan(bad)
    assert editor.to_plan() == saved
    shape["size"] = [10, 10, 10]
    with pytest.raises(ValueError, match="entire turn"):
        editor.preview_cut_shape(shape)
    assert editor.to_plan() == saved


def test_shape_handle_motion_is_finite_and_preserves_other_parameters():
    shape = CutShape.from_dict(dict(kind="box", center=[0, 0, 0], normal=[1, 0, 0], size=[0.003, 0.02, 0.01]))
    moved = shape.moved("center", [0.01, 0.02, 0.03])
    np.testing.assert_array_equal(moved.size, shape.size)
    tilted = moved.moved("normal", moved.center + [0, 3, 0])
    np.testing.assert_array_equal(tilted.normal, [0, 1, 0])
    resized = tilted.moved("size1", tilted.center + tilted.axes[1] * 0.04)
    assert resized.size[1] == pytest.approx(0.04)
    with pytest.raises(ValueError):
        shape.moved("normal", shape.center)


def test_box_tilt_does_not_flip_axes_at_a_coordinate_tie():
    shape = CutShape.from_dict(dict(kind="box", center=[0, 0, 0], normal=[1, 0.0001, 0], size=[0.003, 0.02, 0.01]))
    tilted = shape.moved("normal", [1, 0, 0.0001])
    assert shape.axes[1] @ tilted.axes[1] > 0.999999
    np.testing.assert_allclose(tilted.axes @ tilted.axes.T, np.eye(3), atol=1e-14)
    assert CutShape.from_dict(tilted.to_dict()).to_dict() == tilted.to_dict()
