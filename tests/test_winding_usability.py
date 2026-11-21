"""Persistent cutters, honest previews, local witnesses and winding traversal."""

from copy import deepcopy
from types import SimpleNamespace
import asyncio

import numpy as np
import pytest

from coilgen.interconnection_editor import InterconnectionEditor
from coilgen.interconnection_validation import EditorValidation
from coilgen.winding_review import quick_review, review_issues, winding_sequence, local_frame
from coilgen.winding_bridges import surface_crossing_stations
from test_interconnection_editor import routed as routed, inner_cut
from test_winding_sketch import crossing_editor
from test_browser_interconnection_editor import browser_editor as browser_editor, command
from test_gui_interconnection_editor import Viewport, session_from_case, finish, close
from test_workbench_usability import application as application


def test_persistent_object_exact_preview_replay_and_atomic_undo(routed):
    editor = InterconnectionEditor(routed.part)
    cut = inner_cut(editor)
    initial, revision = deepcopy(editor.to_plan()), editor.revision
    shape = editor.default_cut_shape("plane", cut.key)
    preview = editor.complete_preview(editor.preview_cut_shape(shape, [cut.key]), 0.003)
    assert not preview["error"] and preview["routes"]
    assert editor.to_plan() == initial and editor.revision == revision
    editor.apply_preview(preview)
    for proposed in preview["routes"]:
        np.testing.assert_array_equal(editor.section(proposed["key"]).points, proposed["points"])
    (obj,) = editor.cut_objects
    assert obj["keys"] == [cut.key]
    saved = editor.to_plan()
    replay = InterconnectionEditor(routed.part)
    replay.load_plan(saved)
    assert replay.cut_objects == editor.cut_objects
    shape["size"][0] *= 0.9
    preview = editor.complete_preview(editor.preview_cut_shape(shape, obj["keys"], object_id=obj["id"]))
    editor.apply_preview(preview)
    assert editor.cut_objects[0]["id"] == obj["id"]
    editor.undo()
    assert editor.to_plan() == saved
    editor.set_locked(cut.key, True)
    locked = deepcopy(editor.to_plan())
    blocked = editor.complete_preview(editor.preview_cut_shape(shape, obj["keys"], object_id=obj["id"]))
    assert blocked["error"] and not blocked["routes"]
    with pytest.raises(ValueError):
        editor.apply_preview(blocked)
    assert editor.to_plan() == locked


def test_choose_alternate_opening_then_reopen(routed):
    editor = InterconnectionEditor(routed.part)
    cut = inner_cut(editor)
    shape = editor.default_cut_shape("plane", cut.key)
    preview = editor.preview_cut_shape(shape, [cut.key])
    alternatives = [c for c in preview["alternatives"] if not c["selected"]]
    assert alternatives
    alternative = alternatives[0]
    preview = editor.complete_preview(
        editor.preview_cut_shape(shape, [cut.key], choices={cut.key: alternative["index"]})
    )
    assert not preview["error"]
    np.testing.assert_allclose(preview["cuts"][0]["point"], alternative["point"])
    editor.apply_preview(preview)
    replay = InterconnectionEditor(routed.part)
    replay.load_plan(editor.to_plan())
    np.testing.assert_array_equal(editor.wire_path().v, replay.wire_path().v)


def test_drawing_preview_cancel_and_commit_match():
    editor = crossing_editor()
    row = editor.section("s0")
    original = editor.to_plan()
    preview = editor.preview_drawing(
        row.key, row.target, [row.points[0], [0, 0.01, 0], row.points[-1]], minimum_radius=0.003
    )
    assert preview["routes"] and not preview["error"]
    assert editor.to_plan() == original
    editor.apply_preview(preview)
    np.testing.assert_array_equal(editor.section(row.key).points, preview["routes"][0]["points"])
    editor.undo()
    assert editor.to_plan() == original


def test_bridge_height_preserves_endpoints_and_exact_replay():
    editor = crossing_editor()
    editor.bridge_crossing("s0", "s2", "over", 0.003, 0.02)
    before = editor.section("s0")
    h = editor.bridge_handle("s0")
    editor.set_bridge_height("s0", np.array(h["point"]) + np.array(h["normal"]) * 0.001)
    after = editor.section("s0")
    assert after.bridge["height_m"] == pytest.approx(0.004)
    np.testing.assert_array_equal(after.points[[0, -1]], before.points[[0, -1]])
    replay = crossing_editor()
    replay.load_plan(editor.to_plan())
    np.testing.assert_array_equal(replay.wire_path().v, editor.wire_path().v)
    editor.undo()
    np.testing.assert_array_equal(editor.section("s0").points, before.points)


def test_curved_crossing_uses_physical_facets_and_rejects_near_miss():
    from coilgen.geometry import build_spherical_cap_mesh
    from coilgen.manufacturing import _SurfaceProjector

    mesh = build_spherical_cap_mesh(0.08, np.pi / 2, 24, 8, 1, 0, 0, 0, 0, 0, 0)
    triangle = mesh.vertices[mesh.faces[60]]
    center = triangle.mean(axis=0)
    a, b = triangle[0] - center, triangle[1] - center
    first, second = np.array([center - a * 0.2, center + a * 0.2]), np.array([center - b * 0.2, center + b * 0.2])
    projector = _SurfaceProjector(mesh)
    hits = surface_crossing_stations(projector, first, second)
    assert len(hits) == 1
    np.testing.assert_allclose(hits[0][2], center, atol=1e-12)
    assert not surface_crossing_stations(projector, first, first + b * 0.15)
    from coilgen.models import CoilPart, WirePath, WireSegmentOwner

    points = np.vstack((first, second, first[0]))
    owners = [
        WireSegmentOwner("within_group_connector" if i % 2 == 0 else "contour_group", group_index=i) for i in range(4)
    ]
    part = CoilPart(mesh=mesh, wire_path=WirePath(points, owners), contour_step=1.0)
    editor = InterconnectionEditor(part)
    editor.bridge_crossing("s0", "s2", "over", 0.0003, np.linalg.norm(first[1] - first[0]) * 0.5)
    assert editor.section("s0").bridge and editor.connectivity["complete"]
    replay = InterconnectionEditor(part)
    replay.load_plan(editor.to_plan())
    np.testing.assert_array_equal(editor.wire_path().v, replay.wire_path().v)


def test_playback_follows_graph_and_open_circuit_is_not_playable():
    editor = crossing_editor()
    sequence = winding_sequence(editor)
    assert len(sequence) == len(editor.sections)
    for a, b in zip(sequence, sequence[1:]):
        assert editor.section(a["key"]).target == b["key"]
    frame = local_frame(editor, "s0")
    assert np.linalg.norm(frame["normal"]) == pytest.approx(1)
    editor.disconnect("s0")
    assert winding_sequence(editor) == []
    assert any(w["category"] == "continuity" for w in quick_review(editor, 0.003))


def test_unsupported_certificate_and_advisory_cannot_become_passes():
    editor = crossing_editor()
    report = EditorValidation(
        "x", False, ("Circuit 1: unsupported_nonplanar_support", "route: pointwise_field_delta_budget")
    )
    issues = review_issues(editor, report, 0, 0.003)
    assert any(i["category"] == "unverified" and i["point"] is None for i in issues)
    assert any(i["category"] == "failed" for i in issues)
    assert not report.accepted


def test_measured_bend_uses_exact_final_frame_witness():
    editor = crossing_editor()
    points = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.001, 0.001, 0.0]])
    frames = SimpleNamespace(
        points_m=points,
        normals=np.tile([0.0, 0.0, 1.0], (3, 1)),
        transverse=np.tile([1.0, 0.0, 0.0], (3, 1)),
        cross_section_m=np.array([[-0.001, -0.001], [0.001, -0.001], [0.001, 0.001], [-0.001, 0.001]]),
        closed=False,
    )
    metrics = dict(
        sweep_ring_count=3,
        sweep_final_min_bend_point_index=1,
        sweep_final_bend_point_count=3,
        sweep_final_min_bend_radius_m=0.001,
    )
    report = EditorValidation(
        "x",
        False,
        ("Circuit 1: minimum_bend_radius_violation",),
        dict(circuits=[dict(manufacturing=metrics)]),
        (SimpleNamespace(fabrication_path=frames),),
    )
    issue = next(i for i in review_issues(editor, report, 0, 0.003) if i["category"] == "failed")
    np.testing.assert_array_equal(issue["point"], points[1])
    assert issue["label"] == "Bend 1.00 / required 3 mm"
    assert len(issue["segments"]) == 2
    metrics["sweep_final_bend_point_count"] = 4
    issue = next(i for i in review_issues(editor, report, 0, 0.003) if i["category"] == "failed")
    assert issue["point"] is None


def test_browser_object_and_draw_previews_are_nonmutating(browser_editor):
    adapter, _ = browser_editor

    async def exercise():
        cut = inner_cut(adapter.editor)
        await command(adapter, "select", key=cut.key)
        before = adapter.session.identity
        await command(adapter, "preview_shape", shape_kind="box", group=cut.owner.group_index)
        assert not adapter.state.winding_error
        assert adapter.session.identity == before
        preview = adapter.state.winding_scene["seam_preview"]
        assert not preview["error"]
        assert not adapter.state.winding_scene["can_validate"]
        await command(adapter, "validate")
        assert "Generate or discard" in adapter.state.winding_error
        assert adapter.session.validation is None
        # A not-yet-committed object's proposed id must survive dimension/group edits.
        await command(
            adapter,
            "preview_shape",
            shape=preview["shape"],
            object_id=preview["object_id"],
            group=cut.owner.group_index,
        )
        assert not adapter.state.winding_error
        await command(adapter, "apply_seam")
        assert not adapter.state.winding_error
        obj = adapter.editor.cut_objects[0]
        await command(adapter, "select_cut_object", object_id=obj["id"])
        assert adapter.state.winding_scene["seam_preview"]["object_id"] == obj["id"]
        await command(adapter, "save")
        name = adapter.state.winding_drafts[-1]
        await command(adapter, "open", name=name)
        assert adapter.editor.cut_objects[0]["id"] == obj["id"]
        assert adapter.state.winding_scene["sequence"]
        assert not adapter.state.winding_scene["can_apply"]
        # A proposal retained from another project/circuit must not block checks
        # after its geometry has already disappeared from the canvas.
        adapter.seam_preview = dict(preview, identity="previous-project")
        adapter.refresh()
        assert adapter.seam_preview is None
        assert adapter.state.winding_scene["can_validate"]

    asyncio.run(exercise())


def test_native_object_selection_review_and_playback(application, routed, tmp_path):
    from coilgen.gui.interconnection_editor import InterconnectionEditorDialog

    dialog = InterconnectionEditorDialog(session_from_case(routed, tmp_path), viewport_factory=Viewport)
    cut = inner_cut(dialog.editor)
    dialog.select_section(cut.key)
    dialog.mode.setCurrentIndex(dialog.mode.findData("shape"))
    finish(application, dialog)
    assert dialog._seam_preview["routes"]
    dialog._generate()
    finish(application, dialog)
    obj = dialog.editor.cut_objects[0]
    dialog._preview_object(obj["id"])
    finish(application, dialog)
    assert dialog._seam_preview["object_id"] == obj["id"]
    dialog._isolate_group(str(cut.owner.group_index))
    assert dialog.viewport.isolate_group == cut.owner.group_index
    dialog.play_toggle.setChecked(True)
    dialog._play_pause()
    dialog._play_tick()
    assert dialog._play_fraction > 0 and dialog.play_label.text()
    assert not dialog.apply_button.isEnabled()
    close(dialog)
