"""Manual routing invariants against actual automatically routed contours."""

from copy import deepcopy
from dataclasses import replace
import json
import pickle

import numpy as np
import pytest

from coilgen.application.interconnection_editor import WindingEditorSession
from coilgen.interconnection_editor import InterconnectionEditor, PLAN_SCHEMA, apply_editor_plan
from coilgen.interconnection_validation import EditorValidation, validate_winding_revision
from coilgen.models import WirePath, WireSegmentOwner
from coilgen.physics import biot_savart_wire_path
from coilgen.pipeline_stages import PipelineCheckpoint, PipelineStage, invalidated_stage
from coilgen.routing_ports import ContourArc

from _coilgen_test_support.interconnection_editor import (
    inner_cut as inner_cut,
    routed as routed,
    simple_session as simple_session,
    surface_connection as surface_connection,
)


def test_automatic_route_is_imported_bit_exact_and_source_never_mutated(routed):
    source = pickle.dumps(routed.part)
    editor = InterconnectionEditor(routed.part)
    assert pickle.dumps(editor.wire_path()) == pickle.dumps(routed.part.wire_path)
    cut = inner_cut(editor)
    editor.move_cut(cut.key, cut.cut[0] + 0.0004)
    assert pickle.dumps(routed.part) == source
    editor.undo()
    assert pickle.dumps(editor.wire_path()) == pickle.dumps(routed.part.wire_path)


def test_split_outer_contours_are_not_mistaken_for_complete_openings(routed):
    editor = InterconnectionEditor(routed.part)
    for index in (0, 2):
        pieces = [s for s in editor.sections if s.owner.kind == "contour_loop" and s.owner.loop_index == index]
        assert len(pieces) > 1 and all(s.cut is None for s in pieces)


def test_cut_moves_forward_source_arc_and_both_connections_in_one_undo_step(routed):
    editor = InterconnectionEditor(routed.part)
    original = editor.to_plan()
    cut = inner_cut(editor)
    editor.move_cut(cut.key, cut.cut[0] + 0.0005, cut.cut[1] * 0.9)
    moved = editor.section(cut.key)
    expected, _ = ContourArc(routed.part.contour_lines[1]).open(*moved.cut)
    np.testing.assert_array_equal(moved.points, expected.v)
    rows = editor.sections
    index = next(i for i, s in enumerate(rows) if s.key == cut.key)
    np.testing.assert_array_equal(rows[index - 1].points[-1], moved.points[0])
    np.testing.assert_array_equal(rows[index + 1].points[0], moved.points[-1])
    revised = editor.to_plan()
    editor.undo()
    assert editor.to_plan() == original
    editor.redo()
    assert editor.to_plan() == revised
    assert editor.undo_label == "Move contour cut"


def test_locked_neighbor_rejects_entire_cut_edit(routed):
    editor = InterconnectionEditor(routed.part)
    cut = inner_cut(editor)
    index = next(i for i, s in enumerate(editor.sections) if s.key == cut.key)
    editor.set_locked(editor.sections[index - 1].key, True)
    before = editor.to_plan()
    with pytest.raises(ValueError, match="neighboring"):
        editor.move_cut(cut.key, cut.cut[0] + 0.001)
    assert editor.to_plan() == before


def test_surface_guide_uses_existing_router_preserves_endpoints_and_lock(routed):
    editor = InterconnectionEditor(routed.part)
    section = surface_connection(editor)
    midpoint = section.points.mean(axis=0) + [0, 0.0007, 0.004]
    editor.set_guides(section.key, [midpoint])
    guided = editor.section(section.key)
    np.testing.assert_array_equal(guided.points[[0, -1]], section.points[[0, -1]])
    np.testing.assert_allclose(guided.points[:, 2], 0, atol=1e-12)
    editor.set_locked(section.key, True)
    editor.route_unlocked()
    np.testing.assert_array_equal(editor.section(section.key).points, guided.points)
    with pytest.raises(ValueError, match="unlocked"):
        editor.set_guides(section.key, [])


def test_failed_multi_connection_route_is_atomic(routed, monkeypatch):
    editor = InterconnectionEditor(routed.part)
    before = editor.to_plan()
    real = editor._route
    calls = []

    def route(*args):
        calls.append(1)
        if len(calls) > 1:
            raise ValueError("No feasible corridor")
        return real(*args)

    monkeypatch.setattr(editor, "_route", route)
    with pytest.raises(ValueError, match="corridor"):
        editor.route_unlocked()
    assert editor.to_plan() == before and not editor.undo_label


def test_disconnected_drafts_never_become_wire_paths(routed):
    editor = InterconnectionEditor(routed.part)
    keys = [s.key for s in editor.sections if s.editable][:2]
    editor.disconnect(keys[0])
    assert editor.connectivity == {"complete": False, "components": 1, "free_ends": 2, "closed": True}
    with pytest.raises(ValueError, match="Reconnect"):
        editor.build_part()
    editor.disconnect(keys[1])
    assert editor.connectivity["components"] == 2
    editor.undo()
    editor.undo()
    assert editor.connectivity["complete"]


def test_plan_roundtrip_preserves_fields_and_clears_all_stale_evidence(routed):
    editor = InterconnectionEditor(routed.part)
    cut = inner_cut(editor)
    editor.move_cut(cut.key, cut.cut[0] + 0.0002)
    guide = surface_connection(editor)
    editor.set_guides(guide.key, [guide.points.mean(axis=0) + [0, 0.0002, 0]])
    plan = json.loads(json.dumps({"schema": PLAN_SCHEMA, "circuits": [editor.to_plan()]}))
    original = deepcopy(routed.part)
    original.field_by_layout = np.ones((2, 3))
    original.manufacturing_diagnostics = {"manufacturing_ok": True}
    original.coil_inductance = 5
    result = apply_editor_plan([original], plan)[0]
    assert result.field_by_layout is None and result.manufacturing_diagnostics == {} and result.coil_inductance == 0
    points = np.array([[0, 0, 0.02], [0.01, 0, 0.02]])
    np.testing.assert_array_equal(
        biot_savart_wire_path(result.wire_path.v, points), biot_savart_wire_path(editor.wire_path().v, points)
    )
    reopened = InterconnectionEditor(result)
    reopened.load_plan(plan["circuits"][0])
    np.testing.assert_array_equal(reopened.wire_path().v, result.wire_path.v)


@pytest.mark.parametrize("change", ["mesh", "contour", "current", "owner"])
def test_plan_requires_exact_automatic_source(routed, change):
    original = InterconnectionEditor(routed.part)
    changed = deepcopy(routed.part)
    if change == "mesh":
        changed.mesh.vertices[0, 0] += 1e-6
    if change == "contour":
        changed.contour_lines[0].v[0, 0] += 1e-6
    if change == "current":
        changed.contour_step *= 2
    if change == "owner":
        changed.wire_path.segment_owners[0].connector_role = "changed"
    with pytest.raises(ValueError, match="different automatic route"):
        InterconnectionEditor(changed).load_plan(original.to_plan())


def test_tampered_protected_geometry_or_endpoints_is_rejected(routed):
    editor = InterconnectionEditor(routed.part)
    row = inner_cut(editor)
    plan = editor.to_plan()
    plan["sections"] = [{"key": row.key, "points_m": row.points.tolist()}]
    with pytest.raises(ValueError, match="contour vertices"):
        editor.load_plan(plan)
    section = surface_connection(editor)
    points = section.points.copy()
    points[0, 0] += 0.001
    plan["sections"] = [
        {"key": section.key, "points_m": points.tolist(), "geometry": [g.value for g in section.geometry]}
    ]
    with pytest.raises(ValueError, match="endpoints"):
        editor.load_plan(plan)


def test_session_draft_roundtrip_and_stale_validation_cannot_apply(tmp_path):
    session, _ = simple_session(tmp_path)
    receipt = EditorValidation(session.identity, True, ())
    assert session.accept_validation(receipt) and session.can_apply
    session.editors[0].set_locked("s0", True)
    assert not session.can_apply and not session.accept_validation(receipt)
    path = session.save_draft(tmp_path / "draft.cgwinding")
    other, _ = simple_session(tmp_path)
    other.load_draft(path)
    assert other.plan() == session.plan() and not other.can_apply
    assert not list(tmp_path.glob("*.tmp"))


def test_editor_setting_only_invalidates_routing(tmp_path):
    session, checkpoint = simple_session(tmp_path)
    assert (
        invalidated_stage(checkpoint.config, replace(checkpoint.config, routing_editor_plan=session.plan()))
        == PipelineStage.ROUTED
    )


def test_full_production_validation_of_unchanged_circular_winding(tmp_path):
    session, _ = simple_session(tmp_path)
    report = session.validation_job()()
    assert report.accepted, report.reasons
    assert report.diagnostics["physical"]["longitudinal_change"]["rms_T"] == 0
    assert report.diagnostics["current_refitted"] is False
    assert report.diagnostics["circuits"][0]["manufacturing"]["exact_collision_completed"]
    assert session.accept_validation(report) and session.can_apply


def test_missing_current_and_open_paths_cannot_pass_engineering(tmp_path):
    session, checkpoint = simple_session(tmp_path)
    sources = checkpoint.parts
    sources[0].contour_step = None
    editor = InterconnectionEditor(sources[0])
    plan = {"schema": PLAN_SCHEMA, "circuits": [editor.to_plan()]}
    report = validate_winding_revision("test", sources, plan, session.config, session.target)
    assert not report.accepted and "current" in " ".join(report.reasons)


def test_inherited_bend_failure_is_not_waived_for_a_manual_revision(tmp_path):
    session, _ = simple_session(tmp_path)
    session.config = replace(session.config, sweep_min_bend_radius=0.04)
    report = session.validation_job()()
    assert not report.accepted
    assert any("bend" in reason for reason in report.reasons)


def test_validation_cannot_hide_target_error_by_refitting_current(tmp_path):
    session, checkpoint = simple_session(tmp_path)
    state = checkpoint.state
    state.parts[0].contour_step = 2.0
    config = replace(checkpoint.config, layout_target_error_rms_limit_T=1e-12)
    changed = PipelineCheckpoint(PipelineStage.ROUTED, state, config, tmp_path)
    report = WindingEditorSession(changed).validation_job()()
    assert not report.accepted
    assert report.diagnostics["circuits"][0]["current_A"] == 2.0
    assert report.diagnostics["current_refitted"] is False
    assert any("rms" in reason for reason in report.reasons)


@pytest.mark.parametrize("margin, expected", [(1.0, False), (0.5, True)])
def test_editor_enforces_final_relative_field_contract_at_recorded_current(tmp_path, margin, expected):
    _, checkpoint = simple_session(tmp_path)
    state = checkpoint.state
    state.target_field.b *= 1.04
    checkpoint = PipelineCheckpoint(
        checkpoint.stage, state, replace(checkpoint.config, target_error_tolerance_percent=3.0), tmp_path
    )
    editor = WindingEditorSession(checkpoint, design_margin=margin)
    report = editor.validation_job()()
    assert report.accepted is expected, report.reasons
    check = report.diagnostics["final_field_budget"]
    assert check["passed"] is expected
    assert check["design_margin"] == margin
    assert check["relative_limit"] == pytest.approx(0.03 / margin)
    assert check["best_fit_current_scale"] > 1.03
    assert report.diagnostics["current_refitted"] is False


def test_saved_budget_ratio_is_part_of_validation_identity(tmp_path):
    editor, checkpoint = simple_session(tmp_path)
    report = EditorValidation(editor.identity, True, ())
    changed = WindingEditorSession(checkpoint, design_margin=0.6)
    assert changed.identity != editor.identity
    assert not changed.accept_validation(report)


def test_lifted_connection_keeps_its_offset_when_guided(tmp_path):
    session, checkpoint = simple_session(tmp_path)
    part = checkpoint.parts[0]
    points = np.array([[-0.02, 0, 0.002], [0, 0, 0.002], [0.02, 0, 0.002]])
    part.wire_path = WirePath(points, [WireSegmentOwner("inter_group_connector")] * 2, ["offset_surface"] * 2)
    editor = InterconnectionEditor(part)
    editor.set_guides("s0", [[0, 0.005, 0.05]])
    changed = editor.section("s0")
    np.testing.assert_allclose(changed.points[:, 2], 0.002, atol=1e-12)
    np.testing.assert_array_equal(changed.points[[0, -1]], points[[0, -1]])
    assert {g.value for g in changed.geometry} == {"offset_surface"}


def test_pipeline_replays_saved_plan_after_automatic_routing_and_revalidates(tmp_path):
    from coilgen.pipeline import CoilGenPipeline

    session, checkpoint = simple_session(tmp_path)
    config = replace(
        checkpoint.config,
        coil_mesh_file="create planary mesh",
        skip_inductance_calculation=True,
        joint_group_routing=False,
        progress_flag=False,
    )
    state = checkpoint.state
    state.parts[0].wire_path = None
    loop = state.parts[0].contour_lines[0]
    middle = (loop.v[:-1] + loop.v[1:]) / 2
    loop.segment_faces = np.where(middle[:, 1] <= middle[:, 0], 0, 1)
    barycentric = []
    for a, b, face in zip(loop.v[:-1], loop.v[1:], loop.segment_faces):
        triangle = state.parts[0].mesh.vertices[state.parts[0].mesh.faces[face]]
        matrix = np.vstack((triangle.T, np.ones(3)))
        barycentric.append(np.linalg.lstsq(matrix, np.vstack((np.array([a, b]).T, np.ones(2))), rcond=None)[0].T)
    loop.segment_barycentric = np.asarray(barycentric)
    contours = PipelineCheckpoint(PipelineStage.CONTOURS, state, config, tmp_path)
    automatic = CoilGenPipeline(config, tmp_path).run_stage("routed", contours)
    editor = WindingEditorSession(automatic)
    editor.editors[0].set_locked("s0", True)
    manual_config = replace(config, routing_editor_plan=editor.plan())
    manual = CoilGenPipeline(manual_config, tmp_path).run_stage("routed", contours)
    np.testing.assert_array_equal(manual.parts[0].wire_path.v, automatic.parts[0].wire_path.v)
    assert manual.parts[0].topology_diagnostics["interconnection_editor"]["validation"] == "passed"
    restored = PipelineCheckpoint.load(manual.save(tmp_path / "manual.cgstage"))
    reopened = WindingEditorSession(restored)
    assert reopened.plan() == editor.plan()
