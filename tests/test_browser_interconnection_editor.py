"""Browser messages commit through the same directed winding transactions."""

import asyncio
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np

from coilgen.application.interconnection_editor import WindingEditorSession, winding_model_changes
from coilgen.interconnection_validation import EditorValidation
from coilgen.pipeline_stages import PipelineCheckpoint, PipelineStage, PipelineState
from _coilgen_test_support.interconnection_editor import routed as routed, inner_cut, surface_connection

from _coilgen_test_support.browser_interconnection_editor import (
    State as State,
    browser_editor as browser_editor,
    command as command,
)


def test_apply_requires_the_saved_final_field_budget(browser_editor):
    adapter, _ = browser_editor
    adapter.session.validation = EditorValidation(adapter.session.identity, True, ())
    adapter.refresh()
    assert adapter.state.winding_scene["can_apply"]
    adapter.browser.session.project.study = {"design_margin": 0.6}
    adapter.refresh()
    assert not adapter.state.winding_scene["can_apply"]
    assert not adapter.state.winding_scene["model_matches"]


def test_browser_freehand_draw_and_group_colors(browser_editor):
    adapter, _ = browser_editor

    async def exercise():
        row = surface_connection(adapter.editor)
        await command(adapter, "select", key=row.key)
        assert adapter.state.winding_scene["drawing_ports"]
        assert all(s["color"].startswith("#") for s in adapter.state.winding_scene["sections"])
        before = adapter.session.identity
        stroke = [row.points[0].tolist(), (row.points.mean(axis=0) + [0, 0.0005, 0]).tolist(), row.points[-1].tolist()]
        await command(adapter, "draw_connection", source=row.key, target=row.target, stroke=stroke)
        assert not adapter.state.winding_error
        assert adapter.session.identity != before and adapter.editor.undo_label == "Draw connection"
        await command(adapter, "undo")
        assert adapter.session.identity == before

    asyncio.run(exercise())


def test_browser_cut_volumes_preview_apply_and_reopen(browser_editor):
    adapter, _ = browser_editor

    async def exercise():
        cut = inner_cut(adapter.editor)
        await command(adapter, "select", key=cut.key)
        for kind in ("plane", "sphere", "box"):
            before = adapter.session.identity
            await command(adapter, "preview_shape", shape_kind=kind, group=cut.owner.group_index)
            assert not adapter.state.winding_error
            preview = adapter.state.winding_scene["seam_preview"]
            assert preview["shape"]["kind"] == kind and preview["cuts"] and preview["display"]["handles"]
            assert adapter.session.identity == before
            handle = next(h for h in preview["display"]["handles"] if h["role"] == "size0")
            center = np.array(preview["shape"]["center"])
            await command(
                adapter,
                "shape_handle",
                role="size0",
                point=(center + 0.9 * (np.array(handle["point"]) - center)).tolist(),
                group=cut.owner.group_index,
            )
            assert not adapter.state.winding_error
            await command(adapter, "apply_seam")
            assert not adapter.state.winding_error and adapter.editor.undo_label == "Apply cut shape"
            assert adapter.state.winding_scene["seam_preview"] is None
            await command(adapter, "undo")
            assert adapter.state.winding_scene["seam_preview"] is None
            await command(adapter, "redo")
            exact = adapter.editor.wire_path().v.copy()
            await command(adapter, "save")
            name = adapter.state.winding_download.split("/")[-1]
            await command(adapter, "open", name=name)
            assert not adapter.state.winding_error
            np.testing.assert_array_equal(adapter.editor.wire_path().v, exact)
            assert adapter.editor.section(cut.key).cut_shape["kind"] == kind
        await command(adapter, "clear_preview")
        assert adapter.state.winding_scene["seam_preview"] is None
        identity = adapter.session.identity
        await command(adapter, "preview_shape", shape_kind="plane")
        invalid = dict(adapter.state.winding_scene["seam_preview"]["shape"])
        invalid["size"] = [-1, 1, 1]
        await command(adapter, "preview_shape", shape=invalid)
        assert adapter.state.winding_error
        assert adapter.state.winding_scene["seam_preview"] is None
        assert adapter.session.identity == identity

    asyncio.run(exercise())


def test_browser_cut_scope_can_narrow_an_existing_preview_to_one_turn(browser_editor):
    adapter, _ = browser_editor

    async def exercise():
        cut = inner_cut(adapter.editor)
        await command(adapter, "select", key=cut.key)
        await command(adapter, "preview_shape", shape_kind="plane")
        shape = adapter.seam_preview["shape"]
        await command(adapter, "preview_shape", shape=shape, scope="selected", object_id=None)
        assert adapter.seam_preview["keys"] == [cut.key]
        assert [c["key"] for c in adapter.seam_preview["cuts"]] == [cut.key]
        before = {s.key: s.cut for s in adapter.editor.sections}
        await command(adapter, "apply_seam")
        assert not adapter.state.winding_error
        assert adapter.editor.cut_objects[0]["keys"] == [cut.key]
        assert all(s.cut == before[s.key] for s in adapter.editor.sections if s.key != cut.key)

    asyncio.run(exercise())


def test_browser_cut_drag_connection_save_reopen_and_exact_geometry(browser_editor):
    adapter, _ = browser_editor

    async def exercise():
        cut = inner_cut(adapter.editor)
        await command(adapter, "select", key=cut.key)
        await command(
            adapter,
            "drag_cut",
            key=cut.key,
            point=adapter.editor.cut_at_position(cut.key, cut.cut[0] + 0.0004).tolist(),
        )
        assert adapter.editor.revision == 1 and not adapter.state.winding_error
        connection = surface_connection(adapter.editor)
        guide = connection.points.mean(axis=0) + [0, 0.0003, 0]
        await command(adapter, "guide", key=connection.key, index=0, point=guide.tolist())
        assert adapter.editor.revision == 2 and adapter.editor.section(connection.key).guides
        exact = adapter.editor.wire_path().v.copy()
        identity = adapter.session.identity
        await command(adapter, "save")
        assert not adapter.state.winding_scene["dirty"]
        draft = adapter.state.winding_drafts[0]
        await command(adapter, "undo")
        assert adapter.session.identity != identity
        await command(adapter, "open", name=draft)
        assert adapter.session.identity == identity
        assert adapter.session.validation is None
        np.testing.assert_array_equal(adapter.editor.wire_path().v, exact)
        # A new browser session can reopen the on-disk plan on the same baseline.
        reopened = WindingEditorSession(
            PipelineCheckpoint(
                PipelineStage.ROUTED,
                PipelineState(parts=[adapter.editor.source], target_field=adapter.session.target),
                adapter.session.config,
                adapter.drafts,
            )
        )
        reopened.load_draft(adapter.drafts / draft)
        np.testing.assert_array_equal(reopened.editors[0].wire_path().v, exact)

    asyncio.run(exercise())


def test_browser_rejects_stale_edits_locked_routes_and_wrong_draft(browser_editor):
    adapter, _ = browser_editor

    async def exercise():
        section = surface_connection(adapter.editor)
        stale = dict(kind="disconnect", key=section.key, identity=adapter.session.identity, circuit=0)
        await command(adapter, "lock", key=section.key)
        before = deepcopy(adapter.session.plan())
        await adapter.action(stale)
        assert "older revision" in adapter.state.winding_error
        await command(adapter, "guide", key=section.key, index=0, point=section.points.mean(axis=0).tolist())
        assert adapter.state.winding_error
        assert adapter.session.plan() == before
        await command(adapter, "open", name="../elsewhere.cgwinding")
        assert "Choose a saved" in adapter.state.winding_error
        await command(
            adapter,
            "import",
            content=json.dumps(
                {
                    "schema": "coilgen.winding-draft/1",
                    "plan": {
                        "schema": "coilgen.interconnection-editor/1",
                        "circuits": [{"source_fingerprint": "changed", "sections": []}],
                    },
                }
            ),
        )
        assert "different automatic route" in adapter.state.winding_error
        assert adapter.session.plan() == before

    asyncio.run(exercise())


def test_browser_apply_requires_current_validation_and_matching_model(browser_editor):
    adapter, applied = browser_editor

    async def exercise():
        await command(adapter, "apply")
        assert not applied
        adapter.session.accept_validation(EditorValidation(adapter.session.identity, True, ()))
        await command(adapter, "apply")
        assert applied == [{"routing_editor_plan": adapter.session.plan()}]
        await command(adapter, "lock")
        await command(adapter, "apply")
        assert len(applied) == 1
        adapter.session.accept_validation(EditorValidation(adapter.session.identity, True, ()))
        adapter.browser.session.project.id = "another-project"
        await command(adapter, "apply")
        assert len(applied) == 1 and not adapter.state.winding_scene["can_apply"]

    asyncio.run(exercise())


def test_browser_failed_validation_report_is_saved_without_enabling_apply(browser_editor):
    adapter, applied = browser_editor

    async def exercise():
        # Exercise production validation on the routed geometry, including its failures.
        await command(adapter, "validate")
        assert adapter.session.validated and not adapter.session.can_apply
        report = json.loads((adapter.browser.exports / adapter.state.winding_report_url.split("/")[-1]).read_text())
        assert report["identity"] == adapter.session.identity and report["reasons"]
        assert report["diagnostics"]["current_refitted"] is False
        await command(adapter, "apply")
        assert not applied

    asyncio.run(exercise())


def test_worker_resolved_threads_do_not_make_the_source_model_stale(browser_editor):
    adapter, _ = browser_editor
    original = adapter.session.config
    recorded = replace(original, constrained_optimization_threads=1, parallel_worker_count=1)
    current = replace(original, constrained_optimization_threads=0, parallel_worker_count=0)
    assert winding_model_changes(recorded, current) == []
    assert "levels" in winding_model_changes(recorded, replace(current, levels=current.levels + 1))
    recorded = replace(recorded, research_physics={"objective": "power", "b0_T": [0, 0, 3]})
    current = replace(current, research_physics={"b0_T": [0, 0, 3], "objective": "power"})
    assert winding_model_changes(recorded, current) == []
    assert "research_physics" in winding_model_changes(
        recorded, replace(current, research_physics={"b0_T": [0, 0, 7], "objective": "power"})
    )


def test_browser_sketch_reconnect_and_bridge_messages(browser_editor):
    from _coilgen_test_support.winding_sketch import crossing_editor

    adapter, _ = browser_editor

    async def exercise():
        cut = inner_cut(adapter.editor)
        curve = adapter.editor.cut_curve(cut.key)
        middle = (curve[3] + curve[4]) / 2
        normal = np.cross(curve[4] - curve[3], [0, 0, 1])
        normal *= 0.0003 / np.linalg.norm(normal)
        await command(
            adapter, "preview_seam", stroke=[(middle - normal).tolist(), (middle + normal).tolist()], flat=False
        )
        assert adapter.state.winding_scene["seam_preview"]
        await command(adapter, "apply_seam")
        assert adapter.editor.undo_label == "Move seam", adapter.state.winding_error
        assert adapter.state.winding_scene["seam_preview"] is None
        edges = [s for s in adapter.editor.sections if s.editable][:3]
        for edge in edges:
            await command(adapter, "disconnect", key=edge.key)
        assert not adapter.state.winding_scene["can_validate"]
        for edge, destination in zip(edges, edges[1:] + edges[:1]):
            await command(adapter, "reconnect_endpoints", source=edge.key, target=destination.target)
        assert adapter.state.winding_scene["can_validate"], adapter.state.winding_error
        assert not adapter.state.winding_scene["ports"]
        adapter.session.editors = [crossing_editor()]
        adapter.selected = "s0"
        adapter.refresh()
        await command(adapter, "bridge", other="s2", relation="under", separation_mm=4, span_mm=40)
        assert adapter.selected == "s2" and adapter.editor.section("s2").bridge, adapter.state.winding_error
        await command(adapter, "save")
        assert adapter.state.winding_download
        await command(adapter, "remove_bridge")
        assert not adapter.editor.section("s2").bridge

    asyncio.run(exercise())
