"""Real pipeline, portable project and both frontends share one circuit plan."""

from dataclasses import replace
from dataclasses import asdict
from copy import deepcopy
from hashlib import sha256
import json
import zipfile

import numpy as np
import pytest

from coilgen.application import DesignService, ProjectDocument
from coilgen.application.group_editor import GroupEditor, load_group_editor
from coilgen.pipeline import CoilGenPipeline
from coilgen.pipeline_stages import PipelineCheckpoint, PipelineStage, invalidated_stage
from _coilgen_test_support.pipeline_stages import config as small_config


@pytest.fixture(scope="module")
def prepared_groups():
    config = replace(
        small_config(),
        skip_postprocessing=False,
        skip_sweep=True,
        joint_group_routing=False,
        levels=4,
        conductor_cross_section_width=0.0003,
        conductor_cross_section_height=0.0003,
    )
    checkpoint = CoilGenPipeline(config).run_stage("contours")
    editor = GroupEditor.from_parts(checkpoint.parts, config)
    assert len(editor.rows()) >= 2
    return config, checkpoint, editor


def test_real_pipeline_reuses_contours_and_roundtrips_independent_circuits(prepared_groups, tmp_path):
    config, contours, editor = prepared_groups
    plan = deepcopy(editor.plan)
    plan["mode"] = "per_group"
    changed = replace(config, routing_group_plan=plan)
    assert invalidated_stage(config, changed) is PipelineStage.ROUTED
    checkpoint = CoilGenPipeline(changed).run_stage("evaluated", contours)
    result = checkpoint.result
    assert len(result.coil_parts) == len(editor.rows())
    assert sum(len(part.contour_lines) for part in result.coil_parts) == sum(
        len(part.contour_lines) for part in contours.parts
    )
    np.testing.assert_array_equal(result.b_field_opt_sf, contours.sf_b_field)
    assert len(result.source_parts) == len(contours.parts)
    assert all(part.sensitivity_matrix is None for part in result.coil_parts)
    np.testing.assert_allclose(
        result.field_layout_per1Amp, sum((part.field_by_layout for part in result.coil_parts)), rtol=1e-13, atol=1e-20
    )
    assert result.regularization_diagnostics["group_circuits_fixed_current_field"]["fixed_current_a"] == pytest.approx(
        result.coil_parts[0].contour_step
    )
    restored = PipelineCheckpoint.load(checkpoint.save(tmp_path / "circuits.cgstage"))
    assert len(restored.result.source_parts) == len(contours.parts)
    assert [part.circuit_id for part in restored.result.coil_parts] == [part.circuit_id for part in result.coil_parts]
    for first, second in zip(restored.result.coil_parts, result.coil_parts):
        np.testing.assert_array_equal(first.wire_path.v, second.wire_path.v)
    project = ProjectDocument.from_config(changed)
    path = project.save(tmp_path / "groups.json")
    loaded = ProjectDocument.load(path)
    assert loaded.config.routing_group_plan == plan


def test_circuit_partition_preserves_independent_sheet_fields_and_density(prepared_groups):
    from coilgen.application.gradient_studies import MeasuredFieldMap
    from coilgen.application.research_requirements import (
        DesiredDataset,
        RegionalRequirement,
        ResearchRequirements,
        evaluate_requirements,
    )
    from coilgen.research.convergence import check_sheet_integration
    from coilgen.research.validation import validate_on_independent_grid

    config, contours, source = prepared_groups
    baseline = CoilGenPipeline(config).run_stage("evaluated", contours)
    editor = GroupEditor.from_parts(source.parts, config)
    editor.assign("per_group")
    routed = CoilGenPipeline(replace(config, routing_group_plan=editor.plan)).run_stage("evaluated", contours)
    expected = validate_on_independent_grid(baseline.result, resolution=5)
    actual = validate_on_independent_grid(routed.result, resolution=5)
    assert expected["available"] and actual["available"]
    for key in ("points_m", "target_T", "field_T"):
        np.testing.assert_array_equal(actual[key], expected[key])
    assert np.linalg.norm(actual["field_T"]) > 0
    for key in ("sheet_current_density_A_per_m", "sheet_current_density_points_m"):
        np.testing.assert_array_equal(routed.scientific_arrays()[key], baseline.scientific_arrays()[key])

    points, target = expected["points_m"][:3], expected["target_T"][:3]
    requirements = ResearchRequirements(
        (
            RegionalRequirement(
                "ROI",
                DesiredDataset(MeasuredFieldMap(points, target)),
                peak_limit_T=1e-5,
            ),
        )
    )
    expected_review, expected_arrays = evaluate_requirements(baseline.result.coil_parts, config, requirements)
    actual_review, actual_arrays = evaluate_requirements(
        routed.result.coil_parts,
        config,
        requirements,
        source_parts=routed.result.source_parts,
    )
    assert actual_review["support_geometry_sha256"] == expected_review["support_geometry_sha256"]
    assert actual_review["stream_function_sha256"] == expected_review["stream_function_sha256"]
    expected_sheet = [row for row in expected_review["rows"] if row["scope"] == "continuous_sheet"]
    actual_sheet = [row for row in actual_review["rows"] if row["scope"] == "continuous_sheet"]
    assert expected_sheet and len(expected_sheet) == len(actual_sheet)
    for expected_row, actual_row in zip(expected_sheet, actual_sheet):
        for suffix in ("_points_m", "_field_T", "_target_T"):
            np.testing.assert_array_equal(
                actual_arrays[actual_row["array_prefix"] + suffix],
                expected_arrays[expected_row["array_prefix"] + suffix],
            )
    expected_integration = check_sheet_integration(
        baseline.result,
        points,
        relative_tolerance=0.1,
        orders=(2, 3),
        subdivisions=(0,),
    )
    actual_integration = check_sheet_integration(
        routed.result,
        points,
        relative_tolerance=0.1,
        orders=(2, 3),
        subdivisions=(0,),
    )
    for expected_row, actual_row in zip(expected_integration["comparisons"], actual_integration["comparisons"]):
        np.testing.assert_array_equal(actual_row["field_T"], expected_row["field_T"])
        assert actual_row["triangle_count"] == expected_row["triangle_count"]


@pytest.mark.parametrize("schema_version", [6, 7])
def test_historical_checkpoints_supply_defaults_without_granting_numerical_reuse(
    prepared_groups,
    tmp_path,
    schema_version,
):
    config, contours, _editor = prepared_groups
    original = CoilGenPipeline(config).run_stage("evaluated", contours)
    path = original.save(tmp_path / "historical.cgstage")
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        arrays = archive.read("arrays.npz")

    def historical_fields(value):
        if isinstance(value, list):
            for item in value:
                historical_fields(item)
        elif isinstance(value, dict):
            if value.get("$class") in {"PipelineState", "CoilGenResult"}:
                value["fields"].pop("source_parts")
            if schema_version == 6:
                if value.get("$class") == "CoilGenConfig":
                    value["fields"].pop("routing_group_plan")
                if value.get("$class") == "ConnectedGroup":
                    value["fields"].pop("routing_spiral_order")
            for item in value.values():
                historical_fields(item)

    historical_fields(manifest["payload"])
    manifest["schema"] = f"coilgen.pipeline-checkpoint/{schema_version}"
    manifest["fingerprints"] = {key: "historical_engine" for key in manifest["fingerprints"]}
    manifest.pop("payload_sha256")
    manifest["payload_sha256"] = sha256(
        json.dumps(manifest, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("arrays.npz", arrays)
    restored = PipelineCheckpoint.load(path)
    assert restored.config.routing_group_plan is None and restored.result.source_parts == []
    assert all(group.routing_spiral_order == "auto" for part in restored.parts for group in part.connected_groups)
    np.testing.assert_array_equal(restored.result.field_by_layout, original.result.field_by_layout)
    with pytest.raises(ValueError, match="incompatible|changed|differ|identity"):
        restored.assert_compatible(config, tmp_path)


def test_verified_contour_run_can_be_edited_without_changing_model_defaults(prepared_groups, tmp_path):
    config, _contours, _editor = prepared_groups
    project = ProjectDocument.from_config(config, name="Group preparation")
    DesignService.run(project, tmp_path / "run", pipeline_stage="contours", fabricate=False, threads=1)
    record = DesignService.load_run(tmp_path / "run")
    with pytest.raises(ValueError, match="different project"):
        load_group_editor(ProjectDocument.from_config(config), record)
    editor = load_group_editor(project, record)
    assert not editor.dirty
    assert editor.apply_changes(project) == {}
    assert project.config.routing_group_plan is None
    editor.assign("per_group")
    updated = project.update_inputs(editor.apply_changes(project))
    assert updated.config.routing_group_plan["mode"] == "custom"
    assert project.config.routing_group_plan is None


def test_swept_circuit_packages_keep_circuit_identity_units_and_checks(prepared_groups, tmp_path):
    from coilgen.application.cad import export_catalog, export_geometry

    config, _contours, source = prepared_groups
    config = replace(config, skip_sweep=False, skip_normal_shift=True, target_region_resolution=5)
    contours = CoilGenPipeline(config).run_stage("contours")
    editor = GroupEditor.from_parts(contours.parts, config)
    editor.assign("per_group")
    project = ProjectDocument.from_config(
        replace(config, routing_group_plan=editor.plan), name="Independent group circuits"
    )
    DesignService.run(project, tmp_path / "run", pipeline_stage="evaluated", fabricate=True, threads=1)
    record = DesignService.load_run(tmp_path / "run")
    circuits = record.metadata["routing"]["parts"]
    assert len({row["circuit_id"] for row in circuits}) == len(editor.rows())
    options = [option for option in export_catalog(record) if option.format in {"stl", "csv"}]
    assert options and any(option.format == "stl" for option in options)
    package = export_geometry(record, tmp_path / "cad", [option.name for option in options], "mm", allow_draft=True)
    manifest = package.manifest
    assert manifest["routing_circuits"] == circuits
    assert manifest["requested_stl_unit"] == "mm"
    assert manifest["construction_released"] is False
    assert len({key for row in manifest["routing_circuits"] for key in row["source_group_keys"]}) == len(editor.rows())
    assert (
        record.metadata["summary"]["accepted"] is not True
        or record.metadata["pipeline_sweeps"]["checks_passed"] is True
    )
    routed = PipelineCheckpoint.load(record.directory / "pipeline_checkpoint.cgstage").result
    clearance = routed.regularization_diagnostics["group_circuit_clearance"]
    assert clearance["checked"]
    assert clearance["checked_body_pairs"] == len(circuits) * (len(circuits) - 1) // 2


def test_qt_integrated_editor_uses_shared_group_draft(prepared_groups, windows):
    config, _contours, source = prepared_groups
    window = windows(ProjectDocument.from_config(config))
    window._open_groups_editor()
    widget = window.groups_editor
    editor = GroupEditor.from_parts(source.parts, config)
    widget.set_editor(editor, window.project.fingerprint)
    assert window.plot_stack.currentWidget() is widget
    assert widget.table.rowCount() == len(editor.rows())
    key = editor.rows()[0]["key"]
    widget._change(key, "circuit", "invalid circuit name")
    assert widget.dirty and widget.invalid_fields and not widget.apply_button.isEnabled()
    widget._refresh()
    assert widget.table.cellWidget(0, 3).text() == "invalid circuit name"
    widget.revert()
    widget._change(key, "circuit", "A")
    assert widget.apply_button.isEnabled() and widget.revert_button.isEnabled()
    widget.revert()
    widget.editor.selected = {editor.rows()[0]["key"]}
    widget._assign("separate")
    initial_circuits = [row["circuit"] for row in source.rows()]
    before = window.project
    window._apply_group_plan()
    assert window.project.revision == before.revision + 1
    assert not widget.dirty
    window.select_node("winding")
    assert window._raw_value("winding_intent") == "Groups & circuits plan"
    assert window._draft_controls["winding_intent"].count() == 5
    window.undo_stack.undo()
    assert window.project.config.routing_group_plan is None
    assert [row["circuit"] for row in widget.editor.rows()] == initial_circuits
    assert window._raw_value("winding_intent") == "Connected circuit"
    assert window._draft_controls["winding_intent"].count() == 4
    widget._change(key, "cut", "high")
    retained = deepcopy(widget.editor.plan)
    original = window.project
    window._set_project(window.project.update_inputs({"levels": config.levels + 2}))
    assert widget.stale and widget.dirty and not widget.apply_button.isEnabled()
    assert widget.editor.plan == retained
    window._set_project(original)
    assert not widget.stale and widget.apply_button.isEnabled()
    window._set_project(original.update_inputs({"levels": config.levels + 2}))
    widget.revert()
    assert widget.editor is None and widget.table.rowCount() == 0


def test_qt_group_preparation_waits_for_verified_result_loading(prepared_groups, windows, monkeypatch):
    from PySide6.QtCore import QTimer

    config, _contours, _source = prepared_groups
    window = windows(ProjectDocument.from_config(config))
    window._open_groups_editor()
    pending, shown, loaded = [], [], []
    monkeypatch.setattr(window, "compute", lambda **kwargs: {"id": "preparation"})
    monkeypatch.setattr(QTimer, "singleShot", lambda _delay, callback: pending.append(callback))
    monkeypatch.setattr(window, "show_run", shown.append)
    monkeypatch.setattr(window, "_load_groups", lambda: loaded.append(True))
    monkeypatch.setattr(window, "_run_verification_error", lambda _id: "")
    window.states["preparation"] = {"status": "completed"}
    window._prepare_groups()
    pending.pop(0)()
    assert not shown and pending
    window.runs["preparation"] = object()
    pending.pop(0)()
    assert shown == ["preparation"] and loaded == [True]
    window.runs.clear()


def test_browser_editor_applies_the_same_draft_and_blocks_unapplied_compute(prepared_groups, tmp_path):
    pytest.importorskip("trame")
    from coilgen.web.app import create_app

    config, _contours, source = prepared_groups
    server, browser = create_app(tmp_path / "browser", server_name="group-circuit-integration")
    try:
        browser.session.new_project("Groups")
        browser.session.apply_config(asdict(config))
        browser.groups.editor = GroupEditor.from_parts(source.parts, config)
        browser.groups.fingerprint = browser.session.project.fingerprint
        browser.groups.open()
        browser.groups.refresh()
        assert len(server.state.groups_rows) == len(source.rows())
        key = source.rows()[0]["key"]
        browser.groups.change(key, "circuit", "invalid circuit name")
        assert server.state.groups_dirty and not browser._ready_to_submit()
        browser.groups.select(key, True)
        assert server.state.groups_rows[0]["circuit"] == "invalid circuit name"
        assert server.state.groups_invalid
        browser.groups.revert()
        browser.groups.assign("per_group")
        assert server.state.groups_dirty
        assert not browser._ready_to_submit()
        browser.groups.apply()
        assert not server.state.groups_dirty
        plan = browser.session.project.config.routing_group_plan
        assert len({row["circuit"] for row in plan["groups"]}) == len(source.rows())
        browser.undo()
        assert browser.session.project.config.routing_group_plan is None
        assert [row["circuit"] for row in server.state.groups_rows] == [row["circuit"] for row in source.rows()]
        browser.redo()
        assert browser.session.project.config.routing_group_plan == plan
        assert len({row["circuit"] for row in server.state.groups_rows}) == len(source.rows())
        browser.open_advanced("Routing")
        assert server.state.advanced_key_search == "Routing"
        assert any(
            entry["key"] == "force_cut_selection" and entry["category"] == "Routing"
            for entry in server.state.advanced_entries
        )
        draft = json.loads(server.state.advanced_config)
        draft["tikonov_reg_factor"] = 123.0
        server.state.advanced_config = json.dumps(draft)
        owner = browser.session.project.fingerprint
        browser.open_advanced("Regularization")
        assert json.loads(server.state.advanced_config) == draft
        assert browser.session.project.fingerprint == owner
        browser.groups.change(key, "cut", "high")
        retained = deepcopy(browser.groups.editor.plan)
        browser.session.apply_config({"levels": config.levels + 2})
        assert server.state.groups_stale and server.state.groups_dirty
        assert browser.groups.editor.plan == retained
        changed_owner = browser.session.project.fingerprint
        browser.groups.apply()
        assert browser.session.project.fingerprint == changed_owner
        browser.groups.revert()
        assert browser.groups.editor is None and not server.state.groups_rows
    finally:
        browser.plotter.close()
        browser.session.close()


def test_browser_group_load_rejects_an_identical_model_under_a_new_owner(prepared_groups, tmp_path, monkeypatch):
    import asyncio
    from threading import Event

    pytest.importorskip("trame")
    from coilgen.web.app import create_app
    from coilgen.web import group_circuits

    config, _contours, source = prepared_groups
    server, browser = create_app(tmp_path / "browser", server_name="group-load-owner-race")
    entered, release = Event(), Event()

    def load(project, record):
        entered.set()
        assert release.wait(10), "Group load was not released"
        return GroupEditor.from_parts(source.parts, config)

    monkeypatch.setattr(group_circuits, "load_group_editor", load)

    async def switch_during_load():
        browser.session.new_project("First owner")
        browser.session.apply_config(asdict(config))
        original = browser.session.project
        pending = asyncio.create_task(browser.groups.load(object()))
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            browser.session.new_project("Second owner")
            browser.session.apply_config(asdict(config))
            assert browser.session.project.id != original.id
            assert browser.session.project.fingerprint == original.fingerprint
        finally:
            release.set()
            await pending
        assert browser.groups.editor is None
        assert not server.state.groups_rows
        assert not server.state.groups_busy
        assert "changed" in server.state.groups_note

    try:
        asyncio.run(switch_during_load())
    finally:
        release.set()
        browser.plotter.close()
        browser.session.close()


from _coilgen_test_support.workbench_usability import application as application, windows as windows  # noqa: E402,F401
