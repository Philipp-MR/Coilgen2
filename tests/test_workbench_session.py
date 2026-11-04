"""The Qt-free session shared by the desktop and browser workbenches."""

import subprocess
import sys
import time

import numpy as np
import pytest

from coilgen.application.scene_spec import DATASET_GROUPS, scene_layers
from coilgen.application.session import (
    ACTION_IDS,
    SETTINGS_NODES,
    WorkbenchSession,
    request_controls,
)


def test_session_imports_no_gui_toolkit():
    code = (
        "import sys, coilgen.application.session, coilgen.application.scene_spec;"
        "print(any(name.split('.')[0] in {'PySide6', 'pyvistaqt', 'vtkmodules'} for name in sys.modules))"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "False"


def test_every_setting_reads_and_writes_its_own_value(tmp_path):
    session = WorkbenchSession(tmp_path)
    session.load_template("halbach_gz")
    nodes = {node for node, _label in SETTINGS_NODES}
    values = session.settings_values()
    assert set(values) == {item.key for item in session.settings_fields()}
    assert "cylinder_radius_mm" in values and "biplanar_separation_mm" not in values
    for item in session.settings_fields():
        assert item.node in nodes
        # Writing back the displayed value must not change the model.
        changes = item.set(values[item.key], session.project.config, session.project.study)
        current = session.project.study if item.study else session.project.config
        for name, value in changes.items():
            existing = current[name] if item.study else getattr(current, name)
            if isinstance(value, tuple):
                np.testing.assert_allclose(np.asarray(value, dtype=float), np.asarray(existing, dtype=float))
            else:
                assert value == pytest.approx(existing) if isinstance(value, float) else value == existing


def test_apply_undo_redo_and_validation(tmp_path):
    session = WorkbenchSession(tmp_path)
    session.load_template("cylinder")
    original = session.project
    session.apply_settings({"levels": 14, "interconnection_axis": "support_axis", "min_bend_mm": 8})
    assert session.project.config.levels == 14
    assert session.project.config.interconnection_axis == "support_axis"
    assert session.project.config.sweep_min_bend_radius == pytest.approx(0.008)
    assert session.undo() and session.project.fingerprint == original.fingerprint
    assert session.redo() and session.project.config.levels == 14
    with pytest.raises(ValueError):
        session.apply_settings({"interconnection_axis": "sideways"})
    assert session.project.config.interconnection_axis == "support_axis"


def test_controls_match_the_desktop_request(tmp_path):
    session = WorkbenchSession(tmp_path)
    session.load_template("cylinder")
    project = session.project
    assert session.controls("sheet") == request_controls(
        project,
        pipeline_stage="sheet",
        validation_resolution=project.study["validation_resolution"],
        design_margin=project.study["design_margin"],
        fabricate=project.study["fabricate"],
    )


def test_profile_presets_and_save_open_round_trip(tmp_path):
    session = WorkbenchSession(tmp_path)
    session.load_template("cylinder")
    session.apply_profile_preset("omega")
    assert session.project.config.conductor_path_reference == "profile_origin"
    assert len(session.project.config.cross_sectional_points) > 8
    path = session.save_project()
    reopened = WorkbenchSession(tmp_path)
    reopened.open_project(path)
    assert reopened.assembly.fingerprint == session.assembly.fingerprint


def test_action_registry_and_dataset_menu_are_complete():
    assert {"compute", "export_geometry", "load_template", "undo", "redo", "open_run"} <= ACTION_IDS
    keys = [key for _group, entries in DATASET_GROUPS for _label, key in entries]
    assert len(keys) == len(set(keys)) and "swept" in keys and "routed_residual" in keys


def test_real_prepared_stage_runs_and_previews(tmp_path):
    session = WorkbenchSession(tmp_path)
    session.load_template("cylinder")
    state = session.compute("prepared")
    deadline = time.monotonic() + 240
    while session.active() and time.monotonic() < deadline:
        time.sleep(0.5)
        session.poll()
    try:
        final = session.states[state["id"]]
        assert final["status"] == "completed", final
        record = session.load_run(state["id"])
        layers, reason = scene_layers(record.arrays, "support")
        assert not reason and layers[0].kind == "surface" and len(layers[0].faces)
        assert isinstance(session.run_summary(record)["verdicts"], dict)
        assert session.export_options(record) == ()
    finally:
        session.close()
