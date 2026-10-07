"""Conical shim target amplitudes remain explicit in world coordinates."""

from dataclasses import asdict
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from coilgen.application.presets import preset_study
from coilgen.target import evaluate_field_expression


def _recipe():
    path = Path(__file__).resolve().parents[1] / "examples" / "conical_harmonic_workbench.py"
    spec = importlib.util.spec_from_file_location("conical_workbench_recipe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_conical_recipe_uses_correct_absolute_z2_amplitude_and_physical_study():
    project = _recipe().build_project()
    config = project.config
    radius = config.target_region_radius
    points = radius * np.array(((0, 0, 1), (0, 0, -1), (1, 0, 0), (0, 1, 0)))

    values = evaluate_field_expression(config.field_shape_function, points)

    np.testing.assert_allclose(values, (10e-6, 10e-6, -5e-6, -5e-6), atol=1e-18)
    assert config.target_gradient_normalization == "off"
    assert config.target_field_units == "T" and config.design_context == "mr_shim"
    assert config.surface_is_cylinder_flag is False
    assert config.target_region_resolution == 17
    assert config.conductor_thickness == config.conductor_cross_section_height == pytest.approx(0.001)
    assert {key: project.study[key] for key in preset_study("conical_harmonic")} == preset_study("conical_harmonic")


def test_conical_gui_receives_saved_project_path_and_script_keeps_results_in_json(tmp_path, monkeypatch):
    import coilgen.gui

    module = _recipe()
    launched = {}
    monkeypatch.setattr(coilgen.gui, "launch", lambda project, **kwargs: launched.update(kwargs) or 0)

    assert module.main(["--output", str(tmp_path), "--gui"]) == 0

    assert launched["project_path"] == (tmp_path / "conical_z2.coilgen.json").resolve()
    assert launched["project_path"].is_file()

    # The compute path attaches the exact completed record, rather than only
    # leaving runtime files beside the input JSON. A prepared record suffices
    # to exercise the persistence boundary without repeating the expensive route.
    from coilgen.application import DesignService, ProjectDocument
    from coilgen.research.application import run_design
    from coilgen.research.requests import ResearchRequest

    project = module.build_project()
    request = ResearchRequest.from_config(
        project.config,
        pipeline_stage="prepared",
        project_id=project.id,
        project_revision=project.revision,
        project_fingerprint=project.fingerprint,
        project_model_fingerprint=project.model_fingerprint,
    )
    monkeypatch.setattr(module, "build_project", lambda: project)
    monkeypatch.setattr(module.DesignService, "run", lambda _project, directory: run_design(request, directory))
    computed = tmp_path / "computed"

    assert module.main(["--output", str(computed), "--compute"]) == 0

    record = DesignService.load_run(computed / "run")
    restored = ProjectDocument.load(computed / "conical_z2.coilgen.json")
    assert len(restored.saved_runs) == 1
    assert restored.saved_runs[0].metadata == record.metadata
    assert asdict(restored.config) == asdict(project.config)
