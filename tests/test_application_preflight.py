"""Scientific readiness is usable without Qt, and never runs optimization."""

from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from coilgen.application import ProjectDocument
from coilgen.application import preflight
from coilgen.application.preflight import PreflightIssue, PreflightReport, check_preflight
from coilgen.application.support_parameters import SUPPORT_EDITORS
from coilgen.config import CoilGenConfig


def numeric_config():
    return CoilGenConfig(coil_mesh_file="create cylinder mesh", sf_opt_method="tikhonov", research_physics=None)


def test_preflight_and_support_catalog_import_without_gui_runtime():
    code = """
import builtins
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.startswith(('PySide', 'pyvista', 'coilgen.gui')):
        raise AssertionError('GUI runtime imported: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from coilgen.application.preflight import check_preflight
from coilgen.application.support_parameters import SUPPORT_EDITORS
assert SUPPORT_EDITORS
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_valid_preflight_preserves_configuration_and_never_assembles_operators(monkeypatch):
    monkeypatch.setattr("coilgen.physics.calculate_sensitivity_matrix", lambda *_a, **_k: pytest.fail("Operator built"))
    monkeypatch.setattr(preflight, "find_spec", lambda _name: pytest.fail("Unused solver checked"))
    project = ProjectDocument.from_config(numeric_config())
    before = project.to_dict()
    report = check_preflight(project)
    assert report.ready and not report.blocking
    report.raise_for_errors()
    assert project.to_dict() == before


def test_blocking_and_warning_diagnostics_remain_distinct():
    warning = PreflightIssue("Fallback selected", "optimization", "solver", "Inspect solver", severity="warning")
    error = PreflightIssue("No support", "geometry", "coil_mesh_file", "Choose support")
    report = PreflightReport((warning, error))
    assert report.warnings == (warning,) and report.blocking == (error,)
    with pytest.raises(ValueError, match="coil_mesh_file: No support"):
        report.raise_for_errors()
    PreflightReport((warning,)).raise_for_errors()


@pytest.mark.parametrize("stage", ["unknown", "", 0, [], True])
def test_invalid_requested_stage_is_a_structured_blocker(stage):
    report = check_preflight(ProjectDocument.from_config(numeric_config()), pipeline_stage=stage)
    assert not report.ready
    assert report.blocking[0].key == "pipeline_stage"
    with pytest.raises(ValueError, match="pipeline_stage"):
        report.raise_for_errors()


def test_missing_conic_solver_retains_stage_and_fallback_policy(monkeypatch):
    monkeypatch.setattr(preflight, "find_spec", lambda _name: None)
    strict = ProjectDocument.from_config(
        replace(numeric_config(), sf_opt_method="constrained_quadratic", constrained_quadratic_fallback_policy="strict")
    )
    assert not check_preflight(strict).ready
    assert check_preflight(strict, pipeline_stage="prepared").ready
    fallback = strict.update_config({"constrained_quadratic_fallback_policy": "target_error_scan"})
    report = check_preflight(fallback)
    assert report.ready and any("configured legacy fallback" in issue.message for issue in report.warnings)
    assert fallback.config.sf_opt_method == "constrained_quadratic"


def test_stage_specific_profile_requirement_is_preserved():
    # A concave polygon may remain a sheet-study draft, but not a swept solid.
    config = replace(
        numeric_config(), cross_sectional_points=((0, 0), (0.002, 0), (0.001, 0.001), (0.002, 0.002), (0, 0.002))
    )
    project = ProjectDocument.from_config(config)
    assert check_preflight(project, pipeline_stage="sheet").ready
    issue = next(
        issue
        for issue in check_preflight(project, pipeline_stage="routed").blocking
        if issue.key == "cross_sectional_points"
    )
    assert issue.node == "material" and "convex" in issue.message


def test_owned_input_survives_original_deletion_and_reports_cache_corruption(tmp_path):
    source = tmp_path / "target.csv"
    source.write_text("x,y,z,bz\n0,0,0,0.00001\n0,0,0.01,0.00002\n", encoding="utf-8")
    project = ProjectDocument.from_config(
        replace(
            numeric_config(),
            target_field_definition_file=str(source),
            field_shape_function="none",
            target_gradient_normalization="none",
        )
    )
    source.unlink()
    assert check_preflight(project).ready
    cached = Path(project.config.target_field_definition_file)
    cached.write_bytes(b"corrupt")
    report = check_preflight(project)
    assert not report.ready and report.blocking[0].key == "target_field_definition_file"


def test_shared_cylinder_adapter_preserves_actual_span_and_untouched_parameters():
    editor = next(item for item in SUPPORT_EDITORS if item.source == "create cylinder mesh")
    original = tuple(numeric_config().cylinder_mesh_parameter_list)
    length = next(field for field in editor.fields if field.key == "length_mm")
    updated = editor.apply_edits(original, {"length_divisions": 40})
    assert editor.display_value(length, updated) == pytest.approx(editor.display_value(length, original))
    np.testing.assert_array_equal(updated[1:3], original[1:3])
    np.testing.assert_array_equal(updated[4:], original[4:])
