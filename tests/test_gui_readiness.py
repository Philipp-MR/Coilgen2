"""Readiness points to actual missing/invalid requirements, without solving."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import test_native_workbench as workbench_tests
from coilgen.application import ProjectDocument
from coilgen.application.assets import mesh_asset
from coilgen.application.presets import preset_config
from coilgen.application.targets import SolidHarmonicTarget
from coilgen.config import CoilGenConfig
from coilgen.gui import readiness
from coilgen.gui.editors import SUPPORT_EDITORS
from coilgen.gui.readiness import ReadinessIssue, ReadinessPanel, check_readiness
from coilgen.models import Mesh
from coilgen.optimization_solvers import conic_solver_order

application = workbench_tests.application


def numeric_config():
    return replace(preset_config(), sf_opt_method="tikhonov", research_physics=None)


@pytest.mark.parametrize("geometry", ["cylinder", "cone", "cone_y20"])
def test_valid_generated_models_need_no_imported_inputs_or_unused_solver(geometry, monkeypatch):
    monkeypatch.setattr(readiness, "find_spec", lambda _name: pytest.fail("Unused solver checked"))
    monkeypatch.setattr(
        "coilgen.physics.calculate_sensitivity_matrix", lambda *_a, **_k: pytest.fail("Operator assembled")
    )
    config = numeric_config()
    if geometry != "cylinder":
        config.coil_mesh_file = "create conical frustum mesh"
        config.conical_frustum_mesh_parameter_list = (0.3, 0.18, 0.14, 16, 8, 1, 0, 0, 0, 0, 0, 0)
        config.surface_is_cylinder_flag = False
    if geometry == "cone_y20":
        config = replace(config, **SolidHarmonicTarget(2, 0, 15.8533e-6, 0.05).to_config_overrides())
    project = ProjectDocument.from_config(config)
    before = project.fingerprint
    report = check_readiness(project)
    assert report.ready and not report.blocking
    assert project.fingerprint == before


@pytest.mark.parametrize(
    "field,node",
    [("coil_mesh_file", "geometry"), ("target_field_definition_file", "target"), ("target_mesh_file", "target")],
)
def test_real_missing_paths_have_correct_navigation(field, node, tmp_path):
    # A failed external import cannot become a ProjectDocument. Exercise its
    # actual config validator/path semantics without constructing a fake asset.
    config = replace(numeric_config(), **{field: "missing.npz"})
    project = SimpleNamespace(config=config, project_root=tmp_path, study={})
    report = check_readiness(project)
    issue = next(issue for issue in report.blocking if issue.key == field)
    assert issue.node == node and issue.title == "Missing input" and "not found" in issue.message
    assert "Import" in issue.action


def test_generated_geometry_error_retains_exact_parameter_and_suggestion():
    project = ProjectDocument.from_config(
        replace(numeric_config(), cylinder_mesh_parameter_list=(0.3, -0.16, 24, 12, 1, 0, 0, 0))
    )
    report = check_readiness(project)
    issue = next(issue for issue in report.blocking if issue.key == "cylinder_mesh_parameter_list[1]")
    assert issue.node == "geometry" and "positive" in issue.message
    assert len(report.blocking) == 1


@pytest.mark.parametrize("editor", SUPPORT_EDITORS, ids=lambda editor: editor.label)
def test_all_supported_primitive_defaults_pass_readiness_without_solving(editor, monkeypatch):
    monkeypatch.setattr(
        "coilgen.physics.calculate_sensitivity_matrix", lambda *_a, **_k: pytest.fail("Operator assembled")
    )
    config = replace(numeric_config(), coil_mesh_file=editor.source, surface_is_cylinder_flag=editor.cylindrical)
    project = ProjectDocument.from_config(config)
    fingerprint = project.fingerprint
    assert check_readiness(project).ready
    assert project.fingerprint == fingerprint


@pytest.mark.parametrize(
    "label,changed,diagnostic",
    [
        ("Disc", {1: 2}, "at least four divisions"),
        ("Annulus", {0: 0}, "inner radius must be positive"),
        ("Biplanar", {4: 0, 5: 0, 6: 0}, "normal must be nonzero"),
        ("Spherical cap", {1: np.pi}, "collapsed rim"),
        ("Ellipsoidal cap", {3: np.pi}, "collapsed rim"),
        ("Partial cylinder", {2: 0, 3: 2 * np.pi}, "coincident end seams"),
        ("Toroidal segment", {2: 0, 3: 2 * np.pi}, "coincident end seams"),
        ("Circular biplanar", {0: 1e-5}, "increase the grid-step target"),
    ],
)
def test_primitive_geometry_guards_do_not_misreport_an_undefined_rotation(label, changed, diagnostic):
    editor = next(item for item in SUPPORT_EDITORS if item.label == label)
    parameters = list(getattr(CoilGenConfig(), editor.parameter_field))
    for index, value in changed.items():
        parameters[index] = value
    config = replace(
        numeric_config(),
        coil_mesh_file=editor.source,
        surface_is_cylinder_flag=editor.cylindrical,
        **{editor.parameter_field: parameters},
    )
    report = check_readiness(ProjectDocument.from_config(config))
    assert len(report.blocking) == 1
    issue = report.blocking[0]
    assert issue.node == "geometry" and issue.key == editor.parameter_field
    assert issue.title == "Invalid support geometry" and diagnostic in issue.message
    assert "Geometry" in issue.action and "rotation axis" not in issue.action


def test_undefined_legacy_support_rotation_keeps_its_specific_corrective_action():
    project = ProjectDocument.from_config(
        replace(numeric_config(), cylinder_mesh_parameter_list=(0.3, 0.16, 24, 12, 0, 0, 0, np.pi / 2))
    )
    issue = check_readiness(project).blocking[0]
    assert issue.title == "Undefined support rotation"
    assert issue.node == "geometry" and issue.key == "rotation_axis_x"
    assert issue.action == "Define a nonzero rotation axis or set the angle to zero."


def test_imported_target_missing_field_is_reported_before_compute(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("x,y,z\n0,0,0\n", encoding="utf-8")
    config = replace(
        numeric_config(),
        target_field_definition_file=str(path),
        field_shape_function="none",
        target_gradient_normalization="none",
    )
    project = ProjectDocument.from_config(config)
    report = check_readiness(project)
    issue = next(issue for issue in report.blocking if issue.key == "target_field_definition_file")
    assert issue.node == "target" and "field" in issue.message.lower()
    assert "analytic" in issue.action


def test_original_import_deleted_does_not_invalidate_owned_input(tmp_path):
    path = tmp_path / "good.csv"
    path.write_text("x,y,z,bz\n0,0,0,0.00001\n0,0,0.01,0.00002\n", encoding="utf-8")
    project = ProjectDocument.from_config(
        replace(
            numeric_config(),
            target_field_definition_file=str(path),
            field_shape_function="none",
            target_gradient_normalization="none",
        )
    )
    path.unlink()
    assert check_readiness(project).ready
    cached = Path(project.config.target_field_definition_file)
    cached.unlink()
    assert check_readiness(project).ready  # regenerated from owned bytes
    cached.write_bytes(b"corrupt")
    report = check_readiness(project)
    assert not report.ready and report.blocking[0].node == "target"
    assert report.blocking[0].key == "target_field_definition_file"


def test_invalid_mesh_indices_are_actionable_geometry_error(tmp_path):
    path = tmp_path / "invalid.npz"
    np.savez(
        path,
        schema="coilgen-mesh-v1",
        vertices_m=np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]]),
        faces=np.array([[0, 1, 99]]),
    )
    project = ProjectDocument.from_config(replace(numeric_config(), coil_mesh_file=str(path)))
    issue = next(issue for issue in check_readiness(project).blocking if issue.key == "coil_mesh_file")
    assert issue.node == "geometry" and "indices" in issue.message


def test_physical_and_strict_solvers_block_only_when_actually_unavailable(monkeypatch):
    monkeypatch.setattr(readiness, "find_spec", lambda _name: None)
    physical = ProjectDocument.from_config(preset_config())
    report = check_readiness(physical)
    assert not report.ready and report.blocking[-1].title == "Unavailable solver"
    assert report.blocking[-1].node == "optimization" and "CVXPY" in report.blocking[-1].message
    strict = ProjectDocument.from_config(
        replace(numeric_config(), sf_opt_method="constrained_quadratic", constrained_quadratic_fallback_policy="strict")
    )
    assert not check_readiness(strict).ready
    assert check_readiness(physical, pipeline_stage="prepared").ready


@pytest.mark.parametrize("geometry", ["cylinder", "cone", "cone_y20"])
def test_valid_physical_examples_pass_with_actual_installed_conic_backend(geometry):
    pytest.importorskip("cvxpy")
    if not conic_solver_order():
        pytest.skip("No supported conic backend is installed in this environment.")
    config = preset_config()
    if geometry != "cylinder":
        config.coil_mesh_file = "create conical frustum mesh"
        config.conical_frustum_mesh_parameter_list = (0.3, 0.18, 0.14, 16, 8, 1, 0, 0, 0, 0, 0, 0)
        config.surface_is_cylinder_flag = False
    if geometry == "cone_y20":
        config = replace(config, **SolidHarmonicTarget(2, 0, 15.8533e-6, 0.05).to_config_overrides())
    assert check_readiness(ProjectDocument.from_config(config)).ready


def test_explicit_uninstalled_backend_is_blocked_by_actual_solver_selector(monkeypatch):
    monkeypatch.setattr(readiness, "find_spec", lambda _name: object())
    monkeypatch.setattr(
        "coilgen.optimization_solvers.conic_solver_order",
        lambda requested: conic_solver_order(requested, installed_solvers=()),
    )
    project = ProjectDocument.from_config(replace(preset_config(), constrained_optimization_solver="clarabel"))
    issue = check_readiness(project).blocking[-1]
    assert "CLARABEL" in issue.message and "not installed" in issue.message


def test_explicit_legacy_fallback_is_a_warning_not_silent_method_change(monkeypatch):
    monkeypatch.setattr(readiness, "find_spec", lambda _name: None)
    project = ProjectDocument.from_config(
        replace(
            numeric_config(),
            sf_opt_method="constrained_quadratic",
            constrained_quadratic_fallback_policy="target_error_scan",
        )
    )
    report = check_readiness(project)
    assert report.ready and not report.blocking
    assert any("configured legacy fallback" in issue.message for issue in report.warnings)
    assert project.config.sf_opt_method == "constrained_quadratic"


def test_preoptimized_stream_function_does_not_need_unused_solver(tmp_path, monkeypatch):
    monkeypatch.setattr(readiness, "find_spec", lambda _name: pytest.fail("Unused solver checked"))
    path = tmp_path / "sf.npz"
    np.savez(path, stream_function=np.zeros(24))
    project = ProjectDocument.from_config(
        replace(
            numeric_config(),
            sf_opt_method="constrained_quadratic",
            constrained_quadratic_fallback_policy="strict",
            preoptimized_stream_function_file=str(path),
        )
    )
    assert check_readiness(project).ready


def test_missing_analytic_target_and_validator_warnings_keep_separate_severity():
    project = ProjectDocument.from_config(replace(numeric_config(), field_shape_function="none"))
    issue = check_readiness(project).blocking[0]
    assert issue.node == "target" and issue.title == "Missing target"
    rf = ProjectDocument.from_config(replace(numeric_config(), design_context="mr_rf"))
    report = check_readiness(rf)
    assert report.ready and report.warnings and not report.blocking


def test_readiness_panel_action_emits_issue_and_empty_hides_banner(application):
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QPushButton

    panel = ReadinessPanel()
    issue = ReadinessIssue(
        "A target map is missing.",
        "target",
        "target_field_definition_file",
        "Import the target map.",
        title="Missing input",
    )
    emitted = []
    panel.issue_requested.connect(emitted.append)
    panel.set_issues([issue])
    assert panel.objectName() == "compute_readiness"
    assert not panel.isHidden() and panel.heading.text() == "Cannot compute yet — resolve the item below."
    panel.findChild(QPushButton).click()
    assert emitted == [issue]
    panel.set_issues([issue, replace(issue, key="coil_mesh_file", node="geometry")])
    assert panel.heading.text() == "Cannot compute yet — resolve these 2 items."
    panel.set_issues([replace(issue, severity="warning", title="Modeling note")])
    assert "Compute is available" in panel.heading.text() and not panel.isHidden()
    panel.set_issues([])
    assert panel.isHidden()
    panel.close()
    panel.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_unused_optional_inputs_and_zero_analytic_field_are_not_missing():
    config = CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        field_shape_function="0",
        target_gradient_normalization="none",
        sf_opt_method="tikhonov",
    )
    assert check_readiness(ProjectDocument.from_config(config)).ready


@pytest.mark.parametrize("source", ["none", ""])
def test_support_is_required_even_when_optional_inputs_are_none(source):
    project = ProjectDocument.from_config(replace(numeric_config(), coil_mesh_file=source))
    issue = next(issue for issue in check_readiness(project).blocking if issue.key == "coil_mesh_file")
    assert issue.node == "geometry" and issue.title == "Missing support"
    assert "configured" in issue.message and "generated" in issue.action.lower()


def test_unsupported_generator_is_not_mistaken_for_a_valid_generated_surface():
    project = ProjectDocument.from_config(replace(numeric_config(), coil_mesh_file="create nonexistent mesh"))
    issues = check_readiness(project).blocking
    assert len(issues) == 1 and issues[0].key == "coil_mesh_file"
    assert "Unknown mesh source" in issues[0].message


@pytest.mark.parametrize(
    "expression,diagnostic",
    [
        ("x +", "syntax"),
        ("x + unknown_parameter", "Unknown target symbol"),
        ("sin(not_a_coordinate)", "Unknown target symbol"),
    ],
)
def test_malformed_simple_expression_is_actionable_before_compute(expression, diagnostic):
    project = ProjectDocument.from_config(replace(numeric_config(), field_shape_function=expression))
    issue = next(issue for issue in check_readiness(project).blocking if issue.key == "field_shape_function")
    assert issue.node == "target" and diagnostic in issue.message


@pytest.mark.parametrize(
    "expression", ["1/x", "[i*i for i in x]", "x if True else unknown_dead_branch", "x.mean() + y", "x.^2 + y.*z"]
)
def test_readiness_does_not_probe_domains_or_reject_legacy_binding_expressions(expression, monkeypatch):
    monkeypatch.setattr(
        "coilgen.target.evaluate_field_expression",
        lambda *_a, **_k: pytest.fail("Invented expression probes evaluated"),
    )
    project = ProjectDocument.from_config(replace(numeric_config(), field_shape_function=expression))
    assert check_readiness(project).ready


def test_imported_support_roundtrip_and_point_only_target_mesh_are_valid(tmp_path):
    support = tmp_path / "support.npz"
    support.write_bytes(mesh_asset(Mesh(np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]]), np.array([[0, 1, 2]]))).data)
    target = tmp_path / "roi.npz"
    np.savez(
        target,
        schema="coilgen-mesh-v1",
        vertices_m=np.array([[0, 0, 0], [0, 0, 0.01]]),
        faces=np.empty((0, 3), dtype=int),
    )
    project = ProjectDocument.from_config(
        replace(numeric_config(), coil_mesh_file=str(support), target_mesh_file=str(target))
    )
    assert check_readiness(project).ready


@pytest.mark.parametrize(
    "points,message",
    [
        (((0, 0), (1e-3, 0)), "at least three"),
        (((0, 0), (1e-3, 0), (1e-3, 0), (0, 1e-3)), "zero-length"),
        (((0, 0), (1e-3, 1e-3), (0, 1e-3), (1e-3, 0)), "intersect"),
        (((0, 0), (1e-3, 0), (2e-3, 0)), "area"),
        (((0, 0), (2e-3, 0), (1e-3, 1e-3), (2e-3, 2e-3), (0, 2e-3)), "convex"),
        (((0, 0), (1e-3, 0), (2e-3, 0), (2e-3, 2e-3), (0, 2e-3)), "collinear"),
    ],
)
def test_compute_reports_actual_invalid_profile_in_conductor_step(points, message, monkeypatch):
    monkeypatch.setattr(
        "coilgen.physics.calculate_sensitivity_matrix", lambda *_a, **_k: pytest.fail("Operator assembled")
    )
    project = ProjectDocument.from_config(replace(numeric_config(), cross_sectional_points=points))
    before = project.fingerprint
    issue = next(issue for issue in check_readiness(project).blocking if issue.key == "cross_sectional_points")
    assert issue.node == "material" and issue.title == "Invalid conductor profile"
    assert message in issue.message
    assert "Conductor → Cross-section profile" in issue.action
    assert readiness._readiness_icon(issue) == "conductor-profile"
    assert project.fingerprint == before


def test_nonfinite_profile_import_cannot_become_project_but_has_actionable_readiness(tmp_path):
    config = replace(numeric_config(), cross_sectional_points=((0, 0), (1e-3, 0), (0, np.nan)))
    with pytest.raises(ValueError, match="Nonfinite"):
        ProjectDocument.from_config(config)
    project = SimpleNamespace(config=config, project_root=tmp_path, study={})
    issue = next(issue for issue in check_readiness(project).blocking if issue.key == "cross_sectional_points")
    assert issue.node == "material" and "finite" in issue.message
    assert "Conductor → Cross-section profile" in issue.action


@pytest.mark.parametrize(
    "stage,changes,requires_profile",
    [
        ("prepared", {}, False),
        ("sheet", {}, False),
        ("contours", {}, False),
        ("routed", {}, True),
        ("evaluated", {}, True),
        # Return-path clearance/overpasses consume actual profile geometry,
        # including when the final conductor sweep is disabled.
        ("routed", {"skip_sweep": True}, True),
        ("evaluated", {"skip_sweep": True}, True),
        ("routed", {"skip_sweep": True, "skip_normal_shift": True}, False),
        ("evaluated", {"skip_sweep": True, "skip_normal_shift": True}, False),
        ("routed", {"skip_sweep": True, "skip_postprocessing": True}, False),
        ("evaluated", {"skip_sweep": True, "skip_postprocessing": True}, False),
        ("routed", {"skip_sweep": True, "contour_only_sweep_mode": True}, False),
        ("evaluated", {"skip_sweep": True, "contour_only_sweep_mode": True}, False),
        # Independent-turn sweeps still require a valid convex profile.
        ("routed", {"contour_only_sweep_mode": True}, True),
        ("evaluated", {"contour_only_sweep_mode": True}, True),
        ("routed", {"skip_postprocessing": True}, True),
    ],
)
def test_profile_readiness_follows_geometry_consumer_scope(stage, changes, requires_profile):
    points = ((0, 0), (2e-3, 0), (1e-3, 1e-3), (2e-3, 2e-3), (0, 2e-3))
    project = ProjectDocument.from_config(replace(numeric_config(), cross_sectional_points=points, **changes))
    report = check_readiness(project, pipeline_stage=stage)
    profile_issues = tuple(issue for issue in report.blocking if issue.key == "cross_sectional_points")
    assert bool(profile_issues) == requires_profile
    assert report.ready == (not requires_profile)


@pytest.mark.parametrize("mode", ["centered", "surface_outer", "surface_inner"])
def test_mounting_mode_does_not_skip_profile_validation(mode):
    project = ProjectDocument.from_config(
        replace(numeric_config(), cross_sectional_points=((0, 0), (1e-3, 0)), conductor_mounting_mode=mode)
    )
    assert any(issue.key == "cross_sectional_points" for issue in check_readiness(project).blocking)


def test_profile_readiness_uses_selected_stage_and_does_not_change_legacy_validator(monkeypatch):
    from coilgen.config_validation import validate_config

    config = replace(numeric_config(), cross_sectional_points=((0, 0), (1e-3, 0)))
    assert not validate_config(config).errors
    project = ProjectDocument.from_config(config).update_study({"pipeline_stage": "contours"})
    assert check_readiness(project).ready
    assert not check_readiness(project, pipeline_stage="routed").ready
    monkeypatch.setattr(
        "coilgen.application.conductor_profiles.validate_conductor_profile",
        lambda *_a, **_k: pytest.fail("Unused conductor geometry checked"),
    )
    assert check_readiness(project, pipeline_stage="prepared").ready


@pytest.mark.parametrize("legacy_orientation", [False, True])
def test_offset_clockwise_profile_is_ready_without_recentring(legacy_orientation):
    points = np.array([[1, -1], [2, 1], [4, 1], [5, -1]], dtype=float) * 1e-3
    if legacy_orientation:
        points = points.T
    project = ProjectDocument.from_config(replace(numeric_config(), cross_sectional_points=points))
    before = project.fingerprint
    assert check_readiness(project).ready
    assert project.fingerprint == before
    np.testing.assert_array_equal(project.config.cross_sectional_points, points)


@pytest.mark.parametrize("field", ["conductor_cross_section_width", "conductor_cross_section_height"])
def test_invalid_rectangle_dimensions_keep_single_existing_diagnostic(field):
    project = ProjectDocument.from_config(replace(numeric_config(), **{field: 0.0}))
    report = check_readiness(project)
    assert len(report.blocking) == 1
    assert report.blocking[0].key == field and report.blocking[0].node == "material"
