"""Native controller regressions; no solver or graphics implementation mirrored."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, Signal
from PySide6.QtWidgets import QApplication, QWidget

from coilgen.application import ProjectAssembly, ProjectDocument
from coilgen.application.presets import EXAMPLE_TITLES, preset_config, preset_study
from coilgen.gui.workbench import CoilGenWorkbench


class FakeViewport(QWidget):
    selected = Signal(str)

    def render_config(self, config):
        self.config = config

    def display_run(self, arrays, metadata=None):
        self.arrays = arrays
        self.metadata = metadata


class FakeJobs:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.states = []
        self.cancelled = []

    def list(self):
        return [dict(state) for state in self.states]

    def cancel(self, id_):
        self.cancelled.append(id_)
        self.states[0]["status"] = "cancelled"


@dataclass
class FakeRun:
    directory: Path
    metadata: dict
    arrays: dict


class FakeService:
    def __init__(self):
        self.submissions = []
        self.record = None

    def submit(self, project, jobs, **controls):
        self.submissions.append((project, controls))
        state = {
            "id": "f" * 32,
            "name": project.name,
            "status": "queued",
            "stage": "Waiting for worker",
            "project_revision": project.revision,
            "project_fingerprint": project.fingerprint,
        }
        jobs.states.append(state)
        return dict(state)

    def load_run(self, path):
        return self.record


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["coilgen-gui-test"])


@pytest.fixture
def window(application, tmp_path):
    jobs = FakeJobs(tmp_path / "jobs")
    service = FakeService()
    project = ProjectDocument.from_config(preset_config(), name=EXAMPLE_TITLES["cylinder"]).update_study(preset_study())
    instance = CoilGenWorkbench(project, jobs=jobs, service=service, viewport_factory=FakeViewport)
    instance.timer.stop()
    yield instance
    instance.undo_stack.setClean()
    instance.close()
    instance.undo_stack.clear()
    instance.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_physical_edit_undo_redo_and_project_roundtrip(window, tmp_path):
    before = window.project
    radius = window._draft_controls["radius_mm"]
    radius.setValue(173.5)
    assert window.apply_settings()
    edited = window.project
    assert edited.config.cylinder_mesh_parameter_list[1] == pytest.approx(0.1735)
    assert before.config.cylinder_mesh_parameter_list[1] == pytest.approx(
        preset_config().cylinder_mesh_parameter_list[1]
    )
    assert "prepared" in window.recompute.toPlainText()
    window.undo_stack.undo()
    assert window.project.fingerprint == before.fingerprint
    window.undo_stack.redo()
    assert window.project.fingerprint == edited.fingerprint
    path = window.save_project(tmp_path / "design.coilgen.json")
    window.apply_config({"levels": 11})
    window.open_project(path)
    assert window.project.fingerprint == edited.fingerprint
    assert window.undo_stack.isClean()


def test_invalid_atomic_edit_keeps_model_and_history(window):
    before = window.project
    assert not window.apply_config({"target_region_radius": -1})
    assert window.project is before
    assert window.undo_stack.count() == 0
    assert "rejected" in window.messages.toPlainText()


def test_submission_stage_cancel_and_snapshot_isolation(window):
    window.stage_selector.setCurrentIndex(2)
    state = window.compute()
    submitted, controls = window.service.submissions[0]
    assert controls["pipeline_stage"] == "contours"
    assert controls["validation_resolution"] == preset_study()["validation_resolution"]
    window.apply_config({"levels": 14})
    assert submitted.config.levels == preset_config().levels
    assert state["project_fingerprint"] != window.project.fingerprint
    window.jobs_table.selectRow(0)
    window.cancel_selected_job()
    assert window.jobs.cancelled == [state["id"]]
    assert window.jobs_table.item(0, 3).text() == "cancelled"


def test_run_scope_acceptance_and_stale_results_remain_visible(window, tmp_path):
    record = FakeRun(
        tmp_path / "run",
        {
            "name": "sheet run",
            "project_id": window.project.id,
            "project_fingerprint": window.project.fingerprint,
            "summary": {"scope": "continuous_sheet", "accepted": None, "power_w": 100},
            "constraints": [{"name": "Independent error", "value": None, "limit": 0.1, "passed": None}],
            "warnings": ["Routing and fabrication unverified"],
        },
        {"vertices_m": np.zeros((3, 3)), "faces": np.array([[0, 1, 2]])},
    )
    window.service.record = record
    window.load_run(record.directory)
    assert "Current model" in window.result_state.text()
    assert "Unverified" in window.result_state.text()
    assert window.checks_table.item(0, 5).text() == "Unverified"
    window.apply_config({"levels": 12})
    assert "STALE" in window.result_state.text()
    assert len(window.runs) == 1
    assert window.viewport.arrays is record.arrays


def test_export_preserves_selected_stage_and_native_model(window, tmp_path):
    window.stage_selector.setCurrentIndex(1)
    path = window.export_python(tmp_path / "design.py")
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")
    assert "DesignService.run" in source
    exported = ProjectDocument.load(path.with_suffix(".coilgen.json"))
    assert exported.study["pipeline_stage"] == "sheet"
    assert exported.fingerprint == window.project.fingerprint


def test_biplanar_preset_exposes_physical_spacing(window):
    window.new_project("biplanar")
    window.select_node("geometry")
    assert window.project.config.coil_mesh_file == "create bi-planary mesh"
    window._draft_controls["separation_mm"].setValue(190)
    assert window.apply_settings()
    assert window.project.config.biplanar_mesh_parameter_list[10] == pytest.approx(0.19)


def test_conical_support_editor_geometry_and_export_roundtrip(window, tmp_path):
    from PySide6.QtWidgets import QComboBox
    from coilgen.geometry import load_mesh_from_config

    cylinder = window.project
    window.findChild(QComboBox, "support_type").setCurrentText("Conical frustum")
    assert window.project.config.coil_mesh_file == "create conical frustum mesh"
    assert not window.project.config.surface_is_cylinder_flag
    controls = window._draft_controls
    for key, value in {
        "bottom_radius_mm": 180,
        "top_radius_mm": 110,
        "length_mm": 320,
        "angular_divisions": 12,
        "length_divisions": 6,
        "rotation_angle_deg": 90,
        "center_y_mm": 15,
    }.items():
        controls[key].setValue(value)
    assert window.apply_settings()
    config = window.project.config
    parameters = config.conical_frustum_mesh_parameter_list
    assert parameters[:5] == pytest.approx((0.32, 0.18, 0.11, 12, 6))
    assert parameters[8] == pytest.approx(np.pi / 2)
    assert parameters[10] == pytest.approx(0.015)
    mesh = load_mesh_from_config(config, Path.cwd())
    # The +z axis rotates toward -y; translation is applied in the world frame.
    assert np.min(mesh.vertices[:, 1]) == pytest.approx(-0.145)
    assert np.max(mesh.vertices[:, 1]) == pytest.approx(0.175)
    assert np.all(np.isfinite(mesh.vertices))
    export = window.export_python(tmp_path / "cone.py")
    restored = ProjectDocument.load(export.with_suffix(".coilgen.json"))
    assert restored.config.conical_frustum_mesh_parameter_list == config.conical_frustum_mesh_parameter_list
    window.undo_stack.undo()
    window.undo_stack.undo()
    assert window.project.fingerprint == cylinder.fingerprint


def test_gui_cone_harmonic_configuration_reaches_actual_sheet_solver(window):
    from PySide6.QtWidgets import QComboBox
    from coilgen.pipeline import CoilGenPipeline

    pytest.importorskip("cvxpy")
    window.findChild(QComboBox, "support_type").setCurrentText("Conical frustum")
    for key, value in {
        "bottom_radius_mm": 180,
        "top_radius_mm": 140,
        "length_mm": 300,
        "angular_divisions": 16,
        "length_divisions": 8,
    }.items():
        window._draft_controls[key].setValue(value)
    assert window.apply_settings()
    window.select_node("target")
    window.findChild(QComboBox, "target_mode").setCurrentText("Spherical harmonic")
    window._draft_controls["harmonic_reference_radius_mm"].setValue(50)
    window._draft_controls["target_region_radius"].setValue(50)
    window._draft_controls["target_region_resolution"].setValue(12)
    assert window.apply_settings()
    window.apply_config({"progress_flag": False, "target_error_tolerance_percent": 12})
    config = window.project.config
    state = CoilGenPipeline(config).run_stage("sheet").state
    assert state.regularization_diagnostics["stream_function_scheme"] == "research_convex"
    assert np.all(np.isfinite(state.sf_b_field))
    # The configured scalar Bz budget does not constrain transverse Maxwell fields.
    relative = np.linalg.norm(state.sf_b_field[:, 2] - state.target_field.b[:, 2]) / np.linalg.norm(
        state.target_field.b[:, 2]
    )
    assert relative <= 0.12 + 1e-6


def test_harmonic_target_is_analytic_absolute_and_roi_does_not_rescale_it(window, tmp_path):
    from PySide6.QtWidgets import QComboBox
    from coilgen.application.targets import recognize_solid_harmonic_target
    from coilgen.target import evaluate_field_expression

    window.select_node("target")
    previous = window.project.config
    window.findChild(QComboBox, "target_mode").setCurrentText("Spherical harmonic")
    controls = window._draft_controls
    assert "target_gradient_strength" not in controls
    controls["harmonic_degree"].setValue(2)
    controls["harmonic_order"].setValue(0)
    controls["harmonic_coefficient_uT"].setValue(120)
    controls["harmonic_reference_radius_mm"].setValue(45)
    controls["harmonic_center_x_mm"].setValue(3)
    assert window.apply_settings()
    config = window.project.config
    recipe = recognize_solid_harmonic_target(config.field_shape_function)
    assert recipe.coefficient_T == pytest.approx(120e-6)
    assert recipe.reference_radius_m == pytest.approx(0.045)
    assert recipe.center_m == pytest.approx((0.003, 0, 0))
    assert config.target_gradient_normalization == "off"
    assert config.target_field_units == "T"
    assert config.design_context == "mr_shim"
    values = evaluate_field_expression(config.field_shape_function, np.array([[0.003, 0, 0.045], [0.003, 0, 0]]))
    assert values == pytest.approx([120e-6 * np.sqrt(5 / (4 * np.pi)), 0])
    expression = config.field_shape_function
    window._draft_controls["target_region_radius"].setValue(55)
    assert window.apply_settings()
    assert window.project.config.field_shape_function == expression
    path = window.export_python(tmp_path / "harmonic.py")
    restored = ProjectDocument.load(path.with_suffix(".coilgen.json"))
    assert restored.config.field_shape_function == expression
    window.findChild(QComboBox, "target_mode").setCurrentText("Analytic expression")
    config = window.project.config
    for key in ("field_shape_function", "target_gradient_normalization", "target_gradient_strength", "design_context"):
        assert getattr(config, key) == getattr(previous, key)
    assert config.target_region_radius == pytest.approx(0.055)


def test_harmonic_editor_order_bounds_and_native_expression_preservation(window):
    from PySide6.QtWidgets import QComboBox
    from coilgen.application.targets import recognize_solid_harmonic_target, solid_harmonic_target

    original = solid_harmonic_target(3, -3, 1.234567891234567e-5, 0.0456789123456789, (0.00123456789123456, 0, 0))
    window.apply_config(original.to_config_overrides())
    window.select_node("target")
    window._draft_controls["harmonic_degree"].setValue(1)
    assert window._draft_controls["harmonic_order"].minimum() == -1
    assert window._draft_controls["harmonic_order"].maximum() == 1
    assert window.apply_settings()
    recipe = recognize_solid_harmonic_target(window.project.config.field_shape_function)
    assert recipe.order == -1
    assert recipe.reference_radius_m == original.reference_radius_m
    assert recipe.coefficient_T == original.coefficient_T
    assert recipe.center_m == original.center_m
    window.undo_stack.undo()
    window.findChild(QComboBox, "target_mode").setCurrentText("Analytic expression")
    assert "field_shape_function" in window._draft_controls
    expression = window.project.config.field_shape_function
    window._draft_controls["target_region_radius"].setValue(60)
    assert window.apply_settings()
    assert window.project.config.field_shape_function == expression


def test_solver_acceptance_slack_edits_actual_physics_policy_and_native_policy(window):
    from PySide6.QtWidgets import QLabel

    window.apply_config(
        {
            "research_physics": {"objective": "power", "acceptance_absolute_by_unit": {"T": 2e-10, "N": 7e-5}},
            "target_error_constraint_acceptance_absolute_T": 1e-7,
        }
    )
    window.select_node("optimization")
    assert window._draft_controls["acceptance_slack"].value() == pytest.approx(0.2)
    window._draft_controls["acceptance_slack"].setValue(2.5)
    assert window.apply_settings()
    config = window.project.config
    assert config.research_physics["acceptance_absolute_by_unit"]["T"] == pytest.approx(2.5e-9)
    assert config.research_physics["acceptance_absolute_by_unit"]["N"] == 7e-5
    assert config.target_error_constraint_acceptance_absolute_T == 1e-7
    window._draft_controls["sf_opt_method"].setCurrentText("tikhonov")
    assert window._draft_controls["acceptance_slack"].value() == pytest.approx(100)
    window._draft_controls["acceptance_slack"].setValue(60)
    assert window.apply_settings()
    assert window.project.config.research_physics is None
    assert window.project.config.target_error_constraint_acceptance_absolute_T == pytest.approx(60e-9)
    assert "separate existing policies" in window.findChild(QLabel, "acceptance_policy_info").text()


def test_wire_field_evidence_preserves_current_and_acceptance_scope(window, tmp_path):
    record = FakeRun(
        tmp_path / "run",
        {
            "summary": {"scope": "routed", "accepted": None},
            "wire_fields": {
                "contour": {
                    "available": True,
                    "current_A": 4.2,
                    "current_refitted": False,
                    "rms_error_T": 2e-6,
                    "relative_rms_error": 0.025,
                    "peak_error_T": 8e-6,
                    "sample_count": 17,
                    "current_source": "contour_step_A",
                    "geometry_source": "unconnected loops",
                }
            },
        },
        {},
    )
    window.service.record = record
    window.load_run(record.directory)
    assert window.viewport.metadata == record.metadata
    assert window.wire_fields_table.item(0, 1).text() == "4.2"
    assert window.wire_fields_table.item(0, 3).text() == "2"
    assert window.wire_fields_table.item(0, 4).text() == "2.5"
    assert "does not claim" in window.wire_fields_table.item(0, 0).toolTip()
    assert "Unverified" in window.result_state.text()


def test_fabrication_failures_and_electrical_scope_are_visible_in_results(window, tmp_path):
    from PySide6.QtWidgets import QPlainTextEdit, QTableWidget

    metadata = {
        "summary": {
            "accepted": False,
            "sheet_accepted": True,
            "scope": "fabrication_review",
            "power_w": 0.03,
            "validated_relative_field_error": 0.04,
        },
        "fabrication": {
            "ready": False,
            "failure_reasons": ["body001:route_clearance_violation;support_surface_penetration"],
            "electrical_metrics": {
                "bodies": [
                    {
                        "body_id": "body001",
                        "current_A": 1.5,
                        "resistance_ohm": 0.7,
                        "inductance_H": 1e-4,
                        "joule_power_W": 1.575,
                    }
                ]
            },
        },
        "winding_steps": [
            {
                "body_id": "body001",
                "circuit_complete": True,
                "drive_feed_designed": False,
                "terminal_assumption": "Closed mathematical turn; feed break and drive connection are not designed.",
            }
        ],
    }
    window.service.record = FakeRun(tmp_path / "run", metadata, {})
    window.load_run(tmp_path / "run")
    window.select_node("results")
    text = window.findChild(QPlainTextEdit, "fabrication_failure_reasons").toPlainText()
    assert "body001:" in text
    assert "route_clearance_violation" in text
    assert "support_surface_penetration" in text
    table = window.findChild(QTableWidget, "selected_run_metrics")
    assert table.item(0, 0).text() == "Continuous sheet"
    assert table.item(2, 0).text() == "Review: body001"
    assert table.item(4, 1).text() == "Inductance [µH]"
    assert table.item(4, 2).text() == "100"
    assert "feed break" in window.findChild(QWidget, "drive_feed_scope").text()


def test_default_full_result_with_null_fabrication_opens_review(window, tmp_path):
    from PySide6.QtWidgets import QLabel, QPlainTextEdit, QTableWidget

    metadata = {
        "pipeline_stage": "evaluated",
        "summary": {
            "accepted": True,
            "sheet_accepted": True,
            "scope": "continuous_sheet",
            "construction_released": False,
            "power_w": 337.9808755441829,
            "validated_relative_field_error": 0.03510131108584537,
            "current_a": None,
            "resistance_ohm": None,
            "inductance_h": None,
        },
        "fabrication": None,
        "winding_steps": [],
        "wire_fields": {"contour": {"current_A": 276.49091510116403, "current_refitted": False}},
        "constraints": [],
    }
    window.service.record = FakeRun(tmp_path / "completed_full_run", metadata, {})
    window.load_run(window.service.record.directory)

    window.navigate_workflow("review")

    panel = window.settings.widget()
    table = panel.findChild(QTableWidget, "selected_run_metrics")
    assert table is not None
    assert table.rowCount() == 3
    assert table.item(0, 0).text() == "Continuous sheet"
    assert table.item(1, 0).text() == "Independent sheet grid"
    assert table.item(2, 0).text() == "Original contours"
    texts = [label.text() for label in panel.findChildren(QLabel)]
    assert any("Fabrication review: Not run" in text for text in texts)
    assert any("Continuous-sheet acceptance: Passed" in text and "continuous_sheet" in text for text in texts)
    diagnostics = panel.findChild(QPlainTextEdit, "fabrication_failure_reasons")
    assert "Missing or unverified checks remain unverified." in diagnostics.toPlainText()
    assert window.tree.currentItem() is window.tree_items["results"]
    assert window.page_label.text() == "Section: Results and provenance"


def test_navigation_commits_draft_and_noop_compute_preserves_exact_inputs(window):
    before = window.project.fingerprint
    window._draft_controls["radius_mm"].setValue(171.25)
    window.select_node("target")
    assert window.project.config.cylinder_mesh_parameter_list[1] == pytest.approx(0.17125)
    assert window.project.fingerprint != before
    assert window._draft_controls["target_field_component"].count() == 1
    fingerprint = window.project.fingerprint
    window.compute()
    assert window.project.fingerprint == fingerprint


def test_study_is_undoable_and_saved_with_project(window, tmp_path):
    window.select_node("validation")
    window._draft_controls["validation_resolution"].setValue(17)
    window._draft_controls["design_margin"].setValue(0.8)
    assert window.apply_settings()
    assert window.project.study["validation_resolution"] == 17
    window.undo_stack.undo()
    assert window.project.study["validation_resolution"] == preset_study()["validation_resolution"]
    window.undo_stack.redo()
    path = window.save_project(tmp_path / "study.json")
    window.open_project(path)
    assert window.validation_resolution == 17
    assert window.design_margin == pytest.approx(0.8)


def test_independent_channels_keep_ownership_undo_and_portable_collection(window, tmp_path):
    original_id = window._active_channel_id
    original = window.project
    window.add_channel("Second coil")
    second_id = window._active_channel_id
    assert second_id != original_id
    window.apply_config({"levels": 15})
    assert window.assembly.channel(original_id).project.config.levels == original.config.levels
    assert window.assembly.channel(second_id).project.config.levels == 15
    window.channel_selector.setCurrentIndex(window.channel_selector.findData(original_id))
    window.undo_stack.undo()
    assert window._active_channel_id == second_id
    assert window.project.config.levels == original.config.levels
    assert window.assembly.channel(original_id).project.fingerprint == original.fingerprint
    window.undo_stack.redo()
    window.channel_enabled.setChecked(False)
    path = window.save_project(tmp_path / "channels.json")
    collection = ProjectAssembly.load(path)
    assert len(collection.channels) == 2
    assert not collection.channel(second_id).enabled
    window.open_project(path)
    assert len(window.assembly.channels) == 2
    script = window.export_assembly_python(tmp_path / "channels.py")
    compile(script.read_text(encoding="utf-8"), str(script), "exec")
    assert "run_assembly" in script.read_text(encoding="utf-8")


def test_imports_capture_owned_bytes_and_reject_malformed_field_atomically(window, tmp_path):
    mesh = tmp_path / "support.npz"
    from coilgen.application.assets import mesh_asset
    from coilgen.models import Mesh

    mesh.write_bytes(
        mesh_asset(
            Mesh(vertices=np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]]), faces=np.array([[0, 1, 2]]))
        ).data
    )
    window.import_mesh(mesh)
    mesh.unlink()
    assert Path(window.project.config.coil_mesh_file).is_file()
    field = tmp_path / "target.npz"
    np.savez(
        field, coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.01, 0.0]]), b=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0002]])
    )
    window.import_target(field, field_units="T")
    assert window.project.config.target_gradient_normalization == "none"
    from coilgen.target import load_target_field_definition

    target = load_target_field_definition(
        window.project.config.target_field_definition_file, None, window.project.config
    )
    assert target.b[1, 2] == pytest.approx(0.0002)
    before = window.project
    invalid = tmp_path / "bad.npz"
    np.savez(invalid, coords=[[0.0, 0.0, 0.0]], b=[[float("nan"), 0.0, 0.0]])
    with pytest.raises(ValueError):
        window.import_target(invalid)
    assert window.project is before
    field.unlink()
    saved = window.save_project(tmp_path / "owned.json")
    assert Path(ProjectDocument.load(saved).config.target_field_definition_file).is_file()


def test_poll_preserves_selected_job_and_service_revision(window):
    state = window.compute()
    window.jobs.states[0].pop("project_revision")
    window.jobs_table.selectRow(0)
    window.poll_jobs()
    assert window.jobs_table.currentRow() == 0
    assert window.jobs_table.item(0, 5).text() == str(window.project.revision)
    assert window.jobs_table.item(0, 0).data(256) == state["id"]


def test_resume_and_export_use_verified_selected_run(window, tmp_path):
    from dataclasses import asdict, replace
    from coilgen.pipeline import CoilGenPipeline
    from coilgen.persistence import file_digest
    from test_pipeline_stages import config

    window.apply_config(asdict(config()))
    directory = tmp_path / "run"
    directory.mkdir()
    checkpoint = directory / "pipeline_checkpoint.cgstage"
    settings = replace(
        window.project.config,
        parallel_worker_count=1,
        constrained_optimization_threads=1,
        plot_flag=False,
        save_stl_flag=False,
    )
    CoilGenPipeline(settings).run_stage("prepared").save(checkpoint)
    field = directory / "field.csv"
    field.write_text("x_m,By_T\n0,0\n", encoding="utf-8")
    record = FakeRun(
        directory,
        {
            "project_id": window.project.id,
            "artifacts": [
                {"name": checkpoint.name, "kind": "checkpoint", "sha256": file_digest(checkpoint)},
                {"name": field.name, "kind": "csv", "sha256": file_digest(field)},
            ],
            "summary": {},
        },
        {},
    )
    window.service.record = record
    window.load_run(directory)
    window.resume_selected_run()
    assert window.service.submissions[-1][1]["resume_checkpoint"] == str(checkpoint)
    destination = tmp_path / "export"
    window.export_run_artifacts(destination)
    assert (destination / "field.csv").read_bytes() == field.read_bytes()
    with pytest.raises(FileExistsError):
        window.export_run_artifacts(destination)


class RenderlessPlotter(QWidget):
    """Exercise VTK datasets and Qt controls without an OpenGL context."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.interactor = self
        self.camera_position = [(1, 1, 1), (0, 0, 0), (0, 0, 1)]
        self.plane_widgets = []
        self._plane_clipped_meshes = []
        self.meshes = []
        self.picked_point = None
        self.closed = False

    def __getattr__(self, name):
        if name in {
            "set_background",
            "enable_parallel_projection",
            "disable_parallel_projection",
            "add_axes",
            "show_grid",
            "render",
            "reset_camera",
            "view_isometric",
            "view_xy",
            "view_xz",
            "view_yz",
        }:
            return lambda *args, **kwargs: None
        raise AttributeError(name)

    def enable_mesh_picking(self, **kwargs):
        self.pick = kwargs["callback"]

    def add_mesh(self, mesh, **kwargs):
        actor = object()
        self.meshes.append((mesh, kwargs, actor))
        return actor

    def add_mesh_clip_plane(self, mesh, **kwargs):
        self.plane_widgets.append(object())
        self.plane_clipped_meshes.append(mesh)

    @property
    def plane_clipped_meshes(self):
        return self._plane_clipped_meshes

    def clear_plane_widgets(self):
        self.plane_widgets.clear()

    def clear(self):
        self.meshes.clear()

    def close(self):
        self.closed = True

    def render(self):
        pass


@pytest.fixture
def scientific_viewport(application, monkeypatch):
    pytest.importorskip("pyvista")
    pyvistaqt = pytest.importorskip("pyvistaqt")
    monkeypatch.setattr(pyvistaqt, "QtInteractor", RenderlessPlotter)
    from coilgen.gui.viewport import ScientificViewport

    viewport = ScientificViewport(synchronous_preview=True)
    yield viewport
    viewport.close()
    viewport.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_scientific_viewport_units_sampling_clipping_and_picking(scientific_viewport):
    viewport = scientific_viewport
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.01, 0.0], [0.01, 0.0, 0.0]])
    fields = np.array([[0.0, 0.0, 1e-6], [0.0, 0.0, 2e-6], [0.0, 0.0, 3e-6]])
    arrays = {"vertices_m": points, "faces": np.array([[0, 1, 2]]), "target_points_m": points, "target_field_T": fields}
    viewport.display_run(arrays)
    viewport.units.setCurrentText("µT")
    assert viewport._dataset_mesh.active_scalars.tolist() == pytest.approx([1, 2, 3])
    viewport.slice_axis.setCurrentText("x")
    viewport.slice_position.setValue(0)
    assert viewport._dataset_mesh.n_points == 2
    viewport.clip.setChecked(True)
    viewport.component.setCurrentText("By")
    viewport.component.setCurrentText("Bz")
    assert len(viewport.plotter.plane_widgets) == 1
    assert len(viewport.plotter.plane_clipped_meshes) == 1
    viewport.clip.setChecked(False)
    selected = []
    viewport.selected.connect(selected.append)
    actor = viewport.plotter.meshes[-1][2]
    viewport.plotter.picked_point = points[0]
    viewport._picked_actor(actor)
    assert selected == ["target"]
    assert "Bz [µT]" in viewport.probe.text()
    np.testing.assert_array_equal(arrays["target_field_T"], fields)
    viewport.close()
    assert viewport.plotter.closed


def test_scientific_preview_and_bad_field_are_visible_errors(scientific_viewport):
    viewport = scientific_viewport
    from coilgen.application.presets import preset_config

    viewport.render_config(preset_config())
    assert "target_points_m" in viewport._preview_arrays
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("preview_target"))
    assert "current model target" in viewport.caption.text()
    viewport.display_run(
        {"target_points_m": np.array([[0.0, 0.0, 0.0]]), "target_field_T": np.array([[0.0, 0.0, float("nan")]])}
    )
    assert viewport.dataset.currentData() == "preview_target"
    assert viewport._dataset_mesh is not None
    target = viewport.dataset.model().item(viewport.dataset.findData("target"))
    assert not target.isEnabled()
    assert "finite real numbers" in target.toolTip()
    assert "Saved run has no available plot data" in viewport.caption.text()


def test_background_previews_keep_qt_responsive_and_latest_revision_wins(application, monkeypatch):
    from threading import Event
    import time
    from PySide6.QtCore import QTimer
    from PySide6.QtTest import QTest

    pytest.importorskip("pyvista")
    pyvistaqt = pytest.importorskip("pyvistaqt")
    monkeypatch.setattr(pyvistaqt, "QtInteractor", RenderlessPlotter)
    from coilgen.gui import viewport as viewport_module
    from coilgen.application.presets import preset_config

    original = viewport_module._preview_data
    started, release = Event(), Event()
    calls = []

    def delayed(config, root):
        calls.append(config.cylinder_mesh_parameter_list[1])
        if len(calls) == 1:
            started.set()
            assert release.wait(5)
        return original(config, root)

    monkeypatch.setattr(viewport_module, "_preview_data", delayed)
    viewport = viewport_module.ScientificViewport()
    try:
        viewport.render_config(preset_config())
        assert started.wait(5)
        second = preset_config()
        second.cylinder_mesh_parameter_list = (0.3, 0.18, 24, 12, 1, 0, 0, 0)
        viewport.render_config(second)
        third = preset_config()
        third.cylinder_mesh_parameter_list = (0.3, 0.19, 24, 12, 1, 0, 0, 0)
        viewport.render_config(third)
        responsive = []
        QTimer.singleShot(0, lambda: responsive.append(True))
        application.processEvents()
        assert responsive == [True]
        release.set()
        deadline = time.monotonic() + 5
        while viewport._preview_worker is not None and time.monotonic() < deadline:
            QTest.qWait(10)
        assert calls == [preset_config().cylinder_mesh_parameter_list[1], 0.19]
        assert np.max(np.linalg.norm(viewport._support.points[:, :2], axis=1)) == pytest.approx(0.19)
        assert "Preparing" not in viewport.caption.text()
    finally:
        release.set()
        viewport.close()
        viewport.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_scientific_controls_are_concise_and_do_not_rewrite_untouched_values(window):
    from PySide6.QtGui import QValidator

    radius = window._draft_controls["radius_mm"]
    assert radius.text() == f"{preset_config().cylinder_mesh_parameter_list[1] * 1000:g} mm"
    assert radius.valueFromText("1.735e2 mm") == pytest.approx(173.5)
    assert radius.validate("1.735e2 mm", 5)[0] == QValidator.State.Acceptable
    window.apply_config(
        {"specific_conductivity_conductor": 1.8123456789123456e-8, "conductor_thickness": 0.005123456789123456}
    )
    window.select_node("material")
    original = window.project.config
    window._draft_controls["insulation"].setValue(0.2)
    window.apply_settings()
    assert window.project.config.specific_conductivity_conductor == original.specific_conductivity_conductor
    assert window.project.config.conductor_thickness == original.conductor_thickness


def test_settings_forms_fit_default_and_narrow_docks_without_horizontal_scrolling(window, application):
    from PySide6.QtWidgets import QComboBox, QDockWidget
    from PySide6.QtCore import Qt

    window.show()
    dock = window.findChild(QDockWidget, "settings_dock")

    def assert_fits():
        for width in (390, 285):
            window.resizeDocks([dock], [width], Qt.Orientation.Horizontal)
            application.processEvents()
            assert window.settings.horizontalScrollBar().maximum() == 0
            for combo in window.settings.findChildren(QComboBox):
                if combo.isVisible():
                    assert combo.geometry().right() < window.settings.widget().width()

    for node in ("geometry", "target", "optimization"):
        window.select_node(node)
        assert_fits()
    window.select_node("geometry")
    window.findChild(QComboBox, "support_type").setCurrentText("Conical frustum")
    assert_fits()
    window.select_node("target")
    window.findChild(QComboBox, "target_mode").setCurrentText("Spherical harmonic")
    assert_fits()


def test_empty_collection_is_rejected_before_window_state_changes(window, tmp_path):
    project, path, count = window.project, window.project_path, window.undo_stack.count()
    empty = ProjectAssembly.new("Empty draft").save(tmp_path / "empty.json")
    with pytest.raises(ValueError, match="no channels"):
        window.open_project(empty)
    assert window.project is project
    assert window.project_path == path
    assert window.undo_stack.count() == count


@pytest.mark.parametrize("mode", ["magnetic_energy", "tikhonov"])
def test_selected_optimization_mode_reaches_the_actual_solver(window, mode):
    from coilgen.pipeline import CoilGenPipeline

    if mode == "magnetic_energy":
        pytest.importorskip("cvxpy")
    window.apply_config({"cylinder_mesh_parameter_list": (0.3, 0.16, 12, 6, 1, 0, 0, 0), "progress_flag": False})
    window.select_node("optimization")
    if mode == "magnetic_energy":
        window._draft_controls["objective"].setCurrentText(mode)
    else:
        window._draft_controls["sf_opt_method"].setCurrentText(mode)
        assert not window._draft_controls["objective"].isEnabled()
        assert window._draft_controls["tikonov_reg_factor"].isEnabled()
    assert window.apply_settings()
    config = window.project.config
    if mode == "tikhonov":
        assert config.research_physics is None
    else:
        assert config.research_physics["objective"] == "magnetic_energy"
    state = CoilGenPipeline(config).run_stage("sheet").state
    diagnostics = state.regularization_diagnostics
    assert diagnostics["stream_function_scheme"] == ("tikhonov" if mode == "tikhonov" else "research_convex")
    if mode == "magnetic_energy":
        assert diagnostics["research_objective"] == mode
    assert np.all(np.isfinite(state.sf_b_field))


def test_custom_physics_is_preserved_and_incompatible_tikhonov_switch_is_atomic(window):
    from PySide6.QtWidgets import QLabel

    physics = {
        "objective": "minimax",
        "force_limit_N": 0.2,
        "torque_limit_Nm": 0.3,
        "b0_T": [0.0, 0.0, 3.0],
        "torque_origin_m": [0.01, 0.0, 0.0],
    }
    window.apply_config({"research_physics": physics, "sf_opt_method": "tikhonov"})
    window.select_node("optimization")
    # The displayed formulation follows the backend's actual dispatch.
    assert window._draft_controls["sf_opt_method"].currentText() == "constrained_quadratic"
    window._draft_controls["target_error_tolerance_percent"].setValue(6)
    assert window.apply_settings()
    assert window.project.config.research_physics == physics
    before, history = window.project, window.undo_stack.count()
    window._draft_controls["sf_opt_method"].setCurrentText("tikhonov")
    assert "must be explicitly removed" in window.findChild(QLabel, "optimization_mode_info").text()
    assert not window.apply_settings()
    assert window.project is before
    assert window.undo_stack.count() == history
    assert "force_limit_N" in window.messages.toPlainText()
    # Removing the incompatible specification explicitly permits native mode.
    window.apply_config({"research_physics": None, "sf_opt_method": "constrained_quadratic"})
    window._draft_controls["sf_opt_method"].setCurrentText("tikhonov")
    assert window.apply_settings()
    assert window.project.config.research_physics is None
    assert window.project.config.sf_opt_method == "tikhonov"


def test_switching_basic_physical_mode_to_tikhonov_can_be_undone(window):
    window.select_node("optimization")
    original = window.project
    window._draft_controls["sf_opt_method"].setCurrentText("tikhonov")
    assert window.apply_settings()
    assert window.project.config.research_physics is None
    window.undo_stack.undo()
    assert window.project.fingerprint == original.fingerprint
    assert window.project.config.research_physics == original.config.research_physics


def test_imported_native_quadratic_objective_is_displayed_and_preserved(window):
    from PySide6.QtWidgets import QLabel

    window.apply_config(
        {
            "research_physics": None,
            "sf_opt_method": "constrained_quadratic",
            "constrained_quadratic_objective_mode": "balanced",
            "stream_function_cost_function": "field_error_plus_resistance",
        }
    )
    window.select_node("optimization")
    assert window._draft_controls["objective"].currentText() == "Native objective (see below)"
    message = window.findChild(QLabel, "optimization_mode_info").text()
    assert "mode='balanced'" in message
    assert "cost function='field_error_plus_resistance'" in message
    assert "no research physical objective" in message
    window._draft_controls["target_error_tolerance_percent"].setValue(6)
    assert window.apply_settings()
    assert window.project.config.research_physics is None
    assert window.project.config.sf_opt_method == "constrained_quadratic"
    assert window.project.config.constrained_quadratic_objective_mode == "balanced"
    assert window.project.config.stream_function_cost_function == "field_error_plus_resistance"
    window._draft_controls["objective"].setCurrentText("minimax")
    assert window.apply_settings()
    assert window.project.config.research_physics == {"objective": "minimax"}


@pytest.mark.parametrize("explicit_coordinates", [False, True])
def test_fabrication_review_field_uses_its_actual_sampling_coordinates(scientific_viewport, explicit_coordinates):
    viewport = scientific_viewport
    target = np.array([[0.0, 0.0, 0.0], [0.0, 0.01, 0.0], [0.01, 0.0, 0.0]])
    validation = np.array([[0.0, 0.0, 0.0], [0.0, 0.015, 0.0], [0.015, 0.0, 0.0], [0.0, 0.0, 0.015]])
    field = np.array([[0.0, 0.0, 1e-6], [0.0, 0.0, 2e-6], [0.0, 0.0, 3e-6]])
    arrays = {"target_points_m": target, "validation_points_m": validation, "fabricated_field_T": field}
    expected = target.copy()
    if explicit_coordinates:
        expected = target + [0.02, 0.0, 0.0]
        arrays["fabricated_points_m"] = expected
    viewport.display_run(arrays)
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("fabricated"))
    assert "unavailable" not in viewport.caption.text()
    np.testing.assert_array_equal(viewport._dataset_mesh.points, expected)
    np.testing.assert_array_equal(viewport._dataset_mesh.active_scalars, field[:, 2])


def test_pipeline_and_review_conductor_datasets_are_distinct(scientific_viewport):
    viewport = scientific_viewport
    pipeline = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
    reviewed = pipeline + [0.0, 0.0, 0.005]
    electrical = np.vstack((reviewed, [[0.02, 0.01, 0.005], [0.0, 0.01, 0.005]]))
    viewport.display_run(
        {
            "fabrication_0_points_m": pipeline,
            "review_conductor_0_points_m": reviewed,
            "review_electrical_0_points_m": electrical,
        }
    )
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("fabrication"))
    assert "pipeline conductor" in viewport.caption.text()
    assert len(viewport.plotter.meshes) == 1
    np.testing.assert_array_equal(viewport.plotter.meshes[0][0].points, pipeline)
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("review_conductor"))
    assert "fabrication-review conductor" in viewport.caption.text()
    assert len(viewport.plotter.meshes) == 1
    np.testing.assert_array_equal(viewport.plotter.meshes[0][0].points, reviewed)
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("review_electrical"))
    assert "review electrical field paths" in viewport.caption.text()
    assert len(viewport.plotter.meshes) == 1
    np.testing.assert_array_equal(viewport.plotter.meshes[0][0].points, electrical)
