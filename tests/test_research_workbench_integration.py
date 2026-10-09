"""Exercise project, worker evidence and Workbench research boundaries together."""

from dataclasses import replace

import pytest
import numpy as np

from coilgen.application import ProjectDocument, DesignService
from coilgen.application.comparison import compare_runs
from coilgen.application.gradient_studies import RigidTransform
from coilgen.application.research_requirements import (
    DesiredDataset,
    RegionalRequirement,
    ResearchRequirements,
    PhysicalScenario,
    nominal_requirements,
)
from coilgen.application.spatial_scene import ScenePart, SpatialScene
from coilgen.config import CoilGenConfig
from coilgen.target_regions import TargetRegion


def inputs():
    config = CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        cylinder_mesh_parameter_list=(0.3, 0.12, 8, 4, 1, 0, 0, 0),
        field_shape_function="none",
        research_physics={"objective": "power"},
        symmetry_reduction="off",
        gauss_order=3,
        progress_flag=False,
        levels=3,
        min_point_loop_number=3,
        skip_postprocessing=True,
        skip_sweep=True,
        skip_inductance_calculation=True,
        constrained_optimization_tolerance=1e-10,
    )
    region = TargetRegion("box", size_m=(0.02, 0.02, 0.02))
    design = DesiredDataset.analytic(region, ("0", "0", "1e-6"), resolution=2)
    validation = DesiredDataset.analytic(region, ("0", "0", "1e-6"), resolution=2, independent=True)
    requirements = ResearchRequirements(
        (RegionalRequirement("ROI", design, "Bz", peak_limit_T=3e-7, validation=validation),)
    )
    return config, requirements


def test_named_requirements_preflight_snapshot_and_owned_validation(tmp_path):
    config, spec = inputs()
    project = ProjectDocument.from_config(config).update_study({"requirements": spec.to_dict()})
    assert DesignService.preflight(project).ready
    assert project.config.field_shape_function == "none"  # Legacy definition stays inert.
    assert project.model_fingerprint != ProjectDocument.from_config(config).model_fingerprint
    saved = ProjectDocument.load(project.save(tmp_path / "portable.json"))
    assert saved.fingerprint == project.fingerprint
    DesignService.run(saved, tmp_path / "run")
    record = DesignService.load_run(tmp_path / "run")
    metadata = record.metadata
    report = metadata["requirements_review"]
    assert metadata["validation"]["available"]
    assert metadata["validation"]["scope"] == "named_regional_independent_sheet"
    assert isinstance(metadata["validation"]["passed"], bool)
    assert report["independent_sampling_complete"]
    assert metadata["summary"]["accepted"] == all(row["passed"] is True for row in metadata["constraints"])
    assert not report["external_feeds_included"]
    assert all(row["passed"] is None for row in report["rows"] if row["scope"] == "saved_routed_centroids")
    assert "validation_points_m" not in record.array_names  # A legacy unrelated grid is never substituted.
    independent = next(
        row for row in report["rows"] if row["independent_sampling"] and row["scope"] == "continuous_sheet"
    )
    np.testing.assert_array_equal(record.array(independent["array_prefix"] + "_target_T"), validation_values(spec))
    assert not any(
        row["name"].startswith("ROI") and row["scope"] == "saved_routed_centroids" for row in metadata["constraints"]
    )
    regional_checks = [row for row in metadata["constraints"] if row["name"].startswith("ROI")]
    assert regional_checks and all(row["name"].endswith("peak") for row in regional_checks)
    assert all(row["limit"] == 3e-7 and row["value"] is not None for row in regional_checks)


def validation_values(spec):
    return spec.regions[0].validation.world_values()[1]


def test_nominal_robust_pair_compares_identical_physical_evaluation_contract(tmp_path):
    config, spec = inputs()
    spec = replace(
        spec,
        scenarios=(
            PhysicalScenario("Support +1mm", "design", support_transform=RigidTransform(translation_m=(0, 0, 0.001))),
        ),
    )
    projects = [
        ProjectDocument.from_config(config).update_study({"requirements": value.to_dict()})
        for value in (nominal_requirements(spec), spec)
    ]
    for name, project in zip(("nominal", "robust"), projects, strict=True):
        DesignService.run(project, tmp_path / name)
    records = [DesignService.load_run(tmp_path / name) for name in ("nominal", "robust")]
    comparison = compare_runs(*records)
    rows = [row for row in comparison.metrics if row.name.startswith("Region ROI") and "continuous_sheet" in row.name]
    assert len(rows) == 8  # Two poses x design/independent x RMS/peak.
    assert all(row.comparable for row in rows)
    assert any(row.field.startswith("study.requirements.scenarios") for row in comparison.parameter_deltas)


def test_named_editor_navigation_and_available_review_actions(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import Signal
    from PySide6.QtWidgets import QApplication, QWidget
    from coilgen.gui.workbench import CoilGenWorkbench

    class Viewport(QWidget):
        selected = Signal(str)

        def render_config(self, config):
            self.config = config

    application = QApplication.instance() or QApplication([])
    config, spec = inputs()
    project = ProjectDocument.from_config(config).update_study(
        {"requirements": spec.to_dict(), "scene": SpatialScene((ScenePart.box("Sensor"),)).to_dict()}
    )
    window = CoilGenWorkbench(project, viewport_factory=Viewport, jobs_directory=tmp_path / "jobs")
    window.timer.stop()
    window.select_node("target")
    assert not window._draft_controls
    assert window.research_menu.isEnabled()
    assert not window.research_actions["_research_requirements_review"].isEnabled()
    window._research_busy("Array fit", True)
    assert "Array fit" in window._research_active_tasks
    window._research_busy("Array fit", False)
    assert not window._research_active_tasks
    window.undo_stack.setClean()
    window.close()
    window.deleteLater()
    application.processEvents()


@pytest.mark.parametrize("transition", ["channel_switch", "stale_accept", "changed_inputs", "close_project"])
def test_modeless_spatial_editor_retains_owner_and_rejects_stale_edits(tmp_path, transition):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import Signal
    from PySide6.QtWidgets import QApplication, QDialog, QWidget
    from coilgen.application.assembly import ProjectAssembly
    from coilgen.gui.workbench import CoilGenWorkbench

    class Viewport(QWidget):
        selected = Signal(str)
        point_picked = Signal(object)

        def render_config(self, config):
            self.config = config

        def set_scene(self, scene):
            self.scene = scene

        def set_point_pick_mode(self, enabled):
            self.picking = enabled

    application = QApplication.instance() or QApplication([])
    config, spec = inputs()
    original = ProjectDocument.from_config(config, name="Source").update_study({"requirements": spec.to_dict()})
    duplicate = original.duplicate("Equivalent independent channel")
    assert duplicate.id != original.id
    assert duplicate.fingerprint == original.fingerprint
    assembly = ProjectAssembly.new("Equivalent channels", (original, duplicate))
    window = CoilGenWorkbench(assembly, viewport_factory=Viewport, jobs_directory=tmp_path / "jobs")
    window.timer.stop()
    window._research_scene_dialog()
    editor = window._scene_editor
    assert editor.property("sourceProjectId") == original.id
    editor.pickRequested.emit()
    assert window.viewport.picking
    edited_scene = SpatialScene((ScenePart.box("Source-only exclusion"),)).to_dict()
    editor.result_value = edited_scene

    if transition == "channel_switch":
        window._set_project(duplicate, channel_id=assembly.channels[1].id)
        assert editor.result() == QDialog.DialogCode.Rejected
    elif transition == "stale_accept":
        # The finish guard also protects against an owner change that bypasses
        # the normal navigation lifecycle; numerical identity alone is equal.
        window.project = duplicate
        editor.accept()
        assert "belongs to another project" in window.messages.toPlainText()
    elif transition == "close_project":
        assert window.close_project()
        assert editor.result() == QDialog.DialogCode.Rejected
        assert window.project is None
    else:
        changed = original.update_study({"design_margin": 0.9})
        window._set_project(changed)
        editor.accept()
        assert "Input edit rejected" in window.messages.toPlainText()
        assert window.project.study["design_margin"] == 0.9

    assert window._scene_editor is None
    assert not window.viewport.picking
    if window.project is not None:
        assert window.project.study.get("scene") is None
        assert window.assembly.channel(assembly.channels[0].id).project.study.get("scene") is None
    assert window.viewport.scene is None
    window.undo_stack.setClean()
    window.close()
    window.deleteLater()
    application.processEvents()


@pytest.mark.parametrize("retain_receipt", [False, True])
def test_background_results_respect_owner_but_keep_submission_receipts(tmp_path, retain_receipt):
    from threading import Event
    from PySide6.QtCore import QElapsedTimer, Signal
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QWidget
    from coilgen.gui.workbench import CoilGenWorkbench

    class Viewport(QWidget):
        selected = Signal(str)

        def render_config(self, config):
            self.config = config

    application = QApplication.instance() or QApplication([])
    project = ProjectDocument.from_config(CoilGenConfig(), name="Original owner")
    window = CoilGenWorkbench(project, viewport_factory=Viewport, jobs_directory=tmp_path / "jobs")
    window.timer.stop()
    release, received = Event(), []

    def operation(captured):
        if not release.wait(2):
            raise RuntimeError("Fixture did not release the captured operation")
        return captured.id

    window._gradient_task(operation, received.append, "Fixture submission", retain_receipt=retain_receipt)
    window._set_project(project.duplicate("Equivalent new owner"))
    release.set()
    deadline = QElapsedTimer()
    deadline.start()
    while window._gradient_tasks and deadline.elapsed() < 3000:
        QTest.qWait(10)
    assert not window._gradient_tasks
    assert received == ([project.id] if retain_receipt else [])
    assert window.project.id != project.id
    window.undo_stack.setClean()
    window.close()
    window.deleteLater()
    application.processEvents()


def test_shared_rigid_registration_rejects_complex_values_and_reflections():
    with pytest.raises(ValueError, match="real numeric"):
        RigidTransform(np.eye(3, dtype=complex))
    with pytest.raises(ValueError, match="proper rigid"):
        RigidTransform(np.diag([-1, 1, 1]))
