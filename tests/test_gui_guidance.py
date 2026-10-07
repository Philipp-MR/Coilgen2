"""Evidence-driven guidance without Qt, file access, or numerical computation."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from coilgen.application import ProjectDocument
from coilgen.application.diagnostics import ProblemIssue
from coilgen.application.presets import preset_config, preset_study
from coilgen.application.targets import SolidHarmonicTarget
from coilgen.field_units import PROTON_GYROMAGNETIC_RATIO_HZ_PER_T
from coilgen.gui.guidance import assess_guidance, resolved_model_summary


READY = SimpleNamespace(blocking=())
HASH = "a" * 64


def _project(kind="cylinder"):
    return ProjectDocument.from_config(preset_config(kind)).update_study(preset_study(kind))


def _record(project, **overrides):
    metadata = {
        "project_id": project.id,
        "project_fingerprint": project.fingerprint,
        "project_model_fingerprint": project.model_fingerprint,
        "pipeline_stage": "evaluated",
        "summary": {
            "accepted": True,
            "sheet_accepted": True,
            "scope": "fabrication_review",
            "construction_released": False,
        },
        "pipeline_sweeps": {"checks_passed": True, "body_count": 1},
        "fabrication": {"ready": True},
        "artifacts": [{"name": "fabrication/body_fabrication.stl", "sha256": HASH}],
    }
    metadata.update(overrides)
    return SimpleNamespace(metadata=metadata)


def _stage(state, key):
    return next(item for item in state.stages if item.key == key)


def test_defined_inputs_offer_full_study_without_claiming_engineering_acceptance():
    project = _project()
    original = project.to_dict()
    state = assess_guidance(project, readiness=READY)
    assert state.action_key == "compute_full"
    assert tuple(item.key for item in state.stages) == (
        "target",
        "support",
        "conductor",
        "solve",
        "winding",
        "check",
        "export",
    )
    assert _stage(state, "target").status == "defined"
    assert _stage(state, "check").status == "not_computed"
    assert "feasibility" in state.detail
    assert project.to_dict() == original


def test_missing_target_is_an_input_problem_even_when_a_historical_solid_exists():
    project = ProjectDocument.from_config(replace(preset_config(), field_shape_function="none"))
    state = assess_guidance(project, _record(project), readiness=READY)
    assert state.action_key == "correct_input"
    assert _stage(state, "target").status == "needs_input"
    assert "target" in state.detail.lower()
    assert "engineering failures" not in state.title


def test_backend_validation_report_is_consumed_without_invented_optional_requirements():
    project = _project()
    report = SimpleNamespace(
        errors=(SimpleNamespace(field="conductor_cross_section_width", message="Width must be positive."),)
    )
    state = assess_guidance(project, readiness=report)
    assert state.action_key == "correct_input"
    assert _stage(state, "conductor").status == "needs_input"
    assert "Width must be positive" in state.detail


def test_pending_edits_do_not_offer_export_of_the_draft():
    project = _project()
    state = assess_guidance(project, _record(project), readiness=READY, pending=True)
    assert state.action_key == "compute_full"
    assert state.tone == "warning"
    assert _stage(state, "export").status == "pending"
    assert "recorded revision" in state.detail


@pytest.mark.parametrize("status", ("queued", "running", "cancelling"))
def test_current_worker_takes_precedence_over_an_exportable_result(status):
    project = _project()
    job = {"project_id": project.id, "project_fingerprint": project.fingerprint, "status": status}
    state = assess_guidance(project, _record(project), active_job=job, readiness=READY)
    assert state.action_key == "show_job"
    assert _stage(state, "solve").status == "running"


def test_worker_from_an_earlier_revision_is_identified_without_rewriting_its_inputs():
    project = _project()
    job = {"project_id": project.id, "project_fingerprint": "old-model", "status": "running"}
    state = assess_guidance(project, active_job=job, readiness=READY)
    assert state.action_key == "show_job"
    assert "earlier revision" in state.title.lower()
    assert "do not alter that job" in state.detail


def test_unrelated_channel_worker_does_not_override_selected_channel_export():
    project = _project()
    state = assess_guidance(
        project, _record(project), active_job={"project_id": "other-channel", "status": "running"}, readiness=READY
    )
    assert state.action_key == "export_geometry"


def test_identical_numeric_model_in_another_channel_is_not_current_evidence():
    project = _project()
    channel = project.duplicate("Independent second channel")
    assert project.fingerprint == channel.fingerprint
    state = assess_guidance(channel, _record(project), readiness=READY)
    assert state.action_key == "recompute"
    assert "another channel" in state.title
    assert _stage(state, "check").status == "stale"
    assert _stage(state, "export").status == "stale"


def test_changed_validation_settings_do_not_reuse_previous_acceptance_as_current():
    project = _project()
    updated = project.update_study({"validation_resolution": 20})
    state = assess_guidance(updated, _record(project), readiness=READY)
    assert updated.model_fingerprint == project.model_fingerprint
    assert state.action_key == "recompute"
    assert "Study settings changed" in state.title
    assert "own revision" in state.detail


@pytest.mark.parametrize("stage", ("prepared", "sheet", "contours", "routed"))
def test_partial_boundaries_offer_continuation_without_claiming_final_acceptance(stage):
    project = _project()
    record = _record(
        project,
        pipeline_stage=stage,
        summary={
            "scope": "continuous_sheet" if stage == "sheet" else "prepared_geometry",
            "accepted": True if stage == "sheet" else False,
            "sheet_accepted": True if stage != "prepared" else None,
        },
        fabrication=None,
        pipeline_sweeps={},
        artifacts=[],
    )
    state = assess_guidance(project, record, readiness=READY)
    assert state.action_key == "continue_full"
    assert _stage(state, "check").status == "unverified"
    assert "study complete" in state.title
    assert "Continue explicitly" in state.detail
    assert state.action_label.startswith("Continue to ")


def test_standard_gy_geometry_failures_override_a_passing_sheet():
    project = _project()
    record = _record(project, fabrication={"ready": False}, pipeline_sweeps={"checks_passed": False})
    state = assess_guidance(project, record, readiness=READY)
    assert state.action_key == "inspect_problems"
    assert _stage(state, "check").status == "failed"
    assert "passed sheet cannot override" in state.detail.lower()


def test_saved_field_failure_and_geometry_export_availability_are_distinct():
    project = _project()
    record = _record(
        project, constraints=[{"name": "Fabricated field budget", "scope": "fabrication_review", "passed": False}]
    )
    issue = ProblemIssue(
        "field", "Fabricated field error", "Measured field exceeds the requested budget.", settings_node="winding"
    )
    state = assess_guidance(project, record, issues=(issue,), readiness=READY)
    assert state.action_key == "inspect_problems"
    assert _stage(state, "export").status == "available"
    assert _stage(state, "export").tone == "warning"
    assert "CAD files remain available" in state.detail


def test_a_warning_about_undesigned_feeds_does_not_hide_available_closed_turn_geometry():
    project = _project("independent_turns")
    issue = ProblemIssue("feeds", "Feeds are not designed", "Closed turns have no drive terminals.", severity="warning")
    state = assess_guidance(project, _record(project), issues=(issue,), readiness=READY)
    assert state.action_key == "export_geometry"
    assert _stage(state, "check").status == "passed"
    assert "not construction release" in state.detail


def test_unverified_current_metadata_does_not_create_passed_checks():
    project = _project()
    record = _record(project, summary={}, fabrication={}, pipeline_sweeps={}, artifacts=[])
    state = assess_guidance(project, record, readiness=READY)
    assert state.action_key == "review_fabrication"
    assert _stage(state, "check").status == "unverified"
    assert all(item.status != "passed" for item in state.stages)


def test_legacy_identity_without_fingerprints_is_explicitly_unverified():
    project = _project()
    state = assess_guidance(project, SimpleNamespace(metadata={"summary": {"accepted": True}}), readiness=READY)
    assert state.action_key == "recompute"
    assert "identity is unverified" in state.title
    assert _stage(state, "check").status != "passed"


def test_empty_headless_identity_is_unverified_instead_of_another_channel():
    project = _project()
    record = _record(project, project_id="", project_fingerprint="", project_model_fingerprint=" ")
    state = assess_guidance(project, record, readiness=READY)
    assert state.action_key == "recompute"
    assert state.title == "Saved result identity is unverified"
    assert "another channel" not in state.detail


def test_empty_top_level_identity_uses_recorded_job_provenance():
    project = _project()
    record = _record(
        project,
        project_id="",
        project_fingerprint="",
        project_model_fingerprint="",
        job={
            "project_id": project.id,
            "project_fingerprint": project.fingerprint,
            "project_model_fingerprint": project.model_fingerprint,
        },
    )
    state = assess_guidance(project, record, readiness=READY)
    assert state.action_key == "export_geometry"


@pytest.mark.parametrize(
    "artifact",
    (
        {"name": "fabrication/body_fabrication_centroid.step", "sha256": HASH},
        {"name": "scientific_result.npz", "sha256": HASH},
        {"name": "fabrication/body_fabrication.stl"},
        {"name": "../body.stl", "sha256": HASH},
        {"name": "C:/body.stl", "sha256": HASH},
    ),
)
def test_arrays_centerlines_and_unverified_paths_do_not_enable_solid_export(artifact):
    project = _project()
    state = assess_guidance(project, _record(project, artifacts=[artifact]), readiness=READY)
    assert state.action_key != "export_geometry"
    assert _stage(state, "export").status == "not_run"


def test_recorded_faceted_step_is_solid_export_without_stl():
    project = _project()
    state = assess_guidance(
        project,
        _record(project, artifacts=[{"name": "fabrication/body_fabrication_faceted.step", "sha256": HASH}]),
        readiness=READY,
    )
    assert state.action_key == "export_geometry"


def test_summary_reports_actual_cylinder_span_and_distinct_wire_and_sheet():
    project = _project()
    summary = resolved_model_summary(project)
    assert "physical span 200 mm" in summary["support"]
    assert "Gy = dBz/dy = 20 mT/m" in summary["target"]
    assert "40 mm diameter" in summary["roi"]
    assert "world origin" in summary["roi"]
    assert "2 × 2 mm (4 mm²)" in summary["conductor"]
    assert "sheet thickness 1 mm" in summary["conductor"]
    assert "validation limits ÷ 0.6" in summary["study"]


def test_summary_preserves_absolute_field_and_hz_units_without_gradient_normalization():
    config = replace(
        preset_config("independent_turns"),
        field_shape_function=str(PROTON_GYROMAGNETIC_RATIO_HZ_PER_T * 1e-4),
        target_field_units="Hz",
    )
    summary = resolved_model_summary(ProjectDocument.from_config(config))
    assert "Uniform Bz = 100 µT" in summary["target"]
    assert "absolute Hz input" in summary["target"]
    assert "mT/m" not in summary["target"]


def test_summary_does_not_confuse_harmonic_center_reference_radius_and_roi():
    target = SolidHarmonicTarget(2, 0, 16e-6, 0.05, (0.01, 0, 0))
    config = replace(preset_config(), **target.to_config_overrides(), target_region_radius=0.03)
    summary = resolved_model_summary(ProjectDocument.from_config(config))
    assert "coefficient 16 µT" in summary["target"]
    assert "R=50 mm" in summary["target"]
    assert "center (10, 0, 0) mm" in summary["target"]
    assert "not the pole amplitude" in summary["target"]
    assert "60 mm diameter" in summary["roi"]
    assert "world origin" in summary["roi"]


def test_mesh_center_roi_is_not_assumed_to_be_the_geometric_center():
    project = ProjectDocument.from_config(replace(preset_config(), set_roi_into_mesh_center=True))
    summary = resolved_model_summary(project)
    assert "support vertex centroid" in summary["roi"]
    assert "world origin" not in summary["roi"]


def test_imported_map_and_support_show_owned_provenance_without_temporary_cache(tmp_path, monkeypatch):
    source = tmp_path / "measured_target.csv"
    source.write_text("0,0,0,0,0,1e-4\n", encoding="utf-8")
    support = tmp_path / "support.npz"
    support.write_bytes(b"owned mesh bytes; not loaded by the overview")
    project = ProjectDocument.from_config(
        replace(preset_config(), coil_mesh_file=str(support), target_field_definition_file=str(source))
    )
    monkeypatch.setattr(
        ProjectDocument, "config", property(lambda _self: pytest.fail("Overview must not create input caches."))
    )
    summary = resolved_model_summary(project)
    assert "measured_target.csv" in summary["target"]
    assert "support.npz" in summary["support"]
    assert "coordinates come" not in summary["roi"] or "imported map" in summary["roi"]
    assert "generated spherical ROI controls do not" in summary["roi"]
    assert not any("coilgen-project-" in value for value in summary.values())
    assert assess_guidance(project).action_key == "compute_full"


def test_nonrectangular_profile_is_not_described_as_the_fallback_rectangle():
    config = replace(preset_config(), cross_sectional_points=((0.0, 0.0), (0.001, 0.0), (0.0, 0.001)))
    summary = resolved_model_summary(ProjectDocument.from_config(config))
    assert "Custom conducting cross-section" in summary["conductor"]
    assert "4 mm²" not in summary["conductor"]


def test_winding_overview_separates_derived_components_current_and_sampling_scope():
    project = _project()
    record = _record(
        project,
        summary={"contour_components": 12, "wire_length_m": 2.1},
        wire_fields={
            "contour": {"current_A": 4.8, "current_refitted": False, "sample_count": 257},
            "routed": {"current_A": 5.0, "current_refitted": True, "sample_count": 257},
        },
    )
    detail = _stage(assess_guidance(project, record, readiness=READY), "winding").detail
    assert "Actual contour components: 12" in detail
    assert "distinct from configured contour levels" in detail
    assert "I=4.8 A; not refitted; 257 design-grid samples" in detail
    assert "I=5 A; refitted" in detail
    assert "not independent winding validation" in detail
    assert "electrical-path length: 2.1 m" in detail


def test_missing_winding_metrics_do_not_infer_current_length_or_turn_count():
    project = _project()
    detail = _stage(assess_guidance(project, _record(project), readiness=READY), "winding").detail
    assert "count is not recorded" in detail
    assert "I=" not in detail
    assert "length:" not in detail


def test_malformed_geometry_keeps_overview_usable_and_requests_input_correction():
    project = ProjectDocument.from_config(replace(preset_config(), cylinder_mesh_parameter_list=(0.2, 0.08)))
    summary = resolved_model_summary(project)
    assert "invalid settings" in summary["study"]
    state = assess_guidance(project)
    assert state.action_key == "correct_input"
    assert _stage(state, "support").status == "needs_input"
