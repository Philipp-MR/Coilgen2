"""The system plan respects ownership, recorded boundaries and evidence health."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from coilgen.application.assembly import ChannelDocument, ProjectAssembly
from coilgen.application.project import ProjectDocument
from coilgen.application.presets import preset_config
from coilgen.application.reuse import ReusableStage, ReusePlan
from coilgen.application.service import RunRecord
from coilgen.gui.design_plan import build_design_plan
from coilgen.pipeline_stages import PipelineStage


@pytest.fixture
def assembly():
    project = ProjectDocument.from_config(preset_config(), name="Gx")
    return ProjectAssembly(
        "system",
        "XYZ gradients",
        1,
        tuple(
            ChannelDocument(name.lower(), name, project if name == "Gx" else project.duplicate(name))
            for name in ("Gx", "Gy", "Gz")
        ),
    )


def state(project, *, id_="run", status="running", **values):
    return {
        "id": id_,
        "status": status,
        "project_id": project.id,
        "project_revision": project.revision,
        "project_fingerprint": project.fingerprint,
        "pipeline_stage": project.study["pipeline_stage"],
        **values,
    }


def record(project, *, id_="run", **values):
    return SimpleNamespace(metadata=state(project, id_=id_, status="completed", **values))


def statuses(channel):
    return tuple(stage.status for stage in channel.stages)


def test_fresh_system_has_every_channel_and_independent_pending_steps(assembly):
    plan = build_design_plan(assembly, active_channel_id="gy")
    assert [channel.channel_id for channel in plan.channels] == ["gx", "gy", "gz"]
    assert [channel.selected for channel in plan.channels] == [False, True, False]
    assert all(statuses(channel) == ("waiting",) * 5 for channel in plan.channels)
    assert (plan.completed_count, plan.requested_count, plan.waiting_channels) == (0, 15, 3)
    assert (plan.enabled_channels, plan.completed_channels, plan.disabled_channels) == (3, 0, 0)


def test_single_document_is_supported_without_importing_qt(assembly):
    project = assembly.channel("gx").project
    plan = build_design_plan(project)
    assert plan.channels[0].channel_id == project.id
    assert plan.channels[0].selected and plan.requested_count == 5
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import coilgen.gui.design_plan; "
            "assert not any(k.startswith(('PySide6', 'vtk', 'pyvista')) for k in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_verified_run_plan_reads_immutable_evidence_without_thawing_large_witnesses(assembly, tmp_path):
    class ImmutableRecord(RunRecord):
        @property
        def metadata(self):
            raise AssertionError("Plan presentation must not thaw all conductor witnesses")

    project = assembly.channel("gx").project
    saved = ImmutableRecord(
        tmp_path,
        state(
            project,
            status="completed",
            summary={"scope": "continuous_sheet", "accepted": False},
            constraints=[{"passed": False}],
            fabrication={
                "ready": False,
                "paths": [
                    {"manufacturing_ok": False, "witness_points_m": [[i * 0.001, 0.0, 0.0] for i in range(3000)]}
                ],
            },
        ),
        {},
    )
    original = saved.metadata_view
    plan = build_design_plan(assembly, records=(saved,))
    channel = plan.channels[0]
    assert channel.completed_count == 5 and channel.verification == "verified"
    assert channel.review_needed and all(stage.status == "complete" for stage in channel.stages)
    assert saved.metadata_view is original
    assert len(original["fabrication"]["paths"][0]["witness_points_m"]) == 3000


def test_disabled_channels_remain_visible_but_do_not_inflate_system_totals(assembly):
    assembly = assembly.update_channel("gy", enabled=False)
    project = assembly.channel("gy").project
    plan = build_design_plan(assembly, states=[state(project, progress=0.45)])
    disabled = plan.channels[1]
    assert not disabled.enabled and disabled.status == "running" and disabled.percent == 45
    assert "excluded from system compute" in disabled.activity
    assert plan.running_channels == 0
    assert (plan.enabled_channels, plan.disabled_channels, plan.requested_count) == (2, 1, 10)


def test_channels_keep_distinct_study_boundaries_and_selected_override(assembly):
    gx, gy = (assembly.channel(id_).project for id_ in ("gx", "gy"))
    assembly = assembly.update_channel("gx", project=gx.update_study({"pipeline_stage": "prepared"}))
    assembly = assembly.update_channel("gy", project=gy.update_study({"pipeline_stage": "sheet"}))
    plan = build_design_plan(assembly, active_channel_id="gx", requested_stage="contours")
    assert [channel.requested_stage for channel in plan.channels] == ["contours", "sheet", "evaluated"]
    assert [channel.requested_count for channel in plan.channels] == [3, 2, 5]
    assert statuses(plan.channels[0]) == ("waiting", "waiting", "waiting", "skipped", "skipped")


def test_identical_physics_fingerprints_do_not_share_channel_progress(assembly):
    gx, gy = (assembly.channel(id_).project for id_ in ("gx", "gy"))
    assert gx.fingerprint == gy.fingerprint
    plan = build_design_plan(assembly, states=[state(gx, completed_stages=["prepared"], progress=0.23)])
    assert statuses(plan.channels[0]) == ("complete", "current", "waiting", "waiting", "waiting")
    assert statuses(plan.channels[1]) == ("waiting",) * 5
    assert plan.channels[0].percent == 23
    assert plan.completed_count == 1 and plan.running_channels == 1


@pytest.mark.parametrize("status", ["queued", "running", "completed", "failed", "cancelled", "interrupted"])
def test_percentage_and_free_text_never_complete_steps(assembly, status):
    gx = assembly.channel("gx").project
    channel = build_design_plan(
        assembly, states=[state(gx, status=status, progress=1, stage="All stages complete; accepted")]
    ).channels[0]
    assert channel.completed_count == 0
    assert not any(stage.status == "complete" for stage in channel.stages)
    if status == "completed":
        assert channel.verification == "pending" and channel.busy


def test_partial_explicit_evidence_does_not_fill_holes(assembly):
    gx = assembly.channel("gx").project
    channel = build_design_plan(
        assembly, states=[state(gx, completed_stages=["sheet"], active_stage="routed")]
    ).channels[0]
    assert statuses(channel) == ("waiting", "complete", "waiting", "current", "waiting")
    assert not channel.stages[3].inferred
    assert channel.completed_count == 1


def test_current_job_uses_actual_requested_boundary_and_newest_active_job(assembly):
    gx = assembly.channel("gx").project
    newer_done = state(gx, id_="done", status="completed", created_at="2026-10-09T15:00:00Z")
    queued = state(gx, id_="queued", status="queued", created_at="2026-10-09T14:30:00Z")
    running = state(gx, id_="working", created_at="2026-10-09T14:00:00Z", pipeline_stage="sheet")
    channel = build_design_plan(assembly, states={"done": newer_done, "working": running, "queued": queued}).channels[0]
    assert channel.job_id == "queued" and channel.status == "queued"
    channel = build_design_plan(assembly, states=[newer_done, running]).channels[0]
    assert channel.job_id == "working" and channel.requested_count == 2


def test_current_saved_result_and_nested_job_identity_verify_completion(assembly):
    gx = assembly.channel("gx").project
    saved = record(gx)
    saved.metadata.pop("id")
    saved.metadata["job"] = {"id": "run"}
    plan = build_design_plan(assembly, states=[state(gx, status="completed")], records=[saved])
    channel = plan.channels[0]
    assert channel.verification == "verified" and channel.status == "completed"
    assert statuses(channel) == ("complete",) * 5
    assert plan.completed_channels == 1 and plan.completed_count == 5


def test_archive_alias_is_joined_by_original_request_identity(assembly):
    gx = assembly.channel("gx").project
    saved = record(gx, request_sha256="request-digest")
    saved.metadata.pop("id")
    saved.id, saved.alias = "archive-digest", "Saved Gx"
    job = state(gx, id_="runtime-id", status="completed", request_sha256="request-digest")
    channel = build_design_plan(assembly, states=[job], records=[saved]).channels[0]
    assert channel.verification == "verified" and channel.completed_count == 5


def test_record_mapping_key_is_preserved_for_integrity_errors(assembly):
    gx = assembly.channel("gx").project
    saved = record(gx)
    saved.metadata.pop("id")
    job = state(gx, id_="runtime-id", status="completed", completed_stages=["prepared"])
    channel = build_design_plan(
        assembly, states=[job], records={"runtime-id": saved}, verification_errors={"runtime-id": "Hash mismatch"}
    ).channels[0]
    assert channel.verification == "failed" and channel.completed_count == 0
    assert "Hash mismatch" in channel.tooltip


def test_earlier_model_is_stale_and_cannot_complete_current_steps(assembly):
    gx = assembly.channel("gx").project
    saved = record(gx)
    changed = gx.update_study({"validation_resolution": 20})
    assembly = assembly.update_channel("gx", project=changed)
    channel = build_design_plan(assembly, records=[saved], states=[state(gx, status="completed")]).channels[0]
    assert channel.status == "stale" and channel.completed_count == 0
    assert statuses(channel) == ("stale",) * 5
    assert channel.historical_count == 1
    assert "Earlier model/study" in channel.activity


def test_revision_only_change_does_not_claim_old_live_worker_but_can_retain_saved_physics(assembly):
    gx = assembly.channel("gx").project
    renamed = replace(gx, revision=gx.revision + 1, name="Renamed Gx")
    assert renamed.fingerprint == gx.fingerprint
    assembly = assembly.update_channel("gx", project=renamed)
    plan = build_design_plan(assembly, states=[state(gx, progress=0.7)], records=[record(gx)])
    channel = plan.channels[0]
    assert channel.job_id is None and channel.percent is None
    assert channel.status == "completed" and channel.completed_count == 5


def test_new_running_job_does_not_borrow_old_final_result_as_its_completion(assembly):
    gx = assembly.channel("gx").project
    channel = build_design_plan(
        assembly, states=[state(gx, id_="new", completed_stages=["prepared"])], records=[record(gx, id_="old")]
    ).channels[0]
    assert statuses(channel) == ("complete", "current", "waiting", "waiting", "waiting")
    assert channel.verification is None and channel.completed_count == 1


def test_real_reuse_boundaries_remain_distinct_from_recorded_computation(assembly):
    reuse = ReusePlan((ReusableStage(PipelineStage.SHEET, Path("sheet.cgstage"), Path("run")),), ("contours",), ())
    plan = build_design_plan(assembly, reuse_plans={"gx": reuse}, pending_channel_ids=("gx",))
    channel = plan.channels[0]
    assert statuses(channel) == ("waiting", "reusable", "stale", "waiting", "waiting")
    assert channel.reuse_from == "sheet" and channel.completed_count == 0
    assert channel.pending and "unapplied edits" in channel.activity


def test_engineering_failures_do_not_change_successful_computation_into_execution_failure(assembly):
    gx = assembly.channel("gx").project
    saved = record(gx, constraints=[{"passed": False}], fabrication={"manufacturing_ok": False})
    plan = build_design_plan(assembly, records=[saved], engineering_errors={"gx": 18})
    channel = plan.channels[0]
    assert channel.status == "completed" and channel.completed_count == 5
    assert channel.review_needed and channel.engineering_errors == 18
    assert plan.completed_channels == 1 and plan.failed_channels == 0


@pytest.mark.parametrize(
    "metadata",
    [
        {"summary": {"accepted": False}},
        {"pipeline_sweeps": {"checks_passed": False}},
        {"fabrication": {"ready": False}},
    ],
)
def test_recorded_canonical_engineering_failures_request_review(assembly, metadata):
    gx = assembly.channel("gx").project
    channel = build_design_plan(assembly, records=[record(gx, **metadata)]).channels[0]
    assert channel.review_needed and channel.status == "completed"


def test_failed_current_saved_record_reports_verification_problem_without_current_job(assembly):
    gx = assembly.channel("gx").project
    plan = build_design_plan(assembly, records={"run": record(gx)}, verification_errors={"run": "Corrupt record"})
    channel = plan.channels[0]
    assert channel.verification == "failed" and channel.completed_count == 0
    assert "verification failed" in channel.activity and plan.failed_channels == 1


def test_runtime_refresh_failure_does_not_erase_verified_channel_completion(assembly):
    gx = assembly.channel("gx").project
    plan = build_design_plan(
        assembly,
        states=[state(gx, status="completed")],
        records=[record(gx)],
        runtime_unavailable="Temporary status read error",
    )
    assert plan.channels[0].status == "unavailable"
    assert plan.channels[0].verification == "verified" and plan.completed_channels == 1
    assert plan.completed_count == 5


def test_nearly_finished_work_remains_below_100_percent_and_shows_operation_counters(assembly):
    gx = assembly.channel("gx").project
    job = state(gx, progress=0.9999, phase_progress=0.373, progress_completed=128, progress_total=344)
    channel = build_design_plan(assembly, states=[job]).channels[0]
    assert channel.percent == 99
    assert "operation 37%" in channel.activity and "128/344 work units" in channel.tooltip
    assert channel.completed_count == 0


def test_indeterminate_operation_is_explicit_and_boolean_counters_are_rejected(assembly):
    gx = assembly.channel("gx").project
    job = state(
        gx, progress=0.23, progress_indeterminate=True, phase_progress=0.373, progress_completed=True, progress_total=2
    )
    channel = build_design_plan(assembly, states=[job]).channels[0]
    assert channel.percent == 23 and channel.busy
    assert "no measurable work total" in channel.activity and "work units" not in channel.activity


@pytest.mark.parametrize(
    "status, stage_status", [("failed", "failed"), ("cancelled", "stopped"), ("interrupted", "stopped")]
)
def test_stopped_jobs_preserve_only_recorded_earlier_stages(assembly, status, stage_status):
    gx = assembly.channel("gx").project
    channel = build_design_plan(
        assembly, states=[state(gx, status=status, completed_stages=["prepared", "sheet"])]
    ).channels[0]
    assert statuses(channel) == ("complete", "complete", stage_status, "waiting", "waiting")
    assert channel.percent is None


def test_runtime_failure_retains_evidence_without_current_percentage_or_animation(assembly):
    gx = assembly.channel("gx").project
    job = state(gx, progress=0.45, completed_stages=["prepared"], stage="Optimizing sheet")
    before = deepcopy(job)
    plan = build_design_plan(assembly, states=[job], runtime_unavailable="Status file unreadable")
    channel = plan.channels[0]
    assert channel.status == "unavailable" and channel.percent is None and not channel.busy
    assert statuses(channel) == ("complete", "waiting", "waiting", "waiting", "waiting")
    assert "last known" in channel.activity and "Status file unreadable" in channel.tooltip
    assert job == before


@pytest.mark.parametrize("progress", [True, float("inf"), float("nan"), "0.5", -0.1, 1.1, None])
def test_invalid_worker_percentages_stay_indeterminate(assembly, progress):
    gx = assembly.channel("gx").project
    channel = build_design_plan(assembly, states=[state(gx, progress=progress)]).channels[0]
    assert channel.percent is None and channel.busy


def test_empty_and_all_excluded_systems_have_valid_zero_totals(assembly):
    for channel in assembly.channels:
        assembly = assembly.update_channel(channel.id, enabled=False)
    assert build_design_plan(assembly).requested_count == 0
    empty = build_design_plan(ProjectAssembly("empty", "Empty system", 1))
    assert not empty.channels and empty.completed_count == empty.requested_count == 0


def test_unknown_requested_stage_is_rejected(assembly):
    with pytest.raises(ValueError, match="Unknown design stage"):
        build_design_plan(assembly, requested_stage="fabrication")
