"""Execution workflows for the native workbench.

The window retains Qt widgets and signal/undo adapters; this controller
coordinates application operations without owning a second model.
"""

from __future__ import annotations

from ..application import DesignService
from ..application.execution import plan_execution
from ..application.session import submit_recorded_design
from .readiness import ReadinessIssue
from ..research.jobs import JobManager, RuntimePolicy


class ExecutionController:
    def __init__(self, window):
        self.window = window

    def compute(self, _checked=False, *, resume_checkpoint=None, pipeline_stage=None):
        window = self.window
        if window.project is None:
            window.statusBar().showMessage("Create or open a project before computing.", 5000)
            return None
        if not window._check_expert_draft():
            return None
        if window._submitting:
            return None
        if not window.apply_settings():
            window._report_compute_draft_error()
            return None

        window._submitting = True
        window._update_execution_feedback()
        try:
            if not window._check_compute_readiness(window.project, pipeline_stage=pipeline_stage):
                return None
            window.poll_jobs()
            controls = window._request_controls()
            if pipeline_stage is not None:
                controls = {**controls, "pipeline_stage": pipeline_stage, "fabricate": False}
            plan = plan_execution(window.project, states=window.states.values(), **controls)
            existing = window.states.get(plan.active_job_id)
            if existing is not None and resume_checkpoint is None and not isinstance(window.service, DesignService):
                window._begin_live_follow(existing)
                window.design_plan_dock.show()
                window.design_plan_dock.raise_()
                window._focus_job(existing["id"], open_window=False)
                window.statusBar().showMessage(
                    f"This revision is already {existing['status']}; showing its design progress.", 10000
                )
                return existing
            if window.jobs is None:
                window.jobs = JobManager(window.jobs_directory, RuntimePolicy())
                window._own_jobs = True
            window._snapshot_existing_jobs()
            window._refresh_stage_status()
            state, is_new = submit_recorded_design(
                window.project,
                window.jobs,
                controls,
                workspace=window.jobs_directory.parent,
                states=window.states,
                service=window.service,
                reuse=window._reuse_plan,
                resume_checkpoint=resume_checkpoint,
            )
            reused = not is_new
            if is_new:
                window._session_job_ids.add(state["id"])
            window._begin_live_follow(state)
            window.design_plan_dock.show()
            window.design_plan_dock.raise_()
            window.statusBar().showMessage(
                f"This revision is already {state['status']}; showing its design progress."
                if reused
                else f"Queued immutable revision {window.project.revision}",
                5000,
            )
            window.poll_jobs()
            window._focus_job(state["id"], open_window=False)
            return state
        except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
            window._error(f"Cannot submit run: {exc}")
            return None
        finally:
            window._submitting = False
            window._update_execution_feedback()

    def compute_channels(self):
        window = self.window
        if window.assembly is None:
            window.statusBar().showMessage("Create or open a project before computing.", 5000)
            return None
        if window._submitting:
            return None
        if not window._check_expert_draft():
            return None
        if not window.apply_settings():
            window._report_compute_draft_error()
            return None
        window._submitting = True
        window._update_execution_feedback()
        window.computation_indicator.repaint()
        try:
            enabled = [channel for channel in window.assembly.channels if channel.enabled]
            if not enabled:
                window.readiness_panel.set_issues(
                    (
                        ReadinessIssue(
                            title="No enabled channel",
                            message="Enable at least one channel before computing the assembly.",
                            node="geometry",
                            key="channel_enabled",
                            action="Use Enabled beside the active channel selector.",
                        ),
                    )
                )
                window.statusBar().showMessage("Enable at least one channel before computing.", 15000)
                return None
            for channel in enabled:
                if not window._check_compute_readiness(channel.project, channel_name=channel.name):
                    window.channel_selector.setCurrentIndex(window.channel_selector.findData(channel.id))
                    window._check_compute_readiness(channel.project, channel_name=channel.name)
                    return None
            if window.jobs is None:
                window.jobs = JobManager(window.jobs_directory, RuntimePolicy())
                window._own_jobs = True
            window._snapshot_existing_jobs()
            from ..application.session_evidence import retain_revision

            for channel in enabled:
                retain_revision(window.jobs_directory.parent, channel.project)
            states = window.service.submit_assembly(window.assembly, window.jobs)
            window._session_job_ids.update(state["id"] for state in states.values())
            window.states.update({state["id"]: state for state in states.values()})
            window.design_plan_dock.show()
            window.design_plan_dock.raise_()
            active_state = next((state for state in states.values() if window._live_owner_matches(state)), None)
            if active_state is not None:
                window._begin_live_follow(active_state)
            window.messages.appendPlainText(
                "Submitted independent channels. Coupled fields, circuits, mutual inductance and assembly clearance remain unverified."
            )
            window.poll_jobs()
            if states:
                window._focus_job((active_state or next(iter(states.values())))["id"], open_window=False)
            return states
        except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
            window._error(f"Cannot submit channels: {exc}")
            return None
        finally:
            window._submitting = False
            window._update_execution_feedback()

    def poll_jobs(self):
        window = self.window
        if window.jobs is None:
            window._update_execution_feedback()
            return
        try:
            states = window.jobs.list()
            window._last_job_states = tuple(states)
            if window.project is None:
                window.states.update((state["id"], dict(state)) for state in states)
                window._last_runtime_error = None
                window._fill_jobs(states)
                window._record_job_outcomes(states)
                window._refresh_computation_indicator()
                return
            window._refresh_computation_indicator()
            window._snapshot_existing_jobs(states)
            merged_states = []
            stopped_changed = False
            stopped_states = []
            for state in states:
                previous = window.states.get(state["id"], {})
                # A test/legacy runtime may omit service annotations; preserve them.
                recovered_write = "persistence_error" not in state
                state = {**previous, **state}
                if recovered_write:
                    state.pop("persistence_error", None)
                window.states[state["id"]] = state
                changed = state.get("status") in {"failed", "cancelled", "interrupted"} and state.get(
                    "status"
                ) != previous.get("status")
                stopped_changed |= changed
                if changed:
                    stopped_states.append(state)
                merged_states.append(state)
                if state["status"] == "completed" and state["id"] not in window._loaded_ids:
                    window._start_completed_run_load(state)
                if state["id"] == window.live_preview.job_id:
                    window.live_preview.refresh(state)
            window._fill_jobs(merged_states)
            window._record_job_outcomes(merged_states)
            if stopped_changed:
                window._start_stopped_reuse_load(stopped_states[-1])
            window._refresh_stage_status(refresh_reuse=False)
            window._last_runtime_error = None
            window._refresh_channel_overview()
            window._update_execution_feedback()
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            message = f"Job runtime unavailable: {exc}"
            if message != window._last_runtime_error:
                window._error(message)
                window._last_runtime_error = message
            window._update_execution_feedback()

    def cancel_selected_job(self):
        window = self.window
        id_ = window._selected_job_id()
        if window.jobs is not None and window.states.get(id_, {}).get("status") in {"queued", "running"}:
            try:
                window.jobs.cancel(id_)
                window.poll_jobs()
            except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
                window._error(f"Cannot cancel job: {exc}")
