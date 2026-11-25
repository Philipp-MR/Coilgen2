"""Document workflows for the native workbench.

The window retains Qt widgets and signal/undo adapters; this controller
coordinates application operations without owning a second model.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QDockWidget, QToolButton
from ..application import ProjectAssembly, ProjectDocument
from ..application.bundles import RunArchive
from ..application.templates import template_assembly, template_spec
from ..attribution import AUTHOR_NAME, PROJECT_URL, SOFTWARE_VERSION
from ..persistence import atomic_text
from ..research.storage import result_metadata_digest
from .run_metadata import read_run_metadata


class DocumentController:
    def __init__(self, window):
        self.window = window

    def save_project(self, path):
        window = self.window
        if window.project is None:
            raise ValueError("Create or open a project before saving.")
        if not window._apply_document_inputs():
            raise ValueError(
                "Correct pending settings, or apply/discard the retained Advanced settings draft before saving."
            )
        with window._visible_work("Saving project and results"):
            portable = window._portable_assembly()
            options = {"progress": window._portable_progress} if Path(path).suffix.lower() == ".coilgen" else {}
            window.project_path = Path(
                portable.save(path, **options)
                if window._collection_mode or len(window.assembly.channels) > 1
                else portable.channel(window._active_channel_id).project.save(path, **options)
            )
            window.assembly = portable
            window.project = portable.channel(window._active_channel_id).project
            window.undo_stack.setClean()
            window._save_session()
            window.statusBar().showMessage(f"Saved {window.project_path}", 5000)
            return window.project_path

    def _portable_assembly(self):
        """Embed all available completed evidence, retaining per-channel ownership."""
        window = self.window
        window._save_session()
        channels = []
        for channel in window.assembly.channels:
            project = channel.project
            scope = channel.id if window._collection_mode else None
            try:
                entries = window.history.restore(project.id, channel_id=scope) if window.history is not None else ()
            except (OSError, ValueError, TypeError, RuntimeError) as exc:
                entries = ()
                window.messages.appendPlainText(
                    f"Local history cache unavailable; saving owned project evidence: {exc}"
                )
            aliases = {entry.directory.resolve(): entry.alias for entry in entries}
            records = {
                entry.run.directory.resolve(): entry.run
                for entry in entries
                if entry.verified and not window._run_verification_error(entry.run)
            }
            records.update(
                {
                    record.directory.resolve(): record
                    for id_, record in window.runs.items()
                    if not window._run_verification_error(record)
                    and (
                        read_run_metadata(record).get("project_id") == project.id
                        or (
                            not read_run_metadata(record).get("project_id")
                            and project.id == window.project.id
                            and id_ == window._selected_run
                        )
                    )
                }
            )
            saved = {archive.id: archive for archive in project.saved_runs}
            paths = {}
            for directory, record in records.items():
                metadata = {key: value for key, value in read_run_metadata(record).items() if key != "job"}
                previous = next(
                    (
                        archive
                        for archive in saved.values()
                        if archive.metadata.get("request_sha256") == metadata.get("request_sha256")
                        and result_metadata_digest(archive.metadata) == result_metadata_digest(metadata)
                    ),
                    None,
                )
                archive = (
                    replace(previous, alias=aliases.get(directory, previous.alias))
                    if previous is not None
                    else RunArchive.capture(
                        record,
                        alias=aliases.get(directory),
                        source_project=window._export_source(record)
                        if window._has_export_source(record, retained_only=True)
                        else None,
                    )
                )
                saved[archive.id] = archive
                paths[directory] = archive.id
            session = project.session
            if project.id == window.project.id:
                selected = window.runs.get(window._selected_run)
                session = {
                    **session,
                    "selected_run_id": paths.get(selected.directory.resolve()) if selected else None,
                    "view_state": window._workbench_view_state(),
                }
            elif window.history is not None:
                try:
                    state = window.history.session(project.id, channel_id=scope)
                except (OSError, ValueError, TypeError, RuntimeError):
                    state = session
                selected = next((entry for entry in entries if entry.id == state.get("selected_id")), None)
                session = {
                    **session,
                    "selected_run_id": paths.get(selected.directory.resolve()) if selected else None,
                    "view_state": state.get("view_state", {}),
                }
            if project.id != window.project.id and (project.id, scope) in window._panel_views:
                session = {**session, "view_state": window._panel_views[(project.id, scope)]}
            if any(not entry.verified for entry in entries):
                session["unavailable_runs"] = [
                    {"alias": entry.alias, "diagnostic": entry.diagnostic} for entry in entries if not entry.verified
                ]
            session["active_channel_id"] = window._active_channel_id if window._collection_mode else None
            channels.append(replace(channel, project=project.with_results(tuple(saved.values()), session=session)))
        return replace(window.assembly, channels=tuple(channels))

    def open_project(self, path):
        window = self.window
        window._invalidate_project_open()
        with window._visible_work("Opening project and verifying embedded results"):
            return window._open_project(path)

    def _open_project(self, path):
        window = self.window
        from .project_loading import load_document

        return window._install_project(load_document(path, progress=window._portable_progress), path)

    def _install_project(self, document, path, *, prepared_history=None, prepared_reuse=None, readiness=None):
        window = self.window
        assembly = document if isinstance(document, ProjectAssembly) else None
        project = document if isinstance(document, ProjectDocument) else None
        # Validation precedes discarding the previous owner's editor.
        window._discard_project_editors()
        window._draft_dirty = False
        window.undo_stack.clear()
        window.project_path = Path(path)
        window._history_key = None
        window._panel_views.clear()
        window._collection_mode = assembly is not None
        window._prepared_project_history = prepared_history
        window._prepared_project_reuse = prepared_reuse
        window._restoring_project_view = prepared_history is not None
        if readiness is not None:
            active = assembly.channel(window._saved_active_channel(assembly)).project if assembly else project
            window._guidance_readiness_key = (active.fingerprint, active.study["pipeline_stage"])
            window._guidance_readiness = readiness
        try:
            window._set_assembly(assembly or ProjectAssembly.new(project.name, (project,)))
        finally:
            window._prepared_project_history = None
            window._prepared_project_reuse = None
            window._restoring_project_view = False
        if prepared_history is not None:
            window._node_changed(window._current_node or window.tree.currentItem())
        window.undo_stack.setClean()
        window.statusBar().showMessage(f"Opened {window.project.name} · revision {window.project.revision}", 7000)
        return document

    def _invalidate_project_open(self, *_args):
        window = self.window
        task = window._project_open_task
        if task is None:
            return
        window._project_open_generation += 1
        window._project_open_task = None
        window._foreground_work.pop(task, None)
        task.cancel()
        window._project_open_cancel.hide()
        window._refresh_computation_indicator()
        window.statusBar().showMessage("Project opening cancelled; current work is retained.", 7000)

    def _start_project_open(self, path):
        window = self.window
        from .project_loading import ProjectOpenTask

        window._invalidate_project_open()
        if window._closing:
            return None
        window._project_open_generation += 1
        token = window._project_open_generation
        task = ProjectOpenTask(
            path,
            window.jobs_directory.parent / "embedded-runs",
            history=window.history,
            token=token,
            jobs_directory=window.jobs_directory,
            states=tuple(state for id_, state in window.states.items() if not window._run_verification_error(id_)),
            records=tuple(record for record in window.runs.values() if not window._run_verification_error(record)),
            threads=getattr(getattr(window.jobs, "policy", None), "threads_per_worker", 1),
        )
        task.verification_generations = dict(window._source_verification_generations)
        window._project_open_task = task
        task.signals.progress.connect(window._project_open_progress)
        task.signals.finished.connect(window._project_open_ready)
        if not hasattr(window, "_project_open_cancel"):
            window._project_open_cancel = QToolButton(window)
            window._project_open_cancel.setText("Cancel opening")
            window._project_open_cancel.clicked.connect(window._invalidate_project_open)
            window.statusBar().addPermanentWidget(window._project_open_cancel)
        window._project_open_cancel.show()
        window._foreground_work[task] = f"Opening {Path(path).name} · verifying saved evidence"
        window._refresh_computation_indicator()
        QThreadPool.globalInstance().start(task)
        return task

    def _project_open_progress(self, token, message):
        window = self.window
        task = window._project_open_task
        if task is None or window._closing or token != window._project_open_generation:
            return
        window._foreground_work[task] = "Opening project · " + message
        window.statusBar().showMessage(message)
        window._refresh_computation_indicator()

    def _project_open_ready(self, token, result, error):
        window = self.window
        task = window._project_open_task
        if task is None or window._closing or token != window._project_open_generation:
            return
        window._project_open_task = None
        window._foreground_work.pop(task, None)
        window._project_open_cancel.hide()
        window._refresh_computation_indicator()
        if error:
            if error != "Cancelled":
                window._error(f"Cannot open project: {error}")
            return
        with window._visible_work("Restoring verified project view"):
            try:
                reuse = result.reuse
                if reuse is not None:
                    reuse = replace(
                        reuse,
                        available=tuple(
                            entry
                            for entry in reuse.available
                            if not window._source_health.known_error(entry.run_directory)
                            and window._source_verification_generations.get(entry.run_directory, 0)
                            == task.verification_generations.get(entry.run_directory, 0)
                        ),
                    )
                window._install_project(
                    result.document,
                    result.path,
                    prepared_history=result.history,
                    prepared_reuse=reuse,
                    readiness=result.readiness,
                )
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
                window._error(f"Cannot restore project view: {exc}")

    def new_project(self, kind=None, *, name="New coil design", project=None, discard_checked=False):
        """Create an unconfigured document; explicit legacy kinds load templates."""
        window = self.window
        from .project_start import blank_project

        if kind is not None:
            return window.load_template(kind)
        if not discard_checked and not window._can_discard():
            return None
        document = project if project is not None else blank_project(name)
        window._clear_project()
        window._set_assembly(ProjectAssembly.new(document.name, (document,)))
        window.undo_stack.setClean()
        window._inspect_node("geometry")
        window.statusBar().showMessage("New project · choose a support and target field", 5000)
        return document

    def load_template(self, kind="cylinder"):
        """Load configured inputs ready to compute; settings remain optional."""
        window = self.window
        template = template_spec(kind)
        if not window._can_discard():
            return
        try:
            assembly = template_assembly(kind)
        except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
            window._error(f"Cannot load template: {exc}")
            return
        window._discard_project_editors()
        window.undo_stack.clear()
        window._draft_dirty = False
        window.project_path = None
        window._history_key = None
        window._panel_views.clear()
        window._collection_mode = len(assembly.channels) > 1
        window._set_assembly(assembly)
        window.messages.appendPlainText(template.description)
        window.select_node(f"stage:{window.project.study['pipeline_stage']}")
        window.findChild(QDockWidget, "settings_dock").hide()
        window.undo_stack.setClean()
        window.statusBar().showMessage(
            f"Loaded template: {template.title} · Settings loaded. Click Compute full design; editing is optional.",
            10000,
        )
        return assembly

    def export_python(self, path, *, compact=False, include_results=False, standalone=False):
        """Export a script plus owned project assets, avoiding duplicated inputs."""
        window = self.window
        if not window._apply_document_inputs():
            raise ValueError("Correct the pending settings before exporting.")
        with window._visible_work("Exporting project and Python inputs"):
            project = (
                window._portable_assembly().channel(window._active_channel_id).project
                if include_results
                else window.project.inputs_only()
            )
            options = {"compact": compact, "include_results": include_results} if compact or include_results else {}
            if standalone:
                options["standalone"] = True
            return project.export_python(path, **options)

    def export_assembly_python(self, path, *, compact=False, include_results=False):
        window = self.window
        if not window._apply_document_inputs():
            raise ValueError("Correct pending settings before exporting.")
        with window._visible_work("Exporting channel system and Python inputs"):
            path = Path(path)
            project_path = path.with_suffix(".coilgen" if compact else ".coilgen-assembly.json")
            assembly = (
                window._portable_assembly()
                if include_results
                else replace(
                    window.assembly,
                    channels=tuple(
                        replace(channel, project=channel.project.inputs_only()) for channel in window.assembly.channels
                    ),
                )
            )
            assembly.save(project_path, **({"progress": window._portable_progress} if compact else {}))
            source = (
                '"""Independent CoilGen2 channels; coupling and assembly checks require separate evaluation."""\n'
                f"# Generated with CoilGen2 - version {SOFTWARE_VERSION}.\n"
                f"# Software maintainer: {AUTHOR_NAME}.\n"
                f"# Project: {PROJECT_URL}\n"
                "from pathlib import Path\n"
                "from coilgen.application import DesignService, ProjectAssembly\n\n"
                "if __name__ == '__main__':\n"
                f"    project = ProjectAssembly.load(Path(__file__).with_name({project_path.name!r}))\n"
                "    results = DesignService.run_assembly(project, Path(__file__).with_suffix('') / 'runs')\n"
                "    print(results)\n"
            )
            atomic_text(path, source)
            return path
