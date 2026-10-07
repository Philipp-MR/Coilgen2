"""Channel status uses validated portable evidence before a channel is opened."""

import test_workbench_usability as usability

from coilgen.application import ProjectAssembly, ProjectDocument
from coilgen.application.bundles import RunArchive
from coilgen.application.presets import preset_config


application = usability.application
windows = usability.windows
prepared_run = usability.prepared_run


def test_overview_includes_unvisited_channel_archives_without_loading_their_arrays(windows, prepared_run):
    project, record = prepared_run
    archive = RunArchive.capture(record)
    saved = project.with_results((archive,))
    fresh = ProjectDocument.from_config(preset_config(), name="New channel")
    assembly = ProjectAssembly.new("Portable channels", (fresh, saved))
    window = windows(assembly)
    original, loaded = window.project, dict(window.runs)
    assert window.project.id == fresh.id
    assert not any(run.metadata.get("project_id") == saved.id for run in window.runs.values())

    window._channels_dialog()

    overview = window.channels_window
    assert overview.table.item(1, 7).text().startswith("Matching")
    assert overview.table.item(1, 6).text() == "Saved result only"
    assert "No saved result" in overview.table.item(0, 7).text()
    assert window.project is original and window.runs == loaded
    assert not window.service.submissions
    assert not window.jobs.states
    assert window.undo_stack.count() == 0
