"""The two adapters apply the same project/channel transactions and undo records."""

from coilgen.application.session import WorkbenchSession
from _coilgen_test_support.workbench_usability import application as application, windows as windows


def test_desktop_and_browser_edit_undo_redo_share_project_revisions(windows, tmp_path):
    window = windows()
    session = WorkbenchSession(tmp_path / "browser")
    session._set_assembly(window.assembly, record=False, channel_id=window._active_channel_id)
    original = window.project
    changes = {"levels": original.config.levels + 2}
    study = {"validation_resolution": 16}
    assert window.apply_inputs(changes, study)
    session.apply_inputs(changes, study)
    assert window.project.fingerprint == session.project.fingerprint
    assert window.project.revision == session.project.revision == original.revision + 1
    window.undo_stack.undo()
    assert session.undo()
    assert window.project is session.project is original
    window.undo_stack.redo()
    assert session.redo()
    assert window.project.fingerprint == session.project.fingerprint


def test_channel_duplication_removal_and_undo_preserve_independent_owners(windows, tmp_path):
    window = windows()
    session = WorkbenchSession(tmp_path / "browser")
    session._set_assembly(window.assembly, record=False, channel_id=window._active_channel_id)
    source = window.project
    browser_copy = session.duplicate_channel()
    native_copy = window.add_channel(browser_copy.name)
    assert browser_copy.id != source.id and native_copy.id != source.id
    assert window.project.values == session.project.values
    assert len(window.assembly.channels) == len(session.channels) == 2
    assert window.remove_active_channel()
    session.remove_channel()
    assert window.project is session.project is source
    window.undo_stack.undo()
    assert session.undo()
    assert window.project is native_copy and session.project is browser_copy
    window.undo_stack.undo()
    assert session.undo()
    assert window.project is session.project is source


def test_native_project_undo_preserves_other_channel_session_evidence(windows):
    window = windows()
    first_id = window._active_channel_id
    first = window.project
    window.add_channel("Second")
    original = window.project
    assert window.apply_inputs({"levels": original.config.levels + 2}, {})
    # Completed jobs/save operations may attach session evidence outside edit history.
    retained = first.with_results(first.saved_runs, session={"selected_job": "completed-run"})
    window.assembly = window.assembly.update_channel(first_id, project=retained)
    window.undo_stack.undo()
    assert window.project is original
    assert window.assembly.channel(first_id).project is retained
    window.undo_stack.redo()
    assert window.project.config.levels == original.config.levels + 2
    assert window.assembly.channel(first_id).project is retained
