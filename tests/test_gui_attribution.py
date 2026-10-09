"""Software credits stay reachable without editing a user's coil design."""

import ast

from test_workbench_usability import application as application, windows as windows

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialogButtonBox, QLabel

from coilgen.attribution import AUTHOR_CREDIT, AUTHOR_NAME, AUTHOR_ROLE, COILGEN_CITATION, COILGEN_PUBLICATION_URL
from coilgen.gui.about import AboutDialog


def test_about_dialog_preserves_full_citation_and_exposes_close(application):
    dialog = AboutDialog()
    credits = dialog.findChild(QLabel, "software_author_credit")
    assert credits.text() == f"{AUTHOR_CREDIT}\n{AUTHOR_ROLE}"
    assert credits.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
    assert dialog.findChild(QLabel, "coilgen_citation").text() == COILGEN_CITATION
    publication = dialog.findChild(QLabel, "coilgen_publication")
    assert COILGEN_PUBLICATION_URL in publication.text()
    assert publication.openExternalLinks()
    dialog.show()
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Close).click()
    assert not dialog.isVisible()


def test_workbench_credits_and_about_action_leave_project_unchanged(windows, monkeypatch):
    window = windows()
    project, undo_count = window.project, window.undo_stack.count()
    called = []
    monkeypatch.setattr(AboutDialog, "exec", lambda dialog: called.append(dialog.windowTitle()))
    window.about_action.trigger()
    assert called == ["About CoilGen2"]
    assert window.project is project
    assert window.undo_stack.count() == undo_count
    assert AUTHOR_NAME in window.welcome.findChild(QLabel, "welcome_author_credit").text()
    window._set_workspace_active(False)
    assert window.about_action.isEnabled()


def test_assembly_script_attributes_software_and_preserves_design(windows, tmp_path):
    window = windows()
    project = window.project
    exported = window.export_assembly_python(tmp_path / "coil_channels.py")
    source = exported.read_text(encoding="utf-8")
    ast.parse(source)
    assert f"# CoilGen2 software: {AUTHOR_CREDIT} ({AUTHOR_ROLE})." in source
    assert window.project is project
    assert "DesignService.run_assembly" in source
