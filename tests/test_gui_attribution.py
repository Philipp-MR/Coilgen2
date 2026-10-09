"""Software credits stay reachable without editing a user's coil design."""

import ast
from hashlib import sha256
from importlib.resources import files

import pytest

from test_workbench_usability import application as application, windows as windows

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDialogButtonBox, QLabel, QScrollArea

from coilgen.attribution import (
    AUTHOR_CREDIT,
    AUTHOR_NAME,
    AUTHOR_PROFILE_URL,
    AUTHOR_ROLE,
    COILGEN_CITATION,
    COILGEN_PUBLICATION_URL,
    CONTACT_EMAIL,
    PROJECT_ISSUES_URL,
    PROJECT_URL,
)
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


@pytest.mark.parametrize("page", ["about", "contact", "impressum"])
def test_information_pages_open_the_requested_tab_with_explicit_public_links(application, page):
    dialog = AboutDialog(page=page)
    assert dialog.tabs.currentWidget().objectName() == page
    assert dialog.tabs.count() == 3
    assert AUTHOR_NAME in dialog.findChild(QLabel, "software_contact_name").text()
    for name, url in (
        ("contact_repository", PROJECT_URL),
        ("contact_issues", PROJECT_ISSUES_URL),
        ("contact_profile", AUTHOR_PROFILE_URL),
        ("contact_email", f"mailto:{CONTACT_EMAIL}"),
        ("impressum_email", f"mailto:{CONTACT_EMAIL}"),
        ("impressum_project_contact", PROJECT_ISSUES_URL),
    ):
        label = dialog.findChild(QLabel, name)
        assert f'href="{url}"' in label.text()
        assert label.openExternalLinks()
        assert label.textInteractionFlags() & Qt.TextInteractionFlag.LinksAccessibleByKeyboard
    assert "project notice" in dialog.findChild(QLabel, "impressum_status").text()
    assert "not published here" in dialog.findChild(QLabel, "impressum_contact_policy").text()
    assert CONTACT_EMAIL == "pamrein11@mgh.harvard.edu"


def test_workbench_credits_and_about_action_leave_project_unchanged(windows, monkeypatch):
    window = windows()
    project, undo_count = window.project, window.undo_stack.count()
    called = []
    monkeypatch.setattr(
        AboutDialog,
        "exec",
        lambda dialog: called.append((dialog.windowTitle(), dialog.tabs.currentWidget().objectName())),
    )
    for action in (window.about_action, window.contact_action, window.impressum_action):
        action.trigger()
    assert called == [
        ("About CoilGen2", "about"),
        ("Contact / Kontakt — CoilGen2", "contact"),
        ("Impressum — CoilGen2", "impressum"),
    ]
    assert window.project is project
    assert window.undo_stack.count() == undo_count
    assert AUTHOR_NAME in window.welcome.findChild(QLabel, "welcome_author_credit").text()
    window._set_workspace_active(False)
    assert window.about_action.isEnabled()
    assert window.contact_action.isEnabled()
    assert window.impressum_action.isEnabled()


def test_assembly_script_attributes_software_and_preserves_design(windows, tmp_path):
    window = windows()
    project = window.project
    exported = window.export_assembly_python(tmp_path / "coil_channels.py")
    source = exported.read_text(encoding="utf-8")
    ast.parse(source)
    assert f"# CoilGen2 software: {AUTHOR_CREDIT} ({AUTHOR_ROLE})." in source
    assert window.project is project
    assert "DesignService.run_assembly" in source


def test_martinos_acknowledgment_uses_original_packaged_artwork_and_remains_reachable(application):
    image = files("coilgen.gui").joinpath("assets/martinos-logo.png").read_bytes()
    assert sha256(image).hexdigest() == "2b7e2302add9e453ec57f9afea3ef7f8dc7623558745655329a3cadc4f3617db"
    original = QPixmap()
    assert original.loadFromData(image)
    assert (original.width(), original.height()) == (505, 86)
    dialog = AboutDialog(page="impressum")
    try:
        logo = dialog.findChild(QLabel, "martinos_acknowledgment_logo")
        assert not logo.pixmap().isNull()
        assert logo.pixmap().width() / logo.pixmap().height() == pytest.approx(505 / 86, rel=0.02)
        assert "With gratitude" in dialog.findChild(QLabel, "martinos_gratitude").text()
        assert (
            "no institutional endorsement is implied"
            in dialog.findChild(QLabel, "martinos_acknowledgment_scope").text()
        )
        homepage = dialog.findChild(QLabel, "martinos_homepage")
        assert 'href="https://www.martinos.org/"' in homepage.text()
        assert homepage.openExternalLinks()
        dialog.resize(380, 300)
        dialog.show()
        application.processEvents()
        scroll = dialog.tabs.currentWidget()
        assert isinstance(scroll, QScrollArea)
        assert scroll.verticalScrollBar().maximum() > 0
        scroll.ensureWidgetVisible(homepage)
        application.processEvents()
        position = homepage.mapTo(scroll.viewport(), QPoint(0, 0))
        assert 0 <= position.y() <= scroll.viewport().height() - homepage.height()
    finally:
        dialog.close()
