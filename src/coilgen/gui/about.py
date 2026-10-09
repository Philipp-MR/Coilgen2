"""Software credits, public project contact, and supplied imprint details."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QTabWidget, QVBoxLayout, QWidget

from ..attribution import (
    AUTHOR_CREDIT,
    AUTHOR_NAME,
    AUTHOR_PROFILE_URL,
    AUTHOR_ROLE,
    COILGEN_CITATION,
    COILGEN_PUBLICATION_URL,
    CONTACT_EMAIL,
    PROJECT_ISSUES_URL,
    PROJECT_URL,
    SOFTWARE_NAME,
)


class AboutDialog(QDialog):
    def __init__(self, parent=None, *, page="about"):
        super().__init__(parent)
        self.setWindowTitle(f"About {SOFTWARE_NAME}")
        self.resize(580, 420)
        layout = QVBoxLayout(self)
        heading = QLabel(SOFTWARE_NAME)
        heading.setStyleSheet("font-size: 24px; font-weight: 600;")
        layout.addWidget(heading)
        layout.addWidget(QLabel("Magnetostatic coil design and field analysis."))
        self.tabs = QTabWidget()
        self.tabs.setObjectName("software_information_tabs")
        self.tabs.setAccessibleName("CoilGen2 information pages")
        layout.addWidget(self.tabs)
        about = self._page("about", "About")
        self._text(about, f"{AUTHOR_CREDIT}\n{AUTHOR_ROLE}", "software_author_credit")
        self._link(about, "CoilGen2 source and project", PROJECT_URL, "software_repository")
        about.addSpacing(12)
        about.addWidget(QLabel("Original CoilGen publication"))
        self._text(about, COILGEN_CITATION, "coilgen_citation")
        self._link(about, "Read the publication", COILGEN_PUBLICATION_URL, "coilgen_publication")
        about.addStretch(1)
        contact = self._page("contact", "Contact / Kontakt")
        self._text(contact, f"Software creator and maintainer\n{AUTHOR_NAME}\n{AUTHOR_ROLE}", "software_contact_name")
        self._text(
            contact,
            "Project contact is available by email and on GitHub. Use the issue tracker for bug reports and feature requests.",
        )
        self._link(contact, CONTACT_EMAIL, f"mailto:{CONTACT_EMAIL}", "contact_email")
        self._link(contact, "CoilGen2 on GitHub", PROJECT_URL, "contact_repository")
        self._link(contact, "Report an issue or request a feature", PROJECT_ISSUES_URL, "contact_issues")
        self._link(contact, f"{AUTHOR_NAME} on GitHub", AUTHOR_PROFILE_URL, "contact_profile")
        contact.addStretch(1)
        imprint = self._page("impressum", "Impressum")
        self._text(imprint, "Project information / Impressum — project notice", "impressum_status")
        self._text(
            imprint, f"{AUTHOR_NAME}\n{AUTHOR_ROLE}\nCreator and maintainer of {SOFTWARE_NAME}.", "impressum_name"
        )
        self._text(
            imprint,
            "Public contact: email and GitHub. Postal contact details are not published here.",
            "impressum_contact_policy",
        )
        self._link(imprint, CONTACT_EMAIL, f"mailto:{CONTACT_EMAIL}", "impressum_email")
        self._link(imprint, "Public project contact on GitHub", PROJECT_ISSUES_URL, "impressum_project_contact")
        imprint.addStretch(1)
        pages = {self.tabs.widget(index).objectName(): index for index in range(self.tabs.count())}
        if page not in pages:
            raise ValueError(f"Unknown information page: {page}")
        self.tabs.setCurrentIndex(pages[page])
        self.tabs.currentChanged.connect(self._update_title)
        self._update_title(self.tabs.currentIndex())
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_title(self, index):
        title = self.tabs.tabText(index)
        self.setWindowTitle(f"About {SOFTWARE_NAME}" if title == "About" else f"{title} — {SOFTWARE_NAME}")

    def _page(self, name, title):
        widget = QWidget()
        widget.setObjectName(name)
        self.tabs.addTab(widget, title)
        return QVBoxLayout(widget)

    @staticmethod
    def _text(layout, text, name=None):
        label = QLabel(text)
        if name:
            label.setObjectName(name)
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(label)

    @staticmethod
    def _link(layout, text, url, name):
        label = QLabel(f'<a href="{url}">{text}</a>')
        label.setObjectName(name)
        label.setOpenExternalLinks(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        layout.addWidget(label)
