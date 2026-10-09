"""Software credits, public project contact, and supplied imprint details."""

from importlib.resources import files

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFrame, QLabel, QScrollArea, QTabWidget, QVBoxLayout, QWidget

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
from .branding import LogoLabel


class AboutDialog(QDialog):
    def __init__(self, parent=None, *, page="about"):
        super().__init__(parent)
        self.setWindowTitle(f"About {SOFTWARE_NAME}")
        self.resize(580, 660)
        layout = QVBoxLayout(self)
        logo = LogoLabel(self, width=420)
        logo.setObjectName("about_logo")
        layout.addWidget(logo, alignment=Qt.AlignmentFlag.AlignHCenter)
        heading = QLabel(SOFTWARE_NAME)
        heading.setStyleSheet("font-size: 15px; font-weight: 600;")
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
        imprint.addSpacing(12)
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        imprint.addWidget(divider)
        self._text(imprint, "Acknowledgments", "acknowledgment_heading")
        logo = QLabel()
        logo.setObjectName("martinos_acknowledgment_logo")
        logo.setAccessibleName("Athinoula A. Martinos Center for Biomedical Imaging logo")
        logo.setStyleSheet("background-color: #152a40; padding: 12px; border-radius: 6px;")
        pixmap = QPixmap()
        pixmap.loadFromData(files("coilgen.gui").joinpath("assets/martinos-logo.png").read_bytes())
        logo.setPixmap(
            pixmap.scaled(
                QSize(280, 48), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        )
        imprint.addWidget(logo, alignment=Qt.AlignmentFlag.AlignLeft)
        self._text(
            imprint,
            "With gratitude to the Athinoula A. Martinos Center for Biomedical Imaging community.",
            "martinos_gratitude",
        )
        self._text(
            imprint,
            f"Personal acknowledgment by {AUTHOR_NAME}; no institutional endorsement is implied.",
            "martinos_acknowledgment_scope",
        )
        self._link(imprint, "Martinos Center website", "https://www.martinos.org/", "martinos_homepage")
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
        scroll = QScrollArea()
        scroll.setObjectName(name)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        widget = QWidget()
        scroll.setWidget(widget)
        self.tabs.addTab(scroll, title)
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
