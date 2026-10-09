"""Software credits and the original CoilGen publication."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout

from ..attribution import (
    AUTHOR_CREDIT,
    AUTHOR_ROLE,
    COILGEN_CITATION,
    COILGEN_PUBLICATION_URL,
    PROJECT_URL,
    SOFTWARE_NAME,
)


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"About {SOFTWARE_NAME}")
        self.resize(540, 340)
        layout = QVBoxLayout(self)
        heading = QLabel(SOFTWARE_NAME)
        heading.setStyleSheet("font-size: 24px; font-weight: 600;")
        layout.addWidget(heading)
        layout.addWidget(QLabel("Magnetostatic coil design and field analysis."))
        credits = QLabel(f"{AUTHOR_CREDIT}\n{AUTHOR_ROLE}")
        credits.setObjectName("software_author_credit")
        credits.setTextFormat(Qt.TextFormat.PlainText)
        credits.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(credits)
        repository = QLabel(f'<a href="{PROJECT_URL}">CoilGen2 source and project</a>')
        repository.setObjectName("software_repository")
        repository.setOpenExternalLinks(True)
        layout.addWidget(repository)
        layout.addSpacing(12)
        layout.addWidget(QLabel("Original CoilGen publication"))
        citation = QLabel(COILGEN_CITATION)
        citation.setObjectName("coilgen_citation")
        citation.setTextFormat(Qt.TextFormat.PlainText)
        citation.setWordWrap(True)
        citation.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(citation)
        publication = QLabel(f'<a href="{COILGEN_PUBLICATION_URL}">Read the publication</a>')
        publication.setObjectName("coilgen_publication")
        publication.setOpenExternalLinks(True)
        layout.addWidget(publication)
        layout.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
