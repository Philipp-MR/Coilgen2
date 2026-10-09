"""Selection of immutable conductor solids; export never changes the design."""

from copy import deepcopy
from dataclasses import dataclass, field
from inspect import Parameter, signature
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
from tempfile import TemporaryDirectory

from PySide6.QtCore import QEvent, QObject, QRunnable, QSignalBlocker, QSize, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .colors import apply_tone, item_foreground
from .icons import icon
from .run_context import run_compatibility


@dataclass
class ExportReceipt:
    directory: Path | None = None
    text: str = ""
    error: str = ""
    complete: bool = False


@dataclass
class GeometryExportState:
    names: tuple[str, ...] | None = None
    unit: str = "mm"
    include_project: bool = True
    receipt: ExportReceipt = field(default_factory=ExportReceipt)


def export_record_key(record):
    """Retain choices only for the same immutable source, including its identity."""
    if record is None:
        return None
    from ..research.storage import result_metadata_digest

    metadata = {key: value for key, value in record.metadata.items() if key != "job"}
    return (str(Path(record.directory).resolve()), result_metadata_digest(metadata))


class ExportReceiptPanel(QWidget):
    """A durable action result, separate from the saved run's engineering verdict."""

    retry_requested = Signal()

    def __init__(self, receipt=None, *, retry=False, parent=None):
        super().__init__(parent)
        self.receipt = receipt or ExportReceipt()
        self.setObjectName("export_receipt")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        self.status = QLabel()
        self.status.setObjectName("export_receipt_status")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)
        self.open_button = QPushButton("Open export folder")
        self.open_button.setObjectName("open_export_folder")
        self.open_button.setIcon(icon("open"))
        self.open_button.clicked.connect(self._open_folder)
        layout.addWidget(self.open_button)
        self.retry_button = QPushButton("Retry export…") if retry else None
        if self.retry_button is not None:
            self.retry_button.setObjectName("retry_export")
            self.retry_button.clicked.connect(self.retry_requested)
            layout.addWidget(self.retry_button)
        self.refresh()

    def set_pending(self, directory, text):
        self.receipt.directory = Path(directory).expanduser().resolve()
        self.receipt.text, self.receipt.error, self.receipt.complete = text, "", False
        self.refresh()

    def set_finished(self, directory, text):
        self.receipt.directory = Path(directory).expanduser().resolve()
        self.receipt.text, self.receipt.error, self.receipt.complete = text, "", True
        self.refresh()

    def set_failed(self, error, *, directory=None):
        if directory is not None:
            self.receipt.directory = Path(directory).expanduser().resolve()
        self.receipt.text, self.receipt.error, self.receipt.complete = "", str(error), False
        self.refresh()

    def refresh(self):
        result = self.receipt
        text = "Export failed: " + result.error if result.error else result.text
        if result.directory is not None:
            text += "\nDestination: " + str(result.directory)
        self.status.setText(text)
        apply_tone(self.status, "error" if result.error else "success" if result.complete else "info")
        self.open_button.setVisible(result.complete)
        if self.retry_button is not None:
            self.retry_button.setVisible(bool(result.error))
        self.setVisible(bool(result.text or result.error))

    def _open_folder(self):
        if self.receipt.complete and self.receipt.directory is not None:
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.receipt.directory))):
                self.status.setText(
                    self.status.text() + "\nCould not open the folder. Copy the destination path above."
                )
                apply_tone(self.status, "warning")


def geometry_export_description(package):
    """Describe the committed manifest's actual units, rather than requested units."""
    manifest = getattr(package, "manifest", {})
    geometry = manifest.get("geometry", ())
    formats = tuple(dict.fromkeys(f"{item['format'].upper()} [{item['actual_unit']}]" for item in geometry))
    count = len(package.paths)
    text = f"Exported {count} solid {'file' if count == 1 else 'files'}"
    text += " · " + ", ".join(formats) if formats else ""
    if manifest.get("project_companion"):
        text += " · project JSON and results included"
    return text + f". Units and provenance: {package.manifest_path.name}."


class GeometryExportPanel(QWidget):
    export_requested = Signal(object, str, bool)
    review_requested = Signal()
    problems_requested = Signal()

    def __init__(
        self,
        record=None,
        options=(),
        *,
        diagnostic="",
        parent=None,
        project=None,
        assembly=None,
        pending=False,
        state=None,
    ):
        super().__init__(parent)
        self._has_record = record is not None
        self._record = record
        self.record_key = export_record_key(record)
        self.state = state if state is not None else GeometryExportState()
        self._diagnostic = str(diagnostic)
        self._exporting = False
        self._review_allowed = False
        self.options = tuple(options)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.scope = QLabel()
        self.scope.setWordWrap(True)
        self.scope.setTextFormat(Qt.TextFormat.PlainText)
        self.scope.setObjectName("geometry_export_scope")
        layout.addWidget(self.scope)
        metadata = record.metadata if record is not None else {}
        summary = metadata.get("summary") or {}
        review = metadata.get("fabrication") or {}
        revision = metadata.get("project_revision", (metadata.get("job") or {}).get("project_revision", "unverified"))
        if revision is None or (isinstance(revision, str) and not revision.strip()):
            revision = "unverified"
        if record is None:
            self.scope.setText("No saved run selected. Compute a design and review its conductor geometry first.")
        else:
            self.scope.setText(
                f"Saved run: {metadata.get('name', record.directory.name)} · revision {revision}. "
                "Exports use this run's exact solid geometry. Geometry availability, field acceptance and construction release are separate."
            )
        self.status = QLabel(diagnostic)
        self.status.setObjectName("geometry_export_status")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        self.table = QTableWidget(0, 4)
        self.table.setObjectName("geometry_export_bodies")
        self.table.setHorizontalHeaderLabels(["Conductor", "Units", "Checks", "Extent [mm]"])
        for column, tooltip in enumerate(
            (
                "Saved conductor body and file format. Hover over a body for its exact artifact name.",
                "STL uses the selected coordinate units; STEP retains its declared source units.",
                "Recorded geometry checks within the saved scope; this is not field acceptance or construction release.",
                "Physical x, y and z dimensions in millimetres, independent of the exported file units.",
            )
        ):
            self.table.horizontalHeaderItem(column).setToolTip(tooltip)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3):
            self.table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setMinimumHeight(110)
        self.table.setMaximumHeight(160)
        self.units = QComboBox()
        self.units.setObjectName("geometry_export_units")
        self.units.setAccessibleName("STL coordinate units")
        self.units.addItem("STL coordinates: millimetres [mm]", "mm")
        self.units.addItem("STL coordinates: metres [m]", "m")
        self.units.setCurrentIndex(max(0, self.units.findData(self.state.unit)))
        self.units.setToolTip("STL has no unit declaration. Scaling is explicit and recorded in the CAD manifest.")
        self.unit_note = QLabel(
            "STL coordinates are scaled from the saved SI geometry to the selected units. "
            "STEP files retain their existing declared units. Extents below are physical dimensions in millimetres."
        )
        self.unit_note.setWordWrap(True)
        self.bounds = QLabel()
        self.bounds.setObjectName("geometry_export_bounds")
        self.bounds.setWordWrap(True)
        self.bounds.setTextFormat(Qt.TextFormat.PlainText)
        self.include_project = QCheckBox("Include project JSON and results")
        self.include_project.setIcon(icon("project-json"))
        self.include_project.setIconSize(QSize(18, 18))
        self.include_project.setAccessibleName(self.include_project.text())
        self.include_project.setChecked(self.state.include_project)
        self.include_project.setObjectName("geometry_export_include_project")
        self.include_project.setToolTip(
            "Save one self-contained project JSON containing model inputs and completed results with the CAD export."
        )
        self.companion_context = QLabel()
        self.companion_context.setObjectName("geometry_export_companion_context")
        self.companion_context.setTextFormat(Qt.TextFormat.PlainText)
        self.companion_context.setWordWrap(True)
        self.include_project.toggled.connect(
            lambda checked: self.companion_context.setVisible(checked and bool(self.companion_context.text()))
        )
        self.export_button = QPushButton("Export selected geometry…")
        self.export_button.setObjectName("export_selected_geometry")
        self.export_button.clicked.connect(self._request_export)
        self.review_button = QPushButton("Review current revision")
        self.review_button.setObjectName("export_run_review")
        self.review_button.clicked.connect(self.review_requested)
        self.review_scope = QLabel()
        self.review_scope.setObjectName("geometry_export_review_scope")
        self.review_scope.setTextFormat(Qt.TextFormat.PlainText)
        self.review_scope.setWordWrap(True)
        self.problems_button = QPushButton("Inspect geometry problems")
        self.problems_button.setObjectName("export_inspect_problems")
        self.problems_button.clicked.connect(self.problems_requested)
        for button, name in (
            (self.export_button, "export-conductor"),
            (self.review_button, "constraint"),
            (self.problems_button, "collision"),
        ):
            button.setIcon(icon(name))
            button.setIconSize(QSize(20, 20))
            button.setAccessibleName(button.text())
        if not self.options:
            if not diagnostic:
                self.status.setText(
                    "No solid STL or solid STEP was saved. "
                    + (
                        "Recorded fabrication checks failed; inspect the problems before recomputing."
                        if review.get("ready") is False
                        else "Review the current revision to produce reviewed conductor solids."
                    )
                )
            apply_tone(self.status, "error" if diagnostic or review.get("ready") is False else "warning")
        else:
            failed = (
                review.get("ready") is False
                or any(option.check_status == "failed" for option in self.options)
                or summary.get("sheet_accepted") is False
            )
            self.status.setText(
                f"{len(self.options)} saved solid files available. "
                + (
                    "Recorded checks failed; these files remain available as inspection geometry."
                    if failed
                    else "Some files are inspection geometry with unverified checks."
                    if any(option.inspection_only for option in self.options)
                    else "Inspect the separate field and circuit checks before using the geometry."
                )
            )
            apply_tone(
                self.status,
                "error"
                if failed
                else "warning"
                if any(option.inspection_only for option in self.options)
                else "success",
            )
        self.problems_button.setVisible(review.get("ready") is False or bool(review.get("failure_reasons")))
        self.review_button.setVisible(record is not None and not self.options and review.get("ready") is not False)
        self.review_scope.setVisible(not self.review_button.isHidden())
        if self._diagnostic:
            self.status.setText(self._diagnostic)
            apply_tone(self.status, "error")
        layout.addWidget(self.problems_button)
        layout.addWidget(self.review_button)
        layout.addWidget(self.review_scope)
        layout.addWidget(self.units)
        layout.addWidget(self.include_project)
        layout.addWidget(self.companion_context)
        layout.addWidget(self.export_button)
        self.receipt = ExportReceiptPanel(self.state.receipt)
        layout.addWidget(self.receipt)
        layout.addWidget(self.table)
        layout.addWidget(self.bounds)
        layout.addWidget(self.unit_note)
        for control in (self.units, self.include_project, self.table, self.bounds, self.unit_note):
            control.setVisible(bool(self.options))
        release = QLabel(
            "Construction release: " + ("Released" if summary.get("construction_released") is True else "Not released")
        )
        release.setObjectName("geometry_export_release")
        layout.addWidget(release)
        with QSignalBlocker(self.table):
            self.table.setRowCount(len(self.options))
            for row, option in enumerate(self.options):
                name = QTableWidgetItem(f"{option.body_id} · {option.format.upper()}")
                name.setIcon(icon("conductor-sweep"))
                name.setData(Qt.ItemDataRole.UserRole, option.name)
                name.setFlags(name.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                selected = option.name in self.state.names if self.state.names is not None else option.format == "stl"
                name.setCheckState(Qt.CheckState.Checked if selected else Qt.CheckState.Unchecked)
                name.setToolTip(option.name)
                self.table.setItem(row, 0, name)
                self.table.setItem(row, 1, QTableWidgetItem())
                status = QTableWidgetItem(
                    ("Inspection · " if option.inspection_only else "") + option.check_status.capitalize()
                )
                status.setForeground(
                    item_foreground(
                        "success"
                        if option.check_status == "passed"
                        else "error"
                        if option.check_status == "failed"
                        else "neutral",
                        self.palette(),
                    )
                )
                status.setToolTip(option.scope)
                status.setIcon(
                    icon(
                        "constraint",
                        badge={"passed": "passed", "failed": "failed"}.get(option.check_status, "unverified"),
                    )
                )
                self.table.setItem(row, 2, status)
                extent = [1000 * (hi - lo) for lo, hi in zip(*option.bounds_m)]
                self.table.setItem(row, 3, QTableWidgetItem(" × ".join(f"{value:.5g}" for value in extent)))
        self.table.itemChanged.connect(self._selection_changed)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.currentCellChanged.connect(self._selection_changed)
        self.units.currentIndexChanged.connect(self._selection_changed)
        self.include_project.toggled.connect(self._selection_changed)
        self._selection_changed()
        self.set_context(project, assembly=assembly, pending=pending)

    def set_context(self, project=None, *, assembly=None, pending=False):
        """Distinguish immutable CAD source from the current project companion."""
        metadata = self._record.metadata if self._record is not None else {}
        owner = metadata.get("project_id") or (metadata.get("job") or {}).get("project_id")
        matching_owner = project is not None and owner is not None and owner == project.id
        self._review_allowed = matching_owner and not self._diagnostic
        self.review_button.setEnabled(self._review_allowed and not self._exporting)
        review_text = (
            self._diagnostic
            if self._diagnostic
            else f"Review computes the current channel, applied revision {project.revision}. "
            "Pending edits are applied first; the selected saved run remains immutable."
            if matching_owner
            else "Open the saved run's owning channel to review its current revision."
        )
        self.review_scope.setText(review_text)
        self.review_button.setToolTip(review_text)
        if self._record is None or project is None and assembly is None:
            self.companion_context.clear()
            self.companion_context.hide()
            return
        current = (
            next((channel.project for channel in assembly.channels if channel.project.id == owner), None)
            if assembly is not None
            else project
        )
        current = current or project
        revision = metadata.get("project_revision", (metadata.get("job") or {}).get("project_revision", "unrecorded"))
        context, label = run_compatibility(metadata, current)
        current_revision = current.revision if current is not None else "unverified"
        companion = (
            f"Current assembly ({len(assembly.channels)} channels)" if assembly is not None else "Current project"
        )
        text = f"CAD source: saved revision {revision}. {companion} companion: applied revision {current_revision} · {label.lower()}."
        if pending:
            text += " Pending edits will be applied before saving the companion."
        if context == "earlier":
            text += " The project contains the current settings and saved earlier result; CAD remains the selected run's geometry."
        self.companion_context.setText(text)
        self.companion_context.setToolTip(text)
        apply_tone(self.companion_context, "warning" if context != "current" or pending else "info")
        self.companion_context.setVisible(self.include_project.isChecked())

    def selected_names(self):
        return tuple(
            self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            for row in range(self.table.rowCount())
            if self.table.item(row, 0).checkState() == Qt.CheckState.Checked
        )

    def _selection_changed(self, *_args):
        with QSignalBlocker(self.table):
            for row, option in enumerate(self.options):
                self.table.item(row, 1).setText(
                    f"STL · {self.units.currentData()}" if option.format == "stl" else f"STEP · {option.source_unit}"
                )
        names = self.selected_names()
        self.state.names = names
        self.state.unit = self.units.currentData()
        self.state.include_project = self.include_project.isChecked()
        self.export_button.setEnabled(self._has_record and bool(names) and not self._exporting and not self._diagnostic)
        self.export_button.setToolTip(
            self._diagnostic or "Export the selected saved solids and their recorded provenance."
        )
        self.export_button.setText(
            "Exporting saved geometry…"
            if self._exporting
            else f"Export {len(names)} selected solid files…"
            if names
            else "Select a solid to export"
            if self.options
            else "No saved solids"
        )
        self.export_button.setAccessibleName(self.export_button.text())
        row = self.table.currentRow()
        selected = (
            self.options[row]
            if 0 <= row < len(self.options)
            else next((option for option in self.options if option.name in names), None)
        )
        if selected is None:
            self.bounds.setText("Select the saved solid bodies and formats to export.")
        else:
            self.bounds.setText(
                f"{selected.body_id} · world bounds [mm]: "
                + "; ".join(
                    f"{axis} {lo * 1000:.5g}…{hi * 1000:.5g}" for axis, lo, hi in zip("xyz", *selected.bounds_m)
                )
            )

    def _request_export(self):
        names = self.selected_names()
        if self._has_record and names and not self._exporting and not self._diagnostic:
            self.export_requested.emit(names, self.units.currentData(), self.include_project.isChecked())

    def set_exporting(self, exporting):
        self._exporting = bool(exporting)
        self.review_button.setEnabled(self._review_allowed and not self._exporting)
        self._selection_changed()

    def _refresh_colors(self):
        apply_tone(self.status, self.status.property("semanticTone") or "neutral")
        with QSignalBlocker(self.table):
            for row, option in enumerate(self.options):
                item = self.table.item(row, 2)
                if item is not None:
                    tone = {"passed": "success", "failed": "error"}.get(option.check_status, "neutral")
                    item.setForeground(item_foreground(tone, self.table.palette()))

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange, QEvent.Type.StyleChange):
            if hasattr(self, "status") and hasattr(self, "table"):
                QTimer.singleShot(0, self, self._refresh_colors)


class _ExportSignals(QObject):
    progress = Signal(dict)
    finished = Signal(object, str)


class GeometryExportTask(QRunnable):
    """Write an already detached snapshot without calling any Qt-owned model."""

    def __init__(self, record, directory, names, unit, project=None):
        super().__init__()
        self.signals = _ExportSignals()
        self.record = _detached_record(record)
        self.record_key = export_record_key(record)
        self.directory, self.names, self.unit, self.project = directory, tuple(names), unit, project
        self.verification_error = ""
        self.feedback_state = None
        self.source_id = None

    def run(self):
        from ..application.cad import export_geometry
        from ..application.evidence import EvidenceIntegrityError
        from ..progress import progress_context
        from .design_tasks import ProgressSignalRelay

        relay = ProgressSignalRelay(self.signals.progress.emit)
        try:
            with progress_context(relay.report):
                package = export_geometry(self.record, self.directory, self.names, unit=self.unit, project=self.project)
        except Exception as exc:
            if isinstance(exc, EvidenceIntegrityError):
                self.verification_error = str(exc)
            relay.flush()
            self.signals.finished.emit(None, f"{type(exc).__name__}: {exc}")
        else:
            relay.flush()
            self.signals.finished.emit(package, "")


def _detached_record(record):
    from ..application.service import RunRecord

    # CAD and copies need metadata and verified bytes, without copying large
    # field maps out of the GUI's immutable selected result.
    return RunRecord(record.directory, record.metadata, {}) if isinstance(record, RunRecord) else deepcopy(record)


class ArtifactVerificationError(ValueError):
    """The selected immutable source failed its export-time integrity check."""


def export_verified_artifacts(record, directory, *, loader):
    """Stage, reverify and exclusively publish saved bytes without overwriting."""
    from ..persistence import file_digest
    from ..progress import ProgressReporter, progress_scope
    from ..research.storage import result_metadata_digest

    try:
        parameters = signature(loader).parameters
        selects_arrays = "array_names" in parameters or any(
            parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
    except (TypeError, ValueError):
        selects_arrays = False

    def verified():
        try:
            current = loader(record.directory, array_names=()) if selects_arrays else loader(record.directory)
            old = {key: value for key, value in record.metadata.items() if key != "job"}
            new = {key: value for key, value in current.metadata.items() if key != "job"}
            if result_metadata_digest(old) != result_metadata_digest(new):
                raise ValueError("Saved run metadata changed after selection. Reload the run.")
            return current
        except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
            raise ArtifactVerificationError(str(exc) or type(exc).__name__) from exc

    with progress_scope(0, 15, "Verifying saved export artifacts", label="Artifact export"):
        current = verified()
    directory = Path(directory).expanduser().resolve()
    source_directory = Path(current.directory).resolve()
    sources = []
    destinations = set()
    for artifact in current.metadata.get("artifacts", ()):
        name = artifact.get("name")
        if not isinstance(name, str) or not name or "\\" in name:
            raise ArtifactVerificationError("Saved artifact names must be portable relative paths.")
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or PureWindowsPath(name).drive:
            raise ArtifactVerificationError("Saved artifact path leaves its run directory.")
        source = (source_directory / relative).resolve()
        destination = (directory / relative).resolve()
        if not source.is_relative_to(source_directory) or not source.is_file():
            raise ArtifactVerificationError("Saved artifact must remain a file inside its run directory.")
        if not destination.is_relative_to(directory):
            raise ValueError("Export destination must stay inside the selected directory.")
        if destination in destinations:
            raise ArtifactVerificationError("Saved artifact names must be unique.")
        destinations.add(destination)
        if destination.exists() and destination != source:
            raise FileExistsError(f"Export would replace an existing file: {destination}")
        sources.append((source, destination, artifact.get("sha256")))
    if not sources:
        raise ValueError("The selected run has no saved artifacts to export.")
    if directory == source_directory:
        ProgressReporter(label="Artifact export").report(100, "verified artifacts already reside in this directory")
        return tuple(destination for _, destination, _ in sources)
    if directory.is_relative_to(source_directory):
        raise ValueError("Export artifacts outside the immutable saved run directory.")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".coilgen-artifacts-", dir=directory.parent) as temporary:
        stage = Path(temporary)
        staged = []
        with progress_scope(15, 55, "Staging and checking saved artifact files", label="Artifact export"):
            task = ProgressReporter(label="Artifact export").task("staged files", len(sources))
            for source, destination, digest in sources:
                path = stage / destination.relative_to(directory)
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, path)
                if not isinstance(digest, str) or file_digest(path) != digest:
                    raise ArtifactVerificationError(f"Saved artifact changed while being exported: {source.name}")
                staged.append((path, destination))
                task.advance()
            task.done()
        with progress_scope(55, 65, "Rechecking saved source evidence", label="Artifact export"):
            verified()
        created_files, created_directories = [], []
        try:
            with progress_scope(65, 100, "Publishing verified artifact files", label="Artifact export"):
                task = ProgressReporter(label="Artifact export").task("published files", len(staged))
                for path, destination in staged:
                    missing = []
                    parent = destination.parent
                    while not parent.exists():
                        missing.append(parent)
                        parent = parent.parent
                    for parent in reversed(missing):
                        parent.mkdir(exist_ok=False)
                        created_directories.append(parent)
                    os.link(path, destination)
                    created_files.append(destination)
                    task.advance()
                task.done()
        except BaseException:
            # Roll back only this transaction's checked destination paths;
            # unrelated files or a concurrent writer are never removed.
            for path in reversed(created_files):
                if path.resolve().is_relative_to(directory):
                    path.unlink(missing_ok=True)
            for path in reversed(created_directories):
                if path.resolve() == directory or path.resolve().is_relative_to(directory):
                    try:
                        path.rmdir()
                    except OSError:
                        pass
            raise
    return tuple(destination for _, destination, _ in sources)


class ArtifactExportTask(QRunnable):
    """Copy a detached saved source on a worker; callbacks own the GUI feedback."""

    def __init__(self, record, directory, *, loader):
        super().__init__()
        self.signals = _ExportSignals()
        self.record = _detached_record(record)
        self.directory = Path(directory).expanduser().resolve()
        self.loader = loader
        self.verification_error = ""

    def run(self):
        from ..progress import progress_context
        from .design_tasks import ProgressSignalRelay

        relay = ProgressSignalRelay(self.signals.progress.emit)
        try:
            with progress_context(relay.report):
                paths = export_verified_artifacts(self.record, self.directory, loader=self.loader)
        except Exception as exc:
            if isinstance(exc, ArtifactVerificationError):
                self.verification_error = str(exc)
            relay.flush()
            self.signals.finished.emit(None, f"{type(exc).__name__}: {exc}")
        else:
            relay.flush()
            self.signals.finished.emit(paths, "")
