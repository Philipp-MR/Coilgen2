"""Scoped research evidence tables with explicit unavailable checks."""

import json

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)
from PySide6.QtCore import Qt


class SpatialReviewDialog(QDialog):
    def __init__(self, report, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spatial constraint evidence")
        layout = QVBoxLayout(self)
        self.report = report
        label = QLabel(
            f"Scope: {report['scope']}\nContinuous electrical segments · conservative conductor envelopes · feeds modeled: {report['feeds_modeled']}\n{report.get('coverage', 'Recorded world geometry')}\n{report['routing_policy']}"
        )
        label.setWordWrap(True)
        layout.addWidget(label)
        self.table = QTableWidget(len(report["rows"]), 8)
        self.table.setHorizontalHeaderLabels(
            [
                "Constraint / body",
                "Status",
                "Centerline / anchor [m]",
                "Required [m]",
                "Surface gap [m]",
                "Surface required [m]",
                "Witness segment",
                "Reason",
            ]
        )
        for index, row in enumerate(report["rows"]):
            solid = row.get("realized_conductor_surface", {})
            surface_gap = solid.get("minimum_surface_distance_m")
            if surface_gap is None and solid.get("certified_surface_distance_lower_bound_m") is not None:
                surface_gap = f"> {solid['certified_surface_distance_lower_bound_m']:.6g}"
            values = [
                row["name"],
                "Passed" if row["passed"] is True else "Failed" if row["passed"] is False else "Unavailable",
                row.get("minimum_centerline_surface_distance_m", row.get("distance_m")),
                row.get("required_envelope_gap_m", row.get("limit_m")),
                surface_gap,
                solid.get("required_surface_distance_m"),
                row.get("witness_segment"),
                row.get("reason", ""),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem("—" if value is None else str(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(index, column, item)
        self.table.resizeColumnsToContents()
        layout.addWidget(self.table)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.table.currentCellChanged.connect(self._selected)
        layout.addWidget(self.details)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(1100, 620)
        if report["rows"]:
            self.table.setCurrentCell(0, 0)

    def _selected(self, row, *_args):
        if row >= 0:
            self.details.setPlainText(json.dumps(self.report["rows"][row], indent=2, allow_nan=False))


class ResearchComparisonDialog(QDialog):
    def __init__(self, comparison, parent=None, *, source_fingerprint="Unavailable"):
        super().__init__(parent)
        self.setWindowTitle("Nominal and robust research comparison")
        layout = QVBoxLayout(self)
        identity = QLabel(
            f"Nominal: {comparison.left.metadata.get('name')} · revision {comparison.left.metadata.get('project_revision')} · {comparison.left.metadata.get('request_sha256', '')[:16]}\nRobust: {comparison.right.metadata.get('name')} · revision {comparison.right.metadata.get('project_revision')} · {comparison.right.metadata.get('request_sha256', '')[:16]}\nSource revision fingerprint: {source_fingerprint}"
        )
        identity.setWordWrap(True)
        identity.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(identity)
        notes = QLabel("\n".join(comparison.notes))
        notes.setWordWrap(True)
        layout.addWidget(notes)
        table = QTableWidget(len(comparison.metrics), 6)
        table.setHorizontalHeaderLabels(["Metric", "Unit", "Nominal", "Robust", "Common contract", "Scope / gap"])
        for row, metric in enumerate(comparison.metrics):
            for column, value in enumerate(
                [
                    metric.name,
                    metric.unit,
                    metric.left,
                    metric.right,
                    "Comparable" if metric.comparable else "Separate scopes",
                    metric.reason or "\n".join(dict.fromkeys([metric.left_scope, metric.right_scope])),
                ]
            ):
                item = QTableWidgetItem("Unavailable" if value is None else str(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(row, column, item)
        table.resizeColumnsToContents()
        layout.addWidget(table)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(1200, 760)
