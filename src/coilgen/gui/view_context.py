"""Keep the editable model and the actual displayed scientific source separate.

This is presentation of saved provenance, not an integrity or acceptance check.
Cached display capabilities avoid repeated validation during camera/control
edits; standalone callers validate arrays without copying them. A selected run
never supplies the identity of a model-preview dataset.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import QGridLayout, QLabel, QSizePolicy, QToolButton, QWidget

from .colors import apply_tone
from .icons import icon
from .job_history import format_job_timestamp
from .dataset_capabilities import FIELD_KEYS as _FIELD_KEYS, PATH_PREFIXES, resolve_dataset


@dataclass(frozen=True, slots=True)
class ViewContext:
    editing_text: str
    viewing_text: str
    editing_tooltip: str
    viewing_tooltip: str
    source: str
    dataset_key: str
    identity: str
    tone: str
    available: bool | None
    sampling: str = ""
    current_A: float | None = None


_DATASETS = {
    "support": "Support preview",
    "preview_target": "Support + target preview",
    "stream_function": "Stream function",
    "surface_current_density": "Surface current density |K| [A/m]",
    "contours": "Contours",
    "routes": "Connected routes",
    "evaluated_winding": "Evaluated winding paths",
    "fabrication": "Pipeline conductor",
    "swept": "Pipeline swept conductor",
    "target": "Target field",
    "sheet": "Continuous-sheet field",
    "validation": "Independent sheet validation",
    "contour_field": "Contour wire field",
    "contour_residual": "Contour residual",
    "routed_field": "Winding field",
    "routed_residual": "Winding residual",
    "review_conductor": "Review conductor",
    "review_electrical": "Review electrical paths",
    "fabricated": "Review field",
    "winding_validation": "Independent winding validation",
    "winding_validation_residual": "Independent winding residual",
}


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _text(value, default=""):
    return value.strip() if isinstance(value, str) and value.strip() else default


def _revision(value):
    return (
        f"revision {value}"
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else "revision unrecorded"
    )


def _number(value):
    if not isinstance(value, Real) or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (OverflowError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _shape(arrays, key):
    return getattr(arrays.get(key), "shape", ())


def _points(arrays, key):
    shape = _shape(arrays, key)
    return shape[0] if len(shape) == 2 and shape[1] == 3 and shape[0] > 0 else 0


def _identity(project, provenance, pending):
    owner = _text(provenance("project_id"))
    fingerprint = _text(provenance("project_fingerprint"))
    model_fingerprint = _text(provenance("project_model_fingerprint"))
    if owner and owner != getattr(project, "id", None):
        return "Another channel", "warning"
    if not owner or not fingerprint:
        return "Model identity unverified", "warning"
    if fingerprint == getattr(project, "fingerprint", None):
        return ("Applied revision · edits pending", "warning") if pending else ("Current applied model/study", "info")
    if model_fingerprint and model_fingerprint == getattr(project, "model_fingerprint", None):
        return "Current numerical model · study differs" + (" · edits pending" if pending else ""), "warning"
    return "Earlier model/study revision", "warning"


def _dataset_scope(arrays, metadata, key, original, capability=None):
    """Follow the exact array/current pair selected by ScientificViewport."""
    capability = capability if capability is not None else resolve_dataset(arrays, key, original=original)
    if key in _FIELD_KEYS:
        point_key = "validation_points_m" if key == "validation" else "target_points_m"
        if key == "fabricated" and "fabricated_points_m" in arrays:
            point_key = "fabricated_points_m"
        if key.startswith("winding_validation"):
            point_key = "winding_validation_points_m"
        count = _points(arrays, point_key)
        wire = key.startswith(("contour_", "routed_"))
        available = capability.available
        sampling = (
            "Independent samples" if key == "validation" or key.startswith("winding_validation") else "Design samples"
        )
        sampling += f" · {count} samples" if count else " · samples unavailable"
        current, current_text, detail = None, "", ""
        if wire:
            kind = "contour" if key.startswith("contour") else "routed"
            evidence = _mapping(_mapping(metadata.get("wire_fields")).get(kind))
            metric = _mapping(evidence.get("at_stream_current")) if original else evidence
            current = _number(metric.get("current_A"))
            policy = _text(metric.get("current_source"), "unrecorded")
            current_text = (
                "Original contour current"
                if original or metric.get("current_refitted") is False
                else "Saved refitted current"
                if metric.get("current_refitted") is True
                else "Saved current · policy unverified"
            )
            current_text += f" {current:.6g} A" if current is not None else " · value unrecorded"
            detail = f"Thin-wire field; current policy: {policy}. No independent winding acceptance."
            geometry = evidence.get("geometry_sources")
            if isinstance(geometry, (list, tuple)):
                sources = tuple(dict.fromkeys(_text(item) for item in geometry if _text(item)))
                if sources:
                    detail += " Geometry: " + ", ".join(sources) + "."
            if not metric or metric.get("available") is False:
                detail += " Saved metric scope is unavailable/unverified."
        elif key in {"sheet", "validation"}:
            current_text = "Solved sheet coefficients [A]"
            detail = "Continuous sheet; unchanged saved solution, without current refitting."
        elif key == "fabricated":
            current_source = _text(_mapping(metadata.get("fabricated_physics")).get("current_source"))
            current_text = (
                "Original per-body contour currents"
                if current_source == "original_contour_current_no_refit"
                else "Review current policy unrecorded"
            )
            current = _number(_mapping(metadata.get("summary")).get("current_a")) if current_source else None
            if current is not None:
                current_text += f" · recorded complete circuit {current:.6g} A"
            detail = "Fabrication-review electrical paths on design samples; independent winding evidence has its own dataset."
        elif key.startswith("winding_validation"):
            current_text = "Original signed per-body contour currents · no refit"
            detail = "Exact recorded electrical paths; independent target samples; thin-wire segment model. Display availability does not establish acceptance of additional contracts."
        else:
            detail = "Saved requested field, not a computed coil field."
        if not available:
            detail += " " + capability.explanation
        return available, sampling, current, current_text, detail
    if key == "surface_current_density":
        return (
            capability.available,
            "Saved mesh faces",
            None,
            "Solved sheet coefficients",
            "Recorded surface density [A/m]; copper volume current density and thermal acceptance require separate evidence.",
        )
    if key == "stream_function":
        return (
            capability.available,
            "Mesh vertices",
            None,
            "Solved sheet coefficients [A]",
            "Saved continuous-sheet solution." + (" " + capability.explanation if not capability.available else ""),
        )
    if key in PATH_PREFIXES or key == "swept":
        return (
            capability.available,
            "Geometry [m]",
            None,
            "",
            "Saved geometry; viewing it does not establish manufacturing acceptance."
            + (" " + capability.explanation if not capability.available else ""),
        )
    return False, "Sampling unverified", None, "", "Unknown dataset; no source or acceptance claim."


def build_view_context(
    project,
    record=None,
    *,
    channel_label="",
    dataset_key="support",
    pending=False,
    alias=None,
    state=None,
    wire_current="stored",
    sample_slice="",
    stage_preview=None,
    verification_error="",
    dataset_capability=None,
):
    """Present applied model identity separately from the selected plot's source.

    ``wire_current='original'`` selects the saved at-stream-current field and its
    own metadata; it never falls back to stored/refitted field evidence. ``state``
    may supplement missing execution provenance, but never saved field scopes.
    """
    if wire_current not in {"stored", "original"}:
        raise ValueError("wire_current must be 'stored' or 'original'.")
    model_name = _text(channel_label) or _text(getattr(project, "name", None), "Coil design")
    revision = _revision(getattr(project, "revision", None))
    editing = f"Editing: {model_name} · {revision}" + (" · changes pending" if pending else "")
    editing_tooltip = (
        f"Applied model: {model_name}; {revision}.\nProject/channel ID: {getattr(project, 'id', 'unrecorded')}."
        + ("\nPending inspector edits have not been applied to the model or preview." if pending else "")
    )
    dataset = _DATASETS.get(dataset_key, "Unknown dataset")
    if stage_preview is not None:
        metadata = _mapping(getattr(stage_preview, "metadata", None))
        arrays = _mapping(getattr(stage_preview, "arrays", None))
        identity, tone = _identity(project, metadata.get, pending)
        stage = _text(metadata.get("pipeline_stage"), "unrecorded")
        available, sampling, current, current_text, detail = _dataset_scope(
            arrays, metadata, dataset_key, wire_current == "original", dataset_capability
        )
        detail = detail.replace("Saved", "Recorded stage").replace("saved", "recorded stage")
        submitted = _revision(metadata.get("project_revision"))
        execution_status = _text(_mapping(state).get("status"), _text(metadata.get("execution_status")))
        prefix = (
            f"Run snapshot · {execution_status}" if execution_status in {"cancelled", "failed", "completed"} else "Live"
        )
        snapshot_description = (
            "Live display snapshot only." if prefix == "Live" else "Retained execution snapshot only."
        )
        return ViewContext(
            editing,
            f"Viewing: {prefix} {stage} preview · submitted {submitted} · {dataset}"
            + (" · dataset unavailable" if not available else ""),
            editing_tooltip,
            f"Completed stage boundary: {stage}; {submitted}.\n{identity}.\n"
            f"Project/channel ID: {metadata.get('project_id', 'unrecorded')}.\n"
            f"Exact request: {metadata.get('request_sha256', 'unrecorded')}.\n"
            f"{sampling}. {current_text}. {detail}\n"
            f"{snapshot_description} Final result integrity, independent field validation and "
            "manufacturing acceptance are not established by this preview."
            + (f"\nDisplayed sample selection: {sample_slice}; no interpolation." if sample_slice else ""),
            "live",
            dataset_key,
            identity,
            "warning" if not available else tone,
            available,
            sampling,
            current,
        )
    if dataset_key in {"support", "preview_target"}:
        available = dataset_capability.available if dataset_capability is not None else None
        reason = dataset_capability.explanation if dataset_capability is not None and not available else ""
        return ViewContext(
            editing,
            f"Viewing: {dataset} · applied {revision}" + (" · preview unavailable" if available is False else ""),
            editing_tooltip,
            "Current applied model preview. A selected saved run does not supply this dataset."
            + ("\n" + reason if reason else "")
            + ("\nApply pending edits to update the preview." if pending else ""),
            "model",
            dataset_key,
            "Applied model preview",
            "warning" if available is False else "info",
            available,
            "Model target samples" if dataset_key == "preview_target" else "Geometry [m]",
        )
    if record is None:
        return ViewContext(
            editing,
            f"Viewing: {dataset} · no saved run selected",
            editing_tooltip,
            "Select a saved run containing this dataset.",
            "run",
            dataset_key,
            "No run selected",
            "warning",
            False,
            "Sampling unavailable",
        )
    metadata = _mapping(getattr(record, "metadata", None))
    job, state = _mapping(metadata.get("job")), _mapping(state)

    def provenance(field):
        return next((source[field] for source in (metadata, job, state) if source.get(field) is not None), None)

    arrays = _mapping(getattr(record, "_arrays", None))
    if not arrays:
        arrays = _mapping(getattr(record, "arrays", None))
    identity, tone = _identity(project, provenance, pending)
    date = format_job_timestamp(provenance("created_at"))
    name = _text(alias) or _text(metadata.get("name"), "Saved run")
    saved_revision = _revision(provenance("project_revision"))
    owner = _text(provenance("project_id"), "unrecorded")
    recorded_channel = _text(provenance("channel_name")) or _text(provenance("channel_label"))
    # Matching ownership permits the current channel name as context, but never
    # lends that label to an unrelated or unidentified historical run.
    saved_channel = recorded_channel or (model_name if owner == getattr(project, "id", None) else "Channel unrecorded")
    available, sampling, current, current_text, scope = _dataset_scope(
        arrays, metadata, dataset_key, wire_current == "original", dataset_capability
    )
    date_text = date.text if date.timestamp is not None else "date unrecorded"
    viewing = f"Viewing: {name} · {date_text} · {saved_revision} · {identity} · {dataset}"
    if verification_error:
        viewing = "Viewing: Cached run · verification failed · " + viewing.removeprefix("Viewing: ")
        tone = "warning"
        scope += " Saved source verification failed: " + str(verification_error)
    viewing += f" · {sampling}"
    if current_text:
        viewing += " · " + current_text
    if not available:
        viewing += " · dataset unavailable"
        tone = "warning"
    tooltip = (
        (
            f"Cached arrays from previously loaded run: {name}\n"
            if verification_error
            else f"Immutable saved run: {name}\n"
        )
        + f"Channel: {saved_channel}; ID: {owner}\n{saved_revision}\n"
        f"{identity}\nExecution submitted: {date_text}\n{date.tooltip}\nDataset: {dataset}\n{sampling}"
        + ("\n" + current_text if current_text else "")
        + "\n"
        + scope
    )
    if sample_slice:
        tooltip += f"\nDisplayed sample selection: {sample_slice}; no interpolation."
    if not available:
        tooltip += "\nSelected scientific array pair/geometry is unavailable or invalid in this source."
    return ViewContext(
        editing, viewing, editing_tooltip, tooltip, "run", dataset_key, identity, tone, available, sampling, current
    )


class _ContextLabel(QLabel):
    """Elide visually while retaining full provenance for accessibility/tooltips."""

    def __init__(self, parent):
        super().__init__(parent)
        self._full_text = ""
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setTextFormat(Qt.TextFormat.PlainText)

    def set_full_text(self, text):
        self._full_text = text
        self.setAccessibleName(text)
        self._elide()

    def _elide(self):
        self.setText(self.fontMetrics().elidedText(self._full_text, Qt.TextElideMode.ElideRight, max(0, self.width())))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()


class ViewContextHeader(QWidget):
    """Compact two-row provenance; detailed history opens only when requested."""

    history_requested = Signal()
    model_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("view_context_header")
        self.context = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QGridLayout(self)
        layout.setContentsMargins(4, 3, 4, 3)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(2)
        self.editing_label = _ContextLabel(self)
        self.editing_label.setObjectName("view_context_editing")
        self.viewing_label = _ContextLabel(self)
        self.viewing_label.setObjectName("view_context_viewing")
        self.model_button = QToolButton(self)
        self.model_button.setText("Model")
        self.model_button.setIcon(icon("mesh"))
        self.model_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.model_button.setAccessibleName("View current applied model")
        self.model_button.setToolTip("Switch the plot to the current applied support and target preview.")
        self.model_button.clicked.connect(self.model_requested)
        self.model_button.hide()
        self.model_button.setEnabled(False)
        layout.addWidget(self.editing_label, 0, 0)
        layout.addWidget(self.model_button, 0, 1)
        layout.addWidget(self.viewing_label, 1, 0, 1, 2)
        layout.setColumnStretch(0, 1)

    def set_context(self, context: ViewContext):
        self.context = context
        self.editing_label.set_full_text(context.editing_text)
        self.viewing_label.set_full_text(context.viewing_text)
        self.editing_label.setToolTip(context.editing_tooltip)
        self.viewing_label.setToolTip(context.viewing_tooltip)
        self.editing_label.setAccessibleDescription(context.editing_tooltip)
        self.viewing_label.setAccessibleDescription(context.viewing_tooltip)
        viewing_saved_run = context.source in {"run", "live"} and context.identity != "No run selected"
        self.model_button.setVisible(viewing_saved_run)
        self.model_button.setEnabled(viewing_saved_run)
        apply_tone(self.editing_label, "warning" if "changes pending" in context.editing_text else "neutral")
        apply_tone(self.viewing_label, context.tone)

    def changeEvent(self, event):
        super().changeEvent(event)
        if (
            event.type() in {QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange}
            and getattr(self, "context", None) is not None
        ):
            self.set_context(self.context)
