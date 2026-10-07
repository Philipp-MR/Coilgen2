"""A live stage snapshot must describe its submitted model and exact arrays."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from coilgen.application.service import RunRecord
from coilgen.application.stage_preview import StagePreview
from coilgen.gui.view_context import ViewContextHeader, build_view_context


def _project():
    return SimpleNamespace(
        name="Conical Z2",
        id="channel-z2",
        revision=14,
        fingerprint="current-request",
        model_fingerprint="current-model",
    )


def _metadata(**changes):
    return {
        "pipeline_stage": "sheet",
        "completed_stages": ["prepared", "sheet"],
        "project_id": "channel-z2",
        "project_revision": 12,
        "project_fingerprint": "current-request",
        "project_model_fingerprint": "current-model",
        "request_sha256": "a" * 64,
        "source": "live_stage_preview",
        "live_stage_preview": True,
        "acceptance_claim": False,
        **changes,
    }


def _arrays():
    return {
        "vertices_m": np.zeros((3, 3)),
        "faces": np.array([[0, 1, 2]]),
        "target_points_m": np.zeros((4, 3)),
        "target_field_T": np.zeros((4, 3)),
        "sheet_field_T": np.zeros((4, 3)),
        "stream_function_A": np.zeros(3),
    }


def _preview(tmp_path, *, metadata=None, arrays=None):
    return StagePreview(
        tmp_path, _metadata() if metadata is None else metadata, _arrays() if arrays is None else arrays
    )


def test_live_source_submitted_revision_and_compatibility_remain_distinct_from_applied_model(tmp_path):
    preview = _preview(tmp_path)
    context = build_view_context(_project(), dataset_key="sheet", stage_preview=preview)
    assert context.source == "live"
    assert "Editing: Conical Z2 · revision 14" in context.editing_text
    assert "Live sheet preview · submitted revision 12" in context.viewing_text
    assert context.identity == "Current applied model/study"
    assert "Current applied model/study" in context.viewing_tooltip
    assert "Exact request: " + "a" * 64 in context.viewing_tooltip
    assert "Live display snapshot only" in context.viewing_tooltip
    assert "manufacturing acceptance are not established" in context.viewing_tooltip
    assert "Immutable saved run" not in context.viewing_tooltip
    assert "accepted" not in context.viewing_text.lower()
    assert context.available is True
    assert context.sampling == "Design samples · 4 samples"
    assert context.current_A is None


@pytest.mark.parametrize(
    "changes, identity, tone",
    [
        ({"project_id": "other-channel"}, "Another channel", "warning"),
        ({"project_id": None}, "Model identity unverified", "warning"),
        ({"project_fingerprint": None}, "Model identity unverified", "warning"),
        (
            {"project_fingerprint": "older", "project_model_fingerprint": "older"},
            "Earlier model/study revision",
            "warning",
        ),
        ({"project_fingerprint": "other-study"}, "Current numerical model · study differs", "warning"),
        ({}, "Current applied model/study", "info"),
    ],
)
def test_live_model_identity_uses_snapshot_provenance_only(tmp_path, changes, identity, tone):
    context = build_view_context(
        _project(), dataset_key="sheet", stage_preview=_preview(tmp_path, metadata=_metadata(**changes))
    )
    assert context.source == "live"
    assert context.identity == identity
    assert identity in context.viewing_tooltip
    assert context.tone == tone


def test_selected_history_and_execution_state_cannot_supply_missing_live_identity_or_arrays(tmp_path):
    saved = RunRecord(tmp_path / "saved", {**_metadata(), "name": "Historical shim"}, _arrays())
    preview = _preview(tmp_path / "live", metadata={"pipeline_stage": "prepared"}, arrays={})
    context = build_view_context(
        _project(),
        saved,
        dataset_key="sheet",
        stage_preview=preview,
        alias="Old alias",
        state={**_metadata(), "name": "Selected historical execution"},
    )
    assert context.source == "live"
    assert context.identity == "Model identity unverified"
    assert context.available is False
    assert context.current_A is None
    assert "revision unrecorded" in context.viewing_text
    assert "Historical" not in context.viewing_text
    assert "Old alias" not in context.viewing_text
    assert "channel-z2" not in context.viewing_tooltip
    assert "Exact request: unrecorded" in context.viewing_tooltip


@pytest.mark.parametrize(
    "key, arrays",
    [
        ("target", {"target_points_m": np.zeros((4, 3)), "target_field_T": np.zeros((4, 3))}),
        ("sheet", {"target_points_m": np.zeros((4, 3)), "sheet_field_T": np.zeros((4, 3))}),
        (
            "stream_function",
            {
                "vertices_m": np.zeros((3, 3)),
                "faces": np.array([[0, 1, 2]], dtype=int),
                "stream_function_A": np.zeros(3),
            },
        ),
        ("contours", {"contour_0_0_points_m": np.zeros((2, 3))}),
        ("routes", {"route_0_points_m": np.zeros((2, 3))}),
        ("fabrication", {"fabrication_0_points_m": np.zeros((2, 3))}),
        ("swept", {"swept_0_0_vertices_m": np.zeros((3, 3)), "swept_0_0_faces": np.array([[0, 1, 2]], dtype=int)}),
    ],
)
def test_live_dataset_availability_is_exact_snapshot_geometry_or_array_pair(tmp_path, key, arrays):
    preview = _preview(tmp_path, arrays=arrays)
    before = {name: values.copy() for name, values in preview.arrays.items()}
    context = build_view_context(_project(), dataset_key=key, stage_preview=preview)
    assert context.source == "live"
    assert context.dataset_key == key
    assert context.available is True
    assert "Live display snapshot only" in context.viewing_tooltip
    assert "Immutable saved run" not in context.viewing_tooltip
    for name, values in preview.arrays.items():
        assert values.flags.writeable is False
        np.testing.assert_array_equal(values, before[name])


@pytest.mark.parametrize(
    "key, arrays",
    [
        ("sheet", {"target_points_m": np.zeros((4, 3))}),
        ("target", {"target_points_m": np.zeros((4, 3)), "target_field_T": np.zeros((3, 3))}),
        (
            "stream_function",
            {
                "vertices_m": np.zeros((3, 3)),
                "faces": np.array([[0, 1, 2]], dtype=int),
                "stream_function_A": np.zeros(2),
            },
        ),
        ("contours", {"contour_0_0_points_m": np.zeros((1, 3))}),
        ("swept", {"swept_0_0_vertices_m": np.zeros((3, 3))}),
        ("unknown", _arrays()),
    ],
)
def test_unavailable_live_dataset_is_visible_as_unavailable_and_warning(tmp_path, key, arrays):
    context = build_view_context(_project(), dataset_key=key, stage_preview=_preview(tmp_path, arrays=arrays))
    assert context.available is False
    assert "dataset unavailable" in context.viewing_text
    assert context.tone == "warning"


def test_pending_edits_do_not_relabel_live_applied_snapshot(tmp_path):
    context = build_view_context(_project(), dataset_key="sheet", stage_preview=_preview(tmp_path), pending=True)
    assert "revision 14 · changes pending" in context.editing_text
    assert "submitted revision 12" in context.viewing_text
    assert context.identity == "Applied revision · edits pending"
    assert "Pending inspector edits have not been applied" in context.editing_tooltip
    assert context.tone == "warning"


@pytest.mark.parametrize("status", ["cancelled", "failed", "completed"])
def test_terminal_execution_labels_retained_stage_as_snapshot(tmp_path, status):
    context = build_view_context(
        _project(), dataset_key="sheet", stage_preview=_preview(tmp_path), state={"status": status}
    )
    assert f"Run snapshot · {status}" in context.viewing_text
    assert "Viewing: Live" not in context.viewing_text
    assert context.source == "live"
    assert context.available is True


def test_live_scopes_do_not_describe_snapshot_arrays_as_saved_run_evidence(tmp_path):
    for key in ("stream_function", "contours"):
        arrays = {**_arrays(), "contour_0_0_points_m": np.zeros((2, 3))}
        context = build_view_context(_project(), dataset_key=key, stage_preview=_preview(tmp_path, arrays=arrays))
        assert "Saved geometry" not in context.viewing_tooltip
        assert "Saved continuous-sheet solution" not in context.viewing_tooltip


def test_live_sample_slab_is_described_without_interpolation(tmp_path):
    context = build_view_context(
        _project(), dataset_key="sheet", stage_preview=_preview(tmp_path), sample_slice="z ≈ 0 m · 2 samples"
    )
    assert "Displayed sample selection: z ≈ 0 m · 2 samples; no interpolation" in context.viewing_tooltip


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication(["live-view-context-test"])


@pytest.fixture
def header(app):
    widget = ViewContextHeader()
    widget.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    widget.resize(520, 65)
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize("arrays", [_arrays(), {}])
def test_live_header_offers_model_return_even_for_unavailable_live_dataset(header, app, tmp_path, arrays):
    context = build_view_context(_project(), dataset_key="sheet", stage_preview=_preview(tmp_path, arrays=arrays))
    requests = []
    header.model_requested.connect(lambda: requests.append(True))
    header.set_context(context)
    app.processEvents()
    assert header.model_button.isVisible()
    assert header.model_button.isEnabled()
    assert "support and target preview" in header.model_button.toolTip()
    assert header.viewing_label.accessibleName() == context.viewing_text
    assert header.viewing_label.toolTip() == context.viewing_tooltip
    assert requests == []
    QTest.mouseClick(header.model_button, Qt.MouseButton.LeftButton)
    assert requests == [True]


def test_model_return_disappears_after_return_to_applied_preview(header, tmp_path):
    header.set_context(build_view_context(_project(), dataset_key="sheet", stage_preview=_preview(tmp_path)))
    assert not header.model_button.isHidden()
    header.set_context(build_view_context(_project(), dataset_key="preview_target"))
    assert header.model_button.isHidden()
    assert not header.model_button.isEnabled()
