"""A context header must describe the array displayed, not merely the selected run."""

import os
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QToolButton

from coilgen.application.service import RunRecord
from coilgen.gui.view_context import ViewContextHeader, build_view_context


def _project():
    return SimpleNamespace(name="Gy", id="channel-gy", revision=14, fingerprint="applied", model_fingerprint="model")


def _metadata():
    return {
        "name": "Gy study",
        "project_id": "channel-gy",
        "project_revision": 12,
        "project_fingerprint": "applied",
        "project_model_fingerprint": "model",
        "job": {"created_at": "2026-10-04T12:00:00+00:00"},
        "wire_fields": {
            "contour": {
                "available": True,
                "current_A": 21,
                "current_refitted": True,
                "current_source": "weighted_fit",
                "geometry_sources": ["oriented_unconnected_contours"],
                "at_stream_current": {
                    "available": True,
                    "current_A": 19.5,
                    "current_source": "original_contour_step",
                    "current_refitted": False,
                },
            },
            "routed": {
                "available": True,
                "current_A": 23,
                "current_refitted": False,
                "current_source": "stream_function",
                "geometry_sources": ["fabricated_centroid"],
                "at_stream_current": {"available": True, "current_A": 23, "current_refitted": False},
            },
        },
    }


def _arrays():
    points = np.zeros((3, 3))
    return {
        "target_points_m": points,
        "target_field_T": points,
        "sheet_field_T": points,
        "validation_points_m": np.zeros((7, 3)),
        "validation_field_T": np.zeros((7, 3)),
        "contour_field_T": points,
        "contour_field_at_stream_current_T": points,
        "routed_field_T": points,
        "routed_field_at_stream_current_T": points,
    }


def _record(tmp_path, *, metadata=None, arrays=None):
    return RunRecord(tmp_path, _metadata() if metadata is None else metadata, _arrays() if arrays is None else arrays)


@pytest.mark.parametrize("key", ["support", "preview_target"])
def test_model_preview_does_not_borrow_selected_run_identity(tmp_path, key):
    metadata = _metadata()
    metadata.update(project_id="another-channel", project_revision=0, name="Old unrelated run")
    context = build_view_context(
        _project(), _record(tmp_path, metadata=metadata), dataset_key=key, alias="Old alias", pending=True
    )
    assert context.source == "model"
    assert context.identity == "Applied model preview"
    assert "applied revision 14" in context.viewing_text
    assert "changes pending" in context.editing_text
    assert "Apply pending edits" in context.viewing_tooltip
    assert "Old" not in context.viewing_text
    assert "selected saved run does not supply" in context.viewing_tooltip


def test_pending_edits_stay_visible_when_selected_run_matches_applied_revision(tmp_path):
    context = build_view_context(_project(), _record(tmp_path), dataset_key="sheet", pending=True)
    assert context.identity == "Applied revision · edits pending"
    assert context.tone == "warning"
    assert "revision 14 · changes pending" in context.editing_text
    assert "revision 12" in context.viewing_text
    assert "edits pending" in context.viewing_text
    assert context.available is True


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"project_id": "other"}, "Another channel"),
        ({"project_id": None}, "Model identity unverified"),
        ({"project_fingerprint": None}, "Model identity unverified"),
        ({"project_fingerprint": "older", "project_model_fingerprint": "older"}, "Earlier model/study revision"),
        ({"project_fingerprint": "study-change"}, "Current numerical model · study differs"),
        ({}, "Current applied model/study"),
    ],
)
def test_full_ownership_and_fingerprint_provenance_governs_identity(tmp_path, overrides, expected):
    metadata = _metadata()
    metadata.update(overrides)
    context = build_view_context(_project(), _record(tmp_path, metadata=metadata), dataset_key="sheet")
    assert context.identity == expected
    assert expected in context.viewing_text
    if overrides.get("project_id") == "other":
        assert "Channel: Channel unrecorded; ID: other" in context.viewing_tooltip


def test_missing_run_and_unknown_dataset_have_no_validity_claim():
    context = build_view_context(_project(), dataset_key="sheet")
    assert context.source == "run"
    assert context.available is False
    assert context.identity == "No run selected"
    assert "no saved run selected" in context.viewing_text


def test_unknown_record_and_unrecorded_dates_remain_explicit(tmp_path):
    context = build_view_context(_project(), _record(tmp_path, metadata={}, arrays={}), dataset_key="invented")
    assert context.identity == "Model identity unverified"
    assert context.available is False
    assert context.current_A is None
    assert "date unrecorded" in context.viewing_text
    assert "revision unrecorded" in context.viewing_text
    assert context.sampling == "Sampling unverified"
    assert "Unknown dataset" in context.viewing_text


def test_alias_and_metadata_are_not_rewritten_by_presentation(tmp_path):
    record = _record(tmp_path)
    before_metadata, before_arrays = record.metadata, record.arrays
    context = build_view_context(
        _project(), record, alias="Wider copper", channel_label="Gy channel", dataset_key="sheet"
    )
    assert "Viewing: Wider copper" in context.viewing_text
    assert "Editing: Gy channel" in context.editing_text
    assert record.metadata == before_metadata
    for key, value in before_arrays.items():
        np.testing.assert_array_equal(record.arrays[key], value)


@pytest.mark.parametrize("stamp", [None, "", "2026-10-04", "not-a-time"])
def test_directory_times_and_model_revision_cannot_supply_execution_timestamp(tmp_path, stamp):
    metadata = _metadata()
    metadata["job"]["created_at"] = stamp
    context = build_view_context(_project(), _record(tmp_path / "2026-10-05", metadata=metadata), dataset_key="target")
    assert "date unrecorded" in context.viewing_text
    assert "No valid execution timestamp" in context.viewing_tooltip


def test_execution_state_fills_missing_provenance_but_never_saved_scopes(tmp_path):
    context = build_view_context(
        _project(),
        _record(tmp_path, metadata={}),
        dataset_key="contour_field",
        state={
            "project_id": "channel-gy",
            "project_revision": 2,
            "project_fingerprint": "applied",
            "created_at": "2026-10-03T12:00:00Z",
            "wire_fields": _metadata()["wire_fields"],
        },
    )
    assert context.identity == "Current applied model/study"
    assert "revision 2" in context.viewing_text
    assert "date unrecorded" not in context.viewing_text
    assert context.current_A is None
    assert "policy unverified" in context.viewing_text
    assert "scope is unavailable/unverified" in context.viewing_tooltip


def test_saved_metadata_dominates_execution_state(tmp_path):
    context = build_view_context(
        _project(),
        _record(tmp_path),
        dataset_key="sheet",
        state={"project_id": "other", "project_fingerprint": "old", "project_revision": 9, "created_at": "bad"},
    )
    assert context.identity == "Current applied model/study"
    assert "revision 12" in context.viewing_text
    assert "date unrecorded" not in context.viewing_text


def test_legacy_naive_timestamp_is_explicitly_assumed_utc(tmp_path):
    metadata = _metadata()
    metadata["job"]["created_at"] = datetime(2026, 10, 4, 12, tzinfo=timezone.utc).replace(tzinfo=None).isoformat()
    context = build_view_context(_project(), _record(tmp_path, metadata=metadata), dataset_key="target")
    assert "interpreted as UTC" in context.viewing_tooltip


@pytest.mark.parametrize("key", ["contour_field", "contour_residual"])
def test_original_and_saved_refitted_fields_keep_their_own_currents(tmp_path, key):
    record = _record(tmp_path)
    stored = build_view_context(_project(), record, dataset_key=key)
    original = build_view_context(_project(), record, dataset_key=key, wire_current="original")
    assert stored.current_A == 21
    assert "Saved refitted current 21 A" in stored.viewing_text
    assert "weighted_fit" in stored.viewing_tooltip
    assert original.current_A == 19.5
    assert "Original contour current 19.5 A" in original.viewing_text
    assert "original_contour_step" in original.viewing_tooltip
    assert "weighted_fit" not in original.viewing_tooltip
    assert "oriented_unconnected_contours" in original.viewing_tooltip
    assert "No independent winding acceptance" in original.viewing_tooltip


def test_routed_map_uses_routed_geometry_and_current_only(tmp_path):
    context = build_view_context(_project(), _record(tmp_path), dataset_key="routed_field")
    assert context.current_A == 23
    assert "fabricated_centroid" in context.viewing_tooltip
    assert "oriented_unconnected_contours" not in context.viewing_tooltip


def test_missing_original_map_cannot_fall_back_to_stored_array(tmp_path):
    arrays = _arrays()
    del arrays["contour_field_at_stream_current_T"]
    record = _record(tmp_path, arrays=arrays)
    assert build_view_context(_project(), record, dataset_key="contour_field").available is True
    context = build_view_context(_project(), record, dataset_key="contour_field", wire_current="original")
    assert context.available is False
    assert context.current_A == 19.5
    assert "dataset unavailable" in context.viewing_text


def test_missing_original_metric_cannot_borrow_refitted_current(tmp_path):
    metadata = _metadata()
    del metadata["wire_fields"]["contour"]["at_stream_current"]
    context = build_view_context(
        _project(), _record(tmp_path, metadata=metadata), dataset_key="contour_field", wire_current="original"
    )
    assert context.current_A is None
    assert "value unrecorded" in context.viewing_text
    assert "weighted_fit" not in context.viewing_tooltip


@pytest.mark.parametrize("current", [None, True, float("inf"), float("nan"), "21"])
def test_invalid_current_is_never_formatted_as_an_engineering_value(tmp_path, current):
    metadata = _metadata()
    metadata["wire_fields"]["contour"]["current_A"] = current
    context = build_view_context(_project(), _record(tmp_path, metadata=metadata), dataset_key="contour_field")
    assert context.current_A is None
    assert "value unrecorded" in context.viewing_text


def test_missing_array_at_earlier_pipeline_stage_is_not_an_available_dataset(tmp_path):
    metadata = _metadata()
    metadata["pipeline_stage"] = "prepared"
    arrays = {"target_points_m": np.zeros((3, 3)), "target_field_T": np.zeros((3, 3))}
    context = build_view_context(_project(), _record(tmp_path, metadata=metadata, arrays=arrays), dataset_key="sheet")
    assert context.available is False
    assert "dataset unavailable" in context.viewing_text


def test_independent_grid_counts_and_sample_slab_stay_separate(tmp_path):
    context = build_view_context(
        _project(), _record(tmp_path), dataset_key="validation", sample_slice="z ≈ 0 m · 3 samples"
    )
    assert context.available is True
    assert context.sampling == "Independent samples · 7 samples"
    assert context.current_A is None
    assert "Solved sheet coefficients [A]" in context.viewing_text
    assert "Displayed sample selection: z ≈ 0 m · 3 samples; no interpolation" in context.viewing_tooltip


@pytest.mark.parametrize(
    "key, arrays",
    [
        ("contours", {"contour_0_points_m": np.zeros((2, 3))}),
        ("routes", {"route_0_points_m": np.zeros((2, 3))}),
        ("evaluated_winding", {"evaluated_winding_0_points_m": np.zeros((2, 3))}),
        ("fabrication", {"fabrication_0_points_m": np.zeros((2, 3))}),
        ("review_conductor", {"review_conductor_0_points_m": np.zeros((2, 3))}),
        ("review_electrical", {"review_electrical_0_points_m": np.zeros((2, 3))}),
        ("swept", {"swept_0_vertices_m": np.zeros((3, 3)), "swept_0_faces": np.array([[0, 1, 2]], dtype=int)}),
        (
            "stream_function",
            {
                "vertices_m": np.zeros((3, 3)),
                "faces": np.array([[0, 1, 2]], dtype=int),
                "stream_function_A": np.zeros(3),
            },
        ),
    ],
)
def test_geometry_sources_match_actual_saved_array_prefixes(tmp_path, key, arrays):
    context = build_view_context(_project(), _record(tmp_path, arrays=arrays), dataset_key=key)
    assert context.available is True
    assert context.current_A is None


def test_residual_requires_matching_target_array_shape(tmp_path):
    arrays = _arrays()
    arrays["target_field_T"] = np.zeros((2, 3))
    context = build_view_context(_project(), _record(tmp_path, arrays=arrays), dataset_key="contour_residual")
    assert context.available is False


def test_review_field_uses_its_own_points_and_does_not_invent_drive_current(tmp_path):
    arrays = _arrays()
    arrays.update(fabricated_points_m=np.zeros((5, 3)), fabricated_field_T=np.zeros((5, 3)))
    context = build_view_context(_project(), _record(tmp_path, arrays=arrays), dataset_key="fabricated")
    assert context.available is True
    assert context.sampling == "Design samples · 5 samples"
    assert context.current_A is None
    assert "Review current policy unrecorded" in context.viewing_text


def test_header_model_does_not_copy_immutable_scientific_arrays():
    class Record:
        metadata = _metadata()
        _arrays = _arrays()

        @property
        def arrays(self):
            raise AssertionError("Presentation must not copy large scientific arrays")

    assert build_view_context(_project(), Record(), dataset_key="sheet").available is True


def test_cached_capability_avoids_revalidation_on_header_refresh(tmp_path, monkeypatch):
    from coilgen.gui.dataset_capabilities import resolve_dataset
    import coilgen.gui.view_context as context_module

    record = _record(tmp_path)
    capability = resolve_dataset(record.arrays, "sheet")
    monkeypatch.setattr(
        context_module, "resolve_dataset", lambda *_args, **_kwargs: pytest.fail("Display refresh rescanned arrays")
    )
    context = build_view_context(_project(), record, dataset_key="sheet", dataset_capability=capability)
    assert context.available is True


def test_cached_verified_arrays_keep_explicit_failed_saved_verification(tmp_path):
    context = build_view_context(
        _project(), _record(tmp_path), dataset_key="sheet", verification_error="Array digest mismatch"
    )
    assert context.available is True and context.tone == "warning"
    assert "Cached run · verification failed" in context.viewing_text
    assert "Array digest mismatch" in context.viewing_tooltip


def test_nonfinite_field_is_not_available_header_evidence(tmp_path):
    values = _arrays()
    values["sheet_field_T"][0, 0] = np.nan
    context = build_view_context(_project(), _record(tmp_path, arrays=values), dataset_key="sheet")
    assert context.available is False
    assert "finite real numbers" in context.viewing_tooltip


def test_invalid_current_selector_is_rejected():
    with pytest.raises(ValueError, match="wire_current"):
        build_view_context(_project(), wire_current="refit-now")


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["view-context-test"])


@pytest.fixture
def header(application):
    widget = ViewContextHeader()
    widget.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    widget.resize(520, 65)
    widget.show()
    application.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_header_keeps_full_accessible_context_and_opens_details_on_click(header, application, tmp_path):
    context = build_view_context(_project(), _record(tmp_path), dataset_key="routed_field", pending=True)
    header.set_context(context)
    application.processEvents()
    assert header.editing_label.accessibleName() == context.editing_text
    assert header.viewing_label.accessibleName() == context.viewing_text
    assert header.viewing_label.toolTip() == context.viewing_tooltip
    assert header.viewing_label.text() != context.viewing_text
    assert header.minimumSizeHint().width() <= 520
    history, model = [], []
    header.history_requested.connect(lambda: history.append(True))
    header.model_requested.connect(lambda: model.append(True))
    assert not any(button.text() == "History" for button in header.findChildren(QToolButton))
    header.history_requested.emit()  # Retained API; history is opened from the application's View menu.
    assert header.model_button.isVisible()
    QTest.mouseClick(header.model_button, Qt.MouseButton.LeftButton)
    assert history == [True]
    assert model == [True]
    assert header.editing_label.property("semanticTone") == "warning"


def test_model_preview_button_state_and_context_updates_do_not_emit_requests(header):
    emitted = []
    header.model_requested.connect(lambda: emitted.append(True))
    header.history_requested.connect(lambda: emitted.append(True))
    context = build_view_context(_project())
    header.set_context(context)
    assert not header.model_button.isEnabled()
    assert header.model_button.isHidden()
    assert header.context is context
    assert emitted == []


def test_no_saved_run_does_not_offer_an_unnecessary_model_return(header):
    context = build_view_context(_project(), dataset_key="sheet")
    header.set_context(context)
    assert context.source == "run"
    assert context.identity == "No run selected"
    assert header.model_button.isHidden()
    assert not header.model_button.isEnabled()


def test_model_return_remains_available_for_a_saved_run_missing_the_chosen_dataset(header, tmp_path):
    context = build_view_context(_project(), _record(tmp_path, arrays={}), dataset_key="sheet")
    header.set_context(context)
    assert context.available is False
    assert header.model_button.isVisible()
    assert header.model_button.isEnabled()
