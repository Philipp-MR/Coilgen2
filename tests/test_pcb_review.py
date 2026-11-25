"""Regression coverage for shared graphical review and restart-safe drafts."""

import asyncio
from copy import deepcopy
import json
import threading
import zipfile

import numpy as np
import pytest

pytest.importorskip("shapely")
from _coilgen_test_support.pcb_layout import pcb_session, browser_pcb
from coilgen.application.pcb_review import issue_groups
from coilgen.models import ConnectedGroup


def split_session():
    session = pcb_session()
    a = np.array([[0.02, 0.03, 0], [0.05, 0.03, 0]])
    b = np.array([[0.05, 0.03, 0], [0.08, 0.06, 0]])
    session.parts[0].connected_groups = [ConnectedGroup(a, a, a, b)]
    session.change("settings", mode="spirals")
    return session


def test_groups_prioritize_circuit_and_retain_every_location():
    issues = [
        dict(code=code, uv=[i, 0], trace=f"g{i}")
        for i, code in enumerate(
            ["off_surface", "seam", "edge", "seam", "terminals", "branch", "clearance", "turn_clearance"]
        )
    ]
    groups = issue_groups(issues)
    assert [g["key"] for g in groups] == ["seam", "circuit", "spacing", "edge", "off_surface"]
    assert sorted(i for g in groups for i in g["indices"]) == list(range(len(issues)))
    assert sorted(i for g in groups for i in g["locations"]) == list(range(len(issues)))
    assert groups[0]["action"] == "seam"
    assert groups[-1]["action"] == "compare"


def test_preview_is_read_only_and_matches_explicit_transition():
    session = split_session()
    before, history = session.document(), deepcopy(session.history)
    preview = session.preview_connection("g1-in:end", "g1-out:start", "top")
    assert preview["transitions"] == 1
    assert preview["length_mm"] == pytest.approx(0)
    assert not preview["warnings"]
    assert [p["label"] for p in preview["pins"]] == ["A", "B"]
    assert session.document() == before and session.history == history
    session.change("connect", a=preview["a"], b=preview["b"], layer=preview["layer"])
    assert len(session.design.vias) == preview["transitions"]
    assert session.preview_connection(preview["a"], preview["b"], preview["layer"])["duplicate"]
    assert "Group" in session.scene()["connection_labels"][0]


def test_preview_warns_about_overlapping_return_without_changing_layout():
    session = pcb_session()
    identity = session.identity
    preview = session.preview_connection("wire:start", "wire:end", "top")
    assert preview["length_mm"] == pytest.approx(60)
    assert any("insufficient gap" in w for w in preview["warnings"])
    assert session.identity == identity


def test_original_overlay_reports_physical_layer_displacement():
    session = pcb_session()
    trace = session.scene()["traces"][0]
    np.testing.assert_array_equal(trace["original"], session.parts[0].wire_path.v)
    assert trace["comparison"]["distance_mm"] == pytest.approx(0.1)
    delta = np.array(trace["comparison"]["saved"]) - trace["comparison"]["proposed"]
    assert np.linalg.norm(delta) * 1000 == pytest.approx(trace["comparison"]["distance_mm"])


def test_rejected_edit_preserves_previous_result_and_evaluation():
    session = pcb_session()
    session.evaluate()
    previous = session.design, session.evaluation, session.identity
    with pytest.raises(ValueError):
        session.change("settings", width_m=0)
    assert session.design is previous[0] and session.evaluation is previous[1]
    assert session.identity == previous[2]


def test_workspace_draft_restores_after_restart_and_does_not_replace_corruption(tmp_path):
    session = pcb_session().attach_storage(tmp_path)
    session.change("settings", width_m=0.0015)
    restored = pcb_session().attach_storage(tmp_path)
    assert restored.identity == session.identity
    assert "Saved in workspace" in restored.scene()["persistence"]
    assert not restored.history
    path = session.storage_path
    path.write_text("invalid draft", encoding="utf-8")
    rejected = pcb_session().attach_storage(tmp_path)
    rejected.change("settings", width_m=0.002)
    assert "could not be restored" in rejected.scene()["persistence"]
    assert path.read_text() == "invalid draft"


def test_export_readiness_matches_visible_summary(tmp_path):
    session = pcb_session()
    review = session.scene()["review"]
    assert review["inspection"] and "not evaluated" in review["field"]
    path = session.export(tmp_path / "review.zip")
    with zipfile.ZipFile(path) as package:
        manifest = json.loads(package.read("pcb-manifest.json"))
    assert manifest["review"] == review
    session.evaluation = {"field_budget": {"passed": True}}
    assert not session.scene()["review"]["inspection"]


def test_browser_rejected_edit_clears_updating_status(tmp_path):
    adapter = browser_pcb(tmp_path)
    asyncio.run(adapter.action(dict(kind="settings", identity=adapter.session.identity, values={"width_m": 0})))
    assert not adapter.state.pcb_busy
    assert adapter.state.pcb_status == "Action failed; layout unchanged."
    assert adapter.state.pcb_error
    assert adapter.session.design.settings.width_m == 0.001


def test_browser_ignores_preview_from_superseded_request(tmp_path, monkeypatch):
    adapter = browser_pcb(tmp_path)
    original = adapter.session.preview_connection
    started, release = threading.Event(), threading.Event()

    def slow(a, b, layer):
        if layer == "top":
            started.set()
            assert release.wait(5)
        return original(a, b, layer)

    monkeypatch.setattr(adapter.session, "preview_connection", slow)

    async def run():
        def message(layer):
            return dict(
                kind="preview",
                identity=adapter.session.identity,
                values=dict(a="wire:start", b="wire:end", layer=layer),
            )

        first = asyncio.create_task(adapter.action(message("top")))
        assert await asyncio.to_thread(started.wait, 5)
        await adapter.action(message("bottom"))
        release.set()
        await first
        assert adapter.state.pcb_preview["layer"] == "bottom"
        assert not adapter.state.pcb_busy

    asyncio.run(run())


def test_native_preview_and_grouped_review_share_graphical_selection(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from coilgen.gui.pcb_layout import PcbLayoutDialog

    app = QApplication.instance() or QApplication([])
    dialog = PcbLayoutDialog(split_session())
    try:
        dialog.show()
        dialog.open_tab(2)
        dialog.end_a.setCurrentIndex(dialog.end_a.findData("g1-in:end"))
        dialog.end_b.setCurrentIndex(dialog.end_b.findData("g1-out:start"))
        app.processEvents()
        assert dialog.preview["transitions"] == 1
        assert dialog.connect_button.isEnabled()
        assert dialog.flat.options["pins"] == dialog.wrapped.options["pins"]
        assert dialog.flat.options["proposal"] is dialog.preview
        dialog.open_tab(3)
        assert dialog.preview is None
        assert dialog.checks.count() == len(dialog.scene["issue_groups"])
        assert "inspection" in dialog.export_button.text()
        dialog.compare.setChecked(True)
        assert dialog.flat.options["compare"] and dialog.wrapped.options["compare"]
    finally:
        dialog.close()
        app.processEvents()
