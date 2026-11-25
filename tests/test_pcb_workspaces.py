"""Both GUI adapters retain source identity and consume the same PCB candidate."""

import asyncio
import time
import pytest

pytest.importorskip("shapely")
from _coilgen_test_support.pcb_layout import browser_pcb, pcb_session


def test_browser_rejects_stale_edits_and_exports_current_scene(tmp_path):
    adapter = browser_pcb(tmp_path)
    old = adapter.session.identity
    asyncio.run(adapter.action(dict(kind="settings", identity=old, values={"width_m": 0.002})))
    new = adapter.session.identity
    asyncio.run(adapter.action(dict(kind="settings", identity=old, values={"width_m": 0.003})))
    assert adapter.session.identity == new
    assert "changed" in adapter.state.pcb_error
    asyncio.run(adapter.action(dict(kind="export", identity=new, formats=["svg", "dxf"])))
    assert adapter.state.pcb_download.endswith(".zip")
    assert len(list(tmp_path.glob("*.zip"))) == 1


def test_browser_invalid_source_cannot_leave_previous_layout_editable(tmp_path):
    adapter = browser_pcb(tmp_path)
    adapter.state.pcb_download = "downloads/old.zip"
    asyncio.run(adapter.open())
    assert adapter.session is None
    assert adapter.state.pcb_scene is None
    assert not adapter.state.pcb_download
    assert "routing checkpoint" in adapter.state.pcb_error


def test_native_linked_selection_and_worker_update(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from coilgen.gui.pcb_layout import PcbLayoutDialog

    app = QApplication.instance() or QApplication([])
    dialog = PcbLayoutDialog(pcb_session())
    dialog.show()
    app.processEvents()
    try:
        dialog.flat.selected.emit("wire:start")
        assert dialog.flat.selection == dialog.wrapped.selection == "wire:start"
        dialog.wrapped.selected.emit("wire:end")
        assert dialog.end_a.currentData() == "wire:start"
        assert dialog.end_b.currentData() == "wire:end"
        dialog.edit("settings", width_m=0.002)
        deadline = time.monotonic() + 10
        while dialog._busy and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert not dialog._busy
        assert dialog.scene["settings"]["width_m"] == 0.002
        assert dialog.flat.scene is dialog.wrapped.scene
        assert not dialog.session.design.errors
    finally:
        dialog.close()
        app.processEvents()
