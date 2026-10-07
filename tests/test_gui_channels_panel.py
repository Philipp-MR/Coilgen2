"""Read-only channel navigation and compute requests, without a solver or files."""

import os
import math

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QAbstractItemView, QDialog

from coilgen.application.assembly import ProjectAssembly
from coilgen.application.presets import preset_config
from coilgen.application.project import ProjectDocument
from coilgen.application.system_presets import system_assembly
from coilgen.gui.channels_panel import ChannelOverview
from coilgen.gui.guidance import resolved_model_summary


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["channel-overview-test"])


def test_nine_channel_values_match_current_models_and_remain_read_only(application):
    assembly = system_assembly()
    before = assembly.to_dict()
    panel = ChannelOverview(assembly, "xz")
    assert panel.table.rowCount() == 9
    assert panel.table.currentRow() == 5
    assert panel.table.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers
    for row, channel in enumerate(assembly.channels):
        summary = resolved_model_summary(channel.project)
        assert panel.table.item(row, 0).text() == channel.name
        assert panel.table.item(row, 0).data(Qt.ItemDataRole.UserRole) == channel.id
        assert not panel.table.item(row, 0).icon().isNull()
        assert panel.table.item(row, 1).text() == "Enabled"
        assert summary["target"] in panel.table.item(row, 2).toolTip()
        assert summary["roi"] in panel.table.item(row, 2).toolTip()
        assert channel.project.values["field_shape_function"] in panel.table.item(row, 2).toolTip()
        assert panel.table.item(row, 3).text() == f"300 × 300 mm · gap {100 + 12 * row} mm"
        assert "Primitive normal: (1, 0, 0)" in panel.table.item(row, 3).toolTip()
        assert str(tuple(channel.project.values["biplanar_mesh_parameter_list"])) in panel.table.item(row, 3).toolTip()
        assert panel.table.item(row, 4).text() == ("2 × 2 mm" if row < 3 else "1 × 1 mm")
        assert panel.table.item(row, 5).text() == "Wire evaluation · strict review"
        for column in range(6):
            assert panel.table.item(row, column).data(Qt.ItemDataRole.AccessibleDescriptionRole)
    assert "remain unverified" in panel.scope.text()
    assert assembly.to_dict() == before
    panel.close()


def test_open_selected_channel_emits_only_identity_then_accepts(application):
    assembly = system_assembly()
    panel = ChannelOverview(assembly, "gx")
    opened, computed = [], []
    panel.channel_requested.connect(opened.append)
    panel.compute_requested.connect(lambda: computed.append(True))
    panel.table.selectRow(7)
    panel.open_button.click()
    assert opened == [assembly.channels[7].id]
    assert not computed
    assert panel.result() == QDialog.DialogCode.Accepted


def test_compute_button_only_requests_parent_action(application):
    assembly = system_assembly()
    panel = ChannelOverview(assembly, "gy")
    before = assembly.to_dict()
    opened, computed = [], []
    panel.channel_requested.connect(opened.append)
    panel.compute_requested.connect(lambda: computed.append(True))
    panel.compute_button.click()
    assert computed == [True]
    assert not opened
    assert panel.result() == QDialog.DialogCode.Accepted
    assert assembly.to_dict() == before


def test_disabled_and_empty_collections_have_no_compute_action(application):
    assembly = system_assembly()
    for channel in assembly.channels:
        assembly = assembly.update_channel(channel.id, enabled=False)
    panel = ChannelOverview(assembly, "gx")
    requested = []
    panel.compute_requested.connect(lambda: requested.append(True))
    assert not panel.compute_button.isEnabled()
    panel.compute_button.click()
    assert not requested
    assert panel.open_button.isEnabled()  # Opening its editor can enable a channel.
    assert panel.table.item(0, 1).text() == "Disabled"
    panel.close()
    empty = ChannelOverview(ProjectAssembly.new(), "missing")
    assert empty.table.rowCount() == 0
    assert not empty.open_button.isEnabled()
    assert not empty.compute_button.isEnabled()
    empty.close()


def test_target_edits_override_stale_example_metadata_in_summary(application):
    assembly = system_assembly()
    channel = assembly.channel("gx")
    modified = channel.project.update_config(
        {
            "target_gradient_strength": 0.031,
            "generated_mesh_rotation_axis": (0, 0, 1),
            "generated_mesh_rotation_angle": math.pi / 2,
        }
    )
    assembly = assembly.update_channel(channel.id, project=modified)
    panel = ChannelOverview(assembly, "gx")
    assert "31 mT/m" in panel.table.item(0, 2).text()
    assert "31 mT/m" in panel.table.item(0, 2).toolTip()
    assert "20 mT/m" not in panel.table.item(0, 2).toolTip()
    assert modified.session["system_example"]["mode"] == "Gx"
    support = panel.table.item(0, 3).toolTip()
    assert "Primitive normal" in support
    assert "1.5708 rad" in support
    assert "(0, 0, 1)" in support
    assert "around the world origin" in support
    panel.close()


def test_ordinary_single_channel_and_long_names_fit_without_horizontal_scrolling(application):
    project = ProjectDocument.from_config(preset_config(), name="A" * 120)
    assembly = ProjectAssembly.new("Single edited coil", (project,))
    panel = ChannelOverview(assembly, assembly.channels[0].id)
    panel.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    panel.resize(800, 440)
    panel.show()
    for _ in range(3):
        application.processEvents()
    assert panel.table.horizontalScrollBar().maximum() == 0
    assert panel.table.item(0, 0).toolTip() == "A" * 120
    assert panel.table.item(0, 3).toolTip() == resolved_model_summary(project)["support"]
    panel.close_button.click()
    assert panel.result() == QDialog.DialogCode.Rejected
