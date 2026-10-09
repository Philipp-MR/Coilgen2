from dataclasses import replace
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QWidget
from matplotlib.figure import Figure

from coilgen.application.field_analysis import FieldDataset, analyze_harmonics, metric_map
from coilgen.application.field_sampling import FieldSlice, SliceCollection
from coilgen.application.service import RunRecord
from coilgen.attribution import AUTHOR_NAME, software_attribution
from coilgen.gui import field_analysis as gui


class FakePlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(layout="constrained")

    def draw(self):
        pass

    def save_png(self, path):
        self.figure.savefig(path)


class FakeCloud(QWidget):
    def set_field(self, *values):
        self.values = values


@pytest.fixture
def application():
    app = QApplication.instance() or QApplication([])
    font_path = Path("C:/Windows/Fonts/segoeui.ttf")
    if font_path.exists():
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            app.setFont(QFont(families[0], 9))
    yield app
    app.processEvents()


@pytest.fixture
def dataset():
    points = np.random.default_rng(81).uniform(-0.04, 0.04, (150, 3))
    desired = np.column_stack((points[:, 0], points[:, 1], points[:, 2])) * 1e-3
    actual = desired + [0, 0, 2e-6]
    return FieldDataset("sheet", "Continuous sheet", points, actual, desired, scope="continuous_sheet")


@pytest.fixture
def record(tmp_path, dataset):
    return RunRecord(
        tmp_path,
        {"name": "Gradient", "request_sha256": "a" * 64, "project_revision": 4},
        {
            "target_points_m": dataset.points_m,
            "sheet_field_T": dataset.field_T,
            "target_field_T": dataset.target_T,
        },
    )


def fake_slices(record, source, **kwargs):
    resolution = kwargs["resolution"]
    horizontal, vertical, normal = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}[kwargs["plane"]]
    center = np.asarray(kwargs["centre_m"], dtype=float)
    pieces = []
    for offset in kwargs["offsets_m"]:
        kwargs["cancel_check"]()
        x, y = np.meshgrid(np.linspace(-0.04, 0.04, resolution), np.linspace(-0.04, 0.04, resolution))
        points = np.broadcast_to(center, (resolution, resolution, 3)).copy()
        points[:, :, horizontal] += x
        points[:, :, vertical] += y
        points[:, :, normal] += offset
        target = points * 1e-3
        actual = target + [0, 0, 2e-6]
        mask = np.ones(x.shape, bool)
        mask[0, 0] = False
        actual[0, 0] = np.nan
        target_mask = mask.copy()
        target_mask[1, 1] = False
        target[1, 1] = np.nan
        pieces.append(
            FieldSlice(
                offset_m=offset,
                coordinates_m=points,
                field_T=actual,
                target_T=target,
                available_mask=mask,
                target_valid_mask=target_mask,
                roi_mask=x**2 + y**2 < 0.04**2,
                outlines_m=(),
            )
        )
    return SliceCollection(
        source_key=source.key,
        label=source.label,
        plane=kwargs["plane"],
        slices=tuple(pieces),
        representation="display_interpolation",
        target_representation="display_interpolation",
        roi_representation="sampling_hull",
        provenance={},
    )


def dialog_with_data(record, dataset):
    dialog = gui.FieldAnalysisDialog(record, auto_start=False, plot_factory=FakePlot, cloud_factory=FakeCloud)
    dialog._catalog_loaded((dataset,))
    dialog.debounce.stop()
    dialog._pending = False
    dialog.count.setValue(2)
    dialog.resolution.setValue(16)
    dialog.debounce.stop()
    return dialog


def wait_worker(application, dialog):
    import time

    deadline = time.monotonic() + 10
    while dialog.worker is not None and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.005)
    assert dialog.worker is None, "Background task did not finish."
    application.processEvents()


def test_maps_use_one_percent_normalization_and_preserve_unsupported_pixels(application, record, dataset, monkeypatch):
    monkeypatch.setattr(gui, "sample_slices", fake_slices)
    dialog = dialog_with_data(record, dataset)
    dialog.normalization.setCurrentIndex(1)
    dialog.debounce.stop()
    dialog.refresh_analysis()
    wait_worker(application, dialog)
    state = dialog.state
    assert state is not None, dialog.status.text()
    expected = 100 * 2e-6 / np.sqrt(np.mean(dataset.target_T[:, 2] ** 2))
    assert state.mapped.values == pytest.approx(np.full(150, expected))
    for values in state.slice_values:
        assert values[2:, 2:] == pytest.approx(np.full(values[2:, 2:].shape, expected))
        assert np.isnan(values[0, 0]) and np.isnan(values[1, 1])
    assert dialog.cloud.values[1] == pytest.approx(state.mapped.values)
    assert dialog._color_limits == pytest.approx((-expected, expected))
    images = [axis.collections[0] for axis in dialog.slice_plot.figure.axes[:-1]]
    assert all(image.get_clim() == dialog._color_limits for image in images)
    dialog.reject()


def test_source_capabilities_current_policy_and_old_result_guard(application, record, dataset, monkeypatch):
    monkeypatch.setattr(gui, "sample_slices", fake_slices)
    bare = replace(dataset, key="routed", label="Routed", target_T=None)
    stream = replace(bare, key="routed_at_stream_current", label="Routed original", field_T=bare.field_T * 2)
    dialog = gui.FieldAnalysisDialog(record, auto_start=False, plot_factory=FakePlot, cloud_factory=FakeCloud)
    dialog._catalog_loaded((bare, stream))
    assert dialog.source.count() == 1
    assert dialog.current_policy.isEnabled()
    assert not dialog.metric_selector.model().item(2).isEnabled()
    dialog.current_policy.setCurrentIndex(1)
    assert dialog._dataset().key == stream.key
    dialog.debounce.stop()
    dialog._pending = False
    dialog.refresh_analysis()
    wait_worker(application, dialog)
    assert dialog.state.dataset is stream
    previous = dialog.state
    dialog.component.setCurrentIndex(0)
    dialog.debounce.stop()
    assert not dialog._current() and not dialog.export_data.isEnabled()
    with pytest.raises(ValueError, match="Recompute"):
        dialog.export_to(record.directory / "stale.json")
    assert dialog.state is previous
    dialog.reject()


def test_stale_worker_result_cannot_publish_and_close_waits_for_cancellation(application, record, dataset):
    dialog = dialog_with_data(record, dataset)
    old_task = gui._AnalysisTask(lambda task: None, dialog._generation - 1)
    dialog.worker = old_task
    dialog._pending = False
    dialog._finished(old_task, ("catalog", (replace(dataset, key="wrong"),)), "")
    assert "wrong" not in dialog.datasets
    entered, release = Event(), Event()

    def operation(task):
        entered.set()
        release.wait(3)
        task.check()
        return ("catalog", (dataset,))

    busy, closed = [], []
    dialog.busyChanged.connect(busy.append)
    dialog.closed.connect(lambda: closed.append(True))
    dialog._start(operation)
    assert entered.wait(3)
    dialog.request_close()
    assert dialog.worker is not None and dialog._close_pending and not closed
    release.set()
    wait_worker(application, dialog)
    assert busy == [True, False] and closed == [True]


def test_harmonics_exports_and_plot_tables_remain_run_bound(application, record, dataset, monkeypatch, tmp_path):
    monkeypatch.setattr(gui, "sample_slices", fake_slices)
    dialog = dialog_with_data(record, dataset)
    dialog.order.setValue(2)
    dialog.debounce.stop()
    dialog.refresh_analysis()
    wait_worker(application, dialog)
    dialog.fit_harmonics()
    wait_worker(application, dialog)
    assert dialog.state.harmonics["rank"] == 9
    assert dialog.coefficients.rowCount() == 9
    assert "Rank 9/9" in dialog.harmonic_note.text()
    dialog.harmonic_source.setCurrentIndex(2)
    residual_fit = dialog.state.harmonics["fits"]["residual"]
    assert np.max(np.abs(residual_fit.residual_values)) < 1e-18
    path = tmp_path / "analysis.json"
    dialog.export_to(path)
    value = json.loads(path.read_text())
    assert value["run_request_sha256"] == "a" * 64
    assert value["software"] == software_attribution()
    assert value["source"]["key"] == "sheet"
    assert value["selection"]["harmonic_order"] == 2
    assert value["slice_scalar_values"][0][0][0] is None
    csv = tmp_path / "native.csv"
    dialog.export_to(csv)
    saved = np.genfromtxt(csv, delimiter=",", names=True)
    assert saved["residual_Bz_T"] == pytest.approx(np.full(150, 2e-6))
    png = tmp_path / "coefficients.png"
    dialog.harmonic_plot.save_png(png)
    assert png.stat().st_size > 1000
    assert record.array("sheet_field_T", copy=False) == pytest.approx(dataset.field_T)
    dialog.reject()


def test_actual_field_plot_exports_credit_software_and_preserve_cloud_pixels(application, tmp_path):
    from PIL import Image

    panel = gui._FigurePanel()
    try:
        panel.figure.subplots().plot([0, 1], [2e-6, 3e-6])
        path = tmp_path / "field.png"
        panel.save_png(path)
        with Image.open(path) as image:
            assert AUTHOR_NAME in image.info["Software"]
        svg = tmp_path / "field.svg"
        panel.save_png(svg)
        assert AUTHOR_NAME in svg.read_text(encoding="utf-8")
    finally:
        panel.close()
    pixels = np.random.default_rng(81).integers(0, 256, size=(5, 8, 4), dtype=np.uint8)
    cloud = SimpleNamespace(plotter=SimpleNamespace(screenshot=lambda **_kwargs: pixels))
    path = tmp_path / "cloud.png"
    gui._PointCloudView.save_png(cloud, path)
    with Image.open(path) as image:
        assert AUTHOR_NAME in image.info["Software"]
        np.testing.assert_array_equal(np.asarray(image), pixels)


def test_field_slices_do_not_require_a_valid_desired_field(application, record, dataset):
    dialog = dialog_with_data(record, dataset)
    selection = replace(dialog.selection(), metric="field", normalization="none")
    mapped = metric_map(dataset, component="Bz", metric="field")
    piece = fake_slices(
        record, dataset, plane="xy", resolution=16, centre_m=(0, 0, 0), offsets_m=(0,), cancel_check=lambda: None
    ).slices[0]
    result = dialog._slice_values(dataset, mapped, piece, selection)
    assert np.isfinite(result[1, 1]) and np.isnan(result[0, 0])
    dialog.reject()


def test_rank_deficient_fit_has_explicit_warning(application, record, dataset):
    plane = replace(dataset, points_m=np.column_stack((dataset.points_m[:, :2], np.zeros(150))))
    harmonic = analyze_harmonics(plane, max_order=2)
    assert not harmonic["full_rank"]
    assert harmonic["diagnostics"] and harmonic["source_free_verified"] is False


def test_progress_distinguishes_measured_work_from_opaque_operations(application, record, dataset):
    dialog = dialog_with_data(record, dataset)
    details, messages = [], []
    dialog.progressDetails.connect(details.append)
    dialog.progressChanged.connect(messages.append)
    dialog._progress(42.25, "Sheet nodes 169/400")
    assert dialog.progress.maximum() == 100 and dialog.progress.value() == 42
    dialog._progress(None, "Solve harmonic least squares")
    assert dialog.progress.maximum() == 0
    assert "%" not in messages[-1]
    assert details[-1]["progress_indeterminate"] and details[-1]["percent"] is None
    dialog._progress(78.5, "Slice 4/5")
    assert dialog.progress.maximum() == 100 and dialog.progress.value() == 78
    assert not details[-1]["progress_indeterminate"]
    dialog.reject()


def test_field_plot_publication_retains_computing_status(application, record, dataset, monkeypatch):
    monkeypatch.setattr(gui, "sample_slices", fake_slices)
    dialog = dialog_with_data(record, dataset)
    busy = []
    dialog.busyChanged.connect(busy.append)
    draw = dialog._render

    def render():
        assert busy == [True]
        assert dialog.progress.maximum() == 0
        draw()

    monkeypatch.setattr(dialog, "_render", render)
    dialog.refresh_analysis()
    wait_worker(application, dialog)
    assert busy == [True, False]
    dialog.reject()


@pytest.mark.parametrize("suffix", ["csv", "json"])
def test_large_export_reports_work_and_cancels_atomically(application, record, dataset, tmp_path, suffix):
    from coilgen.progress import DesignCancelled

    dialog = dialog_with_data(record, dataset)
    big = replace(
        dataset,
        points_m=np.tile(dataset.points_m, (60, 1)),
        field_T=np.tile(dataset.field_T, (60, 1)),
        target_T=np.tile(dataset.target_T, (60, 1)),
    )
    selection = dialog.selection()
    state = gui.AnalysisState(selection, big, metric_map(big), None, ())
    path = tmp_path / ("samples." + suffix)
    calls = []
    dialog.export_to(path, state=state, progress=lambda percent, message: calls.append((percent, message)))
    assert calls[-1][0] == 100
    if suffix == "csv":
        assert any(percent is not None and 0 < percent < 99 for percent, message in calls)
        assert len(np.genfromtxt(path, delimiter=",", skip_header=1)) == len(big.points_m)
    else:
        assert sum("Write captured analysis JSON" in message for percent, message in calls) > 2
        assert len(json.loads(path.read_text())["source"]["points_m"]) == len(big.points_m)
    path.write_text("Previous complete export")
    cancelled = False

    def cancel_after_work(percent, message):
        nonlocal cancelled
        if message.startswith("Write native samples") or message.startswith("Write captured analysis JSON"):
            cancelled = True

    with pytest.raises(DesignCancelled, match="cancelled"):
        dialog.export_to(path, state=state, progress=cancel_after_work, cancel_check=lambda: cancelled)
    assert path.read_text() == "Previous complete export"
    assert not tuple(tmp_path.glob(".*.tmp"))
    dialog.reject()


def test_metric_switch_reuses_physics_but_grid_change_resamples(application, record, dataset, monkeypatch):
    calls = []

    def sampler(*args, **kwargs):
        calls.append(kwargs)
        return fake_slices(*args, **kwargs)

    monkeypatch.setattr(gui, "sample_slices", sampler)
    dialog = dialog_with_data(record, dataset)
    dialog.refresh_analysis()
    wait_worker(application, dialog)
    first = dialog.state.slices
    for widget, index in [(dialog.units, 1), (dialog.component, 0), (dialog.metric_selector, 3)]:
        widget.setCurrentIndex(index)
        dialog.debounce.stop()
        dialog.refresh_analysis()
        wait_worker(application, dialog)
        assert dialog.state.slices is first
    assert len(calls) == 1
    dialog.resolution.setValue(20)
    dialog.debounce.stop()
    dialog.refresh_analysis()
    wait_worker(application, dialog)
    assert len(calls) == 2
    dialog.reject()


def test_vector_error_distribution_uses_vector_scale_and_magnitude_residual_stays_signed(
    application, record, dataset, monkeypatch
):
    monkeypatch.setattr(gui, "sample_slices", fake_slices)
    dialog = dialog_with_data(record, dataset)
    dialog.metric_selector.setCurrentIndex(dialog.metric_selector.findData("vector_error"))
    dialog.normalization.setCurrentIndex(1)
    dialog.debounce.stop()
    dialog.refresh_analysis()
    wait_worker(application, dialog)
    assert dialog.state.error_map is dialog.state.mapped
    expected = 100 * 2e-6 / np.sqrt(np.mean(np.sum(dataset.target_T**2, axis=1)))
    assert dialog.state.error_map.values == pytest.approx(np.full(150, expected))
    assert "Vector residual magnitude" in dialog.error_plot.figure.axes[0].get_xlabel()
    assert not dialog.units.isEnabled()
    selection = replace(dialog.selection(), metric="signed_residual", component="magnitude")
    assert dialog._signed(replace(dialog.state, selection=selection))
    dialog.reject()


def test_immediate_close_does_not_start_queued_catalog_worker(application, record):
    class ClosedCloud(FakeCloud):
        calls = 0

        def close(self):
            self.calls += 1
            return super().close()

    dialog = gui.FieldAnalysisDialog(record, plot_factory=FakePlot, cloud_factory=ClosedCloud)
    busy = []
    dialog.busyChanged.connect(busy.append)
    dialog.request_close()
    dialog.request_close()
    application.processEvents()
    assert dialog.worker is None and busy == []
    assert dialog.cloud.calls == 1


def test_off_origin_offsets_and_plane_switch_use_world_positions(application, record, dataset):
    translated = replace(dataset, points_m=dataset.points_m + [0.2, 0.3, 0.4])
    dialog = dialog_with_data(record, translated)
    dialog.count.setValue(1)
    dialog.offsets.setText("0 0")
    assert dialog.selection().offsets_m == pytest.approx((float(dialog.center.text().split()[2]),))
    dialog.plane.setCurrentIndex(dialog.plane.findData("xz"))
    assert dialog.selection().offsets_m == pytest.approx((float(dialog.center.text().split()[1]),))
    dialog.debounce.stop()
    dialog.reject()


def test_owned_roi_wireframe_applies_region_and_registration_pose():
    from coilgen.application.gradient_studies import RigidTransform
    from coilgen.target_regions import TargetRegion

    region = TargetRegion(
        "box", center_m=(1, 2, 3), rotation_axis=(0, 0, 1), rotation_angle_rad=np.pi / 2, size_m=(0.2, 0.4, 0.6)
    )
    registration = RigidTransform(((0, -1, 0), (1, 0, 0), (0, 0, 1)), (0.1, 0.2, 0.3))
    sample = RigidTransform(((0, 0, 1), (0, 1, 0), (-1, 0, 0)), (0.3, 0.2, 0.1))
    lines = gui.owned_roi_polylines(
        {"region": region.to_dict(), "registration": registration.to_dict(), "sample_transform": sample.to_dict()}
    )
    assert len(lines) == 12 and all(not line.flags.writeable for line in lines)
    expected = np.array([[x, y, z] for x in (-0.1, 0.1) for y in (-0.2, 0.2) for z in (-0.3, 0.3)])
    expected = expected @ region.rotation_matrix.T + region.center_m
    for pose in [registration, sample]:
        expected = expected @ np.asarray(pose.rotation).T + pose.translation_m
    assert np.unique(np.vstack(lines), axis=0) == pytest.approx(np.unique(expected, axis=0))
    assert gui.owned_roi_polylines(None) == ()


def test_visible_dialog_does_not_shadow_qt_paint_metrics(application, record, dataset):
    dialog = dialog_with_data(record, dataset)
    dialog.show()
    application.processEvents()
    assert not dialog.grab().isNull()
    dialog.reject()


def test_camera_first_fit_includes_roi_and_metric_switch_keeps_camera():
    class Plotter:
        reset_count = 0
        actors = []

        def clear(self):
            self.actors = []

        def add_mesh(self, mesh, **kwargs):
            self.actors.append(kwargs)

        def show_axes(self):
            pass

        def show_grid(self, **kwargs):
            pass

        def render(self):
            pass

        def view_isometric(self):
            pass

        def reset_camera(self):
            self.reset_count += 1

    plotter = Plotter()
    view = SimpleNamespace(
        plotter=plotter, first=True, pv=SimpleNamespace(PolyData=lambda points: {}, lines_from_points=lambda points: {})
    )
    points, values = np.zeros((1, 3)), np.zeros(1)
    gui._PointCloudView.set_field(view, points, values, "Bz", "T", (-1, 1), True)
    assert plotter.reset_count == 0
    gui._PointCloudView.set_roi(view, (np.array([[0, 0, 0], [1, 1, 1]]),))
    assert plotter.reset_count == 1 and len(plotter.actors) == 2
    assert all(actor["reset_camera"] is False for actor in plotter.actors)
    gui._PointCloudView.set_field(view, points, values, "Residual", "T", (-1, 1), True)
    gui._PointCloudView.set_roi(view, ())
    assert plotter.reset_count == 1


def test_sparse_owned_roi_and_moved_source_fit_preserve_same_volume_grid(application, record):
    from coilgen.application.gradient_studies import RigidTransform

    pose = RigidTransform().to_dict()

    def source(key, center):
        geometry = {
            "region": {"shape": "sphere", "radius_m": 0.02, "center_m": center},
            "registration": pose,
            "sample_transform": pose,
        }
        return FieldDataset(key, key, [center], [[0, 0, 1e-6]], target_geometry=geometry)

    sparse, same, moved = (
        source("native", (0.4, 0.2, 0.1)),
        source("validation", (0.4, 0.2, 0.1)),
        source("moved", (-0.4, 0.3, 0.2)),
    )
    dialog = gui.FieldAnalysisDialog(record, auto_start=False, plot_factory=FakePlot, cloud_factory=FakeCloud)
    dialog._catalog_loaded((sparse, same, moved))
    assert dialog.selection().center_m == pytest.approx((0.4, 0.2, 0.1))
    assert dialog.selection().size_m == pytest.approx((0.042, 0.042))
    assert float(dialog.harmonic_radius.text()) == pytest.approx(0.02)
    assert dialog.selection().harmonic_center_m == pytest.approx((0.4, 0.2, 0.1))
    dialog.center.setText(".5 .5 .5")
    dialog.size.setText(".11 .22")
    dialog.source.setCurrentIndex(dialog.source.findData("validation"))
    assert dialog.center.text() == ".5 .5 .5" and dialog.size.text() == ".11 .22"
    dialog.fit_source.click()
    assert dialog.selection().center_m == pytest.approx((0.4, 0.2, 0.1))
    dialog.source.setCurrentIndex(dialog.source.findData("moved"))
    assert dialog.selection().center_m == pytest.approx((-0.4, 0.3, 0.2))
    assert dialog.selection().harmonic_center_m == pytest.approx((0.4, 0.2, 0.1))
    dialog.debounce.stop()
    dialog.reject()


def test_exact_rotated_primitive_bounds_cover_sphere_and_box():
    from coilgen.application.gradient_studies import RigidTransform
    from coilgen.target_regions import TargetRegion

    pose = RigidTransform().to_dict()
    sphere = TargetRegion(
        "sphere", center_m=(0.2, 0.3, 0.4), radius_m=0.02, rotation_axis=(1, 2, 3), rotation_angle_rad=0.8
    )
    geometry = {"region": sphere.to_dict(), "registration": pose, "sample_transform": pose}
    bounds = gui.owned_roi_bounds(geometry)
    assert bounds == pytest.approx(np.array([[0.18, 0.28, 0.38], [0.22, 0.32, 0.42]]))
    box = TargetRegion("box", size_m=(0.1, 0.2, 0.3), rotation_axis=(1, 2, 3), rotation_angle_rad=0.8)
    geometry["region"] = box.to_dict()
    lines = gui.owned_roi_polylines(geometry)
    exact = np.stack((np.vstack(lines).min(axis=0), np.vstack(lines).max(axis=0)))
    assert gui.owned_roi_bounds(geometry) == pytest.approx(exact)


def test_csv_names_metric_definition_and_preserves_weights(application, record, dataset, monkeypatch, tmp_path):
    monkeypatch.setattr(gui, "sample_slices", fake_slices)
    weighted = replace(dataset, weights=np.linspace(1, 2, len(dataset.points_m)))
    dialog = dialog_with_data(record, weighted)
    dialog.sampling.setCurrentIndex(dialog.sampling.findData("weighted"))
    dialog.normalization.setCurrentIndex(dialog.normalization.findData("target_rms"))
    dialog.debounce.stop()
    dialog.refresh_analysis()
    wait_worker(application, dialog)
    path = tmp_path / "weighted.csv"
    dialog.export_to(path)
    data = np.genfromtxt(path, delimiter=",", names=True)
    column = "Bz_signed_residual_target_rms_weighted_percent"
    assert column in data.dtype.names
    assert data["saved_sample_weight"] == pytest.approx(weighted.weights)
    assert data[column] == pytest.approx(dialog.state.mapped.values)
    dialog.reject()
