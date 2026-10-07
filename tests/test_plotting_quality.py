from types import SimpleNamespace
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.field_maps import ScalarMap
from coilgen.gradient_slice_maps import GradientSliceMapResult, GradientSliceMapSlice, GradientSliceMapSpec
from coilgen.matplotlib_visualization import (
    _linear_color_norm,
    _plot_cross_section_grid,
    _scalar_map_styles,
    plot_field_pointclouds_3d,
    plot_gradient_slice_inspector,
    pointcloud_plot_indices,
)
from coilgen.models import TargetField
from coilgen.plot_book import PlotBookEntry, plot_book_page_from_entry, plot_book_rendering


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _field_result(count=3000):
    coords = np.random.default_rng(14).uniform(-0.1, 0.1, (count, 3))
    field = np.zeros((count, 3))
    field[:, 2] = 1e-5 * (coords[:, 0] + 0.5 * coords[:, 2])
    target = TargetField(
        coords=coords,
        b=field,
        weights=np.ones(count),
        suppressed=np.zeros(count, dtype=bool),
        target_field_group_inds=np.ones(count, dtype=int),
    )
    return SimpleNamespace(
        target_field=target,
        target_field_1A=target,
        input_data=CoilGenConfig(),
        field_by_layout=10 * field,
        field_layout_per1Amp=10 * field,
        b_field_opt_sf=0.9 * field,
        b_field_opt_sf_1A=0.9 * field,
        coil_parts=[],
    )


def test_default_sampling_retains_domain_and_field_coordinate_extrema():
    count = 5000
    rng = np.random.default_rng(21)
    coords = rng.normal(size=(count, 3))
    values = rng.normal(size=(count, 2))
    domains = np.zeros(count, dtype=int)
    domains[-7:] = 1
    kwargs = dict(coordinates=coords, scalar_values=values, domain_labels=domains)
    selected = pointcloud_plot_indices(CoilGenConfig(), count, **kwargs)
    np.testing.assert_array_equal(selected, pointcloud_plot_indices(CoilGenConfig(), count, **kwargs))
    assert len(selected) == 2000
    assert set(domains[selected]) == {0, 1}
    for group in (np.flatnonzero(domains == 0), np.flatnonzero(domains == 1)):
        for column in np.column_stack((values, coords)).T:
            assert group[np.argmin(column[group])] in selected
            assert group[np.argmax(column[group])] in selected
    np.testing.assert_array_equal(
        pointcloud_plot_indices(CoilGenConfig(plot_pointcloud_max_points=0), count), np.arange(count)
    )


def test_field_comparison_shares_full_data_scale_without_mutating_fields():
    result = _field_result()
    original = result.target_field.b.copy()
    figure = plot_field_pointclouds_3d(result, show=False)
    axes = [axis for axis in figure.axes if axis.name == "3d"]
    scatters = [axis.collections[0] for axis in axes]
    assert scatters[0].norm is scatters[1].norm is scatters[2].norm
    expected = np.max(np.abs(result.field_layout_per1Amp[:, 2]))
    assert scatters[0].norm.vmax == pytest.approx(expected)
    assert scatters[0].norm.vmin == pytest.approx(-expected)
    assert scatters[3].norm.vmax == pytest.approx(-scatters[3].norm.vmin)
    assert all(len(scatter._offsets3d[0]) == 2000 for scatter in scatters)
    assert all(len(axis.get_xticks()) == 3 for axis in axes)
    assert all(any("2,000 / 3,000" in text.get_text() for text in axis.texts) for axis in axes)
    np.testing.assert_array_equal(result.target_field.b, original)
    for dimension in range(3):
        low, high = (
            axes[0].get_xlim3d() if dimension == 0 else axes[0].get_ylim3d() if dimension == 1 else axes[0].get_zlim3d()
        )
        assert low <= np.min(result.target_field.coords[:, dimension]) + 1e-14
        assert high >= np.max(result.target_field.coords[:, dimension]) - 1e-14


def test_constant_microtesla_scale_is_relative_and_small_variation_is_preserved():
    norm = _linear_color_norm(np.full(8, 1e-6))
    assert norm.vmin == pytest.approx(0.95e-6)
    assert norm.vmax == pytest.approx(1.05e-6)
    tiny = _linear_color_norm(np.array([1e-18, 2e-18]))
    assert tiny.vmin == 1e-18
    assert tiny.vmax == 2e-18


def test_magnitude_maps_are_sequential_and_signed_errors_diverge_at_zero():
    coords = np.zeros((3, 3))
    maps = [
        ScalarMap("target", "Target |grad Bz|", np.array([1.0, 2.0, 3.0]), "T/m"),
        ScalarMap("layout", "Layout |grad Bz|", np.array([2.0, 4.0, 6.0]), "T/m"),
        ScalarMap("layout_vs_target", "Gradient |grad Bz| Relative Error", np.array([-3.0, 1.0, 2.0]), "%"),
    ]
    styles = _scalar_map_styles(maps, coords)
    assert styles[0][0] is styles[1][0]
    assert styles[0][1] == "viridis"
    assert styles[0][0].vmin == 0
    assert styles[0][0].vmax == 6
    assert styles[2][1] == "RdBu_r"
    assert styles[2][0].vmin == -3
    assert styles[2][0].vmax == 3


def test_suppression_cloud_cannot_displace_dsv_error_cross_sections():
    dsv = np.array(np.meshgrid([-0.01, 0.01], [-0.01, 0.01], [-0.01, 0.01])).reshape(3, -1).T
    coords = np.vstack((dsv, np.random.default_rng(4).uniform(1, 2, (1000, 3))))
    values = np.full(len(coords), np.nan)
    values[: len(dsv)] = np.arange(len(dsv))
    result = SimpleNamespace(target_field=SimpleNamespace(coords=coords))
    figure = _plot_cross_section_grid(
        result,
        [
            ScalarMap(
                "error",
                "DSV error",
                values,
                "T",
                current_convention="Final fitted current",
                normalization="Peak target reference",
            )
        ],
        figure_title="Domain regression",
        show=False,
        block=False,
    )
    axes = [
        axis
        for axis in figure.axes
        if axis.collections and hasattr(axis.collections[0], "get_offsets") and axis.get_xlabel()
    ]
    assert len(axes) == 3
    assert all(len(axis.collections[0].get_offsets()) == len(dsv) for axis in axes)
    footer = " ".join(text.get_text() for text in figure.texts)
    assert "Final fitted current" in footer and "Peak target reference" in footer


def test_plotbook_owns_first_agg_draw():
    draws = []
    original = FigureCanvasAgg.draw

    def counted(canvas):
        draws.append(canvas)
        return original(canvas)

    with patch.object(FigureCanvasAgg, "draw", counted):
        with plot_book_rendering():
            figure = plot_field_pointclouds_3d(_field_result(100), show=False)
        assert not draws
        entry = PlotBookEntry(key="field", short_name="Field", title="Field", category="field", figure=figure)
        page = plot_book_page_from_entry(entry)
        assert len(draws) == 1
        assert page.image_rgba.shape == (800, 1100, 4)


def _slice_data():
    x = np.array([-0.02, 0.0, 0.02])
    y = x.copy()
    slices = []
    for index, z in enumerate((-0.01, 0.01)):
        xx, yy = np.meshgrid(x, y)
        bz = (index + 1) * (xx + yy) * 1e-4
        bz[0, 0] = 999.0
        magnitude = np.full((3, 3), (index + 1) * 1e-3)
        mask = np.ones((3, 3), dtype=bool)
        mask[0, 0] = False
        slices.append(
            GradientSliceMapSlice(
                axis=2,
                value=z,
                plot_axes=(0, 1),
                x=x,
                y=y,
                coords=np.stack((xx, yy, np.full_like(xx, z)), axis=-1),
                mask=mask,
                maps={"Bz": bz, "gradBz_mag": magnitude},
                dsv_outline_center_m=(0.0, 0.0),
                dsv_outline_radius_m=0.025,
            )
        )
    return GradientSliceMapResult(
        spec=GradientSliceMapSpec(enabled=True, source="target", slice_count=2),
        axis=2,
        axis_name="z",
        plot_axes=(0, 1),
        center_m=np.zeros(3),
        reference_radius_m=0.03,
        current_scale_A=1.0,
        gradient_step_m=1e-4,
        slices=tuple(slices),
        component_units={"Bz": "T", "gradBz_mag": "T/m"},
    )


def test_slice_inspector_reuses_arrays_and_artists_for_all_controls():
    evaluated = _slice_data()
    originals = [item.maps["Bz"].copy() for item in evaluated.slices]
    with patch("coilgen.matplotlib_visualization.gradient_slice_map_result", return_value=evaluated) as fetch:
        figure = plot_gradient_slice_inspector(SimpleNamespace(), show=False)
        inspector = figure._coilgen_slice_inspector
        image = inspector.image
        initial_limit = image.norm.vmax
        outline_count = len(inspector.ax.lines)
        assert initial_limit == pytest.approx(8.0)
        np.testing.assert_allclose(inspector.ax.get_xlim(), (-0.03, 0.03))
        np.testing.assert_allclose(inspector.ax.get_ylim(), (-0.03, 0.03))
        assert "outside valid domain" in inspector._format_coord(-0.02, -0.02)
        assert "Outside the evaluated slice grid" in inspector._format_coord(1.0, 1.0)
        inspector.slider.set_val(1)
        assert image.norm.vmax == initial_limit
        inspector.scale_button.set_active(0)
        assert image.norm.vmax == pytest.approx(initial_limit / 2)
        inspector.unit_buttons.set_active(1)
        assert image.norm.vmax == pytest.approx(initial_limit / 2 / 1e6)
        inspector.component_buttons.set_active(1)
        assert image.norm.vmin == 0
        assert image.get_cmap().name == "viridis"
        assert "z=-0.01 m" in inspector._format_coord(0, 0)
        assert "0.001 T/m" in inspector._format_coord(0, 0)
        assert inspector.image is image
        assert len(inspector.ax.lines) == outline_count == 2
        assert fetch.call_count == 1
    for item, original in zip(evaluated.slices, originals):
        np.testing.assert_array_equal(item.maps["Bz"], original)
