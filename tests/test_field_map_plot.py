import numpy as np
import pytest

from coilgen.gui.field_map_plot import draw_sample_sections, sample_section


@pytest.fixture
def grid():
    points = np.array(np.meshgrid(*[[-0.01, 0, 0.01]] * 3, indexing="ij")).reshape(3, -1).T
    values = points @ [1, 2, 3]
    return points, values


@pytest.mark.parametrize("plane, normal, coefficients", [("xy", 2, (1, 2)), ("xz", 1, (1, 3)), ("yz", 0, (2, 3))])
def test_sections_preserve_axis_orientation_and_native_values(grid, plane, normal, coefficients):
    points, values = grid
    original = points.copy()
    section = sample_section(points, values, plane)
    assert section.normal_range_mm == (0, 0)
    assert np.all(section.points_mm[:, normal] == 0)
    u, v, cells = section.grid
    assert u == pytest.approx([-10, 0, 10])
    assert v == pytest.approx([-10, 0, 10])
    assert cells == pytest.approx((u[None, :] * coefficients[0] + v[:, None] * coefficients[1]) / 1000)
    np.testing.assert_array_equal(points, original)


def test_missing_native_sample_is_not_filled(grid):
    points, values = grid
    valid = np.any(points != 0, axis=1)
    _, _, cells = sample_section(points[valid], values[valid], "xy").grid
    assert np.isnan(cells[1, 1])
    assert np.isfinite(cells).sum() == 8


def test_scattered_volume_shows_only_labelled_thin_slab():
    points = np.random.default_rng(32).uniform(-0.02, 0.02, (300, 3))
    values = np.arange(len(points))
    section = sample_section(points, values, "xy")
    assert section.grid is None
    assert 0 < len(section.values) < len(values) / 4
    low, high = section.normal_range_mm
    assert high - low <= np.ptp(points[:, 2]) * 100
    assert np.all(section.points_mm[:, 2] >= low)
    assert np.all(section.points_mm[:, 2] <= high)


def test_duplicate_observations_are_not_silently_averaged(grid):
    points, values = grid
    section = sample_section(np.vstack((points, [0, 0, 0])), np.r_[values, 1], "xy")
    assert section.grid is None
    assert len(section.values) == 10


@pytest.mark.parametrize("size", [(12, 4), (6, 7)])
def test_render_shares_zero_centred_scale_and_exports_without_geometry_changes(grid, size, tmp_path):
    from matplotlib.figure import Figure

    points, values = grid
    figure = Figure(figsize=size)
    draw_sample_sections(figure, points, values, "Signed Bz residual", "µT", (-0.1, 0.1), True)
    maps = figure.axes[:3]
    assert [axis.get_xlabel() for axis in maps] == ["x [mm]", "x [mm]", "y [mm]"]
    assert all(axis.collections[0].get_clim() == (-0.1, 0.1) for axis in maps)
    assert all("saved plane" in axis.get_title(loc="left") for axis in maps)
    figure.savefig(tmp_path / "sections.png")


def test_single_point_and_flat_field_are_renderable(tmp_path):
    from matplotlib.figure import Figure

    figure = Figure()
    draw_sample_sections(figure, [[0, 0, 0]], [0], "Bz", "µT", (-1e-12, 1e-12), True)
    figure.savefig(tmp_path / "point.png")


def test_comparison_preserves_common_mask_and_component_difference():
    from coilgen.application.field_presentation import field_comparison_values

    actual = np.array([[[1.0, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    target = actual * 0.9
    target[0, 1, 2] = np.nan
    valid = np.array([[True, True], [False, True]])
    before = actual.copy()
    desired, field, residual = field_comparison_values(actual, target, "Bz", valid=valid)
    for values in (desired, field, residual):
        np.testing.assert_array_equal(np.isfinite(values), [[True, False], [False, True]])
        assert not values.flags.writeable
    np.testing.assert_allclose(residual, field - desired, equal_nan=True)
    assert residual[1, 1] == pytest.approx(1.2)
    np.testing.assert_array_equal(actual, before)


def test_comparison_magnitude_difference_is_not_vector_error_and_b0_projection_is_shared():
    from coilgen.application.field_presentation import field_comparison_values

    actual, target = np.array([[3.0, 4, 0]]), np.array([[-3.0, -4, 0]])
    desired, field, residual = field_comparison_values(actual, target, "magnitude")
    assert desired == pytest.approx([5]) and field == pytest.approx([5])
    assert residual == pytest.approx([0])
    desired, field, residual = field_comparison_values(actual, target, "Bparallel", b0_direction=(2, 0, 0))
    assert desired == pytest.approx([-3]) and field == pytest.approx([3]) and residual == pytest.approx([6])
    assert all(np.all(np.isnan(values)) for values in field_comparison_values([[1, 2, 3]], None, "Bz"))


@pytest.mark.parametrize("component", ["Bz", "magnitude"])
def test_comparison_uses_identical_field_norm_and_separate_signed_residual_scale(grid, tmp_path, component):
    from matplotlib.figure import Figure
    from coilgen.gui.field_map_plot import draw_field_comparison

    points, target = grid
    if component == "magnitude":
        target = np.abs(target)
    actual = target + 0.001
    residual = actual - target
    figure = Figure(figsize=(11, 4))
    shared, separate = draw_field_comparison(
        figure,
        points,
        (target, actual, residual),
        component,
        "µT",
        plane="xy",
    )
    target_plot, actual_plot, error_plot = [axis.collections[0] for axis in figure.axes[:3]]
    assert target_plot.norm is actual_plot.norm
    assert target_plot.get_clim() == actual_plot.get_clim() == shared
    assert error_plot.get_clim() == separate
    assert separate[0] == -separate[1] and separate[1] < shared[1] / 10
    assert shared[0] == (0 if component == "magnitude" else -shared[1])
    assert len(figure.axes) == 5  # Three maps, one shared field bar, one residual bar.
    np.testing.assert_array_equal(target_plot.cmap(target_plot.norm(0.02)), actual_plot.cmap(actual_plot.norm(0.02)))
    figure.savefig(tmp_path / "comparison.png")
