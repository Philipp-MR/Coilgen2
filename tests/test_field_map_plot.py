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
