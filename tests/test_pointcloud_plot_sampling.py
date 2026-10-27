from __future__ import annotations

from types import SimpleNamespace
import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)

from coilgen.config import CoilGenConfig
from coilgen.matplotlib_visualization import (
    plot_field_pointclouds_3d,
    plot_problem_setup_geometry_3d,
    plot_shim_pointclouds_3d,
    pointcloud_plot_indices,
)
from coilgen.models import CoilPart, Mesh, TargetField


class PointcloudPlotSamplingTests(unittest.TestCase):
    def test_random_display_subset_is_capped_unique_and_reproducible(self) -> None:
        config = CoilGenConfig(plot_pointcloud_max_points=500, plot_pointcloud_random_seed=41)
        eligible = np.ones(1600, dtype=bool)
        eligible[:100] = False

        first = pointcloud_plot_indices(config, len(eligible), eligible_mask=eligible)
        second = pointcloud_plot_indices(config, len(eligible), eligible_mask=eligible)

        self.assertEqual(len(first), 500)
        self.assertEqual(len(np.unique(first)), 500)
        self.assertTrue(np.all(eligible[first]))
        np.testing.assert_array_equal(first, second)

        changed_seed = CoilGenConfig(plot_pointcloud_max_points=500, plot_pointcloud_random_seed=42)
        changed = pointcloud_plot_indices(changed_seed, len(eligible), eligible_mask=eligible)
        self.assertFalse(np.array_equal(first, changed))

    def test_display_subset_does_not_sample_at_or_below_limit(self) -> None:
        config = CoilGenConfig(plot_pointcloud_max_points=500, plot_pointcloud_random_seed=41)

        indices = pointcloud_plot_indices(config, 500)

        np.testing.assert_array_equal(indices, np.arange(500))

    def test_successful_setup_plot_draws_only_500_target_points(self) -> None:
        rng = np.random.default_rng(12)
        coords = rng.normal(size=(1200, 3)) * 0.05
        field = np.zeros((1200, 3), dtype=float)
        field[:, 2] = rng.normal(size=1200) * 1.0e-6
        target = TargetField(
            coords=coords,
            b=field,
            weights=np.ones(1200),
            suppressed=np.zeros(1200, dtype=bool),
            target_field_group_inds=np.ones(1200, dtype=int),
        )
        mesh = Mesh(
            vertices=np.asarray(((-0.1, -0.1, 0.0), (0.1, -0.1, 0.0), (0.0, 0.1, 0.0))),
            faces=np.asarray(((0, 1, 2),), dtype=int),
        )
        result = SimpleNamespace(
            coil_parts=[CoilPart(mesh=mesh)],
            target_field=target,
            target_field_1A=target,
            field_by_layout=0.8 * field,
            field_layout_per1Amp=0.8 * field,
            b_field_opt_sf=0.9 * field,
            b_field_opt_sf_1A=0.9 * field,
            input_data=CoilGenConfig(
                field_display_units="Hz",
                plot_pointcloud_max_points=500,
                plot_pointcloud_random_seed=41,
            ),
        )

        figure = plot_problem_setup_geometry_3d(result, show=False)
        try:
            scatter_counts = [
                len(collection._offsets3d[0])
                for collection in figure.axes[0].collections
                if hasattr(collection, "_offsets3d")
            ]
            self.assertEqual(sum(scatter_counts), 500)
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)

        self.assertEqual(len(result.target_field.coords), 1200)

        figures = (
            plot_field_pointclouds_3d(result, show=False),
            plot_shim_pointclouds_3d(result, show=False),
        )
        try:
            for plot_figure in figures:
                for axis in plot_figure.axes:
                    scatter_counts = [
                        len(collection._offsets3d[0])
                        for collection in axis.collections
                        if hasattr(collection, "_offsets3d")
                    ]
                    if scatter_counts:
                        self.assertEqual(sum(scatter_counts), 500)
        finally:
            import matplotlib.pyplot as plt

            for plot_figure in figures:
                plt.close(plot_figure)


if __name__ == "__main__":
    unittest.main()
