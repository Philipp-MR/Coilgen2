from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import coilgen.gradient_slice_maps as gradient_slice_maps
from coilgen.gradient_slice_maps import (
    GradientSliceMapResult,
    GradientSliceMapSpec,
    evaluate_gradient_slice_maps,
    gradient_slice_map_summary_dict,
)


class GradientSliceMapTest(unittest.TestCase):
    def test_target_slice_maps_use_pixel_centers_and_finite_difference_gradients(self) -> None:
        axis = np.linspace(-0.08, 0.08, 5)
        coords = np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
        inside = np.linalg.norm(coords, axis=1) <= 0.08 * (1.0 + 1.0e-12)
        coords = coords[inside]
        target_b = np.zeros((len(coords), 3), dtype=float)
        target_b[:, 2] = 0.01 * coords[:, 0]
        target_field = SimpleNamespace(
            coords=coords,
            b=target_b,
            suppressed=np.zeros(len(coords), dtype=bool),
        )
        config = SimpleNamespace(
            target_region_radius=0.08,
            target_gradient_strength=0.01,
            target_gradient_axis="x",
            linearity_reference_radius_m=0.08,
            linearity_fit_offset=True,
            b_0_strength_T=1.5,
            b_0_direction=(0.0, 0.0, 1.0),
            gradient_slice_maps_enabled=True,
            gradient_slice_map_axis="z",
            gradient_slice_map_slice_count=3,
            gradient_slice_map_grid_size=24,
            gradient_slice_map_domain="dsv",
            gradient_slice_map_source="target",
            gradient_slice_map_current="target",
            gradient_slice_map_field_components=("Bz",),
            gradient_slice_map_gradient_components=("dBz_dx", "dBz_dy", "dBz_dz", "gradBz_mag", "linearity_residual"),
            gradient_slice_map_concomitant_components=("concomitant_Bz", "concomitant_grad_mag"),
            gradient_slice_map_gradient_method="finite_difference",
            gradient_slice_map_gradient_step_m=1.0e-4,
            gradient_slice_map_overlay=(),
            gradient_slice_map_overlay_mode="none",
            gradient_slice_map_overlay_slab_thickness_m=0.0,
            parallel_worker_count=1,
            biot_savart_chunk_size=512,
            parallel_min_work_size=50000,
            parallel_max_chunk_memory_mb=64.0,
            progress_flag=False,
        )
        result = SimpleNamespace(
            input_data=config,
            target_field=target_field,
            coil_parts=(),
            diagnostic_cache={},
            needed_current_layout=1.0,
            layout_gradient={"achieved_gradient": 0.01},
            potential_step=1.0,
        )

        evaluated = evaluate_gradient_slice_maps(result)

        self.assertEqual(evaluated.axis_name, "z")
        self.assertEqual(len(evaluated.slices), 3)
        self.assertGreater(evaluated.summary["gradient_slice_map_sample_count"], 0)
        dx_values = _component_values(evaluated, "dBz_dx")
        dy_values = _component_values(evaluated, "dBz_dy")
        dz_values = _component_values(evaluated, "dBz_dz")
        residual = _component_values(evaluated, "linearity_residual")
        self.assertAlmostEqual(float(np.nanmedian(dx_values)), 0.01, places=10)
        self.assertLess(float(np.nanmax(np.abs(dy_values))), 1.0e-12)
        self.assertLess(float(np.nanmax(np.abs(dz_values))), 1.0e-12)
        self.assertLess(float(np.nanmax(np.abs(residual))), 1.0e-12)
        self.assertIn("concomitant_Bz", evaluated.component_units)
        self.assertLess(evaluated.summary["gradient_slice_map_linearity_peak_percent"], 1.0e-6)

    def test_layout_slice_maps_match_per_slice_finite_difference_reference(self) -> None:
        result = _layout_slice_map_result(parallel_worker_count=1)

        evaluated = evaluate_gradient_slice_maps(result)

        config = result.input_data
        source = "layout"
        current_scale = gradient_slice_maps._current_scale_A(result, config.gradient_slice_map_current)
        polylines = gradient_slice_maps._source_polylines(result, source)
        for slice_data in evaluated.slices:
            points = slice_data.coords[slice_data.mask]
            base = gradient_slice_maps._evaluate_source_field(
                result,
                source=source,
                points=points,
                current=config.gradient_slice_map_current,
                current_scale_A=current_scale,
                polylines=polylines,
                config=config,
            )
            derivative = gradient_slice_maps._finite_difference_field_gradient(
                result,
                source=source,
                points=points,
                step_m=evaluated.gradient_step_m,
                current=config.gradient_slice_map_current,
                current_scale_A=current_scale,
                polylines=polylines,
                config=config,
            )

            np.testing.assert_allclose(slice_data.maps["Bz"][slice_data.mask], base[:, 2], rtol=2.0e-14, atol=1.0e-18)
            np.testing.assert_allclose(
                slice_data.maps["dBz_dx"][slice_data.mask], derivative[:, 2, 0], rtol=2.0e-13, atol=1.0e-15
            )
            np.testing.assert_allclose(
                slice_data.maps["dBz_dy"][slice_data.mask], derivative[:, 2, 1], rtol=2.0e-13, atol=1.0e-15
            )
            np.testing.assert_allclose(
                slice_data.maps["dBz_dz"][slice_data.mask], derivative[:, 2, 2], rtol=2.0e-13, atol=1.0e-15
            )
            np.testing.assert_allclose(
                slice_data.maps["gradBz_mag"][slice_data.mask],
                np.linalg.norm(derivative[:, 2, :], axis=1),
                rtol=2.0e-13,
                atol=1.0e-15,
            )

    def test_layout_slice_maps_parallel_workers_match_serial(self) -> None:
        serial = evaluate_gradient_slice_maps(_layout_slice_map_result(parallel_worker_count=1))
        parallel = evaluate_gradient_slice_maps(_layout_slice_map_result(parallel_worker_count=2))

        self.assertEqual(len(parallel.slices), len(serial.slices))
        for serial_slice, parallel_slice in zip(serial.slices, parallel.slices):
            self.assertAlmostEqual(parallel_slice.value, serial_slice.value)
            for component in ("Bz", "dBz_dx", "dBz_dy", "dBz_dz", "gradBz_mag"):
                np.testing.assert_allclose(
                    parallel_slice.maps[component],
                    serial_slice.maps[component],
                    rtol=2.0e-13,
                    atol=1.0e-15,
                    equal_nan=True,
                )

    def test_layout_source_field_resolves_zero_worker_count_for_biot_savart(self) -> None:
        config = SimpleNamespace(
            parallel_worker_count=0,
            biot_savart_chunk_size=8,
            parallel_min_work_size=0,
            parallel_max_chunk_memory_mb=64.0,
        )
        result = SimpleNamespace(input_data=config)
        target_points = np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=float)
        wire = np.asarray([[0.0, 0.02, 0.0], [0.0, 0.02, 0.01]], dtype=float)
        worker_counts = []

        def fake_biot_savart(line, target_coords, **kwargs):
            worker_counts.append(int(kwargs["worker_count"]))
            return np.ones((len(target_coords), 3), dtype=float)

        with patch.object(gradient_slice_maps, "resolve_worker_count", return_value=7):
            with patch.object(gradient_slice_maps, "biot_savart_wire_path", side_effect=fake_biot_savart):
                field = gradient_slice_maps._evaluate_source_field(
                    result,
                    source="layout",
                    points=target_points,
                    current="per_amp",
                    current_scale_A=2.0,
                    polylines=(wire,),
                    config=config,
                )

        self.assertEqual(worker_counts, [7])
        np.testing.assert_allclose(field, np.full((len(target_points), 3), 2.0))

    def test_summary_uses_cached_result(self) -> None:
        config = SimpleNamespace(gradient_slice_maps_enabled=True)
        cached = GradientSliceMapResult(
            spec=GradientSliceMapSpec(enabled=True),
            axis=2,
            axis_name="z",
            plot_axes=(0, 1),
            center_m=np.zeros(3),
            reference_radius_m=0.08,
            current_scale_A=1.0,
            gradient_step_m=1.0e-4,
            slices=(),
            component_units={},
            summary={"gradient_slice_maps_status": "evaluated", "gradient_slice_map_sample_count": 5},
        )
        result = SimpleNamespace(input_data=config, diagnostic_cache={"gradient_slice_maps": cached})

        summary = gradient_slice_map_summary_dict(result)

        self.assertEqual(summary["gradient_slice_maps_status"], "evaluated")
        self.assertEqual(summary["gradient_slice_map_sample_count"], 5)

    def test_gradient_slice_mosaic_layout_reserves_colorbar_tick_gutter(self) -> None:
        from coilgen.matplotlib_visualization import (
            _format_gradient_slice_axis_tick,
            _format_gradient_slice_colorbar_tick,
            _gradient_slice_colorbar_unit_label,
            _gradient_slice_axes_rect,
            _gradient_slice_colorbar_rect,
            _gradient_slice_mosaic_layout,
            _gradient_slice_spatial_ticks,
            _style_gradient_slice_colorbar,
        )

        layout = _gradient_slice_mosaic_layout(rows=2, cols=3)
        figure_width = float(layout["figure_width"])
        first_colorbar = _gradient_slice_colorbar_rect(layout, row=0, col=0, rows=2)
        second_tile = _gradient_slice_axes_rect(layout, row=0, col=1, rows=2)
        first_colorbar_right_in = (first_colorbar[0] + first_colorbar[2]) * figure_width
        second_tile_left_in = second_tile[0] * figure_width
        reserved_text_gap_in = second_tile_left_in - first_colorbar_right_in

        self.assertGreaterEqual(reserved_text_gap_in, 0.50)
        self.assertEqual(_format_gradient_slice_colorbar_tick(0.0), "0")
        self.assertEqual(_format_gradient_slice_colorbar_tick(-2.7e-5), "-2.7e-5")
        self.assertEqual(_format_gradient_slice_colorbar_tick(8.923), "8.9")
        self.assertEqual(_gradient_slice_colorbar_unit_label("T/m"), "[T/m]")
        self.assertEqual(_gradient_slice_colorbar_unit_label(""), "")
        self.assertEqual(_gradient_slice_spatial_ticks(np.asarray([-0.08, -0.04, 0.0, 0.04, 0.08])), [-0.08, 0.0, 0.08])
        self.assertEqual(_format_gradient_slice_axis_tick(-0.08), "-0.08")
        self.assertEqual(_format_gradient_slice_axis_tick(0.0), "0")
        self.assertEqual(_format_gradient_slice_axis_tick(1.2e-4), "1.2e-4")

        fake_colorbar = _FakeColorbar()
        _style_gradient_slice_colorbar(fake_colorbar, -1.0, 1.0, unit_label="T")

        self.assertEqual(fake_colorbar.label_text, "[T]")
        self.assertEqual(fake_colorbar.label_kwargs["fontsize"], 5.7)
        self.assertEqual(fake_colorbar.axis.yaxis.tick_position, "right")
        self.assertEqual(fake_colorbar.axis.yaxis.label_position, "right")

    def test_gradient_slice_mosaic_uses_compact_tile_labels(self) -> None:
        try:
            import matplotlib
        except ModuleNotFoundError:
            self.skipTest("matplotlib is not installed")
        matplotlib.use("Agg", force=True)
        from matplotlib.colors import to_rgba
        import matplotlib.pyplot as plt
        from coilgen.gradient_slice_maps import GradientSliceMapSlice
        from coilgen.matplotlib_visualization import plot_gradient_derivative_slice_mosaic

        grid = np.linspace(-0.08, 0.08, 6)
        mask = np.ones((len(grid), len(grid)), dtype=bool)
        slices = []
        for value in (-0.08, 0.0, 0.08):
            data = np.full((len(grid), len(grid)), value, dtype=float)
            if value == 0.0:
                data[0, 0] = np.nan
            slices.append(
                GradientSliceMapSlice(
                    axis=2,
                    value=value,
                    plot_axes=(0, 1),
                    x=grid,
                    y=grid,
                    coords=np.zeros((len(grid), len(grid), 3), dtype=float),
                    mask=mask,
                    maps={"dBz_dx": data},
                )
            )
        cached = GradientSliceMapResult(
            spec=GradientSliceMapSpec(
                enabled=True,
                gradient_components=("dBz_dx",),
                overlay=("wire_path",),
                overlay_mode="project_all",
            ),
            axis=2,
            axis_name="z",
            plot_axes=(0, 1),
            center_m=np.zeros(3),
            reference_radius_m=0.08,
            current_scale_A=1.0,
            gradient_step_m=1.0e-4,
            slices=tuple(slices),
            component_units={"dBz_dx": "T/m"},
            summary={},
        )
        result = SimpleNamespace(
            input_data=SimpleNamespace(
                gradient_slice_maps_enabled=True,
                gradient_slice_map_overlay=("wire_path",),
                gradient_slice_map_overlay_mode="project_all",
            ),
            diagnostic_cache={"gradient_slice_maps": cached},
            coil_parts=(
                SimpleNamespace(
                    wire_path=SimpleNamespace(v=np.asarray([[-0.08, -0.08, 0.0], [0.08, 0.08, 0.0]], dtype=float))
                ),
            ),
        )

        fig = plot_gradient_derivative_slice_mosaic(result, show=False)
        try:
            tile_axes = [ax for ax in fig.axes if ax.images]
            colorbar_axes = [ax for ax in fig.axes if not ax.images]

            self.assertEqual(len(tile_axes), 3)
            self.assertEqual(len(colorbar_axes), 3)
            self.assertTrue(all(ax.get_title() == "" for ax in tile_axes))
            self.assertTrue(all(1 <= len(ax.get_xticks()) <= 3 for ax in tile_axes))
            self.assertTrue(all(1 <= len(ax.get_yticks()) <= 3 for ax in tile_axes))
            self.assertTrue(all(len(ax.lines) == 0 for ax in tile_axes))
            tile_sizes = [
                (
                    round(ax.get_position().width * fig.get_figwidth(), 6),
                    round(ax.get_position().height * fig.get_figheight(), 6),
                )
                for ax in tile_axes
            ]
            self.assertEqual(len(set(tile_sizes)), 1)
            self.assertAlmostEqual(tile_sizes[0][0], tile_sizes[0][1], places=6)
            self.assertEqual(tile_axes[0].images[0].get_interpolation(), "nearest")
            self.assertTrue(np.allclose(tile_axes[0].images[0].cmap.get_bad(), to_rgba("#cbd5e1")))
            labels = [text.get_text() for ax in tile_axes for text in ax.texts]
            self.assertIn("dBz/dx, z=-0.08 m", labels)
            self.assertIn("dBz/dx, z=0 m", labels)
        finally:
            plt.close(fig)


def _component_values(evaluated, component: str) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(slice_data.maps[component], dtype=float)[slice_data.mask]
            for slice_data in evaluated.slices
            if component in slice_data.maps
        ]
    )


class _FakeColorbar:
    def __init__(self):
        self.axis = _FakeAxis()
        self.ax = self.axis
        self.label_text = ""
        self.label_kwargs = {}
        self.ticks = []
        self.ticklabels = []

    def set_ticks(self, ticks):
        self.ticks = list(ticks)

    def set_ticklabels(self, labels):
        self.ticklabels = list(labels)

    def set_label(self, text, **kwargs):
        self.label_text = text
        self.label_kwargs = dict(kwargs)


class _FakeAxis:
    def __init__(self):
        self.yaxis = _FakeYAxis()
        self.tick_kwargs = {}

    def tick_params(self, **kwargs):
        self.tick_kwargs = dict(kwargs)


class _FakeYAxis:
    def __init__(self):
        self.tick_position = ""
        self.label_position = ""

    def set_ticks_position(self, position):
        self.tick_position = position

    def set_label_position(self, position):
        self.label_position = position


def _layout_slice_map_result(*, parallel_worker_count: int):
    axis = np.linspace(-0.02, 0.02, 3)
    coords = np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
    target_field = SimpleNamespace(
        coords=coords,
        b=np.zeros((len(coords), 3), dtype=float),
        suppressed=np.zeros(len(coords), dtype=bool),
    )
    config = SimpleNamespace(
        target_region_radius=0.02,
        target_gradient_strength=0.01,
        target_gradient_axis="x",
        linearity_reference_radius_m=0.02,
        linearity_fit_offset=True,
        b_0_strength_T=1.5,
        b_0_direction=(0.0, 0.0, 1.0),
        gradient_slice_maps_enabled=True,
        gradient_slice_map_axis="z",
        gradient_slice_map_slice_count=3,
        gradient_slice_map_grid_size=8,
        gradient_slice_map_domain="dsv",
        gradient_slice_map_source="layout",
        gradient_slice_map_current="per_amp",
        gradient_slice_map_field_components=("Bz",),
        gradient_slice_map_gradient_components=("dBz_dx", "dBz_dy", "dBz_dz", "gradBz_mag"),
        gradient_slice_map_concomitant_components=(),
        gradient_slice_map_gradient_method="finite_difference",
        gradient_slice_map_gradient_step_m=2.0e-4,
        gradient_slice_map_overlay=(),
        gradient_slice_map_overlay_mode="none",
        gradient_slice_map_overlay_slab_thickness_m=0.0,
        parallel_worker_count=int(parallel_worker_count),
        biot_savart_chunk_size=1,
        parallel_min_work_size=0,
        parallel_max_chunk_memory_mb=1.0,
        progress_flag=False,
    )
    wire = np.asarray(
        [
            [0.04, -0.04, 0.03],
            [0.04, 0.04, 0.03],
            [-0.04, 0.04, 0.03],
            [-0.04, -0.04, 0.03],
            [0.04, -0.04, 0.03],
        ],
        dtype=float,
    )
    return SimpleNamespace(
        input_data=config,
        target_field=target_field,
        coil_parts=(SimpleNamespace(wire_path=SimpleNamespace(v=wire)),),
        diagnostic_cache={},
        needed_current_layout=1.0,
        layout_gradient=None,
        potential_step=1.0,
    )


def test_physical_layout_slices_and_overlays_use_the_evaluated_signed_current():
    from coilgen.physics import biot_savart_wire_path

    result = _layout_slice_map_result(parallel_worker_count=1)
    part = result.coil_parts[0]
    physical = part.wire_path.v + [0, 0, 0.01]
    part.fabrication_path = SimpleNamespace(wire_points_m=physical)
    result.input_data.gradient_slice_map_current = "target"
    result.needed_current_layout = -2.0
    result.layout_gradient = {"achieved_gradient": 0.002}  # A fresh gradient fit would incorrectly use +5 A.
    evaluated = evaluate_gradient_slice_maps(result)
    assert evaluated.current_scale_A == -2
    assert evaluated.summary["gradient_slice_map_geometry_source"] == "fabricated_centroid"
    np.testing.assert_array_equal(gradient_slice_maps._overlay_polylines(result, "wire_path")[0], physical)
    for section in evaluated.slices:
        coords = section.coords[section.mask]
        expected = biot_savart_wire_path(physical, coords, worker_count=1) * -2
        np.testing.assert_allclose(section.maps["Bz"][section.mask], expected[:, 2], rtol=1e-12, atol=1e-18)
        step = evaluated.gradient_step_m
        plus = biot_savart_wire_path(physical, coords + [step, 0, 0], worker_count=1)
        minus = biot_savart_wire_path(physical, coords - [step, 0, 0], worker_count=1)
        expected_dx = -2 * (plus[:, 2] - minus[:, 2]) / (2 * step)
        np.testing.assert_allclose(section.maps["dBz_dx"][section.mask], expected_dx, rtol=1e-12, atol=1e-16)


def test_slice_cache_tracks_physical_geometry_and_source_even_when_raw_path_is_unchanged():
    result = _layout_slice_map_result(parallel_worker_count=1)
    first = gradient_slice_maps.gradient_slice_map_result(result)
    assert gradient_slice_maps.gradient_slice_map_result(result) is first
    part = result.coil_parts[0]
    original = part.wire_path.v.copy()
    part.fabrication_path = SimpleNamespace(wire_points_m=original.copy())
    second = gradient_slice_maps.gradient_slice_map_result(result)
    assert second is not first
    assert second.summary["gradient_slice_map_geometry_source"] == "fabricated_centroid"
    part.fabrication_path.wire_points_m += [0, 0, 0.01]
    third = gradient_slice_maps.gradient_slice_map_result(result)
    assert third is not second
    assert not np.allclose(_component_values(second, "Bz"), _component_values(third, "Bz"), rtol=1e-5, atol=0)
    np.testing.assert_array_equal(part.wire_path.v, original)


def test_contour_only_layout_fallback_matches_final_evaluation_and_turn_multiplicity():
    result = _layout_slice_map_result(parallel_worker_count=1)
    part = result.coil_parts[0]
    original = part.wire_path.v.copy()
    part.fabrication_path = SimpleNamespace(wire_points_m=original + [0, 0, 0.01])
    part.contour_lines = [SimpleNamespace(v=original, turn_multiplicity=2)]
    result.input_data.skip_postprocessing = True
    lines = gradient_slice_maps._source_polylines(result, "layout")
    assert len(lines) == 2
    for line in lines:
        np.testing.assert_array_equal(line, original)
    assert gradient_slice_maps._layout_geometry_sources(result) == ("unconnected_contours",)
    evaluated = evaluate_gradient_slice_maps(result)
    assert evaluated.summary["gradient_slice_map_geometry_source"] == "unconnected_contours"


def test_slice_current_zero_negative_stream_and_legacy_gradient_fallback_are_distinct():
    result = _layout_slice_map_result(parallel_worker_count=1)
    result.needed_current_layout = 0.0
    result.layout_gradient = {"achieved_gradient": 0.002}
    assert gradient_slice_maps._current_scale_A(result, "target") == 0
    result.potential_step = -3
    assert gradient_slice_maps._current_scale_A(result, "stream_function") == -3
    del result.needed_current_layout
    assert gradient_slice_maps._current_scale_A(result, "target") == 5


def test_sampled_field_maps_use_canonical_physical_fields_and_refresh_gradient_cache():
    from coilgen.field_maps import field_scalar_maps, gradient_vector_maps
    from coilgen.physics import biot_savart_wire_path

    result = _layout_slice_map_result(parallel_worker_count=1)
    part = result.coil_parts[0]
    physical = part.wire_path.v + [0, 0, 0.01]
    coords = result.target_field.coords
    expected = biot_savart_wire_path(physical, coords, worker_count=1)
    original = biot_savart_wire_path(part.wire_path.v, coords, worker_count=1)
    result.field_layout_per1Amp = expected
    result.field_by_layout = expected
    result.target_field_1A = SimpleNamespace(b=original)
    result.b_field_opt_sf_1A = original
    result.b_field_opt_sf = original
    np.testing.assert_allclose(field_scalar_maps(result)["layout"].values, expected[:, 2])
    first = gradient_vector_maps(result)
    assert gradient_vector_maps(result) is first
    result.field_layout_per1Amp = biot_savart_wire_path(physical + [0, 0, 0.01], coords, worker_count=1)
    second = gradient_vector_maps(result)
    assert second is not first
    assert not np.allclose(first["layout"].values, second["layout"].values, rtol=1e-5, atol=0, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
