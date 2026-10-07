from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
import coilgen.field_maps as field_maps
import coilgen.gradient_slice_maps as slice_maps
from coilgen.models import TargetField
from coilgen.spherical_harmonics import fit_spherical_harmonics_many, solid_spherical_harmonic_basis


def _result() -> SimpleNamespace:
    axis = np.linspace(-0.04, 0.04, 5)
    coords = np.asarray(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
    field = np.zeros_like(coords)
    field[:, 2] = 0.01 * coords[:, 0] + 0.005 * coords[:, 1]
    target = TargetField(
        coords=coords,
        b=field,
        weights=np.ones(len(coords)),
        suppressed=np.zeros(len(coords), dtype=bool),
        target_field_group_inds=np.ones(len(coords), dtype=int),
    )
    return SimpleNamespace(
        input_data=CoilGenConfig(
            target_region_radius=0.05,
            gradient_slice_maps_enabled=True,
            gradient_slice_map_source="target",
            gradient_slice_map_grid_size=8,
            gradient_slice_map_slice_count=3,
            gradient_slice_map_concomitant_components=(),
        ),
        target_field=target,
        target_field_1A=target,
        field_by_layout=0.9 * field,
        field_layout_per1Amp=0.9 * field,
        b_field_opt_sf=0.95 * field,
        b_field_opt_sf_1A=0.95 * field,
        needed_current_layout=1.0,
        potential_step=1.0,
        layout_gradient={},
        coil_parts=[],
        diagnostic_cache={},
    )


def test_gradient_cache_reuses_geometry_but_detects_fields_units_coords_and_masks() -> None:
    result = _result()
    with patch.object(
        field_maps, "_build_local_gradient_basis", wraps=field_maps._build_local_gradient_basis
    ) as builds:
        first = field_maps.gradient_vector_maps(result)
        assert field_maps.gradient_vector_maps(result) is first
        result.field_layout_per1Amp *= 2.0
        changed_field = field_maps.gradient_vector_maps(result)
        assert changed_field is not first
        np.testing.assert_allclose(changed_field["layout"].values, 2.0 * first["layout"].values, atol=1e-15)
        result.input_data.field_display_units = "Hz"
        changed_unit = field_maps.gradient_vector_maps(result)
        assert changed_unit["target"].unit == "Hz/(A m)"
        assert builds.call_count == 1

        result.target_field.coords[:, 0] *= 1.2
        field_maps.gradient_vector_maps(result)
        assert builds.call_count == 2
        result.target_field.suppressed[:2] = True
        first_mask = field_maps.gradient_vector_maps(result)
        result.target_field.suppressed[:4] = (False, False, True, True)
        changed_mask = field_maps.gradient_vector_maps(result)
        assert builds.call_count == 4
        assert np.all(np.isnan(first_mask["target"].values[:2]))
        assert np.all(np.isfinite(changed_mask["target"].values[:2]))


def test_finite_local_gradient_matches_original_least_squares_stencils() -> None:
    rng = np.random.default_rng(309)
    coords = rng.normal(size=(40, 3)) * 0.04
    values = coords[:, 0] ** 2 + 2.0 * coords[:, 1] - coords[:, 2]
    neighbors = field_maps._nearest_neighbor_indices(coords, 12)
    reference = np.asarray(
        [
            np.linalg.pinv(coords[indices] - point) @ (values[indices] - values[index])
            for index, (point, indices) in enumerate(zip(coords, neighbors))
        ]
    )
    actual = field_maps.local_scalar_gradients(coords, values)
    np.testing.assert_allclose(actual, reference, rtol=2e-13, atol=1e-14)


def test_invalid_gradient_samples_are_not_replaced_by_zero_measurements() -> None:
    rng = np.random.default_rng(17)
    coords = rng.normal(size=(35, 3))
    expected = np.array([0.2, -0.4, 0.6])
    values = coords @ expected + 7.0
    values[[2, 9]] = np.nan
    coords[12] = np.nan
    actual = field_maps.local_scalar_gradients(coords, values)
    assert np.all(np.isnan(actual[[2, 9, 12]]))
    valid = np.ones(len(values), dtype=bool)
    valid[[2, 9, 12]] = False
    np.testing.assert_allclose(actual[valid], np.broadcast_to(expected, actual[valid].shape), atol=1e-13)
    assert np.all(np.isnan(field_maps.local_scalar_gradients(coords, np.full(len(values), np.nan))))


def test_planar_stencil_preserves_identifiable_components_only() -> None:
    coords = np.asarray([[x, y, 0.0] for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0)])
    gradients = field_maps.local_scalar_gradients(coords, 2.0 * coords[:, 0] - coords[:, 1])
    np.testing.assert_allclose(gradients[:, :2], np.tile([2.0, -1.0], (len(coords), 1)), atol=1e-14)
    assert np.all(np.isnan(gradients[:, 2]))
    assert np.all(np.isnan(field_maps.local_scalar_gradients(np.zeros((1, 3)), np.ones(1))))


def test_slice_cache_isolates_specs_and_detects_same_size_mutations() -> None:
    result = _result()
    spec = replace(slice_maps.gradient_slice_map_spec_from_config(result.input_data), axis="x")
    with patch.object(
        slice_maps, "evaluate_gradient_slice_maps", wraps=slice_maps.evaluate_gradient_slice_maps
    ) as evaluate:
        custom = slice_maps.gradient_slice_map_result(result, spec)
        default = slice_maps.gradient_slice_map_result(result)
        assert custom.axis_name == "x" and default.axis_name == "z"
        assert result.diagnostic_cache["gradient_slice_maps"] is default
        assert slice_maps.gradient_slice_map_result(result, spec) is custom
        assert slice_maps.gradient_slice_map_result(result) is default
        assert evaluate.call_count == 2
        result.target_field.b *= 2.0
        changed_field = slice_maps.gradient_slice_map_result(result)
        assert changed_field is not default
        for old_slice, new_slice in zip(default.slices, changed_field.slices):
            np.testing.assert_allclose(new_slice.maps["Bz"], 2.0 * old_slice.maps["Bz"], equal_nan=True)
        result.target_field.coords[:, 0] += 0.001
        changed_coords = slice_maps.gradient_slice_map_result(result)
        assert changed_coords is not changed_field
        assert evaluate.call_count == 4


def test_slice_cache_detects_in_place_wire_path_edits() -> None:
    result = _result()
    result.input_data.gradient_slice_map_source = "layout"
    result.input_data.gradient_slice_map_gradient_components = ()
    wire = np.array([[0.08, -0.08, 0.08], [0.08, 0.08, 0.08], [-0.08, 0.08, 0.08]])
    result.coil_parts = [SimpleNamespace(wire_path=SimpleNamespace(v=wire))]
    before = slice_maps.gradient_slice_map_result(result)
    wire[:, 2] += 0.01
    after = slice_maps.gradient_slice_map_result(result)
    assert after is not before
    assert not np.allclose(
        before.slices[1].maps["Bz"], after.slices[1].maps["Bz"], rtol=1e-8, atol=1e-18, equal_nan=True
    )


@pytest.mark.parametrize("components, factor", [(("linearity_residual",), 1), (("dBz_dx",), 3), (("gradBz_mag",), 7)])
def test_slice_evaluation_only_computes_requested_derivatives(components, factor) -> None:
    result = _result()
    spec = replace(
        slice_maps.gradient_slice_map_spec_from_config(result.input_data),
        gradient_components=components,
        concomitant_components=(),
    )
    with patch.object(
        slice_maps, "_evaluate_source_field_batched", wraps=slice_maps._evaluate_source_field_batched
    ) as fields:
        evaluated = slice_maps.evaluate_gradient_slice_maps(result, spec)
        positions = sum(len(call.kwargs["points"]) for call in fields.call_args_list)
    assert positions == factor * evaluated.summary["gradient_slice_map_sample_count"]
    reference = slice_maps.evaluate_gradient_slice_maps(
        result, replace(spec, gradient_components=("dBz_dx", "dBz_dy", "dBz_dz", "gradBz_mag", "linearity_residual"))
    )
    for actual_slice, reference_slice in zip(evaluated.slices, reference.slices):
        np.testing.assert_allclose(
            actual_slice.maps[components[0]], reference_slice.maps[components[0]], equal_nan=True
        )


def test_interpolation_plans_reuse_geometry_and_preserve_uncached_values() -> None:
    rng = np.random.default_rng(117)
    points = rng.uniform(-1.0, 1.0, (40, 2))
    values = points[:, 0] ** 2 - points[:, 1]
    grid_x = np.r_[np.linspace(-0.9, 0.9, 9), points[0, 0]]
    grid_y = np.r_[np.linspace(-0.9, 0.9, 9), points[0, 1]]
    cache = {}
    with patch.object(field_maps, "_build_interpolation_plan", wraps=field_maps._build_interpolation_plan) as builds:
        for data in (values, 2.0 * values, -values):
            actual, support = field_maps.interpolate_points_to_grid(points, data, grid_x, grid_y, plan_cache=cache)
            reference, reference_support = field_maps.interpolate_points_to_grid(points, data, grid_x, grid_y)
            np.testing.assert_allclose(actual, reference, rtol=1e-14, atol=1e-14, equal_nan=True)
            np.testing.assert_array_equal(support, reference_support)
        assert builds.call_count == 1
        changed_values = values.copy()
        changed_values[0] = np.nan
        actual, _ = field_maps.interpolate_points_to_grid(points, changed_values, grid_x, grid_y, plan_cache=cache)
        reference, _ = field_maps.interpolate_points_to_grid(points, changed_values, grid_x, grid_y)
        np.testing.assert_allclose(actual, reference, equal_nan=True)
        assert builds.call_count == 2
        points[:, 0] *= 1.1
        field_maps.interpolate_points_to_grid(points, values, grid_x, grid_y, plan_cache=cache)
        assert builds.call_count == 3


def test_multi_source_sh_uses_one_factorization_and_recovers_known_terms() -> None:
    rng = np.random.default_rng(550)
    coords = rng.normal(size=(90, 3)) * 0.04
    weights = rng.uniform(0.2, 1.0, len(coords))
    basis, _terms, center, radius = solid_spherical_harmonic_basis(coords, 3, weights=weights)
    coefficients = rng.normal(size=(basis.shape[1], 3))
    values = basis @ coefficients
    with patch.object(np.linalg, "lstsq", wraps=np.linalg.lstsq) as solve:
        fits = fit_spherical_harmonics_many(
            coords,
            {str(index): values[:, index] for index in range(3)},
            max_order=3,
            weights=weights,
            center=center,
            radius=radius,
        )
    assert solve.call_count == 1
    for index, fit in enumerate(fits.values()):
        np.testing.assert_allclose(fit.coefficients, coefficients[:, index], rtol=1e-12, atol=1e-12)
        assert fit.rank == basis.shape[1]
        assert fit.residual_rms < 1e-12


def test_sh_row_helpers_accept_existing_fits_without_repeating_solves() -> None:
    result = _result()
    coil_fits = field_maps.coil_spherical_harmonic_fits(result)
    shim_fits = field_maps.shim_spherical_harmonic_fits(result)
    with patch.object(np.linalg, "lstsq", side_effect=AssertionError("unexpected refit")):
        assert field_maps.coil_spherical_harmonic_coefficient_rows(result, fits=coil_fits)
        assert field_maps.shim_spherical_harmonic_comparison(result, fits=shim_fits)
