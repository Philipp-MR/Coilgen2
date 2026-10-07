"""Plot availability follows complete, finite evidence and exact current policies."""

import numpy as np
import pytest

from coilgen.gui.dataset_capabilities import resolve_dataset


def test_zero_field_and_zero_stream_coefficients_are_available():
    arrays = {
        "vertices_m": np.zeros((3, 3)),
        "faces": np.array([[0, 1, 2]]),
        "stream_function_A": np.zeros(3),
        "target_points_m": np.zeros((2, 3)),
        "target_field_T": np.zeros((2, 3)),
        "routed_field_T": np.zeros((2, 3)),
    }
    assert resolve_dataset(arrays, "stream_function").available
    assert resolve_dataset(arrays, "routed_residual").available
    original = resolve_dataset(arrays, "routed_residual", original=True)
    assert not original.available and not original.invalid
    assert "routed_field_at_stream_current_T" in original.reason
    assert original.required_stage == "Evaluate"


@pytest.mark.parametrize("coordinates", [np.empty((0, 3)), np.zeros((2, 2)), np.array([[0, 0, np.nan]])])
def test_empty_or_invalid_field_coordinates_are_unavailable(coordinates):
    capability = resolve_dataset(
        {"target_points_m": coordinates, "sheet_field_T": np.zeros((len(coordinates), 3))}, "sheet"
    )
    assert not capability.available


def test_residual_needs_the_matching_target_on_the_same_samples():
    arrays = {"target_points_m": np.zeros((2, 3)), "routed_field_T": np.zeros((2, 3))}
    assert resolve_dataset(arrays, "routed_field").available
    assert not resolve_dataset(arrays, "routed_residual").available
    arrays["target_field_T"] = np.zeros((3, 3))
    capability = resolve_dataset(arrays, "routed_residual")
    assert not capability.available and capability.invalid


@pytest.mark.parametrize(
    "faces", [np.empty((0, 3), dtype=int), np.array([[0, 1, 3]]), np.array([[0, 1, -1]]), np.array([[0, 1, 1.5]])]
)
def test_sweep_requires_complete_integer_faces_with_valid_indices(faces):
    arrays = {"swept_0_vertices_m": np.zeros((3, 3)), "swept_0_faces": faces}
    assert not resolve_dataset(arrays, "swept").available


def test_one_invalid_saved_path_does_not_silently_display_a_partial_winding():
    arrays = {"route_0_points_m": np.zeros((2, 3)), "route_1_points_m": np.array([[0, 0, 0], [np.inf, 0, 0]])}
    capability = resolve_dataset(arrays, "routes")
    assert not capability.available and capability.invalid
    assert "route_1_points_m" in capability.reason


def test_review_field_uses_its_own_samples_when_present():
    arrays = {
        "target_points_m": np.zeros((2, 3)),
        "fabricated_points_m": np.zeros((3, 3)),
        "fabricated_field_T": np.zeros((3, 3)),
    }
    assert resolve_dataset(arrays, "fabricated").available
    arrays["fabricated_points_m"][0, 0] = np.nan
    assert not resolve_dataset(arrays, "fabricated").available
