"""Integration refinement retains the physical P1 current and original data."""

from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.models import CoilPart, Mesh
from coilgen.research.convergence import check_sheet_integration


def sheet():
    vertices = np.array([[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.1, 0.1, 0.0], [-0.1, 0.1, 0.0]])
    part = CoilPart(Mesh(vertices, np.array([[0, 1, 2], [0, 2, 3]])), stream_function=vertices[:, 0].copy())
    config = CoilGenConfig(target_field_component="vector", parallel_worker_count=1, progress_flag=False)
    return SimpleNamespace(coil_parts=[part], input_data=config)


def test_refined_integral_agrees_and_source_is_unchanged():
    result = sheet()
    original_vertices = result.coil_parts[0].mesh.vertices.copy()
    original_stream = result.coil_parts[0].stream_function.copy()
    points = np.array([[0.02, 0.03, 0.2], [-0.03, -0.02, 0.3]])
    check = check_sheet_integration(result, points, relative_tolerance=0.005, orders=(4, 7), subdivisions=(0, 1))
    assert check["passed"]
    assert check["independent_quadrature_check"]
    assert check["independent_mesh_solution_check"] is False
    assert check["current_refitted"] is False
    assert check["comparisons"][-1]["triangle_count"] == 8
    assert check["comparisons"][-1]["difference_rms_T"] == 0.0
    np.testing.assert_array_equal(result.coil_parts[0].mesh.vertices, original_vertices)
    np.testing.assert_array_equal(result.coil_parts[0].stream_function, original_stream)
    assert result.coil_parts[0].sensitivity_matrix is None
    strict = check_sheet_integration(result, points, relative_tolerance=1e-14, orders=(1, 5), subdivisions=(0, 1))
    assert not strict["passed"]
    coarse = check_sheet_integration(result, points, relative_tolerance=0.005, orders=(2, 5), subdivisions=(0, 1))
    assert not coarse["passed"]
    assert coarse["comparisons"][0]["relative_difference"] > 0.01


def test_resource_guard_runs_before_field_work(monkeypatch):
    monkeypatch.setattr(
        "coilgen.research.convergence.calculate_sensitivity_matrix", lambda *args: pytest.fail("Computed")
    )
    with pytest.raises(ValueError, match="budget"):
        check_sheet_integration(sheet(), [[0, 0, 0.2]], relative_tolerance=0.01, max_integration_work=1)
    with pytest.raises(ValueError, match="max_operator_bytes"):
        check_sheet_integration(sheet(), [[0, 0, 0.2]], relative_tolerance=0.01, max_operator_bytes=1)


def test_combined_operator_and_extreme_subdivision_guards(monkeypatch):
    monkeypatch.setattr(
        "coilgen.research.convergence.calculate_basis_functions", lambda *args: pytest.fail("Built basis")
    )
    result = sheet()
    result.coil_parts.append(sheet().coil_parts[0])
    with pytest.raises(ValueError, match="Combined.*max_operator_bytes"):
        check_sheet_integration(
            result, [[0, 0, 0.2]], relative_tolerance=0.01, orders=(2, 4), subdivisions=(0,), max_operator_bytes=100
        )
    with pytest.raises(ValueError, match="budget"):
        check_sheet_integration(sheet(), [[0, 0, 0.2]], relative_tolerance=0.01, subdivisions=(0, 10**100))


@pytest.mark.parametrize("point", [[0.0, 0.0, 0.0], [0.1, 0.1, 0.0], [0.0, -0.1, 0.0]])
def test_singular_surface_samples_rejected_before_field_work(monkeypatch, point):
    monkeypatch.setattr(
        "coilgen.research.convergence.calculate_sensitivity_matrix", lambda *args: pytest.fail("Computed")
    )
    with pytest.raises(ValueError, match="singular"):
        check_sheet_integration(sheet(), [point], relative_tolerance=0.01)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"orders": (0, 4)},
        {"subdivisions": (False, 1)},
        {"orders": (2, 2)},
        {"relative_tolerance": np.nan},
        {"orders": (2,), "subdivisions": (0,)},
    ],
)
def test_invalid_comparison_is_rejected(kwargs):
    options = {"relative_tolerance": 0.01, **kwargs}
    with pytest.raises(ValueError):
        check_sheet_integration(sheet(), [[0, 0, 0.2]], **options)
