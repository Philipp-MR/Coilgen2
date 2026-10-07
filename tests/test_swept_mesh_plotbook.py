from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.matplotlib_visualization import build_result_plot_specs, plot_swept_body_3d
from coilgen.models import CoilPart, Mesh, SweptBody


def _part():
    vertices = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    mesh = Mesh(vertices, faces)
    return CoilPart(mesh=mesh, swept_bodies=[SweptBody(mesh=mesh, contour_index=2)])


def test_geometry_plot_requests_expose_generated_sweep_meshes():
    result = SimpleNamespace(coil_parts=[_part()], input_data=CoilGenConfig())
    keys = {spec.key for spec in build_result_plot_specs(result, include=("wire",))}
    assert "swept_body" in keys
    assert "swept_body" not in {spec.key for spec in build_result_plot_specs(result, include=("field_comparison",))}
    result.coil_parts[0].swept_bodies.clear()
    assert "swept_body" not in {spec.key for spec in build_result_plot_specs(result, include=("wire",))}


def test_public_swept_plot_supports_independent_bodies_without_legacy_merged_mesh():
    import matplotlib.pyplot as plt

    part = _part()
    original = part.swept_bodies[0].mesh.vertices.copy()
    figure = plot_swept_body_3d(part, part_index=3, show=False)
    try:
        assert figure.get_label() == "Swept Conductor Body Mesh"
        assert figure._coilgen_swept_mesh_inspector is not None
        figure._coilgen_swept_mesh_inspector.select_body(0)
        assert "part 3" in figure._coilgen_swept_mesh_inspector.selection_label.get_text()
        assert any(axis.name == "3d" for axis in figure.axes)
        np.testing.assert_array_equal(part.swept_bodies[0].mesh.vertices, original)
    finally:
        plt.close(figure)


def test_unavailable_swept_mesh_does_not_leave_a_blank_figure():
    import matplotlib.pyplot as plt

    part = _part()
    part.swept_bodies.clear()
    before = plt.get_fignums()
    with pytest.raises(ValueError):
        plot_swept_body_3d(part, show=False)
    assert plt.get_fignums() == before
