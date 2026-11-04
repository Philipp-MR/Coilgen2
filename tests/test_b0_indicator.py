"""The 3D view shows the configured B0 direction next to the coordinate axes."""

from dataclasses import replace

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from coilgen.application.presets import preset_config  # noqa: E402
from coilgen.application.scene_spec import b0_indicator_mesh  # noqa: E402

import test_native_workbench as workbench_tests  # noqa: E402

application = workbench_tests.application
scientific_viewport = workbench_tests.scientific_viewport


@pytest.mark.parametrize("direction", [(0, 0, 1), (1, 0, 0), (0, -2, 0), (0, 1, 1)])
def test_indicator_arrow_points_along_b0(direction):
    unit = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    mesh = b0_indicator_mesh(direction, pv)
    points = np.asarray(mesh.points)
    projection = points @ unit
    tip = points[int(np.argmax(projection))]
    # The label sits beyond the arrow tip; both lie on the positive B0 side.
    assert float(tip @ unit) > 0.5
    assert float(np.min(projection)) == pytest.approx(-0.55, abs=1e-6)
    assert np.linalg.norm(tip - (tip @ unit) * unit) < 0.25


def test_viewport_tracks_the_model_b0_direction(scientific_viewport):
    viewport = scientific_viewport
    np.testing.assert_allclose(viewport.b0_direction, (0.0, 0.0, 1.0))
    viewport.render_config(replace(preset_config("cylinder"), b_0_direction=(0.0, 3.0, 0.0)))
    np.testing.assert_allclose(viewport.b0_direction, (0.0, 1.0, 0.0))
    # A renderless test plotter has no interactor: no widget, and no scene mesh is added for B0.
    assert viewport._b0_widget is None
