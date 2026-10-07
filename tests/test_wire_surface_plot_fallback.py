"""The wire plot must represent the layout that was actually constructed."""

from types import SimpleNamespace

import matplotlib
import numpy as np

from coilgen.matplotlib_visualization import build_result_plot_specs
from coilgen.models import CoilPart, ContourLine, Mesh, WirePath


matplotlib.use("Agg", force=True)


def _result(*, connected: bool):
    mesh = Mesh(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        faces=np.array([[0, 1, 2]]),
    )
    points = np.array([[0.1, 0.1, 0.0], [0.7, 0.1, 0.0], [0.1, 0.7, 0.0], [0.1, 0.1, 0.0]])
    part = CoilPart(mesh=mesh, contour_lines=[ContourLine(v=points, potential=1.0)])
    if connected:
        part.wire_path = WirePath(v=points)
    return SimpleNamespace(coil_parts=[part], input_data=SimpleNamespace(design_context="mr_shim"))


def test_contour_only_wire_page_draws_independent_paths_on_surface() -> None:
    import matplotlib.pyplot as plt

    specs = build_result_plot_specs(_result(connected=False), include=("wire", "wire_sequence"))
    wire = next(spec for spec in specs if spec.key == "wire")
    assert wire.title == "Coil Part 1: Independent Contour Paths on Coil Surface"
    assert not any(spec.key == "wire_sequence" for spec in specs)

    figure = wire.factory()
    try:
        assert figure.axes[0].lines
        assert any("unconnected contours" in text.get_label() for text in figure.axes[0].lines)
    finally:
        plt.close(figure)


def test_routed_wire_page_keeps_connected_path_and_sequence() -> None:
    import matplotlib.pyplot as plt

    specs = build_result_plot_specs(_result(connected=True), include=("wire", "wire_sequence"))
    wire = next(spec for spec in specs if spec.key == "wire")
    assert wire.title == "Coil Part 1: Connected Wire Path on Coil Surface"
    assert any(spec.key == "wire_sequence" for spec in specs)

    figure = wire.factory()
    try:
        assert any(line.get_label() == "connected wire path" for line in figure.axes[0].lines)
    finally:
        plt.close(figure)


def test_mixed_parts_show_routes_and_independent_contours() -> None:
    import matplotlib.pyplot as plt

    routed = _result(connected=True).coil_parts[0]
    contours = _result(connected=False).coil_parts[0]
    result = SimpleNamespace(coil_parts=[routed, contours], input_data=SimpleNamespace(design_context="mr_shim"))
    wire = next(spec for spec in build_result_plot_specs(result, include=("wire",)) if spec.key == "wire")
    assert wire.title == "All Coil Parts: Wire Paths and Independent Contours"

    figure = wire.factory()
    try:
        labels = {line.get_label() for line in figure.axes[0].lines}
        assert "part 1 connected wire path" in labels
        assert "part 2 unconnected contours" in labels
    finally:
        plt.close(figure)
