"""The realized centroid is authoritative for B, R/L and centerline exports."""

from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.conductor_geometry import conductor_trace
from coilgen.evaluation import layout_field_geometry
from coilgen.inductance import calculate_inductance_by_coil_layout
from coilgen.manufacturing import save_wire_trace_as_csv, save_wire_trace_as_step, wire_trace_rows
from coilgen.models import CoilPart, FabricationPath, Mesh, WirePath


def case():
    route = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.02, 0.02, 0.0]])
    physical = np.array([[0.0, 0.0, 0.005], [0.03, 0.0, 0.005], [0.03, 0.04, 0.005]])
    part = CoilPart(Mesh(route, np.array([[0, 1, 2]])))
    part.wire_path = WirePath(route)
    part.fabrication_path = FabricationPath(
        physical,
        np.tile([1.0, 0.0, 0.0], (3, 1)),
        np.tile([0.0, 0.0, 1.0], (3, 1)),
        np.tile([0.0, 1.0, 0.0], (3, 1)),
        np.array([[0.0, 0.0], [0.001, 0.0], [0.0, 0.001]]),
        route,
    )
    config = CoilGenConfig(
        wire_trace_csv_format="xyz_no_header", conductor_cross_section_width=0.001, conductor_cross_section_height=0.001
    )
    return part, config, physical, route


def test_field_electrical_and_exports_share_mounted_geometry(tmp_path):
    part, config, physical, route = case()
    field_points, source = layout_field_geometry(part, config)
    assert source == "fabricated_centroid"
    np.testing.assert_array_equal(field_points, physical)
    _, matrix, resistance, length, area = calculate_inductance_by_coil_layout([part], config)
    assert length[0] == pytest.approx(0.07)
    assert resistance[0] == pytest.approx(config.specific_conductivity_conductor * 0.07 / area[0])
    assert matrix[0, 0] > 0
    result = SimpleNamespace(coil_parts=[part], input_data=config)
    exported = save_wire_trace_as_csv(result, tmp_path)[0]
    np.testing.assert_allclose(np.loadtxt(exported, delimiter=","), physical)
    rows = wire_trace_rows(result)
    assert rows[-1]["cumulative_length_m"] == pytest.approx(length[0])
    step = save_wire_trace_as_step(result, tmp_path)[0].read_text()
    assert "0.005" in step
    config.inductance_source = "wire_path"
    assert calculate_inductance_by_coil_layout([part], config)[3][0] == pytest.approx(0.04)
    np.testing.assert_allclose(
        np.loadtxt(save_wire_trace_as_csv(result, tmp_path, source="wire_path")[0], delimiter=","), route
    )


def test_invalid_physical_geometry_cannot_fall_back():
    part, _config, _physical, _route = case()
    part.fabrication_path.points_m = np.full((3, 3), np.nan)
    with pytest.raises(ValueError, match="fabricated field trace"):
        conductor_trace(part)
    part.fabrication_path = None
    with pytest.raises(ValueError, match="has not been realized"):
        conductor_trace(part, source="fabricated_centroid")
