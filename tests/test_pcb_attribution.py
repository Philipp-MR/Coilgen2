"""PCB artwork credits the software without altering physical coordinates."""

import json
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

from coilgen.attribution import AUTHOR_CREDIT, AUTHOR_ROLE, SOFTWARE_NAME, software_attribution
from coilgen.models import PcbLayout, PcbWirePart
from coilgen.pcb import save_pcb_centerlines_as_csv, save_pcb_layouts_as_dxf, save_pcb_layouts_as_svg


def _pcb():
    wire = PcbWirePart(
        centerline=np.array([[0, 0], [0.02, 0.01]]),
        track_shape=np.array([[0, 0], [0.02, 0], [0.01, 0.01]]),
    )
    layout = PcbLayout(
        upper_layer=[wire],
        boundary=np.array([[0, 0], [0.02, 0], [0.02, 0.01], [0, 0.01], [0, 0]]),
        track_width=0.001,
        cylinder_radius=0.12,
    )
    return SimpleNamespace(coil_parts=[SimpleNamespace(pcb_layout=layout)]), layout


def test_pcb_svg_credits_software_and_preserves_artwork_dimensions(tmp_path):
    result, layout = _pcb()
    original = layout.upper_layer[0].track_shape.copy()
    path = save_pcb_layouts_as_svg(result, tmp_path)[0]
    svg = ET.parse(path).getroot()
    namespace = {"svg": "http://www.w3.org/2000/svg"}

    assert json.loads(svg.find("svg:metadata[@id='coilgen-software']", namespace).text) == software_attribution()
    assert svg.attrib["width"] == "22.000mm"
    assert svg.attrib["height"] == "11.000mm"
    assert svg.find("svg:polygon", namespace).attrib["points"] == "1.0000,10.5000 21.0000,10.5000 11.0000,0.5000"
    np.testing.assert_array_equal(layout.upper_layer[0].track_shape, original)


def test_pcb_dxf_comment_preserves_millimeter_units_and_copper_entities(tmp_path):
    result, layout = _pcb()
    path = save_pcb_layouts_as_dxf(result, tmp_path)[0]
    lines = path.read_text(encoding="ascii").splitlines()
    pairs = list(zip(lines[::2], lines[1::2]))

    assert pairs[0] == ("999", f"{SOFTWARE_NAME} software: {AUTHOR_CREDIT} ({AUTHOR_ROLE}).")
    assert pairs[1:5] == [("0", "SECTION"), ("2", "HEADER"), ("9", "$INSUNITS"), ("70", "4")]
    copper = pairs[pairs.index(("8", "upper")) + 1 :]
    assert copper[:2] == [("90", "3"), ("70", "1")]
    assert copper[2:8] == [("10", "0"), ("20", "0"), ("10", "20"), ("20", "0"), ("10", "10"), ("20", "10")]
    assert pairs[-2:] == [("0", "ENDSEC"), ("0", "EOF")]
    np.testing.assert_array_equal(layout.upper_layer[0].centerline, [[0, 0], [0.02, 0.01]])


def test_pcb_centerline_csv_retains_its_exact_data_format(tmp_path):
    result, _layout = _pcb()
    path = save_pcb_centerlines_as_csv(result, tmp_path)[0]

    assert path.read_text(encoding="utf-8") == (
        "part,layer,wire_part,point_index,x_m,y_m\n1,upper,1,1,0,0\n1,upper,1,2,0.02,0.01\n"
    )
