"""Native SVG metadata describes software without altering plotted geometry."""

import json
import xml.etree.ElementTree as ET

import numpy as np

from coilgen.attribution import software_attribution
from coilgen.models import CoilPart, Mesh, WirePath
from coilgen.visualization import _write, save_wire_projection


def test_native_svg_has_software_metadata_and_preserves_si_geometry(tmp_path):
    vertices = np.asarray([[0, 0, 0], [0.03, 0, 0], [0, 0.02, 0]])
    part = CoilPart(Mesh(vertices, np.asarray([[0, 1, 2]])))
    part.wire_path = WirePath(vertices.copy())
    path = save_wire_projection(part, tmp_path / "wire.svg")
    svg = ET.parse(path).getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}
    credits = svg.find("svg:metadata[@id='coilgen-software']", ns)
    assert json.loads(credits.text) == software_attribution()
    assert len(svg.findall("svg:polyline", ns)) > 0
    np.testing.assert_array_equal(part.mesh.vertices, vertices)
    np.testing.assert_array_equal(part.wire_path.v, vertices)


def test_non_svg_diagnostics_remain_exact_and_unwrapped(tmp_path):
    content = '[{"value_T": 0.0002}]'
    path = tmp_path / "diagnostics.json"
    _write(path, content)
    assert path.read_text(encoding="utf-8") == content
